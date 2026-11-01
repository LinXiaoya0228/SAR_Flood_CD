"""Fold-restricted matched R0 source dataset and deterministic loading infrastructure."""

import gzip
import hashlib
import os
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
    ContractViolationError,
)


def seed_everything(seed: int) -> None:
    """Sets deterministic seeds across Python, NumPy, PyTorch CPU, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Returns a deterministic worker initialization function."""
    def _worker_init(worker_id: int):
        worker_seed = (base_seed + worker_id + 1) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _worker_init


class RSTR0SourceDataset(Dataset):
    """Event-disjoint matched R0 PyTorch Dataset.

    Filters exclusively through an explicit allowlist of event IDs.
    Enforces the proven linear amplitude SAR preprocessing, double-masking,
    and returns paired pre/post tensors (xA, xB).
    """

    def __init__(
        self,
        allowed_event_ids: List[int],
        dataset_root: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        canonical_preevent: str = "pre_event_2",
        sar_clamp_max: float = 0.15,
        sar_mean: Tuple[float, float] = (0.0953, 0.0264),
        sar_std: Tuple[float, float] = (0.0427, 0.0215),
        max_samples_cap: Optional[int] = None,
        augmenter=None,
        is_training: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        if not allowed_event_ids:
            raise ContractViolationError("RSTR0SourceDataset requires a non-empty allowed_event_ids list.")

        self.allowed_events: Set[int] = set()
        for e in allowed_event_ids:
            if not isinstance(e, int) or isinstance(e, bool):
                raise ContractViolationError(f"Invalid event ID type in dataset: {e} ({type(e)})")
            if e in FORBIDDEN_TARGET_EVENTS:
                raise ContractViolationError(f"CRITICAL: Forbidden target event {e} provided to RSTR0SourceDataset!")
            if e not in AUTHORITATIVE_SOURCE_EVENTS:
                raise ContractViolationError(f"Unknown event {e} not in authorized source events!")
            self.allowed_events.add(e)

        self.canonical_preevent = canonical_preevent
        self.sar_clamp_max = float(sar_clamp_max)
        self.sar_mean = np.array(sar_mean, dtype=np.float32)[:, None, None]
        self.sar_std = np.array(sar_std, dtype=np.float32)[:, None, None]
        self.augmenter = augmenter
        self.is_training = bool(is_training)
        self.seed = int(seed)
        self.current_epoch = 0

        # Resolve paths
        repo_root = Path(__file__).resolve().parents[2]
        if dataset_root:
            self.data_root = Path(dataset_root)
        else:
            default_kuro_data = repo_root.parent / "datasets/KuroSiwo"
            if default_kuro_data.exists():
                self.data_root = default_kuro_data
            else:
                self.data_root = repo_root.parent / "third_party/KuroSiwo"

        if manifest_path:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = repo_root.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict.gz"

        if "test" in self.manifest_path.name.lower():
            raise ContractViolationError(f"CRITICAL: Test manifest passed to RSTR0SourceDataset: {self.manifest_path}")

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        # Load manifest
        with gzip.open(self.manifest_path, "rb") as f:
            all_records = pickle.load(f)

        self.samples: List[Dict[str, Any]] = []
        self.admitted_counts_by_event: Dict[int, int] = {eid: 0 for eid in sorted(self.allowed_events)}
        self.skipped_samples_count: int = 0

        for key, record in all_records.items():
            actid = int(record["info"]["actid"])
            if actid in self.allowed_events:
                self.samples.append({
                    "sample_id": key,
                    "event_id": actid,
                    "relative_path": record["path"],
                })
                self.admitted_counts_by_event[actid] += 1

        # Deterministic sort by sample_id
        self.samples.sort(key=lambda s: s["sample_id"])

        if max_samples_cap is not None and max_samples_cap > 0:
            self.samples = self.samples[:max_samples_cap]
            # recount admitted
            self.admitted_counts_by_event = {eid: 0 for eid in sorted(self.allowed_events)}
            for s in self.samples:
                self.admitted_counts_by_event[s["event_id"]] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_dir(self, relative_path_str: str) -> Path:
        relative = Path(relative_path_str)
        sample_dir = self.data_root / relative
        if not sample_dir.is_dir() and relative.parts and relative.parts[0] == "data":
            sample_dir = self.data_root.joinpath(*relative.parts[1:])
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
        return sample_dir

    def _find_file(self, files: List[Path], prefix: str) -> Path:
        for f in files:
            if f.name.startswith(prefix) and f.suffix == ".tif":
                return f
        raise FileNotFoundError(f"Missing raster with prefix '{prefix}'")

    def _read_raster(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read(1)

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch index for sample-level deterministic augmentation seeding."""
        self.current_epoch = int(epoch)

    def _load_label_and_mask(self, item: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Loads raw ground truth and valid mask for fast class counting."""
        sample_dir = self._resolve_dir(item["relative_path"])
        files = [f for f in sample_dir.iterdir() if f.is_file()]
        valid_path = self._find_file(files, "MK0_MNA")
        label_path = self._find_file(files, "MK0_MLU")
        mna = self._read_raster(valid_path).astype(np.uint8)
        mlu = self._read_raster(label_path).astype(np.int64)
        return mlu, mna

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.samples[index]
        sample_dir = self._resolve_dir(item["relative_path"])
        files = [f for f in sample_dir.iterdir() if f.is_file()]

        # Locate SAR files
        if self.canonical_preevent == "pre_event_2":
            preevent_vv_path = self._find_file(files, "SL2_IVV")
            preevent_vh_path = self._find_file(files, "SL2_IVH")
        elif self.canonical_preevent == "pre_event_1":
            preevent_vv_path = self._find_file(files, "SL1_IVV")
            preevent_vh_path = self._find_file(files, "SL1_IVH")
        else:
            raise ValueError(f"Unknown canonical pre-event: {self.canonical_preevent}")

        post_vv_path = self._find_file(files, "MS1_IVV")
        post_vh_path = self._find_file(files, "MS1_IVH")
        valid_path = self._find_file(files, "MK0_MNA")
        label_path = self._find_file(files, "MK0_MLU")

        # Read SAR: float32
        pre_vv = self._read_raster(preevent_vv_path).astype(np.float32)
        pre_vh = self._read_raster(preevent_vh_path).astype(np.float32)
        post_vv = self._read_raster(post_vv_path).astype(np.float32)
        post_vh = self._read_raster(post_vh_path).astype(np.float32)

        # Read validity and label
        mna = self._read_raster(valid_path).astype(np.uint8)
        mlu = self._read_raster(label_path).astype(np.int64)

        # M3: Apply synchronized paired augmentation before clipping and standardization
        if self.augmenter is not None and self.is_training:
            pre_raw = np.stack([pre_vv, pre_vh], axis=0)
            post_raw = np.stack([post_vv, post_vh], axis=0)
            sample_seed = self.augmenter.derive_sample_seed(self.seed, self.current_epoch, item["sample_id"])
            pre_raw, post_raw, mlu, mna, _ = self.augmenter.augment(
                pre_raw, post_raw, mlu, mna, sample_seed
            )
            pre_vv, pre_vh = pre_raw[0], pre_raw[1]
            post_vv, post_vh = post_raw[0], post_raw[1]

        # Preprocess SAR: clamp -> nan_to_num -> z-score
        def _prep_sar(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
            sar = np.stack([vv, vh], axis=0)  # (2, 224, 224)
            sar = np.clip(sar, 0.0, self.sar_clamp_max)
            sar = np.nan_to_num(sar, nan=self.sar_clamp_max, posinf=self.sar_clamp_max, neginf=0.0)
            return (sar - self.sar_mean) / self.sar_std

        xA_arr = _prep_sar(pre_vv, pre_vh)
        xB_arr = _prep_sar(post_vv, post_vh)

        # Enforce double-masking rule: any invalid pixel becomes 3 (ignore_index)
        mlu[mna == 0] = 3

        return {
            "sample_id": item["sample_id"],
            "event_id": item["event_id"],
            "xA": torch.from_numpy(xA_arr).float(),
            "xB": torch.from_numpy(xB_arr).float(),
            "label": torch.from_numpy(mlu).long(),
            "valid_mask": torch.from_numpy(mna).to(torch.uint8),
        }

    def set_epoch(self, epoch: int) -> None:
        """Sets current epoch for deterministic sampling and augmentation."""
        self.current_epoch = int(epoch)

    def get_sample_paths(self, item: Dict[str, Any]) -> Dict[str, Path]:
        """Resolves file paths for sample label and valid mask."""
        sample_dir = self._resolve_dir(item["relative_path"])
        files = [f for f in sample_dir.iterdir() if f.is_file()]
        return {
            "label_path": self._find_file(files, "MK0_MLU"),
            "valid_mask_path": self._find_file(files, "MK0_MNA"),
        }


def build_r0_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    sampler: Optional[Sampler] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Constructs a reproducible DataLoader with deterministic generator and worker seeding."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    if sampler is not None:
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=make_worker_init_fn(seed),
        generator=generator,
    )


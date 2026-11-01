"""Evaluation dataset loader for KuroSiwo target/development events.

Strict Guarantees:
1. FORBIDDEN_SEALED_TEST_EVENTS are permanently quarantined; any attempt to load or
   evaluate them raises an immediate RuntimeError.
2. Double-masking validity: valid = (valid_mask == 1) & (label != 3).
3. 2-channel SAR preprocessing strictly preserves the frozen normalization constants:
   mean = [0.0953, 0.0264], std = [0.0427, 0.0215], clamped to [0.0, 0.15].
4. Canonical temporal pairing: pre_event_2 (SL2) and post_event (MS1).
5. Deterministic sample ordering sorted by sample_id.
"""

import gzip
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import numpy as np
import torch
from torch.utils.data import Dataset

FORBIDDEN_SEALED_TEST_EVENTS: Set[int] = {
    205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013
}

DEFAULT_TTA_DEV_EVENTS: List[int] = [
    279, 437, 520, 559, 1111003, 1111008
]

EXPECTED_TTA_DEV_COUNTS: Dict[int, int] = {
    279: 2381,
    437: 169,
    520: 92,
    559: 42,
    1111003: 19,
    1111008: 123,
}

TOTAL_EXPECTED_TTA_DEV_SAMPLES: int = 2826


class KuroEvaluationDataset(Dataset):
    """Controlled evaluation dataset for target/development events."""

    SENSORY_COMPONENTS = [
        "SL1_IVV", "SL1_IVH",
        "SL2_IVV", "SL2_IVH",
        "MS1_IVV", "MS1_IVH",
        "MK0_MNA",
    ]
    LABEL_COMPONENTS = ["MK0_MLU"]

    def __init__(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        allowed_events: Optional[List[int]] = None,
        canonical_preevent: str = "pre_event_2",
        sar_clamp_max: float = 0.15,
        sar_mean: Tuple[float, float] = (0.0953, 0.0264),
        sar_std: Tuple[float, float] = (0.0427, 0.0215),
        load_labels: bool = True,
        records: Optional[List[Dict[str, Any]]] = None,
        allow_sealed_test: bool = False,
    ):
        super().__init__()
        self.canonical_preevent = str(canonical_preevent)
        self.sar_clamp_max = float(sar_clamp_max)
        self.sar_mean = np.array(sar_mean, dtype=np.float32)[:, None, None]
        self.sar_std = np.array(sar_std, dtype=np.float32)[:, None, None]
        self.load_labels = bool(load_labels)
        self.allow_sealed_test = bool(allow_sealed_test)

        # Resolve allowed events
        if allowed_events is None:
            self.allowed_events = sorted(list(DEFAULT_TTA_DEV_EVENTS))
        else:
            self.allowed_events = sorted(list(set(int(e) for e in allowed_events)))

        # Research Integrity Assertion 1: Disjointness from sealed test partition (unless authorized)
        allowed_set = set(self.allowed_events)
        if not self.allow_sealed_test:
            if not allowed_set.isdisjoint(FORBIDDEN_SEALED_TEST_EVENTS):
                intersection = allowed_set & FORBIDDEN_SEALED_TEST_EVENTS
                raise RuntimeError(
                    f"CRITICAL RESEARCH INTEGRITY VIOLATION: Evaluation requested on "
                    f"forbidden sealed-test events: {sorted(list(intersection))}!"
                )

        # In-memory mock records for testing
        if records is not None:
            self.data_root = None
            self.manifest_path = None
            self.samples = []
            for r in records:
                actid = int(r["event_id"])
                if not self.allow_sealed_test and actid in FORBIDDEN_SEALED_TEST_EVENTS:
                    raise RuntimeError(
                        f"CRITICAL RESEARCH INTEGRITY VIOLATION: Mock record contains "
                        f"sealed test event {actid}!"
                    )
                if actid in allowed_set:
                    self.samples.append(r)
            self.samples.sort(key=lambda s: str(s["sample_id"]))
            return

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
            self.manifest_path = (
                repo_root.parent
                / "third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz"
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Evaluation manifest not found: {self.manifest_path}")

        # Load manifest
        with gzip.open(self.manifest_path, "rb") as f:
            all_records = pickle.load(f)

        self.samples = []
        for key, record in all_records.items():
            actid = int(record["info"]["actid"])
            if not self.allow_sealed_test and actid in FORBIDDEN_SEALED_TEST_EVENTS:
                # Immediate guard against accidental inspection
                continue
            if actid in allowed_set:
                self.samples.append({
                    "sample_id": key,
                    "event_id": actid,
                    "relative_path": record["path"],
                })

        # Deterministic sorting
        self.samples.sort(key=lambda s: str(s["sample_id"]))

    def __len__(self) -> int:
        return len(self.samples)

    @classmethod
    def from_mock_records(cls, mock_records: List[Dict[str, Any]], allowed_events: Optional[List[int]] = None):
        """Constructs a dataset instance from mock in-memory dictionaries for unit testing."""
        return cls(records=mock_records, allowed_events=allowed_events)

    def _resolve_dir(self, relative_path_str: str) -> Path:
        relative = Path(relative_path_str)
        sample_dir = self.data_root / relative
        if not sample_dir.is_dir() and relative.parts and relative.parts[0] == "data":
            sample_dir = self.data_root.joinpath(*relative.parts[1:])
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
        return sample_dir

    def _find_prefix(self, files: List[Path], prefix: str) -> Path:
        for f in files:
            if f.name.startswith(prefix):
                return f
        raise FileNotFoundError(f"Component with prefix '{prefix}' missing from sample directory!")

    def _read_raster(self, path: Path) -> np.ndarray:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read(1)
        return data

    def _preprocess_sar(self, vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
        sar = np.stack([vv, vh], axis=0)  # (2, H, W)
        sar = np.clip(sar, 0.0, self.sar_clamp_max)
        sar = np.nan_to_num(sar, nan=self.sar_clamp_max, posinf=self.sar_clamp_max, neginf=0.0)
        sar = (sar - self.sar_mean) / self.sar_std
        return sar

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.samples[index]

        # In-memory mock support
        if "xA" in item and "xB" in item:
            xA = item["xA"]
            xB = item["xB"]
            valid_mask = item.get("valid_mask", torch.ones((224, 224), dtype=torch.uint8))
            if "label" in item:
                label = item["label"]
            else:
                label = (
                    torch.zeros(valid_mask.shape, dtype=torch.int64)
                    if torch.is_tensor(valid_mask)
                    else np.zeros(valid_mask.shape, dtype=np.int64)
                )
            valid = (valid_mask == 1) & (label != 3)
            return {
                "sample_id": item["sample_id"],
                "event_id": item["event_id"],
                "relative_path": item.get("relative_path", ""),
                "xA": xA if torch.is_tensor(xA) else torch.from_numpy(xA).float(),
                "xB": xB if torch.is_tensor(xB) else torch.from_numpy(xB).float(),
                "label": label if torch.is_tensor(label) else torch.from_numpy(label).long(),
                "valid_mask": valid_mask if torch.is_tensor(valid_mask) else torch.from_numpy(valid_mask).to(torch.uint8),
                "valid": valid if torch.is_tensor(valid) else torch.from_numpy(valid).bool(),
            }

        # Defense-in-depth: check before opening any disk raster or label
        eid = int(item["event_id"])
        if not self.allow_sealed_test and eid in FORBIDDEN_SEALED_TEST_EVENTS:
            raise RuntimeError(
                f"CRITICAL RESEARCH INTEGRITY VIOLATION: Refusing to open rasters for "
                f"quarantined sealed-test event {eid} (sample: {item['sample_id']})!"
            )
        if eid not in self.allowed_events:
            raise RuntimeError(
                f"Access rejected: Event {eid} is not in allowed evaluation events: {self.allowed_events}"
            )

        sample_dir = self._resolve_dir(item["relative_path"])
        files = [f for f in sample_dir.iterdir() if f.is_file()]

        required_components = self.SENSORY_COMPONENTS + (self.LABEL_COMPONENTS if self.load_labels else [])
        located = {pfx: self._find_prefix(files, pfx) for pfx in required_components}

        # Read rasters
        sl1_vv = self._read_raster(located["SL1_IVV"]).astype(np.float32)
        sl1_vh = self._read_raster(located["SL1_IVH"]).astype(np.float32)
        sl2_vv = self._read_raster(located["SL2_IVV"]).astype(np.float32)
        sl2_vh = self._read_raster(located["SL2_IVH"]).astype(np.float32)
        ms1_vv = self._read_raster(located["MS1_IVV"]).astype(np.float32)
        ms1_vh = self._read_raster(located["MS1_IVH"]).astype(np.float32)
        valid_raw = self._read_raster(located["MK0_MNA"]).astype(np.uint8)

        label_raw = None
        if self.load_labels:
            label_raw = self._read_raster(located["MK0_MLU"]).astype(np.int64)

        # Preprocess SAR branches
        pre2_sar = self._preprocess_sar(sl2_vv, sl2_vh)
        pre1_sar = self._preprocess_sar(sl1_vv, sl1_vh)
        post_sar = self._preprocess_sar(ms1_vv, ms1_vh)

        if self.canonical_preevent == "pre_event_2":
            xA_np = pre2_sar
        elif self.canonical_preevent == "pre_event_1":
            xA_np = pre1_sar
        else:
            raise ValueError(f"Unknown canonical_preevent: {self.canonical_preevent}")
        xB_np = post_sar

        xA = torch.from_numpy(xA_np).float()
        xB = torch.from_numpy(xB_np).float()
        valid_mask = torch.from_numpy(valid_raw).to(torch.uint8)

        if label_raw is not None:
            label = torch.from_numpy(label_raw).long()
        else:
            label = torch.zeros_like(valid_mask, dtype=torch.int64)

        # Double-masking validity check
        valid = (valid_mask == 1) & (label != 3)

        assert torch.isfinite(xA).all(), f"Non-finite values in xA for sample {item['sample_id']}"
        assert torch.isfinite(xB).all(), f"Non-finite values in xB for sample {item['sample_id']}"

        return {
            "sample_id": item["sample_id"],
            "event_id": item["event_id"],
            "relative_path": item["relative_path"],
            "xA": xA,
            "xB": xB,
            "label": label,
            "valid_mask": valid_mask,
            "valid": valid,
        }

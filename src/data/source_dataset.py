"""Kuro Siwo PyTorch Dataset for controlled source-model training.

Strictly enforces:
1. Frozen 22/5 source-inner split (Alternative A1) via configs/splits/kuro_source_inner_split_v1.json.
2. Complete research-integrity isolation: rejects any attempt to load the 6 TTA-validation
   or 10 sealed-test events.
3. Model A ([VV, VH]) and Model B ([VV, VH, slope]) input tensor generation.
4. Policy S2 physical metric slope computation with train-only statistics normalization.
5. Canonical pre-event selection (default pre_event_2: SL2_IVV, SL2_IVH).
6. Double-masking convention: label[MK0_MNA == 0] = 3 (ignore_index).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, distance_transform_edt
import torch
from torch.utils.data import Dataset
from compress_pickle import load


def compute_policy_s2_slope(
    dem: np.ndarray,
    dx: float,
    dy: float,
    crs: Any,
    meta_nodata: Optional[float] = None,
    nodata_threshold: float = -430.0,
    dilation_iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes physical metric slope angle in degrees under Policy S2.
    
    Returns:
        slope_deg: (H, W) float32 slope angle in degrees [0, 90].
        valid_mask: (H, W) bool, True where elevation and slope are valid.
    """
    if crs is None or getattr(crs, "is_geographic", False):
        raise ValueError(f"DEM has unsupported geographic or missing CRS: {crs}")

    raw_nodata = (~np.isfinite(dem)) | (dem < float(nodata_threshold))
    if meta_nodata is not None and np.isfinite(meta_nodata):
        raw_nodata = raw_nodata | (np.isclose(dem, meta_nodata))

    if raw_nodata.all():
        return np.zeros_like(dem, dtype=np.float32), np.zeros_like(dem, dtype=bool)

    if raw_nodata.any():
        nodata_mask = binary_dilation(raw_nodata, iterations=dilation_iterations)
        valid_mask = ~nodata_mask
        # Inpaint with nearest valid elevation to prevent artificial cliff at boundary
        indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
        dem_clean = dem[tuple(indices)]
    else:
        valid_mask = np.ones_like(dem, dtype=bool)
        dem_clean = dem

    gy, gx = np.gradient(dem_clean, abs(dy), abs(dx))
    rise_run = np.sqrt(gx**2 + gy**2)
    slope_deg = (np.arctan(rise_run) * (180.0 / np.pi)).astype(np.float32)
    return slope_deg, valid_mask


class KuroSourceDataset(Dataset):
    """Controlled source-model training and validation dataset."""

    SEALED_TEST_EVENTS = [205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]
    TTA_VAL_EVENTS = [279, 437, 520, 559, 1111003, 1111008]
    EXPECTED_SPLIT_HASH = "7afd9768da9f37031a93a3aa813e3cd9812a110ee463e1c9442998b0f3e486d3"
    EXPECTED_ALL_SOURCE_SPLIT_HASH = "601b21c81f7f1e26984fe6507a1532dbfa373d72b29346103fb5f0a1ac38d5e1"
    ALLOWED_SPLIT_HASHES = {
        EXPECTED_SPLIT_HASH,
        EXPECTED_ALL_SOURCE_SPLIT_HASH,
    }

    def __init__(
        self,
        split_config_path: str = "configs/splits/kuro_source_inner_split_v1.json",
        split: str = "train",
        dataset_root: Optional[str] = None,
        manifest_path: Optional[str] = None,
        include_slope: bool = False,
        slope_stats_path: Optional[str] = None,
        canonical_preevent: str = "pre_event_2",
        sar_clamp_max: float = 0.15,
        sar_mean: Tuple[float, float] = (0.0953, 0.0264),
        sar_std: Tuple[float, float] = (0.0427, 0.0215),
    ):
        super().__init__()
        self.split = str(split).lower()
        if self.split not in ("train", "val", "all_source", "all"):
            raise ValueError(f"split must be 'train', 'val', or 'all_source', got '{self.split}'")

        self.include_slope = bool(include_slope)
        self.canonical_preevent = str(canonical_preevent)
        self.sar_clamp_max = float(sar_clamp_max)
        self.sar_mean = np.array(sar_mean, dtype=np.float32)[:, None, None]
        self.sar_std = np.array(sar_std, dtype=np.float32)[:, None, None]

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
            default_manifest = repo_root.parent / "third_party/KuroSiwo/pickle/KuroV2_grid_dict.gz"
            if default_manifest.exists():
                self.manifest_path = default_manifest
            else:
                self.manifest_path = self.data_root / "pickle/KuroV2_grid_dict.gz"

        if "test" in self.manifest_path.name.lower():
            raise RuntimeError(
                f"CRITICAL RESEARCH INTEGRITY VIOLATION: Test manifest provided to source dataset: {self.manifest_path}"
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Source manifest not found: {self.manifest_path}")

        # Verify split config
        split_p = Path(split_config_path)
        if not split_p.is_absolute():
            split_p = repo_root / split_p
        if not split_p.exists():
            raise FileNotFoundError(f"Split config not found: {split_p}")

        split_bytes = split_p.read_bytes()
        split_sha = hashlib.sha256(split_bytes).hexdigest()
        if split_sha not in self.ALLOWED_SPLIT_HASHES and split_sha != self.EXPECTED_SPLIT_HASH:
            raise RuntimeError(
                f"Split config SHA-256 mismatch! Expected one of {self.ALLOWED_SPLIT_HASHES}, got {split_sha}"
            )

        with open(split_p, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        inner_train_events = set(int(e) for e in split_data.get("source_inner_train_events", split_data.get("inner_train_events", [])))
        inner_val_events = set(int(e) for e in split_data.get("source_inner_validation_events", split_data.get("inner_val_events", [])))
        all_source_events = set(int(e) for e in split_data.get("source_all_events", []))
        if not all_source_events:
            all_source_events = inner_train_events | inner_val_events

        sealed_test_set = set(self.SEALED_TEST_EVENTS)
        tta_val_set = set(self.TTA_VAL_EVENTS)

        # Integrity check: zero overlap with quarantined events
        forbidden = sealed_test_set | tta_val_set
        if (
            not inner_train_events.isdisjoint(forbidden)
            or not inner_val_events.isdisjoint(forbidden)
            or not all_source_events.isdisjoint(forbidden)
        ):
            raise RuntimeError("CRITICAL RESEARCH INTEGRITY VIOLATION: Split file contains forbidden events!")

        if self.split in ("all_source", "all"):
            self.allowed_events = all_source_events
            expected_sample_count = int(split_data.get("sample_counts", {}).get("total_source_samples", 21462))
            if len(self.allowed_events) != 27:
                raise RuntimeError(f"Expected exactly 27 source events for all-source split, got {len(self.allowed_events)}")
        elif self.split == "train":
            if len(inner_train_events) == 27 and len(inner_val_events) == 0:
                self.allowed_events = inner_train_events
                expected_sample_count = int(split_data.get("sample_counts", {}).get("train_sample_count", 21462))
            else:
                self.allowed_events = inner_train_events
                expected_sample_count = int(split_data.get("sample_counts", {}).get("source_inner_train_sample_count", 19178))
        else:
            self.allowed_events = inner_val_events
            expected_sample_count = int(split_data.get("sample_counts", {}).get("source_inner_validation_sample_count", 2284))

        # Load slope stats if Model B
        if self.include_slope:
            if slope_stats_path:
                stats_p = Path(slope_stats_path)
            else:
                stats_p = repo_root / "artifacts/week1/source_train_slope_statistics.json"
                if not stats_p.exists():
                    stats_p = repo_root / "outputs/week1/source_train_slope_statistics.json"
            if not stats_p.exists():
                raise FileNotFoundError(
                    f"Model B slope statistics artifact missing: {stats_p}. "
                    "Run tools/generate_train_slope_statistics.py first!"
                )
            with open(stats_p, "r", encoding="utf-8") as f:
                slope_stats = json.load(f)
            stats_dict = slope_stats.get("statistics", slope_stats)
            self.slope_mean = float(stats_dict.get("mean_degrees", stats_dict.get("slope_mean_deg", 1.492574)))
            self.slope_std = float(stats_dict.get("std_degrees", stats_dict.get("slope_std_deg", 2.372311)))
            print(f"Loaded train-only slope normalization: mean={self.slope_mean:.4f} deg, std={self.slope_std:.4f} deg")

        # Load and filter manifest
        print(f"Loading source manifest: {self.manifest_path} for split '{self.split}'...")
        with open(self.manifest_path, "rb") as f:
            all_records = load(f)

        self.samples = []
        for key, record in all_records.items():
            actid = int(record["info"]["actid"])
            if actid in self.allowed_events:
                if actid in forbidden:
                    raise RuntimeError(
                        f"CRITICAL RESEARCH INTEGRITY VIOLATION: Quarantined event {actid} was in allowed_events!"
                    )
                self.samples.append({
                    "sample_id": key,
                    "event_id": actid,
                    "relative_path": record["path"],
                })

        # Deterministic sorting
        self.samples.sort(key=lambda s: s["sample_id"])
        print(f"Loaded {len(self.samples)} samples for split '{self.split}' across {len(self.allowed_events)} events.")
        if len(self.samples) != expected_sample_count:
            raise RuntimeError(
                f"Sample count mismatch for split '{self.split}'! Expected {expected_sample_count}, got {len(self.samples)}"
            )

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
            data = src.read(1)
        return data

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

        # Read SAR
        pre_vv = self._read_raster(preevent_vv_path).astype(np.float32)
        pre_vh = self._read_raster(preevent_vh_path).astype(np.float32)
        post_vv = self._read_raster(post_vv_path).astype(np.float32)
        post_vh = self._read_raster(post_vh_path).astype(np.float32)

        # Preprocess SAR: clamp -> nan_to_num -> z-score
        def _prep_sar(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
            sar = np.stack([vv, vh], axis=0)  # (2, 224, 224)
            sar = np.clip(sar, 0.0, self.sar_clamp_max)
            sar = np.nan_to_num(sar, nan=self.sar_clamp_max, posinf=self.sar_clamp_max, neginf=0.0)
            return (sar - self.sar_mean) / self.sar_std

        pre_sar = _prep_sar(pre_vv, pre_vh)
        post_sar = _prep_sar(post_vv, post_vh)

        # Process slope if Model B
        if self.include_slope:
            dem_path = self._find_file(files, "MK0_DEM")
            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float32)
                crs = src.crs
                dx, dy = src.res
                meta_nodata = src.nodata

            slope_deg, valid_slope_mask = compute_policy_s2_slope(
                dem=dem, dx=dx, dy=dy, crs=crs, meta_nodata=meta_nodata
            )
            # Neutral prior: invalid pixels get training slope mean, which yields 0.0 after standardization
            norm_slope = np.zeros_like(slope_deg, dtype=np.float32)
            norm_slope[valid_slope_mask] = (slope_deg[valid_slope_mask] - self.slope_mean) / self.slope_std
            norm_slope_ch = norm_slope[None, :, :]  # (1, 224, 224)

            xA_arr = np.concatenate([pre_sar, norm_slope_ch], axis=0)  # (3, 224, 224)
            xB_arr = np.concatenate([post_sar, norm_slope_ch], axis=0)  # (3, 224, 224)
        else:
            xA_arr = pre_sar  # (2, 224, 224)
            xB_arr = post_sar  # (2, 224, 224)

        # Read validity and label
        mna = self._read_raster(valid_path).astype(np.uint8)
        mlu = self._read_raster(label_path).astype(np.int64)

        # Enforce double-masking rule: any invalid pixel becomes 3 (ignore_index)
        mlu[mna == 0] = 3

        res = {
            "sample_id": item["sample_id"],
            "event_id": item["event_id"],
            "xA": torch.from_numpy(xA_arr).float(),
            "xB": torch.from_numpy(xB_arr).float(),
            "label": torch.from_numpy(mlu).long(),
            "valid_mask": torch.from_numpy(mna).to(torch.uint8),
        }
        if self.include_slope:
            res["slope_valid_mask"] = torch.from_numpy(valid_slope_mask).to(torch.uint8)
        return res

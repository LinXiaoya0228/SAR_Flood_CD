"""Kuro Siwo PyTorch Dataset for validation evaluation with strict event-disjoint isolation."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from compress_pickle import load


def sanitize_and_normalize_dem(
    raw_dem: np.ndarray,
    dem_mean: float = 67.0293,
    dem_std: float = 1765.0062,
    nodata_threshold: float = -1000.0,
    replacement_norm_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sanitize raw DEM values and normalize to standardized space.

    Policy (kuro_dem_sanitize_v1):
    1. Read / convert raw DEM as float32.
    2. Define invalid DEM pixels as:
       - non-finite values, including NaN, +Inf, -Inf;
       - raw DEM values < nodata_threshold (-1000.0 m), such as -32768.
    3. Apply committed DEM normalization: (raw_dem - dem_mean) / dem_std to valid values.
    4. Set invalid DEM locations to replacement_norm_value (0.0), corresponding to
       neutral / training-mean elevation in standardized space.
    5. Valid high elevations (e.g. +1000 m, +1800 m) are preserved and NOT clipped.
    6. Physically valid below-sea-level elevations (e.g. -100 m, -430 m) are preserved.
    7. Return:
       - norm_dem: np.ndarray of same shape, float32, normalized and sanitized.
       - dem_invalid_mask: np.ndarray of same shape, bool, True where DEM was invalid.
    """
    raw_float = np.asarray(raw_dem, dtype=np.float32)
    invalid_mask = (~np.isfinite(raw_float)) | (raw_float < float(nodata_threshold))
    valid_mask = ~invalid_mask

    norm_dem = np.full_like(raw_float, replacement_norm_value, dtype=np.float32)
    norm_dem[valid_mask] = (raw_float[valid_mask] - float(dem_mean)) / float(dem_std)
    norm_dem[invalid_mask] = replacement_norm_value

    return norm_dem, invalid_mask


class KuroValidationDataset(Dataset):
    """Dataset loader for Kuro Siwo validation events only.
    
    Guarantees:
    - Never touches sealed test events.
    - Explicit double-mask validity check: valid = (valid_mask == 1) & (label != 3).
    - Preprocessing is strictly documented and deterministic.
    """

    SENSORY_COMPONENTS = [
        "SL1_IVV", "SL1_IVH",
        "SL2_IVV", "SL2_IVH",
        "MS1_IVV", "MS1_IVH",
        "MK0_DEM", "MK0_MNA"
    ]
    LABEL_COMPONENTS = ["MK0_MLU"]
    REQUIRED_COMPONENTS = SENSORY_COMPONENTS + LABEL_COMPONENTS

    def __init__(
        self,
        dataset_root: str,
        manifest_path: str,
        allowed_events: List[int],
        sealed_test_events: List[int],
        canonical_preevent: str = "pre_event_2",
        sar_clamp_max: float = 0.15,
        sar_mean: List[float] = (0.0953, 0.0264),
        sar_std: List[float] = (0.0427, 0.0215),
        dem_mean: float = 67.0293,
        dem_std: float = 1765.0062,
        dem_nodata_threshold: float = -1000.0,
        dem_replacement_value: float = 0.0,
        preprocessing_version: str = "kuro_dem_sanitize_v1",
        load_labels: bool = True,
    ):
        super().__init__()
        self.data_root = Path(dataset_root)
        self.canonical_preevent = canonical_preevent
        self.sar_clamp_max = sar_clamp_max
        self.sar_mean = np.array(sar_mean, dtype=np.float32)[:, None, None]
        self.sar_std = np.array(sar_std, dtype=np.float32)[:, None, None]
        self.dem_mean = float(dem_mean)
        self.dem_std = float(dem_std)
        self.dem_nodata_threshold = float(dem_nodata_threshold)
        self.dem_replacement_value = float(dem_replacement_value)
        self.preprocessing_version = str(preprocessing_version)
        self.load_labels = bool(load_labels)

        # Strict security assertions on events
        allowed_set = set(int(e) for e in allowed_events)
        sealed_set = set(int(e) for e in sealed_test_events)
        assert allowed_set.isdisjoint(sealed_set), "Fatal: Validation set overlaps with sealed test events!"

        # Load manifest
        print(f"Loading manifest: {manifest_path}")
        with open(manifest_path, "rb") as f:
            all_records = load(f)

        # Filter to allowed validation events only
        self.samples = []
        for key, record in all_records.items():
            actid = int(record["info"]["actid"])
            if actid in allowed_set:
                assert actid not in sealed_set, f"Security violation: Attempted to load test event {actid}!"
                self.samples.append({
                    "sample_id": key,
                    "event_id": actid,
                    "relative_path": record["path"],
                })

        # Deterministic sorting
        self.samples.sort(key=lambda s: s["sample_id"])
        print(f"Loaded {len(self.samples)} validation samples across events: {sorted(allowed_set)}")

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

    def _find_prefix(self, files: List[Path], prefix: str) -> Path:
        for f in files:
            if f.name.startswith(prefix):
                return f
        raise FileNotFoundError(f"Component with prefix '{prefix}' missing from sample directory!")

    def _read_raster(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            data = src.read(1)
        return data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.samples[index]
        sample_dir = self._resolve_dir(item["relative_path"])
        files = [f for f in sample_dir.iterdir() if f.is_file()]

        # Locate files conditionally based on load_labels
        required_components = self.SENSORY_COMPONENTS + (self.LABEL_COMPONENTS if self.load_labels else [])
        located = {pfx: self._find_prefix(files, pfx) for pfx in required_components}

        # Read rasters
        sl1_vv = self._read_raster(located["SL1_IVV"]).astype(np.float32)
        sl1_vh = self._read_raster(located["SL1_IVH"]).astype(np.float32)
        sl2_vv = self._read_raster(located["SL2_IVV"]).astype(np.float32)
        sl2_vh = self._read_raster(located["SL2_IVH"]).astype(np.float32)
        ms1_vv = self._read_raster(located["MS1_IVV"]).astype(np.float32)
        ms1_vh = self._read_raster(located["MS1_IVH"]).astype(np.float32)
        dem = self._read_raster(located["MK0_DEM"]).astype(np.float32)
        valid = self._read_raster(located["MK0_MNA"]).astype(np.uint8)
        
        label = None
        if self.load_labels:
            label = self._read_raster(located["MK0_MLU"]).astype(np.int64)

        # Validate spatial shape
        shape_checks = [
            ("sl1_vv", sl1_vv), ("sl1_vh", sl1_vh),
            ("sl2_vv", sl2_vv), ("sl2_vh", sl2_vh),
            ("ms1_vv", ms1_vv), ("ms1_vh", ms1_vh),
            ("dem", dem), ("valid", valid),
        ]
        if self.load_labels and label is not None:
            shape_checks.append(("label", label))

        for name, arr in shape_checks:
            if arr.shape != (224, 224):
                raise ValueError(f"Sample {item['sample_id']} has unexpected shape {arr.shape} in {name}")

        # Record raw summary statistics before any scaling
        raw_stats = {
            "sl1_vv": (float(sl1_vv.min()), float(sl1_vv.max()), float(sl1_vv.mean()), float(sl1_vv.std())),
            "sl1_vh": (float(sl1_vh.min()), float(sl1_vh.max()), float(sl1_vh.mean()), float(sl1_vh.std())),
            "sl2_vv": (float(sl2_vv.min()), float(sl2_vv.max()), float(sl2_vv.mean()), float(sl2_vv.std())),
            "sl2_vh": (float(sl2_vh.min()), float(sl2_vh.max()), float(sl2_vh.mean()), float(sl2_vh.std())),
            "ms1_vv": (float(ms1_vv.min()), float(ms1_vv.max()), float(ms1_vv.mean()), float(ms1_vv.std())),
            "ms1_vh": (float(ms1_vh.min()), float(ms1_vh.max()), float(ms1_vh.mean()), float(ms1_vh.std())),
            "dem": (float(dem.min()), float(dem.max()), float(dem.mean()), float(dem.std())),
        }

        # Preprocess SAR: clamp -> nan_to_num -> normalize
        def preprocess_sar(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
            sar = np.stack([vv, vh], axis=0)  # (2, 224, 224)
            sar = np.clip(sar, 0.0, self.sar_clamp_max)
            sar = np.nan_to_num(sar, nan=self.sar_clamp_max, posinf=self.sar_clamp_max, neginf=0.0)
            sar = (sar - self.sar_mean) / self.sar_std
            return sar

        pre1_sar = preprocess_sar(sl1_vv, sl1_vh)
        pre2_sar = preprocess_sar(sl2_vv, sl2_vh)
        post_sar = preprocess_sar(ms1_vv, ms1_vh)

        # Preprocess DEM: explicit NoData sanitization or raw behavior
        if self.preprocessing_version == "kuro_raw":
            dem_norm = (dem - self.dem_mean) / self.dem_std
            dem_norm = np.nan_to_num(dem_norm, nan=0.0, posinf=0.0, neginf=0.0)
            dem_invalid_mask = (~np.isfinite(dem)) | (dem < self.dem_nodata_threshold)
        elif self.preprocessing_version == "kuro_dem_sanitize_v1":
            dem_norm, dem_invalid_mask = sanitize_and_normalize_dem(
                dem,
                dem_mean=self.dem_mean,
                dem_std=self.dem_std,
                nodata_threshold=self.dem_nodata_threshold,
                replacement_norm_value=self.dem_replacement_value,
            )
        else:
            raise ValueError(f"Unknown preprocessing_version: {self.preprocessing_version}")
        dem_norm_ch = dem_norm[None, :, :]  # (1, 224, 224)

        # Assemble 3-channel temporal inputs: [VV, VH, DEM]
        pre1_3ch = torch.from_numpy(np.concatenate([pre1_sar, dem_norm_ch], axis=0)).float()
        pre2_3ch = torch.from_numpy(np.concatenate([pre2_sar, dem_norm_ch], axis=0)).float()
        post_3ch = torch.from_numpy(np.concatenate([post_sar, dem_norm_ch], axis=0)).float()

        # Select canonical pre-event branch
        if self.canonical_preevent == "pre_event_2":
            xA = pre2_3ch
        elif self.canonical_preevent == "pre_event_1":
            xA = pre1_3ch
        else:
            raise ValueError(f"Unknown canonical pre-event option: {self.canonical_preevent}")
        xB = post_3ch

        # Verify finite values
        assert torch.isfinite(xA).all(), f"Non-finite values in xA for sample {item['sample_id']}"
        assert torch.isfinite(xB).all(), f"Non-finite values in xB for sample {item['sample_id']}"

        ret = {
            "sample_id": item["sample_id"],
            "event_id": item["event_id"],
            "xA": xA,
            "xB": xB,
            "pre1": pre1_3ch,
            "pre2": pre2_3ch,
            "post": post_3ch,
            "valid_mask": torch.from_numpy(valid).to(torch.uint8),
            "dem_invalid_mask": torch.from_numpy(dem_invalid_mask).to(torch.bool),
            "raw_stats": raw_stats,
            "component_paths": {k: str(v) for k, v in located.items()},
        }
        if self.load_labels and label is not None:
            ret["label"] = torch.from_numpy(label).long()
        return ret


class KuroUnlabeledValidationDataset(KuroValidationDataset):
    """Auditable unlabeled view of KuroValidationDataset.

    Guarantees:
    - Never resolves, reads, or returns MK0_MLU ground-truth rasters.
    - Strictly label-free for risk generation and TTA adaptation.
    """

    def __init__(self, *args, **kwargs):
        kwargs["load_labels"] = False
        super().__init__(*args, **kwargs)


"""
GEOID-Flood Dataset Adapter and Evaluation Pipeline for ChangeGuard.

Provides:
1. Tile discovery and validation for GEOID-Flood held-out partition.
2. Exact spatial tiling of 1024x1024 tiles into non-overlapping 256x256 chips (16 chips/tile).
3. Frozen ChangeGuard 2-channel SAR preprocessing (clamping, nan_to_num, standardization).
4. Double-masking validity logic (validity == 1 and label != 255).
5. Fast, memory-efficient evaluation engine yielding tile-level and event-level confusion counts.
"""

import os
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.evaluation.metrics import (
    compute_confusion_counts,
    compute_flood_metrics,
    compute_macro_metrics,
    compute_pooled_metrics,
)


@dataclass
class GEOIDTileRecord:
    event_id: str
    bbox_id: str
    tile_id: str
    tile_key: str
    pre_path: str
    post_path: str
    label_path: str
    validity_path: str
    country: Optional[str] = None
    event_time: Optional[str] = None


def discover_geoid_heldout_tiles(
    base_dir: Union[str, Path],
    catalog_path: Optional[Union[str, Path]] = None,
    filter_events: Optional[List[str]] = None,
) -> List[GEOIDTileRecord]:
    """Discovers all valid 4-component tiles from GEOID-Flood heldout partition.

    Args:
        base_dir: Directory containing geoid-flood-heldout AOI subdirectories.
        catalog_path: Optional path to heldout_tile_catalog.json for extra metadata.
        filter_events: Optional list of event IDs to retain (e.g. ['EMSR857']).
    Returns:
        List of verified GEOIDTileRecord instances, sorted by tile_key.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        raise FileNotFoundError(f"GEOID heldout directory not found: {base_path}")

    # Load catalog metadata if available
    metadata_by_key = {}
    if catalog_path and Path(catalog_path).is_file():
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        for row in catalog:
            # Catalog keys can be mapped by event_id, tile_id, bbox_id
            ev = str(row.get("event_id", ""))
            tid = str(row.get("tile_id", ""))
            bb = str(row.get("bbox_id", ""))
            metadata_by_key[(ev, tid, bb)] = row

    # Search for all label files on disk
    label_pattern = str(base_path / "*" / "label" / "*_label.tif")
    label_files = sorted(glob.glob(label_pattern))
    if not label_files:
        raise FileNotFoundError(f"No label GeoTIFFs found with pattern: {label_pattern}")

    records: List[GEOIDTileRecord] = []
    for l_path in label_files:
        filename = os.path.basename(l_path)
        prefix = filename.replace("_label.tif", "")
        parts = prefix.split("-")
        if len(parts) < 3:
            continue
        event_id = parts[0]
        if filter_events and event_id not in filter_events:
            continue

        aoi_dir = os.path.dirname(os.path.dirname(l_path))
        pre_files = glob.glob(os.path.join(aoi_dir, "s1grd", f"{prefix}_s1grd_pre_*.tif"))
        post_files = glob.glob(os.path.join(aoi_dir, "s1grd", f"{prefix}_s1grd_post_*.tif"))
        val_file = os.path.join(aoi_dir, "validity", f"{prefix}_validity.tif")

        if len(pre_files) != 1 or len(post_files) != 1 or not os.path.isfile(val_file):
            continue

        tile_id = parts[1]
        bbox_id = parts[2]
        meta = metadata_by_key.get((event_id, tile_id, bbox_id), {})

        records.append(
            GEOIDTileRecord(
                event_id=event_id,
                bbox_id=bbox_id,
                tile_id=tile_id,
                tile_key=prefix,
                pre_path=pre_files[0],
                post_path=post_files[0],
                label_path=l_path,
                validity_path=val_file,
                country=meta.get("countries", None),
                event_time=meta.get("event_time", None),
            )
        )

    records.sort(key=lambda r: r.tile_key)
    return records


class GEOIDTileDataset(Dataset):
    """Dataset delivering full 1024x1024 tiles decomposed into 16 non-overlapping 256x256 chips."""

    def __init__(
        self,
        records: List[GEOIDTileRecord],
        sar_clamp_max: float = 0.15,
        sar_nan_val: float = 0.0,
        sar_mean: Tuple[float, float] = (0.0953, 0.0264),
        sar_std: Tuple[float, float] = (0.0427, 0.0215),
    ):
        super().__init__()
        self.records = records
        self.sar_clamp_max = float(sar_clamp_max)
        self.sar_nan_val = float(sar_nan_val)
        self.sar_mean = np.array(sar_mean, dtype=np.float32)[:, None, None]
        self.sar_std = np.array(sar_std, dtype=np.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.records)

    def _preprocess_sar_tile(self, tif_path: str) -> np.ndarray:
        """Reads 2-channel SAR GeoTIFF, clips, fills NaNs, and standardizes."""
        with rasterio.open(tif_path) as src:
            # Bands: 1 = VV, 2 = VH
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32)

        sar = np.stack([vv, vh], axis=0)  # (2, 1024, 1024)
        sar = np.clip(sar, 0.0, self.sar_clamp_max)
        sar = np.nan_to_num(sar, nan=self.sar_nan_val, posinf=self.sar_clamp_max, neginf=0.0)
        sar = (sar - self.sar_mean) / self.sar_std
        return sar  # (2, 1024, 1024)

    def _slice_to_chips_2ch(self, sar_1024: np.ndarray) -> np.ndarray:
        """Slices (2, 1024, 1024) into 16 non-overlapping (2, 256, 256) chips."""
        # (2, 4, 256, 4, 256) -> transpose(1, 3, 0, 2, 4) -> (4, 4, 2, 256, 256) -> (16, 2, 256, 256)
        chips = sar_1024.reshape(2, 4, 256, 4, 256).transpose(1, 3, 0, 2, 4).reshape(16, 2, 256, 256)
        return chips

    def _slice_to_chips_1ch(self, arr_1024: np.ndarray) -> np.ndarray:
        """Slices (1024, 1024) into 16 non-overlapping (256, 256) chips."""
        chips = arr_1024.reshape(4, 256, 4, 256).swapaxes(1, 2).reshape(16, 256, 256)
        return chips

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[index]

        # 1. Preprocess SAR
        pre_sar = self._preprocess_sar_tile(rec.pre_path)
        post_sar = self._preprocess_sar_tile(rec.post_path)

        xA_chips = self._slice_to_chips_2ch(pre_sar)   # (16, 2, 256, 256)
        xB_chips = self._slice_to_chips_2ch(post_sar)  # (16, 2, 256, 256)

        # 2. Read Labels and Validity
        with rasterio.open(rec.label_path) as sl, rasterio.open(rec.validity_path) as sv:
            lbl = sl.read(1).astype(np.int64)
            val = sv.read(1).astype(np.uint8)

        # Map invalid / outside AOI to ignore class 3
        lbl_cleaned = lbl.copy()
        lbl_cleaned[(val == 0) | (lbl == 255)] = 3

        valid_mask = ((val == 1) & (lbl != 255)).astype(np.uint8)

        lbl_chips = self._slice_to_chips_1ch(lbl_cleaned)     # (16, 256, 256)
        val_chips = self._slice_to_chips_1ch(valid_mask)      # (16, 256, 256)

        return {
            "tile_key": rec.tile_key,
            "event_id": rec.event_id,
            "xA": torch.from_numpy(xA_chips).float(),        # (16, 2, 256, 256)
            "xB": torch.from_numpy(xB_chips).float(),        # (16, 2, 256, 256)
            "labels": torch.from_numpy(lbl_chips).long(),    # (16, 256, 256)
            "valid_mask": torch.from_numpy(val_chips).to(torch.uint8),  # (16, 256, 256)
        }


def evaluate_model_on_geoid(
    model: nn.Module,
    dataset: GEOIDTileDataset,
    device: str = "cuda:0",
    batch_size_tiles: int = 1,
    num_workers: int = 4,
    flood_class: int = 2,
    ignore_class: int = 3,
) -> Dict[str, Any]:
    """Runs high-performance evaluation of a frozen model on GEOIDTileDataset.

    Returns:
        Dictionary containing:
        - 'tile_metrics': List of confusion count dicts for every tile.
        - 'event_metrics': List of rate dicts for every distinct physical event.
        - 'macro_metrics': Unweighted macro-averaged precision, recall, F1, IoU across events.
        - 'pooled_metrics': Dataset-level pooled precision, recall, F1, IoU.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size_tiles,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() and device.startswith("cuda"),
    )

    tile_records = []
    # Event accumulators for confusion counts
    event_accumulators: Dict[str, Dict[str, int]] = {}

    with torch.no_grad():
        for batch in loader:
            tile_keys = batch["tile_key"]
            event_ids = batch["event_id"]
            # xA: (B_tiles, 16, 2, 256, 256) -> flatten tiles and chips to (B_tiles * 16, 2, 256, 256)
            xA = batch["xA"].view(-1, 2, 256, 256).to(device, non_blocking=True)
            xB = batch["xB"].view(-1, 2, 256, 256).to(device, non_blocking=True)
            labels = batch["labels"].view(-1, 256, 256).cpu().numpy()
            valid_masks = batch["valid_mask"].view(-1, 256, 256).cpu().numpy()

            logits = model(xA, xB)
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B_tiles * 16, 256, 256)

            # Accumulate per tile
            chips_per_tile = 16
            for t_idx, (t_key, ev_id) in enumerate(zip(tile_keys, event_ids)):
                c_start = t_idx * chips_per_tile
                c_end = c_start + chips_per_tile
                t_preds = preds[c_start:c_end]
                t_labels = labels[c_start:c_end]
                t_val = valid_masks[c_start:c_end]

                t_counts = compute_confusion_counts(
                    pred_classes=t_preds,
                    labels=t_labels,
                    valid_mask=t_val,
                    flood_class=flood_class,
                    ignore_class=ignore_class,
                )
                t_counts["tile_key"] = t_key
                t_counts["event_id"] = ev_id
                tile_records.append(t_counts)

                if ev_id not in event_accumulators:
                    event_accumulators[ev_id] = {
                        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                        "valid_pixels": 0, "gt_flood_pixels": 0, "pred_flood_pixels": 0,
                    }
                for k in ["tp", "fp", "fn", "tn", "valid_pixels", "gt_flood_pixels", "pred_flood_pixels"]:
                    event_accumulators[ev_id][k] += t_counts[k]

    # Compute event-level metrics
    event_rows = []
    for ev_id, counts in sorted(event_accumulators.items()):
        rates = compute_flood_metrics(counts)
        row = {
            "event_id": ev_id,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "tn": counts["tn"],
            "valid_pixels": counts["valid_pixels"],
            "gt_flood_pixels": counts["gt_flood_pixels"],
            "pred_flood_pixels": counts["pred_flood_pixels"],
            "precision": rates["precision"],
            "recall": rates["recall"],
            "f1": rates["f1"],
            "iou": rates["iou"],
            "is_evaluable": rates["f1"] is not None,
        }
        event_rows.append(row)

    macro_metrics = compute_macro_metrics(event_rows)
    pooled_metrics = compute_pooled_metrics(event_rows)

    return {
        "tile_records": tile_records,
        "event_metrics": event_rows,
        "macro_metrics": macro_metrics,
        "pooled_metrics": pooled_metrics,
    }

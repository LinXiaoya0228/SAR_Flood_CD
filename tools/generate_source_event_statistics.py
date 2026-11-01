"""Generate exact event-level source statistics across the 27 source-training events

and propose a deterministic 22/5 source-inner split.
"""

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import rasterio
import numpy as np
import pandas as pd
from compress_pickle import load


def compute_sample_stats(args):
    sample_key, record, dataset_root = args
    rel_path = Path(record["path"])
    if rel_path.parts and rel_path.parts[0] == "data":
        sample_dir = dataset_root.joinpath(*rel_path.parts[1:])
    else:
        sample_dir = dataset_root / rel_path

    datasets = record["info"]["datasets"]
    mlu_name = [k for k in datasets if k.startswith("MK0_MLU")][0]
    mna_name = [k for k in datasets if k.startswith("MK0_MNA")][0]

    mlu_path = sample_dir / f"{mlu_name}.tif"
    mna_path = sample_dir / f"{mna_name}.tif"

    with rasterio.open(mlu_path) as s:
        lbl = s.read(1)
    with rasterio.open(mna_path) as s:
        vld = s.read(1)

    total_pixels = lbl.size  # 50176
    valid_mask = (vld == 1) & (lbl != 3)
    valid_pixels = int(valid_mask.sum())

    c0 = int(((lbl == 0) & valid_mask).sum())
    c1 = int(((lbl == 1) & valid_mask).sum())
    c2 = int(((lbl == 2) & valid_mask).sum())
    c3 = int((lbl == 3).sum() + ((vld != 1) & (lbl != 3)).sum())

    has_flood = c2 > 0

    return {
        "sample_id": sample_key,
        "event_id": int(record["info"]["actid"]),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "c0_pixels": c0,
        "c1_pixels": c1,
        "c2_pixels": c2,
        "c3_pixels": c3,
        "has_flood": has_flood,
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    config_split_path = repo_root / "configs" / "kuro_event_split.json"
    dataset_root = Path("/usr1/home/s126mdg22_06/FYP_SARCD/datasets/KuroSiwo")
    pickle_path = Path("/usr1/home/s126mdg22_06/FYP_SARCD/third_party/KuroSiwo/pickle/KuroV2_grid_dict.gz")
    out_dir = repo_root / "outputs" / "week0"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_split_path) as f:
        split_cfg = json.load(f)

    source_events = sorted(int(e) for e in split_cfg["effective_splits"]["train"])
    val_events = sorted(int(e) for e in split_cfg["effective_splits"]["validation"])
    test_events = sorted(int(e) for e in split_cfg["effective_splits"]["test"])

    print(f"Configured 27 source events: {source_events}")
    assert len(source_events) == 27, f"Expected 27 source events, got {len(source_events)}"
    assert set(source_events).isdisjoint(set(val_events)), "Security violation: Source overlaps with val!"
    assert set(source_events).isdisjoint(set(test_events)), "Security violation: Source overlaps with test!"

    print(f"Loading train manifest from: {pickle_path}")
    with open(pickle_path, "rb") as f:
        all_records = load(f)

    source_records = [
        (k, r, dataset_root)
        for k, r in all_records.items()
        if int(r["info"]["actid"]) in source_events
    ]
    print(f"Found {len(source_records)} source samples across {len(source_events)} events.")

    t0 = time.time()
    print("Reading rasters across samples using ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=24) as pool:
        results = list(pool.map(compute_sample_stats, source_records))
    t_elapsed = time.time() - t0
    print(f"Processed {len(results)} samples in {t_elapsed:.1f}s ({len(results)/t_elapsed:.1f} samples/sec)")

    df_samples = pd.DataFrame(results)

    # Aggregate by event
    rows = []
    for event_id in source_events:
        sub = df_samples[df_samples["event_id"] == event_id]
        n_samples = len(sub)
        v_pixels = int(sub["valid_pixels"].sum())
        c0 = int(sub["c0_pixels"].sum())
        c1 = int(sub["c1_pixels"].sum())
        c2 = int(sub["c2_pixels"].sum())
        c3 = int(sub["c3_pixels"].sum())
        flood_samples = int(sub["has_flood"].sum())

        rows.append({
            "event_id": event_id,
            "sample_count": n_samples,
            "valid_pixel_count": v_pixels,
            "class0_no_water_pixels": c0,
            "class0_proportion": c0 / v_pixels if v_pixels > 0 else 0.0,
            "class1_perm_water_pixels": c1,
            "class1_proportion": c1 / v_pixels if v_pixels > 0 else 0.0,
            "class2_flood_pixels": c2,
            "class2_flood_proportion": c2 / v_pixels if v_pixels > 0 else 0.0,
            "class3_invalid_pixels": c3,
            "class3_invalid_proportion": c3 / (n_samples * 50176) if n_samples > 0 else 0.0,
            "samples_with_flood_count": flood_samples,
            "samples_with_flood_proportion": flood_samples / n_samples if n_samples > 0 else 0.0,
        })

    df_events = pd.DataFrame(rows)
    df_events.sort_values(by="event_id", inplace=True)

    csv_path = out_dir / "source_event_statistics.csv"
    json_path = out_dir / "source_event_statistics.json"

    df_events.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(df_events.to_dict(orient="records"), f, indent=2)

    print(f"Saved source event statistics to:\n  - {csv_path}\n  - {json_path}")


if __name__ == "__main__":
    main()

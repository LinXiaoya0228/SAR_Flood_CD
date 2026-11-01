from pathlib import Path
from collections import Counter
import json
import random
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from compress_pickle import load


PROJECT_ROOT = Path.home() / "FYP_SARCD"
DATA_ROOT = PROJECT_ROOT / "datasets/KuroSiwo"
KURO_REPO = PROJECT_ROOT / "third_party/KuroSiwo"
OUTPUT_ROOT = PROJECT_ROOT / "ChangeGuard/outputs/day1"
CONFIG_ROOT = PROJECT_ROOT / "ChangeGuard/configs"

TRAIN_MANIFEST = KURO_REPO / "pickle/KuroV2_grid_dict.gz"
TEST_MANIFEST = (
    KURO_REPO
    / "pickle/KuroV2_grid_dict_test_0_100.gz"
)
OFFICIAL_CONFIG = KURO_REPO / "configs/train/data_config.json"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)


def load_records(path):
    with path.open("rb") as file:
        return load(file)


def parse_split(config_text, name):
    pattern = rf'"{name}"\s*:\s*\[(.*?)\]'
    match = re.search(pattern, config_text, flags=re.DOTALL)

    if match is None:
        raise RuntimeError(f"Cannot find {name} in {OFFICIAL_CONFIG}")

    return [
        int(value)
        for value in re.findall(r"\d+", match.group(1))
    ]


def event_ids(records):
    return {
        int(record["info"]["actid"])
        for record in records.values()
    }


def resolve_sample_dir(record):
    relative = Path(record["path"])
    sample_dir = DATA_ROOT / relative

    if (
        not sample_dir.is_dir()
        and relative.parts
        and relative.parts[0] == "data"
    ):
        sample_dir = DATA_ROOT.joinpath(*relative.parts[1:])

    return sample_dir


def find_prefix(files, prefix):
    for file in files:
        if file.name.startswith(prefix):
            return file
    return None


def read_prefix(files, prefix):
    path = find_prefix(files, prefix)

    if path is None:
        raise FileNotFoundError(f"Missing prefix: {prefix}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError(f"Cannot read {path}")

    return image


def display_sar(image):
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    lower, upper = np.percentile(finite, [2, 98])

    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, lower, upper)
    return (image - lower) / (upper - lower)


print("===== Loading manifests =====")

train_records = load_records(TRAIN_MANIFEST)
test_records = load_records(TEST_MANIFEST)

train_manifest_events = event_ids(train_records)
test_manifest_events = event_ids(test_records)
all_data_events = train_manifest_events | test_manifest_events

print("Train manifest records:", len(train_records))
print("Test manifest records:", len(test_records))
print("Train manifest events:", len(train_manifest_events))
print("Test manifest events:", len(test_manifest_events))
print("All data events:", len(all_data_events))


print("\n===== Official split audit =====")

config_text = OFFICIAL_CONFIG.read_text(encoding="utf-8")

official_splits = {
    "train": parse_split(config_text, "train_acts"),
    "validation": parse_split(config_text, "val_acts"),
    "test": parse_split(config_text, "test_acts"),
}

for split_name, events in official_splits.items():
    print(f"{split_name}: {len(events)} events")
    print(sorted(events))

train_set = set(official_splits["train"])
validation_set = set(official_splits["validation"])
test_set = set(official_splits["test"])

overlaps = {
    "train_validation": sorted(train_set & validation_set),
    "train_test": sorted(train_set & test_set),
    "validation_test": sorted(validation_set & test_set),
}

configured_events = train_set | validation_set | test_set
unassigned_data_events = sorted(all_data_events - configured_events)
configured_but_absent = sorted(configured_events - all_data_events)

print("Split overlaps:", overlaps)
print("Data events not assigned:", unassigned_data_events)
print("Configured events absent from data:", configured_but_absent)

effective_splits = {
    split_name: sorted(set(events) & all_data_events)
    for split_name, events in official_splits.items()
}

split_output = {
    "label_map": {
        "0": "no_water",
        "1": "permanent_water",
        "2": "flood_water",
        "3": "invalid_no_data",
    },
    "binary_target": {
        "negative_labels": [0, 1],
        "positive_label": 2,
        "ignore_label": 3,
        "valid_mask_value": 1,
    },
    "official_splits": official_splits,
    "effective_splits": effective_splits,
    "configured_but_absent": configured_but_absent,
    "unassigned_data_events": unassigned_data_events,
}

split_path = CONFIG_ROOT / "kuro_event_split.json"

with split_path.open("w", encoding="utf-8") as file:
    json.dump(split_output, file, indent=2)

print("Saved:", split_path)


print("\n===== Label consistency audit =====")

random.seed(20260916)

record_keys = list(train_records.keys())
selected_keys = random.sample(
    record_keys,
    min(500, len(record_keys)),
)

raw_class_pixels = Counter()
valid_class_pixels = Counter()
valid_mask_values = Counter()
class3_marked_valid = 0
tested_labels = 0

for key in selected_keys:
    record = train_records[key]
    sample_dir = resolve_sample_dir(record)

    if not sample_dir.is_dir():
        continue

    files = [
        path
        for path in sample_dir.iterdir()
        if path.is_file()
    ]

    label_path = find_prefix(files, "MK0_MLU")
    valid_path = find_prefix(files, "MK0_MNA")

    if label_path is None or valid_path is None:
        continue

    label = cv2.imread(
        str(label_path),
        cv2.IMREAD_UNCHANGED,
    )
    valid = cv2.imread(
        str(valid_path),
        cv2.IMREAD_UNCHANGED,
    )

    if label is None or valid is None:
        continue

    tested_labels += 1

    values, counts = np.unique(label, return_counts=True)
    for value, count in zip(values, counts):
        raw_class_pixels[int(value)] += int(count)

    values, counts = np.unique(valid, return_counts=True)
    for value, count in zip(values, counts):
        valid_mask_values[int(value)] += int(count)

    valid_pixels = valid == 1

    values, counts = np.unique(
        label[valid_pixels],
        return_counts=True,
    )
    for value, count in zip(values, counts):
        valid_class_pixels[int(value)] += int(count)

    class3_marked_valid += int(
        np.count_nonzero((label == 3) & valid_pixels)
    )

print("Labels tested:", tested_labels)
print("Raw class pixel counts:", dict(sorted(raw_class_pixels.items())))
print("Valid class pixel counts:", dict(sorted(valid_class_pixels.items())))
print("Valid-mask values:", dict(sorted(valid_mask_values.items())))
print("Class-3 pixels marked valid:", class3_marked_valid)


print("\n===== Creating visualizations =====")

label_cmap = ListedColormap(
    [
        "black",       # 0 no water
        "deepskyblue", # 1 permanent water
        "magenta",     # 2 flood water
        "gray",        # 3 invalid
    ]
)

selected_samples = []
used_events = set()

for record in train_records.values():
    event_id = int(record["info"]["actid"])

    if event_id in used_events:
        continue

    sample_dir = resolve_sample_dir(record)

    if not sample_dir.is_dir():
        continue

    files = [
        path
        for path in sample_dir.iterdir()
        if path.is_file()
    ]

    label_path = find_prefix(files, "MK0_MLU")

    if label_path is None:
        continue

    label = cv2.imread(
        str(label_path),
        cv2.IMREAD_UNCHANGED,
    )

    if label is None or 2 not in np.unique(label):
        continue

    selected_samples.append(
        (event_id, sample_dir, files)
    )
    used_events.add(event_id)

    if len(selected_samples) == 3:
        break

for index, (event_id, sample_dir, files) in enumerate(
    selected_samples,
    start=1,
):
    pre1_vv = read_prefix(files, "SL1_IVV")
    pre1_vh = read_prefix(files, "SL1_IVH")
    pre2_vv = read_prefix(files, "SL2_IVV")
    pre2_vh = read_prefix(files, "SL2_IVH")
    post_vv = read_prefix(files, "MS1_IVV")
    post_vh = read_prefix(files, "MS1_IVH")
    label = read_prefix(files, "MK0_MLU")
    valid = read_prefix(files, "MK0_MNA")

    binary_flood = (label == 2).astype(np.uint8)
    binary_flood[valid != 1] = 0

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(12, 12),
    )

    sar_panels = [
        (pre1_vv, "Pre-1 VV"),
        (pre2_vv, "Pre-2 VV"),
        (post_vv, "Post VV"),
        (pre1_vh, "Pre-1 VH"),
        (pre2_vh, "Pre-2 VH"),
        (post_vh, "Post VH"),
    ]

    for axis, (image, title) in zip(
        axes.flat[:6],
        sar_panels,
    ):
        axis.imshow(display_sar(image), cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    axes[2, 0].imshow(
        label,
        cmap=label_cmap,
        vmin=0,
        vmax=3,
    )
    axes[2, 0].set_title(
        "Label: 0 NW, 1 PW, 2 Flood, 3 Invalid"
    )
    axes[2, 0].axis("off")

    axes[2, 1].imshow(
        binary_flood,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2, 1].set_title("Binary flood-change target")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(
        valid,
        cmap="gray",
    )
    axes[2, 2].set_title("Valid mask")
    axes[2, 2].axis("off")

    figure.suptitle(
        f"Kuro Siwo event {event_id}\n{sample_dir.name}"
    )
    figure.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / f"sample_{index}_event_{event_id}.png"
    )
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("Saved:", output_path)


no_overlap = all(
    len(values) == 0
    for values in overlaps.values()
)

split_pass = (
    no_overlap
    and len(unassigned_data_events) == 0
)

visualization_pass = len(selected_samples) == 3

print("\n==============================")
print("SPLIT AUDIT:", "PASS" if split_pass else "CHECK")
print(
    "LABEL SEMANTICS:",
    "PASS"
    if class3_marked_valid == 0
    else "CHECK",
)
print(
    "VISUALIZATIONS:",
    "PASS"
    if visualization_pass
    else "CHECK",
)
print("==============================")
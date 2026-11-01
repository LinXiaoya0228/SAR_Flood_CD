from pathlib import Path
from collections import Counter
import random

import cv2
import numpy as np
from compress_pickle import load


DATA_ROOT = Path.home() / "FYP_SARCD/datasets/KuroSiwo"
MANIFEST = (
    Path.home()
    / "FYP_SARCD/third_party/KuroSiwo/pickle/KuroV2_grid_dict_test_0_100.gz"
)

REQUIRED = [
    "MK0_MNA",  # valid mask
    "MS1_IVV",  # post-event VV
    "MS1_IVH",  # post-event VH
    "SL1_IVV",  # pre-event 1 VV
    "SL1_IVH",  # pre-event 1 VH
    "SL2_IVV",  # pre-event 2 VV
    "SL2_IVH",  # pre-event 2 VH
]


def find_prefix(files, prefix):
    for file in files:
        if file.name.startswith(prefix):
            return file
    return None


print("===== Loading manifest =====")
records = load(str(MANIFEST))
print("Manifest records:", len(records))

event_dirs = [
    path
    for path in DATA_ROOT.iterdir()
    if path.is_dir() and path.name.isdigit()
]
print("Top-level event directories:", len(event_dirs))

missing_directories = []
missing_components = []
zero_byte_files = []
readable_records = []
event_counts = Counter()

for record_id, record in records.items():
    relative_path = Path(record["path"])
    sample_dir = DATA_ROOT / relative_path

    # Handle a possible extra "data/" prefix in manifest paths.
    if (
        not sample_dir.is_dir()
        and relative_path.parts
        and relative_path.parts[0] == "data"
    ):
        sample_dir = DATA_ROOT.joinpath(*relative_path.parts[1:])

    info = record.get("info", {})
    event_id = str(
        info.get(
            "actid",
            relative_path.parts[0] if relative_path.parts else "unknown",
        )
    )
    event_counts[event_id] += 1

    if not sample_dir.is_dir():
        missing_directories.append((record_id, str(relative_path)))
        continue

    files = [path for path in sample_dir.iterdir() if path.is_file()]
    located = {
        prefix: find_prefix(files, prefix)
        for prefix in REQUIRED
    }

    missing = [
        prefix
        for prefix, path in located.items()
        if path is None
    ]

    if missing:
        missing_components.append(
            (record_id, str(relative_path), missing)
        )
        continue

    zero_files = [
        str(path)
        for path in located.values()
        if path.stat().st_size == 0
    ]

    if zero_files:
        zero_byte_files.extend(zero_files)
        continue

    label_path = find_prefix(files, "MK0_MLU")
    dem_path = find_prefix(files, "MK0_DEM")

    readable_records.append(
        {
            "record_id": record_id,
            "event_id": event_id,
            "sample_dir": sample_dir,
            "located": located,
            "label_path": label_path,
            "dem_path": dem_path,
        }
    )

print("\n===== Structural results =====")
print("Events represented in manifest:", len(event_counts))
print("Missing sample directories:", len(missing_directories))
print("Samples missing core components:", len(missing_components))
print("Zero-byte core files:", len(zero_byte_files))
print("Structurally valid records:", len(readable_records))

print("\nFirst missing directories:")
for item in missing_directories[:10]:
    print(item)

print("\nFirst missing-component cases:")
for item in missing_components[:10]:
    print(item)

print("\n===== Random readability test =====")

random.seed(20260916)
selected = random.sample(
    readable_records,
    min(100, len(readable_records)),
)

bad_reads = []
shape_mismatches = []
label_values = Counter()
labels_found = 0
dem_found = 0

for sample in selected:
    shapes = set()

    for prefix, path in sample["located"].items():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if image is None or image.size == 0:
            bad_reads.append(
                (sample["record_id"], prefix, str(path))
            )
            continue

        shapes.add(tuple(image.shape[:2]))

    if len(shapes) > 1:
        shape_mismatches.append(
            (sample["record_id"], sorted(shapes))
        )

    label_path = sample["label_path"]
    if label_path is not None:
        labels_found += 1
        label = cv2.imread(
            str(label_path),
            cv2.IMREAD_UNCHANGED,
        )

        if label is None or label.size == 0:
            bad_reads.append(
                (
                    sample["record_id"],
                    "MK0_MLU",
                    str(label_path),
                )
            )
        else:
            for value in np.unique(label):
                label_values[int(value)] += 1

    if sample["dem_path"] is not None:
        dem_found += 1

print("Random samples tested:", len(selected))
print("Unreadable files:", len(bad_reads))
print("Shape mismatches:", len(shape_mismatches))
print("Samples with labels:", labels_found)
print("Samples with DEM:", dem_found)
print("Observed label values:", dict(sorted(label_values.items())))

print("\nFirst unreadable files:")
for item in bad_reads[:10]:
    print(item)

print("\nFirst shape mismatches:")
for item in shape_mismatches[:10]:
    print(item)

passed = (
    len(event_dirs) == 43
    and len(missing_directories) == 0
    and len(missing_components) == 0
    and len(zero_byte_files) == 0
    and len(bad_reads) == 0
    and len(shape_mismatches) == 0
)

print("\n==============================")
print("STRUCTURAL AUDIT:", "PASS" if passed else "FAIL")
print("==============================")
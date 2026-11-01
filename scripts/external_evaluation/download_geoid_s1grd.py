#!/usr/bin/env python3
"""
Download and extract GEOID-Flood held-out Sentinel-1 GRD (s1grd) shards.
Downloads shards one-by-one, extracts to data/external_geoid/geoid-flood-heldout,
and removes the tarball after extraction to conserve disk space.
"""
import argparse
import os
import sys
import time
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "links-ads/geoid-flood"
TOTAL_SHARDS = 5


def main():
    parser = argparse.ArgumentParser(description="Download and extract GEOID-Flood held-out Sentinel-1 GRD shards.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data/external_geoid/geoid-flood-heldout"),
        help="Destination directory for extracted dataset",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="v1.1",
        help="Hugging Face dataset revision tag",
    )
    args = parser.parse_args()

    target_dir = Path(args.output_dir)
    base_dir = target_dir.parent
    staging_dir = base_dir / ".staging"

    base_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting download of {TOTAL_SHARDS} s1grd shards for geoid-flood-heldout...")
    t0 = time.time()

    for shard_idx in range(TOTAL_SHARDS):
        shard_filename = f"geoid-flood-heldout/shards/test/s1grd/test-s1grd-{shard_idx:05d}-of-{TOTAL_SHARDS:05d}.tar"
        print(f"\n[{shard_idx+1}/{TOTAL_SHARDS}] Downloading {shard_filename}...")
        t_shard_start = time.time()
        
        tar_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=shard_filename,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(staging_dir)
        )
        download_time = time.time() - t_shard_start
        file_size_gb = os.path.getsize(tar_path) / (1024 ** 3)
        print(f"  Downloaded {file_size_gb:.2f} GB in {download_time:.1f}s ({file_size_gb * 1024 / download_time:.1f} MB/s).")

        print(f"  Extracting to {target_dir}...")
        t_extract = time.time()
        with tarfile.open(tar_path) as tar:
            tar.extractall(target_dir)
        extract_time = time.time() - t_extract
        print(f"  Extracted in {extract_time:.1f}s.")

        print(f"  Removing tarball {tar_path}...")
        try:
            os.remove(tar_path)
        except OSError:
            pass
            print("  Tarball removed successfully.")
        except Exception as e:
            print(f"  Warning: failed to remove {tar_path}: {e}")

    total_time = time.time() - t0
    print(f"\nAll {TOTAL_SHARDS} shards downloaded and extracted successfully in {total_time/60:.2f} min!")

if __name__ == "__main__":
    main()


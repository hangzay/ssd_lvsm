#!/usr/bin/env python3
"""
Scan the DA3 cache root and write scene-id list files for each dataset.

Output files (one scene_id per line, only scenes with done.txt):
  <output_dir>/objaverse/full_list.txt      -- Objaverse training set
  <output_dir>/abo_render_f/full_list.txt   -- ABO test set (front)
  <output_dir>/abo_render_t/full_list.txt   -- ABO test set (top)
  <output_dir>/abo_render3/full_list.txt    -- ABO test set (3-view)
  <output_dir>/gso_render_f/full_list.txt   -- GSO test set (front)
  <output_dir>/gso_render6/full_list.txt    -- GSO test set (6-view)

Usage:
  python generate_scene_lists.py \
      --da3_root path_to_objaverse_da3_cache \
      --output_dir path_to_objaverse_scene_lists
"""

import argparse
import os
import re

# Objaverse split dirs look like "000-000", "000-001", ...
_SPLIT_RE = re.compile(r"^\d{3}-\d{3}$")

# ABO / GSO dataset sub-directories (flat layout: <da3_root>/<dataset>/<scene_id>/)
FLAT_DATASETS = [
    "abo_render_f",
    "abo_render_t",
    "abo_render3",
    "gso_render_f",
    "gso_render6",
]


def scan_objaverse(da3_objaverse_dir: str) -> list[str]:
    """Scan <da3_root>/objaverse/<split>/<uid>/done.txt  -> 'split/uid'."""
    results = []
    if not os.path.isdir(da3_objaverse_dir):
        print(f"[WARN] objaverse DA3 dir not found: {da3_objaverse_dir}")
        return results

    for split in sorted(os.listdir(da3_objaverse_dir)):
        split_dir = os.path.join(da3_objaverse_dir, split)
        if not os.path.isdir(split_dir) or not _SPLIT_RE.match(split):
            continue
        for uid in sorted(os.listdir(split_dir)):
            uid_dir = os.path.join(split_dir, uid)
            if os.path.isfile(os.path.join(uid_dir, "done.txt")):
                results.append(f"{split}/{uid}")

    return results


def scan_flat(da3_dataset_dir: str) -> list[str]:
    """Scan <da3_root>/<dataset>/<scene_id>/done.txt  -> 'scene_id'."""
    results = []
    if not os.path.isdir(da3_dataset_dir):
        print(f"[WARN] dataset DA3 dir not found: {da3_dataset_dir}")
        return results

    for scene_id in sorted(os.listdir(da3_dataset_dir)):
        scene_dir = os.path.join(da3_dataset_dir, scene_id)
        if os.path.isfile(os.path.join(scene_dir, "done.txt")):
            results.append(scene_id)

    return results


def write_list(path: str, entries: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(entries))
        if entries:
            f.write("\n")
    print(f"  -> {len(entries):>7,d} scenes  {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--da3_root",
        default="path_to_objaverse_da3_cache",
    )
    parser.add_argument(
        "--output_dir",
        default="./scene_lists",
    )
    args = parser.parse_args()

    print(f"DA3 root : {args.da3_root}")
    print(f"Output   : {args.output_dir}\n")

    # --- Objaverse ---
    objaverse_dir = os.path.join(args.da3_root, "objaverse")
    print(f"Scanning objaverse ...")
    ids = scan_objaverse(objaverse_dir)
    write_list(os.path.join(args.output_dir, "objaverse", "full_list.txt"), ids)

    # --- ABO / GSO ---
    for dataset in FLAT_DATASETS:
        dataset_dir = os.path.join(args.da3_root, dataset)
        print(f"Scanning {dataset} ...")
        ids = scan_flat(dataset_dir)
        write_list(os.path.join(args.output_dir, dataset, "full_list.txt"), ids)

    print("\nAll done.")


if __name__ == "__main__":
    main()

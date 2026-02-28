#!/usr/bin/env python3
"""Prepare ORIGA dataset in PaliGemma detection JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

PREFIX = "detect optic-disc ; optic-cup"
SPLIT_TO_JSONL = {
    "train": "_annotations.train.jsonl",
    "val": "_annotations.valid.jsonl",
    "test": "_annotations.test.jsonl",
}
DEFAULT_TARGET_SIZE = 448


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ORIGA masks into PaliGemma detection JSONL dataset."
    )
    parser.add_argument(
        "--origa-root",
        type=Path,
        default=Path("/home/ubuntu/glaucoma/dataset/ORIGA"),
        help="Path to ORIGA root containing train/val/test and Masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/teamlastmin/dataset/origa_paligemma/dataset"),
        help="Output directory for generated JSONL files and images/.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="Square output image size used for saved images and box encoding.",
    )
    return parser.parse_args()


def to_loc(value: int, size: int) -> int:
    if size <= 1:
        return 0
    loc = int(round((value / (size - 1)) * 1024))
    return max(0, min(1024, loc))


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    return y1, x1, y2, x2


def bbox_to_loc_tokens(bbox: tuple[int, int, int, int], h: int, w: int) -> str:
    y1, x1, y2, x2 = bbox
    return (
        f"<loc{to_loc(y1, h)}><loc{to_loc(x1, w)}>"
        f"<loc{to_loc(y2, h)}><loc{to_loc(x2, w)}>"
    )


def build_suffix(mask_array: np.ndarray) -> str | None:
    # ORIGA mask convention:
    # 0 = background, 1 = disc-only ring, 2 = cup region.
    # optic-disc uses (1 or 2), optic-cup uses (2 only).
    disc_mask = np.isin(mask_array, [1, 2])
    cup_mask = mask_array == 2

    disc_bbox = mask_to_bbox(disc_mask)
    cup_bbox = mask_to_bbox(cup_mask)
    if disc_bbox is None or cup_bbox is None:
        return None

    h, w = mask_array.shape[:2]
    disc_token = bbox_to_loc_tokens(disc_bbox, h, w)
    cup_token = bbox_to_loc_tokens(cup_bbox, h, w)
    return f"{disc_token} optic-disc ; {cup_token} optic-cup"


def load_aligned_mask(origa_root: Path, image_name: str, image_size: tuple[int, int]) -> np.ndarray | None:
    # Prefer square masks because split images are 512x512 square fundus crops.
    candidate_paths = [
        origa_root / "Masks_Square" / image_name,
        origa_root / "Masks" / image_name,
    ]
    for mask_path in candidate_paths:
        if not mask_path.exists():
            continue
        mask_img = Image.open(mask_path)
        if mask_img.size != image_size:
            # Nearest-neighbor keeps class ids (0/1/2) intact.
            mask_img = mask_img.resize(image_size, resample=Image.NEAREST)
        return np.array(mask_img)
    return None


def convert_split(
    origa_root: Path, output_dir: Path, split: str, target_size: int
) -> tuple[int, dict[str, int]]:
    src_split_dir = origa_root / split
    dst_split_dir = output_dir / "images" / split
    dst_split_dir.mkdir(parents=True, exist_ok=True)

    skipped = {"missing_mask": 0, "empty_mask": 0}
    kept = 0
    rows: list[dict[str, str]] = []

    for image_path in sorted(src_split_dir.glob("*.png")):
        image = Image.open(image_path).convert("RGB")
        image_size = image.size
        mask_array = load_aligned_mask(origa_root, image_path.name, image_size)
        if mask_array is None:
            skipped["missing_mask"] += 1
            continue

        target_hw = (target_size, target_size)
        if image.size != target_hw:
            image = image.resize(target_hw, resample=Image.BILINEAR)
            mask_array = np.array(
                Image.fromarray(mask_array).resize(target_hw, resample=Image.NEAREST)
            )

        suffix = build_suffix(mask_array)
        if suffix is None:
            skipped["empty_mask"] += 1
            continue

        dst_image_path = dst_split_dir / image_path.name
        if dst_image_path.is_symlink():
            # Replace legacy symlinks so resized images live inside the dataset folder.
            dst_image_path.unlink()
        image.save(dst_image_path)

        rows.append(
            {
                "image": f"images/{split}/{image_path.name}",
                "prefix": PREFIX,
                "suffix": suffix,
            }
        )
        kept += 1

    jsonl_path = output_dir / SPLIT_TO_JSONL[split]
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return kept, skipped


def main() -> None:
    args = parse_args()
    if args.target_size <= 0:
        raise ValueError("--target-size must be a positive integer.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        count, skipped = convert_split(
            args.origa_root, args.output_dir, split, args.target_size
        )
        split_counts[split] = count
        print(
            f"{split}: kept={count} "
            f"skipped_missing_mask={skipped['missing_mask']} "
            f"skipped_empty_mask={skipped['empty_mask']}"
        )

    print("Converted ORIGA to detection JSONL.")
    print(
        f"train={split_counts['train']}  "
        f"val={split_counts['val']}  "
        f"test={split_counts['test']}"
    )
    print(f"Dataset saved to: {args.output_dir} (image size {args.target_size}x{args.target_size})")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Run one-shot PaliGemma LoRA inference on a single image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.tools.paligemma_tool import run_paligemma_detection

# PaliGemma loc tokens span [0, 1024) on each axis.
LOC_SCALE = 1024

# Per-label colours (BGR-independent; PIL uses RGB tuples).
_LABEL_COLOURS: list[tuple[int, int, int]] = [
    (255, 80, 80),   # red   – first label (optic-disc)
    (80, 200, 80),   # green – second label (optic-cup)
    (80, 160, 255),  # blue  – any extra labels
    (255, 200, 0),   # yellow
    (200, 0, 255),   # purple
]


def _colour_for(label: str, seen: dict[str, int]) -> tuple[int, int, int]:
    if label not in seen:
        seen[label] = len(seen)
    return _LABEL_COLOURS[seen[label] % len(_LABEL_COLOURS)]


def draw_boxes(
    image: Image.Image,
    boxes: list[dict[str, Any]],
    *,
    line_width: int = 3,
    font_size: int = 18,
) -> Image.Image:
    """Return a copy of *image* with bounding boxes drawn on it."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    img_w, img_h = out.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    label_colours: dict[str, tuple[int, int, int]] = {}
    for box in boxes:
        label = box.get("label", "obj")
        colour = _colour_for(label, label_colours)

        # Scale from 0-1024 loc-token space to pixel space.
        x1 = int(box["x1"] / LOC_SCALE * img_w)
        y1 = int(box["y1"] / LOC_SCALE * img_h)
        x2 = int(box["x2"] / LOC_SCALE * img_w)
        y2 = int(box["y2"] / LOC_SCALE * img_h)

        draw.rectangle([x1, y1, x2, y2], outline=colour, width=line_width)

        # Label background + text.
        text = label
        bbox = draw.textbbox((x1, y1), text, font=font)
        pad = 2
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=colour,
        )
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-image inference using a fine-tuned PaliGemma LoRA adapter."
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("/home/ubuntu/teamlastmin/finetuned_paligemma2_det_lora"),
        help="Path to LoRA dir (root, final, or checkpoint directory).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="detect optic-disc ; optic-cup",
        help="Prompt/query context for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save output JSON.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help=(
            "Path to save the overlay image with bounding boxes drawn. "
            "If omitted, defaults to <input_stem>_overlay.<ext> in the same directory."
        ),
    )
    return parser.parse_args()


def resolve_adapter_dir(adapter_dir: Path) -> Path:
    if (adapter_dir / "adapter_config.json").exists():
        return adapter_dir

    final_dir = adapter_dir / "final"
    if final_dir.exists() and (final_dir / "adapter_config.json").exists():
        return final_dir

    candidates = [
        p for p in adapter_dir.iterdir() if p.is_dir() and (p / "adapter_config.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No adapter directory found under {adapter_dir}. "
            "Expected adapter_config.json in this path, in 'final/', or in a checkpoint folder."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_dir}")

    selected_adapter = resolve_adapter_dir(args.adapter_dir)
    result = run_paligemma_detection(
        image_path=str(args.image),
        query_context=args.prompt,
        max_new_tokens=args.max_new_tokens,
        adapter_dir=selected_adapter,
    )

    result["input_image"] = str(args.image)
    result["selected_adapter"] = str(selected_adapter)

    payload = json.dumps(result, indent=2)
    print(payload)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(payload + "\n", encoding="utf-8")
        print(f"Saved JSON to: {args.save_json}")

    # Determine output image path.
    if args.output_image is not None:
        overlay_path = args.output_image
    else:
        overlay_path = args.image.parent / (args.image.stem + "_overlay" + args.image.suffix)

    image = Image.open(args.image).convert("RGB")
    boxes = result.get("boxes", [])
    overlay = draw_boxes(image, boxes)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(str(overlay_path))
    print(f"Saved overlay image ({len(boxes)} box(es)) to: {overlay_path}")


if __name__ == "__main__":
    main()

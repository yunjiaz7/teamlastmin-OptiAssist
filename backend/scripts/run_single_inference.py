#!/usr/bin/env python3
"""Run one-shot PaliGemma LoRA inference on a single image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.tools.paligemma_tool import run_paligemma_detection


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

    # Add CLI context fields for easier debugging/reproducibility.
    result["input_image"] = str(args.image)
    result["selected_adapter"] = str(selected_adapter)

    payload = json.dumps(result, indent=2)
    print(payload)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(payload + "\n", encoding="utf-8")
        print(f"Saved JSON to: {args.save_json}")


if __name__ == "__main__":
    main()

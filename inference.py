#!/usr/bin/env python3
"""Run PaliGemma LoRA inference on a few images and save overlays."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

LOC_RE = re.compile(r"<loc(\d+)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained PaliGemma LoRA adapter."
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("/home/ubuntu/teamlastmin/finetuned_paligemma2_det_lora/final"),
        help="Directory containing adapter_config.json and adapter_model.safetensors.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "/home/ubuntu/teamlastmin/dataset/origa_paligemma/dataset/_annotations.train.jsonl"
        ),
        help="JSONL file used to pick sample images/prompts.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/home/ubuntu/teamlastmin/dataset/origa_paligemma/dataset"),
        help="Root directory prepended to each annotation image path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="How many samples to run.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Generation length for model outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/teamlastmin/outputs/inference_samples"),
        help="Directory to save visualizations and prediction report.",
    )
    return parser.parse_args()


def load_samples(jsonl_path: Path, count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= count:
                break
            row = json.loads(line)
            if "image" in row and "prefix" in row:
                rows.append(row)
    if not rows:
        raise ValueError(f"No usable records found in {jsonl_path}")
    return rows


def pick_device_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return device, dtype
    return torch.device("cpu"), torch.float32


def parse_prediction_boxes(prediction: str) -> list[tuple[int, int, int, int, str]]:
    boxes: list[tuple[int, int, int, int, str]] = []
    parts = [part.strip() for part in prediction.split(";") if part.strip()]
    for part in parts:
        locs = [int(x) for x in LOC_RE.findall(part)]
        if len(locs) < 4:
            continue
        y1, x1, y2, x2 = locs[:4]
        label = part.split(">")[-1].strip() or "obj"
        boxes.append((y1, x1, y2, x2, label))
    return boxes


def is_oversized_box(
    y1: int, x1: int, y2: int, x2: int, *, area_threshold: float = 0.85
) -> bool:
    y_min, y_max = sorted((max(0, min(1024, y1)), max(0, min(1024, y2))))
    x_min, x_max = sorted((max(0, min(1024, x1)), max(0, min(1024, x2))))
    height = max(0, y_max - y_min)
    width = max(0, x_max - x_min)
    area_ratio = (height * width) / float(1024 * 1024)
    return area_ratio >= area_threshold


def loc_to_px(v: int, size: int) -> int:
    v = max(0, min(1024, v))
    return int(round((v / 1024.0) * (size - 1)))


def strip_prompt_prefix(decoded: str, prompt: str) -> str:
    clean = decoded.strip()
    if clean.startswith(prompt):
        return clean[len(prompt) :].strip(" \n:")
    # Some tokenizers drop/normalize whitespace around <image>.
    prompt_no_image = prompt.replace("<image>", "", 1).strip()
    if prompt_no_image and clean.startswith(prompt_no_image):
        return clean[len(prompt_no_image) :].strip(" \n:")
    return clean


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config_path = args.adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {adapter_config_path}")
    adapter_cfg = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model_id = adapter_cfg["base_model_name_or_path"]

    samples = load_samples(args.annotations, args.num_samples)
    device, dtype = pick_device_dtype()

    processor_source = args.adapter_dir if (args.adapter_dir / "preprocessor_config.json").exists() else base_model_id
    processor = AutoProcessor.from_pretrained(processor_source, local_files_only=True)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir), local_files_only=True)
    model.to(device)
    model.eval()

    rows: list[dict[str, str | int]] = []
    for idx, sample in enumerate(samples, start=1):
        image_path = args.image_root / sample["image"]
        image = Image.open(image_path).convert("RGB")
        prompt = "<image> " + sample["prefix"]

        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        decoded = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        prediction = strip_prompt_prefix(decoded, prompt)
        boxes = parse_prediction_boxes(prediction)
        oversized_count = sum(is_oversized_box(y1, x1, y2, x2) for y1, x1, y2, x2, _ in boxes)

        vis = image.copy()
        draw = ImageDraw.Draw(vis)
        w, h = vis.size
        for y1, x1, y2, x2, label in boxes:
            x1p, y1p = loc_to_px(x1, w), loc_to_px(y1, h)
            x2p, y2p = loc_to_px(x2, w), loc_to_px(y2, h)
            x_min, x_max = sorted((x1p, x2p))
            y_min, y_max = sorted((y1p, y2p))
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=3)
            draw.text((x_min + 2, max(0, y_min - 12)), label, fill=(255, 0, 0))

        out_image = args.output_dir / f"sample_{idx:02d}.png"
        vis.save(out_image)
        rows.append(
            {
                "sample": idx,
                "source_image": sample["image"],
                "prompt": sample["prefix"],
                "raw_decoded": decoded,
                "prediction": prediction,
                "predicted_boxes": len(boxes),
                "oversized_boxes": oversized_count,
                "saved_image": out_image.name,
            }
        )

    results_json = args.output_dir / "results.json"
    results_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    md_lines = [
        "# Inference Samples",
        "",
        f"- Adapter: `{args.adapter_dir}`",
        f"- Base model: `{base_model_id}`",
        f"- Processor source: `{processor_source}`",
        f"- Device/dtype: `{device}` / `{dtype}`",
        f"- Samples: `{len(rows)}`",
        "",
    ]
    for row in rows:
        md_lines += [
            f"## Sample {row['sample']}",
            f"- Source: `{row['source_image']}`",
            f"- Prompt: `{row['prompt']}`",
            f"- Prediction: `{row['prediction']}`",
            f"- Predicted boxes: `{row['predicted_boxes']}`",
            f"- Oversized boxes: `{row['oversized_boxes']}`",
            "",
            f"![sample-{row['sample']}]({row['saved_image']})",
            "",
        ]
    (args.output_dir / "results.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

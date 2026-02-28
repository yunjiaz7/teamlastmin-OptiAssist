from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

LOC_RE = re.compile(r"<loc(\d+)>")
DEFAULT_ADAPTER_ROOT = Path("/home/ubuntu/teamlastmin/finetuned_paligemma2_det_lora")
CANONICAL_DETECTION_PROMPT = "detect optic-disc ; optic-cup"

_MODEL_BUNDLE: dict[str, Any] | None = None


def _pick_device_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return device, dtype
    return torch.device("cpu"), torch.float32


def _select_latest_adapter_dir(adapter_root: Path = DEFAULT_ADAPTER_ROOT) -> Path:
    if not adapter_root.exists():
        raise FileNotFoundError(f"Adapter root does not exist: {adapter_root}")

    candidates = [p for p in adapter_root.iterdir() if p.is_dir() and (p / "adapter_config.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No adapter directories with adapter_config.json in {adapter_root}")

    # Choose newest directory by mtime so "latest fine-tuned model" is automatic.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_bundle(adapter_dir: Path | None = None) -> dict[str, Any]:
    selected_adapter = adapter_dir or _select_latest_adapter_dir()
    adapter_config_path = selected_adapter / "adapter_config.json"
    adapter_cfg = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model_id = adapter_cfg["base_model_name_or_path"]

    device, dtype = _pick_device_dtype()
    processor_source = selected_adapter if (selected_adapter / "preprocessor_config.json").exists() else base_model_id

    processor = AutoProcessor.from_pretrained(processor_source, local_files_only=True)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, str(selected_adapter), local_files_only=True)
    model.to(device)
    model.eval()

    return {
        "adapter_dir": selected_adapter,
        "base_model_id": base_model_id,
        "processor": processor,
        "model": model,
        "device": device,
        "dtype": dtype,
    }


def get_paligemma_bundle(adapter_dir: Path | None = None) -> dict[str, Any]:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        _MODEL_BUNDLE = _load_bundle(adapter_dir=adapter_dir)
    return _MODEL_BUNDLE


def _strip_prompt_prefix(decoded: str, prompt: str) -> str:
    clean = decoded.strip()
    if clean.startswith(prompt):
        return clean[len(prompt) :].strip(" \n:")
    prompt_no_image = prompt.replace("<image>", "", 1).strip()
    if prompt_no_image and clean.startswith(prompt_no_image):
        return clean[len(prompt_no_image) :].strip(" \n:")
    return clean


def _parse_prediction_boxes(prediction: str) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    parts = [part.strip() for part in prediction.split(";") if part.strip()]
    for part in parts:
        locs = [int(x) for x in LOC_RE.findall(part)]
        if len(locs) < 4:
            continue
        y1, x1, y2, x2 = locs[:4]
        label = part.split(">")[-1].strip() or "obj"
        boxes.append(
            {
                "label": label,
                "y1": y1,
                "x1": x1,
                "y2": y2,
                "x2": x2,
            }
        )
    return boxes


def run_paligemma_detection(
    image_path: str,
    query_context: str,
    *,
    max_new_tokens: int = 128,
    adapter_dir: Path | None = None,
) -> dict[str, Any]:
    bundle = get_paligemma_bundle(adapter_dir=adapter_dir)
    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]

    image = Image.open(image_path).convert("RGB")
    prompt_text = (query_context or "").strip() or CANONICAL_DETECTION_PROMPT

    def _generate_for_prompt(text_prompt: str) -> tuple[str, str, list[dict[str, Any]]]:
        full_prompt = "<image> " + text_prompt
        inputs = processor(text=full_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        decoded_local = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        prediction_local = _strip_prompt_prefix(decoded_local, full_prompt)
        boxes_local = _parse_prediction_boxes(prediction_local)
        return decoded_local, prediction_local, boxes_local

    decoded, prediction, boxes = _generate_for_prompt(prompt_text)

    # If a free-form query prompt yields no location tokens, retry once with the
    # training-time canonical prompt to keep detection behavior robust.
    used_fallback_prompt = False
    if not boxes and prompt_text != CANONICAL_DETECTION_PROMPT:
        decoded, prediction, boxes = _generate_for_prompt(CANONICAL_DETECTION_PROMPT)
        used_fallback_prompt = True

    return {
        "tool": "paligemma_lora_latest",
        "adapter_dir": str(bundle["adapter_dir"]),
        "base_model": bundle["base_model_id"],
        "prompt": prompt_text,
        "canonical_prompt": CANONICAL_DETECTION_PROMPT,
        "used_fallback_prompt": used_fallback_prompt,
        "prediction": prediction,
        "boxes": boxes,
        "box_count": len(boxes),
    }

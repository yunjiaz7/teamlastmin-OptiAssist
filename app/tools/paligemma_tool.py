"""
paligemma_tool.py

Run detection inference with a fine-tuned PaliGemma 2 LoRA adapter.
Compatible with run_single_inference.py CLI helper.

Usage (from segmenter or CLI):
    from app.tools.paligemma_tool import run_paligemma_detection

    result = run_paligemma_detection(
        image_path="path/to/image.jpg",
        query_context="detect optic-disc ; optic-cup",
        max_new_tokens=128,
        adapter_dir=Path("path/to/finetuned_paligemma2_det_lora/final"),
    )
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import traceback
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

logger = logging.getLogger(__name__)

# Project-local fine-tuned adapter directory.
_LOCAL_ADAPTER_FINAL = (
    Path(__file__).parent.parent.parent
    / "backend"
    / "models"
    / "paligemma2-finetuned"
    / "finetuned_paligemma2_det_lora"
    / "final"
)

# Fallback: official PaliGemma 2 mix model (supports the same <loc####>
# detection prompt format without needing a LoRA adapter).
_BASE_MIX_MODEL_ID = "google/paligemma2-3b-mix-448"

# ---------------------------------------------------------------------------
# Internal cache so repeated calls don't reload the model from disk
# ---------------------------------------------------------------------------
_LOADED: dict = {}   # keys: "processor", "model", "adapter_path", "base_id", "device", "dtype"


def _pick_runtime_settings() -> tuple[torch.device, torch.dtype, Optional[str]]:
    # Keep runtime settings aligned with `backend/scripts/train_paligemma.py`.
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16, "auto"
    return torch.device("cpu"), torch.float32, None


def _load_model_and_processor(adapter_dir: Path) -> tuple:
    """
    Load (processor, model) for the given adapter, using an in-process cache
    so subsequent calls skip the expensive from_pretrained step.

    If ``adapter_config.json`` is not found inside ``adapter_dir``, falls back
    to the official ``google/paligemma2-3b-mix-448`` model which supports the
    same ``<loc####>`` detection prompt format without a LoRA adapter.
    """
    adapter_key = str(adapter_dir)
    if _LOADED.get("adapter_path") == adapter_key:
        return _LOADED["processor"], _LOADED["model"], _LOADED["device"], _LOADED["dtype"]

    device, dtype, device_map = _pick_runtime_settings()
    config_path = adapter_dir / "adapter_config.json"

    if config_path.exists():
        # ----------------------------------------------------------------
        # Path A: fine-tuned LoRA adapter
        # ----------------------------------------------------------------
        with config_path.open() as f:
            adapter_cfg = json.load(f)

        base_id: str = adapter_cfg.get("base_model_name_or_path", "google/paligemma2-3b-pt-448")
        logger.info("Loading fine-tuned model. base=%s adapter=%s", base_id, adapter_dir)

        processor = PaliGemmaProcessor.from_pretrained(base_id, use_fast=True)
        logger.info("Processor loaded from %s", base_id)

        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_id,
            device_map=device_map,
            torch_dtype=dtype,
        )
        if device_map is None:
            base_model = base_model.to(device)

        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model = model.merge_and_unload()
        if device_map is None:
            model = model.to(device)
        model.eval()
        logger.info("LoRA adapter merged. Fine-tuned model ready.")

    else:
        # ----------------------------------------------------------------
        # Path B: adapter files missing — fall back to official mix model.
        # google/paligemma2-3b-mix-448 was trained on detection tasks and
        # natively supports the <loc####> token format used here.
        # ----------------------------------------------------------------
        logger.warning(
            "adapter_config.json not found in %s — falling back to %s",
            adapter_dir,
            _BASE_MIX_MODEL_ID,
        )

        processor = PaliGemmaProcessor.from_pretrained(_BASE_MIX_MODEL_ID, use_fast=True)
        logger.info("Fallback processor loaded from %s", _BASE_MIX_MODEL_ID)

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            _BASE_MIX_MODEL_ID,
            device_map=device_map,
            torch_dtype=dtype,
        )
        if device_map is None:
            model = model.to(device)
        model.eval()
        logger.info("Fallback model loaded: %s", _BASE_MIX_MODEL_ID)

    _LOADED["processor"] = processor
    _LOADED["model"] = model
    _LOADED["adapter_path"] = adapter_key
    _LOADED["device"] = device
    _LOADED["dtype"] = dtype

    return processor, model, device, dtype


# ---------------------------------------------------------------------------
# Regex that captures <loc####> tokens (PaliGemma detection format)
# ---------------------------------------------------------------------------
_LOC_RE = re.compile(r"<loc(\d+)>")


def _loc_to_px(v: int, size: int) -> int:
    v = max(0, min(1024, int(v)))
    return int(round((v / 1024.0) * (size - 1)))


def _parse_detections(raw: str, w: int, h: int) -> list[dict]:
    """Convert raw <loc####> token output into pixel-space bounding boxes."""
    detections = []
    parts = [part.strip() for part in raw.split(";") if part.strip()]
    for idx, part in enumerate(parts, start=1):
        locs = [int(x) for x in _LOC_RE.findall(part)]
        if len(locs) < 4:
            continue
        y0, x0, y1, x1 = locs[:4]
        x_min, x_max = sorted((_loc_to_px(x0, w), _loc_to_px(x1, w)))
        y_min, y_max = sorted((_loc_to_px(y0, h), _loc_to_px(y1, h)))
        label = part.split(">")[-1].strip() or f"region_{idx}"
        box = {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }
        detections.append({"label": label, "bounding_box": box})
    return detections


def _annotate(image: Image.Image, detections: list[dict]) -> str:
    """Draw bounding boxes and return a base64 PNG data-URI."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for det in detections:
        b = det["bounding_box"]
        draw.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]],
                       outline="red", width=2)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_paligemma_detection(
    image_path: str,
    query_context: str,
    max_new_tokens: int = 128,
    adapter_dir: Optional[Path] = None,
) -> dict:
    """
    Run PaliGemma 2 LoRA detection on a single image.

    Args:
        image_path:     Path to the input image file.
        query_context:  Detection prompt, e.g. "detect optic-disc ; optic-cup".
        max_new_tokens: Maximum tokens to generate.
        adapter_dir:    Path to the LoRA adapter directory (must contain
                        adapter_config.json).  Defaults to the project-local
                        finetuned_paligemma2_det_lora/final path.

    Returns:
        dict with keys:
            raw_output              (str)  — decoded model output
            detections              (list) — list of {label, bounding_box} dicts
            annotated_image_base64  (str)  — base64 PNG with boxes drawn
            summary                 (str)  — human-readable detection summary
    """
    if adapter_dir is None:
        adapter_dir = _LOCAL_ADAPTER_FINAL
    else:
        adapter_dir = Path(adapter_dir)

    try:
        processor, model, device, dtype = _load_model_and_processor(adapter_dir)
    except Exception:
        logger.error("Model load failed:\n%s", traceback.format_exc())
        raise

    image = Image.open(image_path).convert("RGB")
    w, h = image.size  # keep originals for bounding-box rescaling

    # Keep prompt format aligned with training (`<image> detect ...`).
    prompt_text = query_context.strip()
    if not prompt_text.startswith("detect"):
        prompt_text = f"detect {prompt_text}"
    prompt = prompt_text if prompt_text.startswith("<image>") else f"<image> {prompt_text}"
    logger.info("Running inference. prompt=%r image=%dx%d", prompt, w, h)

    inputs = processor(text=prompt, images=image, return_tensors="pt", truncation=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    for key, value in inputs.items():
        if torch.is_floating_point(value):
            inputs[key] = value.to(dtype)
    input_len = inputs["input_ids"].shape[-1]
    logger.info("Processor keys: %s", list(inputs.keys()))

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = outputs[0][input_len:]
    raw_output = processor.decode(generated, skip_special_tokens=True).strip()
    logger.info("Raw output: %s", raw_output[:300])

    detections = _parse_detections(raw_output, w, h)
    annotated = _annotate(image, detections)
    summary = (
        f"{len(detections)} region(s) detected."
        if detections else "No regions detected."
    )

    return {
        "raw_output": raw_output,
        "detections": detections,
        "annotated_image_base64": annotated,
        "summary": summary,
    }

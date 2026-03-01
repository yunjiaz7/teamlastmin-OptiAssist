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
from transformers import AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal cache so repeated calls don't reload the model from disk
# ---------------------------------------------------------------------------
_LOADED: dict = {}   # keys: "processor", "model", "adapter_path", "base_id"


def _find_cached_base(preferred: str, fallback: str) -> str:
    """Return whichever of preferred / fallback is already in the HF cache."""
    try:
        import huggingface_hub
        cached = {r.repo_id for r in huggingface_hub.scan_cache_dir().repos}
        if preferred in cached:
            return preferred
        if fallback in cached:
            logger.warning(
                "%s not in HF cache — falling back to %s", preferred, fallback
            )
            return fallback
    except Exception:
        pass
    return preferred


def _load_model_and_processor(adapter_dir: Path) -> tuple:
    """
    Load (processor, model) for the given adapter, using an in-process cache
    so subsequent calls skip the expensive from_pretrained step.
    """
    adapter_key = str(adapter_dir)
    if _LOADED.get("adapter_path") == adapter_key:
        return _LOADED["processor"], _LOADED["model"]

    # ----------------------------------------------------------------
    # 1. Read the adapter config to discover the base model
    # ----------------------------------------------------------------
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")

    with config_path.open() as f:
        adapter_cfg = json.load(f)

    declared_base: str = adapter_cfg.get("base_model_name_or_path", "google/paligemma2-3b-pt-448")
    fallback_base = "google/paligemma2-3b-pt-224"
    base_id = _find_cached_base(declared_base, fallback_base)
    logger.info("Loading base model: %s", base_id)

    # ----------------------------------------------------------------
    # 2. Processor — always from the HF base model so resolution matches
    # ----------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(base_id)
    logger.info("Processor loaded from %s", base_id)

    # ----------------------------------------------------------------
    # 3. Base model + LoRA adapter
    # ----------------------------------------------------------------
    base_model = AutoModelForVision2Seq.from_pretrained(base_id, dtype=torch.float32)
    logger.info("Base model loaded.")

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model = model.merge_and_unload()
    model.eval()
    logger.info("LoRA adapter merged. Model ready.")

    _LOADED["processor"] = processor
    _LOADED["model"] = model
    _LOADED["adapter_path"] = adapter_key
    _LOADED["base_id"] = base_id

    return processor, model


# ---------------------------------------------------------------------------
# Regex that captures <loc####> tokens (PaliGemma detection format)
# ---------------------------------------------------------------------------
_LOC_RE = re.compile(r"<loc(\d{4})>")


def _parse_detections(raw: str, w: int, h: int) -> list[dict]:
    """Convert raw <loc####> token output into pixel-space bounding boxes."""
    tokens = _LOC_RE.findall(raw)
    detections = []
    for i in range(0, len(tokens) - 3, 4):
        y0, x0, y1, x1 = (int(tokens[i + j]) for j in range(4))
        box = {
            "x_min": int(x0 / 1024 * w),
            "y_min": int(y0 / 1024 * h),
            "x_max": int(x1 / 1024 * w),
            "y_max": int(y1 / 1024 * h),
        }
        # Label: text after the 4th loc token, before the next special token
        parts = re.split(r"(?:<loc\d{4}>){4}", raw)
        label = ""
        idx = i // 4 + 1
        if idx < len(parts):
            candidate = re.split(r"<", parts[idx])[0].strip()
            label = candidate or f"region_{idx}"
        detections.append({"label": label or f"region_{i//4+1}", "bounding_box": box})
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
        adapter_dir = (
            Path(__file__).parent.parent.parent
            / "backend" / "models" / "paligemma2-finetuned"
            / "finetuned_paligemma2_det_lora" / "final"
        )

    try:
        processor, model = _load_model_and_processor(adapter_dir)
    except Exception:
        logger.error("Model load failed:\n%s", traceback.format_exc())
        raise

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # PaliGemma detection prompt always starts with "detect"
    prompt = query_context if query_context.startswith("detect") else f"detect {query_context}"
    logger.info("Running inference. prompt=%r image=%dx%d", prompt, w, h)

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]
    logger.info("Processor keys: %s", list(inputs.keys()))

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
        )

    generated = outputs[0][input_len:]
    raw_output = processor.decode(generated, skip_special_tokens=False)
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

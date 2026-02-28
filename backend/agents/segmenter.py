from __future__ import annotations

"""
segmenter.py

Runs PaliGemma 2 inference for retinal image segmentation. Accepts raw image
bytes and a location query, returns detected anatomical regions with bounding
boxes drawn on an annotated copy of the original image.

Mask decoding (from <seg> tokens) requires vae-oid.npz — coordinate with teammate.
Until that asset is available, has_mask is always set to False.
"""

import asyncio
import base64
import io
import logging
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

logger = logging.getLogger(__name__)

MODEL_PATH = "./models/paligemma2-finetuned"

# ---------------------------------------------------------------------------
# Model loading — happens once at module level to avoid per-call overhead
# ---------------------------------------------------------------------------
_model_path = Path(MODEL_PATH)
if not _model_path.exists():
    raise FileNotFoundError(
        f"PaliGemma 2 model not found at '{MODEL_PATH}'. "
        "Please download or fine-tune the model and place it at that path."
    )

logger.info("Loading PaliGemma 2 processor and model from %s", MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_PATH)
logger.info("PaliGemma 2 loaded successfully.")

# Regex to capture all <loc####> tokens in sequence
_LOC_PATTERN = re.compile(r"<loc(\d{4})>")


def _run_inference(pil_image: Image.Image, prompt: str) -> str:
    """
    Execute blocking PaliGemma 2 inference synchronously.

    Args:
        pil_image: PIL Image object of the retinal scan.
        prompt: Text prompt for the model (e.g. "segment optic disc\\n").

    Returns:
        Raw decoded output string including special tokens.
    """
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    raw_output = processor.decode(outputs[0], skip_special_tokens=False)
    return raw_output


def _parse_detections(raw_output: str, img_width: int, img_height: int) -> list[dict]:
    """
    Parse <loc####> tokens from raw PaliGemma output into structured detections.

    PaliGemma encodes bounding boxes as four consecutive <loc####> tokens in the
    order: y_min, x_min, y_max, x_max. Each token value is in [0, 1023] and must
    be scaled to the actual image dimensions.

    Args:
        raw_output: Full decoded model output string with special tokens intact.
        img_width:  Width of the original image in pixels.
        img_height: Height of the original image in pixels.

    Returns:
        List of detection dicts, each containing label, bounding_box, and has_mask.
    """
    loc_tokens = _LOC_PATTERN.findall(raw_output)
    detections = []

    # Group loc tokens in blocks of 4; anything left over is incomplete and skipped
    for i in range(0, len(loc_tokens) - 3, 4):
        y_min_raw, x_min_raw, y_max_raw, x_max_raw = (int(loc_tokens[i + j]) for j in range(4))

        y_min = int((y_min_raw / 1024) * img_height)
        x_min = int((x_min_raw / 1024) * img_width)
        y_max = int((y_max_raw / 1024) * img_height)
        x_max = int((x_max_raw / 1024) * img_width)

        # Attempt to extract a label string following the 4th loc token in the raw text
        after_locs = re.split(r"(?:<loc\d{4}>){4}", raw_output)
        label_text = ""
        label_idx = i // 4 + 1
        if label_idx < len(after_locs):
            # Take only the first word/phrase before the next special token
            label_candidate = re.split(r"<", after_locs[label_idx])[0].strip()
            label_text = label_candidate if label_candidate else f"region_{label_idx}"

        # <seg> tokens indicate mask availability; decoding requires vae-oid.npz (TBD)
        has_mask = False

        detections.append(
            {
                "label": label_text or f"region_{i // 4 + 1}",
                "confidence": 0.9,
                "bounding_box": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                },
                "has_mask": has_mask,
            }
        )

    return detections


def _draw_boxes(pil_image: Image.Image, detections: list[dict]) -> str:
    """
    Draw red bounding boxes on the image and encode it as a base64 PNG data URI.

    Args:
        pil_image:  Original PIL Image to annotate.
        detections: List of detection dicts produced by _parse_detections().

    Returns:
        Base64-encoded PNG string prefixed with the data URI scheme.
    """
    annotated = pil_image.copy()
    draw = ImageDraw.Draw(annotated)

    for det in detections:
        box = det["bounding_box"]
        draw.rectangle(
            [box["x_min"], box["y_min"], box["x_max"], box["y_max"]],
            outline="red",
            width=2,
        )

    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _build_summary(detections: list[dict]) -> str:
    """
    Produce a human-readable summary of detected regions.

    Args:
        detections: List of detection dicts.

    Returns:
        A plain-text summary string.
    """
    if not detections:
        return "No regions detected."

    labels = [d["label"] for d in detections]
    label_phrases = ", ".join(f"{lbl} in region" for lbl in labels)
    return f"{len(detections)} regions detected: {label_phrases}"


async def run_segmentation(image_bytes: bytes, query: str) -> dict:
    """
    Segment anatomical structures in a retinal image using PaliGemma 2.

    Args:
        image_bytes: Raw bytes of the input retinal image.
        query: Location query describing what to segment (e.g. "optic disc").

    Returns:
        A dict with keys:
            "detections"            (list): Detected regions with labels and bounding boxes.
            "annotated_image_base64" (str): Base64 PNG of the image with boxes drawn.
            "summary"               (str): Human-readable detection summary.

    Raises:
        RuntimeError: If inference or post-processing fails unexpectedly.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to decode image bytes into PIL Image: {e}") from e

    img_width, img_height = pil_image.size
    prompt = f"segment {query}\n"

    logger.info("Running segmentation. query=%s image_size=%dx%d", query, img_width, img_height)

    try:
        # Offload blocking transformer inference to a thread so the event loop stays free
        raw_output = await asyncio.to_thread(_run_inference, pil_image, prompt)
    except Exception as e:
        raise RuntimeError(f"PaliGemma 2 inference failed: {e}") from e

    logger.debug("Raw model output: %s", raw_output[:200])

    detections = _parse_detections(raw_output, img_width, img_height)
    annotated_image_base64 = _draw_boxes(pil_image, detections)
    summary = _build_summary(detections)

    logger.info("Segmentation complete. detections=%d", len(detections))

    return {
        "detections": detections,
        "annotated_image_base64": annotated_image_base64,
        "summary": summary,
    }

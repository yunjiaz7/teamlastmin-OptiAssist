from __future__ import annotations

"""
diagnostician.py

Runs MedGemma inference to produce a structured medical diagnosis for a retinal
image. Accepts raw image bytes and a clinical question, and returns a parsed JSON
dict containing condition, severity, findings, and recommendations.
"""

import asyncio
import io
import json
import logging
from pathlib import Path

from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_PATH = "./models/medgemma-finetuned"

SYSTEM_PROMPT = (
    "You are an expert ophthalmology AI assistant. "
    "Analyze the retinal image and answer the clinical question. "
    "Always respond with valid JSON only, no extra text. "
    "JSON fields required:\n"
    "  condition: string (disease name or 'Normal')\n"
    "  severity: string (None/Mild/Moderate/Severe/Proliferative)\n"
    "  severity_level: integer 0-4\n"
    "  confidence: float 0.0-1.0\n"
    "  findings: list of strings (specific observations)\n"
    "  recommendation: string (follow-up advice)\n"
    "  disclaimer: always set to "
    "'For research use only. Not intended for clinical diagnosis.'"
)

FALLBACK_RESULT = {
    "condition": "Analysis unavailable",
    "severity": "None",
    "severity_level": 0,
    "confidence": 0.0,
    "findings": [],
    "recommendation": "Please retry or consult a qualified ophthalmologist.",
    "disclaimer": "For research use only. Not intended for clinical diagnosis.",
}

# ---------------------------------------------------------------------------
# Model loading — happens once at module level to avoid per-call overhead
# ---------------------------------------------------------------------------
_model_path = Path(MODEL_PATH)
if not _model_path.exists():
    raise FileNotFoundError(
        f"MedGemma model not found at '{MODEL_PATH}'. "
        "Please download or fine-tune the model and place it at that path."
    )

logger.info("Loading MedGemma pipeline from %s", MODEL_PATH)
pipe = pipeline("image-text-to-text", model=MODEL_PATH)
logger.info("MedGemma pipeline loaded successfully.")


def _run_inference(messages: list[dict]) -> str:
    """
    Execute blocking MedGemma pipeline inference synchronously.

    Args:
        messages: Chat-formatted message list with system and user turns.

    Returns:
        Raw text content of the model's last response turn.
    """
    output = pipe(text=messages, max_new_tokens=512)
    raw_text: str = output[0]["generated_text"][-1]["content"]
    return raw_text


def _parse_json(raw_text: str) -> dict:
    """
    Parse a JSON dict from the model's raw output string.

    Tries strict json.loads first; if that fails, falls back to extracting
    the substring between the first '{' and last '}' before retrying.
    Returns FALLBACK_RESULT if both attempts fail.

    Args:
        raw_text: The raw string output from the MedGemma pipeline.

    Returns:
        Parsed diagnosis dict, or FALLBACK_RESULT on failure.
    """
    # Attempt 1: direct parse (model followed instructions and returned clean JSON)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract the outermost JSON object from the string
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("JSON parsing failed for model output. Returning fallback result.")
    return FALLBACK_RESULT


async def run_diagnosis(image_bytes: bytes | None, query: str) -> dict:
    """
    Produce a structured ophthalmological diagnosis using MedGemma.

    Args:
        image_bytes: Raw bytes of the retinal image, or None for text-only queries.
        query: The clinician's diagnostic question.

    Returns:
        A dict with keys: condition, severity, severity_level, confidence,
        findings, recommendation, disclaimer.

    Raises:
        RuntimeError: If inference fails unexpectedly.
    """
    pil_image: Image.Image | None = None
    if image_bytes is not None:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to decode image bytes into PIL Image: {e}") from e

    # Build the user message, attaching the image when available
    user_content: list[dict] | str
    if pil_image is not None:
        user_content = [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": query},
        ]
    else:
        user_content = query

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info("Running MedGemma diagnosis. query=%s has_image=%s", query[:80], pil_image is not None)

    try:
        # Offload blocking pipeline call to a thread to keep the event loop free
        raw_text = await asyncio.to_thread(_run_inference, messages)
    except Exception as e:
        raise RuntimeError(f"MedGemma inference failed: {e}") from e

    logger.debug("Raw MedGemma output: %s", raw_text[:200])

    result = _parse_json(raw_text)
    logger.info("Diagnosis complete. condition=%s severity=%s", result.get("condition"), result.get("severity"))

    return result

from __future__ import annotations

"""
merger.py

Merges segmentation results from PaliGemma 2 and diagnosis results from MedGemma
into a single unified response. Uses Gemma 3 via Ollama to generate a concise
natural-language summary suitable for clinical review.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"
DISCLAIMER = "For research use only. Not intended for clinical diagnosis."


def _build_context(location: dict | None, diagnosis: dict | None) -> str:
    """
    Construct a plain-text context string from available analysis results.

    Args:
        location:  Segmentation result dict from PaliGemma 2, or None.
        diagnosis: Diagnosis result dict from MedGemma, or None.

    Returns:
        A human-readable string summarising whichever results are present.
    """
    parts: list[str] = []

    if location is not None:
        summary = location.get("summary", "")
        detections = location.get("detections", [])
        parts.append(f"Location analysis: {summary} ({len(detections)} region(s) detected)")

    if diagnosis is not None:
        condition = diagnosis.get("condition", "Unknown")
        severity = diagnosis.get("severity", "Unknown")
        findings = diagnosis.get("findings", [])
        findings_text = "; ".join(findings) if findings else "none recorded"
        parts.append(
            f"Diagnosis: {condition}, severity {severity}. Findings: {findings_text}."
        )

    return " ".join(parts) if parts else "No analysis results available."


def _determine_result_type(location: dict | None, diagnosis: dict | None) -> str:
    """
    Determine which result types are present in this response.

    Args:
        location:  Segmentation result dict, or None.
        diagnosis: Diagnosis result dict, or None.

    Returns:
        One of: "full", "location", "diagnosis".
    """
    if location is not None and diagnosis is not None:
        return "full"
    if location is not None:
        return "location"
    return "diagnosis"


async def merge_results(
    location: dict | None,
    diagnosis: dict | None,
    question: str,
) -> dict:
    """
    Merge PaliGemma 2 and MedGemma outputs into a unified clinical response.

    Args:
        location:  Segmentation result dict from run_segmentation(), or None.
        diagnosis: Diagnosis result dict from run_diagnosis(), or None.
        question:  The original clinical question asked by the user.

    Returns:
        A dict with keys:
            "type"       (str):        "full", "location", or "diagnosis".
            "location"   (dict|None):  Raw segmentation result.
            "diagnosis"  (dict|None):  Raw diagnosis result.
            "summary"    (str):        Gemma 3 generated narrative summary.
            "disclaimer" (str):        Standard research-use disclaimer.
    """
    context_string = _build_context(location, diagnosis)
    result_type = _determine_result_type(location, diagnosis)

    prompt = (
        f"You are a medical AI assistant. Summarize these ophthalmology analysis results "
        f"in 2-3 clear sentences for a doctor. "
        f"Question asked: {question}. "
        f"Results: {context_string}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    summary: str
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            summary = data["response"]
            logger.info("Merger summary generated. length=%d", len(summary))
    except Exception as e:
        # Fall back to the raw context string so the response remains useful
        logger.warning("Ollama summarisation failed, using raw context. reason=%s", str(e))
        summary = context_string

    return {
        "type": result_type,
        "location": location,
        "diagnosis": diagnosis,
        "summary": summary,
        "disclaimer": DISCLAIMER,
    }

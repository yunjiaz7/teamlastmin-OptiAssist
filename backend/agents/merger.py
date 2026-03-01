from __future__ import annotations

"""
merger.py

Merges segmentation results from PaliGemma 2, diagnosis results from MedGemma,
and cup-to-disc ratio metrics into a single unified response.

MedGemma (the same model used for diagnosis) is used to generate the final
clinical narrative so the summary benefits from its ophthalmology fine-tuning.
Falls back to the raw context string if MedGemma is unavailable.
"""

import logging

logger = logging.getLogger(__name__)

DISCLAIMER = "For research use only. Not intended for clinical diagnosis."


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _build_context(
    location: dict | None,
    diagnosis: dict | None,
    cdr_metrics: dict | None,
) -> str:
    """
    Construct a plain-text context string from all available analysis results.

    Args:
        location:    Segmentation result dict from PaliGemma 2, or None.
        diagnosis:   Diagnosis result dict from MedGemma, or None.
        cdr_metrics: Dict of CDR metric results keyed by tool name, or None.

    Returns:
        A human-readable string summarising whichever results are present.
    """
    parts: list[str] = []

    if diagnosis is not None:
        condition = diagnosis.get("condition", "Unknown")
        severity = diagnosis.get("severity", "Unknown")
        confidence = diagnosis.get("confidence")
        findings = diagnosis.get("findings", [])
        recommendation = diagnosis.get("recommendation", "")
        findings_text = "; ".join(findings) if findings else "none recorded"
        conf_text = f" (confidence: {confidence:.0%})" if isinstance(confidence, float) else ""
        parts.append(
            f"Diagnosis: {condition}, severity {severity}{conf_text}. "
            f"Findings: {findings_text}. Recommendation: {recommendation}"
        )

    if location is not None:
        summary = location.get("summary", "")
        detections = location.get("detections", [])
        parts.append(
            f"Segmentation: {summary} ({len(detections)} region(s) detected)"
        )

    if cdr_metrics:
        cdr_lines: list[str] = []
        for tool_name, result in cdr_metrics.items():
            if result.get("error"):
                cdr_lines.append(f"  {tool_name}: {result['error']}")
                continue
            metric = result.get("metric", tool_name)
            value = result.get("value") or result.get("value_px")
            interp = result.get("interpretation", "")
            if value is not None:
                unit = "px" if "diameter" in metric else ""
                cdr_lines.append(
                    f"  {metric}: {value}{unit}"
                    + (f" — {interp}" if interp else "")
                )
        if cdr_lines:
            parts.append("Cup-to-disc metrics:\n" + "\n".join(cdr_lines))

    return "\n\n".join(parts) if parts else "No analysis results available."


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def merge_results(
    location: dict | None,
    diagnosis: dict | None,
    question: str,
    cdr_metrics: dict | None = None,
) -> dict:
    """
    Merge PaliGemma 2, MedGemma, and CDR metric outputs into a unified clinical
    response.  MedGemma is used to generate the final narrative summary.

    Args:
        location:    Segmentation result dict from run_segmentation(), or None.
        diagnosis:   Diagnosis result dict from run_diagnosis(), or None.
        question:    The original clinical question asked by the user.
        cdr_metrics: Dict of CDR metric results from the agentic loop, or None.

    Returns:
        A dict with keys:
            "type"        (str):         "full", "location", or "diagnosis".
            "location"    (dict|None):   Raw segmentation result.
            "diagnosis"   (dict|None):   Raw diagnosis result.
            "cdr_metrics" (dict):        All computed CDR metric results.
            "summary"     (str):         MedGemma-generated narrative summary.
            "disclaimer"  (str):         Standard research-use disclaimer.
    """
    cdr_metrics = cdr_metrics or {}
    context_string = _build_context(location, diagnosis, cdr_metrics)
    result_type = _determine_result_type(location, diagnosis)

    # ----------------------------------------------------------------
    # Use MedGemma (general-purpose medical AI) to generate the final
    # clinical summary from all pipeline outputs.
    # Falls back to the raw context string on any failure.
    # ----------------------------------------------------------------
    summary: str = context_string  # safe default
    try:
        from agents.diagnostician import summarize_with_medgemma
        summary = await summarize_with_medgemma(context=context_string, question=question)
        logger.info("MedGemma summary generated. length=%d", len(summary))
    except Exception as e:
        logger.warning(
            "MedGemma summarization failed; using raw context. reason=%s", str(e)
        )
        summary = context_string

    return {
        "type": result_type,
        "location": location,
        "diagnosis": diagnosis,
        "cdr_metrics": cdr_metrics,
        "summary": summary,
        "disclaimer": DISCLAIMER,
    }

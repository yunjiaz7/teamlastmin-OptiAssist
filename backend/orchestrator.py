from __future__ import annotations

"""
orchestrator.py

Main pipeline orchestration for OptiAssist, an ophthalmology AI assistant.

Coordinates input parsing, image pre-scanning, the FunctionGemma agentic loop,
and result merging into a single async pipeline exposed via run_pipeline().

Architecture
------------
The routing and sequencing logic now lives entirely inside the FunctionGemma
agentic loop (router.run_agentic_loop). The orchestrator is responsible for:

  1. Input validation and pre-scanning.
  2. Defining concrete tool callbacks (which call MedGemma / PaliGemma and
     emit SSE progress events around each model call).
  3. Handing the callbacks into the agentic loop.
  4. Merging the results returned by the loop.

SSE events emitted (unchanged from previous version):
  input_received, prescanning, prescan_complete,
  routing, route_decided,               ← route_decided is emitted from router
  medgemma_start, medgemma_complete,    ← emitted by _run_diagnosis callback
  paligemma_start, paligemma_complete,  ← emitted by _run_segmentation callback
  merging, complete
"""

import asyncio
import json
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


async def run_pipeline(
    image_bytes: bytes | None,
    question: str,
    emit: Callable[[str, str], Awaitable[None]],
) -> dict:
    """
    Execute the full OptiAssist analysis pipeline.

    Args:
        image_bytes: Raw image bytes from the client, or None for text-only questions.
        question:    The clinician's question string describing what to analyze.
        emit:        Async callback used to push SSE progress updates.
                     Called as: await emit(event: str, message: str)

    Returns:
        A dict with shape:
            {
                "route":  str,   # route name chosen by FunctionGemma
                "result": dict   # merged output from executed agents
            }

    Raises:
        ValueError:   If both image_bytes and question are absent/empty.
        RuntimeError: If any pipeline stage fails unexpectedly.
    """

    # -------------------------------------------------------------------------
    # Stage 1 — Input validation
    # -------------------------------------------------------------------------
    if not image_bytes and not question:
        raise ValueError("At least one of image_bytes or question must be provided.")

    logger.info(
        "Pipeline started. has_image=%s, question_len=%d",
        image_bytes is not None,
        len(question),
    )
    await emit("input_received", "Image and question received")

    # -------------------------------------------------------------------------
    # Stage 2 — Image pre-scan (skip for text-only requests)
    # -------------------------------------------------------------------------
    image_description: str = ""

    if image_bytes is not None:
        await emit("prescanning", "Scanning image content...")
        try:
            from agents.prescanner import prescan_image
            image_description = await prescan_image(image_bytes)
        except Exception as exc:
            raise RuntimeError(f"Image pre-scan failed: {exc}") from exc

        await emit("prescan_complete", f"Image identified: {image_description}")
        logger.info("Pre-scan complete: %s", image_description)

    # -------------------------------------------------------------------------
    # Stage 3 — FunctionGemma agentic loop
    #
    # Tool callbacks are defined here so they can close over image_bytes,
    # question, and emit. The actual model calls stay in orchestrator;
    # only the routing/sequencing logic lives in router.run_agentic_loop.
    # -------------------------------------------------------------------------
    await emit("routing", "FunctionGemma orchestrating analysis...")

    async def _run_diagnosis() -> dict:
        """Execute MedGemma inference with SSE progress events."""
        await emit("medgemma_start", "Analyzing for pathological conditions...")
        try:
            from agents.diagnostician import run_diagnosis
            result = await run_diagnosis(image_bytes, question, image_description)
        except Exception as exc:
            raise RuntimeError(f"MedGemma diagnosis failed: {exc}") from exc
        await emit(
            "medgemma_complete",
            json.dumps({
                "text": "Diagnosis analysis complete",
                "diagnosis": {
                    "condition": result.get("condition"),
                    "severity": result.get("severity"),
                    "findings": result.get("findings"),
                    "recommendation": result.get("recommendation"),
                },
            }),
        )
        logger.info(
            "Diagnosis complete. condition=%s severity=%s",
            result.get("condition"),
            result.get("severity"),
        )
        return result

    async def _run_segmentation(query: str) -> dict:
        """Execute PaliGemma 2 inference with SSE progress events."""
        await emit("paligemma_start", f"Locating {query}...")
        try:
            from agents.segmenter import run_segmentation
            result = await run_segmentation(image_bytes, query)
        except (FileNotFoundError, ImportError):
            raise  # propagate as-is so router.py can detect permanent model-loading errors
        except Exception as exc:
            raise RuntimeError(f"PaliGemma segmentation failed: {exc}") from exc
        n = len(result.get("detections", []))
        raw_output = result.get("raw_output", "")
        annotated_image_base64 = result.get("annotated_image_base64", "")
        await emit(
            "paligemma_complete",
            json.dumps({
                "text": f"Found {n} regions of interest",
                "segmentation": {
                    "raw_output": raw_output,
                    "detection_count": n,
                    "annotated_image_base64": annotated_image_base64,
                },
            }),
        )
        logger.info("Segmentation complete. detections=%d query=%s", n, query)
        return result

    try:
        from agents.router import run_agentic_loop
        loop_result = await run_agentic_loop(
            question=question,
            image_description=image_description,
            run_diagnosis_cb=_run_diagnosis,
            run_segmentation_cb=_run_segmentation,
            emit=emit,
        )
    except Exception as exc:
        raise RuntimeError(f"Agentic loop failed: {exc}") from exc

    diagnosis: dict | None = loop_result["diagnosis"]
    location: dict | None = loop_result["location"]

    # Derive a human-readable route label for the result payload
    if diagnosis is not None and location is not None:
        route_label = "Full Analysis"
    elif location is not None:
        route_label = "Segmentation"
    else:
        route_label = "Diagnosis"

    logger.info("Agentic loop finished. route_label=%s", route_label)

    # -------------------------------------------------------------------------
    # Stage 4 — Merge results
    # PaliGemma detections + MedGemma diagnosis are passed directly to MedGemma
    # for the final clinical narrative summary.
    # -------------------------------------------------------------------------
    await emit("merging", "MedGemma summarising all results...")
    try:
        from agents.merger import merge_results
        final = await merge_results(location, diagnosis, question)
    except Exception as exc:
        raise RuntimeError(f"Result merging failed: {exc}") from exc

    await emit("complete", "Analysis complete")
    logger.info("Pipeline finished successfully.")

    return {
        "route": route_label,
        "result": final,
    }

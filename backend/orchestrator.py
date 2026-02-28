from __future__ import annotations

"""
orchestrator.py

Main pipeline orchestration for OptiAssist, an ophthalmology AI assistant.
Coordinates input parsing, image pre-scanning, request routing, model execution,
and result merging into a single async pipeline exposed via run_pipeline().
"""

import asyncio
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
        question: The clinician's question string describing what to analyze.
        emit: Async callback used to push SSE progress updates.
              Called as: await emit(event: str, message: str)

    Returns:
        A dict with shape:
            {
                "route": str,   # the function name chosen by the router
                "result": dict  # merged output from executed agents
            }

    Raises:
        ValueError: If both image_bytes and question are absent/empty.
        RuntimeError: If any pipeline stage fails unexpectedly.
    """

    # -------------------------------------------------------------------------
    # Stage 1 — Input parsing
    # -------------------------------------------------------------------------
    if not image_bytes and not question:
        raise ValueError("At least one of image_bytes or question must be provided.")

    logger.info("Pipeline started. has_image=%s, question_len=%d", image_bytes is not None, len(question))
    await emit("input_received", "Image and question received")

    # -------------------------------------------------------------------------
    # Stage 2 — Image pre-scan (skip for text-only requests)
    # -------------------------------------------------------------------------
    image_description: str = ""

    if image_bytes is not None:
        await emit("prescanning", "Scanning image content...")
        try:
            from agents.prescanner import prescan_image
            result = await prescan_image(image_bytes)
        except Exception as e:
            raise RuntimeError(f"Image pre-scan failed: {e}") from e

        await emit("prescan_complete", f"Image identified: {result}")
        image_description = result
        logger.info("Pre-scan complete: %s", image_description)

    # -------------------------------------------------------------------------
    # Stage 3 — FunctionGemma routing
    # -------------------------------------------------------------------------
    await emit("routing", "Deciding analysis type...")
    try:
        from agents.router import route_request
        route = await route_request(question, image_description)
    except Exception as e:
        raise RuntimeError(f"Request routing failed: {e}") from e

    await emit("route_decided", f"Route: {route['function']}")
    logger.info("Route decided: %s", route["function"])

    # -------------------------------------------------------------------------
    # Stage 4 — Execute the routed function
    # -------------------------------------------------------------------------
    location: dict | None = None
    diagnosis: dict | None = None

    if route["function"] == "analyze_location":
        await emit("paligemma_start", "Locating anatomical structures...")
        try:
            from agents.segmenter import run_segmentation
            location = await run_segmentation(image_bytes, route["query"])
        except Exception as e:
            raise RuntimeError(f"Segmentation failed: {e}") from e

        await emit("paligemma_complete", f"Found {len(location['detections'])} regions of interest")
        logger.info("Segmentation complete. detections=%d", len(location["detections"]))

    elif route["function"] == "analyze_diagnosis":
        await emit("medgemma_start", "Analyzing for pathological conditions...")
        try:
            from agents.diagnostician import run_diagnosis
            diagnosis = await run_diagnosis(image_bytes, route["query"])
        except Exception as e:
            raise RuntimeError(f"Diagnosis failed: {e}") from e

        await emit("medgemma_complete", "Diagnosis analysis complete")
        logger.info("Diagnosis complete.")

    elif route["function"] == "analyze_full":
        await emit("paligemma_start", "Locating anatomical structures...")
        await emit("medgemma_start", "Analyzing for pathological conditions...")
        try:
            from agents.segmenter import run_segmentation
            from agents.diagnostician import run_diagnosis
            # Run segmentation and diagnosis concurrently to reduce total latency
            location, diagnosis = await asyncio.gather(
                run_segmentation(image_bytes, route["query"]),
                run_diagnosis(image_bytes, route["query"]),
            )
        except Exception as e:
            raise RuntimeError(f"Full analysis (segmentation + diagnosis) failed: {e}") from e

        await emit("paligemma_complete", "Segmentation complete")
        await emit("medgemma_complete", "Diagnosis complete")
        logger.info("Full analysis complete.")

    else:
        # Guard against unexpected route values returned by the router
        raise ValueError(f"Unknown route function: '{route['function']}'")

    # -------------------------------------------------------------------------
    # Stage 5 — Merge results
    # -------------------------------------------------------------------------
    await emit("merging", "Combining results...")
    try:
        from agents.merger import merge_results
        final = await merge_results(location, diagnosis, question)
    except Exception as e:
        raise RuntimeError(f"Result merging failed: {e}") from e

    await emit("complete", "Analysis complete")
    logger.info("Pipeline finished successfully.")

    return {
        "route": route["function"],
        "result": final,
    }

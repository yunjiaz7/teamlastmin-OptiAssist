from __future__ import annotations

"""
main.py

FastAPI application entry point for OptiAssist. Exposes a health check endpoint
and a streaming /analyze endpoint that runs the full ophthalmology AI pipeline
and pushes progress updates to the client via Server-Sent Events (SSE).
"""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from orchestrator import run_pipeline

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OptiAssist API")

# Allow all origins so the local Next.js dev server can reach this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sentinel object placed on the queue to signal that the pipeline has finished
_DONE = object()


@app.get("/health")
async def health() -> dict:
    """
    Return API health status and the list of models used in the pipeline.

    Returns:
        A dict with keys "status" and "models".
    """
    return {
        "status": "ok",
        "models": ["functiongemma", "gemma3:4b", "paligemma2", "medgemma"],
    }


@app.post("/analyze")
async def analyze(
    question: str = Form(...),
    image: Optional[UploadFile] = File(default=None),
) -> StreamingResponse:
    """
    Run the OptiAssist pipeline and stream progress events to the client via SSE.

    Args:
        question: The clinician's question (required form field).
        image:    Optional retinal image upload.

    Returns:
        A StreamingResponse with content-type "text/event-stream".
        Each chunk is a JSON-encoded SSE event terminated by a double newline.
    """
    image_bytes: bytes | None = None
    if image is not None:
        image_bytes = await image.read()

    # Queue is the bridge between the pipeline coroutine and the SSE generator.
    # The pipeline puts (event, message) tuples; _DONE signals completion.
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(event: str, message: str) -> None:
        """Push a progress event onto the SSE queue."""
        await queue.put((event, message))

    async def run_and_signal() -> None:
        """Run the pipeline, then put the final result or an error onto the queue."""
        try:
            result = await run_pipeline(image_bytes, question, emit)
            await queue.put(("__result__", result))
        except Exception as exc:
            logger.error("Pipeline error: %s", str(exc))
            await queue.put(("__error__", str(exc)))
        finally:
            await queue.put(_DONE)

    async def event_stream() -> AsyncGenerator[str, None]:
        """
        Async generator that reads from the queue and yields SSE-formatted strings.

        Yields:
            SSE event strings in the format:  data: {...}\\n\\n
        """
        # Start the pipeline as a concurrent task so this generator can stream
        # its events without blocking
        asyncio.create_task(run_and_signal())

        while True:
            item = await queue.get()

            if item is _DONE:
                break

            event, payload = item

            if event == "__result__":
                # Final payload carries the full result dict
                chunk = json.dumps({"event": "complete", "result": payload})
            elif event == "__error__":
                chunk = json.dumps({"event": "error", "message": payload})
            else:
                chunk = json.dumps({"event": event, "message": payload})

            yield f"data: {chunk}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# OptiAssist — System Technical Deep Dive

> **What it is**: A fully local AI-assisted analysis system built for ophthalmology clinical settings. Zero cloud dependency. Patient data never leaves the clinic.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Backend Tech Stack](#3-backend-tech-stack)
4. [The Five-Stage AI Pipeline](#4-the-five-stage-ai-pipeline)
   - [Stage 1 — Input Reception & Validation](#stage-1--input-reception--validation)
   - [Stage 2 — Image Pre-Scan (Prescanner)](#stage-2--image-pre-scan-prescanner)
   - [Stage 3 — Intelligent Routing (Router)](#stage-3--intelligent-routing-router)
   - [Stage 4 — Parallel Expert Inference](#stage-4--parallel-expert-inference)
   - [Stage 5 — Result Merging (Merger)](#stage-5--result-merging-merger)
5. [Models In Depth](#5-models-in-depth)
6. [SSE Real-Time Streaming Mechanism](#6-sse-real-time-streaming-mechanism)
7. [Async Concurrency Design](#7-async-concurrency-design)
8. [Frontend Tech Stack](#8-frontend-tech-stack)
9. [Frontend Page Architecture](#9-frontend-page-architecture)
10. [Frontend → Backend Communication Protocol](#10-frontend--backend-communication-protocol)
11. [API Reference](#11-api-reference)
12. [Data Structure Definitions](#12-data-structure-definitions)
13. [Key Design Decisions](#13-key-design-decisions)
14. [Dependency Manifest](#14-dependency-manifest)

---

## 1. System Overview

OptiAssist is a **fully on-device AI diagnostic assistant purpose-built for ophthalmologists**, addressing three core pain points:

| Problem | OptiAssist Solution |
|---------|-------------------|
| Patient privacy (HIPAA / GDPR) | All inference runs locally — images never leave the machine |
| Cloud latency (500 ms+) | On-device inference, sub-second response |
| Lack of specialized tooling | Multi-model pipeline trained specifically on retinal imagery |

The system consists of two independent services:

- **Backend**: A Python FastAPI service responsible for receiving requests, orchestrating the AI pipeline, and streaming progress updates to the client via Server-Sent Events (SSE).
- **Frontend**: A Next.js application providing a marketing landing page and an interactive live demo interface.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js 16)                     │
│                                                              │
│  ┌──────────────┐     ┌─────────────────────────────────┐   │
│  │  Landing Page│     │         Demo Page (/demo)        │   │
│  │  /           │     │                                  │   │
│  │  - Hero      │     │  ┌─────────────┐ ┌───────────┐  │   │
│  │  - Problem   │     │  │ Input Panel │ │ Analysis  │  │   │
│  │  - HowItWrks │     │  │ (image +    │ │ Panel     │  │   │
│  │  - TechStack │     │  │  question)  │ │ (SSE feed)│  │   │
│  └──────────────┘     │  └──────┬──────┘ └───────────┘  │   │
│                       └─────────┼───────────────────────┘   │
└─────────────────────────────────┼───────────────────────────┘
                                  │  POST /analyze (multipart/form-data)
                                  │  ← SSE Stream (text/event-stream)
┌─────────────────────────────────▼───────────────────────────┐
│                  BACKEND (FastAPI + Uvicorn)                  │
│                                                              │
│  main.py ──► orchestrator.py                                 │
│                    │                                         │
│         ┌──────────▼──────────────────────────┐             │
│         │           5-Stage Pipeline           │             │
│         │                                      │             │
│         │  Stage 1: Input Parsing              │             │
│         │      ↓                               │             │
│         │  Stage 2: prescanner.py              │             │
│         │    └─► Gemma 3 4B (via Ollama)       │             │
│         │      ↓                               │             │
│         │  Stage 3: router.py                  │             │
│         │    └─► FunctionGemma 270M (Ollama)   │             │
│         │      ↓                               │             │
│         │  Stage 4: Branch execution           │             │
│         │    ├─► segmenter.py                  │             │
│         │    │     └─► PaliGemma 2 3B (HF)     │             │
│         │    ├─► diagnostician.py              │             │
│         │    │     └─► MedGemma 4B (HF)        │             │
│         │    └─► Both in parallel (analyze_full)│            │
│         │      ↓                               │             │
│         │  Stage 5: merger.py                  │             │
│         │    └─► Gemma 3 4B (via Ollama)       │             │
│         └──────────────────────────────────────┘             │
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │  Local Model Infrastructure                │             │
│  │  - Ollama (localhost:11434): Gemma3, FuncG │             │
│  │  - HuggingFace Transformers: PaliG2, MedG  │             │
│  └────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Backend Tech Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Web framework | FastAPI | 0.128.8 | API routing, SSE streaming, CORS |
| ASGI server | Uvicorn | 0.39.0 | Async HTTP server |
| Async HTTP client | httpx | 0.28.1 | Calling the Ollama REST API |
| Deep learning | PyTorch | 2.8.0 | Transformers inference backend |
| Model library | HuggingFace Transformers | 4.57.6 | Loading PaliGemma 2 / MedGemma |
| Image processing | Pillow | 11.3.0 | Image decoding, annotation drawing, encoding |
| Local LLM runtime | Ollama | localhost:11434 | Running Gemma 3 / FunctionGemma |
| Env management | python-dotenv | 1.2.1 | Configuration isolation |
| Numerical computing | NumPy | 2.0.2 | Tensor post-processing |

---

## 4. The Five-Stage AI Pipeline

The pipeline entry point is the `run_pipeline()` function in `orchestrator.py`, accepting three parameters:

```python
async def run_pipeline(
    image_bytes: bytes | None,   # Raw image bytes; None for text-only mode
    question: str,               # The clinician's clinical question
    emit: Callable[[str, str], Awaitable[None]],  # SSE push callback
) -> dict
```

---

### Stage 1 — Input Reception & Validation

**File**: `orchestrator.py` (lines 47–51)

A minimal guard: if both image and question are absent, a `ValueError` is raised immediately before entering the pipeline.

```python
if not image_bytes and not question:
    raise ValueError("At least one of image_bytes or question must be provided.")

await emit("input_received", "Image and question received")
```

SSE event name: `input_received`

---

### Stage 2 — Image Pre-Scan (Prescanner)

**File**: `agents/prescanner.py`

**Trigger condition**: Only runs when `image_bytes is not None`. Text-only queries skip this stage entirely.

**Core logic**:

1. Base64-encode the raw image bytes
2. Call the **Ollama local API** (`http://localhost:11434/api/generate`)
3. Use **Gemma 3 4B** (`gemma3:4b`) with `stream: false` for a synchronous response
4. Return a 1–2 sentence natural-language description (e.g., *"A retinal fundus image showing the optic disc and vascular structure; no obvious macular abnormality detected."*)

```python
payload = {
    "model": "gemma3:4b",
    "prompt": "Describe this medical retinal image in 1-2 sentences...",
    "images": [base64_image],   # Ollama multimodal input format
    "stream": False,
}
```

**Why this step exists**: The generated `image_description` is forwarded to the Router in the next stage, giving the routing decision visual context and improving routing accuracy.

**Fault tolerance**: If the Ollama call fails, the function returns the safe default `"Retinal fundus image"` and the pipeline continues uninterrupted.

SSE events: `prescanning` → `prescan_complete`

---

### Stage 3 — Intelligent Routing (Router)

**File**: `agents/router.py`

**Core logic**: Uses **FunctionGemma 270M** (a function-calling-specialized fine-tune of Gemma) to semantically classify the request and select an execution path.

**Tool Calling Schema**:

The router registers three tools with the model; the model selects one via the function calling mechanism:

| Tool Name | Trigger Scenario | Trigger Keywords |
|-----------|-----------------|-----------------|
| `analyze_location` | Locating an anatomical structure or lesion | where, locate, show me, detect, segment, find |
| `analyze_diagnosis` | Medical judgment, disease classification, risk assessment | is this, diagnosis, what disease, severity, risk |
| `analyze_full` | Combined location + diagnosis analysis | full analysis, everything, complete report |

**Invocation**:

```python
payload = {
    "model": "functiongemma",
    "messages": [{"role": "user", "content": f"{question}\n\nImage context: {image_description}"}],
    "tools": TOOLS,   # JSON Schema definitions for all three tools
    "stream": False,
}
```

**Output format**:

```python
{
    "function": "analyze_diagnosis",   # Selected route
    "query": "..."                     # Refined query extracted from the tool call arguments
}
```

**Fault tolerance**: If the model returns no tool call (e.g., due to a model error), the router defaults to `analyze_full`, ensuring the user always receives a useful response.

SSE events: `routing` → `route_decided`

---

### Stage 4 — Parallel Expert Inference

Based on the Router's decision, Stage 4 executes along one of three paths:

#### Path A: `analyze_location` → Segmenter

**File**: `agents/segmenter.py`
**Model**: `PaliGemma 2 3B` (Google, loaded locally)

**Execution flow**:

```
image_bytes
    ↓
PIL.Image.open()   # Decode to RGB PIL Image
    ↓
prompt = f"segment {query}\n"
    ↓
processor(text=prompt, images=pil_image, return_tensors="pt")
    ↓
model.generate(**inputs, max_new_tokens=256)
    ↓
raw_output = processor.decode(outputs[0], skip_special_tokens=False)
    ↓
_parse_detections(raw_output, img_width, img_height)
    ↓
_draw_boxes(pil_image, detections)
    ↓
return {detections, annotated_image_base64, summary}
```

**`<loc>` Token Parsing Mechanism**:

PaliGemma 2 encodes bounding boxes using special tokens `<loc####>`. Each bounding box is represented as four consecutive tokens in the order `y_min, x_min, y_max, x_max`. Each value is in the range `[0, 1023]` (normalized coordinate × 1024) and must be scaled back to actual image pixel dimensions.

```python
_LOC_PATTERN = re.compile(r"<loc(\d{4})>")

# Scale back to real pixel coordinates
y_min = int((y_min_raw / 1024) * img_height)
x_min = int((x_min_raw / 1024) * img_width)
```

**Annotated image generation**: Pillow draws red bounding boxes (`outline="red", width=2`) on a copy of the original image, then encodes the result as a Base64 PNG data URI for the frontend.

**Async handling**: PaliGemma inference is a blocking synchronous operation. It is offloaded to a thread pool via `asyncio.to_thread()` to prevent blocking the FastAPI event loop.

```python
raw_output = await asyncio.to_thread(_run_inference, pil_image, prompt)
```

SSE events: `paligemma_start` → `paligemma_complete`

---

#### Path B: `analyze_diagnosis` → Diagnostician

**File**: `agents/diagnostician.py`
**Model**: `MedGemma 4B` (Google DeepMind medical-specialized model, loaded locally)

**System Prompt Design** (structured output constraint):

```
You are an expert ophthalmology AI assistant.
Analyze the retinal image and answer the clinical question.
Always respond with valid JSON only, no extra text.
JSON fields required:
  condition: string (disease name or 'Normal')
  severity: string (None/Mild/Moderate/Severe/Proliferative)
  severity_level: integer 0-4
  confidence: float 0.0-1.0
  findings: list of strings (specific observations)
  recommendation: string (follow-up advice)
  disclaimer: always set to 'For research use only...'
```

**Execution flow**:

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": query},
    ]},
]

pipe = pipeline("image-text-to-text", model=MODEL_PATH)
output = pipe(text=messages, max_new_tokens=512)
```

**JSON parsing with fallback**:

```python
# Attempt 1: direct json.loads
# Attempt 2: extract substring between first '{' and last '}', then re-parse
# Final fallback: FALLBACK_RESULT (a pre-defined safe default dict)
```

Blocking pipeline calls are similarly offloaded via `asyncio.to_thread()`.

SSE events: `medgemma_start` → `medgemma_complete`

---

#### Path C: `analyze_full` → Segmenter + Diagnostician in Parallel

This is the most critical optimization: both heavyweight models run **simultaneously**, not sequentially.

```python
location, diagnosis = await asyncio.gather(
    run_segmentation(image_bytes, route["query"]),
    run_diagnosis(image_bytes, route["query"]),
)
```

`asyncio.gather()` schedules both coroutines concurrently. Since each internally uses `asyncio.to_thread()`, both models actually run in separate threads simultaneously, maximizing GPU/CPU utilization. The total wall-clock time is `max(T_segmentation, T_diagnosis)` rather than the sum.

SSE events: `paligemma_start` + `medgemma_start` emitted simultaneously → each fires its corresponding `_complete` when done

---

### Stage 5 — Result Merging (Merger)

**File**: `agents/merger.py`
**Model**: `Gemma 3 4B` (via Ollama)

Receives `location` (from PaliGemma 2) and/or `diagnosis` (from MedGemma), then generates a natural-language summary suitable for clinical review.

**Context construction**:

```python
def _build_context(location, diagnosis) -> str:
    # If location is present: include detected region count and summary
    # If diagnosis is present: include condition name, severity, and findings list
    # Returns a single-paragraph string for Gemma 3 to summarize
```

**Summarization prompt**:

```
You are a medical AI assistant. Summarize these ophthalmology analysis results
in 2-3 clear sentences for a doctor.
Question asked: {question}.
Results: {context_string}
```

**Output structure**:

```python
{
    "type": "full" | "location" | "diagnosis",
    "location": { ... } | None,
    "diagnosis": { ... } | None,
    "summary": "Gemma 3 generated narrative summary",
    "disclaimer": "For research use only. Not intended for clinical diagnosis."
}
```

**Fault tolerance**: If the Ollama call fails, `summary` falls back to the raw `context_string` (the plain concatenated text), keeping the response useful.

SSE events: `merging` → `complete`

---

## 5. Models In Depth

| Model | Parameters | Runtime | Call Interface | Role in Pipeline |
|-------|-----------|---------|---------------|-----------------|
| **Gemma 3 4B** | 4B | Ollama local service | REST API (`/api/generate`) | Image pre-scan description + final narrative summary |
| **FunctionGemma** | 270M | Ollama local service | REST API (`/api/chat` + tools) | Semantic routing via function calling |
| **PaliGemma 2** | 3B | HuggingFace Transformers (local weights) | Python API (`AutoProcessor` + `PaliGemmaForConditionalGeneration`) | Retinal anatomical structure segmentation + bounding box prediction |
| **MedGemma** | 4B | HuggingFace Transformers (local weights) | Python API (`pipeline("image-text-to-text")`) | Ophthalmic disease diagnosis + structured clinical report |

**Model file locations**:
- `backend/models/paligemma2-finetuned/` — Fine-tuned PaliGemma 2 weights
- `backend/models/medgemma-finetuned/` — Fine-tuned MedGemma weights
- Gemma 3 / FunctionGemma — Managed by Ollama, auto-cached in `~/.ollama/`

---

## 6. SSE Real-Time Streaming Mechanism

The central design of `main.py` is an **asyncio.Queue-based SSE streaming architecture** that lets the frontend observe every step of the pipeline in real time.

### How it works

```
Client POST /analyze
         │
         │  FastAPI immediately returns StreamingResponse(text/event-stream)
         │
         ▼
  asyncio.Queue  ←───────────────────────────────────┐
         │                                            │
         │  event_stream() async generator            │
         │  continuously reads from queue             │
         │                                            │
         ▼                                            │
  yield "data: {...}\n\n"  ──→ pushed to client       │
                                                      │
                            run_pipeline() runs in    │
                            the background; each      │
                            stage calls emit() which  │
                            puts events onto the queue┘
```

### The emit callback design

```python
async def emit(event: str, message: str) -> None:
    await queue.put((event, message))
```

Each pipeline stage calls `await emit("event_name", "message")` to push progress updates without needing to know anything about the underlying SSE transport.

### SSE event wire format

Each SSE chunk sent to the client looks like:

```
data: {"event": "prescanning", "message": "Scanning image content..."}\n\n
data: {"event": "routing", "message": "Deciding analysis type..."}\n\n
...
data: {"event": "complete", "result": { ... full result ... }}\n\n
```

### Complete event sequence

```
input_received    → Input accepted
prescanning       → Gemma 3 begins scanning the image
prescan_complete  → Image description generated
routing           → FunctionGemma begins routing
route_decided     → Routing decision confirmed
paligemma_start   → PaliGemma 2 inference started (analyze_location / analyze_full)
medgemma_start    → MedGemma inference started (analyze_diagnosis / analyze_full)
paligemma_complete→ Segmentation done
medgemma_complete → Diagnosis done
merging           → Gemma 3 begins generating summary
complete          → Final event carrying the full result payload
```

### Termination sentinel

When the pipeline finishes, the sentinel object `_DONE` is placed on the queue. The `event_stream()` generator detects it and exits, cleanly closing the connection.

```python
_DONE = object()  # Identity-unique sentinel; comparison uses `is`, not `==`

if item is _DONE:
    break
```

---

## 7. Async Concurrency Design

OptiAssist handles CPU-intensive inference and I/O-bound network calls through two key mechanisms, keeping the code clean while maximizing throughput.

### Mechanism 1: `asyncio.to_thread()` — Making blocking inference async

HuggingFace Transformers inference is a synchronous blocking operation. Calling it directly inside a coroutine would stall the entire FastAPI event loop. The solution:

```python
# Used in both segmenter.py and diagnostician.py
raw_output = await asyncio.to_thread(_run_inference, pil_image, prompt)
```

`asyncio.to_thread()` submits the blocking function to the default `ThreadPoolExecutor`. The coroutine suspends until the thread finishes; the event loop remains free to handle other requests in the meantime.

### Mechanism 2: `asyncio.gather()` — Parallel dual-model inference

In the `analyze_full` path, segmentation and diagnosis are launched simultaneously:

```python
location, diagnosis = await asyncio.gather(
    run_segmentation(image_bytes, route["query"]),
    run_diagnosis(image_bytes, route["query"]),
)
```

Both coroutines internally call `asyncio.to_thread()`, so two threads actually run in parallel, maximizing GPU/CPU utilization. Total wall-clock time equals `max(T_seg, T_diag)` rather than their sum.

### Mechanism 3: Pipeline / SSE generator decoupling

```python
asyncio.create_task(run_and_signal())  # Pipeline runs as a background task
# event_stream() is an independent coroutine draining the queue and pushing to the client
```

The two sides communicate exclusively through `asyncio.Queue` — fully decoupled, neither blocking the other.

---

## 8. Frontend Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Next.js** | 16.1.6 | Full-stack React framework, App Router |
| **React** | 19.2.4 | Component-based UI |
| **TypeScript** | 5.7.3 | Type safety |
| **Tailwind CSS** | 4.2.0 | Utility-first CSS, dark theme |
| **Radix UI** | multiple | Accessible unstyled primitive components |
| **Lucide React** | 0.564.0 | Icon library |
| **Vercel Analytics** | 1.6.1 | Usage analytics |
| **Geist Font** | Google Fonts | Sans-serif + monospace typefaces |

---

## 9. Frontend Page Architecture

### Route structure

```
/        → app/page.tsx          (Landing Page)
/demo    → app/demo/page.tsx     (Interactive Demo)
```

### Landing Page (`/`)

Six components assembled linearly:

```
<Navbar>          Navigation bar (with smooth scroll to Problem section)
<Hero>            Hero section (product positioning + faux terminal + CTA buttons)
<ProblemSection>  Pain point cards (Privacy / Speed / No cloud upload / Smarter tools)
<HowItWorks>      Three-step workflow (Capture → Ask → Analyze)
<TechStack>       Model cards (four models)
<SiteFooter>      Footer
```

**Visual design highlights**:
- Dark theme (`bg-background: #0A0A0A`)
- `glass-card` glassmorphism (`backdrop-blur` + low-opacity background)
- Green (`#4ADE80`) as the brand primary color
- `glow-green` glow effect for a high-tech aesthetic

### Demo Page (`/demo`)

**Layout**: Two-column split (`grid-cols-[1fr_1.2fr]`)

**Left column — Input Panel**:
- Drag-and-drop upload zone (`onDrop` + `onDragOver`) + `FileReader` API preview
- Clinical question input (Enter key submits)
- Analyze / Reset button group

**Right column — Analysis Panel**:
- Pipeline feed: real-time display of each SSE event with ✅ / ⏳ / ❌ status icons
- Segmentation result image: displays the Base64 PNG annotated image
- Diagnosis card: Condition name + Severity badge + Findings list + Recommendation text

---

## 10. Frontend → Backend Communication Protocol

### Request format

```
POST http://localhost:8000/analyze
Content-Type: multipart/form-data

Fields:
  question: string  (required)
  image:    File    (optional — retinal image JPG/PNG)
```

The frontend uses the browser-native `FormData` + `fetch` API:

```typescript
const formData = new FormData()
if (imageFile) formData.append("image", imageFile)
formData.append("question", question)

const response = await fetch(`${BACKEND_URL}/analyze`, {
    method: "POST",
    body: formData,
})
```

### Response: Manual SSE stream parsing

The frontend uses `response.body.getReader()` to manually read the SSE stream rather than `EventSource`, because `EventSource` does not support POST requests:

```typescript
const reader = response.body.getReader()
const decoder = new TextDecoder()
let buffer = ""

while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n")
    buffer = lines.pop() || ""   // Hold incomplete lines for the next iteration

    for (const line of lines) {
        if (!line.startsWith("data: ")) continue
        const data = JSON.parse(line.slice(6))
        // Handle data.event
    }
}
```

### Backend URL configuration

```typescript
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"
```

Configured via the `NEXT_PUBLIC_BACKEND_URL` environment variable; defaults to local port 8000.

---

## 11. API Reference

### `GET /health`

**Description**: Health check. Returns service status and the list of models in use.

**Response**:
```json
{
    "status": "ok",
    "models": ["functiongemma", "gemma3:4b", "paligemma2", "medgemma"]
}
```

---

### `POST /analyze`

**Description**: Executes the full retinal analysis pipeline and returns an SSE stream.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | ✅ | Clinical question |
| `image` | File | ❌ | Retinal image (JPG / PNG) |

**Response**: `Content-Type: text/event-stream`

SSE event types:

| `event` value | Payload | Meaning |
|--------------|---------|---------|
| `input_received` | message: string | Input accepted |
| `prescanning` | message: string | Image pre-scan started |
| `prescan_complete` | message: string (includes description) | Pre-scan finished |
| `routing` | message: string | Routing decision started |
| `route_decided` | message: string (includes route name) | Route confirmed |
| `paligemma_start` | message: string | PaliGemma 2 inference started |
| `medgemma_start` | message: string | MedGemma inference started |
| `paligemma_complete` | message: string (includes detection count) | Segmentation done |
| `medgemma_complete` | message: string | Diagnosis done |
| `merging` | message: string | Result merging started |
| `complete` | result: PipelineResult | Analysis complete — full result payload |
| `error` | message: string | Error description |

---

## 12. Data Structure Definitions

### PipelineResult (final output)

```typescript
interface PipelineResult {
    route: "analyze_location" | "analyze_diagnosis" | "analyze_full"
    result: MergedResult
}
```

### MergedResult (aggregated result)

```typescript
interface MergedResult {
    type: "full" | "location" | "diagnosis"
    location: SegmentationResult | null
    diagnosis: DiagnosisResult | null
    summary: string      // Gemma 3 generated narrative summary
    disclaimer: string   // Standard research-use disclaimer
}
```

### SegmentationResult

```typescript
interface SegmentationResult {
    detections: Detection[]
    annotated_image_base64: string  // "data:image/png;base64,..."
    summary: string
}

interface Detection {
    label: string          // Anatomical structure label
    confidence: number     // Fixed at 0.9 in current version
    bounding_box: {
        x_min: number
        y_min: number
        x_max: number
        y_max: number
    }
    has_mask: boolean      // Currently always false; awaiting vae-oid.npz decoding support
}
```

### DiagnosisResult

```typescript
interface DiagnosisResult {
    condition: string        // Disease name or "Normal"
    severity: "None" | "Mild" | "Moderate" | "Severe" | "Proliferative"
    severity_level: 0 | 1 | 2 | 3 | 4
    confidence: number       // 0.0 – 1.0
    findings: string[]       // List of specific clinical observations
    recommendation: string   // Follow-up advice
    disclaimer: string
}
```

---

## 13. Key Design Decisions

### Why SSE instead of WebSocket?

- SSE is **unidirectional** server-to-client push — exactly what this scenario requires (backend pushes progress; frontend only receives)
- SSE runs over plain HTTP: no handshake, simple CORS policy
- Native browser support with automatic reconnection (when using `EventSource`)
- WebSocket's bidirectional capability is entirely unnecessary here

### Why `asyncio.Queue` instead of `asyncio.Event`?

A Queue supports multiple producers and consumers and carries typed data (event name + message), perfectly matching the need for each pipeline stage to push different payloads to the frontend. An Event can only signal state transitions; it cannot carry data.

### Why PaliGemma 2 and MedGemma are loaded via HuggingFace, while Gemma 3 / FunctionGemma use Ollama?

- PaliGemma 2 and MedGemma are **fine-tuned versions** (the `-finetuned` directories). Loading custom weights directly requires HuggingFace Transformers — the most flexible approach.
- Gemma 3 and FunctionGemma use **base weights** managed by Ollama, which provides friendlier local management (automatic quantization, memory management, HTTP API).
- Ollama's built-in function calling support is the key that makes FunctionGemma routing work.

### When are models loaded?

All models are loaded at **module import time** (module level), not on each request. This eliminates per-request model loading overhead (Transformers cold-loading can take tens of seconds).

```python
# segmenter.py — loaded once at module level
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_PATH)
```

### Error handling strategy

Each agent implements two layers of fault tolerance:

1. **Inner layer**: `try/except` catches inference errors and returns a safe fallback (e.g., prescanner's `FALLBACK_DESCRIPTION`)
2. **Outer layer**: The orchestrator wraps agent exceptions as `RuntimeError`, propagating them upward to `main.py`, which converts them into SSE `error` events

This ensures the system degrades gracefully on any single-point failure rather than crashing.

---

## 14. Dependency Manifest

### Backend (`requirements.txt`)

```
fastapi==0.128.8         # Web framework
uvicorn==0.39.0          # ASGI server
httpx==0.28.1            # Async HTTP client (calls Ollama)
transformers==4.57.6     # PaliGemma 2 / MedGemma loading
torch==2.8.0             # Deep learning framework
pillow==11.3.0           # Image processing
python-dotenv==1.2.1     # Environment variable management
pydantic==2.12.5         # Data validation
python-multipart==0.0.20 # multipart/form-data parsing
safetensors==0.7.0       # Efficient model weight loading format
```

### Frontend (core `package.json` dependencies)

```
next@16.1.6              # Full-stack framework
react@19.2.4             # UI library
typescript@5.7.3         # Type system
tailwindcss@4.2.0        # CSS framework
@radix-ui/*              # Accessible primitive components
lucide-react@0.564.0     # Icon library
```

### External runtime dependencies

- **Ollama** (must be installed locally): runs `gemma3:4b` and `functiongemma`
  - Start: `ollama serve`
  - Pull models: `ollama pull gemma3:4b && ollama pull functiongemma`
- **CUDA / Metal** (optional): GPU acceleration for PaliGemma 2 / MedGemma inference

---

## Appendix: Startup Procedure

```bash
# 1. Start the Ollama service
ollama serve

# 2. Start the backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 3. Start the frontend
cd frontend
npm install
npm run dev        # → http://localhost:3000
```

### End-to-end latency estimate (analyze_full path)

```
Stage 1: Input validation          < 1 ms
Stage 2: Gemma 3 pre-scan          ~2–5 s   (Ollama; hardware-dependent)
Stage 3: FunctionGemma routing     ~0.5–1 s
Stage 4: PaliGemma 2               ~3–8 s   ┐ run in parallel
         MedGemma                  ~3–8 s   ┘ total ≈ max(both)
Stage 5: Gemma 3 summary           ~1–3 s
──────────────────────────────────────────────────────────────
Total (GPU):   ~8–18 s
Total (CPU):   ~30–90 s
```

---

*Documentation generated from the actual codebase. Version date: 2026-02-28*

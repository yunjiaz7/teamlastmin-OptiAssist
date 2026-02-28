# OptiAssist — Technical Documentation & Operations Manual

> On-Device AI Assistant for Ophthalmologists | v1.0 | 2026.02.28

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Final Product Form](#2-final-product-form)
3. [Technical Architecture](#3-technical-architecture)
4. [Directory Structure](#4-directory-structure)
5. [Pre-Hackathon Setup (Day 0)](#5-pre-hackathon-setup-day-0)
6. [Hackathon Day Operations (228)](#6-hackathon-day-operations-228)
7. [Teammate Handover Interface](#7-teammate-handover-interface)
8. [Prompt Reference Summary](#8-prompt-reference-summary)
9. [Risk Register](#9-risk-register)
10. [Demo Script](#10-demo-script)

---

## 1. Project Overview

### One-Line Description

Upload or capture a retinal fundus image, ask a clinical question in natural language, and receive an on-device AI analysis — combining precise anatomical segmentation and medical diagnosis — in under 30 seconds. Patient data never leaves the device.

### Core Value

- Patient retinal images are protected under HIPAA and GDPR — they legally cannot be transmitted to cloud servers
- Ophthalmologists need real-time AI assistance during consultations, but cloud round-trips add 500ms+ latency
- Existing AI tools require uploading sensitive patient data to external APIs — a compliance and privacy risk
- Manual image review is slow and subject to human error; AI catches patterns instantly
- OptiAssist runs entirely on-device using Google DeepMind Gemma family models — zero data transmission

### Technical Highlights

- On-device multi-model routing: FunctionGemma 270M as intelligent orchestrator
- PaliGemma 2 3B (fine-tuned by teammate) for pixel-level retinal structure segmentation
- MedGemma 4B (fine-tuned by teammate) for ophthalmological diagnosis and classification
- Gemma 3 4B for image pre-scanning and result synthesis
- Real-time SSE streaming — judges see every reasoning step as it happens
- Handles text-only questions, image+question, and full analysis requests

---

## 2. Final Product Form

### Landing Page (`/`)

Style reference: ship26.instalily.ai — dark background `#0A0A0A`, green accent `#2C7A4B`, monospace typography, sharp minimal layout.

- Hero section: project name + one-line description + two CTA buttons
- Problem section (4 cards): Patient Data Privacy | Real-time Speed | Cloud Tools Violate Compliance | Doctors Need Smarter Tools
- How It Works: 3-step visual flow
- Tech Stack: list all 4 models with one-line role descriptions
- `Try the Demo` button → navigates to `/demo`

### Demo Page (`/demo`)

- Left panel: image upload / camera capture + text question input + Analyze button
- Right panel top: annotated retinal image with segmentation mask overlay
- Right panel middle: diagnosis result card — condition, severity, findings, recommendation
- Right panel bottom: live SSE processing feed — each step appears in real time
- Route badge: shows which models were called (`Segmentation` / `Diagnosis` / `Full Analysis`)
- Disclaimer: "For research use only. Not intended for clinical diagnosis."

---

## 3. Technical Architecture

### System Overview

```
User Input (image optional + question)
            ↓
    ┌─────────────────────┐
    │   Input Parser      │  Has image? Has question?
    └─────────────────────┘
            ↓
    ┌─────────────────────┐
    │  Gemma 3 Pre-scan   │  (only if image present)
    │  image → text desc  │  "Retinal fundus showing..."
    └─────────────────────┘
            ↓
    ┌─────────────────────┐
    │   FunctionGemma     │  Reads question + image description
    │   Router            │  Decides which function to call
    └─────────────────────┘
            ↓
    ┌───────────────────────────────────────┐
    │            Tool Executor              │
    │                                       │
    │  analyze_location()                   │
    │  → PaliGemma 2 (fine-tuned)          │
    │    outputs <loc><seg> tokens          │
    │    post-process → mask overlay        │
    │                                       │
    │  analyze_diagnosis()                  │
    │  → MedGemma (fine-tuned)             │
    │    outputs natural language JSON      │
    │                                       │
    │  analyze_full()                       │
    │  → both in parallel                   │
    └───────────────────────────────────────┘
            ↓
    ┌─────────────────────┐
    │   Result Merger     │  Gemma 3 synthesizes
    │                     │  unified natural language
    └─────────────────────┘
            ↓
       SSE stream → Frontend
```

### Routing Decision Table

| Input Type | Question Keywords | Route |
|------------|-------------------|-------|
| Image + question | "where", "locate", "show me", "detect", "segment" | `analyze_location()` → PaliGemma 2 |
| Image + question | "is this", "diagnosis", "what disease", "severity", "risk" | `analyze_diagnosis()` → MedGemma |
| Image + question | "what is wrong", "full analysis", "everything" | `analyze_full()` → Both |
| Text only | Any medical question | `analyze_diagnosis()` → MedGemma text mode |

### Technology Stack

| Layer | Tool | Notes |
|-------|------|-------|
| Orchestrator | FunctionGemma 270M | Text-only router, outputs structured function calls |
| Pre-scanner | Gemma 3 4B | Multimodal, converts image to text description |
| Segmentation | PaliGemma 2 3B (fine-tuned) | Outputs `<loc>` and `<seg>` tokens |
| Diagnosis | MedGemma 4B (fine-tuned) | Outputs natural language medical text |
| Result merger | Gemma 3 4B | Synthesizes multi-model outputs |
| Model serving | Ollama + HuggingFace Transformers | Local inference, no internet required |
| Backend | FastAPI | API endpoints + SSE streaming |
| Frontend | Next.js 14 + Tailwind CSS | Landing page + demo interface |
| Deployment | Vercel (frontend) + Local (backend) | Backend runs on-device on demo day |

### SSE Event Stream

Each processing step emits a named event to the frontend:

| Event | Message Shown to User |
|-------|-----------------------|
| `input_received` | Image and question received |
| `prescanning` | Scanning image content... |
| `prescan_complete` | Image identified: retinal fundus photograph |
| `routing` | Deciding analysis type... |
| `route_decided` | Route: Full Analysis (location + diagnosis) |
| `paligemma_start` | Locating anatomical structures... |
| `medgemma_start` | Analyzing for pathological conditions... |
| `paligemma_complete` | Found 2 regions of interest |
| `medgemma_complete` | Diagnosis analysis complete |
| `merging` | Combining results... |
| `complete` | {final JSON payload} |
| `error` | {error details} |

### Final JSON Response Shape

**analyze_location:**
```json
{
  "request_id": "abc123",
  "route": "analyze_location",
  "status": "success",
  "result": {
    "type": "location",
    "detections": [
      {
        "label": "hemorrhage",
        "confidence": 0.91,
        "bounding_box": { "x_min": 187, "y_min": 423, "x_max": 634, "y_max": 821 },
        "has_mask": true
      }
    ],
    "annotated_image_base64": "data:image/png;base64,...",
    "summary": "2 regions detected: hemorrhage in upper right quadrant."
  }
}
```

**analyze_diagnosis:**
```json
{
  "request_id": "abc124",
  "route": "analyze_diagnosis",
  "status": "success",
  "result": {
    "type": "diagnosis",
    "diagnosis": {
      "condition": "Non-Proliferative Diabetic Retinopathy",
      "severity": "Moderate",
      "severity_level": 3,
      "confidence": 0.84
    },
    "findings": [
      "Multiple hemorrhages in superior temporal quadrant",
      "Microaneurysms near macula"
    ],
    "recommendation": "Follow-up with ophthalmologist within 3-6 months",
    "disclaimer": "For research use only. Not intended for clinical diagnosis."
  }
}
```

**analyze_full:**
```json
{
  "request_id": "abc125",
  "route": "analyze_full",
  "status": "success",
  "result": {
    "type": "full",
    "location": {
      "detections": [...],
      "annotated_image_base64": "data:image/png;base64,..."
    },
    "diagnosis": {
      "condition": "Non-Proliferative Diabetic Retinopathy",
      "severity": "Moderate",
      "severity_level": 3,
      "findings": [...],
      "recommendation": "Follow-up within 3-6 months"
    },
    "summary": "Hemorrhage detected in upper right quadrant. Moderate NPDR confirmed.",
    "disclaimer": "For research use only. Not intended for clinical diagnosis."
  }
}
```

---

## 4. Directory Structure

```
optiassist/
├── agents.md                          # Coding standards (Cursor reads this)
├── backend/
│   ├── main.py                        # FastAPI entry point + SSE
│   ├── orchestrator.py                # Main pipeline logic
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── prescanner.py              # Stage 2: Gemma 3 image → text
│   │   ├── router.py                  # Stage 3: FunctionGemma routing
│   │   ├── segmenter.py               # Stage 4a: PaliGemma 2 + mask decode
│   │   ├── diagnostician.py           # Stage 4b: MedGemma inference
│   │   └── merger.py                  # Stage 4c: Gemma 3 result synthesis
│   ├── models/                        # Fine-tuned weights (from teammate)
│   │   ├── paligemma2-finetuned/      # config.json + model.safetensors
│   │   └── medgemma-finetuned/        # config.json + model.safetensors
│   ├── utils/
│   │   ├── image_utils.py             # Image preprocessing + mask overlay
│   │   └── token_parser.py            # Parse <loc> and <seg> tokens
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── app/
│   │   ├── page.tsx                   # Landing Page (/)
│   │   ├── demo/
│   │   │   └── page.tsx               # Demo Page (/demo)
│   │   └── layout.tsx
│   ├── components/
│   │   ├── ImageUpload.tsx            # Image drop zone + preview
│   │   ├── ProcessingFeed.tsx         # SSE live status feed
│   │   ├── AnnotatedImage.tsx         # Image + mask overlay display
│   │   ├── DiagnosisCard.tsx          # Diagnosis result display
│   │   └── RouteBadge.tsx             # Shows which models were called
│   ├── package.json
│   └── tailwind.config.js
└── data/
    └── sample_images/
        ├── normal_fundus.jpg
        ├── moderate_dr.jpg            # Main demo image
        └── glaucoma_suspect.jpg
```

---

## 5. Pre-Hackathon Setup (Day 0)

---

### Step 1 — YOU: Pull models via Ollama

Open terminal and run:

```bash
ollama pull gemma3:4b
ollama pull functiongemma
```

Test both are working:

```bash
ollama run gemma3:4b "Describe what a normal retinal fundus image looks like"
```

Type `/bye` to exit. Models are saved to `~/.ollama/models/`.

---

### Step 2 — YOU: Create project directory and Python environment

```bash
mkdir -p optiassist/backend/agents optiassist/backend/models/paligemma2-finetuned optiassist/backend/models/medgemma-finetuned optiassist/backend/utils optiassist/data/sample_images
cd optiassist/backend
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn python-multipart sse-starlette
pip install transformers torch pillow numpy
pip install python-dotenv requests
```

Verify:

```bash
python -c "import fastapi, transformers, PIL; print('All good!')"
```

---

### Step 3 — YOU: Download sample retinal images

Download publicly available retinal fundus images. Recommended source: DRIVE dataset or EyePACS public samples. You need at minimum:

- One **normal** fundus image
- One image showing **diabetic retinopathy** (moderate severity)
- One **glaucoma suspect** image

Place all in `optiassist/data/sample_images/`.

---

### Step 4 — YOU: Create GitHub repo and push skeleton

```bash
cd optiassist
git init
echo "backend/venv/" >> .gitignore
echo "backend/models/" >> .gitignore
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
touch backend/__init__.py
touch backend/agents/__init__.py
touch agents.md
git add .
git commit -m "Initial commit: project skeleton"
git remote add origin https://github.com/YOUR_USERNAME/optiassist.git
git push -u origin main
```

> Note: The repo can be **private**. v0 supports private repos — connect it at v0.dev → Import Project → authorize GitHub → select your private repo. v0 will raise PRs directly into the private repo.

---

### Step 5 — YOU: Initialize Next.js frontend skeleton and push

Before giving v0 any prompts, the repo needs a working Next.js skeleton. Run:

```bash
cd optiassist
npx create-next-app@latest frontend --typescript --tailwind --app --no-src-dir --import-alias "@/*"
cd frontend
git add .
git commit -m "Add Next.js frontend skeleton"
git push origin main
```

Now connect v0 to this repo at v0.dev. From this point v0 raises PRs into the repo.

---

### Step 6 — CURSOR: Create `agents.md` coding standards

Open Cursor in the `optiassist/` root directory. Create `agents.md`.

**Cursor Prompt:**

```
Create agents.md in the project root.

This file defines coding standards for all backend Python files in this project.

Include the following rules:
1. All functions must have type hints
2. All functions must have docstrings explaining input, output, and purpose
3. Use async/await for all I/O operations (model inference, file reading)
4. Every file must have a module-level docstring explaining what it does
5. Error handling: wrap all model inference calls in try/except, raise descriptive exceptions
6. No hardcoded paths — use environment variables or constants defined at the top of each file
7. Logging: use Python's built-in logging module, not print statements
8. Keep functions small and single-purpose — one function does one thing
9. All model loading should happen once at module level, not inside functions
10. Comments should explain WHY, not WHAT

Do not do anything beyond what I described above.
All code must be written in English. Reply to me in Chinese when chatting.
```

---

### Step 7 — CURSOR: Create `orchestrator.py`

Open Cursor in `optiassist/backend/`. Create `orchestrator.py`.

**Cursor Prompt:**

```
Create backend/orchestrator.py. Code style follows agents.md in the root directory.

This is the main pipeline orchestration for OptiAssist, an ophthalmology AI assistant.

The orchestrator exposes one async function:
  async def run_pipeline(image_bytes: bytes | None, question: str, emit) -> dict

Parameters:
  - image_bytes: raw image bytes, or None if text-only question
  - question: user's clinical question string
  - emit: async callback function, call it as await emit(event, message) to push SSE updates

Pipeline stages in order:

Stage 1 — Input parsing:
  - Check if image_bytes is present
  - If no image and no question, raise ValueError
  - await emit("input_received", "Image and question received")

Stage 2 — Image pre-scan (only if image_bytes is not None):
  - await emit("prescanning", "Scanning image content...")
  - Call: from agents.prescanner import prescan_image
  - result = await prescan_image(image_bytes)
  - await emit("prescan_complete", f"Image identified: {result}")
  - Store result as image_description

Stage 3 — FunctionGemma routing:
  - await emit("routing", "Deciding analysis type...")
  - Call: from agents.router import route_request
  - route = await route_request(question, image_description)
  - await emit("route_decided", f"Route: {route['function']}")

Stage 4 — Execute routed function:
  - If route["function"] == "analyze_location":
      await emit("paligemma_start", "Locating anatomical structures...")
      from agents.segmenter import run_segmentation
      location = await run_segmentation(image_bytes, route["query"])
      await emit("paligemma_complete", f"Found {len(location['detections'])} regions of interest")
      diagnosis = None

  - If route["function"] == "analyze_diagnosis":
      await emit("medgemma_start", "Analyzing for pathological conditions...")
      from agents.diagnostician import run_diagnosis
      diagnosis = await run_diagnosis(image_bytes, route["query"])
      await emit("medgemma_complete", "Diagnosis analysis complete")
      location = None

  - If route["function"] == "analyze_full":
      await emit("paligemma_start", "Locating anatomical structures...")
      await emit("medgemma_start", "Analyzing for pathological conditions...")
      from agents.segmenter import run_segmentation
      from agents.diagnostician import run_diagnosis
      import asyncio
      location, diagnosis = await asyncio.gather(
          run_segmentation(image_bytes, route["query"]),
          run_diagnosis(image_bytes, route["query"])
      )
      await emit("paligemma_complete", "Segmentation complete")
      await emit("medgemma_complete", "Diagnosis complete")

Stage 5 — Merge results:
  - await emit("merging", "Combining results...")
  - from agents.merger import merge_results
  - final = await merge_results(location, diagnosis, question)
  - await emit("complete", "Analysis complete")
  - Return final dict

Return shape: { "route": str, "result": dict }

Do not do anything beyond what I described above.
All code must be written in English. Reply to me in Chinese when chatting.
```

---

### Step 8 — CURSOR: Create `agents/prescanner.py`

**Cursor Prompt:**

```
Create backend/agents/prescanner.py. Code style follows agents.md in the root directory.

This module pre-scans a retinal image using Gemma 3 via Ollama and returns a brief text description.

Implement one async function:
  async def prescan_image(image_bytes: bytes) -> str

Steps:
1. Convert image_bytes to base64 string
2. Call Ollama API at http://localhost:11434/api/generate with:
   - model: "gemma3:4b"
   - prompt: "Describe this medical retinal image in 1-2 sentences. Focus on visible structures and any abnormalities. Be factual and concise."
   - images: [base64_string]
   - stream: false
3. Parse response JSON, return the "response" field as a string
4. If call fails, return "Retinal fundus image" as fallback

Use httpx for async HTTP calls.

Do not do anything beyond what I described above.
All code must be written in English. Reply to me in Chinese when chatting.
```

---

### Step 9 — CURSOR: Create `agents/router.py`

**Cursor Prompt:**

```
Create backend/agents/router.py. Code style follows agents.md in the root directory.

This module routes the user request to the correct analysis function using FunctionGemma via Ollama.

Implement one async function:
  async def route_request(question: str, image_description: str = "") -> dict

Steps:
1. Build a combined context string: question + image_description
2. Define 3 tools as a list of dicts in Ollama tool format:
   Tool 1 name: "analyze_location"
     description: "Use when user wants to locate, detect, or segment a specific anatomical structure or lesion in the retinal image. Trigger keywords: where, locate, show me, detect, segment, find"
     parameters: { query: { type: string, description: "the location query" } }
   Tool 2 name: "analyze_diagnosis"
     description: "Use when user wants a medical judgment, classification, disease identification, or risk assessment. Trigger keywords: is this, diagnosis, what disease, severity, risk, normal, condition"
     parameters: { query: { type: string, description: "the diagnostic query" } }
   Tool 3 name: "analyze_full"
     description: "Use when user wants both location information AND medical diagnosis together. Trigger keywords: full analysis, everything, what is wrong and where, complete report"
     parameters: { query: { type: string, description: "the full analysis query" } }

3. Call Ollama chat API at http://localhost:11434/api/chat with:
   - model: "functiongemma"
   - messages: [{ role: "user", content: combined context }]
   - tools: the 3 tools defined above
   - stream: false

4. Parse response to extract tool call name and arguments
5. Return: { "function": tool_name, "query": query_argument }
6. If parsing fails or no tool call, default to: { "function": "analyze_full", "query": question }

Use httpx for async HTTP calls.

Do not do anything beyond what I described above.
All code must be written in English. Reply to me in Chinese when chatting.
```

---

### Step 10 — CURSOR: Create `agents/segmenter.py`

**Cursor Prompt:**

```
Create backend/agents/segmenter.py. Code style follows agents.md in the root directory.

This module runs PaliGemma 2 inference for retinal image segmentation and returns annotated results.

MODEL_PATH constant at top of file: "./models/paligemma2-finetuned"

Load model once at module level using HuggingFace transformers:
  from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
  processor = AutoProcessor.from_pretrained(MODEL_PATH)
  model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_PATH)

Implement one async function:
  async def run_segmentation(image_bytes: bytes, query: str) -> dict

Steps:
1. Convert image_bytes to PIL Image
2. Build prompt: f"segment {query}\n"
3. Run inference (wrap blocking call in asyncio.to_thread):
   inputs = processor(text=prompt, images=pil_image, return_tensors="pt")
   outputs = model.generate(**inputs, max_new_tokens=256)
   raw_output = processor.decode(outputs[0], skip_special_tokens=False)

4. Parse <loc> tokens from raw_output:
   - Pattern: <loc(\d{4})> appears 4 times per detection (y_min, x_min, y_max, x_max)
   - Convert: coord = (token_value / 1024) * image_dimension
   - Extract label text after the 4 loc tokens

5. For mask decoding: if <seg> tokens are present, note them but for now
   set has_mask to False (mask decoding requires vae-oid.npz — coordinate with teammate)

6. Draw bounding boxes on original image using PIL ImageDraw
   Use red color with 2px width
   Convert annotated image to base64 PNG

7. Build summary string: "X regions detected: label1 in region, label2 in region"

Return:
{
  "detections": [
    {
      "label": str,
      "confidence": 0.9,
      "bounding_box": { "x_min": int, "y_min": int, "x_max": int, "y_max": int },
      "has_mask": bool
    }
  ],
  "annotated_image_base64": "data:image/png;base64,...",
  "summary": str
}

If model not found at MODEL_PATH, raise FileNotFoundError with clear message.

Do not do anything beyond what I described above.
All code must be written in English. Reply to me in Chinese when chatting.
```

---

### Step 11 — CURSOR: Create `agents/diagnostician.py`

**Cursor Prompt:**

```
Create backend/agents/diagnostician.py. Code style follows agents.md in the root directory.

This module runs MedGemma inference for medical diagnosis of retinal images.

MODEL_PATH constant at top of file: "./models/medgemma-finetuned"

Load model once at module level using HuggingFace transformers pipeline:
  from transformers import pipeline
  pipe = pipeline("image-text-to-text", model=MODEL_PATH)

Implement one async function:
  async def run_diagnosis(image_bytes: bytes | None, query: str) -> dict

Steps:
1. Convert image_bytes to PIL Image if present, else None
2. Build messages list:
   system message: "You are an expert ophthalmology AI assistant.
     Analyze the retinal image and answer the clinical question.
     Always respond with valid JSON only, no extra text.
     JSON fields required:
       condition: string (disease name or 'Normal')
       severity: string (None/Mild/Moderate/Severe/Proliferative)
       severity_level: integer 0-4
       confidence: float 0.0-1.0
       findings: list of strings (specific observations)
       recommendation: string (follow-up advice)
       disclaimer: always set to 'For research use only. Not intended for clinical diagnosis.'"
   user message: query (with image if present)

3. Run inference (wrap blocking call in asyncio.to_thread):
   output = pipe(text=messages, max_new_tokens=512)
   raw_text = output[0]["generated_text"][-1]["content"]

4. Parse JSON from raw_text:
   - Try json.loads(raw_text) directly
   - If fails, extract JSON block between first { and last }
   - If still fails, return a safe fallback dict with condition "Analysis unavailable"

Return the parsed dict.

If model not found at MODEL_PATH, raise FileNotFoundError with clear message.

Do not do anything beyond what I described above.
All code must be written in English. Reply to me in Chinese when chatting.
```

---

### Step 12 — CURSOR: Create `agents/merger.py`

**Cursor Prompt:**

```
Create backend/agents/merger.py. Code style follows agents.md in the root directory.

This module merges results from PaliGemma 2 and MedGemma into a unified response.

Implement one async function:
  async def merge_results(location: dict | None, diagnosis: dict | None, question: str) -> dict

Steps:
1. Build a context string summarizing available results:
   - If location is not None: include detections summary
   - If diagnosis is not None: include condition, severity, findings

2. Call Ollama API at http://localhost:11434/api/generate with:
   - model: "gemma3:4b"
   - prompt: f"You are a medical AI assistant. Summarize these ophthalmology analysis results in 2-3 clear sentences for a doctor. Question asked: {question}. Results: {context_string}"
   - stream: false

3. Parse response to get summary string

4. Determine result type:
   - Both present: "full"
   - Only location: "location"
   - Only diagnosis: "diagnosis"

5. Return:
{
  "type": result_type,
  "location": location,      # may be None
  "diagnosis": diagnosis,    # may be None
  "summary": summary_string,
  "disclaimer": "For research use only. Not intended for clinical diagnosis."
}

Use httpx for async HTTP calls.

Do not do anything beyond what I described above.
All code must be written in English. Reply to me in Chinese when chatting.
```

---

### Step 13 — CURSOR: Create `main.py`

**Cursor Prompt:**

```
Create backend/main.py. Code style follows agents.md in the root directory.

FastAPI application for OptiAssist. Two endpoints:

1. GET /health
   Returns: { "status": "ok", "models": ["functiongemma", "gemma3:4b", "paligemma2", "medgemma"] }

2. POST /analyze
   Accepts multipart form data:
     - image: UploadFile (optional)
     - question: str (required)
   Returns StreamingResponse with content-type "text/event-stream"

   SSE format for each event:
     data: {"event": "event_name", "message": "human readable message"}\n\n

   Final complete event format:
     data: {"event": "complete", "result": {full result dict}}\n\n

   Error event format:
     data: {"event": "error", "message": "error description"}\n\n

   Implementation:
   - Read image bytes from UploadFile if present, else None
   - Define an async emit(event, message) function that yields SSE formatted string
   - Use an asyncio.Queue to pass emitted events to the StreamingResponse generator
   - Call orchestrator.run_pipeline(image_bytes, question, emit)
   - Stream all queue items as SSE

Include CORS middleware allowing all origins (needed for local frontend dev).
Load environment variables from .env using python-dotenv.

Do not do anything beyond what I described above.
All code must be written in English. Reply to me in Chinese when chatting.
```

---

### Step 14 — YOU: Install dependencies and test backend

```bash
cd optiassist/backend
source venv/bin/activate
pip install httpx
pip freeze > requirements.txt
uvicorn main:app --port 8000 --reload
```

In a separate terminal, test health endpoint:

```bash
curl http://localhost:8000/health
```

Should return: `{"status":"ok","models":[...]}`

Test with a sample image (text-only first since models may not be ready):

```bash
curl -X POST http://localhost:8000/analyze \
  -F "question=What is diabetic retinopathy?"
```

Confirm SSE events stream correctly.

---

### Step 15 — v0: Build Landing Page

Go to v0.dev. Connect to your GitHub repo. Open a new chat.

**v0 Prompt:**

```
Build a Next.js landing page at app/page.tsx for "OptiAssist", an on-device AI assistant for ophthalmologists.

Style reference: ship26.instalily.ai
- Background: #0A0A0A
- Accent color: #2C7A4B (green)
- Monospace font for headings and labels
- Minimal sharp layout, no gradients, no rounded corners on main elements
- White body text, muted gray for secondary text

Sections in this exact order:

1. NAVBAR
   - Left: "OptiAssist" in monospace, green color
   - Right: two links — "Why We Built This" (scrolls to problem section) and "Try the Demo" button (green, links to /demo)

2. HERO
   - Large headline: "AI Assistant for Ophthalmologists"
   - Subheadline: "Real-time retinal analysis. Entirely on-device. Patient data never leaves the clinic."
   - Two buttons: [Try the Demo] links to /demo | [Why We Built This] scrolls to #problem

3. PROBLEM SECTION — id="problem"
   - Section title: "Why We Built This"
   - 4 cards in a 2x2 grid, dark card background (#141414), green left border
   - Card 1 — icon: lock | title: "Patient Data Privacy" | body: "Patient retinal images are protected under HIPAA and GDPR. They cannot leave the clinic. Uploading to cloud AI tools creates serious compliance and legal risk."
   - Card 2 — icon: zap | title: "Real-time Decisions Need Speed" | body: "Cloud round-trips add 500ms or more of delay. On-device inference responds in under 50ms — fast enough for live clinical consultation."
   - Card 3 — icon: globe | title: "Existing Tools Require Cloud Upload" | body: "Sending patient scans to external servers violates HIPAA and GDPR. Existing AI medical tools are not built for regulated clinical environments."
   - Card 4 — icon: brain | title: "Doctors Need Smarter Tools" | body: "Manual retinal image review is slow and prone to human error. AI detects patterns and anomalies instantly, every time."

4. HOW IT WORKS
   - Section title: "How It Works"
   - 3 horizontal steps with arrows between them:
     Step 1: "Capture or upload retinal image"
     Step 2: "Ask your clinical question"
     Step 3: "Get instant on-device analysis"

5. TECH STACK
   - Section title: "Powered By"
   - 4 items in a row, each showing model name + one-line role:
     FunctionGemma 270M — "Intelligent request routing"
     PaliGemma 2 3B — "Retinal structure segmentation"
     MedGemma 4B — "Ophthalmological diagnosis"
     Gemma 3 4B — "Image understanding & synthesis"

6. FOOTER
   - "OptiAssist — On-Device AI for Ophthalmology"
   - "For research use only. Not intended for clinical diagnosis."

Use React with useState for the smooth scroll behavior. Tailwind only, no external UI libraries. Lucide React for icons is allowed.

Do not do anything beyond what I described above.
All code must be written in English. Reply to me in Chinese when chatting.
```

---

### Step 16 — YOU: Merge Landing Page PR

1. In v0 right panel, click **Merge PR**
2. Locally:

```bash
cd optiassist
git pull origin main
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` to verify.

---

### Step 17 — v0: Build Demo Page

**v0 Prompt:**

```
Build a Next.js demo page at app/demo/page.tsx for OptiAssist.

Same dark style as landing page: background #0A0A0A, accent #2C7A4B, monospace font for labels.

LAYOUT: Two equal columns side by side (grid-cols-2), full viewport height.

LEFT COLUMN — Input Panel:
  - Section label: "INPUT" in monospace, small, muted
  - Image drop zone:
    - Dashed border, dark background
    - Text: "Drop retinal image here or click to upload"
    - On hover: green dashed border
    - On upload: show image preview filling the zone
    - Accepts: image/*
  - Text input below the drop zone:
    - Placeholder: "Ask a clinical question..."
    - Dark background, green border on focus
    - Full width
  - Two buttons below:
    - [Analyze] — green, full width, disabled if no question entered
    - [Reset] — outline style, full width, clears everything
  - Small note: "Supported: JPG, PNG retinal fundus images"

RIGHT COLUMN — Output Panel:
  - Section label: "ANALYSIS" in monospace, small, muted
  - Default state (before analysis): show placeholder text "Analysis results will appear here"
  
  - During and after analysis, show three stacked sections:

  SECTION A — Processing Feed (shows immediately when Analyze is clicked):
    - Label: "Processing Steps"
    - Each SSE event appears as a new row:
      - Completed steps: green checkmark + event message + timestamp
      - Current step: spinning indicator + message
    - Monospace font, small text

  SECTION B — Annotated Image (shows after paligemma_complete event):
    - Label: "Segmentation Result" — only show if route includes segmentation
    - Display the annotated_image_base64 from result
    - Full width image

  SECTION C — Diagnosis Card (shows after complete event):
    - Route badge top right: "Segmentation" (blue) / "Diagnosis" (green) / "Full Analysis" (purple)
    - Condition name: large text, white
    - Severity badge: color coded — None=gray, Mild=yellow, Moderate=orange, Severe=red, Proliferative=dark red
    - Findings: bullet list
    - Recommendation: italic text
    - Disclaimer: small muted text at bottom

STATE MANAGEMENT:
  - useState for: image file, question, processing steps array, final result, isLoading
  - On Analyze click: use fetch with ReadableStream to consume SSE from backend
  - Backend URL: process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"
  - Parse each SSE line: lines starting with "data: " contain JSON
  - On each event: append to processing steps array
  - On "complete" event: parse result and set final result state
  - On "error" event: show error message in processing feed

SSE CONSUMPTION PATTERN:
  const response = await fetch(`${BACKEND_URL}/analyze`, { method: "POST", body: formData })
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    const text = decoder.decode(value)
    // parse lines starting with "data: "
  }

Tailwind only. No external UI libraries. Lucide React for icons is allowed.

Do not do anything beyond what I described above.
All code must be written in English. Reply to me in Chinese when chatting.
```

---

### Step 18 — YOU: Merge Demo Page PR and Deploy Vercel

```bash
cd optiassist
git pull origin main
cd frontend
npm install
npm run dev
```

Test full flow at `http://localhost:3000/demo`.

Deploy to Vercel:
1. Go to vercel.com → Add New Project → select `optiassist` repo
2. **IMPORTANT:** Framework Preset = `Next.js`, Root Directory = `frontend`
3. Add environment variable: `NEXT_PUBLIC_BACKEND_URL` = your ngrok URL (see Day-of step)
4. Deploy

If 404 after deploy: Settings → Build and Deployment → confirm Framework = Next.js, Root Directory = frontend → Redeploy.

---

## 6. Hackathon Day Operations (2.28)

### 9:00 AM — Run immediately on arrival

```bash
# Start Ollama if not running
ollama serve

# Start backend
cd optiassist/backend
source venv/bin/activate
uvicorn main:app --port 8000

# Confirm backend is live
curl http://localhost:8000/health
```

### If teammate has finished fine-tuning

Copy model folders into:
```
optiassist/backend/models/paligemma2-finetuned/
optiassist/backend/models/medgemma-finetuned/
```

Then restart backend. Test with sample image end-to-end.

### If you need to expose backend to Vercel frontend

```bash
ngrok http 8000
```

Copy the HTTPS URL. Go to Vercel project → Settings → Environment Variables → update `NEXT_PUBLIC_BACKEND_URL` → Redeploy.

### Demo Day Checklist

- [ ] `ollama list` confirms `functiongemma` and `gemma3:4b` are present
- [ ] Fine-tuned PaliGemma 2 weights in `backend/models/paligemma2-finetuned/`
- [ ] Fine-tuned MedGemma weights in `backend/models/medgemma-finetuned/`
- [ ] `curl http://localhost:8000/health` returns ok
- [ ] Landing page loads correctly with dark theme
- [ ] Demo page upload + question flow works end-to-end
- [ ] SSE stream visible in browser (live steps appear in real time)
- [ ] Sample images ready: `normal_fundus.jpg`, `moderate_dr.jpg`, `glaucoma_suspect.jpg`
- [ ] ngrok installed as backup
- [ ] Demo script rehearsed

---

## 7. Teammate Handover Interface

> Align on everything in this section before either person writes code.

### What the teammate delivers

Two folders placed in `optiassist/backend/models/`:

```
backend/models/
├── paligemma2-finetuned/
│   ├── config.json
│   ├── model.safetensors
│   └── processor_config.json
└── medgemma-finetuned/
    ├── config.json
    ├── model.safetensors
    └── tokenizer_config.json
```

Optionally: `vae-oid.npz` in `backend/utils/` for PaliGemma 2 segmentation mask decoding.

### Questions to confirm before fine-tuning starts

| Question | For PaliGemma 2 | For MedGemma |
|----------|-----------------|--------------|
| Exact prompt format used during training | `segment {query}\n` or different? | System prompt format? |
| Image resolution trained on | 224px, 448px, or 896px? | Standard |
| Labels / classes fine-tuned for | e.g. hemorrhage, optic disc, blood vessels, macula | DR severity 0-4? Other conditions? |
| VQ-VAE decoder included? | Need `vae-oid.npz` for pixel mask — confirm | N/A |
| Output format | Raw `<loc><seg>` tokens | JSON string or free text? |
| HuggingFace loading class | `PaliGemmaForConditionalGeneration` | `pipeline("image-text-to-text")` |

### Backend integration test (run after teammate delivers models)

```bash
cd optiassist/backend
source venv/bin/activate
python -c "
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
model = PaliGemmaForConditionalGeneration.from_pretrained('./models/paligemma2-finetuned')
print('PaliGemma 2 loaded OK')
"

python -c "
from transformers import pipeline
pipe = pipeline('image-text-to-text', model='./models/medgemma-finetuned')
print('MedGemma loaded OK')
"
```

---

## 8. Prompt Reference Summary

### Cursor Prompts (Backend)

| Step | File | Purpose |
|------|------|---------|
| Step 6 | `agents.md` | Coding standards for all backend Python |
| Step 7 | `orchestrator.py` | Main pipeline: input → prescan → route → execute → merge |
| Step 8 | `agents/prescanner.py` | Gemma 3 image pre-scan → text description |
| Step 9 | `agents/router.py` | FunctionGemma routing → function call output |
| Step 10 | `agents/segmenter.py` | PaliGemma 2 inference + loc/seg token parsing + bounding box overlay |
| Step 11 | `agents/diagnostician.py` | MedGemma inference + JSON output parsing |
| Step 12 | `agents/merger.py` | Gemma 3 result synthesis → unified summary |
| Step 13 | `main.py` | FastAPI endpoints + SSE streaming + CORS |

### v0 Prompts (Frontend)

| Step | Page | Purpose |
|------|------|---------|
| Step 15 | `app/page.tsx` | Landing page: dark theme, 4 problem cards, how it works, tech stack |
| Step 17 | `app/demo/page.tsx` | Demo page: two-column layout, SSE live feed, segmentation + diagnosis results |

### Manual Steps (No Prompts)

| Step | Action |
|------|--------|
| Step 1 | `ollama pull gemma3:4b && ollama pull functiongemma` |
| Step 2 | Create directory structure and Python venv |
| Step 3 | Download sample retinal images |
| Step 4 | Create GitHub repo and push skeleton |
| Step 5 | Initialize Next.js skeleton and push (required before connecting v0) |
| Step 14 | Test backend end-to-end with curl |
| Step 16 | Merge v0 landing page PR → git pull → verify localhost:3000 |
| Step 18 | Merge v0 demo page PR → git pull → npm install → deploy Vercel |

---

## 9. Risk Register

| Risk | Trigger | Mitigation |
|------|---------|------------|
| Fine-tuned models not ready in time | Teammate cannot finish before demo | Use base PaliGemma 2 + base MedGemma — still demonstrates the routing architecture, just without fine-tuning performance delta |
| PaliGemma 2 seg tokens fail to decode | `vae-oid.npz` decoder missing or broken | Fall back to bounding boxes only (no pixel mask) — still visually compelling for demo |
| MedGemma output not valid JSON | Free text instead of structured output | Add stricter JSON instruction to system prompt + add retry with simplified prompt |
| FunctionGemma routes incorrectly | Wrong model called for question type | Fallback: if parsing fails, default to `analyze_full` (calls both models) |
| SSE stream drops in browser | Frontend ReadableStream disconnects | Fall back to polling: GET `/status/{job_id}` every 2 seconds |
| Models too slow for live demo | Inference takes >60 seconds | Pre-run on demo image before presentation, show cached result + narrate the pipeline |
| v0 generates incompatible component structure | Next.js App Router mismatch | Fix in Cursor after merging PR — v0 generates the skeleton, Cursor does cleanup |
| ngrok not available on demo day | Cannot expose local backend | Run entire demo on one machine with frontend pointing to `localhost:8000` directly |
| Teammate fine-tuned on different prompt format | Segmenter.py prompt doesn't match training | Update prompt constant in `segmenter.py` to match whatever format teammate used |

---

## 10. Demo Script

### 30-Second Pitch

**Opening (10 seconds):**
> "A doctor takes a retinal scan. That image is protected by HIPAA and GDPR — it cannot leave the clinic. But the doctor still needs AI assistance in real time. That's exactly why this has to run on-device."

**Solution (10 seconds):**
> "OptiAssist runs entirely on this machine using Google DeepMind's Gemma family models. The patient image never touches a cloud server. The doctor asks a clinical question — the system routes it intelligently to the right AI model and returns an answer in seconds."

**Live Demo — run in this order:**

1. Open landing page — point out the 4 problem cards, emphasize **Patient Data Privacy**
2. Click **Try the Demo**
3. Upload `moderate_dr.jpg`
4. Type: `"Does this look like diabetic retinopathy, and where are the lesions?"`
5. Click **Analyze** — narrate the live SSE feed as each step appears:
   - "Here you can see FunctionGemma deciding which models to call..."
   - "It routed to Full Analysis — both PaliGemma 2 and MedGemma are running in parallel..."
   - "PaliGemma 2 has found the lesion regions and drawn the segmentation overlay..."
   - "MedGemma has confirmed: Moderate NPDR, follow-up recommended within 3-6 months..."
6. Point out the annotated image with bounding boxes
7. Point out the diagnosis card: condition, severity badge, findings list, recommendation

**Closing (10 seconds):**
> "Same architecture, any medical imaging specialty — dermatology, radiology, pathology. Swap the fine-tuned models, same pipeline. And because it's on-device, it works in hospitals with no internet, in rural clinics, anywhere a doctor needs AI without compromising patient privacy."

---

*Document version: v1.0 | OptiAssist Team | 2026.02.28*

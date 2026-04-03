# OptiAssist 👁️

> **Fully local AI diagnostic assistant for ophthalmologists.** Patient data never leaves the clinic.

OptiAssist is a multi-model AI pipeline that analyzes retinal fundus images and answers clinical questions in real time — all on-device, with zero cloud dependency. It combines four specialized Google AI models orchestrated through a 5-stage pipeline, streaming live progress updates to the clinician as each analysis step completes.

---

## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/Hzx4GWAeHUg/maxresdefault.jpg)](https://youtu.be/Hzx4GWAeHUg)

- **Image:** `data/sample_images/issue-img.png`
- **Question:** `What is the cup-to-disc ratio?`

---

## ✨ Key Features

- **100% Local Inference** — HIPAA/GDPR-friendly; images and patient data never leave the machine
- **Multi-Model Pipeline** — Four specialized AI models working in concert: FunctionGemma, Gemma 3, PaliGemma 2, and MedGemma
- **Real-Time Streaming** — Server-Sent Events push live progress updates for every pipeline stage
- **Intelligent Routing** — FunctionGemma 270M autonomously decides which models to invoke based on the clinical question
- **Optic Structure Segmentation** — Fine-tuned PaliGemma 2 detects optic disc and optic cup bounding boxes for CDR / glaucoma assessment
- **Structured Diagnosis** — Fine-tuned MedGemma 4B returns condition, severity (None/Mild/Moderate/Severe/Proliferative), confidence, findings, and recommendations as structured JSON

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js 16)                     │
│                                                              │
│  /  Landing Page          /demo  Interactive Demo           │
│     Hero, Problem,               Input Panel (upload +      │
│     HowItWorks, TechStack        question) + SSE feed       │
└──────────────────────────────┬──────────────────────────────┘
                               │  POST /analyze (multipart)
                               │  ← SSE Stream (text/event-stream)
┌──────────────────────────────▼──────────────────────────────┐
│                  BACKEND (FastAPI + Uvicorn)                  │
│                                                              │
│  Stage 1  Input Validation                                   │
│      ↓                                                       │
│  Stage 2  prescanner.py  →  Gemma 3 4B (Ollama)             │
│      ↓                                                       │
│  Stage 3  router.py      →  FunctionGemma 270M (Ollama)     │
│      ↓                                                       │
│  Stage 4  segmenter.py   →  PaliGemma 2 3B (HuggingFace)   │
│           diagnostician.py → MedGemma 4B (HuggingFace)      │
│      ↓                                                       │
│  Stage 5  merger.py      →  Gemma 3 4B (Ollama)             │
│                                                              │
│  Local Model Runtime:                                        │
│  · Ollama (localhost:11434): Gemma 3, FunctionGemma         │
│  · HuggingFace Transformers: PaliGemma 2, MedGemma          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Models

| Model | Parameters | Runtime | Role |
|-------|-----------|---------|------|
| **Gemma 3 4B** | 4B | Ollama | Image pre-scan description + final narrative summary |
| **FunctionGemma** | 270M | Ollama | Intelligent routing via function calling |
| **PaliGemma 2** *(custom LoRA)* | 3B | HuggingFace Transformers | Optic disc/cup bounding box segmentation |
| **MedGemma** | 4B | HuggingFace Transformers | Ophthalmic disease diagnosis + structured report |

### 🔬 Custom Fine-Tuned PaliGemma 2

The PaliGemma 2 model used in OptiAssist is **our own LoRA fine-tune**, trained specifically for retinal optic structure detection (optic disc and optic cup bounding boxes).

| Training Detail | Value |
|----------------|-------|
| Base model | `google/paligemma2-3b-pt-448` |
| Fine-tuning method | LoRA (PEFT) |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs | 8 |
| Total steps | 488 |
| Batch size | 4 |
| Best checkpoint | Step 427 (eval loss `0.5716`) |
| Input resolution | 448 × 448 |
| Task | Object detection via `<loc####>` tokens |

The adapter weights (`adapter_model.safetensors`) are loaded at runtime via `peft.PeftModel`, then merged into the base model for inference. Weights are stored locally under `backend/models/paligemma2-finetuned/`.

---

## 📋 Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **npm**
- **[Ollama](https://ollama.com/)** installed and running locally
- *(Optional)* CUDA-capable GPU or Apple Silicon for faster inference

---

## 🚀 Quick Start

### 1. Start Ollama and pull models

```bash
ollama serve

# In a new terminal:
ollama pull gemma3:4b
ollama pull functiongemma
```

### 2. Start the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.  
PaliGemma 2 is preloaded at startup — the first request will not be stalled by model loading.

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 4. Try the demo

Navigate to [http://localhost:3000/demo](http://localhost:3000/demo) and use the following sample inputs:

| Field | Value |
|-------|-------|
| **Image** | `data/sample_images/issue-img.png` |
| **Question** | `What is the cup-to-disc ratio?` |

---

## 📁 Project Structure

```
teamlastmin/
├── backend/
│   ├── main.py               # FastAPI entry point + SSE streaming
│   ├── orchestrator.py       # 5-stage pipeline coordinator
│   ├── agents/
│   │   ├── prescanner.py     # Stage 2: image description (Gemma 3)
│   │   ├── router.py         # Stage 3: agentic routing (FunctionGemma)
│   │   ├── segmenter.py      # Stage 4a: optic structure detection (PaliGemma 2)
│   │   ├── diagnostician.py  # Stage 4b: disease diagnosis (MedGemma)
│   │   ├── merger.py         # Stage 5: result synthesis (Gemma 3)
│   │   └── cup_disc_tools.py # CDR calculation utilities
│   ├── models/
│   │   ├── paligemma2-finetuned/   # Fine-tuned PaliGemma 2 weights
│   │   └── medgemma-finetuned/     # Fine-tuned MedGemma weights
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx          # Landing page
│   │   └── demo/page.tsx     # Interactive demo
│   ├── components/           # Reusable UI components
│   └── package.json
├── app/
│   └── tools/
│       └── paligemma_tool.py # Shared PaliGemma inference tool
└── data/
    └── sample_images/        # Sample retinal fundus images for testing
```

---

## 🔌 API Reference

### `GET /health`

Returns service health and active models.

```json
{
  "status": "ok",
  "models": ["functiongemma", "gemma3:4b", "paligemma2", "medgemma"]
}
```

### `POST /analyze`

Runs the full analysis pipeline and streams results via SSE.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | ✅ | Clinical question (e.g., *"What is the cup-to-disc ratio?"*) |
| `image` | File | ❌ | Retinal fundus image (JPG / PNG) |

**Response**: `text/event-stream`

SSE events emitted in order:

```
input_received      → Input accepted
prescanning         → Gemma 3 scanning the image
prescan_complete    → Image description generated
routing             → FunctionGemma deciding route
route_decided       → Route confirmed
paligemma_start     → PaliGemma 2 inference started  (if applicable)
medgemma_start      → MedGemma inference started
paligemma_complete  → Segmentation result ready
medgemma_complete   → Diagnosis result ready
merging             → Gemma 3 synthesising summary
complete            → Final result payload delivered
```

**Final `complete` payload shape**:

```typescript
{
  route: "Diagnosis" | "Segmentation" | "Full Analysis",
  result: {
    type: "full" | "location" | "diagnosis",
    location: SegmentationResult | null,
    diagnosis: DiagnosisResult | null,
    summary: string,        // Gemma 3 narrative
    disclaimer: string
  }
}
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_BACKEND_URL` | `http://localhost:8000` | Backend URL used by the frontend |

Set environment variables in `frontend/.env.local`.

---

## ⏱️ Latency Estimates (Full Analysis)

| Stage | Estimated Time |
|-------|---------------|
| Input validation | < 1 ms |
| Gemma 3 pre-scan | 2–5 s |
| FunctionGemma routing | 0.5–1 s |
| PaliGemma 2 + MedGemma *(parallel)* | 3–8 s each → `max` of both |
| Gemma 3 summary | 1–3 s |
| **Total (GPU)** | **~8–18 s** |
| **Total (CPU)** | **~30–90 s** |

PaliGemma 2 and MedGemma run in **parallel** via `asyncio.gather()` — total time equals the slower of the two, not their sum.

---

## 🛡️ Privacy & Disclaimer

- All model inference runs **entirely on-device**. No data is sent to external servers.
- OptiAssist is a **research prototype** and is **not intended for clinical diagnosis**. Always consult a qualified ophthalmologist.

---

## 🧰 Tech Stack

**Backend**: FastAPI · Uvicorn · HuggingFace Transformers · PyTorch · Pillow · httpx · Ollama

**Frontend**: Next.js 16 · React 19 · TypeScript · Tailwind CSS · Radix UI · Lucide React

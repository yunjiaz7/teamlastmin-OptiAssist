## Doctor Fundus AI Assistant - Build Plan

### Goal
Build an app where a doctor:
- uploads a fundus image,
- asks a clinical question,
- an agent chooses tool calls to `MedGemma` and/or `finetuned_paligemma2_det_lora`,
- and returns one final grounded answer with evidence.

### Product Scope (V1)
- Single-image input (JPG/PNG), single doctor query.
- Tool-enabled agent with deterministic routing + optional dual-run fallback.
- Final response includes:
  - concise clinical answer,
  - findings from each tool,
  - confidence and uncertainty statement,
  - recommendation to verify with clinical judgment.

### Proposed System Architecture

1) Frontend (doctor UI)
- Upload component for fundus image.
- Query textbox ("What do you want to know?").
- Result panel:
  - Final answer,
  - Structured findings (disc/cup, abnormalities, lesion hints),
  - Optional image overlay from PaliGemma detections.

2) Backend API (FastAPI)
- `POST /analyze`
  - multipart form: `image`, `query`
  - returns JSON with final answer + tool traces.
- `GET /health` for readiness checks.

3) Agent Orchestrator (tool-calling core)
- Input: doctor query + image path.
- Tool registry:
  - `run_paligemma_detection(image, query_context)`
  - `run_medgemma_vqa(image, query)`
- Router policy:
  - If query asks location/objects/lesions/disc-cup geometry -> prioritize PaliGemma.
  - If query asks diagnosis/explanation/risk/management -> prioritize MedGemma.
  - If query is broad/ambiguous -> call both, then synthesize.
- Synthesis step:
  - merge outputs into one final response template,
  - resolve conflicts by stating disagreement explicitly,
  - never fabricate unsupported findings.

4) Model Tool Adapters
- PaliGemma adapter:
  - load base model + LoRA adapter from `finetuned_paligemma2_det_lora`,
  - run generation,
  - parse `<loc####>` boxes and labels (reuse logic from `inference.py`),
  - return normalized structured JSON.
- MedGemma adapter:
  - standardized callable interface (same output schema),
  - include model prompt template for clinical Q/A grounded in image.

5) Response Contract (single schema for UI + audit)
- `final_answer: str`
- `decision_path: {router_reason, tools_called}`
- `tool_outputs: {paligemma: ..., medgemma: ...}`
- `confidence: low|medium|high`
- `safety_notes: [..]`
- `disclaimer: str`

### Routing Logic (simple, reliable first)
- Start rule-based (keyword + intent tags) for predictability and debugging.
- Upgrade later to LLM-router once logs are available.
- Always keep a hard fallback:
  - if one tool errors, continue with the other and mark partial result.

### Safety + Clinical Guardrails
- Add hard disclaimer: "Decision support only; not a diagnosis."
- Block unsupported claims:
  - synthesis can only use evidence present in tool outputs.
- Include uncertainty language when confidence is low.
- Log anonymized traces for review (no patient identifiers in logs).

### Implementation Steps

Phase 1 - Backend skeleton
- Create FastAPI service + upload handling.
- Add response schema models (Pydantic).

Phase 2 - Tool adapters
- Refactor `inference.py` logic into reusable `tools/paligemma_tool.py`.
- Implement `tools/medgemma_tool.py` with same interface.

Phase 3 - Orchestrator
- Build `agent/router.py` and `agent/synthesizer.py`.
- Implement rule-based routing + synthesis templates.

Phase 4 - Frontend
- Lightweight UI (Streamlit or React) with upload + query + results.

Phase 5 - Validation
- Run 20-50 curated cases.
- Compare tool outputs vs final synthesis for hallucination leakage.
- Tune routing rules and confidence thresholds.

### Recommended Initial File Layout
- `app/main.py` (FastAPI entrypoint)
- `app/schemas.py` (request/response models)
- `app/agent/router.py`
- `app/agent/synthesizer.py`
- `app/tools/paligemma_tool.py`
- `app/tools/medgemma_tool.py`
- `app/services/image_store.py`
- `ui/app.py` (Streamlit UI)

### Definition of Done (V1)
- Doctor can upload image + ask question in UI.
- Agent calls at least one tool and can call both when needed.
- Final answer is returned with evidence and disclaimer.
- Tool-call trace is visible for debugging.
- Basic test coverage for router + output schema + one end-to-end happy path.

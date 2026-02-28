# TeamLastMin - Ophthalmology Fundus Assistant

This repository contains a local-first workflow for:
- converting ORIGA retinal data into PaliGemma detection format,
- fine-tuning `google/paligemma2-3b-pt-448` with LoRA for optic disc/cup detection,
- running detection inference with visual overlays,
- orchestrating a multi-tool clinical assistant (PaliGemma + MedGemma + FuncGemma) with LangGraph.

The project is intended for research and decision support. It is **not** a diagnostic system.

## What is included

- **Dataset conversion**: `data/scripts/convert_origa.py`
- **LoRA training**: `train.py`
- **Detection inference + visual reports**: `inference.py`
- **Agent runner**: `run_agent.py`
- **LangGraph orchestration**: `app/agent/langgraph_agent.py`
- **Tool adapters**:
  - `app/tools/paligemma_tool.py`
  - `app/tools/ollama_tools.py`

## Repository layout

```text
teamlastmin/
├── app/
│   ├── agent/langgraph_agent.py
│   └── tools/
│       ├── ollama_tools.py
│       └── paligemma_tool.py
├── data/scripts/convert_origa.py
├── dataset/origa_paligemma/
├── finetuned_paligemma2_det_lora/
├── outputs/inference_samples/
├── train.py
├── inference.py
└── run_agent.py
```

## Requirements

- Linux with NVIDIA GPU recommended for training/inference
- Python 3.10+
- Local model availability for offline (`local_files_only=True`) loading:
  - base model: `google/paligemma2-3b-pt-448`
  - LoRA adapter checkpoints under `finetuned_paligemma2_det_lora/`
- Optional for agent workflow: Ollama running locally

## Setup

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install core packages used by the agent:

```bash
pip install -r requirements-agent.txt
```

Install training/inference dependencies:

```bash
pip install datasets numpy
```

If not already installed in your environment, also install:

```bash
pip install ollama
```

## 1) Convert ORIGA to PaliGemma JSONL format

`convert_origa.py` reads ORIGA split images and masks, computes optic-disc/optic-cup bounding boxes, and writes JSONL rows with:
- `prefix`: `detect optic-disc ; optic-cup`
- `suffix`: `<loc...><loc...><loc...><loc...> optic-disc ; ... optic-cup`

Run:

```bash
python data/scripts/convert_origa.py \
  --origa-root /home/ubuntu/glaucoma/dataset/ORIGA \
  --output-dir /home/ubuntu/teamlastmin/dataset/origa_paligemma/dataset
```

Expected outputs:
- `dataset/origa_paligemma/dataset/_annotations.train.jsonl`
- `dataset/origa_paligemma/dataset/_annotations.valid.jsonl`
- `dataset/origa_paligemma/dataset/_annotations.test.jsonl`
- copied images under `dataset/origa_paligemma/dataset/images/{train,val,test}/`

## 2) Train PaliGemma LoRA detector

Training script: `train.py`

Key points:
- uses `google/paligemma2-3b-pt-448`,
- freezes vision tower/projector,
- applies LoRA on language modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`),
- logs validation mIoU by decoding generated suffixes and comparing IoU to ground truth.

Run:

```bash
python train.py
```

Default output directory:
- `finetuned_paligemma2_det_lora/`
- final adapter in `finetuned_paligemma2_det_lora/final/`

## 3) Run detection inference samples

Inference script: `inference.py`

It loads the LoRA adapter, runs generation on samples from a JSONL annotation file, draws predicted boxes, and writes:
- `outputs/inference_samples/sample_XX.png`
- `outputs/inference_samples/results.json`
- `outputs/inference_samples/results.md`

Run:

```bash
python inference.py \
  --adapter-dir /home/ubuntu/teamlastmin/finetuned_paligemma2_det_lora/final \
  --annotations /home/ubuntu/teamlastmin/dataset/origa_paligemma/dataset/_annotations.valid.jsonl \
  --image-root /home/ubuntu/teamlastmin/dataset/origa_paligemma/dataset \
  --num-samples 5 \
  --output-dir /home/ubuntu/teamlastmin/outputs/inference_samples
```

## 4) Run the LangGraph clinical assistant

Entrypoint: `run_agent.py`

Flow:
1. Router classifies query intent (detection vs interpretation vs both),
2. Runs tool(s):
   - PaliGemma detection tool for structured box output,
   - MedGemma VQA via Ollama for interpretation,
3. FuncGemma synthesizer combines tool evidence into final answer + confidence.

Routing policy:
- Interpretation-only questions (for example DR presence/risk wording) call MedGemma and skip PaliGemma.
- Structural localization queries (disc/cup boundaries, segmentation, box location) call PaliGemma.
- CDR queries call both PaliGemma and `cdr_calculator`.

Start Ollama and ensure required models are available:

```bash
ollama serve
ollama pull bentplau/medgemma1.5-4b-it
ollama pull funcgemma
```

Run agent:

```bash
python run_agent.py \
  --image /path/to/fundus.png \
  --query "Locate optic disc/cup and summarize glaucoma risk"
```

Model configuration defaults are stored in `config.json`:
- `ollama_host` (default `http://127.0.0.1:11434`)
- `medgemma_model` (default `bentplau/medgemma1.5-4b-it`)
- `funcgemma_model` (default `funcgemma`)

MedGemma troubleshooting:
- If Ollama returns an image-input error (for example missing image data/status 500), verify `medgemma_model` in `config.json` points to a vision-capable model tag such as `bentplau/medgemma1.5-4b-it`.
- Make sure the image path passed to `run_agent.py` exists and is readable.

## Notes and limitations

- Paths in scripts default to local absolute paths in this workspace; adjust flags if your layout differs.
- Inference and tool adapters use local-only model loading; required model artifacts must already exist on disk.
- Outputs are for research and decision support only, and must be reviewed by qualified clinicians.

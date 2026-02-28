# OptiAssist — 完整技术文档与操作手册

> 眼科医生本地端 AI 助手 | v1.0 | 2026.02.28

---

## 目录

1. [项目概述](#1-项目概述)
2. [最终产品形态](#2-最终产品形态)
3. [技术架构](#3-技术架构)
4. [目录结构](#4-目录结构)
5. [比赛前准备（Day 0）](#5-比赛前准备day-0)
6. [比赛当天操作（2.28）](#6-比赛当天操作228)
7. [队友交接接口](#7-队友交接接口)
8. [Prompt 大全](#8-prompt-大全)
9. [风险备案](#9-风险备案)
10. [Demo 脚本](#10-demo-脚本)

---

## 1. 项目概述

### 一句话描述

上传或拍摄一张眼底图像，用自然语言提问，系统在本地完成 AI 分析——结合精确的解剖结构分割和医学诊断——30 秒内返回结果。患者数据全程不离开设备。

### 核心价值

- 眼底图像受 HIPAA 和 GDPR 保护——法律上不允许传输到云端服务器
- 眼科医生在问诊时需要实时 AI 辅助，但云端来回延迟超过 500ms
- 现有 AI 工具要求将敏感患者数据上传到外部 API——存在合规和隐私风险
- 人工看图慢且容易出错；AI 能即时识别异常模式
- OptiAssist 完全使用 Google DeepMind Gemma 家族模型在本地运行——零数据外传

### 技术亮点

- 本地多模型路由：FunctionGemma 270M 作为智能调度器
- PaliGemma 2 3B（队友微调版）负责像素级眼底结构分割
- MedGemma 4B（队友微调版）负责眼科诊断与分类
- Gemma 3 4B 负责图像预扫描和结果综合
- SSE 实时流式推送——评审能看到每一步推理过程
- 支持纯文字提问、图片+提问、完整分析三种输入模式

---

## 2. 最终产品形态

### 首页（`/`）

风格参考：ship26.instalily.ai——深色背景 `#0A0A0A`，绿色强调色 `#2C7A4B`，等宽字体，极简锐利布局。

- Hero 区：项目名 + 一句话描述 + 两个按钮（Try the Demo / Why We Built This）
- 痛点区（4 张卡片）：患者数据隐私 | 实时响应速度 | 现有工具违反合规 | 医生需要更智能的工具
- 使用流程：3 步可视化说明
- 技术栈：列出全部 4 个模型及其一句话职责
- `Try the Demo` 按钮 → 跳转 `/demo`

### Demo 页（`/demo`）

- 左侧面板：图片上传 / 摄像头拍摄 + 文字问题输入框 + Analyze 按钮
- 右侧上方：带分割 mask 叠加的标注眼底图
- 右侧中间：诊断结果卡片——疾病名称、严重程度、发现、建议
- 右侧下方：SSE 实时处理进度——每一步骤实时出现
- 路由标签：显示调用了哪些模型（Segmentation / Diagnosis / Full Analysis）
- 免责声明："仅供研究使用，不作为临床诊断依据。"

---

## 3. 技术架构

### 系统总览

```
用户输入（可选图片 + 问题文字）
            ↓
    ┌─────────────────────┐
    │   输入解析器         │  有图片？有问题？
    └─────────────────────┘
            ↓
    ┌─────────────────────┐
    │  Gemma 3 预扫描      │  （仅当有图片时）
    │  图片 → 文字描述     │  "眼底图显示..."
    └─────────────────────┘
            ↓
    ┌─────────────────────┐
    │   FunctionGemma     │  读取问题 + 图片描述
    │   路由器             │  决定调用哪个函数
    └─────────────────────┘
            ↓
    ┌───────────────────────────────────────┐
    │            工具执行层                  │
    │                                       │
    │  analyze_location()                   │
    │  → PaliGemma 2（微调版）              │
    │    输出 <loc><seg> tokens             │
    │    后处理 → mask 叠加图               │
    │                                       │
    │  analyze_diagnosis()                  │
    │  → MedGemma（微调版）                 │
    │    输出自然语言 JSON                   │
    │                                       │
    │  analyze_full()                       │
    │  → 两个模型并行运行                    │
    └───────────────────────────────────────┘
            ↓
    ┌─────────────────────┐
    │   结果合并器         │  Gemma 3 综合输出
    │                     │  统一自然语言回答
    └─────────────────────┘
            ↓
       SSE 流 → 前端
```

### 路由决策表

| 输入类型 | 问题关键词 | 路由结果 |
|----------|-----------|---------|
| 图片 + 问题 | "在哪里"、"定位"、"显示"、"检测"、"分割" | `analyze_location()` → PaliGemma 2 |
| 图片 + 问题 | "是什么病"、"诊断"、"严重程度"、"风险"、"正常吗" | `analyze_diagnosis()` → MedGemma |
| 图片 + 问题 | "哪里有问题"、"完整分析"、"全部告诉我" | `analyze_full()` → 两个都调用 |
| 纯文字 | 任何医学问题 | `analyze_diagnosis()` → MedGemma 纯文字模式 |

### 技术栈

| 层级 | 工具 | 说明 |
|------|------|------|
| 调度器 | FunctionGemma 270M | 纯文字路由器，输出结构化函数调用 |
| 预扫描 | Gemma 3 4B | 多模态，将图片转为文字描述 |
| 分割 | PaliGemma 2 3B（微调版） | 输出 `<loc>` 和 `<seg>` tokens |
| 诊断 | MedGemma 4B（微调版） | 输出自然语言医学文本 |
| 结果合并 | Gemma 3 4B | 综合多模型输出 |
| 模型服务 | Ollama + HuggingFace Transformers | 本地推理，无需联网 |
| 后端框架 | FastAPI | API 接口 + SSE 推送 |
| 前端框架 | Next.js 14 + Tailwind CSS | 首页 + Demo 界面 |
| 部署 | Vercel（前端）+ 本地（后端） | 比赛当天后端跑在本机 |

### SSE 事件流

每个处理步骤向前端推送一个命名事件：

| 事件名 | 前端显示内容 |
|--------|------------|
| `input_received` | 图片和问题已接收 |
| `prescanning` | 正在扫描图片内容... |
| `prescan_complete` | 图片识别完成：眼底照片 |
| `routing` | 正在判断分析类型... |
| `route_decided` | 路由：完整分析（定位 + 诊断） |
| `paligemma_start` | 正在定位解剖结构... |
| `medgemma_start` | 正在分析病理状况... |
| `paligemma_complete` | 发现 2 个感兴趣区域 |
| `medgemma_complete` | 诊断分析完成 |
| `merging` | 正在整合结果... |
| `complete` | {最终 JSON 结果} |
| `error` | {错误详情} |

### 最终 JSON 返回结构

**analyze_location（定位分析）：**
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
    "summary": "检测到 2 个区域：右上象限出血点。"
  }
}
```

**analyze_diagnosis（诊断分析）：**
```json
{
  "request_id": "abc124",
  "route": "analyze_diagnosis",
  "status": "success",
  "result": {
    "type": "diagnosis",
    "diagnosis": {
      "condition": "非增殖性糖尿病视网膜病变",
      "severity": "Moderate",
      "severity_level": 3,
      "confidence": 0.84
    },
    "findings": [
      "颞上象限多处点状出血",
      "黄斑附近微血管瘤"
    ],
    "recommendation": "3-6 个月内随访眼科医生",
    "disclaimer": "仅供研究使用，不作为临床诊断依据。"
  }
}
```

**analyze_full（完整分析）：**
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
      "condition": "非增殖性糖尿病视网膜病变",
      "severity": "Moderate",
      "severity_level": 3,
      "findings": [...],
      "recommendation": "3-6 个月内随访"
    },
    "summary": "右上象限发现出血点，综合判断为中度非增殖性糖尿病视网膜病变。",
    "disclaimer": "仅供研究使用，不作为临床诊断依据。"
  }
}
```

---

## 4. 目录结构

```
optiassist/
├── agents.md                          # 代码规范（Cursor 读这个）
├── backend/
│   ├── main.py                        # FastAPI 入口 + SSE
│   ├── orchestrator.py                # 主 Pipeline 逻辑
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── prescanner.py              # 阶段2：Gemma 3 图片→文字
│   │   ├── router.py                  # 阶段3：FunctionGemma 路由
│   │   ├── segmenter.py               # 阶段4a：PaliGemma 2 + mask 解码
│   │   ├── diagnostician.py           # 阶段4b：MedGemma 推理
│   │   └── merger.py                  # 阶段4c：Gemma 3 结果综合
│   ├── models/                        # 微调模型权重（队友提供）
│   │   ├── paligemma2-finetuned/      # config.json + model.safetensors
│   │   └── medgemma-finetuned/        # config.json + model.safetensors
│   ├── utils/
│   │   ├── image_utils.py             # 图片预处理 + mask 叠加
│   │   └── token_parser.py            # 解析 <loc> 和 <seg> tokens
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── app/
│   │   ├── page.tsx                   # 首页（/）
│   │   ├── demo/
│   │   │   └── page.tsx               # Demo 页（/demo）
│   │   └── layout.tsx
│   ├── components/
│   │   ├── ImageUpload.tsx            # 图片拖拽上传 + 预览
│   │   ├── ProcessingFeed.tsx         # SSE 实时进度显示
│   │   ├── AnnotatedImage.tsx         # 标注图片 + mask 叠加展示
│   │   ├── DiagnosisCard.tsx          # 诊断结果展示
│   │   └── RouteBadge.tsx             # 显示调用了哪些模型
│   ├── package.json
│   └── tailwind.config.js
└── data/
    └── sample_images/
        ├── normal_fundus.jpg
        ├── moderate_dr.jpg            # Demo 主图
        └── glaucoma_suspect.jpg
```

---

## 5. 比赛前准备（Day 0）

---

### Step 1｜你来操作：通过 Ollama 拉取模型

打开终端运行：

```bash
ollama pull gemma3:4b
ollama pull functiongemma
```

测试是否正常：

```bash
ollama run gemma3:4b "描述一张正常眼底图像的特征"
```

输入 `/bye` 退出。模型保存在 `~/.ollama/models/`。

---

### Step 2｜你来操作：创建项目目录和 Python 环境

```bash
mkdir -p optiassist/backend/agents optiassist/backend/models/paligemma2-finetuned optiassist/backend/models/medgemma-finetuned optiassist/backend/utils optiassist/data/sample_images
cd optiassist/backend
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn python-multipart sse-starlette
pip install transformers torch pillow numpy
pip install python-dotenv requests
```

验证安装：

```bash
python -c "import fastapi, transformers, PIL; print('All good!')"
```

---

### Step 3｜你来操作：下载示例眼底图片

下载公开可用的眼底图像。推荐来源：DRIVE 数据集或 EyePACS 公开样本。至少需要：

- 一张**正常**眼底图
- 一张**中度糖尿病视网膜病变**图（Demo 主图）
- 一张**青光眼疑似**图

全部放到 `optiassist/data/sample_images/`。

---

### Step 4｜你来操作：创建 GitHub 仓库并推送骨架

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

> 注意：仓库可以是**私有的**。v0 支持私有仓库——在 v0.dev → Import Project → 授权 GitHub → 选择私有仓库。v0 会直接向私有仓库提 PR。

---

### Step 5｜你来操作：初始化 Next.js 前端骨架并推送

在给 v0 任何 prompt 之前，仓库里必须有一个可运行的 Next.js 骨架。运行：

```bash
cd optiassist
npx create-next-app@latest frontend --typescript --tailwind --app --no-src-dir --import-alias "@/*"
cd frontend
git add .
git commit -m "Add Next.js frontend skeleton"
git push origin main
```

然后在 v0.dev 连接这个仓库。之后 v0 的所有操作都以 PR 形式进入仓库。

---

### Step 6｜让 Cursor 来做：创建 `agents.md` 代码规范

在 Cursor 中打开 `optiassist/` 根目录，创建 `agents.md`。

**Cursor Prompt：**

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

### Step 7｜让 Cursor 来做：创建 `orchestrator.py`

在 Cursor 中打开 `optiassist/backend/`，创建 `orchestrator.py`。

**Cursor Prompt：**

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

### Step 8｜让 Cursor 来做：创建 `agents/prescanner.py`

**Cursor Prompt：**

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

### Step 9｜让 Cursor 来做：创建 `agents/router.py`

**Cursor Prompt：**

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

### Step 10｜让 Cursor 来做：创建 `agents/segmenter.py`

**Cursor Prompt：**

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

### Step 11｜让 Cursor 来做：创建 `agents/diagnostician.py`

**Cursor Prompt：**

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

### Step 12｜让 Cursor 来做：创建 `agents/merger.py`

**Cursor Prompt：**

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
  "location": location,
  "diagnosis": diagnosis,
  "summary": summary_string,
  "disclaimer": "For research use only. Not intended for clinical diagnosis."
}

Use httpx for async HTTP calls.

Do not do anything beyond what I described above.
All code must be written in English. Reply to me in Chinese when chatting.
```

---

### Step 13｜让 Cursor 来做：创建 `main.py`

**Cursor Prompt：**

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

### Step 14｜你来操作：安装依赖并测试后端

```bash
cd optiassist/backend
source venv/bin/activate
pip install httpx
pip freeze > requirements.txt
uvicorn main:app --port 8000 --reload
```

另开一个终端测试：

```bash
curl http://localhost:8000/health
```

应返回：`{"status":"ok","models":[...]}`

再测试纯文字请求（先不传图，因为模型可能还没准备好）：

```bash
curl -X POST http://localhost:8000/analyze \
  -F "question=什么是糖尿病视网膜病变？"
```

确认 SSE 事件正常流出。

---

### Step 15｜v0：构建首页

打开 v0.dev，连接你的 GitHub 仓库，新建对话。

**v0 Prompt：**

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

### Step 16｜你来操作：合并首页 PR

1. 在 v0 右侧面板点击 **Merge PR**
2. 本地拉取：

```bash
cd optiassist
git pull origin main
cd frontend
npm install
npm run dev
```

打开 `http://localhost:3000` 确认效果。

---

### Step 17｜v0：构建 Demo 页

**v0 Prompt：**

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

### Step 18｜你来操作：合并 Demo 页 PR 并部署 Vercel

```bash
cd optiassist
git pull origin main
cd frontend
npm install
npm run dev
```

在 `http://localhost:3000/demo` 测试完整流程。

部署到 Vercel：
1. 打开 vercel.com → Add New Project → 选择 `optiassist` 仓库
2. **重要：** Framework Preset = `Next.js`，Root Directory = `frontend`
3. 添加环境变量：`NEXT_PUBLIC_BACKEND_URL` = 你的 ngrok URL（见比赛当天步骤）
4. 点击 Deploy

如果部署后 404：Settings → Build and Deployment → 确认 Framework = Next.js，Root Directory = frontend → Redeploy。

---

## 6. 比赛当天操作（2.28）

### 9:00 AM——到场后立即执行

```bash
# 启动 Ollama（如果没有自动启动）
ollama serve

# 启动后端
cd optiassist/backend
source venv/bin/activate
uvicorn main:app --port 8000

# 确认后端正常
curl http://localhost:8000/health
```

### 队友微调完成后

将模型文件夹复制到：
```
optiassist/backend/models/paligemma2-finetuned/
optiassist/backend/models/medgemma-finetuned/
```

重启后端，用示例图片做一次端到端测试。

### 如果需要将后端暴露给 Vercel 前端

```bash
ngrok http 8000
```

复制 HTTPS URL。在 Vercel 项目 → Settings → Environment Variables → 更新 `NEXT_PUBLIC_BACKEND_URL` → Redeploy。

### 比赛当天检查清单

- [ ] `ollama list` 确认 `functiongemma` 和 `gemma3:4b` 已下载
- [ ] 微调版 PaliGemma 2 权重在 `backend/models/paligemma2-finetuned/`
- [ ] 微调版 MedGemma 权重在 `backend/models/medgemma-finetuned/`
- [ ] `curl http://localhost:8000/health` 返回 ok
- [ ] 首页深色主题正常显示
- [ ] Demo 页上传+提问流程端到端可用
- [ ] SSE 流在浏览器可见（实时步骤依次出现）
- [ ] 示例图片就位：`normal_fundus.jpg`、`moderate_dr.jpg`、`glaucoma_suspect.jpg`
- [ ] ngrok 已安装备用
- [ ] Demo 脚本已排练

---

## 7. 队友交接接口

> 在任何人开始写代码之前，先对齐这一节的所有内容。

### 队友交付什么

两个文件夹，放在 `optiassist/backend/models/` 下：

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

可选：`vae-oid.npz` 放在 `backend/utils/`，用于 PaliGemma 2 像素级 segmentation mask 解码。

### 微调开始前需要确认的问题

| 问题 | PaliGemma 2 | MedGemma |
|------|------------|---------|
| 训练时使用的 prompt 格式 | `segment {query}\n` 还是其他？ | system prompt 是什么格式？ |
| 训练时的图片分辨率 | 224px、448px 还是 896px？ | 标准输入尺寸 |
| 微调了哪些标签/类别 | 如：hemorrhage、optic disc、blood vessels、macula | DR 严重程度 0-4？还有其他疾病？ |
| 是否包含 VQ-VAE 解码器 | 需要 `vae-oid.npz` 才能出像素 mask——请确认 | 不适用 |
| 输出格式 | 原始 `<loc><seg>` tokens | JSON 字符串还是自由文本？ |
| HuggingFace 加载方式 | `PaliGemmaForConditionalGeneration` | `pipeline("image-text-to-text")` |

### 后端集成测试（队友交付模型后运行）

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

## 8. Prompt 大全

### Cursor Prompts（后端）

| 步骤 | 文件 | 用途 |
|------|------|------|
| Step 6 | `agents.md` | 所有后端 Python 文件的代码规范 |
| Step 7 | `orchestrator.py` | 主 Pipeline：输入→预扫描→路由→执行→合并 |
| Step 8 | `agents/prescanner.py` | Gemma 3 图片预扫描→文字描述 |
| Step 9 | `agents/router.py` | FunctionGemma 路由→函数调用输出 |
| Step 10 | `agents/segmenter.py` | PaliGemma 2 推理 + loc/seg token 解析 + 边框叠加 |
| Step 11 | `agents/diagnostician.py` | MedGemma 推理 + JSON 输出解析 |
| Step 12 | `agents/merger.py` | Gemma 3 结果综合→统一摘要 |
| Step 13 | `main.py` | FastAPI 接口 + SSE 推送 + CORS |

### v0 Prompts（前端）

| 步骤 | 页面 | 用途 |
|------|------|------|
| Step 15 | `app/page.tsx` | 首页：深色主题、4 张痛点卡片、使用流程、技术栈 |
| Step 17 | `app/demo/page.tsx` | Demo 页：双列布局、SSE 实时进度、分割+诊断结果展示 |

### 你手动操作（无需 Prompt）

| 步骤 | 内容 |
|------|------|
| Step 1 | `ollama pull gemma3:4b && ollama pull functiongemma` |
| Step 2 | 创建目录结构和 Python venv |
| Step 3 | 下载示例眼底图片 |
| Step 4 | 创建 GitHub 仓库并推送骨架 |
| Step 5 | 初始化 Next.js 骨架并推送（v0 连接前必须先做） |
| Step 14 | 用 curl 做后端端到端测试 |
| Step 16 | 合并 v0 首页 PR → git pull → 确认 localhost:3000 |
| Step 18 | 合并 v0 Demo 页 PR → git pull → npm install → 部署 Vercel |

---

## 9. 风险备案

| 风险 | 触发条件 | 备案方案 |
|------|---------|---------|
| 微调模型没来得及完成 | 队友在 Demo 前无法交付 | 使用原版 PaliGemma 2 + 原版 MedGemma——仍然能展示路由架构，只是没有微调带来的性能提升 |
| PaliGemma 2 seg tokens 解码失败 | `vae-oid.npz` 缺失或损坏 | 降级为仅显示 bounding box（无像素 mask）——Demo 仍然直观 |
| MedGemma 输出不是有效 JSON | 返回自由文本而非结构化输出 | 在 system prompt 中加更严格的 JSON 格式要求 + 加重试逻辑 |
| FunctionGemma 路由错误 | 调用了错误的模型 | 兜底：解析失败时默认走 `analyze_full`（两个模型都调） |
| SSE 流在浏览器中断 | ReadableStream 断连 | 改成轮询：每 2 秒 GET `/status/{job_id}` |
| 模型推理太慢 | 推理超过 60 秒 | 演示前提前跑好，展示缓存结果 + 解说 Pipeline 逻辑 |
| v0 生成的组件结构不兼容 | Next.js App Router 不匹配 | 合并 PR 后在 Cursor 里修——v0 生成骨架，Cursor 做清理 |
| 比赛当天 ngrok 不可用 | 无法暴露本地后端 | 整个 Demo 在一台机器上跑，前端直接指向 `localhost:8000` |
| 队友微调用了不同的 prompt 格式 | segmenter.py 的 prompt 和训练时不匹配 | 更新 `segmenter.py` 顶部的 prompt 常量，改成队友实际使用的格式 |

---

## 10. Demo 脚本

### 30 秒 Pitch

**开场（10 秒）：**
> "医生拍了一张眼底图。那张图受 HIPAA 和 GDPR 保护——它不能离开诊室。但医生仍然需要实时 AI 辅助。这就是为什么它必须跑在本地设备上。"

**解决方案（10 秒）：**
> "OptiAssist 完全运行在这台机器上，使用 Google DeepMind 的 Gemma 家族模型。患者图像从不接触云端服务器。医生提出临床问题——系统智能地将其路由到正确的 AI 模型，几秒钟内返回答案。"

**现场演示——按这个顺序：**

1. 打开首页——指出 4 张痛点卡片，重点强调**患者数据隐私**
2. 点击 **Try the Demo**
3. 上传 `moderate_dr.jpg`
4. 输入问题：`"这看起来像糖尿病视网膜病变吗？病变在哪里？"`
5. 点击 **Analyze**——边看 SSE 实时进度边解说：
   - "这里你可以看到 FunctionGemma 在决定调用哪些模型..."
   - "它路由到了完整分析——PaliGemma 2 和 MedGemma 正在并行运行..."
   - "PaliGemma 2 找到了病变区域并画出了分割叠加..."
   - "MedGemma 确认：中度非增殖性 DR，建议 3-6 个月内随访..."
6. 指出标注图片上的 bounding box
7. 指出诊断结果卡片：疾病名称、严重程度标签、发现列表、建议

**收尾（10 秒）：**
> "同样的架构适用于任何医学影像专科——皮肤科、放射科、病理科。换掉微调模型，Pipeline 不变。而且因为跑在本地，它可以在没有网络的医院、偏远诊所、任何医生需要 AI 而不能妥协患者隐私的地方工作。"

---

*文档版本：v1.0 | OptiAssist Team | 2026.02.28*

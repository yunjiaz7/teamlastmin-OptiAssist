# OptiAssist 系统技术深度解析

> **定位**：面向眼科临床场景的全本地化 AI 辅助分析系统，零云依赖，患者数据不离院。

---

## 目录

1. [系统总览](#1-系统总览)
2. [整体架构图](#2-整体架构图)
3. [后端技术栈](#3-后端技术栈)
4. [五阶段 AI Pipeline 详解](#4-五阶段-ai-pipeline-详解)
   - [Stage 1 — 输入接收与验证](#stage-1--输入接收与验证)
   - [Stage 2 — 图像预扫描（Prescanner）](#stage-2--图像预扫描prescanner)
   - [Stage 3 — 智能路由（Router）](#stage-3--智能路由router)
   - [Stage 4 — 并行专家推理](#stage-4--并行专家推理)
   - [Stage 5 — 结果聚合（Merger）](#stage-5--结果聚合merger)
5. [模型详解](#5-模型详解)
6. [SSE 实时流式传输机制](#6-sse-实时流式传输机制)
7. [异步并发设计](#7-异步并发设计)
8. [前端技术栈](#8-前端技术栈)
9. [前端页面架构](#9-前端页面架构)
10. [前端 → 后端通信协议](#10-前端--后端通信协议)
11. [API 接口规范](#11-api-接口规范)
12. [数据结构定义](#12-数据结构定义)
13. [关键设计决策](#13-关键设计决策)
14. [依赖清单](#14-依赖清单)

---

## 1. 系统总览

OptiAssist 是一个**专为眼科医生设计的本地化 AI 辅助诊断系统**，解决三个核心痛点：

| 问题 | OptiAssist 方案 |
|------|----------------|
| 患者隐私（HIPAA/GDPR）| 所有推理在本地完成，图像不离院 |
| 云端延迟（500ms+）| 本地设备推理，亚秒级响应 |
| 专业工具匮乏 | 专门针对视网膜图像训练的多模型 pipeline |

系统由两个独立服务组成：

- **Backend**：Python FastAPI 服务，负责接收请求、协调 AI pipeline、通过 SSE 推流进度
- **Frontend**：Next.js 应用，提供展示落地页 + 交互式 Demo 界面

---

## 2. 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js 16)                     │
│                                                              │
│  ┌──────────────┐     ┌─────────────────────────────────┐   │
│  │  Landing Page│     │         Demo Page (/demo)        │   │
│  │  /           │     │                                  │   │
│  │  - Hero      │     │  ┌─────────────┐ ┌───────────┐  │   │
│  │  - Problem   │     │  │  Input Panel│ │Analysis   │  │   │
│  │  - HowItWrks │     │  │  (image +   │ │Panel      │  │   │
│  │  - TechStack │     │  │   question) │ │(SSE feed) │  │   │
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
│         │  Stage 4: 分支执行 (按路由决策)        │             │
│         │    ├─► segmenter.py                  │             │
│         │    │     └─► PaliGemma 2 3B (HF)     │             │
│         │    ├─► diagnostician.py              │             │
│         │    │     └─► MedGemma 4B (HF)        │             │
│         │    └─► 两者并行 (analyze_full)        │             │
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

## 3. 后端技术栈

| 组件 | 技术选型 | 版本 | 用途 |
|------|---------|------|------|
| Web 框架 | FastAPI | 0.128.8 | API 路由、SSE 推流、CORS |
| ASGI 服务器 | Uvicorn | 0.39.0 | 异步 HTTP 服务器 |
| 异步 HTTP 客户端 | httpx | 0.28.1 | 调用 Ollama REST API |
| 深度学习框架 | PyTorch | 2.8.0 | Transformers 推理后端 |
| 模型库 | HuggingFace Transformers | 4.57.6 | PaliGemma 2 / MedGemma 加载 |
| 图像处理 | Pillow | 11.3.0 | 图像解码、标注绘制、编码 |
| 本地 LLM 运行时 | Ollama | localhost:11434 | 运行 Gemma3 / FunctionGemma |
| 环境变量管理 | python-dotenv | 1.2.1 | 配置隔离 |
| 数值计算 | NumPy | 2.0.2 | 张量后处理 |

---

## 4. 五阶段 AI Pipeline 详解

Pipeline 的入口在 `orchestrator.py` 的 `run_pipeline()` 函数，接收三个参数：

```python
async def run_pipeline(
    image_bytes: bytes | None,   # 原始图像字节，可为 None（纯文本模式）
    question: str,               # 医生提出的临床问题
    emit: Callable[[str, str], Awaitable[None]],  # SSE 推流回调
) -> dict
```

---

### Stage 1 — 输入接收与验证

**文件**：`orchestrator.py`（第 47–51 行）

最简单的防护：如果图像和问题都为空，直接抛出 `ValueError`，不进入后续流程。

```python
if not image_bytes and not question:
    raise ValueError("At least one of image_bytes or question must be provided.")

await emit("input_received", "Image and question received")
```

SSE 事件名：`input_received`

---

### Stage 2 — 图像预扫描（Prescanner）

**文件**：`agents/prescanner.py`

**触发条件**：仅当 `image_bytes is not None` 时执行，纯文本问答跳过此阶段。

**核心逻辑**：

1. 将原始图像字节用 Base64 编码
2. 调用 **Ollama 本地 API**（`http://localhost:11434/api/generate`）
3. 使用 **Gemma 3 4B** 模型（`gemma3:4b`），以 `stream: false` 同步获取响应
4. 返回 1-2 句自然语言描述（如 "视网膜眼底图像，可见视盘及血管分布，黄斑区未见明显异常"）

```python
payload = {
    "model": "gemma3:4b",
    "prompt": "Describe this medical retinal image in 1-2 sentences...",
    "images": [base64_image],  # Ollama 多模态输入格式
    "stream": False,
}
```

**为什么需要这一步**：生成的 `image_description` 会在下一阶段传给 Router，让路由决策时具备视觉上下文，提高路由准确率。

**容错机制**：Ollama 调用失败时返回默认值 `"Retinal fundus image"`，Pipeline 不中断。

SSE 事件名：`prescanning` → `prescan_complete`

---

### Stage 3 — 智能路由（Router）

**文件**：`agents/router.py`

**核心逻辑**：使用 **FunctionGemma 270M**（基于 Gemma 的 function calling 专用微调模型）对请求进行语义分类，选择执行路径。

**工具定义（Tool Calling Schema）**：

Router 向模型注册了三个工具，模型通过 tool calling 机制选择其中之一：

| 工具名 | 触发场景 | 关键词 |
|--------|---------|--------|
| `analyze_location` | 定位解剖结构或病变位置 | where, locate, show me, detect, segment, find |
| `analyze_diagnosis` | 医学判断、疾病分类、风险评估 | is this, diagnosis, what disease, severity, risk |
| `analyze_full` | 位置 + 诊断综合分析 | full analysis, everything, complete report |

**调用方式**：

```python
payload = {
    "model": "functiongemma",
    "messages": [{"role": "user", "content": f"{question}\n\nImage context: {image_description}"}],
    "tools": TOOLS,  # 三个工具的 JSON Schema 定义
    "stream": False,
}
```

**输出格式**：

```python
{
    "function": "analyze_diagnosis",  # 选择的路由
    "query": "..."                    # 从 tool call arguments 提取的精炼查询
}
```

**容错机制**：若模型没有返回 tool call（例如模型异常），默认回退到 `analyze_full`，确保用户总能获得响应。

SSE 事件名：`routing` → `route_decided`

---

### Stage 4 — 并行专家推理

根据 Router 的决策，Stage 4 分三条路径执行：

#### 路径 A：`analyze_location` → Segmenter（分割器）

**文件**：`agents/segmenter.py`  
**模型**：`PaliGemma 2 3B`（Google，本地加载）

**执行流程**：

```
image_bytes
    ↓
PIL.Image.open()   # 解码为 RGB PIL Image
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
返回 {detections, annotated_image_base64, summary}
```

**`<loc>` Token 解析机制**：

PaliGemma 2 使用特殊 token `<loc####>` 表示边界框，格式为四个连续 token，顺序为 `y_min, x_min, y_max, x_max`，每个值范围 `[0, 1023]`（对应归一化坐标 × 1024）。

解析代码：

```python
_LOC_PATTERN = re.compile(r"<loc(\d{4})>")

# 坐标还原到真实像素
y_min = int((y_min_raw / 1024) * img_height)
x_min = int((x_min_raw / 1024) * img_width)
```

**标注图像生成**：使用 Pillow 在原图上绘制红色边界框（`outline="red", width=2`），最终以 Base64 PNG 格式返回给前端。

**异步处理**：PaliGemma 的推理是阻塞的同步操作，通过 `asyncio.to_thread()` 放入线程池，避免阻塞 FastAPI 事件循环。

```python
raw_output = await asyncio.to_thread(_run_inference, pil_image, prompt)
```

SSE 事件名：`paligemma_start` → `paligemma_complete`

---

#### 路径 B：`analyze_diagnosis` → Diagnostician（诊断器）

**文件**：`agents/diagnostician.py`  
**模型**：`MedGemma 4B`（Google DeepMind 医疗专用模型，本地加载）

**System Prompt 设计**（结构化输出约束）：

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

**执行流程**：

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

**JSON 解析容错**：

```python
# 尝试 1：直接 json.loads
# 尝试 2：提取 { ... } 之间的子串再解析
# 最终回退：FALLBACK_RESULT（预定义的安全默认值）
```

同样通过 `asyncio.to_thread()` 异步化阻塞调用。

SSE 事件名：`medgemma_start` → `medgemma_complete`

---

#### 路径 C：`analyze_full` → 并行执行 Segmenter + Diagnostician

这是最关键的优化：两个重量级模型**同时运行**，而不是串行。

```python
location, diagnosis = await asyncio.gather(
    run_segmentation(image_bytes, route["query"]),
    run_diagnosis(image_bytes, route["query"]),
)
```

`asyncio.gather()` 并发调度两个协程，实际上两个模型各自在独立线程中运行（通过内部的 `asyncio.to_thread()`），理论总耗时 = max(T_segmentation, T_diagnosis)，而非两者之和。

SSE 事件名：`paligemma_start` + `medgemma_start` 同时发出 → 各自完成后发出对应的 `_complete`

---

### Stage 5 — 结果聚合（Merger）

**文件**：`agents/merger.py`  
**模型**：`Gemma 3 4B`（via Ollama）

接收 `location`（来自 PaliGemma 2）和/或 `diagnosis`（来自 MedGemma），生成一份面向临床医生的自然语言摘要。

**Context 构建**：

```python
def _build_context(location, diagnosis) -> str:
    # location 有值时：拼入检测到的区域数量和摘要
    # diagnosis 有值时：拼入病症名称、严重程度、发现列表
    # 返回单段文本供 Gemma 3 生成摘要
```

**Summarization Prompt**：

```
You are a medical AI assistant. Summarize these ophthalmology analysis results
in 2-3 clear sentences for a doctor.
Question asked: {question}.
Results: {context_string}
```

**输出结构**：

```python
{
    "type": "full" | "location" | "diagnosis",
    "location": { ... } | None,
    "diagnosis": { ... } | None,
    "summary": "Gemma 3 生成的叙述性摘要",
    "disclaimer": "For research use only. Not intended for clinical diagnosis."
}
```

**容错机制**：Ollama 调用失败时，`summary` 直接回退为 `context_string`（原始拼接文本），保证响应可用性。

SSE 事件名：`merging` → `complete`

---

## 5. 模型详解

| 模型 | 参数量 | 运行方式 | 调用接口 | 在 Pipeline 中的角色 |
|------|-------|---------|---------|-------------------|
| **Gemma 3 4B** | 4B | Ollama 本地服务 | REST API (`/api/generate`) | 图像预扫描描述 + 最终摘要生成 |
| **FunctionGemma** | 270M | Ollama 本地服务 | REST API (`/api/chat` + tools) | 语义路由（function calling） |
| **PaliGemma 2** | 3B | HuggingFace Transformers (本地权重) | Python API (`AutoProcessor` + `PaliGemmaForConditionalGeneration`) | 视网膜解剖结构分割 + 边界框预测 |
| **MedGemma** | 4B | HuggingFace Transformers (本地权重) | Python API (`pipeline("image-text-to-text")`) | 眼科疾病诊断 + 结构化临床报告 |

**模型文件位置**：
- `backend/models/paligemma2-finetuned/`：PaliGemma 2 微调权重
- `backend/models/medgemma-finetuned/`：MedGemma 微调权重
- Gemma 3 / FunctionGemma：由 Ollama 管理，自动缓存于 `~/.ollama/`

---

## 6. SSE 实时流式传输机制

`main.py` 的核心设计是**基于 asyncio.Queue 的 SSE 推流架构**，让前端能实时看到 Pipeline 的每一步进度。

### 工作原理

```
客户端 POST /analyze
         │
         │  FastAPI 立即返回 StreamingResponse(text/event-stream)
         │
         ▼
  asyncio.Queue  ←───────────────────────────────────┐
         │                                            │
         │  event_stream() 异步生成器                │
         │  持续从 queue 读取                         │
         │                                            │
         ▼                                            │
  yield "data: {...}\n\n"   ──→ 推给客户端            │
                                                      │
                            run_pipeline() 在后台运行  │
                            每个阶段完成后调用 emit()  │
                            emit() 将事件放入 queue ──┘
```

### emit 回调设计

```python
async def emit(event: str, message: str) -> None:
    await queue.put((event, message))
```

Pipeline 的每个阶段通过 `await emit("event_name", "message")` 推送进度，无需关心底层 SSE 实现。

### SSE 事件格式

每个 SSE chunk 的格式为：

```
data: {"event": "prescanning", "message": "Scanning image content..."}\n\n
data: {"event": "routing", "message": "Deciding analysis type..."}\n\n
...
data: {"event": "complete", "result": { ... 完整结果 ... }}\n\n
```

### 事件序列（完整流程）

```
input_received    → 输入接收确认
prescanning       → Gemma 3 开始扫描图像
prescan_complete  → 图像描述生成完毕
routing           → FunctionGemma 开始路由
route_decided     → 路由结果确定
paligemma_start   → PaliGemma 2 开始推理（analyze_location / analyze_full）
medgemma_start    → MedGemma 开始推理（analyze_diagnosis / analyze_full）
paligemma_complete→ 分割完成
medgemma_complete → 诊断完成
merging           → Gemma 3 开始生成摘要
complete          → 携带完整结果的最终事件
```

### 终止信号

Pipeline 结束时将哨兵对象 `_DONE` 放入 queue，`event_stream()` 检测到后退出循环，连接自然关闭。

```python
_DONE = object()  # 身份唯一的哨兵对象，不需要值比较

if item is _DONE:
    break
```

---

## 7. 异步并发设计

OptiAssist 在保持代码简洁的前提下，通过两个关键机制解决 CPU 密集型推理与 IO 密集型网络请求的并发问题。

### 机制 1：asyncio.to_thread()（阻塞推理异步化）

HuggingFace Transformers 的推理是同步阻塞操作，直接在协程中调用会阻塞整个 FastAPI 事件循环。解决方案：

```python
# segmenter.py 和 diagnostician.py 中均使用此模式
raw_output = await asyncio.to_thread(_run_inference, pil_image, prompt)
```

`asyncio.to_thread()` 将阻塞函数放入默认线程池（`ThreadPoolExecutor`），协程挂起等待完成，事件循环可以继续处理其他请求。

### 机制 2：asyncio.gather()（并行双模型推理）

在 `analyze_full` 路径中，分割和诊断同时启动：

```python
location, diagnosis = await asyncio.gather(
    run_segmentation(image_bytes, route["query"]),
    run_diagnosis(image_bytes, route["query"]),
)
```

两个协程内部各自调用 `asyncio.to_thread()`，因此实际上是两个线程同时跑，最大化 GPU/CPU 利用率。

### 机制 3：Pipeline 与 SSE 生成器解耦

```python
asyncio.create_task(run_and_signal())  # Pipeline 后台运行
# event_stream() 作为独立协程，从 queue 读取并推给客户端
```

两者通过 `asyncio.Queue` 通信，完全解耦，互不阻塞。

---

## 8. 前端技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **Next.js** | 16.1.6 | 全栈 React 框架，App Router |
| **React** | 19.2.4 | UI 组件化 |
| **TypeScript** | 5.7.3 | 类型安全 |
| **Tailwind CSS** | 4.2.0 | 原子化 CSS，深色主题 |
| **Radix UI** | 多版本 | 无障碍可访问的原语组件 |
| **Lucide React** | 0.564.0 | 图标库 |
| **Vercel Analytics** | 1.6.1 | 访问统计 |
| **Geist Font** | Google Fonts | 无衬线 + 等宽字体 |

---

## 9. 前端页面架构

### 路由结构

```
/          → app/page.tsx          （Landing Page，落地页）
/demo      → app/demo/page.tsx     （交互式 Demo 界面）
```

### Landing Page (`/`)

由六个组件线性拼接：

```
<Navbar>          导航栏（含跳转到 Problem Section 的平滑滚动）
<Hero>            英雄区（产品定位 + 伪终端动态展示 + CTA 按钮）
<ProblemSection>  痛点区（四张卡片：隐私/速度/云依赖/工具匮乏）
<HowItWorks>      工作流程（三步：Capture → Ask → Analyze）
<TechStack>       技术栈展示（四个模型卡片）
<SiteFooter>      页脚
```

**视觉设计亮点**：
- 深色主题（`bg-background: #0A0A0A`）
- `glass-card` 玻璃拟态效果（`backdrop-blur` + 低透明度背景）
- 绿色（`#4ADE80`）作为品牌主色调
- `glow-green` 发光效果增强科技感

### Demo Page (`/demo`)

**布局**：两列分割（`grid-cols-[1fr_1.2fr]`）

**左列（Input Panel）**：
- 拖拽上传区（`onDrop` + `onDragOver`）+ `FileReader` API 预览
- 临床问题输入框（Enter 键触发分析）
- Analyze / Reset 按钮组

**右列（Analysis Panel）**：
- Pipeline Feed：实时显示每个 SSE 事件（✅ / ⏳ / ❌ 状态图标）
- 分割结果图：显示 Base64 PNG 标注图像
- 诊断卡片：Condition + Severity Badge + Findings 列表 + Recommendation

---

## 10. 前端 → 后端通信协议

### 请求格式

```
POST http://localhost:8000/analyze
Content-Type: multipart/form-data

Fields:
  question: string (必填)
  image: File (可选，图像文件)
```

前端使用浏览器原生 `FormData` + `fetch` API：

```typescript
const formData = new FormData()
if (imageFile) formData.append("image", imageFile)
formData.append("question", question)

const response = await fetch(`${BACKEND_URL}/analyze`, {
    method: "POST",
    body: formData,
})
```

### 响应：SSE 流解析

前端使用 `response.body.getReader()` 手动读取 SSE 流（而非 `EventSource`），因为 `EventSource` 不支持 POST 请求：

```typescript
const reader = response.body.getReader()
const decoder = new TextDecoder()
let buffer = ""

while (true) {
    const { done, value } = await reader.read()
    if (done) break
    
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n")
    buffer = lines.pop() || ""  // 保留未完整的行到下次拼接
    
    for (const line of lines) {
        if (!line.startsWith("data: ")) continue
        const data = JSON.parse(line.slice(6))
        // 处理 data.event
    }
}
```

### 后端 URL 配置

```typescript
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"
```

通过环境变量 `NEXT_PUBLIC_BACKEND_URL` 配置，默认指向本地 8000 端口。

---

## 11. API 接口规范

### GET `/health`

**描述**：健康检查，返回服务状态和使用的模型列表。

**响应**：
```json
{
    "status": "ok",
    "models": ["functiongemma", "gemma3:4b", "paligemma2", "medgemma"]
}
```

---

### POST `/analyze`

**描述**：执行完整的视网膜分析 Pipeline，返回 SSE 流。

**请求**：`multipart/form-data`

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `question` | string | ✅ | 临床问题 |
| `image` | File | ❌ | 视网膜图像（JPG/PNG） |

**响应**：`Content-Type: text/event-stream`

SSE 事件类型：

| event 值 | 携带数据 | 含义 |
|---------|---------|------|
| `input_received` | message: string | 输入接收确认 |
| `prescanning` | message: string | 图像预扫描开始 |
| `prescan_complete` | message: string（含图像描述） | 预扫描完成 |
| `routing` | message: string | 路由决策开始 |
| `route_decided` | message: string（含路由名） | 路由确定 |
| `paligemma_start` | message: string | PaliGemma 2 推理开始 |
| `medgemma_start` | message: string | MedGemma 推理开始 |
| `paligemma_complete` | message: string（含检测数量） | 分割完成 |
| `medgemma_complete` | message: string | 诊断完成 |
| `merging` | message: string | 结果聚合开始 |
| `complete` | result: PipelineResult | 分析完成，携带完整结果 |
| `error` | message: string | 错误信息 |

---

## 12. 数据结构定义

### PipelineResult（最终结果）

```typescript
interface PipelineResult {
    route: "analyze_location" | "analyze_diagnosis" | "analyze_full"
    result: MergedResult
}
```

### MergedResult（聚合结果）

```typescript
interface MergedResult {
    type: "full" | "location" | "diagnosis"
    location: SegmentationResult | null
    diagnosis: DiagnosisResult | null
    summary: string      // Gemma 3 生成的叙述性摘要
    disclaimer: string   // 免责声明
}
```

### SegmentationResult（分割结果）

```typescript
interface SegmentationResult {
    detections: Detection[]
    annotated_image_base64: string  // "data:image/png;base64,..."
    summary: string
}

interface Detection {
    label: string          // 解剖结构标签
    confidence: number     // 固定为 0.9（当前版本）
    bounding_box: {
        x_min: number
        y_min: number
        x_max: number
        y_max: number
    }
    has_mask: boolean      // 暂为 false，等待 vae-oid.npz 解码支持
}
```

### DiagnosisResult（诊断结果）

```typescript
interface DiagnosisResult {
    condition: string        // 疾病名称或 "Normal"
    severity: "None" | "Mild" | "Moderate" | "Severe" | "Proliferative"
    severity_level: 0 | 1 | 2 | 3 | 4
    confidence: number       // 0.0 - 1.0
    findings: string[]       // 具体观察发现列表
    recommendation: string   // 随访建议
    disclaimer: string
}
```

---

## 13. 关键设计决策

### 为什么用 SSE 而非 WebSocket？

- SSE 是**单向**的服务器推流，完全符合本场景（后端推进度，前端只接收）
- SSE 基于 HTTP，无需握手，CORS 策略简单
- 浏览器原生支持，断线自动重连（使用 `EventSource` 时）
- WebSocket 的双向能力在此场景完全多余

### 为什么用 asyncio.Queue 而非 asyncio.Event？

Queue 支持多个生产者和消费者，且可以携带数据（事件名 + 消息），完美匹配 Pipeline 各阶段向前端推送不同 payload 的需求。Event 只能表示状态切换，无法传递数据。

### 为什么 PaliGemma 2 和 MedGemma 用 HuggingFace 加载，而 Gemma 3 / FunctionGemma 用 Ollama？

- PaliGemma 2 和 MedGemma 是**微调版本**（`-finetuned` 目录），需要直接加载自定义权重，HuggingFace Transformers 最灵活
- Gemma 3 和 FunctionGemma 使用**原始权重**，Ollama 提供更友好的本地管理（自动量化、内存管理、HTTP API）
- Ollama 的 function calling 支持是 FunctionGemma 路由工作的关键

### 模型加载时机

所有模型在**模块导入时**（module level）加载，而非每次请求时加载。这消除了每次请求的模型加载开销（Transformers 模型冷加载可能需要数十秒）。

```python
# segmenter.py — 模块级加载
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_PATH)
```

### 错误处理策略

每个 Agent 都实现两层容错：

1. **内层**：`try/except` 捕获推理错误，返回 fallback 默认值（如 prescanner 的 `FALLBACK_DESCRIPTION`）
2. **外层**：orchestrator 将 Agent 异常包装为 `RuntimeError`，向上传播，最终在 `main.py` 中转为 SSE `error` 事件

这确保系统在单点故障时能优雅降级，而不是崩溃。

---

## 14. 依赖清单

### Backend (`requirements.txt`)

```
fastapi==0.128.8         # Web 框架
uvicorn==0.39.0          # ASGI 服务器
httpx==0.28.1            # 异步 HTTP 客户端（调用 Ollama）
transformers==4.57.6     # PaliGemma 2 / MedGemma 加载
torch==2.8.0             # 深度学习框架
pillow==11.3.0           # 图像处理
python-dotenv==1.2.1     # 环境变量
pydantic==2.12.5         # 数据验证
python-multipart==0.0.20 # multipart/form-data 解析
safetensors==0.7.0       # 模型权重高效加载格式
```

### Frontend (`package.json` 核心依赖)

```
next@16.1.6              # 全栈框架
react@19.2.4             # UI 库
typescript@5.7.3         # 类型系统
tailwindcss@4.2.0        # CSS 框架
@radix-ui/*              # 可访问性原语组件
lucide-react@0.564.0     # 图标
```

### 外部运行时依赖

- **Ollama**（需本地安装）：运行 `gemma3:4b` 和 `functiongemma` 模型
  - 启动命令：`ollama serve`
  - 拉取模型：`ollama pull gemma3:4b && ollama pull functiongemma`
- **CUDA / Metal**（可选）：GPU 加速 PaliGemma 2 / MedGemma 推理

---

## 附录：启动流程

```bash
# 1. 启动 Ollama 服务
ollama serve

# 2. 启动后端
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 3. 启动前端
cd frontend
npm install
npm run dev        # http://localhost:3000
```

**数据流全链路耗时估算（analyze_full 路径）**：

```
Stage 1: 输入验证          < 1ms
Stage 2: Gemma 3 预扫描    ~2-5s  (Ollama，取决于硬件)
Stage 3: FunctionGemma 路由 ~0.5-1s
Stage 4: PaliGemma 2       ~3-8s  ┐ 并行执行
         MedGemma           ~3-8s  ┘ 总耗时 ≈ max(二者)
Stage 5: Gemma 3 摘要       ~1-3s
─────────────────────────────────────────────────────
总计（GPU）:   ~8-18s
总计（CPU）:   ~30-90s
```

---

*本文档基于代码库实际实现生成，版本时间：2026-02-28*

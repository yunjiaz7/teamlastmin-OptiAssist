"use client"

import { useState, useRef, useCallback, type DragEvent, type ChangeEvent } from "react"
import { Upload, X, CheckCircle2, Loader2, AlertCircle, RotateCcw, Send, ArrowLeft } from "lucide-react"
import Link from "next/link"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

interface DiagnosisData {
  condition?: string
  severity?: string
  findings?: string[]
  recommendation?: string
}

interface SegmentationData {
  raw_output?: string
  detection_count?: number
}

interface SummaryData {
  summary: string
  disclaimer?: string
}

interface ProcessingStep {
  event: string
  message: string
  timestamp: string
  status: "complete" | "processing" | "error"
  diagnosisData?: DiagnosisData
  segmentationData?: SegmentationData
  summaryData?: SummaryData
}

interface AnalysisResult {
  route?: string
  condition?: string
  severity?: string
  findings?: string[]
  recommendation?: string
  disclaimer?: string
}

const SEVERITY_COLORS: Record<string, string> = {
  None: "bg-[rgba(255,255,255,0.06)] text-muted-foreground",
  Mild: "bg-amber-500/10 text-amber-400",
  Moderate: "bg-orange-500/10 text-orange-400",
  Severe: "bg-red-500/10 text-red-400",
  Proliferative: "bg-red-500/15 text-red-300",
}

const ROUTE_COLORS: Record<string, string> = {
  Segmentation: "bg-sky-500/10 text-sky-400",
  Diagnosis: "bg-emerald-500/10 text-emerald-400",
  "Full Analysis": "bg-violet-500/10 text-violet-400",
}

interface RouteDecisionInfo {
  explanation: string
  agents: string[]
}

const ROUTE_DECISION_CONFIG: Record<string, RouteDecisionInfo> = {
  analyze_location: {
    explanation:
      "Structural anomaly detected in image. Routing to PaliGemma 2 segmentation expert to locate regions of interest.",
    agents: ["PaliGemma 2"],
  },
  analyze_diagnosis: {
    explanation:
      "Clinical diagnosis query detected. Routing to MedGemma diagnostic expert for disease classification and severity assessment.",
    agents: ["MedGemma 4B"],
  },
  analyze_full: {
    explanation:
      "Complex query requiring full analysis. Spawning both segmentation and diagnostic agents in parallel.",
    agents: ["PaliGemma 2", "MedGemma 4B"],
  },
}

function AgentDecisionCard({ message }: { message: string }) {
  const routeKey = message.split(": ")[1]?.trim() ?? ""
  const info = ROUTE_DECISION_CONFIG[routeKey]

  return (
    <div
      className="agent-fade-in my-1 rounded-xl border-l-2 border-[#4ADE80] px-4 py-3 flex flex-col gap-2"
      style={{ background: "rgba(74,222,128,0.05)", border: "1px solid rgba(74,222,128,0.15)", borderLeft: "2px solid #4ADE80" }}
    >
      <span className="text-[11px] font-semibold uppercase tracking-widest text-[#4ADE80]">
        🤖 Agent Decision
      </span>
      <p className="text-[13px] leading-snug text-foreground/85">
        {info?.explanation ?? `Routing to: ${routeKey}`}
      </p>
      {info && info.agents.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-0.5">
          {info.agents.map((agent) => (
            <span
              key={agent}
              className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 font-mono text-[11px] font-medium text-[#4ADE80]"
              style={{ background: "rgba(74,222,128,0.1)", border: "1px solid rgba(74,222,128,0.2)" }}
            >
              → {agent}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function MedGemmaDiagnosisCard({ data }: { data: DiagnosisData }) {
  return (
    <div
      className="agent-fade-in my-1 rounded-xl px-4 py-3 flex flex-col gap-2"
      style={{
        background: "rgba(96,165,250,0.05)",
        border: "1px solid rgba(96,165,250,0.18)",
        borderLeft: "2px solid #60A5FA",
      }}
    >
      <span className="text-[11px] font-semibold uppercase tracking-widest text-[#60A5FA]">
        🔬 MedGemma Diagnosis
      </span>
      {data.condition && (
        <p className="text-[14px] font-bold text-foreground leading-snug">
          {data.condition}
        </p>
      )}
      {data.severity && (
        <span
          className={`inline-block w-fit px-2 py-0.5 rounded-md font-mono text-[11px] font-medium ${
            SEVERITY_COLORS[data.severity] ?? "bg-[rgba(255,255,255,0.06)] text-muted-foreground"
          }`}
        >
          {data.severity}
        </span>
      )}
      {data.findings && data.findings.length > 0 && (
        <ul className="flex flex-col gap-1 pl-3 mt-0.5">
          {data.findings.map((f, i) => (
            <li key={i} className="list-disc text-[12px] text-foreground/70 leading-relaxed">
              {f}
            </li>
          ))}
        </ul>
      )}
      {data.recommendation && (
        <p className="italic text-[12px] text-foreground/50 leading-relaxed">
          {data.recommendation}
        </p>
      )}
    </div>
  )
}

function PaliGemmaSegmentationCard({ data }: { data: SegmentationData }) {
  return (
    <div
      className="agent-fade-in my-1 rounded-xl px-4 py-3 flex flex-col gap-2"
      style={{
        background: "rgba(168,85,247,0.05)",
        border: "1px solid rgba(168,85,247,0.18)",
        borderLeft: "2px solid #A855F7",
      }}
    >
      <span className="text-[11px] font-semibold uppercase tracking-widest text-[#A855F7]">
        📍 PaliGemma Detection
      </span>
      <p className="text-[13px] text-foreground/80">
        {data.detection_count ?? 0} region{(data.detection_count ?? 0) !== 1 ? "s" : ""} of interest detected
      </p>
      {data.raw_output && (
        <p className="font-mono text-[11px] text-foreground/60 break-all leading-relaxed">
          {data.raw_output}
        </p>
      )}
    </div>
  )
}

function FinalSummaryCard({ data }: { data: SummaryData }) {
  return (
    <div
      className="agent-fade-in my-1 rounded-xl px-4 py-4 flex flex-col gap-3"
      style={{
        background: "rgba(74,222,128,0.04)",
        border: "1px solid rgba(74,222,128,0.18)",
        borderLeft: "2px solid #4ADE80",
      }}
    >
      <span className="text-[11px] font-semibold uppercase tracking-widest text-[#4ADE80]">
        📋 Final Summary
      </span>
      <p className="text-[15px] font-medium text-foreground/90 leading-relaxed">
        {data.summary}
      </p>
      {data.disclaimer && (
        <p className="border-t border-border pt-2 text-[11px] text-faint leading-relaxed">
          ⚠️ {data.disclaimer}
        </p>
      )}
    </div>
  )
}

export default function DemoPage() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [question, setQuestion] = useState("")
  const [steps, setSteps] = useState<ProcessingStep[]>([])
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((file: File) => {
    setImageFile(file)
    const reader = new FileReader()
    reader.onload = (e) => setImagePreview(e.target?.result as string)
    reader.readAsDataURL(file)
  }, [])

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith("image/")) handleFile(file)
  }, [handleFile])

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => setIsDragging(false), [])

  const handleFileInput = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }, [handleFile])

  const reset = useCallback(() => {
    setImageFile(null)
    setImagePreview(null)
    setQuestion("")
    setSteps([])
    setResult(null)
    setIsLoading(false)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }, [])

  const analyze = useCallback(async () => {
    if (!question.trim()) return
    setIsLoading(true)
    setSteps([])
    setResult(null)

    const formData = new FormData()
    if (imageFile) formData.append("image", imageFile)
    formData.append("question", question)

    try {
      const response = await fetch(`${BACKEND_URL}/analyze`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok || !response.body) {
        setSteps((prev) => [
          ...prev,
          {
            event: "error",
            message: `Server responded with ${response.status}`,
            timestamp: new Date().toLocaleTimeString(),
            status: "error",
          },
        ])
        setIsLoading(false)
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          const jsonStr = line.slice(6).trim()
          if (!jsonStr) continue

          try {
            const data = JSON.parse(jsonStr)

            if (data.event === "complete") {
              // Extract summary from the merged result
              const summaryData: SummaryData | undefined =
                data.result?.result?.summary
                  ? {
                      summary: data.result.result.summary,
                      disclaimer: data.result.result.disclaimer,
                    }
                  : undefined

              setSteps((prev) =>
                prev.map((s) =>
                  s.status === "processing" ? { ...s, status: "complete" } : s
                )
              )
              setSteps((prev) => [
                ...prev,
                {
                  event: "complete",
                  message: "Analysis complete",
                  timestamp: new Date().toLocaleTimeString(),
                  status: "complete",
                  summaryData,
                },
              ])
              setResult(data.result ?? data)
            } else if (data.event === "error") {
              setSteps((prev) => [
                ...prev,
                {
                  event: "error",
                  message: data.message ?? "Unknown error",
                  timestamp: new Date().toLocaleTimeString(),
                  status: "error",
                },
              ])
            } else {
              // Try to parse structured data from JSON-encoded messages
              let displayMessage: string = data.message ?? data.event ?? ""
              let diagnosisData: DiagnosisData | undefined
              let segmentationData: SegmentationData | undefined

              if (data.event === "medgemma_complete" && data.message) {
                try {
                  const parsed = JSON.parse(data.message)
                  displayMessage = parsed.text ?? displayMessage
                  diagnosisData = parsed.diagnosis
                } catch { /* leave as-is */ }
              } else if (data.event === "paligemma_complete" && data.message) {
                try {
                  const parsed = JSON.parse(data.message)
                  displayMessage = parsed.text ?? displayMessage
                  segmentationData = parsed.segmentation
                } catch { /* leave as-is */ }
              }

              setSteps((prev) => {
                const updated = prev.map((s) =>
                  s.status === "processing" ? { ...s, status: "complete" as const } : s
                )
                return [
                  ...updated,
                  {
                    event: data.event ?? data.step ?? "step",
                    message: displayMessage,
                    timestamp: new Date().toLocaleTimeString(),
                    status: "processing",
                    diagnosisData,
                    segmentationData,
                  },
                ]
              })
            }
          } catch {
            // skip malformed JSON
          }
        }
      }
    } catch (err) {
      setSteps((prev) => [
        ...prev,
        {
          event: "error",
          message: err instanceof Error ? err.message : "Connection failed",
          timestamp: new Date().toLocaleTimeString(),
          status: "error",
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }, [imageFile, question])

  const canAnalyze = question.trim().length > 0 && !isLoading

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      {/* Top Nav */}
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-background/60 backdrop-blur-xl px-6">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors p-1.5 -ml-1.5 rounded-lg hover:bg-secondary"
          >
            <ArrowLeft className="size-4" />
          </Link>
          <div className="h-4 w-px bg-border" />
          <div className="flex items-center gap-2">
            <div className="size-5 rounded-md bg-primary/10 flex items-center justify-center">
              <div className="size-1.5 rounded-full bg-primary" />
            </div>
            <span className="text-sm font-medium text-foreground">
              OptiAssist
            </span>
            <span className="text-muted-foreground text-sm">/</span>
            <span className="text-sm text-muted-foreground">Demo</span>
          </div>
        </div>
        <div className="flex items-center gap-2.5">
          <span
            className={`inline-block size-2 rounded-full ${
              isLoading ? "animate-pulse bg-amber-400" : "bg-primary"
            }`}
          />
          <span className="text-[13px] text-muted-foreground">
            {isLoading ? "Processing..." : "Ready"}
          </span>
        </div>
      </header>

      {/* Main Two-Column Layout */}
      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[1fr_1.2fr]">
        {/* Left Panel */}
        <div className="flex flex-col gap-5 border-b border-border p-6 lg:border-r lg:border-b-0 overflow-y-auto">
          <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-widest">
            Input
          </span>

          {/* Drop Zone */}
          <div
            role="button"
            tabIndex={0}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click()
            }}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`relative flex min-h-[240px] cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed transition-all duration-200 ${
              isDragging
                ? "border-primary bg-primary/5"
                : "border-border bg-[rgba(255,255,255,0.02)] hover:border-muted-foreground hover:bg-[rgba(255,255,255,0.04)]"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileInput}
            />
            {imagePreview ? (
              <div className="relative h-full w-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imagePreview}
                  alt="Uploaded retinal image"
                  className="h-full w-full object-contain p-3"
                />
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setImageFile(null)
                    setImagePreview(null)
                    if (fileInputRef.current) fileInputRef.current.value = ""
                  }}
                  className="absolute right-3 top-3 bg-background/80 backdrop-blur-sm p-1.5 rounded-lg text-muted-foreground transition-colors hover:text-foreground hover:bg-secondary"
                  aria-label="Remove image"
                >
                  <X className="size-4" />
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <div className="size-12 rounded-xl bg-[rgba(255,255,255,0.04)] flex items-center justify-center">
                  <Upload className="size-5" />
                </div>
                <div className="flex flex-col items-center gap-1">
                  <span className="text-sm font-medium text-foreground/80">Drop retinal image here</span>
                  <span className="text-xs text-muted-foreground">or click to browse</span>
                </div>
              </div>
            )}
          </div>

          {/* Question Input */}
          <div className="relative">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && canAnalyze) analyze()
              }}
              placeholder="Ask a clinical question..."
              className="w-full border border-border bg-[rgba(255,255,255,0.02)] rounded-xl px-4 py-3 pr-10 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20 transition-all"
            />
            <Send className="pointer-events-none absolute right-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          </div>

          {/* Buttons */}
          <div className="flex gap-3">
            <button
              disabled={!canAnalyze}
              onClick={analyze}
              className="flex-1 bg-primary py-3 text-sm font-medium text-primary-foreground rounded-xl transition-all duration-200 hover:brightness-110 hover:shadow-[0_0_20px_-4px_rgba(74,222,128,0.4)] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:brightness-100 disabled:hover:shadow-none"
            >
              {isLoading ? "Analyzing..." : "Analyze"}
            </button>
            <button
              onClick={reset}
              className="flex items-center justify-center gap-2 border border-border bg-transparent px-5 py-3 text-sm text-muted-foreground rounded-xl transition-all duration-200 hover:text-foreground hover:bg-secondary"
            >
              <RotateCcw className="size-3.5" />
              Reset
            </button>
          </div>

          <p className="text-[11px] text-muted-foreground">
            Supported formats: JPG, PNG retinal fundus images
          </p>
        </div>

        {/* Right Panel */}
        <div className="flex flex-col gap-5 overflow-y-auto p-6">
          <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-widest">
            Analysis
          </span>

          {steps.length === 0 && !result ? (
            <div className="flex flex-1 items-center justify-center">
              <div className="flex flex-col items-center gap-3 text-center">
                <div className="size-12 rounded-xl bg-[rgba(255,255,255,0.04)] flex items-center justify-center">
                  <Send className="size-5 text-muted-foreground" />
                </div>
                <p className="text-sm text-muted-foreground">
                  Upload an image and ask a question to start analysis
                </p>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              {/* Processing Feed */}
              {steps.length > 0 && (
                <div className="glass-card rounded-2xl p-5 flex flex-col gap-1">
                  <span className="mb-3 text-[11px] font-medium text-muted-foreground uppercase tracking-widest">
                    Pipeline
                  </span>
                  {steps.map((step, i) =>
                    step.event === "route_decided" ? (
                      <div key={i}>
                        <AgentDecisionCard message={step.message} />
                      </div>
                    ) : step.event === "medgemma_complete" && step.diagnosisData ? (
                      <div key={i}>
                        <div className="flex items-center gap-3 py-1.5">
                          <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-primary" />
                          <span className="text-[13px] leading-snug text-foreground/80">
                            {step.message}
                          </span>
                          <span className="ml-auto shrink-0 font-mono text-[10px] text-faint">
                            {step.timestamp}
                          </span>
                        </div>
                        <MedGemmaDiagnosisCard data={step.diagnosisData} />
                      </div>
                    ) : step.event === "paligemma_complete" && step.segmentationData ? (
                      <div key={i}>
                        <div className="flex items-center gap-3 py-1.5">
                          <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-primary" />
                          <span className="text-[13px] leading-snug text-foreground/80">
                            {step.message}
                          </span>
                          <span className="ml-auto shrink-0 font-mono text-[10px] text-faint">
                            {step.timestamp}
                          </span>
                        </div>
                        <PaliGemmaSegmentationCard data={step.segmentationData} />
                      </div>
                    ) : step.event === "complete" && step.summaryData ? (
                      <div key={i}>
                        <div className="flex items-center gap-3 py-1.5">
                          <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-primary" />
                          <span className="text-[13px] leading-snug text-foreground/80">
                            {step.message}
                          </span>
                          <span className="ml-auto shrink-0 font-mono text-[10px] text-faint">
                            {step.timestamp}
                          </span>
                        </div>
                        <FinalSummaryCard data={step.summaryData} />
                      </div>
                    ) : (
                      <div key={i} className="flex items-start gap-3 py-1.5">
                        {step.status === "complete" ? (
                          <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-primary" />
                        ) : step.status === "error" ? (
                          <AlertCircle className="mt-0.5 size-4 shrink-0 text-destructive" />
                        ) : (
                          <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin text-amber-400" />
                        )}
                        <span
                          className={`text-[13px] leading-snug ${
                            step.status === "error"
                              ? "text-destructive"
                              : "text-foreground/80"
                          }`}
                        >
                          {step.message}
                        </span>
                        <span className="ml-auto shrink-0 font-mono text-[10px] text-faint">
                          {step.timestamp}
                        </span>
                      </div>
                    )
                  )}
                </div>
              )}

              {/* Diagnosis Card */}
              {result && (result.condition || result.findings) && (
                <div className="glass-card rounded-2xl p-5 flex flex-col gap-4 relative">
                  {/* Route Badge */}
                  {result.route && (
                    <span
                      className={`absolute right-5 top-5 px-2.5 py-1 rounded-lg font-mono text-[10px] font-medium uppercase ${
                        ROUTE_COLORS[result.route] ?? "bg-[rgba(255,255,255,0.06)] text-muted-foreground"
                      }`}
                    >
                      {result.route}
                    </span>
                  )}

                  {/* Condition */}
                  {result.condition && (
                    <h2 className="text-xl font-bold text-foreground pr-24">
                      {result.condition}
                    </h2>
                  )}

                  {/* Severity */}
                  {result.severity && (
                    <span
                      className={`inline-block w-fit px-2.5 py-1 rounded-lg font-mono text-xs font-medium ${
                        SEVERITY_COLORS[result.severity] ?? "bg-[rgba(255,255,255,0.06)] text-muted-foreground"
                      }`}
                    >
                      {result.severity}
                    </span>
                  )}

                  {/* Findings */}
                  {result.findings && result.findings.length > 0 && (
                    <ul className="flex flex-col gap-2 pl-4">
                      {result.findings.map((f, i) => (
                        <li
                          key={i}
                          className="list-disc text-sm text-foreground/70 leading-relaxed"
                        >
                          {f}
                        </li>
                      ))}
                    </ul>
                  )}

                  {/* Recommendation */}
                  {result.recommendation && (
                    <p className="italic text-sm text-subtle leading-relaxed">
                      {result.recommendation}
                    </p>
                  )}

                  {/* Disclaimer */}
                  {result.disclaimer && (
                    <p className="mt-1 border-t border-border pt-3 text-[11px] text-faint leading-relaxed">
                      {result.disclaimer}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

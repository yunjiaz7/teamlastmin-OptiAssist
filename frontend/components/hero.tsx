"use client"

import Link from "next/link"
import { ArrowRight, Play } from "lucide-react"

function TerminalWindow() {
  const lines = [
    { tag: "router", color: "text-amber-400", text: "Loading FunctionGemma 270M..." },
    { tag: "router", color: "text-primary", text: "Request classified: diagnosis" },
    { tag: "segmenter", color: "text-sky-400", text: "PaliGemma 2 analyzing retinal scan..." },
    { tag: "diagnosis", color: "text-amber-400", text: "MedGemma 4B running analysis..." },
    { tag: "synthesis", color: "text-subtle", text: "Gemma 3 4B generating report" },
    { tag: "pipeline", color: "text-subtle", text: "Complete \u2014 2 findings, 0 warnings" },
  ]

  return (
    <div className="glass-card rounded-2xl overflow-hidden glow-green">
      {/* Title bar */}
      <div className="flex items-center gap-2 px-5 py-3.5 border-b border-border">
        <span className="size-3 rounded-full bg-[#FF5F57]" />
        <span className="size-3 rounded-full bg-[#FEBC2E]" />
        <span className="size-3 rounded-full bg-[#28C840]" />
        <span className="ml-3 font-mono text-xs text-muted-foreground">
          optiassist pipeline
        </span>
      </div>
      {/* Terminal body */}
      <div className="px-6 py-5 font-mono text-[13px] leading-[2] text-muted-foreground">
        {lines.map((line, i) => (
          <p key={i}>
            <span className="text-faint">{'> '}</span>
            <span className={line.color}>{'['}{line.tag}{']'}</span>
            {' '}{line.text}
          </p>
        ))}
        <p className="text-primary mt-1">
          <span className="text-faint">{'> '}</span>
          {'analysis complete \u2713'}
        </p>
      </div>
    </div>
  )
}

export function Hero({ onScrollToProblem }: { onScrollToProblem: () => void }) {
  return (
    <section className="relative min-h-screen flex items-center px-6 pt-20 overflow-hidden">
      {/* Background radial glow */}
      <div className="absolute top-[-200px] left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-primary/5 rounded-full blur-[120px] pointer-events-none" />

      <div className="grid grid-cols-1 gap-16 lg:grid-cols-2 lg:gap-20 items-center w-full max-w-6xl mx-auto py-24 relative z-10">
        {/* Left */}
        <div className="flex flex-col gap-8">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass-card w-fit text-[13px] text-muted-foreground">
            <span className="size-2 rounded-full bg-primary animate-pulse" />
            On-device AI for clinical use
          </div>
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl xl:text-[68px] text-balance leading-[1.08]">
            <span className="text-foreground">AI Assistant for{" "}</span>
            <span className="text-gradient-green">Ophthalmologists.</span>
          </h1>
          <p className="max-w-lg text-base text-subtle leading-relaxed sm:text-[17px]">
            Real-time retinal analysis powered by on-device models. Patient data never
            leaves the clinic. Zero cloud dependency.
          </p>
          <div className="flex flex-col gap-1 text-xs text-muted-foreground">
            <p>Built for Google DeepMind X InstaLILY AI Hackathon 2026.</p>
            <p>Built this system from 0 to 1 in 8 hours - won 2nd Prize.</p>
          </div>
          <div className="flex items-center gap-3 pt-2">
            <button
              onClick={onScrollToProblem}
              className="flex items-center gap-2 border border-border text-subtle px-5 py-3 text-[13px] font-medium rounded-xl hover:bg-secondary hover:text-foreground transition-all duration-200"
            >
              <Play className="size-3.5" />
              Watch Demo
            </button>
            <Link
              href="/demo"
              className="flex items-center gap-2 bg-primary text-primary-foreground px-5 py-3 text-[13px] font-medium rounded-xl hover:brightness-110 transition-all duration-200 hover:shadow-[0_0_20px_-4px_rgba(74,222,128,0.4)]"
            >
              Try the Demo
              <ArrowRight className="size-3.5" />
            </Link>
          </div>
        </div>
        {/* Right */}
        <div className="hidden lg:block">
          <TerminalWindow />
        </div>
      </div>
    </section>
  )
}

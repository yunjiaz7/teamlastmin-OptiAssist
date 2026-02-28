import { Lock, Zap, Globe, Brain } from "lucide-react"
import type { ReactNode } from "react"

const cards: { icon: ReactNode; title: string; body: string }[] = [
  {
    icon: <Lock className="size-5" />,
    title: "Patient Data Privacy",
    body: "Patient retinal images are protected under HIPAA and GDPR. Uploading to cloud AI creates serious compliance risk.",
  },
  {
    icon: <Zap className="size-5" />,
    title: "Real-time Speed",
    body: "Cloud round-trips add 500ms+ of delay. On-device inference responds in under 50ms \u2014 fast enough for live consultation.",
  },
  {
    icon: <Globe className="size-5" />,
    title: "No Cloud Upload Required",
    body: "Existing tools send scans to external servers. OptiAssist runs entirely on-device, built for regulated clinical environments.",
  },
  {
    icon: <Brain className="size-5" />,
    title: "Smarter Clinical Tools",
    body: "Manual retinal review is slow and error-prone. AI detects patterns and anomalies instantly, augmenting clinical decision-making.",
  },
]

export function ProblemSection() {
  return (
    <section id="problem" className="px-6 py-32 max-w-6xl mx-auto">
      <div className="flex flex-col gap-4 mb-16 max-w-2xl">
        <p className="text-[13px] font-medium text-primary tracking-wide uppercase">
          The Problem
        </p>
        <h2 className="text-3xl font-bold text-foreground sm:text-4xl lg:text-[42px] tracking-tight leading-[1.15]">
          Why We Built This
        </h2>
        <p className="text-subtle text-base leading-relaxed sm:text-[17px]">
          Ophthalmologists face unique challenges that existing AI tools cannot solve.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {cards.map((card) => (
          <div
            key={card.title}
            className="glass-card rounded-2xl p-7 flex flex-col gap-4 group hover:bg-[rgba(255,255,255,0.05)] transition-all duration-300"
          >
            <div className="flex items-center gap-3">
              <div className="size-9 rounded-xl bg-primary/10 flex items-center justify-center text-primary">
                {card.icon}
              </div>
              <h3 className="text-[15px] font-semibold text-card-foreground">
                {card.title}
              </h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {card.body}
            </p>
          </div>
        ))}
      </div>
    </section>
  )
}

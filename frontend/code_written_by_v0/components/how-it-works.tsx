import { Camera, MessageSquare, Activity } from "lucide-react"
import type { ReactNode } from "react"

const steps: { number: string; icon: ReactNode; label: string; description: string }[] = [
  {
    number: "01",
    icon: <Camera className="size-5" />,
    label: "Capture",
    description: "Upload or capture a retinal fundus image directly from your equipment.",
  },
  {
    number: "02",
    icon: <MessageSquare className="size-5" />,
    label: "Ask",
    description: "Type your clinical question in natural language. The AI understands ophthalmological context.",
  },
  {
    number: "03",
    icon: <Activity className="size-5" />,
    label: "Analyze",
    description: "Get instant on-device analysis with segmentation overlays and diagnostic insights.",
  },
]

export function HowItWorks() {
  return (
    <section className="px-6 py-32 max-w-6xl mx-auto">
      <div className="flex flex-col gap-4 mb-16 max-w-2xl">
        <p className="text-[13px] font-medium text-primary tracking-wide uppercase">
          How It Works
        </p>
        <h2 className="text-3xl font-bold text-foreground sm:text-4xl lg:text-[42px] tracking-tight leading-[1.15]">
          Three Steps. <span className="text-gradient-green">Zero Cloud.</span>
        </h2>
        <p className="text-subtle text-base leading-relaxed sm:text-[17px]">
          From image to insight in seconds, entirely on your device.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {steps.map((step) => (
          <div
            key={step.number}
            className="glass-card rounded-2xl p-7 flex flex-col gap-5 group hover:bg-[rgba(255,255,255,0.05)] transition-all duration-300"
          >
            <div className="flex items-center justify-between">
              <div className="size-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary">
                {step.icon}
              </div>
              <span className="font-mono text-3xl font-bold text-faint group-hover:text-muted-foreground transition-colors">
                {step.number}
              </span>
            </div>
            <h3 className="text-lg font-semibold text-foreground">
              {step.label}
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {step.description}
            </p>
          </div>
        ))}
      </div>
    </section>
  )
}

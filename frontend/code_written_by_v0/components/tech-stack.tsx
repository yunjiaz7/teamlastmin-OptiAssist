const models = [
  { name: "FunctionGemma", size: "270M", role: "Intelligent request routing" },
  { name: "PaliGemma 2", size: "3B", role: "Retinal structure segmentation" },
  { name: "MedGemma", size: "4B", role: "Ophthalmological diagnosis" },
  { name: "Gemma 3", size: "4B", role: "Image understanding & synthesis" },
]

export function TechStack() {
  return (
    <section className="px-6 py-32 max-w-6xl mx-auto">
      <div className="flex flex-col gap-4 mb-16 max-w-2xl">
        <p className="text-[13px] font-medium text-primary tracking-wide uppercase">
          Architecture
        </p>
        <h2 className="text-3xl font-bold text-foreground sm:text-4xl lg:text-[42px] tracking-tight leading-[1.15]">
          Powered By
        </h2>
        <p className="text-subtle text-base leading-relaxed sm:text-[17px]">
          A pipeline of specialized on-device models, each optimized for a specific clinical task.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {models.map((model) => (
          <div
            key={model.name}
            className="glass-card rounded-2xl p-6 flex flex-col gap-3 group hover:bg-[rgba(255,255,255,0.05)] transition-all duration-300"
          >
            <div className="flex items-baseline gap-2">
              <span className="text-[15px] font-semibold text-foreground">
                {model.name}
              </span>
              <span className="font-mono text-xs text-primary font-medium">
                {model.size}
              </span>
            </div>
            <span className="text-sm text-muted-foreground leading-relaxed">
              {model.role}
            </span>
          </div>
        ))}
      </div>
    </section>
  )
}

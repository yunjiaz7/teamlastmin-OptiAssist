export function SiteFooter() {
  return (
    <footer className="border-t border-border px-6 py-16 max-w-6xl mx-auto">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-col gap-1.5">
          <p className="text-sm font-medium text-foreground">
            OptiAssist
          </p>
          <p className="text-xs text-muted-foreground">
            On-Device AI for Ophthalmology
          </p>
        </div>
        <p className="text-[11px] text-faint">
          For research use only. Not intended for clinical diagnosis.
        </p>
      </div>
    </footer>
  )
}

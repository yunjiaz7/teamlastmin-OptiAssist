"use client"

import Link from "next/link"
import { Github } from "lucide-react"

export function Navbar({ onScrollToProblem }: { onScrollToProblem: () => void }) {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border bg-background/60 backdrop-blur-xl">
      <div className="flex items-center justify-between px-6 py-4 max-w-6xl mx-auto">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="size-7 rounded-lg bg-primary/10 flex items-center justify-center">
            <div className="size-2.5 rounded-full bg-primary" />
          </div>
          <span className="font-semibold text-foreground tracking-tight text-[15px]">
            OptiAssist
          </span>
        </Link>
        <div className="flex items-center gap-2">
          <button
            onClick={onScrollToProblem}
            className="hidden sm:block text-[13px] text-muted-foreground hover:text-foreground transition-colors px-3 py-1.5 rounded-lg hover:bg-secondary"
          >
            Why We Built This
          </button>
          <a
            href="https://github.com/yunjiaz7/teamlastmin-OptiAssist/tree/main"
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground transition-colors p-2 rounded-lg hover:bg-secondary"
            aria-label="GitHub"
          >
            <Github className="size-[18px]" />
          </a>
          <Link
            href="/demo"
            className="bg-primary text-primary-foreground px-4 py-2 text-[13px] font-medium rounded-xl hover:brightness-110 transition-all duration-200 hover:shadow-[0_0_20px_-4px_rgba(74,222,128,0.4)]"
          >
            Try the Demo
          </Link>
        </div>
      </div>
    </nav>
  )
}

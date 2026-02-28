"use client"

import { useCallback, useRef } from "react"
import { Navbar } from "@/components/navbar"
import { Hero } from "@/components/hero"
import { ProblemSection } from "@/components/problem-section"
import { HowItWorks } from "@/components/how-it-works"
import { TechStack } from "@/components/tech-stack"
import { SiteFooter } from "@/components/site-footer"

export default function Home() {
  const problemRef = useRef<HTMLDivElement>(null)

  const scrollToProblem = useCallback(() => {
    problemRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [])

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar onScrollToProblem={scrollToProblem} />
      <main>
        <Hero onScrollToProblem={scrollToProblem} />
        <div ref={problemRef}>
          <ProblemSection />
        </div>
        <HowItWorks />
        <TechStack />
      </main>
      <SiteFooter />
    </div>
  )
}

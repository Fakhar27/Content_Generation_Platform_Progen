import type { ReactNode } from "react"

interface DashboardHeaderProps {
  heading: string
  subheading?: string
  children?: ReactNode
}

export function DashboardHeader({ heading, subheading, children }: DashboardHeaderProps) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between mb-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-neon-pink">{heading}</h1>
        {subheading && <p className="text-zinc-400 mt-1">{subheading}</p>}
      </div>
      {children}
    </div>
  )
}

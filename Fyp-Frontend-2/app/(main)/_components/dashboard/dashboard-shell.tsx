import type { ReactNode } from "react"

interface DashboardShellProps {
  children: ReactNode
}

export function DashboardShell({ children }: DashboardShellProps) {
  return (
    <div className="flex-1 w-full overflow-hidden">
      <div className="noise-overlay"></div>
      <div className="h-full px-4 py-6 lg:px-8 bg-cyberpunk-grid">
        <div className="max-w-7xl mx-auto space-y-6">{children}</div>
      </div>
    </div>
  )
}

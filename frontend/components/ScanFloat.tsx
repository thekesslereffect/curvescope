"use client"

import { useEffect, useState } from "react"
import { usePathname } from "next/navigation"
import Link from "next/link"
import { getScanStatus, type ScanStatus } from "@/lib/api"

const PHASE_SHORT: Record<string, string> = {
  Resolving: "Resolve",
  "Downloading light curve": "Download LC",
  Cleaning: "Clean",
  Autoencoder: "Autoencoder",
  BLS: "BLS",
  Wavelet: "Wavelet",
  Centroid: "Centroid",
  Classifying: "Classify",
  Technosignature: "Techno",
}

function shortPhase(phase: string): string {
  if (!phase) return ""
  for (const [match, label] of Object.entries(PHASE_SHORT)) {
    if (phase.includes(match)) return label
  }
  return phase.slice(0, 20)
}

export function ScanFloat() {
  const pathname = usePathname()
  const [status, setStatus] = useState<ScanStatus | null>(null)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    let active = true
    const poll = () => {
      getScanStatus()
        .then((s) => active && setStatus(s))
        .catch(() => {})
    }
    poll()
    const id = setInterval(poll, 2000)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  if (!status?.running) return null
  if (pathname === "/scan") return null

  const pct =
    status.total > 0
      ? Math.round((100 * status.completed) / status.total)
      : 0

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={() => setCollapsed(false)}
        className="fixed bottom-4 right-4 z-50 flex items-center gap-2 rounded-2xl bg-foreground/90 px-3.5 py-2.5 text-xs text-background shadow-lg backdrop-blur hover:bg-foreground transition-colors"
      >
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-background opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-background" />
        </span>
        Scanning {pct}%
      </button>
    )
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 w-72 rounded-2xl border border-border bg-card/95 shadow-xl backdrop-blur text-xs">
      <div className="flex items-center justify-between px-3.5 py-2.5 border-b border-border/60">
        <span className="flex items-center gap-1.5 text-blue-400">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-400" />
          </span>
          Sector {status.sector} scan
        </span>
        <div className="flex items-center gap-1">
          <Link
            href="/scan"
            className="text-muted-foreground hover:text-foreground px-1 transition-colors"
            title="Open scan page"
          >
            ↗
          </Link>
          <button
            type="button"
            onClick={() => setCollapsed(true)}
            className="text-muted-foreground hover:text-foreground px-1 transition-colors"
            title="Minimize"
          >
            _
          </button>
        </div>
      </div>

      <div className="px-3.5 py-2.5 space-y-1.5">
        <div className="flex justify-between text-dim">
          <span>{status.completed} / {status.total} done</span>
          <span>{pct}%</span>
        </div>

        <div className="h-1.5 bg-accent rounded-full overflow-hidden">
          <div
            className="h-full bg-foreground rounded-full transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>

        {status.current_tic && (
          <div className="text-muted-foreground truncate">
            TIC {status.current_tic}
            {status.current_phase && (
              <span className="text-blue-400/80 ml-1">
                · {shortPhase(status.current_phase)}
              </span>
            )}
          </div>
        )}

        {status.errors.length > 0 && (
          <div className="text-red-400/70">
            {status.errors.length} error{status.errors.length > 1 ? "s" : ""}
          </div>
        )}
      </div>
    </div>
  )
}

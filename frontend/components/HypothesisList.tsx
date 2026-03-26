"use client"
import { useState } from "react"
import type { Hypothesis } from "@/lib/types"

const CATEGORY_STYLES = {
  natural: {
    badge: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    bar: "bg-blue-500",
    icon: "text-blue-400",
  },
  artificial: {
    badge: "bg-red-500/10 text-red-400 border-red-500/20",
    bar: "bg-red-500",
    icon: "text-red-400",
  },
} as const

function ScoreBar({ score, category }: { score: number; category: "natural" | "artificial" }) {
  const style = CATEGORY_STYLES[category]
  const pct = Math.round(score * 100)
  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${style.bar}`}
          style={{ width: `${pct}%`, opacity: 0.3 + score * 0.7 }}
        />
      </div>
      <span className="text-[11px] font-mono text-muted-foreground w-10 text-right">
        {pct}%
      </span>
    </div>
  )
}

function HypothesisRow({ h, rank }: { h: Hypothesis; rank: number }) {
  const [open, setOpen] = useState(false)
  const style = CATEGORY_STYLES[h.category]

  return (
    <div className="hover:bg-white/5 transition-colors">
      <button
        onClick={() => setOpen(!open)}
        className="w-full text-left px-4 py-3 flex items-start gap-3"
      >
        <span className="text-[11px] font-mono text-muted-foreground mt-0.5 w-5 shrink-0">
          {rank}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-medium text-dim">{h.name}</span>
            <span className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border ${style.badge}`}>
              {h.category}
            </span>
          </div>
          <p className="text-[11px] text-muted-foreground leading-relaxed line-clamp-2">
            {h.description}
          </p>
        </div>
        <div className="shrink-0 mt-0.5">
          <ScoreBar score={h.score} category={h.category} />
        </div>
        <span className="text-muted-foreground mt-0.5 shrink-0 text-xs">
          {open ? "−" : "+"}
        </span>
      </button>

      {open && h.reasoning.length > 0 && (
        <div className="px-4 pb-3 pl-12">
          <div className="border-l-2 border-border/60 pl-3 space-y-1">
            {h.reasoning.map((r, i) => (
              <p key={i} className="text-[11px] text-dim leading-relaxed flex gap-2">
                <span className={`shrink-0 mt-0.5 ${
                  r.toLowerCase().includes("inconsistent") ||
                  r.toLowerCase().includes("no ") ||
                  r.toLowerCase().includes("less ")
                    ? "text-muted-foreground"
                    : style.icon
                }`}>
                  {r.toLowerCase().includes("inconsistent") ||
                   r.toLowerCase().includes("no ") ||
                   r.toLowerCase().includes("less ")
                    ? "−"
                    : "+"}
                </span>
                {r}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export function HypothesisList({ hypotheses }: { hypotheses: Hypothesis[] }) {
  if (!hypotheses || hypotheses.length === 0) return null

  const [filter, setFilter] = useState<"all" | "natural" | "artificial">("all")

  const filtered = filter === "all"
    ? hypotheses
    : hypotheses.filter((h) => h.category === filter)

  const topNatural = hypotheses.find((h) => h.category === "natural")
  const topArtificial = hypotheses.find((h) => h.category === "artificial")

  return (
    <div className="card-surface overflow-hidden">
      <div className="px-4 py-3 border-b border-border/60 bg-card flex items-center justify-between">
        <div>
          <p className="label-upper">Hypothesis Analysis</p>
          <p className="text-[11px] text-muted-foreground mt-0.5">
            Ranked candidate explanations for unclassified events
          </p>
        </div>
        <div className="flex gap-1">
          {(["all", "natural", "artificial"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`text-[10px] uppercase tracking-wider px-2 py-1 rounded transition-colors ${
                filter === f
                  ? "bg-white/10 text-foreground"
                  : "text-muted-foreground hover:text-dim"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {topNatural && topArtificial && (
        <div className="px-4 py-3 border-b border-border/40 bg-card/50 flex gap-4 text-[11px]">
          <span className="text-muted-foreground">
            Top natural:{" "}
            <span className="text-blue-400 font-medium">{topNatural.name}</span>
            <span className="text-muted-foreground font-mono ml-1">({Math.round(topNatural.score * 100)}%)</span>
          </span>
          <span className="text-muted-foreground">
            Top artificial:{" "}
            <span className="text-red-400 font-medium">{topArtificial.name}</span>
            <span className="text-muted-foreground font-mono ml-1">({Math.round(topArtificial.score * 100)}%)</span>
          </span>
        </div>
      )}

      <div className="divide-y divide-border/40">
        {filtered.map((h, i) => (
          <HypothesisRow key={h.id} h={h} rank={i + 1} />
        ))}
      </div>
    </div>
  )
}

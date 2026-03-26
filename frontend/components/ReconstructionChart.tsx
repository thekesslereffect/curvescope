"use client"

import { useState } from "react"
import {
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from "recharts"
import type { ReconstructionSample } from "@/lib/api"

const LABEL_META: Record<string, { title: string; color: string; desc: string }> = {
  best: { title: "Best", color: "#22c55e", desc: "Lowest error - the model nailed these" },
  typical: { title: "Typical", color: "#3b82f6", desc: "Mid-range error - representative of training data" },
  worst: { title: "Worst", color: "#ef4444", desc: "Highest error - hardest to reconstruct" },
  live: { title: "Live", color: "#f59e0b", desc: "Current model output - updates each epoch" },
}

function SingleRecon({ sample }: { sample: ReconstructionSample }) {
  const meta = LABEL_META[sample.label] ?? LABEL_META.typical
  const data = sample.original.map((val, i) => {
    const recon = sample.reconstructed[i]
    return { i, original: +val.toFixed(4), reconstructed: +recon.toFixed(4), residual: +(val - recon).toFixed(4) }
  })

  return (
    <div className="card-surface p-3">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] uppercase tracking-wider font-medium" style={{ color: meta.color }}>
          {meta.title}
        </span>
        <span className="text-[10px] font-mono text-muted-foreground">
          MSE {sample.error.toFixed(6)}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={130}>
        <ComposedChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2e2e2e" />
          <XAxis dataKey="i" tick={false} axisLine={false} />
          <YAxis
            tick={{ fontSize: 9, fontFamily: "var(--font-mono, monospace)", fill: "#525252" }}
            tickLine={false}
            width={36}
            tickFormatter={(v) => (typeof v === "number" ? v.toFixed(1) : String(v))}
            stroke="#2e2e2e"
          />
          <Tooltip
            contentStyle={{
              fontSize: 10,
              fontFamily: "var(--font-mono, monospace)",
              background: "#1a1a1a",
              border: "1px solid #2e2e2e",
              borderRadius: "0.75rem",
            }}
            formatter={(v, name) => [Number(v).toFixed(4), String(name)]}
          />
          <Area
            dataKey="residual"
            fill="#ef4444"
            fillOpacity={0.08}
            stroke="none"
            isAnimationActive={false}
          />
          <Line
            dataKey="original"
            stroke="#b4b4b4"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
            name="Original"
          />
          <Line
            dataKey="reconstructed"
            stroke={meta.color}
            strokeWidth={1.5}
            strokeDasharray="4 2"
            dot={false}
            isAnimationActive={false}
            name="Reconstructed"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

export function ReconstructionChart({ samples }: { samples: ReconstructionSample[] }) {
  const [activeLabel, setActiveLabel] = useState<string>("all")

  if (!samples.length) {
    return (
      <div className="text-xs text-muted-foreground py-6 text-center card-surface">
        Reconstruction samples appear after training completes
      </div>
    )
  }

  const isLive = samples.length === 1 && (samples[0].label as string) === "live"
  const uniqueLabels = Array.from(new Set(samples.map((s) => s.label)))
  const filterLabels = isLive ? [] : ["all", ...uniqueLabels]
  const filtered = activeLabel === "all" || isLive ? samples : samples.filter((s) => s.label === activeLabel)

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="label-upper">
          {isLive ? "Live reconstruction (updates each epoch)" : "Reconstruction samples"}
        </p>
        {filterLabels.length > 1 && (
          <div className="flex gap-1">
            {filterLabels.map((l) => (
              <button
                key={l}
                type="button"
                onClick={() => setActiveLabel(l)}
                className={`px-2 py-0.5 rounded-lg text-[10px] transition-colors ${
                  activeLabel === l
                    ? "bg-white/10 text-foreground"
                    : "text-muted-foreground hover:text-dim"
                }`}
              >
                {l === "all" ? "All" : LABEL_META[l]?.title ?? l}
              </button>
            ))}
          </div>
        )}
      </div>
      <p className="text-[10px] text-muted-foreground leading-relaxed">
        {isLive
          ? "Gray = original training window, amber dashed = current model output. Watch the reconstruction improve as training progresses."
          : "Gray = original training window, colored dashed = model reconstruction. The pink residual area shows where they disagree. Best = model learned this pattern perfectly. Worst = patterns the model finds hardest to reproduce (edge of \"normal\")."}
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {filtered.map((s, i) => (
          <SingleRecon key={`${s.label}-${i}`} sample={s} />
        ))}
      </div>
    </div>
  )
}

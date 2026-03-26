"use client"
import { AreaChart, Area, XAxis, YAxis, Tooltip, ReferenceLine, ReferenceArea, ResponsiveContainer } from "recharts"
import type { FlaggedEvent } from "@/lib/types"

interface Props {
  time: number[]
  scores: number[]
  events?: FlaggedEvent[]
  height?: number
}

export function AnomalyScoreChart({ time, scores, events = [], height = 100 }: Props) {
  const MAX_POINTS = 3000
  const step = Math.max(1, Math.floor(time.length / MAX_POINTS))

  const data = time
    .filter((_, i) => i % step === 0)
    .map((t, i) => ({ t: +t.toFixed(2), s: +(scores[i * step] ?? 0).toFixed(4) }))
    .filter(d => !isNaN(d.s))

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
        <XAxis dataKey="t" tick={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }} tickLine={false} stroke="#2e2e2e" />
        <YAxis
          domain={[0, 1]}
          tick={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }}
          tickLine={false}
          width={30}
          tickFormatter={(v: number) => v.toFixed(1)}
          stroke="#2e2e2e"
        />
        <Tooltip
          contentStyle={{ fontSize: 11, fontFamily: "var(--font-mono, monospace)", background: "#1a1a1a", border: "1px solid #2e2e2e", borderRadius: "0.75rem" }}
          formatter={(v) => [Number(v).toFixed(4), "score"]}
        />
        {events.map((ev) => (
          <ReferenceArea
            key={ev.id}
            x1={+(ev.time_center - ev.duration_hours / 48).toFixed(2)}
            x2={+(ev.time_center + ev.duration_hours / 48).toFixed(2)}
            fill={ev.event_type === "unknown" ? "rgba(220,38,38,0.15)" : "rgba(59,130,246,0.10)"}
            stroke={ev.event_type === "unknown" ? "#dc2626" : "#3b82f6"}
            strokeWidth={0.5}
          />
        ))}
        <ReferenceLine y={0.5} stroke="#dc2626" strokeDasharray="4 4" strokeOpacity={0.5} />
        <Area type="monotone" dataKey="s" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} strokeWidth={1} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  )
}

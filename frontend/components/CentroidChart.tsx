"use client"
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ReferenceArea, ResponsiveContainer } from "recharts"
import type { CentroidResult, FlaggedEvent } from "@/lib/types"

interface Props {
  centroid: CentroidResult
  events?: FlaggedEvent[]
  height?: number
}

export function CentroidChart({ centroid, events = [], height = 160 }: Props) {
  if (!centroid?.available || !centroid.time) {
    return (
      <div className="flex items-center justify-center text-xs text-muted-foreground" style={{ height }}>
        TPF centroid data unavailable for this target
      </div>
    )
  }

  const data = centroid.time.map((t, i) => ({
    t: +t.toFixed(2),
    d: +(centroid.displacement_arcsec?.[i] ?? 0).toFixed(2),
  })).filter(d => !isNaN(d.d))

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
        <XAxis dataKey="t" tick={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }} tickLine={false} stroke="#2e2e2e" />
        <YAxis
          tick={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }}
          tickLine={false}
          tickFormatter={(v: number) => `${v}"`}
          width={36}
          stroke="#2e2e2e"
        />
        <Tooltip
          contentStyle={{ fontSize: 11, fontFamily: "var(--font-mono, monospace)", background: "#1a1a1a", border: "1px solid #2e2e2e", borderRadius: "0.75rem" }}
          formatter={(v) => [`${Number(v).toFixed(1)}"`, "shift"]}
        />
        {events.map(ev => (
          <ReferenceArea
            key={ev.id}
            x1={+(ev.time_center - ev.duration_hours / 48).toFixed(2)}
            x2={+(ev.time_center + ev.duration_hours / 48).toFixed(2)}
            fill={ev.event_type === "contamination" ? "rgba(220,38,38,0.12)" : "rgba(59,130,246,0.06)"}
          />
        ))}
        <ReferenceLine y={10} stroke="#dc2626" strokeDasharray="4 4" />
        <Line
          type="monotone" dataKey="d"
          stroke={centroid.shift_flagged ? "#dc2626" : "#3b82f6"}
          strokeWidth={1} dot={false} isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

"use client"
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceArea, ResponsiveContainer } from "recharts"
import type { FlaggedEvent } from "@/lib/types"

interface Props {
  time: number[]
  flux: number[]
  events?: FlaggedEvent[]
  height?: number
}

export function LightCurveChart({ time, flux, events = [], height = 220 }: Props) {
  const MAX_POINTS = 4000
  const step = Math.max(1, Math.floor(time.length / MAX_POINTS))

  const data = time
    .filter((_, i) => i % step === 0)
    .map((t, i) => ({ t: +t.toFixed(2), f: flux[i * step] }))
    .filter(d => d.f !== undefined && !isNaN(d.f))

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
        <XAxis
          dataKey="t"
          tick={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }}
          tickLine={false}
          stroke="#2e2e2e"
          label={{ value: "BTJD", position: "insideBottomRight", fontSize: 10, fill: "#737373" }}
        />
        <YAxis
          domain={["auto", "auto"]}
          tick={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }}
          tickLine={false}
          tickFormatter={(v: number) => v.toFixed(4)}
          width={58}
          stroke="#2e2e2e"
        />
        <Tooltip
          contentStyle={{ fontSize: 11, fontFamily: "var(--font-mono, monospace)", background: "#1a1a1a", border: "1px solid #2e2e2e", borderRadius: "0.75rem" }}
          labelStyle={{ color: "#737373" }}
          formatter={(v) => [Number(v).toFixed(5), "flux"]}
          labelFormatter={(t) => `t = ${t} BTJD`}
        />
        {events.map((ev) => (
          <ReferenceArea
            key={ev.id}
            x1={+(ev.time_center - ev.duration_hours / 48).toFixed(2)}
            x2={+(ev.time_center + ev.duration_hours / 48).toFixed(2)}
            fill={ev.event_type === "unknown" ? "rgba(220,38,38,0.12)" : "rgba(59,130,246,0.08)"}
            stroke={ev.event_type === "unknown" ? "#dc2626" : "#3b82f6"}
            strokeWidth={1}
          />
        ))}
        <Line type="monotone" dataKey="f" stroke="#3b82f6" strokeWidth={1} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

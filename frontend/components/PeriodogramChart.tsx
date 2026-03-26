"use client"
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer } from "recharts"

interface Props {
  periods: number[]
  powers: number[]
  bestPeriod: number | null
  height?: number
}

function fmtPower(v: number): string {
  if (v === 0) return "0"
  if (Math.abs(v) < 0.001) return v.toExponential(2)
  return v.toPrecision(4)
}

export function PeriodogramChart({ periods, powers, bestPeriod, height = 220 }: Props) {
  if (!periods.length || !powers.length) {
    return <div className="h-[220px] flex items-center justify-center text-xs text-muted-foreground">No periodogram data</div>
  }

  const MAX_POINTS = 2000
  const step = Math.max(1, Math.floor(periods.length / MAX_POINTS))

  const data: { p: number; pw: number }[] = []
  for (let i = 0; i < periods.length; i += step) {
    const p = periods[i]
    const pw = powers[i] ?? 0
    if (p > 0) data.push({ p, pw })
  }

  const pMin = data[0]?.p ?? 0.1
  const pMax = data[data.length - 1]?.p ?? 1

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
        <XAxis
          dataKey="p"
          type="number"
          tick={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }}
          tickLine={false}
          scale="log"
          domain={[pMin, pMax]}
          allowDataOverflow
          tickFormatter={(v: number) => v < 1 ? v.toFixed(2) : v.toFixed(1)}
          label={{ value: "Period (days)", position: "insideBottomRight", fontSize: 10, fill: "#737373" }}
          stroke="#2e2e2e"
        />
        <YAxis
          tick={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }}
          tickLine={false}
          width={52}
          tickFormatter={(v: number) => fmtPower(v)}
          label={{ value: "Power", angle: -90, position: "insideLeft", fontSize: 10, fill: "#737373" }}
          stroke="#2e2e2e"
        />
        <Tooltip
          contentStyle={{ fontSize: 11, fontFamily: "var(--font-mono, monospace)", background: "#1a1a1a", border: "1px solid #2e2e2e", borderRadius: "0.75rem" }}
          formatter={(v) => [fmtPower(Number(v)), "power"]}
          labelFormatter={(p) => `P = ${Number(p).toPrecision(4)} days`}
        />
        {bestPeriod != null && bestPeriod > 0 && (
          <ReferenceLine
            x={bestPeriod}
            stroke="#22c55e"
            strokeDasharray="4 4"
            label={{ value: `${bestPeriod.toFixed(3)}d`, fontSize: 10, fill: "#22c55e", position: "top" }}
          />
        )}
        <Line type="monotone" dataKey="pw" stroke="#8b5cf6" strokeWidth={1} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts"
import type { ErrorHistogram } from "@/lib/api"

export function ErrorHistogramChart({
  histogram,
  height = 180,
}: {
  histogram: ErrorHistogram
  height?: number
}) {
  if (!histogram.bin_centers?.length) {
    return (
      <div
        className="flex items-center justify-center text-[11px] text-muted-foreground card-surface"
        style={{ height }}
      >
        Error distribution appears after training completes
      </div>
    )
  }

  const data = histogram.bin_centers.map((center, i) => ({
    error: center,
    count: histogram.counts[i],
  }))

  return (
    <div className="card-surface p-3">
      <div className="flex items-center justify-between mb-1 px-1">
        <p className="label-upper">
          Training error distribution
        </p>
        <div className="flex gap-3 text-[10px] font-mono text-muted-foreground">
          <span>mean {histogram.mean.toFixed(4)}</span>
          <span>p99 {histogram.p99.toFixed(4)}</span>
          <span>{histogram.total_windows.toLocaleString()} windows</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2e2e2e" vertical={false} />
          <XAxis
            dataKey="error"
            tick={{ fontSize: 9, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }}
            tickLine={false}
            tickFormatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))}
            label={{ value: "Reconstruction error (MSE)", position: "insideBottomRight", fontSize: 10, fill: "#525252" }}
            stroke="#2e2e2e"
          />
          <YAxis
            tick={{ fontSize: 9, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }}
            tickLine={false}
            width={44}
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
            labelFormatter={(v) => `Error: ${Number(v).toFixed(5)}`}
            formatter={(v) => [Number(v).toLocaleString(), "windows"]}
          />
          <ReferenceLine
            x={histogram.mean}
            stroke="#3b82f6"
            strokeDasharray="4 4"
            strokeWidth={1.5}
            label={{ value: "mean", position: "top", fontSize: 9, fill: "#3b82f6" }}
          />
          <ReferenceLine
            x={histogram.p99}
            stroke="#ef4444"
            strokeDasharray="4 4"
            strokeWidth={1.5}
            label={{ value: "p99", position: "top", fontSize: 9, fill: "#ef4444" }}
          />
          <Bar
            dataKey="count"
            fill="#f59e0b"
            fillOpacity={0.7}
            radius={[2, 2, 0, 0]}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

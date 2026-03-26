"use client"

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from "recharts"

export function TrainingLossChart({
  history,
  height = 200,
}: {
  history: { epoch: number; loss: number; val_loss?: number }[]
  height?: number
}) {
  if (!history.length) {
    return (
      <div
        className="flex items-center justify-center text-[11px] text-muted-foreground card-surface"
        style={{ height }}
      >
        Loss curve appears after the first training epoch
      </div>
    )
  }

  const hasVal = history.some((p) => p.val_loss != null)

  const data = history.map((p) => ({
    epoch: p.epoch,
    loss: +p.loss.toFixed(6),
    ...(p.val_loss != null ? { val_loss: +p.val_loss.toFixed(6) } : {}),
  }))

  return (
    <div className="card-surface p-3">
      <p className="label-upper mb-1 px-1">
        Loss (MSE) vs epoch
      </p>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2e2e2e" />
          <XAxis
            dataKey="epoch"
            tick={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }}
            tickLine={false}
            label={{ value: "Epoch", position: "insideBottomRight", fontSize: 10, fill: "#737373" }}
            stroke="#2e2e2e"
          />
          <YAxis
            tick={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)", fill: "#737373" }}
            tickLine={false}
            width={52}
            tickFormatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))}
            stroke="#2e2e2e"
          />
          <Tooltip
            contentStyle={{
              fontSize: 11,
              fontFamily: "var(--font-mono, monospace)",
              background: "#1a1a1a",
              border: "1px solid #2e2e2e",
              borderRadius: "0.75rem",
            }}
            labelFormatter={(e) => `Epoch ${e}`}
            formatter={(v, name) => [
              Number(v).toFixed(6),
              name === "val_loss" ? "val loss" : "train loss",
            ]}
          />
          {hasVal && (
            <Legend
              verticalAlign="top"
              height={20}
              wrapperStyle={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)" }}
            />
          )}
          <Line
            type="monotone"
            dataKey="loss"
            name="train"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          {hasVal && (
            <Line
              type="monotone"
              dataKey="val_loss"
              name="val"
              stroke="#3b82f6"
              strokeWidth={2}
              strokeDasharray="4 2"
              dot={false}
              isAnimationActive={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

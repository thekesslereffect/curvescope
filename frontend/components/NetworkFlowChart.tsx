"use client"

import { useEffect, useRef } from "react"
import type { ActivationLayer } from "@/lib/api"

const INFERNO: [number, number, number][] = [
  [0, 0, 4], [20, 11, 53], [58, 12, 96], [96, 21, 103],
  [132, 37, 92], [165, 54, 74], [196, 73, 51], [222, 101, 28],
  [241, 137, 10], [249, 178, 23], [246, 220, 76], [252, 255, 164],
]

function inferno(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t))
  const scaled = clamped * (INFERNO.length - 1)
  const lo = Math.floor(scaled)
  const hi = Math.min(lo + 1, INFERNO.length - 1)
  const frac = scaled - lo
  return [
    Math.round(INFERNO[lo][0] + frac * (INFERNO[hi][0] - INFERNO[lo][0])),
    Math.round(INFERNO[lo][1] + frac * (INFERNO[hi][1] - INFERNO[lo][1])),
    Math.round(INFERNO[lo][2] + frac * (INFERNO[hi][2] - INFERNO[lo][2])),
  ]
}

function HeatmapLayer({ layer, maxWidth }: { layer: ActivationLayer; maxWidth: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const data = layer.data as number[][]
  const rows = data.length
  const cols = data[0]?.length ?? 0

  const cellW = Math.max(2, Math.min(8, Math.floor(maxWidth / cols)))
  const cellH = Math.max(2, Math.min(6, Math.floor(120 / rows)))
  const w = cellW * cols
  const h = cellH * rows

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let vmin = Infinity
    let vmax = -Infinity
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = data[r][c]
        if (v < vmin) vmin = v
        if (v > vmax) vmax = v
      }
    }
    if (vmax === vmin) vmax = vmin + 1
    const range = vmax - vmin

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const t = (data[r][c] - vmin) / range
        const [rv, gv, bv] = inferno(t)
        ctx.fillStyle = `rgb(${rv},${gv},${bv})`
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH)
      }
    }
  }, [data, rows, cols, cellW, cellH])

  return (
    <canvas
      ref={canvasRef}
      width={w}
      height={h}
      className="rounded-lg"
      style={{ imageRendering: "pixelated", maxWidth: "100%" }}
    />
  )
}

function BarLayer({ layer, maxWidth }: { layer: ActivationLayer; maxWidth: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const data = layer.data as number[]
  const n = data.length

  const barW = Math.max(2, Math.min(8, Math.floor(maxWidth / n)))
  const w = barW * n
  const h = 48

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctx.fillStyle = "#181818"
    ctx.fillRect(0, 0, w, h)

    const absMax = Math.max(...data.map(Math.abs), 0.001)

    for (let i = 0; i < n; i++) {
      const v = data[i]
      const norm = v / absMax
      const barH = Math.abs(norm) * (h / 2)
      const t = (norm + 1) / 2
      const [rv, gv, bv] = inferno(t)
      ctx.fillStyle = `rgb(${rv},${gv},${bv})`
      if (v >= 0) {
        ctx.fillRect(i * barW, h / 2 - barH, barW - 0.5, barH)
      } else {
        ctx.fillRect(i * barW, h / 2, barW - 0.5, barH)
      }
    }

    ctx.strokeStyle = "rgba(255,255,255,0.1)"
    ctx.beginPath()
    ctx.moveTo(0, h / 2)
    ctx.lineTo(w, h / 2)
    ctx.stroke()
  }, [data, n, barW, w, h])

  return (
    <canvas
      ref={canvasRef}
      width={w}
      height={h}
      className="rounded-lg"
      style={{ imageRendering: "pixelated", maxWidth: "100%" }}
    />
  )
}

function Arrow() {
  return (
    <div className="flex justify-center py-1">
      <svg width="12" height="16" viewBox="0 0 12 16" className="text-muted-foreground">
        <path d="M6 0 L6 12 M2 8 L6 14 L10 8" stroke="currentColor" strokeWidth="1.5" fill="none" />
      </svg>
    </div>
  )
}

export function NetworkFlowChart({ activations }: { activations: ActivationLayer[] }) {
  if (!activations.length) {
    return (
      <div className="text-xs text-muted-foreground py-6 text-center card-surface">
        Network activation flow appears after training completes
      </div>
    )
  }

  const maxWidth = 520

  return (
    <div className="card-surface p-5 space-y-0">
      <p className="label-upper mb-3">
        Network activation flow (typical sample)
      </p>
      <p className="text-[10px] text-muted-foreground leading-relaxed mb-4">
        How data flows through the autoencoder. Each heatmap shows neuron activations at that layer
        (rows = filters, columns = spatial position). Bright = high activation.
        The bottleneck compresses the signal to 64 numbers, forcing the network to learn
        only the most important patterns.
      </p>

      {activations.map((layer, i) => {
        const isEncoder = layer.name.startsWith("Encoder") || layer.name.startsWith("Bottleneck")
        const isBottleneck = layer.name.startsWith("Bottleneck")

        return (
          <div key={i}>
            {i > 0 && <Arrow />}
            <div className={`flex items-start gap-3 ${isBottleneck ? "justify-center" : ""}`}>
              <div className={`shrink-0 ${isBottleneck ? "text-center" : ""}`}>
                <p className={`text-[10px] mb-1 font-medium ${
                  isBottleneck ? "text-amber-400" : isEncoder ? "text-blue-400" : "text-emerald-400"
                }`}>
                  {layer.name}
                </p>
                <p className="text-[9px] font-mono text-muted-foreground">
                  {layer.shape.join(" x ")}
                </p>
              </div>
              <div className="flex-1 min-w-0 overflow-hidden">
                {layer.type === "heatmap" ? (
                  <HeatmapLayer layer={layer} maxWidth={maxWidth} />
                ) : (
                  <BarLayer layer={layer} maxWidth={maxWidth} />
                )}
              </div>
            </div>
          </div>
        )
      })}

      <div className="flex justify-center gap-6 mt-4 pt-3 border-t border-border/50">
        <span className="flex items-center gap-1.5 text-[9px] text-muted-foreground">
          <span className="inline-block w-2 h-2 rounded-sm bg-blue-400" />
          Encoder layers
        </span>
        <span className="flex items-center gap-1.5 text-[9px] text-muted-foreground">
          <span className="inline-block w-2 h-2 rounded-sm bg-amber-400" />
          Bottleneck
        </span>
        <span className="flex items-center gap-1.5 text-[9px] text-muted-foreground">
          <span className="inline-block w-2 h-2 rounded-sm bg-emerald-400" />
          Decoder layers
        </span>
      </div>
    </div>
  )
}

"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import type { TPFData } from "@/lib/types"

interface Props {
  tpf: TPFData
  height?: number
}

const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
  [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112],
  [121, 209, 81], [189, 222, 38], [253, 231, 37],
]

function viridis(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t))
  const scaled = clamped * (VIRIDIS.length - 1)
  const lo = Math.floor(scaled)
  const hi = Math.min(lo + 1, VIRIDIS.length - 1)
  const frac = scaled - lo
  return [
    Math.round(VIRIDIS[lo][0] + frac * (VIRIDIS[hi][0] - VIRIDIS[lo][0])),
    Math.round(VIRIDIS[lo][1] + frac * (VIRIDIS[hi][1] - VIRIDIS[lo][1])),
    Math.round(VIRIDIS[lo][2] + frac * (VIRIDIS[hi][2] - VIRIDIS[lo][2])),
  ]
}

function drawColorBar(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, vmin: number, vmax: number) {
  for (let i = 0; i < h; i++) {
    const t = 1 - i / h
    const [r, g, b] = viridis(t)
    ctx.fillStyle = `rgb(${r},${g},${b})`
    ctx.fillRect(x, y + i, w, 1)
  }
  ctx.fillStyle = "#b4b4b4"
  ctx.font = "10px monospace"
  ctx.textAlign = "left"
  ctx.fillText(vmax.toFixed(0), x + w + 4, y + 10)
  ctx.fillText(vmin.toFixed(0), x + w + 4, y + h)
  ctx.fillText("e⁻/s", x + w + 4, y + h / 2 + 3)
}

export function TPFViewer({ tpf, height = 360 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(60)
  const [hoveredPixel, setHoveredPixel] = useState<{ r: number; c: number; val: number } | null>(null)
  const animRef = useRef<number | null>(null)
  const lastTimeRef = useRef(0)

  const nFrames = tpf.n_frames ?? 0
  const nRows = tpf.n_rows ?? 0
  const nCols = tpf.n_cols ?? 0
  const flux = tpf.flux
  const time = tpf.time

  const colorBarWidth = 16
  const colorBarMargin = 50
  const pixelSize = nRows > 0 && nCols > 0
    ? Math.floor(Math.min((height - 40) / nRows, (height * 1.4 - colorBarWidth - colorBarMargin) / nCols))
    : 20
  const gridW = pixelSize * nCols
  const gridH = pixelSize * nRows
  const canvasW = gridW + colorBarWidth + colorBarMargin + 20
  const canvasH = gridH + 40

  const getFrameMinMax = useCallback((f: number) => {
    if (!flux || !flux[f]) return { vmin: 0, vmax: 1 }
    let vmin = Infinity
    let vmax = -Infinity
    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const v = flux[f][r][c]
        if (v < vmin) vmin = v
        if (v > vmax) vmax = v
      }
    }
    if (vmin === vmax) vmax = vmin + 1
    return { vmin, vmax }
  }, [flux, nRows, nCols])

  const draw = useCallback((f: number) => {
    const canvas = canvasRef.current
    if (!canvas || !flux || !flux[f]) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const { vmin, vmax } = getFrameMinMax(f)
    const range = vmax - vmin

    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const val = flux[f][r][c]
        const t = (val - vmin) / range
        const [rv, gv, bv] = viridis(t)
        ctx.fillStyle = `rgb(${rv},${gv},${bv})`
        ctx.fillRect(c * pixelSize, r * pixelSize, pixelSize, pixelSize)
      }
    }

    ctx.strokeStyle = "rgba(255,255,255,0.08)"
    ctx.lineWidth = 0.5
    for (let r = 0; r <= nRows; r++) {
      ctx.beginPath()
      ctx.moveTo(0, r * pixelSize)
      ctx.lineTo(gridW, r * pixelSize)
      ctx.stroke()
    }
    for (let c = 0; c <= nCols; c++) {
      ctx.beginPath()
      ctx.moveTo(c * pixelSize, 0)
      ctx.lineTo(c * pixelSize, gridH)
      ctx.stroke()
    }

    if (tpf.aperture_mask) {
      ctx.strokeStyle = "rgba(255, 100, 100, 0.9)"
      ctx.lineWidth = 2
      for (let r = 0; r < nRows; r++) {
        for (let c = 0; c < nCols; c++) {
          if (!tpf.aperture_mask[r]?.[c]) continue
          const x = c * pixelSize
          const y = r * pixelSize
          if (r === 0 || !tpf.aperture_mask[r - 1]?.[c]) {
            ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + pixelSize, y); ctx.stroke()
          }
          if (r === nRows - 1 || !tpf.aperture_mask[r + 1]?.[c]) {
            ctx.beginPath(); ctx.moveTo(x, y + pixelSize); ctx.lineTo(x + pixelSize, y + pixelSize); ctx.stroke()
          }
          if (c === 0 || !tpf.aperture_mask[r]?.[c - 1]) {
            ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + pixelSize); ctx.stroke()
          }
          if (c === nCols - 1 || !tpf.aperture_mask[r]?.[c + 1]) {
            ctx.beginPath(); ctx.moveTo(x + pixelSize, y); ctx.lineTo(x + pixelSize, y + pixelSize); ctx.stroke()
          }
        }
      }
    }

    drawColorBar(ctx, gridW + 16, 4, colorBarWidth, gridH - 8, vmin, vmax)

    ctx.fillStyle = "#b4b4b4"
    ctx.font = "11px monospace"
    ctx.textAlign = "left"
    const t = time?.[f]
    ctx.fillText(
      `Frame ${f + 1}/${nFrames}${t != null ? `  BTJD ${t.toFixed(4)}` : ""}`,
      0,
      gridH + 16
    )
    if (tpf.column != null && tpf.row != null) {
      ctx.fillText(`CCD col ${tpf.column} row ${tpf.row}`, 0, gridH + 30)
    }
  }, [flux, nRows, nCols, nFrames, pixelSize, gridW, gridH, colorBarWidth, getFrameMinMax, time, tpf.aperture_mask, tpf.column, tpf.row])

  useEffect(() => { draw(frame) }, [frame, draw])

  useEffect(() => {
    if (!playing) {
      if (animRef.current != null) cancelAnimationFrame(animRef.current)
      animRef.current = null
      return
    }
    const interval = 1000 / speed
    const tick = (ts: number) => {
      if (ts - lastTimeRef.current >= interval) {
        lastTimeRef.current = ts
        setFrame((prev) => (prev + 1) % nFrames)
      }
      animRef.current = requestAnimationFrame(tick)
    }
    animRef.current = requestAnimationFrame(tick)
    return () => {
      if (animRef.current != null) cancelAnimationFrame(animRef.current)
    }
  }, [playing, speed, nFrames])

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas || !flux) return
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const mx = (e.clientX - rect.left) * scaleX
    const my = (e.clientY - rect.top) * scaleY
    const c = Math.floor(mx / pixelSize)
    const r = Math.floor(my / pixelSize)
    if (r >= 0 && r < nRows && c >= 0 && c < nCols && flux[frame]) {
      setHoveredPixel({ r, c, val: flux[frame][r][c] })
    } else {
      setHoveredPixel(null)
    }
  }

  if (!tpf.available || nFrames === 0 || !flux) {
    return (
      <div className="text-xs text-muted-foreground py-8 text-center">
        No pixel data available for this target.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setPlaying(!playing)}
            className="btn-secondary text-xs py-1.5"
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button
            type="button"
            onClick={() => { setFrame((f) => (f - 1 + nFrames) % nFrames) }}
            disabled={playing}
            className="btn-ghost text-xs py-1.5 disabled:opacity-40"
          >
            &#9664;
          </button>
          <button
            type="button"
            onClick={() => { setFrame((f) => (f + 1) % nFrames) }}
            disabled={playing}
            className="btn-ghost text-xs py-1.5 disabled:opacity-40"
          >
            &#9654;
          </button>
          <label className="text-[10px] text-muted-foreground flex items-center gap-1">
            Speed
            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="input-field text-[10px] py-0.5 px-1"
            >
              <option value={10}>10 fps</option>
              <option value={30}>30 fps</option>
              <option value={60}>60 fps</option>
              <option value={120}>120 fps</option>
            </select>
          </label>
        </div>

        {hoveredPixel && (
          <span className="text-[10px] font-mono text-dim">
            Pixel ({hoveredPixel.c + (tpf.column ?? 0)}, {hoveredPixel.r + (tpf.row ?? 0)})
            {" = "}{hoveredPixel.val.toFixed(2)} e⁻/s
            {tpf.aperture_mask?.[hoveredPixel.r]?.[hoveredPixel.c] ? " (in aperture)" : ""}
          </span>
        )}
      </div>

      <canvas
        ref={canvasRef}
        width={canvasW}
        height={canvasH}
        onMouseMove={handleCanvasMouseMove}
        onMouseLeave={() => setHoveredPixel(null)}
        className="max-w-full rounded-xl"
        style={{ imageRendering: "pixelated" }}
      />

      <div className="flex items-center gap-3">
        <input
          type="range"
          min={0}
          max={nFrames - 1}
          value={frame}
          onChange={(e) => { setFrame(Number(e.target.value)); setPlaying(false) }}
          className="flex-1 accent-foreground h-1.5"
        />
        <span className="text-[10px] font-mono text-muted-foreground tabular-nums w-24 text-right shrink-0">
          {frame + 1} / {nFrames}
        </span>
      </div>

      <div className="flex gap-4 text-[10px] text-muted-foreground">
        <span>{nRows} x {nCols} pixels</span>
        <span>{nFrames} frames (downsampled)</span>
        {tpf.aperture_mask && (
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-1.5 rounded-sm border border-red-400/70" />
            Pipeline aperture
          </span>
        )}
      </div>
    </div>
  )
}

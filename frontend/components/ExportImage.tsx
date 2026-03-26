"use client"
import { useCallback } from "react"
import type { Analysis } from "@/lib/types"

const W = 1200
const H = 675
const PAD = 48
const CHART_TOP = 200
const CHART_BOTTOM = H - 120
const CHART_LEFT = PAD + 10
const CHART_RIGHT = W - PAD - 10

function drawExport(canvas: HTMLCanvasElement, analysis: Analysis) {
  const ctx = canvas.getContext("2d")!
  canvas.width = W
  canvas.height = H

  // Background
  ctx.fillStyle = "#000000"
  ctx.fillRect(0, 0, W, H)

  // Subtle grid lines
  ctx.strokeStyle = "rgba(255,255,255,0.03)"
  ctx.lineWidth = 1
  for (let y = CHART_TOP; y <= CHART_BOTTOM; y += 40) {
    ctx.beginPath()
    ctx.moveTo(CHART_LEFT, y)
    ctx.lineTo(CHART_RIGHT, y)
    ctx.stroke()
  }
  for (let x = CHART_LEFT; x <= CHART_RIGHT; x += 80) {
    ctx.beginPath()
    ctx.moveTo(x, CHART_TOP)
    ctx.lineTo(x, CHART_BOTTOM)
    ctx.stroke()
  }

  // --- Light curve ---
  const flux = analysis.raw_flux
  if (flux && flux.time.length > 0) {
    const time = flux.time
    const vals = flux.flux

    let fMin = Infinity, fMax = -Infinity
    for (const v of vals) {
      if (v < fMin) fMin = v
      if (v > fMax) fMax = v
    }
    const fRange = fMax - fMin || 1
    const fPad = fRange * 0.08
    const yMin = fMin - fPad
    const yMax = fMax + fPad

    const tMin = time[0]
    const tMax = time[time.length - 1]
    const tRange = tMax - tMin || 1

    const toX = (t: number) => CHART_LEFT + ((t - tMin) / tRange) * (CHART_RIGHT - CHART_LEFT)
    const toY = (f: number) => CHART_BOTTOM - ((f - yMin) / (yMax - yMin)) * (CHART_BOTTOM - CHART_TOP)

    // Event bands
    if (analysis.events) {
      for (const ev of analysis.events) {
        if (ev.event_type === "systematic" || ev.event_type === "contamination") continue
        const halfDur = (ev.duration_hours / 2 / 24)
        const x1 = toX(ev.time_center - halfDur)
        const x2 = toX(ev.time_center + halfDur)
        ctx.fillStyle = ev.event_type === "unknown"
          ? "rgba(220,38,38,0.12)"
          : "rgba(255,255,255,0.04)"
        ctx.fillRect(x1, CHART_TOP, x2 - x1, CHART_BOTTOM - CHART_TOP)
      }
    }

    // Glow layer
    ctx.strokeStyle = "rgba(255,255,255,0.15)"
    ctx.lineWidth = 4
    ctx.lineJoin = "round"
    ctx.beginPath()
    let started = false
    for (let i = 0; i < time.length; i++) {
      const x = toX(time[i])
      const y = toY(vals[i])
      if (!started) { ctx.moveTo(x, y); started = true }
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Main line
    ctx.strokeStyle = "#ffffff"
    ctx.lineWidth = 1.5
    ctx.beginPath()
    started = false
    for (let i = 0; i < time.length; i++) {
      const x = toX(time[i])
      const y = toY(vals[i])
      if (!started) { ctx.moveTo(x, y); started = true }
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Y-axis labels
    ctx.fillStyle = "rgba(255,255,255,0.25)"
    ctx.font = "11px monospace"
    ctx.textAlign = "right"
    const ySteps = 5
    for (let i = 0; i <= ySteps; i++) {
      const val = yMin + (i / ySteps) * (yMax - yMin)
      const y = toY(val)
      ctx.fillText(val.toFixed(4), CHART_LEFT - 8, y + 4)
    }

    // X-axis labels
    ctx.textAlign = "center"
    const xSteps = 6
    for (let i = 0; i <= xSteps; i++) {
      const t = tMin + (i / xSteps) * tRange
      const x = toX(t)
      ctx.fillText(t.toFixed(1), x, CHART_BOTTOM + 16)
    }

    // Axis label
    ctx.fillStyle = "rgba(255,255,255,0.2)"
    ctx.font = "10px monospace"
    ctx.textAlign = "right"
    ctx.fillText("BTJD", CHART_RIGHT, CHART_BOTTOM + 30)
  }

  // Chart border
  ctx.strokeStyle = "rgba(255,255,255,0.08)"
  ctx.lineWidth = 1
  ctx.strokeRect(CHART_LEFT, CHART_TOP, CHART_RIGHT - CHART_LEFT, CHART_BOTTOM - CHART_TOP)

  // --- Header ---
  const ticId = analysis.target?.tic_id || "Unknown"

  // TIC label
  ctx.fillStyle = "rgba(255,255,255,0.35)"
  ctx.font = "600 11px monospace"
  ctx.textAlign = "left"
  ctx.fillText("TIC", PAD, 52)

  // TIC number
  ctx.fillStyle = "#ffffff"
  ctx.font = "600 38px monospace"
  ctx.fillText(ticId, PAD, 95)

  // Sector
  if (analysis.sector) {
    ctx.fillStyle = "rgba(255,255,255,0.3)"
    ctx.font = "12px monospace"
    ctx.fillText(`SECTOR ${analysis.sector.toUpperCase()}`, PAD, 120)
  }

  // Common name
  if (analysis.target?.common_name) {
    ctx.fillStyle = "rgba(255,255,255,0.2)"
    ctx.font = "italic 13px monospace"
    ctx.fillText(analysis.target.common_name, PAD, 142)
  }

  // --- Anomaly score (right side) ---
  const score = analysis.anomaly_score ?? 0
  ctx.textAlign = "right"

  ctx.fillStyle = "rgba(255,255,255,0.35)"
  ctx.font = "600 11px monospace"
  ctx.fillText("ANOMALY SCORE", W - PAD, 52)

  // Score with intensity-based brightness
  const intensity = Math.round(155 + score * 100)
  ctx.fillStyle = `rgb(${intensity},${intensity},${intensity})`
  ctx.font = "600 52px monospace"
  ctx.fillText(score.toFixed(3), W - PAD, 100)

  // --- Stats row ---
  const statsY = CHART_BOTTOM + 55
  ctx.font = "11px monospace"
  ctx.textAlign = "left"

  const stats: [string, string][] = []
  if (analysis.known_period) stats.push(["PERIOD", `${analysis.known_period.toFixed(3)}d`])
  if (analysis.flag_count) stats.push(["EVENTS", `${analysis.flag_count}`])

  const eventTypes = new Set(analysis.events?.map(e => e.event_type).filter(t => t !== "systematic" && t !== "contamination") ?? [])
  if (eventTypes.size > 0) {
    const primary = [...eventTypes][0].replace(/_/g, " ").toUpperCase()
    stats.push(["PRIMARY", primary])
  }

  if (analysis.raw_flux) {
    const t = analysis.raw_flux.time
    stats.push(["RANGE", `${t[0].toFixed(1)}–${t[t.length - 1].toFixed(1)} BTJD`])
  }

  let statX = PAD
  for (const [label, value] of stats) {
    ctx.fillStyle = "rgba(255,255,255,0.25)"
    ctx.fillText(label, statX, statsY)
    ctx.fillStyle = "rgba(255,255,255,0.6)"
    ctx.fillText(value, statX + ctx.measureText(label).width + 8, statsY)
    statX += ctx.measureText(label).width + ctx.measureText(value).width + 36
  }

  // --- Branding ---
  ctx.textAlign = "right"
  ctx.fillStyle = "rgba(255,255,255,0.15)"
  ctx.font = "10px monospace"
  ctx.fillText("CURVESCOPE — TESS ANOMALY SCANNER", W - PAD, statsY)

  // Top-right decorative line
  ctx.strokeStyle = "rgba(255,255,255,0.06)"
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(PAD, 160)
  ctx.lineTo(W - PAD, 160)
  ctx.stroke()
}

export function ExportImageButton({ analysis }: { analysis: Analysis }) {
  const handleExport = useCallback(() => {
    const canvas = document.createElement("canvas")
    drawExport(canvas, analysis)

    canvas.toBlob((blob) => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      const tic = analysis.target?.tic_id || "target"
      a.download = `TIC-${tic}-anomaly.png`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }, "image/png")
  }, [analysis])

  return (
    <button
      onClick={handleExport}
      className="btn-secondary text-[11px] py-1 px-3"
      title="Export cover image"
    >
      Export image
    </button>
  )
}

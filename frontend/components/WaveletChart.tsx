"use client"
import { useEffect, useRef } from "react"
import type { WaveletResult } from "@/lib/types"

interface Props {
  wavelet: WaveletResult
  height?: number
}

const SYS_LABELS: Record<string, string> = {
  orbital: "13.7d orbital",
  momentum_dump: "3.1d momentum",
  scattered_light: "1.0d scatter",
  half_orbital: "6.9d harmonic",
}

export function WaveletChart({ wavelet, height = 200 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !wavelet?.power?.length || !wavelet.periods?.length || !wavelet.time?.length) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const nP = wavelet.periods.length
    const nT = wavelet.time.length
    const W = canvas.width
    const H = canvas.height

    let maxPow = 0
    for (const row of wavelet.power) for (const v of row) if (v > maxPow) maxPow = v
    if (maxPow === 0) maxPow = 1

    const img = ctx.createImageData(W, H)
    for (let py = 0; py < H; py++) {
      const pIdx = Math.floor((py / H) * nP)
      for (let px = 0; px < W; px++) {
        const tIdx = Math.floor((px / W) * nT)
        const norm = Math.min((wavelet.power[pIdx]?.[tIdx] ?? 0) / maxPow, 1)
        const idx = (py * W + px) * 4
        img.data[idx] = Math.floor(255 * Math.pow(norm, 0.6))
        img.data[idx + 1] = Math.floor(255 * Math.pow(norm, 1.8))
        img.data[idx + 2] = Math.floor(255 * (norm < 0.5 ? norm * 1.5 : 1 - norm))
        img.data[idx + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)

    const logMin = Math.log10(wavelet.periods[0])
    const logMax = Math.log10(wavelet.periods[wavelet.periods.length - 1])
    ctx.setLineDash([4, 4])
    ctx.lineWidth = 1

    for (const sys of wavelet.tess_systematic_periods) {
      const logP = Math.log10(sys.period_days)
      const py = Math.floor(((logP - logMin) / (logMax - logMin)) * H)
      ctx.strokeStyle = "rgba(255,255,255,0.6)"
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = "rgba(255,255,255,0.8)"
      ctx.font = "10px monospace"
      ctx.fillText(SYS_LABELS[sys.name] ?? sys.name, 4, py - 3)
      ctx.setLineDash([4, 4])
    }
    ctx.setLineDash([])
  }, [wavelet])

  return (
    <div className="relative rounded-xl overflow-hidden">
      <canvas ref={canvasRef} width={800} height={height} className="w-full block" style={{ height: `${height}px` }} />
      <div className="absolute bottom-1 right-2 text-[10px] font-mono text-white/40">
        period (days, log) &uarr; &middot; time &rarr;
      </div>
    </div>
  )
}

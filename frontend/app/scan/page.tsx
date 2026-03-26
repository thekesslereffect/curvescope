"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import {
  getSectors,
  getScanStatus,
  startScan,
  stopScan,
  type ScanStatus,
} from "@/lib/api"
import { extractApiErrorMessage } from "@/lib/api-errors"
import { APP_NAME } from "@/lib/brand"

const PIPELINE_STEPS = [
  { key: "resolve", label: "Resolve target", match: "resolving" },
  { key: "download", label: "Download light curve", match: "downloading light curve" },
  { key: "clean", label: "Clean & detrend", match: "cleaning" },
  { key: "autoencoder", label: "Autoencoder scoring", match: "autoencoder" },
  { key: "bls", label: "BLS periodogram", match: "bls" },
  { key: "wavelet", label: "Wavelet transform", match: "wavelet" },
  { key: "centroid", label: "Centroid / TPF", match: "centroid" },
  { key: "classify", label: "Classify events", match: "classifying" },
  { key: "techno", label: "Technosignature catalogs", match: "technosignature" },
]

function PipelinePhases({ currentPhase }: { currentPhase: string }) {
  const [highWater, setHighWater] = useState(-1)

  const liveIdx = currentPhase
    ? PIPELINE_STEPS.findIndex((s) => currentPhase.toLowerCase().includes(s.match))
    : -1

  useEffect(() => {
    if (liveIdx > highWater) setHighWater(liveIdx)
  }, [liveIdx, highWater])

  useEffect(() => {
    setHighWater(-1)
  }, [currentPhase?.startsWith("Resolving")])

  const activeIdx = liveIdx >= 0 ? liveIdx : highWater

  return (
    <div className="flex flex-wrap gap-x-1.5 gap-y-1.5 items-center">
      {PIPELINE_STEPS.map((step, i) => {
        const done = i < activeIdx
        const active = i === activeIdx
        return (
          <div key={step.key} className="flex items-center gap-1">
            {i > 0 && <span className="text-muted-foreground text-[8px]">&rsaquo;</span>}
            <span
              className={`text-[10px] px-2 py-0.5 rounded-lg transition-colors duration-200 ${
                active
                  ? "bg-blue-500/15 text-blue-400 ring-1 ring-blue-500/30"
                  : done
                  ? "bg-emerald-500/10 text-emerald-400/80"
                  : "text-muted-foreground"
              }`}
            >
              {done && "✓ "}
              {active && "● "}
              {step.label}
            </span>
          </div>
        )
      })}
    </div>
  )
}

export default function ScanPage() {
  const [sectors, setSectors] = useState<number[]>([])
  const [sector, setSector] = useState(1)
  const [limit, setLimit] = useState("")
  const [skipExisting, setSkipExisting] = useState(true)
  const [status, setStatus] = useState<ScanStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getSectors().then((r) => setSectors(r.sectors)).catch(() => {})
  }, [])

  const isRunning = status?.running ?? false
  useEffect(() => {
    const interval = isRunning ? 600 : 3000
    const id = setInterval(() => {
      getScanStatus().then(setStatus).catch(() => {})
    }, interval)
    getScanStatus().then(setStatus).catch(() => {})
    return () => clearInterval(id)
  }, [isRunning])

  const onStart = async () => {
    setError(null)
    try {
      await startScan(sector, limit ? parseInt(limit, 10) : null, skipExisting)
      const s = await getScanStatus()
      setStatus(s)
    } catch (e: unknown) {
      setError(extractApiErrorMessage(e))
    }
  }

  const onStop = async () => {
    setError(null)
    try {
      await stopScan()
      setStatus(await getScanStatus())
    } catch {
      setError("Stop failed")
    }
  }

  const pct =
    status && status.total > 0
      ? Math.round((100 * status.completed) / status.total)
      : 0

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <p className="label-upper mb-1.5">{APP_NAME} · Scan</p>
      <h1 className="text-xl font-semibold mb-2">Sector scan</h1>
      <p className="text-sm text-dim mb-8 leading-relaxed">
        Discover every TESS light-curve target in a sector via MAST, then run {APP_NAME}&apos;s full pipeline on each
        (downloads cache to your data directory). Full sectors can take hours—set a max target count for dry runs.
      </p>

      <div className="card-surface p-6 mb-8 space-y-4">
        <div className="flex flex-wrap gap-4 items-end">
          <div>
            <label className="block label-upper mb-1.5">Sector</label>
            <select
              value={sector}
              onChange={(e) => setSector(parseInt(e.target.value, 10))}
              disabled={!!status?.running}
              className="input-field text-sm min-w-[120px]"
            >
              {(sectors.length ? sectors : Array.from({ length: 87 }, (_, i) => i + 1)).map((n) => (
                <option key={n} value={n}>
                  Sector {n}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block label-upper mb-1.5">Max targets (optional)</label>
            <input
              type="number"
              min={1}
              placeholder="All in sector"
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
              disabled={!!status?.running}
              className="input-field text-sm w-40"
            />
          </div>
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={skipExisting}
              onChange={(e) => setSkipExisting(e.target.checked)}
              disabled={!!status?.running}
              className="accent-blue-500 w-3.5 h-3.5"
            />
            <span className="text-xs text-dim">Skip already analyzed</span>
          </label>
          <button
            type="button"
            onClick={onStart}
            disabled={!!status?.running}
            className="btn-primary disabled:opacity-40 text-sm"
          >
            Start scan
          </button>
          <button
            type="button"
            onClick={onStop}
            disabled={!status?.running}
            className="btn-destructive disabled:opacity-40 text-sm"
          >
            Stop
          </button>
        </div>
        {error && <p className="text-sm text-red-400">{error}</p>}
      </div>

      {status && (
        <div className="card-surface p-6 mb-8">
          <p className="label-upper mb-2">Status</p>
          <p className="text-sm text-dim mb-2">{status.message}</p>
          <p className="text-xs text-muted-foreground mb-3 font-mono">
            {status.running ? "Running" : "Idle"} · sector {status.sector ?? "—"} · {status.completed} / {status.total}{" "}
            done
            {status.skipped > 0 && ` · ${status.skipped} skipped`}
            {status.current_tic && ` · current TIC ${status.current_tic}`}
          </p>

          <div className="h-2 bg-accent rounded-full overflow-hidden mb-1">
            <div
              className="h-full bg-foreground rounded-full transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
          <p className="text-[10px] text-muted-foreground font-mono mb-4">{pct}% overall</p>

          {status.running && status.current_tic && (
            <div className="card-surface bg-background/50 p-3 mb-4">
              <p className="label-upper mb-2">
                Current target · TIC {status.current_tic}
              </p>
              <PipelinePhases currentPhase={status.current_phase} />
            </div>
          )}

          {status.errors.length > 0 && (
            <div className="mt-4 max-h-32 overflow-y-auto text-xs font-mono text-red-400/90">
              {status.errors.slice(-10).map((err, i) => (
                <div key={`${err.tic ?? "?"}-${i}`}>
                  {err.tic || "?"}: {err.error}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {status && status.results_preview.length > 0 && (
        <div>
          <p className="label-upper mb-2">Top scores this scan</p>
          <div className="card-surface divide-y divide-border overflow-hidden">
            {status.results_preview.map((r) => (
              <Link
                key={r.analysis_id}
                href={r.tic_id ? `/analyze/${encodeURIComponent(r.tic_id)}` : "#"}
                className="flex justify-between px-4 py-2.5 hover:bg-white/5 text-xs font-mono transition-colors"
              >
                <span className="text-blue-400">TIC {r.tic_id}</span>
                <span className="text-muted-foreground">
                  anomaly {r.anomaly_score?.toFixed(3) ?? "—"} · techno {r.technosignature_score?.toFixed(3) ?? "—"} · flags{" "}
                  {r.flag_count ?? 0}
                </span>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

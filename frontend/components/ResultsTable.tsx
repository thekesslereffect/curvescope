"use client"

import Link from "next/link"
import type { AnalysisSummary } from "@/lib/api"

export const EVENT_TYPES = [
  "",
  "transit",
  "unknown",
  "exocomet",
  "asymmetric",
  "depth_anomaly",
  "non_periodic",
  "stellar_flare",
  "stellar_spot",
  "eclipsing_binary",
  "stellar_variability",
  "systematic",
  "contamination",
]

function hrefForRow(row: AnalysisSummary) {
  const id = row.tic_id || row.common_name || String(row.id)
  return `/analyze/${encodeURIComponent(id)}`
}

export function ResultsTable({
  items,
  loading,
}: {
  items: AnalysisSummary[]
  loading?: boolean
}) {
  if (loading) {
    return (
      <div className="card-surface p-12 text-center text-sm text-muted-foreground">
        Loading analyses…
      </div>
    )
  }

  if (items.length === 0) {
    return (
      <div className="card-surface p-12 text-center text-sm text-muted-foreground">
        No analyses yet. Run a sector scan or search for a target.
      </div>
    )
  }

  return (
    <div className="card-surface overflow-hidden overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="bg-card text-muted-foreground uppercase tracking-wider">
          <tr>
            <th className="px-4 py-2.5 font-medium">TIC / name</th>
            <th className="px-4 py-2.5 text-right font-medium">Anomaly</th>
            <th className="px-4 py-2.5 text-right font-medium">Techno</th>
            <th className="px-4 py-2.5 text-right font-medium">Flags</th>
            <th className="px-4 py-2.5 font-medium">Sector</th>
            <th className="px-4 py-2.5 font-medium">When</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/50">
          {items.map((row) => (
            <tr key={row.id} className="hover:bg-white/5 transition-colors">
              <td className="px-4 py-2.5">
                <Link href={hrefForRow(row)} className="text-blue-400 hover:text-blue-300 transition-colors">
                  {row.tic_id ? `TIC ${row.tic_id}` : "—"}
                  {row.common_name && (
                    <span className="text-muted-foreground ml-2">{row.common_name}</span>
                  )}
                </Link>
              </td>
              <td className="px-4 py-2.5 text-right tabular-nums font-mono text-dim">
                {row.anomaly_score != null ? row.anomaly_score.toFixed(3) : "—"}
              </td>
              <td className="px-4 py-2.5 text-right tabular-nums font-mono text-dim">
                {row.technosignature_score.toFixed(3)}
              </td>
              <td className="px-4 py-2.5 text-right font-mono text-dim">{row.flag_count}</td>
              <td className="px-4 py-2.5 text-muted-foreground">{row.sector ?? "—"}</td>
              <td className="px-4 py-2.5 text-muted-foreground">
                {row.created_at ? new Date(row.created_at).toLocaleString() : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

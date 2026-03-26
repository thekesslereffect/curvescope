"use client"

import { useCallback, useEffect, useState } from "react"
import { SearchBar } from "@/components/SearchBar"
import { ResultsTable, EVENT_TYPES } from "@/components/ResultsTable"
import { getAnalyses, deleteAllAnalyses, type AnalysesResponse } from "@/lib/api"
import { extractApiErrorMessage } from "@/lib/api-errors"
import { APP_NAME, APP_ONE_LINER } from "@/lib/brand"

export default function DashboardPage() {
  const [data, setData] = useState<AnalysesResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [sortBy, setSortBy] = useState<"anomaly_score" | "technosignature_score" | "created_at">("anomaly_score")
  const [eventType, setEventType] = useState("")
  const [minScore, setMinScore] = useState("")
  const [search, setSearch] = useState("")
  const [debouncedSearch, setDebouncedSearch] = useState("")

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search.trim()), 400)
    return () => clearTimeout(t)
  }, [search])

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const res = await getAnalyses({
        page,
        page_size: 50,
        sort_by: sortBy,
        event_type: eventType || undefined,
        min_score: minScore ? parseFloat(minScore) : 0,
        search: debouncedSearch || undefined,
      })
      setData(res)
      setError(null)
    } catch (e) {
      setData(null)
      setError(extractApiErrorMessage(e))
    } finally {
      setLoading(false)
    }
  }, [page, sortBy, eventType, minScore, debouncedSearch])

  useEffect(() => {
    load()
  }, [load])

  const sum = data?.summary

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <div className="mb-8">
        <p className="label-upper mb-1.5">{APP_NAME} · Dashboard</p>
        <h1 className="text-2xl font-semibold tracking-tight mb-2">Analyzed targets</h1>
        <p className="text-sm text-dim max-w-2xl leading-relaxed">
          {APP_ONE_LINER} Open any row for light curves, periodograms, and flags—or run{" "}
          <strong className="text-foreground font-medium">Scan</strong> to queue a whole TESS sector from MAST.
        </p>
      </div>

      {error && (
        <div className="mb-6 card-surface bg-red-500/5 border-red-500/20 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {sum && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
          {[
            { label: "Complete analyses", value: String(sum.total_complete) },
            { label: "Interesting events", value: String(sum.interesting_event_count) },
            { label: "Max anomaly score", value: sum.max_anomaly_score.toFixed(3) },
            { label: "Max technosignature", value: sum.max_technosignature.toFixed(3) },
          ].map((c) => (
            <div key={c.label} className="card-surface p-4">
              <p className="label-upper mb-1.5">{c.label}</p>
              <p className="text-lg font-semibold font-mono text-foreground">{c.value}</p>
            </div>
          ))}
        </div>
      )}

      <div className="mb-6 flex flex-col gap-4">
        <SearchBar />
        <div className="flex flex-wrap gap-3 items-end">
          <div>
            <label className="block label-upper mb-1.5">Sort</label>
            <select
              value={sortBy}
              onChange={(e) => {
                setPage(1)
                setSortBy(e.target.value as typeof sortBy)
              }}
              className="input-field text-xs min-w-[140px]"
            >
              <option value="anomaly_score">Anomaly score</option>
              <option value="technosignature_score">Technosignature</option>
              <option value="created_at">Date</option>
            </select>
          </div>
          <div>
            <label className="block label-upper mb-1.5">Event type</label>
            <select
              value={eventType}
              onChange={(e) => {
                setPage(1)
                setEventType(e.target.value)
              }}
              className="input-field text-xs min-w-[140px]"
            >
              {EVENT_TYPES.map((et) => (
                <option key={et || "any"} value={et}>
                  {et || "Any"}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block label-upper mb-1.5">Min anomaly</label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              placeholder="0"
              value={minScore}
              onChange={(e) => {
                setPage(1)
                setMinScore(e.target.value)
              }}
              className="input-field text-xs w-24"
            />
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="block label-upper mb-1.5">Filter table (TIC / name)</label>
            <input
              type="text"
              value={search}
              onChange={(e) => {
                setPage(1)
                setSearch(e.target.value)
              }}
              placeholder="Substring match…"
              className="input-field text-xs w-full"
            />
          </div>
          <button type="button" onClick={() => load()} className="btn-secondary text-xs">
            Refresh
          </button>
          <button
            type="button"
            onClick={async () => {
              if (!confirm("Delete ALL analyses and events? Targets are kept. This cannot be undone.")) return
              try {
                await deleteAllAnalyses()
                setError(null)
                setData(null)
                load()
              } catch (e) {
                setError(extractApiErrorMessage(e))
              }
            }}
            className="btn-destructive text-xs"
          >
            Clear all
          </button>
        </div>
      </div>

      <ResultsTable items={data?.items ?? []} loading={loading} />

      {data && data.total > data.page_size && (
        <div className="flex justify-between items-center mt-4 text-xs text-muted-foreground">
          <span className="font-mono">
            Page {data.page} of {Math.ceil(data.total / data.page_size)} ({data.total} total)
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="btn-secondary text-xs disabled:opacity-40"
            >
              Prev
            </button>
            <button
              type="button"
              disabled={page >= Math.ceil(data.total / data.page_size)}
              onClick={() => setPage((p) => p + 1)}
              className="btn-secondary text-xs disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

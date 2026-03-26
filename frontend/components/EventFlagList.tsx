import type { FlaggedEvent } from "@/lib/types"
import { StatusBadge } from "./StatusBadge"

export function EventFlagList({ events }: { events: FlaggedEvent[] }) {
  if (events.length === 0) return null

  return (
    <div className="card-surface overflow-hidden">
      <div className="px-4 py-3 border-b border-border/60 bg-card">
        <p className="label-upper">
          Flagged Events ({events.length})
        </p>
      </div>
      <div className="divide-y divide-border/40">
        {events.map((ev) => (
          <div key={ev.id} className="px-4 py-3 hover:bg-white/5 transition-colors">
            <div className="flex items-start justify-between gap-3 mb-1.5">
              <div className="flex items-center gap-2">
                <StatusBadge type={ev.event_type} />
                <span className="text-xs font-mono text-muted-foreground">
                  t={ev.time_center.toFixed(2)} BTJD
                </span>
              </div>
              <span className="text-xs font-mono font-semibold text-dim">
                score {ev.anomaly_score.toFixed(3)}
              </span>
            </div>
            <div className="flex gap-4 text-[11px] font-mono text-muted-foreground mb-1">
              <span>depth {ev.depth_ppm.toFixed(0)} ppm</span>
              <span>{ev.duration_hours.toFixed(1)}h</span>
              <span>conf {ev.confidence.toFixed(2)}</span>
              {ev.centroid_shift_arcsec >= 0 && (
                <span>centroid {ev.centroid_shift_arcsec.toFixed(1)}&quot;</span>
              )}
            </div>
            <p className="text-xs text-dim leading-relaxed">{ev.notes}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

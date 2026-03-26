import type { EventType } from "@/lib/types"
import { EVENT_LABELS, EVENT_COLORS } from "@/lib/types"

export function StatusBadge({ type }: { type: EventType }) {
  const color = EVENT_COLORS[type] || "#6b7280"
  const label = EVENT_LABELS[type] || type

  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-lg text-[11px] font-medium"
      style={{ backgroundColor: `${color}15`, color, border: `1px solid ${color}30` }}
    >
      {label}
    </span>
  )
}

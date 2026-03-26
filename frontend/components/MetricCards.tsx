import type { Analysis } from "@/lib/types"

export function MetricCards({ analysis }: { analysis: Analysis }) {
  const cards = [
    {
      label: "Anomaly Score",
      value: analysis.anomaly_score?.toFixed(3) ?? "—",
      highlight: (analysis.anomaly_score ?? 0) > 0.7,
    },
    {
      label: "Best Period",
      value: analysis.known_period ? `${analysis.known_period.toFixed(3)}d` : "—",
      highlight: false,
    },
    {
      label: "Flagged Events",
      value: String(analysis.flag_count),
      highlight: analysis.flag_count > 0,
    },
    {
      label: "Technosignature",
      value: analysis.technosignature?.composite_score?.toFixed(3) ?? "—",
      highlight: (analysis.technosignature?.composite_score ?? 0) > 0.3,
    },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
      {cards.map((c) => (
        <div
          key={c.label}
          className={`card-surface p-4 ${
            c.highlight ? "border-red-500/20 bg-red-500/5" : ""
          }`}
        >
          <p className="label-upper mb-1.5">{c.label}</p>
          <p className={`text-lg font-semibold font-mono ${c.highlight ? "text-red-400" : "text-foreground"}`}>
            {c.value}
          </p>
        </div>
      ))}
    </div>
  )
}

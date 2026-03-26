export type AnalysisStatus = "pending" | "running" | "complete" | "failed"

export type EventType =
  | "transit"
  | "asymmetric"
  | "depth_anomaly"
  | "non_periodic"
  | "exocomet"
  | "stellar_flare"
  | "stellar_spot"
  | "eclipsing_binary"
  | "stellar_variability"
  | "systematic"
  | "contamination"
  | "unknown"

export interface Target {
  id: number
  tic_id: string
  common_name: string | null
  ra: number | null
  dec: number | null
  magnitude: number | null
  stellar_type: string | null
}

export interface WaveletSystematic {
  name: string
  period_days: number
  relative_power: number
}

export interface WaveletResult {
  time: number[]
  periods: number[]
  power: number[][]
  tess_systematic_periods: WaveletSystematic[]
  dominant_periods: number[]
}

export interface TPFData {
  available: boolean
  time?: number[]
  flux?: number[][][]
  n_rows?: number
  n_cols?: number
  n_frames?: number
  aperture_mask?: number[][]
  column?: number
  row?: number
}

export interface CentroidResult {
  available: boolean
  time?: number[]
  col?: number[]
  row?: number[]
  col_baseline?: number
  row_baseline?: number
  displacement_arcsec?: number[]
  max_shift_arcsec?: number
  rms_shift_arcsec?: number
  shift_flagged?: boolean
}

export interface Hypothesis {
  id: string
  name: string
  category: "natural" | "artificial"
  description: string
  score: number
  reasoning: string[]
}

export interface TechnosignatureResult {
  ran: boolean
  reason?: string
  composite_score: number
  morphology?: { score: number; events: unknown[]; note: string }
  timing_entropy?: { score: number; entropy: number | null; n_events: number; note: string }
  ir_excess?: { score: number; available: boolean; note: string }
  catalog_membership?: { score: number; available: boolean; simbad_type: string | null; note: string }
  summary?: string
  hypotheses?: Hypothesis[]
}

export interface FlaggedEvent {
  id: number
  event_type: EventType
  time_center: number
  duration_hours: number
  depth_ppm: number
  anomaly_score: number
  confidence: number
  notes: string
  centroid_shift_arcsec: number
  systematic_match: string | null
}

export interface LossEntry {
  epoch: number
  loss: number
  val_loss?: number
}

export interface Analysis {
  id: number
  target: Target | null
  sector: string
  status: AnalysisStatus
  anomaly_score: number | null
  known_period: number | null
  flag_count: number
  raw_flux: { time: number[]; flux: number[] } | null
  detrended_flux: { time: number[]; flux: number[] } | null
  score_timeline: { time: number[]; score: number[] } | null
  periodogram: { period: number[]; power: number[] } | null
  wavelet: WaveletResult | null
  centroid: CentroidResult | null
  tpf_data: TPFData | null
  technosignature: TechnosignatureResult | null
  events: FlaggedEvent[]
  error_message: string | null
  created_at: string | null
}

export const EVENT_LABELS: Record<EventType, string> = {
  transit: "Transit",
  asymmetric: "Asymmetric",
  depth_anomaly: "Depth anomaly",
  non_periodic: "Non-periodic",
  exocomet: "Exocomet",
  stellar_flare: "Stellar flare",
  stellar_spot: "Stellar spot",
  eclipsing_binary: "Eclipsing binary",
  stellar_variability: "Stellar variability",
  systematic: "Systematic",
  contamination: "Contamination",
  unknown: "Unknown",
}

export const EVENT_COLORS: Record<EventType, string> = {
  transit: "#3b82f6",
  asymmetric: "#ef4444",
  depth_anomaly: "#ef4444",
  non_periodic: "#f59e0b",
  exocomet: "#22c55e",
  stellar_flare: "#f59e0b",
  stellar_spot: "#6b7280",
  eclipsing_binary: "#a855f7",
  stellar_variability: "#ec4899",
  systematic: "#9ca3af",
  contamination: "#9ca3af",
  unknown: "#dc2626",
}

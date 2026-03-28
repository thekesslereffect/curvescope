import axios from "axios"
import type { Analysis } from "./types"

/** Match `npm run dev:backend` (127.0.0.1). Override in frontend/.env.local if needed. */
const API_BASE =
  typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_URL
    ? process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "")
    : "http://127.0.0.1:8000/api"

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30_000,
})

export interface AnalysisSummary {
  id: number
  status: "pending" | "running" | "complete" | "failed"
  sector: string | null
  anomaly_score: number | null
  technosignature_score: number
  known_period: number | null
  flag_count: number
  tic_id: string | null
  common_name: string | null
  created_at: string | null
  error_message: string | null
}

export interface AnalysesResponse {
  items: AnalysisSummary[]
  total: number
  page: number
  page_size: number
  summary: {
    total_complete: number
    interesting_event_count: number
    max_technosignature: number
    max_anomaly_score: number
  }
}

export async function getAnalyses(params?: {
  page?: number
  page_size?: number
  sort_by?: "anomaly_score" | "technosignature_score" | "created_at"
  event_type?: string
  min_score?: number
  search?: string
  include_failed?: boolean
}): Promise<AnalysesResponse> {
  const { data } = await client.get("/analyses", { params })
  return data
}

export async function deleteAllAnalyses() {
  const { data } = await client.delete("/analyses")
  return data as { ok: boolean; deleted_analyses: number; deleted_events: number }
}

export async function getLatestAnalysis(identifier: string): Promise<Analysis | null> {
  const res = await client.get("/analysis/latest", {
    params: { identifier },
    // First visit has no saved run — 404 is normal; avoids red console / failed resource noise
    validateStatus: (s) => s === 200 || s === 404,
  })
  if (res.status === 404) return null
  return res.data as Analysis
}

export async function startAnalysis(identifier: string, sector = "all") {
  const { data } = await client.post("/analyze", { identifier, sector })
  return data as { analysis_id: number; status: string }
}

export async function getAnalysis(id: number): Promise<Analysis> {
  const { data } = await client.get(`/analysis/${id}`)
  return data
}

export async function pollAnalysis(
  id: number,
  onUpdate: (a: Analysis) => void,
  intervalMs = 2000
): Promise<Analysis> {
  const pollOnce = async () => {
    const analysis = await getAnalysis(id)
    onUpdate(analysis)
    return analysis
  }

  let analysis = await pollOnce()
  if (analysis.status === "complete") return analysis
  if (analysis.status === "failed") {
    throw new Error(analysis.error_message || "Analysis failed")
  }

  return new Promise((resolve, reject) => {
    let polling = false
    const timer = setInterval(async () => {
      if (polling) return
      polling = true
      try {
        analysis = await pollOnce()
        if (analysis.status === "complete") {
          clearInterval(timer)
          resolve(analysis)
        } else if (analysis.status === "failed") {
          clearInterval(timer)
          reject(new Error(analysis.error_message || "Analysis failed"))
        }
      } catch (e) {
        clearInterval(timer)
        reject(e)
      } finally {
        polling = false
      }
    }, intervalMs)
  })
}

export async function getTargets() {
  const { data } = await client.get("/targets")
  return data
}

export async function getEvents(params?: { event_type?: string; min_score?: number; limit?: number }) {
  const { data } = await client.get("/events", { params })
  return data
}

export interface ScanStatus {
  running: boolean
  sector: number | null
  total: number
  completed: number
  skipped: number
  current_tic: string | null
  current_phase: string
  errors: { tic: string; error: string }[]
  results_preview: {
    analysis_id: number
    tic_id: string | null
    anomaly_score: number | null
    technosignature_score: number
    flag_count: number | null
  }[]
  message: string
}

export async function getSectors() {
  const { data } = await client.get("/sectors")
  return data as { sectors: number[]; note?: string }
}

export async function startScan(sector: number, limit?: number | null, skipExisting?: boolean) {
  const { data } = await client.post("/scan/start", {
    sector,
    limit: limit ?? null,
    skip_existing: skipExisting ?? false,
  })
  return data as { ok: boolean; message: string }
}

export async function getScanStatus(): Promise<ScanStatus> {
  const { data } = await client.get("/scan/status")
  return data
}

export async function stopScan() {
  const { data } = await client.post("/scan/stop")
  return data as { ok: boolean; message: string }
}

export interface AppSettings {
  data_dir: string
  mast_cache_dir: string
  model_weights_dir: string
  database_url: string
  database_size_bytes: number
  model_weights_exist: boolean
  model_stats_exist: boolean
  counts: { targets: number; analyses: number; events: number }
  gpu: {
    cuda_available: boolean
    name: string | null
    vram_total_bytes: number | null
    vram_free_bytes: number | null
    ready?: boolean
  }
}

export async function getAppSettings(): Promise<AppSettings> {
  const { data } = await client.get("/settings", { timeout: 8_000 })
  return data
}

export async function updateDataDir(path: string) {
  const { data } = await client.put("/settings/data-dir", { path })
  return data as { ok: boolean; data_dir: string; message: string }
}

export interface TrainDefaults {
  epochs: number
  batch_size: number
  learning_rate: number
  max_targets: number | null
  available_training_targets: number
}

export async function getTrainDefaults(): Promise<TrainDefaults> {
  const { data } = await client.get("/train/defaults")
  return data
}

export interface TrainingTarget {
  tic_id: string
  anomaly_score: number | null
  source: string
}

export async function getTrainingTargets() {
  const { data } = await client.get("/train/targets")
  return data as { count: number; targets: TrainingTarget[] }
}

export async function addTrainingTargets(targets: { tic_id: string; anomaly_score?: number | null; source?: string }[]) {
  const { data } = await client.post("/train/targets", { targets })
  return data as { ok: boolean; added: number; total: number }
}

export async function removeTrainingTargets(ticIds: string[]) {
  const { data } = await client.delete("/train/targets", { data: { tic_ids: ticIds } })
  return data as { ok: boolean; removed: number; total: number }
}

export async function importQuietStarsToTargets(maxScore = 0.25, maxFlags = 0, limit = 200) {
  const { data } = await client.post("/train/targets/import-quiet", null, {
    params: { max_score: maxScore, max_flags: maxFlags, limit },
  })
  return data as { ok: boolean; added: number; total: number; scanned: number }
}

export async function getQuietStars(maxScore = 0.25, maxFlags = 0) {
  const { data } = await client.get("/train/quiet-stars", {
    params: { max_score: maxScore, max_flags: maxFlags },
  })
  return data as {
    count: number
    max_score: number
    max_flags: number
    stars: { tic_id: string; anomaly_score: number; flag_count: number }[]
  }
}

export async function startTraining(opts: {
  epochs: number
  batch_size: number
  learning_rate: number
  max_targets: number | null
  use_quiet_stars?: boolean
  quiet_max_score?: number
  quiet_max_flags?: number
}) {
  const { data } = await client.post("/train", opts)
  return data as { ok: boolean; message: string }
}

export interface ReconstructionSample {
  label: "best" | "typical" | "worst"
  original: number[]
  reconstructed: number[]
  error: number
}

export interface ErrorHistogram {
  bin_centers: number[]
  counts: number[]
  mean: number
  std: number
  p99: number
  total_windows: number
}

export interface ActivationLayer {
  name: string
  shape: number[]
  data: number[][] | number[]
  type: "heatmap" | "bar"
}

export interface TrainStatus {
  running: boolean
  phase: string
  epoch: number
  total_epochs: number
  loss: number | null
  targets_fetched: number
  total_targets: number
  windows_count: number
  message: string
  error: string | null
  hyperparams?: {
    epochs: number
    batch_size: number
    learning_rate: number
    max_targets: number | null
  }
  loss_history?: { epoch: number; loss: number; val_loss?: number }[]
  reconstruction_samples?: ReconstructionSample[]
  error_histogram?: ErrorHistogram
  network_activations?: ActivationLayer[]
}

export async function getTrainStatus(): Promise<TrainStatus> {
  const { data } = await client.get("/train/status")
  return data as TrainStatus
}

export function getExportUrl(type: "analyses" | "events", sector?: number | null): string {
  const params = new URLSearchParams()
  if (sector != null) params.set("sector", String(sector))
  return `${API_BASE}/export/${type}${params.toString() ? "?" + params : ""}`
}

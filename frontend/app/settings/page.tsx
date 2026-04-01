"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import {
  getAppSettings,
  updateDataDir,
  clearMastCache,
  startTraining,
  getTrainStatus,
  getTrainDefaults,
  getQuietStars,
  getTrainingTargets,
  removeTrainingTargets,
  importQuietStarsToTargets,
  type AppSettings,
  type TrainStatus,
  type TrainingTarget,
} from "@/lib/api"
import { extractApiErrorMessage } from "@/lib/api-errors"
import { TrainingLossChart } from "@/components/TrainingLossChart"
import { ReconstructionChart } from "@/components/ReconstructionChart"
import { ErrorHistogramChart } from "@/components/ErrorHistogramChart"
import { NetworkFlowChart } from "@/components/NetworkFlowChart"
import { APP_DESCRIPTION, APP_NAME, APP_TAGLINE } from "@/lib/brand"

function fmtBytes(n: number | null | undefined, fallback = "—") {
  if (n == null) return fallback
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

function Hint({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <p className={["text-[10px] text-muted-foreground leading-snug mt-1.5", className].filter(Boolean).join(" ")}>
      {children}
    </p>
  )
}

export default function SettingsPage() {
  const [s, setS] = useState<AppSettings | null>(null)
  const [train, setTrain] = useState<TrainStatus | null>(null)
  const [dataDirInput, setDataDirInput] = useState("")
  const [msg, setMsg] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const refreshedWeightsAfterRun = useRef(false)
  const [trainEpochs, setTrainEpochs] = useState(50)
  const [trainBatch, setTrainBatch] = useState(64)
  const [trainLr, setTrainLr] = useState("0.001")
  const [trainMaxTargets, setTrainMaxTargets] = useState("")
  const [availableTrainStars, setAvailableTrainStars] = useState(25)
  const [targets, setTargets] = useState<TrainingTarget[]>([])
  const [targetsLoading, setTargetsLoading] = useState(false)
  const [targetsExpanded, setTargetsExpanded] = useState(false)
  const [selectedForRemoval, setSelectedForRemoval] = useState<Set<string>>(new Set())
  const [importingQuiet, setImportingQuiet] = useState(false)
  const [quietAvailable, setQuietAvailable] = useState<number | null>(null)
  const [settingsLoad, setSettingsLoad] = useState<"loading" | "ok" | "error">("loading")
  const [settingsLoadError, setSettingsLoadError] = useState<string | null>(null)
  const [trainStarting, setTrainStarting] = useState(false)
  const [clearingCache, setClearingCache] = useState(false)

  const load = useCallback(async () => {
    setSettingsLoad("loading")
    setSettingsLoadError(null)
    try {
      const app = await getAppSettings()
      setS(app)
      setDataDirInput(app.data_dir ?? "")
      setSettingsLoad("ok")
    } catch (e) {
      setS(null)
      setSettingsLoad("error")
      setSettingsLoadError(extractApiErrorMessage(e))
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    let pollId: ReturnType<typeof setInterval> | null = null

    async function init() {
      for (let attempt = 0; attempt < 6; attempt++) {
        if (cancelled) return
        try {
          const app = await getAppSettings()
          if (cancelled) return
          setS(app)
          setDataDirInput(app.data_dir ?? "")
          setSettingsLoad("ok")
          break
        } catch (e) {
          if (attempt < 5) {
            setSettingsLoadError(`Connecting to backend (attempt ${attempt + 1}/6)...`)
            await new Promise((r) => setTimeout(r, 2000))
          } else {
            setS(null)
            setSettingsLoad("error")
            setSettingsLoadError(extractApiErrorMessage(e))
          }
        }
      }

      if (cancelled) return

      try {
        const d = await getTrainDefaults()
        if (!cancelled) {
          setTrainEpochs(d.epochs)
          setTrainBatch(d.batch_size)
          setTrainLr(String(d.learning_rate))
          setAvailableTrainStars(d.available_training_targets)
          setTrainMaxTargets(d.max_targets != null ? String(d.max_targets) : "")
        }
      } catch {}

      if (cancelled) return

      try {
        const t = await getTrainStatus()
        if (!cancelled) setTrain(t)
      } catch {}

      pollId = setInterval(async () => {
        try {
          const t = await getTrainStatus()
          if (!cancelled) setTrain(t)
        } catch {}
      }, 2000)
    }

    void init()
    return () => {
      cancelled = true
      if (pollId != null) clearInterval(pollId)
    }
  }, [])

  const loadTargets = useCallback(async () => {
    setTargetsLoading(true)
    try {
      const r = await getTrainingTargets()
      setTargets(r.targets)
      setAvailableTrainStars(r.count)
    } catch {}
    try {
      const q = await getQuietStars(0.25, 0)
      setQuietAvailable(q.count)
    } catch {
      setQuietAvailable(null)
    }
    setTargetsLoading(false)
  }, [])

  useEffect(() => { void loadTargets() }, [loadTargets])

  useEffect(() => {
    if (train?.running) {
      refreshedWeightsAfterRun.current = false
      return
    }
    if (train?.phase === "complete" && !refreshedWeightsAfterRun.current) {
      refreshedWeightsAfterRun.current = true
      void load()
    }
  }, [train?.phase, train?.running, load])

  const saveDataDir = async () => {
    setErr(null)
    setMsg(null)
    try {
      const r = await updateDataDir(dataDirInput.trim())
      setMsg(r.message)
      await load()
    } catch (e: unknown) {
      setErr(extractApiErrorMessage(e))
    }
  }

  const onTrain = async () => {
    setErr(null)
    setMsg(null)
    const lr = Number(trainLr)
    if (!Number.isFinite(lr) || lr <= 0) {
      setErr("Learning rate must be a positive number (e.g. 0.001 or 1e-4)")
      return
    }
    const maxT = trainMaxTargets.trim()
    let maxTargetsParsed: number | null = null
    if (maxT !== "") {
      const n = parseInt(maxT, 10)
      if (!Number.isFinite(n) || n < 1) {
        setErr("Max training stars: leave empty for all, or enter 1-" + availableTrainStars)
        return
      }
      maxTargetsParsed = Math.min(n, availableTrainStars)
    }
    setTrainStarting(true)
    try {
      const r = await startTraining({
        epochs: trainEpochs,
        batch_size: trainBatch,
        learning_rate: lr,
        max_targets: maxTargetsParsed,
      })
      setMsg(r.message ?? "Training started")
      setTrain(await getTrainStatus())
    } catch (e: unknown) {
      setErr(extractApiErrorMessage(e))
    } finally {
      setTrainStarting(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <p className="label-upper mb-1.5">
        {APP_NAME} &middot; {APP_TAGLINE}
      </p>
      <h1 className="text-xl font-semibold mb-2">Settings</h1>
      <p className="text-sm text-dim mb-3 leading-relaxed max-w-2xl">{APP_DESCRIPTION}</p>
      <p className="text-xs text-muted-foreground mb-8 leading-relaxed max-w-2xl">
        Configure where data lives, inspect GPU status, and train the autoencoder. Changing the data directory moves
        the SQLite database and MAST cache root; the API reconnects immediately.
      </p>

      {settingsLoad === "error" && settingsLoadError && (
        <div className="mb-6 card-surface border-amber-500/30 bg-amber-500/5 px-4 py-3 text-sm text-amber-200 space-y-2">
          <p>
            <strong>Could not load settings from the API.</strong> {settingsLoadError}
          </p>
          <button type="button" onClick={() => void load()} className="btn-secondary text-xs">
            Retry
          </button>
        </div>
      )}

      {(msg || err) && (
        <div
          className={`mb-6 card-surface px-4 py-3 text-sm ${
            err ? "border-red-500/30 bg-red-500/5 text-red-400" : "border-emerald-500/20 bg-emerald-500/5 text-emerald-400"
          }`}
        >
          {err || msg}
        </div>
      )}

      <section className="card-surface p-6 mb-6">
        <h2 className="label-upper mb-4">Data directory</h2>
        <div className="flex gap-2 flex-wrap items-start">
          <div className="flex-1 min-w-[240px]">
            <input
              type="text"
              value={dataDirInput}
              onChange={(e) => setDataDirInput(e.target.value)}
              disabled={settingsLoad === "loading"}
              placeholder={settingsLoad === "loading" ? "Loading from API..." : "e.g. E:\\tess-data"}
              title="Root folder for SQLite DB, MAST/lightkurve cache, and model weights"
              className="input-field w-full disabled:opacity-60"
            />
            <Hint>
              Root folder for everything heavy: <span className="text-dim">tess_anomaly.db</span>, MAST downloads,
              and <span className="text-dim">weights/autoencoder_v1.pt</span>. Use a large drive (e.g.{" "}
              <code className="text-dim">E:\tess-data</code>). Persists to <code className="text-dim">backend/.env</code>{" "}
              as <code className="text-dim">DATA_DIR</code>.
            </Hint>
          </div>
          <button type="button" onClick={saveDataDir} className="btn-primary shrink-0 text-sm">
            Update
          </button>
        </div>
        {s && (
          <div className="mt-4 space-y-3">
            <ul className="text-xs font-mono text-muted-foreground space-y-1">
              <li>MAST cache: {s.mast_cache_dir}</li>
              <li>Weights: {s.model_weights_dir}</li>
              <li>Database file size: {fmtBytes(s.database_size_bytes)}</li>
            </ul>
            <div>
              <button
                type="button"
                disabled={clearingCache}
                onClick={async () => {
                  if (!confirm(
                    "Delete all downloaded FITS files from the MAST cache?\n\n" +
                    "This frees disk space but means light curves will need to be re-downloaded " +
                    "if you re-analyze a target. Your database, model weights, and all results are kept."
                  )) return
                  setClearingCache(true)
                  setErr(null)
                  setMsg(null)
                  try {
                    const r = await clearMastCache()
                    setMsg(`MAST cache cleared — freed ${fmtBytes(r.freed_bytes)}`)
                  } catch (e) {
                    setErr(extractApiErrorMessage(e))
                  } finally {
                    setClearingCache(false)
                  }
                }}
                className="btn-secondary text-xs disabled:opacity-40"
              >
                {clearingCache ? "Clearing…" : "Clear MAST cache"}
              </button>
              <Hint>
                Deletes downloaded FITS files to free disk space. Your database, results, and model weights are not
                affected. Light curves will be re-downloaded from MAST if you analyze a target again.
              </Hint>
            </div>
          </div>
        )}
      </section>

      <section className="card-surface p-6 mb-6">
        <h2 className="label-upper mb-2">GPU</h2>
        <Hint className="mb-3">
          Used for training the autoencoder and (if CUDA is available) faster inference. Install a CUDA build of PyTorch
          in <code className="text-dim">backend/venv</code> for GPU training.
        </Hint>
        {settingsLoad === "loading" && !s ? (
          <p className="text-sm text-muted-foreground">Waiting for settings...</p>
        ) : settingsLoad === "error" ? (
          <p className="text-sm text-muted-foreground">Unavailable until settings load succeeds.</p>
        ) : s?.gpu ? (
          s.gpu.ready ? (
            <ul className="text-sm text-dim space-y-1">
              <li>CUDA: {s.gpu.cuda_available ? "yes" : "no"}</li>
              {s.gpu.name && <li>Device: {s.gpu.name}</li>}
              {s.gpu.vram_total_bytes != null && <li>VRAM: {fmtBytes(s.gpu.vram_total_bytes)}</li>}
              {s.gpu.vram_free_bytes != null && <li>VRAM free: {fmtBytes(s.gpu.vram_free_bytes)}</li>}
            </ul>
          ) : (
            <p className="text-sm text-dim">
              GPU probe still running in the background (PyTorch/CUDA init can be slow on first load).
              Click <strong className="text-foreground">Refresh file status</strong> below to re-check.
            </p>
          )
        ) : (
          <p className="text-sm text-muted-foreground">No GPU data in response.</p>
        )}
      </section>

      <section className="card-surface p-6 mb-6">
        <h2 className="label-upper mb-2">Database</h2>
        <Hint className="mb-3">
          SQLite file lives under your data directory. Counts update as you run analyses or sector scans.
        </Hint>
        {s?.counts && (
          <ul className="text-sm text-dim space-y-1">
            <li>Targets: {s.counts.targets}</li>
            <li>Analyses: {s.counts.analyses}</li>
            <li>Flagged events: {s.counts.events}</li>
          </ul>
        )}
      </section>

      <section className="card-surface p-6 space-y-6">
        <h2 className="label-upper">Model training</h2>
        <div className="mb-2">
          <p className="text-xs text-muted-foreground font-mono">
            Weights: {s?.model_weights_exist ? "present" : "missing"} &middot; stats:{" "}
            {s?.model_stats_exist ? "present" : "missing"}
            {s?.model_weights_dir && (
              <span className="block mt-1 text-muted-foreground/70 break-all">{s.model_weights_dir}/autoencoder_v1.pt</span>
            )}
          </p>
          <Hint>
            The anomaly pipeline scores light curves against this model. <strong>Stats</strong> (
            <code className="text-dim">.stats.npz</code>) store training-time errors for calibration. Both should be
            present after a successful run. Use <strong>Refresh file status</strong> if the UI looks stale.
          </Hint>
        </div>

        <div className="card-surface bg-background/50 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={() => setTargetsExpanded(!targetsExpanded)}
              className="flex items-center gap-2 text-xs text-dim hover:text-foreground transition-colors"
            >
              <span className={`transition-transform ${targetsExpanded ? "rotate-90" : ""}`}>&#9654;</span>
              Training targets ({targets.length})
            </button>
            <div className="flex gap-2">
              {quietAvailable != null && quietAvailable > 0 && (
                <button
                  type="button"
                  disabled={importingQuiet || train?.running}
                  onClick={async () => {
                    setImportingQuiet(true)
                    try {
                      const r = await importQuietStarsToTargets(0.25, 0, 500)
                      setMsg(`Imported ${r.added} quiet stars (${r.total} total targets)`)
                      await loadTargets()
                    } catch (e) {
                      setErr(extractApiErrorMessage(e))
                    } finally {
                      setImportingQuiet(false)
                    }
                  }}
                  className="px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-400 text-[10px] hover:bg-emerald-500/20 disabled:opacity-40 transition-colors"
                >
                  {importingQuiet ? "Importing..." : `Import ${quietAvailable} quiet stars from scans`}
                </button>
              )}
              <button
                type="button"
                onClick={() => void loadTargets()}
                disabled={targetsLoading}
                className="btn-ghost text-[10px]"
              >
                {targetsLoading ? "..." : "Reload"}
              </button>
            </div>
          </div>
          <Hint>
            These stars are used to train the autoencoder. The model learns &quot;normal&quot; light curves from these
            targets. Import quiet stars (anomaly {"<"} 0.25, 0 flags) from your scans to diversify training data.
            Stored in <code className="text-dim">training_targets.json</code>.
          </Hint>

          {targetsExpanded && (
            <div className="space-y-2">
              {selectedForRemoval.size > 0 && (
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={async () => {
                      const ids = Array.from(selectedForRemoval)
                      try {
                        const r = await removeTrainingTargets(ids)
                        setMsg(`Removed ${r.removed} targets (${r.total} remaining)`)
                        setSelectedForRemoval(new Set())
                        await loadTargets()
                      } catch (e) {
                        setErr(extractApiErrorMessage(e))
                      }
                    }}
                    className="btn-destructive text-[10px] py-1 px-2.5"
                  >
                    Remove {selectedForRemoval.size} selected
                  </button>
                  <button
                    type="button"
                    onClick={() => setSelectedForRemoval(new Set())}
                    className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Clear selection
                  </button>
                </div>
              )}
              <div className="max-h-60 overflow-y-auto card-surface bg-background/50 overflow-hidden">
                <table className="w-full text-[11px] font-mono">
                  <thead className="sticky top-0 bg-card text-muted-foreground">
                    <tr>
                      <th className="w-8 px-2 py-1.5"></th>
                      <th className="text-left px-2 py-1.5">TIC ID</th>
                      <th className="text-right px-2 py-1.5">Anomaly</th>
                      <th className="text-left px-2 py-1.5">Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {targets.map((t) => (
                      <tr
                        key={t.tic_id}
                        className={`border-t border-border/40 hover:bg-white/5 transition-colors ${
                          selectedForRemoval.has(t.tic_id) ? "bg-red-500/5" : ""
                        }`}
                      >
                        <td className="px-2 py-1.5 text-center">
                          <input
                            type="checkbox"
                            checked={selectedForRemoval.has(t.tic_id)}
                            onChange={(e) => {
                              const next = new Set(selectedForRemoval)
                              if (e.target.checked) next.add(t.tic_id)
                              else next.delete(t.tic_id)
                              setSelectedForRemoval(next)
                            }}
                            className="accent-red-500 w-3 h-3"
                          />
                        </td>
                        <td className="px-2 py-1.5 text-dim">{t.tic_id}</td>
                        <td className="px-2 py-1.5 text-right text-muted-foreground">
                          {t.anomaly_score != null ? t.anomaly_score.toFixed(4) : "-"}
                        </td>
                        <td className="px-2 py-1.5 text-muted-foreground">{t.source}</td>
                      </tr>
                    ))}
                    {targets.length === 0 && (
                      <tr>
                        <td colSpan={4} className="px-2 py-3 text-center text-muted-foreground">No training targets</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] text-muted-foreground">
                {targets.filter((t) => t.source === "built-in").length} built-in
                {" + "}
                {targets.filter((t) => t.source === "scan").length} from scans
              </p>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="block label-upper mb-1.5">Epochs</label>
            <input
              type="number"
              min={1}
              max={500}
              value={trainEpochs}
              onChange={(e) => setTrainEpochs(parseInt(e.target.value, 10) || 1)}
              disabled={train?.running}
              title="How many full passes over the training windows"
              className="input-field w-full text-xs font-mono"
            />
            <Hint>
              Full passes over all training windows. Increase if the loss curve is still dropping at the end; lower for
              quick tests (allowed 1-500).
            </Hint>
          </div>
          <div>
            <label className="block label-upper mb-1.5">Batch size</label>
            <input
              type="number"
              min={4}
              max={512}
              value={trainBatch}
              onChange={(e) => setTrainBatch(parseInt(e.target.value, 10) || 64)}
              disabled={train?.running}
              title="Samples per gradient update"
              className="input-field w-full text-xs font-mono"
            />
            <Hint>
              Larger batches train faster per epoch and use more VRAM; smaller can generalize slightly differently. Try
              32-128 on a 3090 (allowed 4-512).
            </Hint>
          </div>
          <div>
            <label className="block label-upper mb-1.5">Learning rate</label>
            <input
              type="text"
              value={trainLr}
              onChange={(e) => setTrainLr(e.target.value)}
              disabled={train?.running}
              placeholder="0.001"
              title="Adam step size; try 0.001 default, or 1e-4 if training is unstable"
              className="input-field w-full text-xs font-mono"
            />
            <Hint>
              Step size for Adam. Default <code className="text-dim">3e-4</code> is recommended.
              Use <code className="text-dim">1e-4</code> for more stable but slower convergence.
              A cosine schedule automatically decays the rate during training.
            </Hint>
          </div>
          <div>
            <label className="block label-upper mb-1.5">
              Max stars (optional)
            </label>
            <input
              type="number"
              min={1}
              max={availableTrainStars}
              value={trainMaxTargets}
              onChange={(e) => setTrainMaxTargets(e.target.value)}
              disabled={train?.running}
              placeholder={`All (${availableTrainStars})`}
              title="Use only the first N stars from the training targets list"
              className="input-field w-full text-xs font-mono"
            />
            <Hint>
              First N of {availableTrainStars} targets in <code className="text-dim">training_targets.json</code>.
              Leave empty for all.
            </Hint>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 items-start">
          <div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={onTrain}
                disabled={train?.running || trainStarting || settingsLoad !== "ok" || targets.length === 0}
                title="Train autoencoder on training_targets.json, overwrite autoencoder_v1.pt and stats"
                className="btn-primary disabled:opacity-40 text-sm"
              >
                {trainStarting ? "Starting..." : `Train on ${availableTrainStars} targets`}
              </button>
              <button
                type="button"
                onClick={() => void load()}
                title="Re-check whether weight and stats files exist on disk"
                className="btn-secondary text-sm"
              >
                Refresh file status
              </button>
            </div>
            <Hint>
              <strong>Train model</strong> runs in the background; the loss chart updates each epoch. Only one run at a
              time.
            </Hint>
          </div>
        </div>

        <TrainingLossChart history={train?.loss_history ?? []} height={220} />
        <Hint>
          Each point is the average reconstruction MSE for that epoch (lower is better). A smooth curve dropping
          to <strong>0.05-0.30</strong> and flattening is ideal. The best checkpoint is automatically saved,
          so occasional spikes won&apos;t ruin the final model.
        </Hint>

        {train?.error_histogram?.bin_centers?.length ? (
          <>
            <ErrorHistogramChart histogram={train.error_histogram} height={160} />
            <Hint>
              Distribution of reconstruction errors across all training windows. The blue line marks the mean,
              the red line marks the 99th percentile. During scanning, windows with errors far beyond this
              distribution get flagged as anomalies.
            </Hint>
          </>
        ) : null}

        {train?.reconstruction_samples?.length ? (
          <>
            <ReconstructionChart samples={train.reconstruction_samples} />
            <Hint>
              How well the trained model reproduces actual training windows. &quot;Best&quot; shows near-perfect
              reconstruction, &quot;Worst&quot; shows patterns at the edge of what the model considers normal.
              Anything that reconstructs worse than the worst examples here will be flagged as anomalous during scanning.
            </Hint>
          </>
        ) : null}

        {train?.network_activations?.length ? (
          <NetworkFlowChart activations={train.network_activations} />
        ) : null}

        {train && (
          <div className="text-xs text-dim space-y-2">
            {train.hyperparams && (
              <p className="text-muted-foreground font-mono">
                Last run: {train.hyperparams.epochs} epochs &middot; batch {train.hyperparams.batch_size} &middot; lr{" "}
                {train.hyperparams.learning_rate}
                {train.hyperparams.max_targets != null
                  ? ` \u00b7 ${train.hyperparams.max_targets} stars`
                  : ` \u00b7 all ${availableTrainStars} stars`}
              </p>
            )}
            <p className="font-mono">
              Phase: <span className="text-foreground">{train.phase}</span>
              {train.running ? " (running)" : ""}
            </p>
            <p className="font-mono">
              Epoch: {train.epoch} / {train.total_epochs} &middot; loss: {train.loss ?? "-"}
            </p>
            <p className="font-mono">
              Targets fetched: {train.targets_fetched} / {train.total_targets} &middot; windows: {train.windows_count}
            </p>
            <p className="text-muted-foreground">{train.message}</p>
            {train.error && <p className="text-red-400">{train.error}</p>}
            {train.phase === "training" && train.total_epochs > 0 && (
              <div className="h-2 bg-accent rounded-full overflow-hidden mt-2">
                <div
                  className="h-full bg-foreground rounded-full transition-all"
                  style={{ width: `${Math.min(100, (100 * train.epoch) / train.total_epochs)}%` }}
                />
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  )
}

"use client"
import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { getLatestAnalysis, startAnalysis, pollAnalysis } from "@/lib/api"
import { LightCurveChart } from "@/components/LightCurveChart"
import { AnomalyScoreChart } from "@/components/AnomalyScoreChart"
import { PeriodogramChart } from "@/components/PeriodogramChart"
import { WaveletChart } from "@/components/WaveletChart"
import { CentroidChart } from "@/components/CentroidChart"
import { TPFViewer } from "@/components/TPFViewer"
import { EventFlagList } from "@/components/EventFlagList"
import { HypothesisList } from "@/components/HypothesisList"
import { ExportImageButton } from "@/components/ExportImage"
import { MetricCards } from "@/components/MetricCards"
import type { Analysis } from "@/lib/types"
import { APP_NAME } from "@/lib/brand"

const TABS = ["raw", "detrended", "pixels", "periodogram", "wavelet", "centroid"] as const
type Tab = (typeof TABS)[number]

const TAB_LABELS: Record<Tab, string> = {
  raw: "Raw",
  detrended: "Detrended",
  pixels: "Pixels",
  periodogram: "Periodogram",
  wavelet: "Wavelet",
  centroid: "Centroid",
}

function hasChartData(a: Analysis): boolean {
  return Boolean(
    a.raw_flux &&
      Array.isArray(a.raw_flux.time) &&
      a.raw_flux.time.length > 0 &&
      a.detrended_flux &&
      Array.isArray(a.detrended_flux.time) &&
      a.detrended_flux.time.length > 0,
  )
}

const TAB_DESCRIPTIONS: Record<Tab, string> = {
  raw:
    "The star's brightness over time, measured by the TESS spacecraft (one point every ~2 minutes). " +
    "A value of 1.0 is the star's normal brightness — dips below that mean the star appeared dimmer, " +
    "which could indicate a planet passing in front of it, or other phenomena. " +
    "Colored bands mark events the pipeline flagged.",
  detrended:
    "Same brightness data, but with the star's slow natural variability removed. " +
    "This makes short events like transits (planet crossings) and flares much easier to see as sudden dips or spikes " +
    "against a flat baseline. Colored bands mark the same flagged events.",
  pixels:
    "The actual CCD pixels around the target star, animated over time. Each cell is one detector pixel " +
    "colored by flux (electrons/second). The red outline marks the photometric aperture the TESS pipeline " +
    "uses to measure the star's brightness. Watch for: the target brightening/dimming, nearby stars " +
    "contaminating the aperture, or the centroid shifting between pixels during events.",
  periodogram:
    "A search for repeating patterns in the brightness data. The X-axis is the trial period (days) " +
    "and the Y-axis is how strongly that period fits the data. A tall peak means the star's brightness " +
    "dips at a regular interval — for example, a planet orbiting every 33 days would produce a peak at 33 on this chart. " +
    "The green dashed line marks the strongest period found.",
  wavelet:
    "Shows how periodic signals change over time. Horizontal axis is time, vertical is period (days). " +
    "Bright or warm colors indicate strong periodic power. " +
    "Dashed white lines mark known TESS spacecraft artifacts (orbital period, momentum dumps) — " +
    "signals at those periods are instrumental, not from the star.",
  centroid:
    "Tracks whether the star's measured position on the detector shifted during brightness dips. " +
    "If the position moves significantly during a dimming event, the brightness drop may come from a nearby star " +
    "(contamination), not the target. The dashed red line at 10\" is the contamination threshold. " +
    "Colored bands show flagged event windows.",
}

function EventLegend({ hasUnknown, hasClassified }: { hasUnknown: boolean; hasClassified: boolean }) {
  if (!hasUnknown && !hasClassified) return null
  return (
    <div className="flex gap-4 mt-2">
      {hasUnknown && (
        <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <span className="inline-block w-3 h-2 rounded-sm bg-red-500/30 border border-red-500/60" />
          Unclassified anomaly
        </span>
      )}
      {hasClassified && (
        <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <span className="inline-block w-3 h-2 rounded-sm bg-blue-500/20 border border-blue-500/50" />
          Classified event
        </span>
      )}
    </div>
  )
}

export default function AnalyzePage() {
  const params = useParams()
  const tic = decodeURIComponent(params.tic as string)

  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [status, setStatus] = useState<string>("idle")
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<Tab>("raw")
  const [runKey, setRunKey] = useState(0)
  const [sectorForRerun, setSectorForRerun] = useState<string>("all")

  useEffect(() => {
    let cancelled = false
    const run = async () => {
      try {
        setStatus("starting")
        setError(null)

        if (runKey === 0) {
          const cached = await getLatestAnalysis(tic)
          if (cancelled) return
          if (cached && cached.status === "complete") {
            setSectorForRerun(cached.sector || "all")
            if (hasChartData(cached)) {
              setAnalysis(cached)
              setStatus("complete")
              return
            }
            setAnalysis(cached)
            setStatus("running")
            const sectorArg = cached.sector || "all"
            const { analysis_id } = await startAnalysis(tic, sectorArg)
            if (cancelled) return
            await pollAnalysis(analysis_id, (a) => {
              if (!cancelled && a.status === "complete") setAnalysis(a)
            })
            if (!cancelled) setStatus("complete")
            return
          }
        }

        setAnalysis(null)
        const { analysis_id } = await startAnalysis(tic, sectorForRerun)
        if (cancelled) return
        setStatus("running")
        await pollAnalysis(analysis_id, (a) => {
          if (!cancelled) setAnalysis(a)
        })
        if (!cancelled) setStatus("complete")
      } catch (e) {
        if (!cancelled) {
          setStatus("failed")
          setError(e instanceof Error ? e.message : "Analysis failed")
        }
      }
    }
    run()
    return () => {
      cancelled = true
    }
  }, [tic, runKey])

  const handleReanalyze = () => {
    setRunKey((k) => k + 1)
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-8">
      <div className="mb-6">
        <p className="label-upper mb-1.5">
          {APP_NAME} · Analysis
        </p>
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold">{tic}</h1>
          {status === "complete" && (
            <div className="flex gap-2">
              <button
                onClick={handleReanalyze}
                className="btn-secondary text-[11px] py-1 px-3"
              >
                Re-analyze
              </button>
              {analysis && hasChartData(analysis) && (
                <ExportImageButton analysis={analysis} />
              )}
            </div>
          )}
        </div>
        <p className="text-xs text-muted-foreground mt-1 max-w-xl">
          {status === "starting" &&
            "Checking for a saved analysis (404 in the network tab here is normal if you have not run this target before)."}
          {status === "running" &&
            (analysis && !hasChartData(analysis)
              ? "Downloading MAST data and generating charts (light curves stay off disk). Metrics above are from your saved scan — often 1–3 minutes."
              : "Pipeline running: BLS, wavelet, centroid (TPF download), catalogs — often 1–3 minutes. Charts appear when the run finishes.")}
          {status === "complete" && analysis?.target && `TIC ${analysis.target.tic_id}`}
          {status === "failed" && <span className="text-red-400">Failed: {error}</span>}
        </p>
      </div>

      {analysis && analysis.status === "complete" && <MetricCards analysis={analysis} />}

      {analysis?.technosignature?.ran && (analysis.technosignature.composite_score ?? 0) > 0.1 && (
        <div className={`card-surface p-3 mb-4 text-xs ${
          (analysis.technosignature.composite_score ?? 0) > 0.5
            ? "border-red-500/30 bg-red-500/5 text-red-400"
            : "border-amber-500/20 bg-amber-500/5 text-amber-300"
        }`}>
          {analysis.technosignature.summary}
        </div>
      )}

      <div className="flex gap-0 mb-4 border-b border-border">
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2.5 text-[11px] uppercase tracking-wider border-b-2 -mb-px transition-colors ${
              activeTab === tab
                ? "border-foreground text-foreground"
                : "border-transparent text-muted-foreground hover:text-dim"
            }`}
          >
            {TAB_LABELS[tab]}
          </button>
        ))}
      </div>

      {analysis && analysis.status === "complete" && (
        <div className="card-surface p-5 mb-4">
          <p className="text-xs text-muted-foreground leading-relaxed mb-3">
            {TAB_DESCRIPTIONS[activeTab]}
          </p>

          {status === "running" && !hasChartData(analysis) && (
            <div className="card-surface bg-background/50 px-4 py-8 text-center mb-4">
              <div className="inline-block w-6 h-6 border-2 border-muted-foreground/30 border-t-foreground rounded-full animate-spin mb-3" />
              <p className="text-sm text-dim">Downloading data and generating charts…</p>
              <p className="text-[11px] text-muted-foreground mt-2 max-w-md mx-auto">
                FITS files are removed after each run so disk use stays small. Open this page again within ~10 minutes to avoid a re-download.
              </p>
            </div>
          )}

          {activeTab === "raw" && analysis.raw_flux && (
            <>
              <LightCurveChart
                time={analysis.raw_flux.time}
                flux={analysis.raw_flux.flux}
                events={analysis.events}
                height={260}
              />
              <EventLegend
                hasUnknown={analysis.events.some(e => e.event_type === "unknown")}
                hasClassified={analysis.events.some(e => e.event_type !== "unknown")}
              />
            </>
          )}
          {activeTab === "detrended" && analysis.detrended_flux && (
            <>
              <LightCurveChart
                time={analysis.detrended_flux.time}
                flux={analysis.detrended_flux.flux}
                events={analysis.events}
                height={260}
              />
              <EventLegend
                hasUnknown={analysis.events.some(e => e.event_type === "unknown")}
                hasClassified={analysis.events.some(e => e.event_type !== "unknown")}
              />
            </>
          )}
          {activeTab === "pixels" && (
            <TPFViewer tpf={analysis.tpf_data ?? { available: false }} height={360} />
          )}
          {activeTab === "periodogram" && analysis.periodogram && (
            <PeriodogramChart
              periods={analysis.periodogram.period}
              powers={analysis.periodogram.power}
              bestPeriod={analysis.known_period}
              height={260}
            />
          )}
          {activeTab === "wavelet" && analysis.wavelet && (
            <>
              {analysis.wavelet.tess_systematic_periods.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-3">
                  {analysis.wavelet.tess_systematic_periods.map((s) => (
                    <span key={s.name} className="text-[10px] px-2 py-0.5 rounded-lg bg-accent text-dim">
                      {s.name} artifact
                    </span>
                  ))}
                </div>
              )}
              <WaveletChart wavelet={analysis.wavelet} height={220} />
              <p className="text-[10px] text-muted-foreground mt-2 font-mono">
                Dominant periods: {analysis.wavelet.dominant_periods.map((p) => `${p}d`).join(", ") || "none"}
              </p>
            </>
          )}
          {activeTab === "centroid" && analysis.centroid && (
            <>
              <div className="flex justify-end items-center mb-2">
                {analysis.centroid.available && (
                  <span className={`text-[10px] px-2 py-0.5 rounded-lg ${
                    analysis.centroid.shift_flagged
                      ? "bg-red-500/10 text-red-400"
                      : "bg-emerald-500/10 text-emerald-400"
                  }`}>
                    {analysis.centroid.shift_flagged
                      ? `max ${analysis.centroid.max_shift_arcsec}" — possible contamination`
                      : `max ${analysis.centroid.max_shift_arcsec}" — on target`}
                  </span>
                )}
              </div>
              <CentroidChart centroid={analysis.centroid} events={analysis.events} height={180} />
              <EventLegend
                hasUnknown={analysis.events.some(e => e.event_type === "unknown")}
                hasClassified={analysis.events.some(e => e.event_type !== "unknown")}
              />
            </>
          )}
        </div>
      )}

      {analysis?.score_timeline && analysis.status === "complete" && (
        <div className="card-surface p-5 mb-4">
          <p className="label-upper mb-1">Anomaly Score Timeline</p>
          <p className="text-xs text-muted-foreground leading-relaxed mb-3">
            How &ldquo;unusual&rdquo; each part of the light curve looks to our autoencoder neural network.
            The model was trained on normal light curves &mdash; regions that look different from normal
            get higher scores (closer to 1.0). The dashed red line at 0.5 is the anomaly threshold;
            anything above it is flagged for review.
          </p>
          <AnomalyScoreChart
            time={analysis.score_timeline.time}
            scores={analysis.score_timeline.score}
            events={analysis.events}
            height={120}
          />
          <EventLegend
            hasUnknown={analysis.events.some(e => e.event_type === "unknown")}
            hasClassified={analysis.events.some(e => e.event_type !== "unknown")}
          />
        </div>
      )}

      {analysis?.technosignature?.hypotheses && analysis.technosignature.hypotheses.length > 0 && (
        <div className="mb-4">
          <HypothesisList hypotheses={analysis.technosignature.hypotheses} />
        </div>
      )}

      {analysis && analysis.status === "complete" && analysis.events.length > 0 && (
        <EventFlagList events={analysis.events} />
      )}

      {(status === "starting" || status === "running") &&
        !(analysis?.status === "complete" && analysis && !hasChartData(analysis)) &&
        !analysis?.raw_flux && (
        <div className="text-center py-16 max-w-md mx-auto">
          <div className="inline-block w-6 h-6 border-2 border-muted-foreground/30 border-t-foreground rounded-full animate-spin mb-4" />
          <p className="text-sm text-dim mb-2">
            {status === "starting" ? "Starting…" : "Working…"}
          </p>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {status === "starting"
              ? "Looking up a previous completed analysis, or queueing a new run."
              : "Light curve may already be downloaded; the backend is still scoring, period-searching, and vetting. Nothing is wrong if this screen stays empty for a couple of minutes."}
          </p>
        </div>
      )}
    </div>
  )
}

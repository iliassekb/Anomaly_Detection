"use client"

import { useState, useEffect, useRef } from "react"
import { Play, Square, RefreshCw, CheckCircle, Clock, AlertCircle, ChevronDown, ChevronUp } from "lucide-react"
import {
  TrainingStatus, TrainingRun,
  getTrainingStatus, triggerTraining, stopTraining,
  getTrainingRuns, getRunMetricHistory, activateRun,
} from "@/lib/dataset-api"

// The 8 metrics shown in the run-card history, split into two groups
// Keys are sanitized: (B) → _B  and  (M) → _M  (MLflow forbids parentheses)
const HISTORY_METRICS = {
  Box: {
    "metrics/precision_B":  "Precision",
    "metrics/recall_B":     "Recall",
    "metrics/mAP50_B":      "mAP50",
    "metrics/mAP50-95_B":   "mAP50-95",
  },
  Mask: {
    "metrics/precision_M":  "Precision",
    "metrics/recall_M":     "Recall",
    "metrics/mAP50_M":      "mAP50",
    "metrics/mAP50-95_M":   "mAP50-95",
  },
}

// Flat label map still used for live-training chips
const METRIC_LABELS: Record<string, string> = {
  "train/box_loss":        "Box",
  "train/seg_loss":        "Seg",
  "train/cls_loss":        "Cls",
  "train/dfl_loss":        "DFL",
  "metrics/mAP50(M)":      "mAP50",
  "metrics/mAP50-95(M)":   "mAP50-95",
  "metrics/precision(M)":  "Precision",
  "metrics/recall(M)":     "Recall",
}

const KEY_METRICS = Object.keys(METRIC_LABELS)

// ── Run history card ──────────────────────────────────────────────

type HistoryMap = Record<string, { step: number; value: number }[]>

function RunCard({
  run, onActivate, activating,
}: {
  run: TrainingRun
  onActivate: (id: string) => void
  activating: string | null
}) {
  const [expanded, setExpanded] = useState(false)
  const [history, setHistory] = useState<HistoryMap | null>(null)
  const [loadingHistory, setLoadingHistory] = useState(false)

  const finished = run.status === "FINISHED"
  const crashed  = run.status === "FAILED" || run.status === "KILLED"
  const mAP = run.metrics["metrics/mAP50(M)"] ?? run.metrics["mAP50"] ?? null
  const date = run.start_time ? new Date(run.start_time).toLocaleDateString() : "—"

  async function toggleExpand() {
    if (!expanded && !history) {
      setLoadingHistory(true)
      try {
        const res = await getRunMetricHistory(run.run_id)
        setHistory(res.history)
      } catch {}
      finally { setLoadingHistory(false) }
    }
    setExpanded(v => !v)
  }

  const statusColor = finished ? "var(--success)"
    : crashed ? "var(--danger)"
    : "var(--accent)"

  const statusLabel = finished ? "FINISHED"
    : crashed ? "CRASHED"
    : run.status === "RUNNING" ? "INTERRUPTED"
    : run.status

  // Build a simple sparkline from history values for a given key
  function Sparkline({ values }: { values: number[] }) {
    if (values.length < 2) return null
    const min = Math.min(...values), max = Math.max(...values)
    const range = max - min || 1
    const w = 80, h = 24
    const pts = values.map((v, i) => {
      const x = (i / (values.length - 1)) * w
      const y = h - ((v - min) / range) * h
      return `${x},${y}`
    }).join(" ")
    return (
      <svg width={w} height={h} style={{ display: "block" }}>
        <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
      </svg>
    )
  }

  return (
    <div className="panel" style={{ padding: "12px 14px" }}>
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 12, fontFamily: "monospace", color: "var(--text-secondary)" }}>
            {run.run_id.slice(0, 12)}…
          </div>
          <div className="flex items-center gap-3 mt-1 flex-wrap">
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              <Clock size={10} style={{ display: "inline", marginRight: 3 }} />{date}
            </span>
            <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.5px", textTransform: "uppercase", color: statusColor }}>
              {statusLabel}
            </span>
            {mAP !== null && (
              <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                mAP50 <strong>{(mAP * 100).toFixed(1)}%</strong>
              </span>
            )}
            {run.params.total_images && (
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{run.params.total_images} imgs</span>
            )}
            {run.params.epochs && (
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{run.params.epochs} epochs</span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {finished && (
            <button
              className="btn btn-secondary"
              style={{ padding: "4px 10px", fontSize: 12 }}
              onClick={() => onActivate(run.run_id)}
              disabled={activating === run.run_id}
            >
              {activating === run.run_id ? "Activating…" : "Activate"}
            </button>
          )}
          <button
            className="btn btn-secondary"
            style={{ padding: "4px 8px" }}
            onClick={toggleExpand}
            disabled={loadingHistory}
          >
            {loadingHistory
              ? <RefreshCw size={12} className="animate-spin" />
              : expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
        </div>
      </div>

      {/* Expanded metrics history */}
      {expanded && history && (
        <div style={{ marginTop: 12, borderTop: "1px solid var(--border)", paddingTop: 10 }}>
          {(() => {
            const hasAny = Object.values(HISTORY_METRICS).some(group =>
              Object.keys(group).some(k => history[k]?.length > 0)
            )
            if (!hasAny)
              return <p style={{ fontSize: 12, color: "var(--text-muted)" }}>No metric history recorded.</p>

            return (
              <div className="space-y-3">
                {(Object.entries(HISTORY_METRICS) as [string, Record<string, string>][]).map(([groupName, keys]) => (
                  <div key={groupName}>
                    <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase",
                      letterSpacing: "0.5px", color: "var(--text-muted)", marginBottom: 6 }}>
                      {groupName}
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
                      {Object.entries(keys).map(([key, label]) => {
                        const pts = history[key]
                        if (!pts?.length) return (
                          <div key={key} style={{
                            background: "var(--surface-alt)", borderRadius: "var(--radius-sm)",
                            padding: "8px 10px", opacity: 0.4,
                          }}>
                            <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{label}</div>
                            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>—</div>
                          </div>
                        )
                        const vals = pts.map(p => p.value)
                        const last = vals[vals.length - 1]
                        return (
                          <div key={key} style={{
                            background: "var(--surface-alt)", borderRadius: "var(--radius-sm)",
                            padding: "8px 10px",
                          }}>
                            <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>{label}</div>
                            <Sparkline values={vals} />
                            <div style={{ fontSize: 12, fontWeight: 600, fontFamily: "monospace", marginTop: 2 }}>
                              {(last * 100).toFixed(1)}%
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )
          })()}
        </div>
      )}
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────

export default function TrainingPanel() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [triggering, setTriggering] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [activating, setActivating] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [runsLoading, setRunsLoading] = useState(false)
  const wasRunning = useRef(false)

  async function loadStatus() {
    try { setStatus(await getTrainingStatus()) } catch {}
  }
  async function loadRuns() {
    setRunsLoading(true)
    try { setRuns(await getTrainingRuns()) } catch {}
    finally { setRunsLoading(false) }
  }

  useEffect(() => { loadStatus(); loadRuns() }, [])

  useEffect(() => {
    if (!status?.running) {
      if (wasRunning.current) { loadRuns(); setStopping(false) }
      wasRunning.current = false
      return
    }
    wasRunning.current = true
    const id = setInterval(loadStatus, 2000)
    return () => clearInterval(id)
  }, [status?.running])

  async function handleTrigger() {
    setTriggering(true); setError(null)
    try { await triggerTraining(); await loadStatus() }
    catch (e: any) { setError(e.message) }
    finally { setTriggering(false) }
  }

  async function handleStop() {
    setStopping(true); setError(null)
    try { await stopTraining() }
    catch (e: any) { setError(e.message); setStopping(false) }
  }

  async function handleActivate(runId: string) {
    if (!confirm("Replace active model weights with this run's best checkpoint?")) return
    setActivating(runId)
    try { await activateRun(runId); setError(null) }
    catch (e: any) { setError(e.message) }
    finally { setActivating(null) }
  }

  const progressPct = status && status.total_epochs > 0
    ? Math.round((status.current_epoch / status.total_epochs) * 100) : 0

  const liveMetrics = status?.current_metrics
    ? KEY_METRICS.filter(k => status.current_metrics[k] !== undefined) : []

  return (
    <div className="space-y-5">
      <span className="section-label">Training</span>

      {/* Status card */}
      <div className="panel" style={{ padding: "16px 18px" }}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {status?.running
              ? <RefreshCw size={15} className="animate-spin" style={{ color: "var(--accent)" }} />
              : status?.error
              ? <AlertCircle size={15} style={{ color: "var(--danger)" }} />
              : <CheckCircle size={15} style={{ color: "var(--success)" }} />}
            <span style={{ fontSize: 13, fontWeight: 600 }}>
              {status?.running
                ? status.stop_requested
                  ? `Stopping after epoch ${status.current_epoch}…`
                  : `Training — epoch ${status.current_epoch} / ${status.total_epochs}`
                : status?.error ? "Training failed" : "Idle"}
            </span>
          </div>

          <div className="flex gap-2">
            {status?.running && (
              <button
                className="btn btn-secondary"
                style={{ padding: "6px 14px", color: "var(--danger)" }}
                onClick={handleStop}
                disabled={stopping || status.stop_requested}
              >
                <Square size={12} />
                {stopping || status.stop_requested ? "Stopping…" : "Stop"}
              </button>
            )}
            <button
              className="btn btn-primary"
              style={{ padding: "6px 14px" }}
              onClick={handleTrigger}
              disabled={triggering || status?.running}
            >
              <Play size={13} />
              {triggering ? "Starting…" : "Start Training"}
            </button>
          </div>
        </div>

        {status?.running && (
          <div style={{ marginTop: 8 }}>
            <div style={{ height: 6, background: "var(--surface-alt)", borderRadius: 3, overflow: "hidden" }}>
              <div style={{
                height: "100%", borderRadius: 3,
                background: status.stop_requested ? "var(--danger)" : "var(--accent)",
                width: `${progressPct}%`, transition: "width 0.5s",
              }} />
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
              {progressPct}% complete
              {status.run_id && <span> · run {status.run_id.slice(0, 8)}</span>}
            </div>
            {liveMetrics.length > 0 && (
              <div className="flex flex-wrap gap-2" style={{ marginTop: 10 }}>
                {liveMetrics.map(key => (
                  <div key={key} style={{
                    background: "var(--surface-alt)", borderRadius: "var(--radius-sm)",
                    padding: "3px 8px", fontSize: 11,
                  }}>
                    <span style={{ color: "var(--text-muted)" }}>{METRIC_LABELS[key]} </span>
                    <span style={{ fontWeight: 600, fontFamily: "monospace" }}>
                      {status.current_metrics[key].toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {status?.error && (
          <p style={{ fontSize: 12, color: "var(--danger)", marginTop: 6 }}>{status.error}</p>
        )}
      </div>

      {error && <p style={{ fontSize: 12, color: "var(--danger)" }}>{error}</p>}

      {/* Runs list */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <span className="section-label">MLflow Runs</span>
          <button className="btn btn-secondary" style={{ padding: "5px 10px" }} onClick={loadRuns} disabled={runsLoading}>
            <RefreshCw size={13} className={runsLoading ? "animate-spin" : ""} />
          </button>
        </div>

        {runs.length === 0 && !runsLoading && (
          <p style={{ fontSize: 13, color: "var(--text-muted)", textAlign: "center", padding: "20px 0" }}>
            No training runs yet.
          </p>
        )}

        <div className="space-y-2">
          {runs.map(run => (
            <RunCard
              key={run.run_id}
              run={run}
              onActivate={handleActivate}
              activating={activating}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

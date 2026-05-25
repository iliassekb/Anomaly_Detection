"use client"

import { useState, useEffect } from "react"
import { Play, RefreshCw, CheckCircle, Clock, AlertCircle } from "lucide-react"
import {
  TrainingStatus, TrainingRun,
  getTrainingStatus, triggerTraining, getTrainingRuns, activateRun,
} from "@/lib/dataset-api"

export default function TrainingPanel() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [triggering, setTriggering] = useState(false)
  const [activating, setActivating] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [runsLoading, setRunsLoading] = useState(false)

  async function loadStatus() {
    try { setStatus(await getTrainingStatus()) } catch {}
  }

  async function loadRuns() {
    setRunsLoading(true)
    try { setRuns(await getTrainingRuns()) } catch {}
    finally { setRunsLoading(false) }
  }

  useEffect(() => {
    loadStatus()
    loadRuns()
  }, [])

  // Poll while training
  useEffect(() => {
    if (!status?.running) return
    const id = setInterval(loadStatus, 3000)
    return () => clearInterval(id)
  }, [status?.running])

  async function handleTrigger() {
    setTriggering(true)
    setError(null)
    try {
      await triggerTraining()
      await loadStatus()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setTriggering(false)
    }
  }

  async function handleActivate(runId: string) {
    if (!confirm("Replace the active model weights with this run's best checkpoint?")) return
    setActivating(runId)
    try {
      await activateRun(runId)
      setError(null)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setActivating(null)
    }
  }

  const progressPct = status && status.total_epochs > 0
    ? Math.round((status.current_epoch / status.total_epochs) * 100)
    : 0

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
                ? `Training — epoch ${status.current_epoch}/${status.total_epochs}`
                : status?.error
                ? "Training failed"
                : "Idle"}
            </span>
          </div>
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

        {status?.running && (
          <div style={{ marginTop: 8 }}>
            <div style={{
              height: 6, background: "var(--surface-alt)", borderRadius: 3, overflow: "hidden",
            }}>
              <div style={{
                height: "100%", borderRadius: 3,
                background: "var(--accent)",
                width: `${progressPct}%`,
                transition: "width 0.5s",
              }} />
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
              {progressPct}% complete
              {status.run_id && <span> · run {status.run_id.slice(0, 8)}</span>}
            </div>
          </div>
        )}

        {status?.error && (
          <p style={{ fontSize: 12, color: "var(--danger)", marginTop: 6 }}>{status.error}</p>
        )}
      </div>

      {error && <p style={{ fontSize: 12, color: "var(--danger)" }}>{error}</p>}

      {/* Runs table */}
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
          {runs.map(run => {
            const mAP = run.metrics["metrics/mAP50(M)"] ?? run.metrics["mAP50"] ?? null
            const date = run.start_time ? new Date(run.start_time).toLocaleDateString() : "—"
            const finished = run.status === "FINISHED"
            return (
              <div key={run.run_id} className="panel" style={{ padding: "12px 14px" }}>
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <div style={{ fontSize: 12, fontFamily: "monospace", color: "var(--text-secondary)" }}>
                      {run.run_id.slice(0, 12)}…
                    </div>
                    <div className="flex items-center gap-3 mt-1 flex-wrap">
                      <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                        <Clock size={10} style={{ display: "inline", marginRight: 3 }} />{date}
                      </span>
                      <span style={{
                        fontSize: 10, fontWeight: 700, letterSpacing: "0.5px", textTransform: "uppercase",
                        color: finished ? "var(--success)" : run.status === "FAILED" ? "var(--danger)" : "var(--accent)",
                      }}>
                        {run.status}
                      </span>
                      {mAP !== null && (
                        <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                          mAP50: <strong>{(mAP * 100).toFixed(1)}%</strong>
                        </span>
                      )}
                      {run.params.total_images && (
                        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                          {run.params.total_images} images
                        </span>
                      )}
                    </div>
                  </div>
                  {finished && (
                    <button
                      className="btn btn-secondary"
                      style={{ padding: "5px 12px", fontSize: 12 }}
                      onClick={() => handleActivate(run.run_id)}
                      disabled={activating === run.run_id}
                    >
                      {activating === run.run_id ? "Activating…" : "Activate"}
                    </button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

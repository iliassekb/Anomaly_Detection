"use client"

import { useState, useCallback } from "react"
import { Film, Upload, Download, Loader2 } from "lucide-react"
import { predictVideo, VideoStats, ModelConfig } from "@/lib/api"
import { HistoryEntry } from "@/hooks/useHistory"

interface Props {
  config: ModelConfig
  onDetection: (entry: Omit<HistoryEntry, "id" | "timestamp">) => void
}

export default function VideoDetector({ config, onDetection }: Props) {
  const [file,     setFile]     = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [stats,    setStats]    = useState<VideoStats | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [step,     setStep]     = useState(2)
  const [error,    setError]    = useState<string | null>(null)

  const handleFile = (f: File) => { setFile(f); setStats(null); setVideoUrl(null); setError(null) }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f?.type.startsWith("video/")) handleFile(f)
  }, [])

  const run = async () => {
    if (!file) return
    setLoading(true); setError(null); setStats(null); setVideoUrl(null)
    try {
      const { blob, stats: s } = await predictVideo(file, config, step)
      setStats(s); setVideoUrl(URL.createObjectURL(blob))
      onDetection({
        type: "video",
        filename: file.name,
        is_anomaly: s.anomaly_frames > 0,
        detections_count: s.total_detections,
        defect_classes: Object.keys(s.detections_by_class),
        anomaly_rate: s.anomaly_rate,
        _videoStats: s,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed")
    } finally { setLoading(false) }
  }

  const download = () => {
    if (!videoUrl) return
    const a = document.createElement("a")
    a.href = videoUrl; a.download = `annotated_${file?.name || "output.mp4"}`; a.click()
  }

  const stepPct = ((step - 1) / (10 - 1)) * 100

  return (
    <div className="space-y-5">

      {/* Drop zone */}
      <div
        className={`dropzone${dragging ? " active" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => document.getElementById("vid-input")?.click()}
      >
        <input
          id="vid-input" type="file" accept="video/*" className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        <div className="py-10 flex flex-col items-center gap-3 pointer-events-none">
          <div
            className="w-10 h-10 rounded flex items-center justify-center"
            style={{ background: "var(--accent-surface)", border: "1px solid var(--accent-border)" }}
          >
            <Upload size={18} style={{ color: "var(--accent)" }} />
          </div>
          <div className="text-center" style={{ lineHeight: 1.6 }}>
            <p style={{ fontSize: 13, fontWeight: 600, color: "var(--text-secondary)" }}>
              Drop a video here or click to browse
            </p>
            <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>
              MP4, AVI, MOV
            </p>
          </div>
        </div>
      </div>

      {/* File card */}
      {file && (
        <div className="panel" style={{ padding: "16px 18px" }}>
          <div className="space-y-4">

            {/* File info */}
            <div className="flex items-center gap-3">
              <div
                className="w-9 h-9 rounded flex items-center justify-center flex-shrink-0"
                style={{ background: "var(--accent-surface)", border: "1px solid var(--accent-border)" }}
              >
                <Film size={15} style={{ color: "var(--accent)" }} />
              </div>
              <div className="min-w-0">
                <p style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }} className="truncate">
                  {file.name}
                </p>
                <p style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  {(file.size / 1024 / 1024).toFixed(1)} MB
                </p>
              </div>
            </div>

            {/* Frame step slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="field-label">Frame sampling</span>
                <span style={{ fontSize: 12, fontFamily: "monospace", fontWeight: 600, color: "var(--accent)" }}>
                  every {step}{step === 1 ? "st" : step === 2 ? "nd" : "th"} frame
                </span>
              </div>
              <input
                type="range" min={1} max={10} step={1} value={step}
                onChange={(e) => setStep(Number(e.target.value))}
                style={{
                  background: `linear-gradient(to right, var(--accent) ${stepPct}%, var(--border) ${stepPct}%)`,
                }}
              />
              <p style={{ fontSize: 11, color: "var(--text-muted)" }}>
                Lower value = higher accuracy, longer processing time
              </p>
            </div>

            <button
              className="btn btn-primary"
              onClick={run}
              disabled={loading}
              style={{ width: "100%", justifyContent: "center" }}
            >
              {loading
                ? <><Loader2 size={13} className="animate-spin" /> Processing video…</>
                : "Scan for defects"}
            </button>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          className="panel"
          style={{ padding: "12px 16px", background: "var(--danger-surface)", borderColor: "var(--danger-border)" }}
        >
          <p style={{ fontSize: 13, color: "var(--danger)" }}>{error}</p>
        </div>
      )}

      {/* Results */}
      {stats && (
        <div className="space-y-4">

          {/* Video player */}
          {videoUrl && (
            <div className="panel" style={{ padding: "16px 18px" }}>
              <div className="flex items-center justify-between mb-3">
                <span style={{
                  fontSize: 10, fontWeight: 700, letterSpacing: "0.8px",
                  textTransform: "uppercase", color: "var(--text-muted)",
                }}>
                  Annotated output
                </span>
                <button
                  className="btn btn-secondary"
                  style={{ padding: "5px 10px", fontSize: 12 }}
                  onClick={download}
                >
                  <Download size={12} /> Export
                </button>
              </div>
              <video
                src={videoUrl} controls className="w-full"
                style={{ borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", display: "block" }}
              />
            </div>
          )}

          {/* Stats grid */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              { label: "Frames scanned",  value: stats.total_frames       },
              { label: "With defects",    value: stats.anomaly_frames     },
              { label: "Defect rate",     value: `${stats.anomaly_rate}%` },
              { label: "Total findings",  value: stats.total_detections   },
            ].map((s) => (
              <div key={s.label} className="panel" style={{ padding: "14px 16px" }}>
                <div style={{
                  fontSize: 10, fontWeight: 700, letterSpacing: "0.8px",
                  textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 6,
                }}>
                  {s.label}
                </div>
                <div style={{
                  fontSize: 24, fontWeight: 700, color: "var(--text-primary)",
                  fontVariantNumeric: "tabular-nums", lineHeight: 1,
                }}>
                  {s.value}
                </div>
              </div>
            ))}
          </div>

          {/* Breakdown by class */}
          {Object.keys(stats.detections_by_class).length > 0 && (
            <div className="panel" style={{ padding: "16px 18px" }}>
              <div style={{
                fontSize: 10, fontWeight: 700, letterSpacing: "0.8px",
                textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 14,
              }}>
                Findings by defect type
              </div>
              <div className="space-y-3">
                {Object.entries(stats.detections_by_class).map(([cls, n]) => {
                  const pct = Math.round((n / stats.total_detections) * 100)
                  return (
                    <div key={cls}>
                      <div className="flex justify-between items-baseline mb-1.5">
                        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", textTransform: "capitalize" }}>
                          {cls}
                        </span>
                        <span style={{ fontSize: 11, fontFamily: "monospace", color: "var(--text-muted)" }}>
                          {n} · {pct}%
                        </span>
                      </div>
                      <div style={{ height: 4, borderRadius: 2, background: "var(--surface-alt)", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${pct}%`, background: "var(--danger)", borderRadius: 2 }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

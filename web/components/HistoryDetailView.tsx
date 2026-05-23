"use client"

import { HistoryEntry } from "@/hooks/useHistory"
import { ArrowLeft, AlertTriangle, CheckCircle, FileImage, Film, Camera } from "lucide-react"
import DetectionResults from "./DetectionResults"

interface Props {
  entry: HistoryEntry
  onBack: () => void
}

function formatDate(ts: number): string {
  return new Date(ts).toLocaleString(undefined, {
    weekday: "short", year: "numeric", month: "short",
    day: "numeric", hour: "2-digit", minute: "2-digit",
  })
}

const TYPE_LABEL: Record<HistoryEntry["type"], string> = {
  image: "Image scan",
  video: "Video scan",
  camera: "Camera session",
}

const TypeIcon = ({ type }: { type: HistoryEntry["type"] }) => {
  const props = { size: 14, style: { color: "var(--accent)", flexShrink: 0 as const } }
  if (type === "image") return <FileImage {...props} />
  if (type === "video") return <Film {...props} />
  return <Camera {...props} />
}

export default function HistoryDetailView({ entry, onBack }: Props) {
  const hasImageData = !!entry._imageData
  const hasVideoStats = !!entry._videoStats

  return (
    <div className="space-y-5">

      {/* Back + title */}
      <div className="flex items-start gap-4">
        <button
          onClick={onBack}
          className="btn btn-secondary"
          style={{ padding: "6px 12px", flexShrink: 0, marginTop: 3 }}
        >
          <ArrowLeft size={13} /> Back
        </button>
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <TypeIcon type={entry.type} />
            <span style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 600, letterSpacing: "0.5px", textTransform: "uppercase" }}>
              {TYPE_LABEL[entry.type]}
            </span>
          </div>
          <h2 style={{
            fontSize: 17, fontWeight: 700, color: "var(--text-primary)",
            letterSpacing: "-0.3px", overflow: "hidden",
            textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {entry.filename}
          </h2>
          <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 3 }}>
            {formatDate(entry.timestamp)}
          </p>
        </div>
      </div>

      {/* Status banner */}
      <div
        className="flex items-center gap-3 px-4 py-3"
        style={{
          background: entry.is_anomaly ? "var(--danger-surface)" : "var(--success-surface)",
          border: `1px solid ${entry.is_anomaly ? "var(--danger-border)" : "var(--success-border)"}`,
          borderRadius: "var(--radius-sm)",
        }}
      >
        <div className="w-2 h-2 rounded-full flex-shrink-0"
          style={{ background: entry.is_anomaly ? "var(--danger)" : "var(--success)" }} />
        <span style={{
          fontSize: 13, fontWeight: 600,
          color: entry.is_anomaly ? "var(--danger)" : "var(--success)",
          flex: 1,
        }}>
          {entry.is_anomaly
            ? `Defect detected — ${entry.defect_classes.map((c) => `${c}`).join(", ")} (${entry.detections_count} instance${entry.detections_count !== 1 ? "s" : ""})`
            : "No defects detected"}
        </span>
        {entry.inference_ms !== undefined && (
          <span style={{ fontSize: 11, fontFamily: "monospace", color: "var(--text-muted)", flexShrink: 0 }}>
            {entry.inference_ms} ms
          </span>
        )}
      </div>

      {/* ── Image result ── */}
      {hasImageData && entry._imageData && (
        <div className="panel" style={{ padding: "18px 20px" }}>
          <div className="grid grid-cols-2 gap-3 mb-5">
            {[
              { label: "Original",  src: entry._imageData.original_b64  },
              { label: "Annotated", src: entry._imageData.annotated_b64 },
            ].map(({ label, src }) => (
              <div key={label}>
                <div style={{
                  fontSize: 10, fontWeight: 700, letterSpacing: "0.8px",
                  textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 6,
                }}>
                  {label}
                </div>
                <img
                  src={`data:image/jpeg;base64,${src}`}
                  alt={label}
                  className="w-full"
                  style={{
                    borderRadius: "var(--radius-sm)",
                    border: "1px solid var(--border)", display: "block",
                  }}
                />
              </div>
            ))}
          </div>
          <DetectionResults
            detections={entry._imageData.detections}
            inference_ms={entry.inference_ms ?? 0}
            is_anomaly={entry.is_anomaly}
          />
        </div>
      )}

      {/* ── Video stats ── */}
      {hasVideoStats && entry._videoStats && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              { label: "Frames scanned", value: entry._videoStats.total_frames      },
              { label: "With defects",   value: entry._videoStats.anomaly_frames    },
              { label: "Defect rate",    value: `${entry._videoStats.anomaly_rate}%` },
              { label: "Total findings", value: entry._videoStats.total_detections  },
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

          {Object.keys(entry._videoStats.detections_by_class).length > 0 && (
            <div className="panel" style={{ padding: "16px 18px" }}>
              <div style={{
                fontSize: 10, fontWeight: 700, letterSpacing: "0.8px",
                textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 14,
              }}>
                Findings by defect type
              </div>
              <div className="space-y-3">
                {Object.entries(entry._videoStats.detections_by_class).map(([cls, n]) => {
                  const pct = Math.round((n / entry._videoStats!.total_detections) * 100)
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

      {/* ── Camera session summary ── */}
      {entry.type === "camera" && !hasVideoStats && (
        <div className="panel" style={{ padding: "16px 18px" }}>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 6 }}>
                Total detections
              </div>
              <div style={{ fontSize: 28, fontWeight: 700, color: "var(--text-primary)", lineHeight: 1 }}>
                {entry.detections_count}
              </div>
            </div>
            {entry.defect_classes.length > 0 && (
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 6 }}>
                  Classes found
                </div>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                  {entry.defect_classes.map((cls) => (
                    <span key={cls} style={{
                      fontSize: 11, fontWeight: 700, color: "var(--danger)",
                      background: "var(--danger-surface)",
                      border: "1px solid var(--danger-border)",
                      padding: "2px 7px", borderRadius: "var(--radius-sm)",
                      textTransform: "capitalize",
                    }}>
                      {cls}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Fallback — entry loaded from localStorage, no session data */}
      {!hasImageData && !hasVideoStats && entry.type !== "camera" && (
        <div className="panel" style={{ padding: "32px 20px", textAlign: "center" }}>
          <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
            Full result images are only available for scans done in the current session.
          </p>
          <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 6 }}>
            Re-run the scan to see the annotated output.
          </p>
        </div>
      )}
    </div>
  )
}

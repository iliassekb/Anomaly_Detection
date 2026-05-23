"use client"

import { HistoryEntry } from "@/hooks/useHistory"
import { FileImage, Film, Camera, Trash2, X, Clock } from "lucide-react"

interface Props {
  entries: HistoryEntry[]
  onClear: () => void
  onRemove: (id: string) => void
  onSelect: (id: string) => void
  selectedId: string | null
}

function ago(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000)
  if (s < 60) return "just now"
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  const d = Math.floor(h / 24)
  if (d < 7) return `${d}d ago`
  return new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric" })
}

function dayLabel(ts: number): string {
  const d = new Date(ts)
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(today.getDate() - 1)
  if (d.toDateString() === today.toDateString()) return "Today"
  if (d.toDateString() === yesterday.toDateString()) return "Yesterday"
  return d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })
}

function groupByDay(entries: HistoryEntry[]): { label: string; items: HistoryEntry[] }[] {
  const map = new Map<string, HistoryEntry[]>()
  for (const e of entries) {
    const key = new Date(e.timestamp).toDateString()
    if (!map.has(key)) map.set(key, [])
    map.get(key)!.push(e)
  }
  return Array.from(map.entries()).map(([key, items]) => ({
    label: dayLabel(items[0].timestamp),
    items,
  }))
}

const TypeIcon = ({ type }: { type: HistoryEntry["type"] }) => {
  const props = { size: 12, style: { flexShrink: 0 as const } }
  if (type === "image") return <FileImage {...props} />
  if (type === "video") return <Film {...props} />
  return <Camera {...props} />
}

export default function HistoryPanel({ entries, onClear, onRemove, onSelect, selectedId }: Props) {
  const anomalyCount = entries.filter((e) => e.is_anomaly).length
  const groups = groupByDay(entries)

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>

      {/* Header */}
      <div style={{
        flexShrink: 0, display: "flex", alignItems: "center",
        justifyContent: "space-between", padding: "11px 14px",
        borderBottom: "1px solid var(--border)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{
            fontSize: 10, fontWeight: 700, letterSpacing: "0.8px",
            textTransform: "uppercase", color: "var(--text-muted)",
          }}>
            History
          </span>
          {entries.length > 0 && (
            <span style={{
              fontSize: 10, fontWeight: 700, color: "var(--text-muted)",
              background: "var(--surface-alt)", padding: "1px 6px",
              borderRadius: 10, border: "1px solid var(--border)",
              lineHeight: 1.6,
            }}>
              {entries.length}
            </span>
          )}
        </div>
        {entries.length > 0 && (
          <button
            onClick={onClear}
            title="Clear all history"
            style={{
              background: "none", border: "none", cursor: "pointer",
              color: "var(--text-muted)", padding: 4,
              borderRadius: "var(--radius-sm)",
              display: "flex", alignItems: "center", transition: "color 0.12s",
            }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.color = "var(--danger)")}
            onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.color = "var(--text-muted)")}
          >
            <Trash2 size={12} />
          </button>
        )}
      </div>

      {/* List */}
      <div style={{ flex: 1, overflowY: "auto" }} className="scroll">
        {entries.length === 0 ? (
          <div style={{
            display: "flex", flexDirection: "column", alignItems: "center",
            justifyContent: "center", height: "100%", gap: 12, padding: "32px 16px",
          }}>
            <div style={{
              width: 38, height: 38, borderRadius: "var(--radius-md)",
              background: "var(--surface-alt)", border: "1px solid var(--border)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <Clock size={16} style={{ color: "var(--text-muted)", opacity: 0.6 }} />
            </div>
            <p style={{
              fontSize: 11, color: "var(--text-muted)", textAlign: "center",
              lineHeight: 1.65, maxWidth: 148,
            }}>
              Scan results appear here automatically
            </p>
          </div>
        ) : (
          <div style={{ padding: "4px 0 8px" }}>
            {groups.map((group) => (
              <div key={group.label}>
                {/* Day separator */}
                <div style={{
                  padding: "10px 14px 5px",
                  fontSize: 10, fontWeight: 700, letterSpacing: "0.6px",
                  textTransform: "uppercase", color: "var(--text-muted)",
                  userSelect: "none",
                }}>
                  {group.label}
                </div>

                {group.items.map((entry) => {
                  const isSelected = entry.id === selectedId
                  return (
                  <div
                    key={entry.id}
                    className="history-row"
                    onClick={() => onSelect(entry.id)}
                    style={{
                      position: "relative", padding: "8px 14px 8px 16px",
                      cursor: "pointer",
                      background: isSelected ? "var(--accent-surface)" : undefined,
                    }}
                  >
                    {/* Left accent bar */}
                    <div style={{
                      position: "absolute", left: 0, top: 8, bottom: 8, width: 2.5,
                      borderRadius: "0 2px 2px 0",
                      background: isSelected
                        ? "var(--accent)"
                        : entry.is_anomaly ? "var(--danger)" : "var(--success)",
                      opacity: isSelected ? 1 : 0.65,
                    }} />

                    <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                      {/* Type icon */}
                      <div style={{
                        width: 26, height: 26, borderRadius: "var(--radius-sm)", flexShrink: 0,
                        background: entry.is_anomaly ? "var(--danger-surface)" : "var(--accent-surface)",
                        border: `1px solid ${entry.is_anomaly ? "var(--danger-border)" : "var(--accent-border)"}`,
                        display: "flex", alignItems: "center", justifyContent: "center", marginTop: 1,
                        color: entry.is_anomaly ? "var(--danger)" : "var(--accent)",
                      }}>
                        <TypeIcon type={entry.type} />
                      </div>

                      {/* Text content */}
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 }}>
                          <span style={{
                            fontSize: 12, fontWeight: 600, color: "var(--text-primary)",
                            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                            flex: 1,
                          }}>
                            {entry.filename}
                          </span>
                          <button
                            onClick={() => onRemove(entry.id)}
                            className="history-remove"
                            style={{
                              background: "none", border: "none", cursor: "pointer",
                              padding: 2, color: "var(--text-muted)", borderRadius: 3,
                              flexShrink: 0, display: "flex", alignItems: "center",
                              opacity: 0, transition: "opacity 0.1s",
                            }}
                          >
                            <X size={10} />
                          </button>
                        </div>

                        <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 2 }}>
                          <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                            {ago(entry.timestamp)}
                          </span>
                          <span style={{
                            width: 2, height: 2, borderRadius: "50%",
                            background: "var(--border-strong)", flexShrink: 0,
                          }} />
                          <span style={{
                            fontSize: 10, fontWeight: 700,
                            color: entry.is_anomaly ? "var(--danger)" : "var(--success)",
                          }}>
                            {entry.is_anomaly
                              ? `${entry.detections_count} defect${entry.detections_count !== 1 ? "s" : ""}`
                              : "Pass"}
                          </span>
                          {entry.anomaly_rate !== undefined && (
                            <>
                              <span style={{ width: 2, height: 2, borderRadius: "50%", background: "var(--border-strong)", flexShrink: 0 }} />
                              <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                                {entry.anomaly_rate}%
                              </span>
                            </>
                          )}
                        </div>

                        {entry.defect_classes.length > 0 && (
                          <div style={{ display: "flex", gap: 3, marginTop: 5, flexWrap: "wrap" }}>
                            {entry.defect_classes.slice(0, 3).map((cls) => (
                              <span key={cls} style={{
                                fontSize: 9, fontWeight: 700, letterSpacing: "0.5px",
                                textTransform: "uppercase", color: "var(--danger)",
                                background: "var(--danger-surface)",
                                border: "1px solid var(--danger-border)",
                                padding: "1px 5px", borderRadius: 3,
                              }}>
                                {cls}
                              </span>
                            ))}
                            {entry.defect_classes.length > 3 && (
                              <span style={{ fontSize: 9, color: "var(--text-muted)", alignSelf: "center" }}>
                                +{entry.defect_classes.length - 3}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )})}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer stats */}
      {entries.length > 0 && (
        <div style={{
          flexShrink: 0, borderTop: "1px solid var(--border)",
          padding: "9px 14px", display: "flex",
          justifyContent: "space-between", alignItems: "center",
        }}>
          <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
            {entries.length} scan{entries.length !== 1 ? "s" : ""}
          </span>
          <span style={{
            fontSize: 10, fontWeight: 600,
            color: anomalyCount > 0 ? "var(--danger)" : "var(--text-muted)",
          }}>
            {anomalyCount} anomal{anomalyCount === 1 ? "y" : "ies"}
          </span>
        </div>
      )}
    </div>
  )
}

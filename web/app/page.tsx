"use client"

import { useState, useEffect, useCallback } from "react"
import Link from "next/link"
import { Settings, Sun, Moon, ChevronDown, History, Database } from "lucide-react"
import ImageDetector from "@/components/ImageDetector"
import VideoDetector from "@/components/VideoDetector"
import WebcamDetector from "@/components/WebcamDetector"
import HistoryPanel from "@/components/HistoryPanel"
import HistoryDetailView from "@/components/HistoryDetailView"
import { ModelConfig, apiHealth, apiModels } from "@/lib/api"
import { useHistory } from "@/hooks/useHistory"

type Tab = "image" | "video" | "camera"

const DEFAULTS: ModelConfig = {
  weights: "weights/best_m.pt",
  conf: 0.25,
  iou: 0.45,
  imgsz: 800,
  isolate: true,
  mask_overlap_thr: 0.20,
}

function ControlSlider({
  label, value, min, max, step, format, onChange,
}: {
  label: string; value: number; min: number; max: number
  step: number; format?: (v: number) => string; onChange: (v: number) => void
}) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="field-label">{label}</span>
        <span className="text-xs font-mono font-semibold tabular-nums"
          style={{ color: "var(--accent)", letterSpacing: "0.3px" }}>
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          background: `linear-gradient(to right, var(--accent) ${pct}%, var(--border) ${pct}%)`,
        }}
      />
    </div>
  )
}

function SidebarDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 py-1">
      <span className="section-label">{label}</span>
      <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
    </div>
  )
}

export default function Page() {
  const [tab,          setTab]          = useState<Tab>("image")
  const [cfg,          setCfg]          = useState<ModelConfig>(DEFAULTS)
  const [models,       setModels]       = useState<string[]>([])
  const [status,       setStatus]       = useState<"loading" | "ok" | "error">("loading")
  const [device,       setDevice]       = useState("")
  const [sidebarOpen,    setSidebarOpen]    = useState(true)
  const [historyOpen,    setHistoryOpen]    = useState(true)
  const [dark,           setDark]           = useState(true)
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null)

  const { entries, addEntry, removeEntry, clearAll } = useHistory()

  const selectedEntry = entries.find((e) => e.id === selectedEntryId) ?? null

  const handleSelect = (id: string) => setSelectedEntryId((prev) => (prev === id ? null : id))
  const handleBack   = () => setSelectedEntryId(null)

  useEffect(() => { document.documentElement.classList.toggle("dark", dark) }, [dark])

  useEffect(() => {
    document.documentElement.classList.add("dark")
    apiHealth()
      .then((h) => { setStatus("ok"); setDevice(h.device.toUpperCase()) })
      .catch(() => setStatus("error"))
    apiModels().then((ms) => { setModels(ms); if (ms.length > 0) set({ weights: ms[0] }) }).catch(() => {})
  }, [])

  const set = useCallback((p: Partial<ModelConfig>) => setCfg((c) => ({ ...c, ...p })), [])

  const tabs: { id: Tab; label: string; desc: string }[] = [
    { id: "image",  label: "Image",  desc: "Upload one or more images" },
    { id: "video",  label: "Video",  desc: "Process an MP4/AVI/MOV file" },
    { id: "camera", label: "Camera", desc: "Real-time webcam detection" },
  ]

  return (
    <div className="flex flex-col min-h-screen" style={{ background: "var(--bg)" }}>

      {/* ── Header ──────────────────────────────────────────────── */}
      <header
        className="flex-shrink-0 flex items-center justify-between px-5 border-b"
        style={{
          height: 50,
          background: "var(--header-bg)",
          borderColor: "var(--border)",
          boxShadow: "var(--shadow-sm)",
          position: "sticky", top: 0, zIndex: 40,
        }}
      >
        {/* Brand */}
        <div className="flex items-center gap-3">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
            <rect x="1" y="1" width="22" height="22" rx="5" stroke="var(--accent)" strokeWidth="1.5" />
            <path d="M7 12 L10.5 15.5 L17 8.5" stroke="var(--accent)" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" opacity="0.3" />
            <circle cx="12" cy="12" r="3.5" fill="var(--accent)" />
            <circle cx="12" cy="12" r="1.5" fill="var(--header-bg)" />
          </svg>
          <div className="flex items-baseline gap-2">
            <span style={{
              fontSize: 14, fontWeight: 700, letterSpacing: "-0.3px",
              color: "var(--text-primary)",
            }}>
              DefectVision
            </span>
            <span style={{
              fontSize: 9, fontWeight: 700, letterSpacing: "1.1px",
              textTransform: "uppercase", color: "var(--text-muted)",
              padding: "2px 6px", background: "var(--surface-alt)",
              borderRadius: "var(--radius-sm)", border: "1px solid var(--border)",
            }}>
              Inspector
            </span>
          </div>
        </div>

        {/* Right controls */}
        <div className="flex items-center gap-2">
          {/* API status */}
          <div
            className="flex items-center gap-2 px-3 py-1.5 rounded"
            style={{
              fontSize: 11, background: "var(--surface-alt)",
              border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
            }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{
              background: status === "ok" ? "var(--success)" : status === "error" ? "var(--danger)" : "var(--text-muted)",
              boxShadow: status === "ok" ? "0 0 0 2px var(--success-surface)" : undefined,
            }} />
            <span style={{ color: "var(--text-secondary)" }}>
              {status === "ok" ? device : status === "error" ? "Offline" : "Connecting…"}
            </span>
          </div>

          <div style={{ width: 1, height: 18, background: "var(--border)", margin: "0 2px" }} />

          {/* Dataset link */}
          <Link href="/dataset" className="btn btn-secondary" style={{ padding: "5px 10px", textDecoration: "none" }} title="Dataset Manager">
            <Database size={13} />
          </Link>

          <div style={{ width: 1, height: 18, background: "var(--border)", margin: "0 2px" }} />

          {/* History toggle */}
          <button
            onClick={() => setHistoryOpen((v) => !v)}
            className="btn btn-secondary"
            style={{
              padding: "5px 9px",
              background: historyOpen ? "var(--accent-surface)" : undefined,
              borderColor: historyOpen ? "var(--accent-border)" : undefined,
              color: historyOpen ? "var(--accent)" : undefined,
            }}
            title="Toggle history"
          >
            <History size={13} />
          </button>

          {/* Theme */}
          <button
            onClick={() => setDark((d) => !d)}
            className="btn btn-secondary"
            style={{ padding: "5px 9px" }}
            title="Toggle theme"
          >
            {dark ? <Sun size={13} /> : <Moon size={13} />}
          </button>

          {/* Settings toggle */}
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="btn btn-secondary"
            style={{
              padding: "5px 9px",
              background: sidebarOpen ? "var(--accent-surface)" : undefined,
              borderColor: sidebarOpen ? "var(--accent-border)" : undefined,
              color: sidebarOpen ? "var(--accent)" : undefined,
            }}
            title="Settings"
          >
            <Settings size={13} />
          </button>
        </div>
      </header>

      {/* ── Body ────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── History Panel ─────────────────────────────────────── */}
        {historyOpen && (
          <aside
            className="flex-shrink-0 overflow-hidden border-r"
            style={{
              width: 220,
              background: "var(--sidebar-bg)",
              borderColor: "var(--border)",
            }}
          >
            <HistoryPanel
              entries={entries}
              onClear={clearAll}
              onRemove={removeEntry}
              onSelect={handleSelect}
              selectedId={selectedEntryId}
            />
          </aside>
        )}

        {/* ── Main ──────────────────────────────────────────────── */}
        <main className="flex-1 overflow-y-auto scroll flex flex-col min-w-0">

          {selectedEntry ? (
            /* ── History detail view ── */
            <div style={{ maxWidth: 820, margin: "0 auto", padding: "24px 28px", width: "100%" }}>
              <HistoryDetailView entry={selectedEntry} onBack={handleBack} />
            </div>
          ) : (
            <>
              {/* Page header + tabs */}
              <div
                className="flex-shrink-0 border-b"
                style={{ borderColor: "var(--border)", background: "var(--surface)" }}
              >
                <div style={{ maxWidth: 820, margin: "0 auto", padding: "20px 28px 0" }}>
                  <h1 style={{
                    fontSize: 18, fontWeight: 700, color: "var(--text-primary)",
                    letterSpacing: "-0.4px", lineHeight: 1.2,
                  }}>
                    Surface Inspection
                  </h1>
                  <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>
                    {tabs.find((t) => t.id === tab)?.desc}
                  </p>
                </div>

                {/* Tab bar */}
                <div
                  className="flex mt-4"
                  style={{
                    borderBottom: "1px solid var(--border)", marginBottom: -1,
                    maxWidth: 820, margin: "16px auto -1px", padding: "0 28px",
                  }}
                >
                  {tabs.map((t) => (
                    <button
                      key={t.id}
                      onClick={() => setTab(t.id)}
                      style={{
                        padding: "9px 18px", fontSize: 13,
                        fontWeight: tab === t.id ? 600 : 500,
                        color: tab === t.id ? "var(--accent)" : "var(--text-muted)",
                        marginBottom: -1, cursor: "pointer",
                        background: "none",
                        borderTop: "none", borderLeft: "none", borderRight: "none",
                        borderBottom: `2px solid ${tab === t.id ? "var(--accent)" : "transparent"}`,
                        transition: "color 0.15s, border-color 0.15s",
                        fontFamily: "inherit", letterSpacing: "-0.1px",
                      }}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Scanner content */}
              <div style={{ maxWidth: 820, margin: "0 auto", padding: "24px 28px", width: "100%" }}>
                {tab === "image"  && <ImageDetector  config={cfg} onDetection={addEntry} />}
                {tab === "video"  && <VideoDetector  config={cfg} onDetection={addEntry} />}
                {tab === "camera" && <WebcamDetector config={cfg} onDetection={addEntry} />}
              </div>
            </>
          )}
        </main>

        {/* ── Settings Sidebar ──────────────────────────────────── */}
        {sidebarOpen && (
          <aside
            className="flex-shrink-0 overflow-y-auto scroll border-l"
            style={{
              width: 240,
              background: "var(--sidebar-bg)",
              borderColor: "var(--border)",
              padding: "18px 16px",
            }}
          >
            <div className="space-y-5">

              {/* Model */}
              <div className="space-y-2">
                <SidebarDivider label="Model" />
                {models.length > 0 ? (
                  <div style={{ position: "relative" }}>
                    <select
                      value={cfg.weights}
                      onChange={(e) => set({ weights: e.target.value })}
                      className="select-field"
                    >
                      {models.map((m) => (
                        <option key={m} value={m}>{m.split("/").pop()}</option>
                      ))}
                    </select>
                    <ChevronDown size={12} style={{
                      position: "absolute", right: 10, top: "50%",
                      transform: "translateY(-50%)",
                      pointerEvents: "none", color: "var(--text-muted)",
                    }} />
                  </div>
                ) : (
                  <div style={{
                    fontSize: 12, color: "var(--text-muted)", padding: "7px 10px",
                    border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
                    background: "var(--surface)",
                  }}>
                    {status === "error" ? "Backend not running" : "Loading…"}
                  </div>
                )}
              </div>

              {/* Detection */}
              <div className="space-y-4">
                <SidebarDivider label="Detection" />

                <ControlSlider
                  label="Confidence threshold"
                  value={cfg.conf} min={0.05} max={0.95} step={0.05}
                  format={(v) => `${Math.round(v * 100)}%`}
                  onChange={(v) => set({ conf: v })}
                />

                <ControlSlider
                  label="IoU (NMS)"
                  value={cfg.iou} min={0.10} max={0.95} step={0.05}
                  format={(v) => v.toFixed(2)}
                  onChange={(v) => set({ iou: v })}
                />

                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="field-label">Inference size</span>
                    <span className="text-xs font-mono font-semibold"
                      style={{ color: "var(--accent)" }}>{cfg.imgsz}px</span>
                  </div>
                  <div className="flex gap-1">
                    {[320, 480, 640, 800, 1024].map((s) => (
                      <button
                        key={s}
                        className={`seg-btn flex-1 ${cfg.imgsz === s ? "active" : ""}`}
                        onClick={() => set({ imgsz: s })}
                      >
                        {s >= 1000 ? "1k" : s}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Isolation */}
              <div className="space-y-4">
                <SidebarDivider label="Isolation" />

                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="field-label">Product boundary</p>
                    <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2, lineHeight: 1.4 }}>
                      Ignore detections outside the product area
                    </p>
                  </div>
                  <div
                    className="toggle-track"
                    style={{ background: cfg.isolate ? "var(--accent)" : "var(--border-strong)" }}
                    onClick={() => set({ isolate: !cfg.isolate })}
                  >
                    <span
                      className="toggle-thumb"
                      style={{ transform: `translateX(${cfg.isolate ? 14 : 2}px)` }}
                    />
                  </div>
                </div>

                {cfg.isolate && (
                  <ControlSlider
                    label="Min. overlap"
                    value={cfg.mask_overlap_thr} min={0} max={0.50} step={0.05}
                    format={(v) => v.toFixed(2)}
                    onChange={(v) => set({ mask_overlap_thr: v })}
                  />
                )}
              </div>

              {/* System */}
              <div className="space-y-2">
                <SidebarDivider label="System" />
                <div style={{
                  fontSize: 11, padding: "8px 10px",
                  borderRadius: "var(--radius-sm)",
                  border: "1px solid var(--border)",
                  background: "var(--surface)",
                  color: "var(--text-muted)",
                  fontFamily: "monospace", lineHeight: 1.5,
                }}>
                  <div style={{
                    color: "var(--text-secondary)", fontWeight: 500,
                    fontSize: 11, marginBottom: 2,
                  }}>
                    API endpoint
                  </div>
                  {process.env.NEXT_PUBLIC_API_URL || "localhost:8000"}
                </div>
              </div>

            </div>
          </aside>
        )}
      </div>
    </div>
  )
}

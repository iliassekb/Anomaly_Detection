"use client"

import { useState, useEffect } from "react"
import { ThumbsUp, ThumbsDown, X, CheckCircle } from "lucide-react"
import { confirmFeedback, correctFeedback, getClasses, ClassInfo, Polygon } from "@/lib/dataset-api"
import { Detection } from "@/lib/api"
import AnnotationCanvas from "./dataset/AnnotationCanvas"

interface Props {
  imageB64: string
  filename: string
  detections: Detection[]
  annotatedB64: string
}

type State = "idle" | "confirming" | "correcting" | "done"

export default function FeedbackModal({ imageB64, filename, detections, annotatedB64 }: Props) {
  const [state, setState] = useState<State>("idle")
  const [classes, setClasses] = useState<ClassInfo[]>([])
  const [className, setClassName] = useState("")
  const [toast, setToast] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getClasses().then(cls => {
      setClasses(cls)
      // Pre-select the first anomaly class if present
      const anomaly = detections.find(d => d.class.toLowerCase() !== "good")
      const match = cls.find(c => c.name === anomaly?.class) ?? cls[0]
      if (match) setClassName(match.name)
    }).catch(() => {})
  }, [])

  function showToast(msg: string) {
    setToast(msg)
    setTimeout(() => setToast(null), 3500)
  }

  async function handleConfirm() {
    setState("confirming")
    setError(null)
    try {
      const payload = detections.map(d => ({
        label: d.class,
      }))
      await confirmFeedback(imageB64, filename, className || detections[0]?.class || "unknown", payload)
      setState("done")
      showToast("Saved to dataset as confirmed annotation.")
    } catch (e: any) {
      setError(e.message)
      setState("idle")
    }
  }

  if (state === "done") {
    return (
      <div className="flex items-center gap-2 px-4 py-3 rounded" style={{
        background: "var(--success-surface)", border: "1px solid var(--success-border)",
        borderRadius: "var(--radius-sm)",
      }}>
        <CheckCircle size={14} style={{ color: "var(--success)", flexShrink: 0 }} />
        <span style={{ fontSize: 12, color: "var(--success)" }}>Saved to dataset. Thank you for your feedback!</span>
      </div>
    )
  }

  if (state === "correcting") {
    return (
      <div style={{ position: "fixed", inset: 0, zIndex: 60, background: "rgba(0,0,0,0.7)", display: "flex", flexDirection: "column", padding: 24 }}>
        <div className="panel" style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div className="flex items-center justify-between" style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>Correct annotation: {filename}</span>
            <button style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: 4 }} onClick={() => setState("idle")}>
              <X size={18} />
            </button>
          </div>
          <div style={{ flex: 1, padding: 16, overflow: "hidden" }}>
            <AnnotationCanvas
              imageUrl={`data:image/jpeg;base64,${imageB64}`}
              imageName={filename}
              className={className}
              onSaved={() => { setState("done"); showToast("Correction saved to dataset.") }}
              onClose={() => setState("idle")}
            />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div>
      {toast && (
        <div style={{
          position: "fixed", bottom: 24, left: "50%", transform: "translateX(-50%)", zIndex: 100,
          background: "var(--success)", color: "white", padding: "8px 18px", borderRadius: 99,
          fontSize: 13, fontWeight: 500, boxShadow: "var(--shadow-md)",
        }}>
          {toast}
        </div>
      )}

      <div className="panel" style={{ padding: "14px 16px" }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 10 }}>
          Was this prediction correct?
        </div>

        {classes.length > 0 && (
          <div style={{ marginBottom: 10 }}>
            <label className="field-label" style={{ display: "block", marginBottom: 4, fontSize: 11 }}>Class</label>
            <select
              className="select-field"
              value={className}
              onChange={e => setClassName(e.target.value)}
              style={{ fontSize: 12, padding: "6px 28px 6px 8px" }}
            >
              {classes.filter(c => c.name.toLowerCase() !== "good").map(c => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
          </div>
        )}

        {error && <p style={{ fontSize: 11, color: "var(--danger)", marginBottom: 8 }}>{error}</p>}

        <div className="flex gap-2">
          <button
            className="btn btn-secondary"
            style={{ flex: 1, justifyContent: "center", fontSize: 12, padding: "7px" }}
            onClick={handleConfirm}
            disabled={state === "confirming"}
          >
            <ThumbsUp size={13} style={{ color: "var(--success)" }} />
            {state === "confirming" ? "Saving…" : "Yes, save to dataset"}
          </button>
          <button
            className="btn btn-secondary"
            style={{ flex: 1, justifyContent: "center", fontSize: 12, padding: "7px" }}
            onClick={() => setState("correcting")}
          >
            <ThumbsDown size={13} style={{ color: "var(--danger)" }} />
            No, fix it
          </button>
        </div>
      </div>
    </div>
  )
}

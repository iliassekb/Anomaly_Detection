"use client"

import { useCallback, useState } from "react"
import { Upload, Download, Loader2, FileImage } from "lucide-react"
import { predictImage, ImageResult, ModelConfig } from "@/lib/api"
import { HistoryEntry } from "@/hooks/useHistory"
import DetectionResults from "./DetectionResults"

interface Props {
  config: ModelConfig
  onDetection: (entry: Omit<HistoryEntry, "id" | "timestamp">) => void
}

export default function ImageDetector({ config, onDetection }: Props) {
  const [files,    setFiles]    = useState<File[]>([])
  const [results,  setResults]  = useState<(ImageResult | { error: string })[]>([])
  const [loading,  setLoading]  = useState(false)
  const [dragging, setDragging] = useState(false)

  const handleFiles = (incoming: File[]) => {
    setFiles(incoming.filter((f) => f.type.startsWith("image/")))
    setResults([])
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false)
    handleFiles(Array.from(e.dataTransfer.files))
  }, [])

  const run = async () => {
    if (!files.length) return
    setLoading(true); setResults([])
    const out: (ImageResult | { error: string })[] = []
    for (let i = 0; i < files.length; i++) {
      const f = files[i]
      try {
        const result = await predictImage(f, config)
        out.push(result)
        setResults([...out])
        onDetection({
          type: "image",
          filename: f.name,
          is_anomaly: result.is_anomaly,
          detections_count: result.detections.length,
          defect_classes: [...new Set(result.detections.map((d) => d.class))],
          inference_ms: result.inference_ms,
          _imageData: {
            original_b64: result.original_b64,
            annotated_b64: result.annotated_b64,
            detections: result.detections,
          },
        })
      } catch (err) {
        out.push({ error: err instanceof Error ? err.message : "Request failed" })
        setResults([...out])
      }
    }
    setLoading(false)
  }

  const download = (b64: string, name: string) => {
    const a = document.createElement("a")
    a.href = `data:image/jpeg;base64,${b64}`; a.download = name; a.click()
  }

  return (
    <div className="space-y-5">

      {/* Drop zone */}
      <div
        className={`dropzone${dragging ? " active" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => document.getElementById("img-input")?.click()}
      >
        <input
          id="img-input" type="file" accept="image/*" multiple className="hidden"
          onChange={(e) => e.target.files && handleFiles(Array.from(e.target.files))}
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
              Drop images here or click to browse
            </p>
            <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>
              PNG, JPG — multiple files supported
            </p>
          </div>
        </div>
      </div>

      {/* Selected files */}
      {files.length > 0 && (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {files.map((f) => (
              <div
                key={f.name}
                className="flex items-center gap-2 px-3 py-1.5"
                style={{
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  fontSize: 12,
                  color: "var(--text-secondary)",
                  borderRadius: "var(--radius-sm)",
                }}
              >
                <FileImage size={12} style={{ color: "var(--accent)", flexShrink: 0 }} />
                <span className="truncate" style={{ maxWidth: 180 }}>{f.name}</span>
              </div>
            ))}
          </div>
          <button className="btn btn-primary" onClick={run} disabled={loading}>
            {loading
              ? <><Loader2 size={13} className="animate-spin" /> Scanning…</>
              : `Scan ${files.length > 1 ? `${files.length} images` : "image"} for defects`}
          </button>
        </div>
      )}

      {/* Results */}
      {results.map((r, idx) => (
        <div key={idx} className="panel" style={{ padding: "18px 20px" }}>
          {"error" in r ? (
            <p style={{ fontSize: 13, color: "var(--danger)" }}>{r.error}</p>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span style={{
                  fontSize: 13, fontWeight: 600, color: "var(--text-secondary)",
                  maxWidth: "70%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}>
                  {files[idx]?.name}
                </span>
                <button
                  className="btn btn-secondary"
                  style={{ padding: "5px 10px", fontSize: 12 }}
                  onClick={() => download(r.annotated_b64, `annotated_${files[idx]?.name}`)}
                >
                  <Download size={12} /> Export
                </button>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Original",  src: r.original_b64  },
                  { label: "Annotated", src: r.annotated_b64 },
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
                detections={r.detections}
                inference_ms={r.inference_ms}
                is_anomaly={r.is_anomaly}
              />
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

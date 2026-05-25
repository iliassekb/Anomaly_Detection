"use client"

import { useState, useRef } from "react"
import { Upload, FileText } from "lucide-react"
import { ClassInfo, uploadRaw, uploadAnnotated } from "@/lib/dataset-api"

interface Props {
  classes: ClassInfo[]
  onUploaded: () => void
}

type Mode = "raw" | "annotated"

export default function ImageUploader({ classes, onUploaded }: Props) {
  const [mode, setMode] = useState<Mode>("raw")
  const [className, setClassName] = useState(classes[0]?.name ?? "")
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [annotationFile, setAnnotationFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const imgRef = useRef<HTMLInputElement>(null)
  const annRef = useRef<HTMLInputElement>(null)

  const selectedClass = className || classes[0]?.name || ""

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    const files = Array.from(e.dataTransfer.files)
    const img = files.find(f => f.type.startsWith("image/"))
    const ann = files.find(f => f.name.endsWith(".txt") || f.name.endsWith(".json") || (f.name.endsWith(".png") && !f.type.startsWith("image/")))
    if (img) setImageFile(img)
    if (ann) setAnnotationFile(ann)
  }

  async function handleUpload() {
    if (!imageFile || !selectedClass) return
    setUploading(true)
    setResult(null)
    setError(null)
    try {
      if (mode === "raw") {
        const r = await uploadRaw(selectedClass, imageFile)
        setResult(`Uploaded as pending: ${r.filename}`)
      } else {
        if (!annotationFile) { setError("Annotation file required"); setUploading(false); return }
        const r = await uploadAnnotated(selectedClass, imageFile, annotationFile)
        setResult(`Uploaded annotated: ${r.filename}`)
      }
      setImageFile(null)
      setAnnotationFile(null)
      onUploaded()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="space-y-4">
      <span className="section-label">Upload Images</span>

      {/* Mode toggle */}
      <div className="flex gap-1" style={{ background: "var(--surface-alt)", padding: 3, borderRadius: "var(--radius-sm)", display: "inline-flex" }}>
        {(["raw", "annotated"] as Mode[]).map(m => (
          <button key={m} className={`seg-btn ${mode === m ? "active" : ""}`} onClick={() => setMode(m)}>
            {m === "raw" ? "Raw (pending)" : "Annotated"}
          </button>
        ))}
      </div>

      {/* Class selector */}
      <div>
        <label className="field-label" style={{ display: "block", marginBottom: 4 }}>Class</label>
        <select
          className="select-field"
          value={className}
          onChange={e => setClassName(e.target.value)}
        >
          {classes.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
        </select>
      </div>

      {/* Drop zone */}
      <div
        className={`dropzone ${dragging ? "active" : ""}`}
        style={{ padding: "32px 16px", textAlign: "center" }}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => imgRef.current?.click()}
      >
        <Upload size={24} style={{ color: "var(--text-muted)", margin: "0 auto 8px" }} />
        <p style={{ fontSize: 13, color: "var(--text-secondary)" }}>
          {imageFile ? imageFile.name : "Drop image here or click to browse"}
        </p>
        {imageFile && (
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
            {(imageFile.size / 1024).toFixed(1)} KB
          </p>
        )}
        <input ref={imgRef} type="file" accept="image/*" hidden onChange={e => e.target.files?.[0] && setImageFile(e.target.files[0])} />
      </div>

      {/* Annotation file (annotated mode) */}
      {mode === "annotated" && (
        <div
          className={`dropzone ${annotationFile ? "" : ""}`}
          style={{ padding: "16px", textAlign: "center", cursor: "pointer" }}
          onClick={() => annRef.current?.click()}
        >
          <FileText size={18} style={{ color: "var(--text-muted)", margin: "0 auto 6px" }} />
          <p style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            {annotationFile
              ? annotationFile.name
              : "Drop annotation (.txt YOLO · .json COCO · .png mask)"}
          </p>
          <input ref={annRef} type="file" accept=".txt,.json,.png" hidden onChange={e => e.target.files?.[0] && setAnnotationFile(e.target.files[0])} />
        </div>
      )}

      {error && <p style={{ fontSize: 12, color: "var(--danger)" }}>{error}</p>}
      {result && <p style={{ fontSize: 12, color: "var(--success)" }}>{result}</p>}

      <button
        className="btn btn-primary"
        style={{ width: "100%" }}
        onClick={handleUpload}
        disabled={uploading || !imageFile || !selectedClass}
      >
        <Upload size={14} />
        {uploading ? "Uploading…" : "Upload"}
      </button>
    </div>
  )
}

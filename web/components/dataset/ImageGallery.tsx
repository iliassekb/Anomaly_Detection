"use client"

import { useState } from "react"
import { X, Edit3 } from "lucide-react"
import { ClassInfo, ImageEntry, imageUrl } from "@/lib/dataset-api"
import AnnotationCanvas from "./AnnotationCanvas"

interface Props {
  classes: ClassInfo[]
  selectedClass: string
  onSelectClass: (c: string) => void
  images: ImageEntry[]
  onRefresh: () => void
}

export default function ImageGallery({ classes, selectedClass, onSelectClass, images, onRefresh }: Props) {
  const [editing, setEditing] = useState<ImageEntry | null>(null)

  function handleSaved() {
    setEditing(null)
    onRefresh()
  }

  return (
    <div className="space-y-4" style={{ height: "100%" }}>
      {/* Class tabs */}
      <div className="flex gap-1 flex-wrap">
        {classes.map(c => (
          <button
            key={c.name}
            className={`seg-btn ${selectedClass === c.name ? "active" : ""}`}
            onClick={() => onSelectClass(c.name)}
          >
            {c.name}
            <span style={{
              marginLeft: 4, fontSize: 10,
              color: selectedClass === c.name ? "rgba(255,255,255,0.7)" : "var(--text-muted)",
            }}>
              {c.annotated + c.pending}
            </span>
          </button>
        ))}
      </div>

      {images.length === 0 && (
        <p style={{ fontSize: 13, color: "var(--text-muted)", textAlign: "center", padding: "32px 0" }}>
          No images in this class yet.
        </p>
      )}

      {/* Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10 }}>
        {images.map(img => {
          const url = imageUrl(selectedClass, img.filename)
          const isAnnotated = img.status === "annotated"
          return (
            <div
              key={img.key}
              className="panel"
              style={{ overflow: "hidden", cursor: "pointer", position: "relative" }}
              onClick={() => setEditing(img)}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={url}
                alt={img.filename}
                style={{ width: "100%", aspectRatio: "1", objectFit: "cover", display: "block" }}
              />
              <div style={{ padding: "6px 8px" }}>
                <div style={{ fontSize: 11, color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {img.filename}
                </div>
              </div>
              {/* Status badge */}
              <div style={{
                position: "absolute", top: 6, right: 6,
                fontSize: 9, fontWeight: 700, letterSpacing: "0.5px", textTransform: "uppercase",
                padding: "2px 6px", borderRadius: 99,
                background: isAnnotated ? "var(--success-surface)" : "rgba(251,191,36,0.15)",
                color: isAnnotated ? "var(--success)" : "#D97706",
                border: `1px solid ${isAnnotated ? "var(--success-border)" : "rgba(217,119,6,0.25)"}`,
              }}>
                {isAnnotated ? "done" : "pending"}
              </div>
              {/* Edit overlay */}
              <div style={{
                position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
                background: "rgba(0,0,0,0.35)", opacity: 0, transition: "opacity 0.15s",
              }}
                onMouseEnter={e => (e.currentTarget.style.opacity = "1")}
                onMouseLeave={e => (e.currentTarget.style.opacity = "0")}
              >
                <Edit3 size={24} color="white" />
              </div>
            </div>
          )
        })}
      </div>

      {/* Annotation modal */}
      {editing && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 50, background: "rgba(0,0,0,0.7)",
          display: "flex", flexDirection: "column", padding: 24,
        }}>
          <div className="panel" style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            {/* Modal header */}
            <div className="flex items-center justify-between" style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
              <span style={{ fontSize: 14, fontWeight: 600 }}>
                Annotate: {editing.filename}
              </span>
              <button
                style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: 4 }}
                onClick={() => setEditing(null)}
              >
                <X size={18} />
              </button>
            </div>
            <div style={{ flex: 1, padding: 16, overflow: "hidden" }}>
              <AnnotationCanvas
                imageUrl={imageUrl(selectedClass, editing.filename)}
                imageName={editing.filename}
                className={selectedClass}
                sourceKey={editing.status === "pending" ? editing.key : undefined}
                onSaved={handleSaved}
                onClose={() => setEditing(null)}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

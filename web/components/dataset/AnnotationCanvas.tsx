"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { Undo2, Trash2, Save, MousePointer, Crosshair } from "lucide-react"
import { saveAnnotation, samSegment, Polygon } from "@/lib/dataset-api"

interface Props {
  imageUrl: string
  imageName: string
  className: string
  initialPolygons?: Polygon[]
  sourceKey?: string
  onSaved?: () => void
  onClose?: () => void
}

type Tool = "polygon" | "sam"

const CLASS_COLORS = [
  "#818CF8", "#34D399", "#F87171", "#FBBF24", "#60A5FA",
  "#A78BFA", "#F472B6", "#4ADE80", "#FB923C", "#38BDF8",
]

export default function AnnotationCanvas({
  imageUrl, imageName, className, initialPolygons = [], sourceKey, onSaved, onClose,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [tool, setTool] = useState<Tool>("polygon")
  const [polygons, setPolygons] = useState<Polygon[]>(initialPolygons)
  const [currentPoints, setCurrentPoints] = useState<[number, number][]>([])
  const [imgSize, setImgSize] = useState({ w: 1, h: 1 })
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [saving, setSaving] = useState(false)
  const [samLoading, setSamLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [imgB64, setImgB64] = useState("")
  const imgRef = useRef<HTMLImageElement | null>(null)

  // Load image and compute scale
  useEffect(() => {
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.onload = () => {
      imgRef.current = img
      setImgSize({ w: img.naturalWidth, h: img.naturalHeight })

      // Convert image to base64 for SAM
      const c = document.createElement("canvas")
      c.width = img.naturalWidth
      c.height = img.naturalHeight
      const ctx = c.getContext("2d")!
      ctx.drawImage(img, 0, 0)
      setImgB64(c.toDataURL("image/jpeg").split(",")[1])
    }
    img.src = imageUrl
  }, [imageUrl])

  // Compute canvas layout
  useEffect(() => {
    if (!containerRef.current || !imgRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const cw = rect.width || 800
    const ch = rect.height || 500
    const sw = cw / imgSize.w
    const sh = ch / imgSize.h
    const s = Math.min(sw, sh, 1)
    setScale(s)
    setOffset({ x: (cw - imgSize.w * s) / 2, y: (ch - imgSize.h * s) / 2 })
  }, [imgSize])

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !imgRef.current) return
    const ctx = canvas.getContext("2d")!
    const rect = containerRef.current!.getBoundingClientRect()
    canvas.width = rect.width || 800
    canvas.height = rect.height || 500
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw image
    ctx.drawImage(imgRef.current, offset.x, offset.y, imgSize.w * scale, imgSize.h * scale)

    // Draw saved polygons
    polygons.forEach((poly, i) => {
      const pts = poly.points
      if (pts.length < 2) return
      ctx.beginPath()
      ctx.moveTo(pts[0][0] * scale + offset.x, pts[0][1] * scale + offset.y)
      pts.slice(1).forEach(([x, y]) => ctx.lineTo(x * scale + offset.x, y * scale + offset.y))
      ctx.closePath()
      const color = CLASS_COLORS[i % CLASS_COLORS.length]
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.fillStyle = color + "33"
      ctx.fill()

      // Vertex dots
      pts.forEach(([x, y]) => {
        ctx.beginPath()
        ctx.arc(x * scale + offset.x, y * scale + offset.y, 4, 0, Math.PI * 2)
        ctx.fillStyle = color
        ctx.fill()
      })
    })

    // Draw current in-progress polygon
    if (currentPoints.length > 0) {
      ctx.beginPath()
      ctx.moveTo(currentPoints[0][0] * scale + offset.x, currentPoints[0][1] * scale + offset.y)
      currentPoints.slice(1).forEach(([x, y]) => ctx.lineTo(x * scale + offset.x, y * scale + offset.y))
      ctx.strokeStyle = "#FBBF24"
      ctx.lineWidth = 2
      ctx.setLineDash([4, 3])
      ctx.stroke()
      ctx.setLineDash([])

      currentPoints.forEach(([x, y]) => {
        ctx.beginPath()
        ctx.arc(x * scale + offset.x, y * scale + offset.y, 5, 0, Math.PI * 2)
        ctx.fillStyle = "#FBBF24"
        ctx.fill()
      })
    }
  }, [polygons, currentPoints, imgSize, scale, offset])

  function toImageCoords(e: React.MouseEvent<HTMLCanvasElement>): [number, number] {
    const rect = canvasRef.current!.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top
    return [(cx - offset.x) / scale, (cy - offset.y) / scale]
  }

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const [ix, iy] = toImageCoords(e)
    if (ix < 0 || iy < 0 || ix > imgSize.w || iy > imgSize.h) return

    if (tool === "polygon") {
      // Close polygon if clicking near first point
      if (currentPoints.length >= 3) {
        const [fx, fy] = currentPoints[0]
        const dist = Math.hypot(ix - fx, iy - fy)
        if (dist < 10 / scale) {
          setPolygons(p => [...p, { points: currentPoints, class_name: className }])
          setCurrentPoints([])
          return
        }
      }
      setCurrentPoints(p => [...p, [ix, iy]])
    }
  }

  function handleCanvasDblClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (tool === "polygon" && currentPoints.length >= 3) {
      setPolygons(p => [...p, { points: currentPoints, class_name: className }])
      setCurrentPoints([])
    }
  }

  async function handleSamClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (tool !== "sam" || !imgB64) return
    const [ix, iy] = toImageCoords(e)
    if (ix < 0 || iy < 0 || ix > imgSize.w || iy > imgSize.h) return
    setSamLoading(true)
    setError(null)
    try {
      const result = await samSegment(imgB64, ix, iy)
      // SAM returns normalized coords, convert to pixel
      const pts: [number, number][] = result.polygon.map(([nx, ny]) => [nx * imgSize.w, ny * imgSize.h])
      setPolygons(p => [...p, { points: pts, class_name: className }])
    } catch (e: any) {
      setError(`SAM error: ${e.message}`)
    } finally {
      setSamLoading(false)
    }
  }

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (tool === "sam") handleSamClick(e)
    else handleCanvasClick(e)
  }

  async function handleSave() {
    if (polygons.length === 0) { setError("Draw at least one polygon first"); return }
    setSaving(true)
    setError(null)
    try {
      await saveAnnotation(className, imageName, polygons, imgSize.w, imgSize.h, sourceKey)
      onSaved?.()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 8 }}>
      {/* Toolbar */}
      <div className="flex items-center gap-2 flex-wrap">
        <div className="flex gap-1" style={{ background: "var(--surface-alt)", padding: 3, borderRadius: "var(--radius-sm)", display: "inline-flex" }}>
          <button className={`seg-btn ${tool === "polygon" ? "active" : ""}`} onClick={() => setTool("polygon")} title="Polygon tool">
            <MousePointer size={12} style={{ display: "inline" }} /> Polygon
          </button>
          <button className={`seg-btn ${tool === "sam" ? "active" : ""}`} onClick={() => setTool("sam")} title="SAM click-to-segment">
            <Crosshair size={12} style={{ display: "inline" }} /> SAM
          </button>
        </div>

        <button
          className="btn btn-secondary"
          style={{ padding: "5px 10px" }}
          onClick={() => {
            if (currentPoints.length > 0) setCurrentPoints([])
            else setPolygons(p => p.slice(0, -1))
          }}
          title="Undo"
        >
          <Undo2 size={13} />
        </button>

        <button
          className="btn btn-secondary"
          style={{ padding: "5px 10px", color: "var(--danger)" }}
          onClick={() => { setPolygons([]); setCurrentPoints([]) }}
          title="Clear all"
        >
          <Trash2 size={13} />
        </button>

        <div style={{ flex: 1 }} />

        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          {polygons.length} polygon{polygons.length !== 1 ? "s" : ""}
          {currentPoints.length > 0 && ` · ${currentPoints.length} pts`}
          {samLoading && " · SAM running…"}
        </span>

        {onClose && (
          <button className="btn btn-secondary" style={{ padding: "5px 12px" }} onClick={onClose}>
            Cancel
          </button>
        )}
        <button className="btn btn-primary" style={{ padding: "5px 12px" }} onClick={handleSave} disabled={saving}>
          <Save size={13} />
          {saving ? "Saving…" : "Save"}
        </button>
      </div>

      {error && <p style={{ fontSize: 12, color: "var(--danger)", margin: 0 }}>{error}</p>}

      {/* Canvas */}
      <div
        ref={containerRef}
        style={{
          flex: 1, position: "relative", overflow: "hidden",
          background: "var(--surface-alt)", borderRadius: "var(--radius-md)",
          border: "1px solid var(--border)", minHeight: 400,
          cursor: tool === "sam" ? "crosshair" : "default",
        }}
      >
        <canvas
          ref={canvasRef}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
          onClick={handleClick}
          onDoubleClick={handleCanvasDblClick}
        />
        {tool === "polygon" && (
          <div style={{
            position: "absolute", bottom: 8, left: "50%", transform: "translateX(-50%)",
            fontSize: 11, color: "var(--text-muted)", background: "var(--surface)",
            padding: "3px 10px", borderRadius: 99, border: "1px solid var(--border)",
            pointerEvents: "none",
          }}>
            Click to add points · Double-click or click first point to close
          </div>
        )}
        {tool === "sam" && (
          <div style={{
            position: "absolute", bottom: 8, left: "50%", transform: "translateX(-50%)",
            fontSize: 11, color: "var(--text-muted)", background: "var(--surface)",
            padding: "3px 10px", borderRadius: 99, border: "1px solid var(--border)",
            pointerEvents: "none",
          }}>
            Click on the object to auto-segment
          </div>
        )}
      </div>
    </div>
  )
}

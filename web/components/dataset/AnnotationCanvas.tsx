"use client"

import { useState, useRef, useEffect } from "react"
import { Undo2, Trash2, Save, MousePointer, Crosshair, CheckCircle } from "lucide-react"
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

interface SamPoint { x: number; y: number; label: number }

const CLASS_COLORS = [
  "#818CF8", "#34D399", "#F87171", "#FBBF24", "#60A5FA",
  "#A78BFA", "#F472B6", "#4ADE80", "#FB923C", "#38BDF8",
]

function pointInPolygon(px: number, py: number, pts: [number, number][]): boolean {
  let inside = false
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j]
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi)
      inside = !inside
  }
  return inside
}

export default function AnnotationCanvas({
  imageUrl, imageName, className, initialPolygons = [], sourceKey, onSaved, onClose,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const [tool, setTool] = useState<Tool>("polygon")
  const [polygons, setPolygons] = useState<Polygon[]>(initialPolygons)
  const [currentPoints, setCurrentPoints] = useState<[number, number][]>([])   // polygon mode
  const [samPoints, setSamPoints] = useState<SamPoint[]>([])                   // SAM mode accumulated
  const [currentSamPolygon, setCurrentSamPolygon] = useState<[number, number][] | null>(null)
  const [imgSize, setImgSize] = useState({ w: 1, h: 1 })
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [saving, setSaving] = useState(false)
  const [samLoading, setSamLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [imgB64, setImgB64] = useState("")
  const imgRef = useRef<HTMLImageElement | null>(null)

  // Load image for display + fetch blob → base64 for SAM (avoids canvas taint issues)
  useEffect(() => {
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.onload = () => {
      imgRef.current = img
      setImgSize({ w: img.naturalWidth, h: img.naturalHeight })
    }
    img.src = imageUrl

    setImgB64("")
    fetch(imageUrl)
      .then(r => r.blob())
      .then(blob => new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve((reader.result as string).split(",")[1] ?? "")
        reader.onerror = reject
        reader.readAsDataURL(blob)
      }))
      .then(b64 => { if (b64) setImgB64(b64) })
      .catch(() => {})
  }, [imageUrl])

  // Compute scale/offset
  useEffect(() => {
    if (!containerRef.current || !imgRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const cw = rect.width || 800, ch = rect.height || 500
    const s = Math.min(cw / imgSize.w, ch / imgSize.h, 1)
    setScale(s)
    setOffset({ x: (cw - imgSize.w * s) / 2, y: (ch - imgSize.h * s) / 2 })
  }, [imgSize])

  // Draw everything
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !imgRef.current) return
    const ctx = canvas.getContext("2d")!
    const rect = containerRef.current!.getBoundingClientRect()
    canvas.width = rect.width || 800
    canvas.height = rect.height || 500
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(imgRef.current, offset.x, offset.y, imgSize.w * scale, imgSize.h * scale)

    const toC = (x: number, y: number) => [x * scale + offset.x, y * scale + offset.y] as [number, number]

    // Finalized polygons
    polygons.forEach((poly, i) => {
      const pts = poly.points
      if (pts.length < 2) return
      const color = CLASS_COLORS[i % CLASS_COLORS.length]
      ctx.beginPath()
      const [cx0, cy0] = toC(pts[0][0], pts[0][1])
      ctx.moveTo(cx0, cy0)
      pts.slice(1).forEach(([x, y]) => { const [cx, cy] = toC(x, y); ctx.lineTo(cx, cy) })
      ctx.closePath()
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke()
      ctx.fillStyle = color + "33"; ctx.fill()
      pts.forEach(([x, y]) => {
        const [cx, cy] = toC(x, y)
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2)
        ctx.fillStyle = color; ctx.fill()
      })
    })

    // Current SAM polygon (being refined)
    if (currentSamPolygon) {
      const color = CLASS_COLORS[polygons.length % CLASS_COLORS.length]
      ctx.beginPath()
      const [cx0, cy0] = toC(currentSamPolygon[0][0], currentSamPolygon[0][1])
      ctx.moveTo(cx0, cy0)
      currentSamPolygon.slice(1).forEach(([x, y]) => { const [cx, cy] = toC(x, y); ctx.lineTo(cx, cy) })
      ctx.closePath()
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke()
      ctx.fillStyle = color + "44"; ctx.fill()
    }

    // SAM prompt points
    samPoints.forEach(p => {
      const [cx, cy] = toC(p.x, p.y)
      ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2)
      ctx.fillStyle = p.label === 1 ? "#22C55E" : "#EF4444"
      ctx.fill()
      ctx.strokeStyle = "white"; ctx.lineWidth = 2; ctx.stroke()
    })

    // In-progress polygon (polygon mode)
    if (currentPoints.length > 0) {
      ctx.beginPath()
      const [cx0, cy0] = toC(currentPoints[0][0], currentPoints[0][1])
      ctx.moveTo(cx0, cy0)
      currentPoints.slice(1).forEach(([x, y]) => { const [cx, cy] = toC(x, y); ctx.lineTo(cx, cy) })
      ctx.strokeStyle = "#FBBF24"; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([])
      currentPoints.forEach(([x, y]) => {
        const [cx, cy] = toC(x, y)
        ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2)
        ctx.fillStyle = "#FBBF24"; ctx.fill()
      })
    }
  }, [polygons, currentPoints, currentSamPolygon, samPoints, imgSize, scale, offset])

  function toImageCoords(e: React.MouseEvent<HTMLCanvasElement>): [number, number] {
    const rect = canvasRef.current!.getBoundingClientRect()
    return [(e.clientX - rect.left - offset.x) / scale, (e.clientY - rect.top - offset.y) / scale]
  }

  function handlePolygonClick(ix: number, iy: number) {
    if (currentPoints.length >= 3) {
      const [fx, fy] = currentPoints[0]
      if (Math.hypot(ix - fx, iy - fy) < 10 / scale) {
        setPolygons(p => [...p, { points: currentPoints, class_name: className }])
        setCurrentPoints([])
        return
      }
    }
    setCurrentPoints(p => [...p, [ix, iy]])
  }

  async function handleSamClick(ix: number, iy: number) {
    if (!imgB64) return
    // Determine label: inside current polygon → reduce (0), outside → extend (1)
    let label = 1
    if (currentSamPolygon && pointInPolygon(ix, iy, currentSamPolygon)) {
      label = 0
    }
    const newPoints = [...samPoints, { x: ix, y: iy, label }]
    setSamPoints(newPoints)
    setSamLoading(true)
    setError(null)
    try {
      const result = await samSegment(imgB64, newPoints)
      // SAM returns normalized coords → convert to pixel
      const pts: [number, number][] = result.polygon.map(([nx, ny]) => [nx * imgSize.w, ny * imgSize.h])
      setCurrentSamPolygon(pts)
    } catch (e: any) {
      setError(`SAM error: ${e.message}`)
    } finally {
      setSamLoading(false)
    }
  }

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const [ix, iy] = toImageCoords(e)
    if (ix < 0 || iy < 0 || ix > imgSize.w || iy > imgSize.h) return
    if (tool === "sam") handleSamClick(ix, iy)
    else handlePolygonClick(ix, iy)
  }

  function handleCanvasDblClick() {
    if (tool === "polygon" && currentPoints.length >= 3) {
      setPolygons(p => [...p, { points: currentPoints, class_name: className }])
      setCurrentPoints([])
    }
  }

  function doneSamPolygon() {
    if (!currentSamPolygon) return
    setPolygons(p => [...p, { points: currentSamPolygon, class_name: className }])
    setCurrentSamPolygon(null)
    setSamPoints([])
  }

  function handleUndo() {
    if (tool === "sam") {
      if (samPoints.length > 0) {
        const newPts = samPoints.slice(0, -1)
        setSamPoints(newPts)
        if (newPts.length === 0) { setCurrentSamPolygon(null); return }
        // Re-run SAM with remaining points
        if (imgB64) {
          setSamLoading(true)
          samSegment(imgB64, newPts).then(result => {
            const pts: [number, number][] = result.polygon.map(([nx, ny]) => [nx * imgSize.w, ny * imgSize.h])
            setCurrentSamPolygon(pts)
          }).catch(() => setCurrentSamPolygon(null)).finally(() => setSamLoading(false))
        }
      } else {
        setPolygons(p => p.slice(0, -1))
      }
    } else {
      if (currentPoints.length > 0) setCurrentPoints(p => p.slice(0, -1))
      else setPolygons(p => p.slice(0, -1))
    }
  }

  function handleClear() {
    setPolygons([]); setCurrentPoints([])
    setCurrentSamPolygon(null); setSamPoints([])
  }

  async function handleSave() {
    // Auto-finalize current SAM polygon if pending
    const allPolygons = currentSamPolygon
      ? [...polygons, { points: currentSamPolygon, class_name: className }]
      : polygons
    if (allPolygons.length === 0) { setError("Draw at least one polygon first"); return }
    setSaving(true); setError(null)
    try {
      await saveAnnotation(className, imageName, allPolygons, imgSize.w, imgSize.h, sourceKey)
      onSaved?.()
    } catch (e: any) { setError(e.message) }
    finally { setSaving(false) }
  }

  const hasSamInProgress = tool === "sam" && currentSamPolygon !== null

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 8 }}>
      {/* Toolbar */}
      <div className="flex items-center gap-2 flex-wrap">
        <div className="flex gap-1" style={{ background: "var(--surface-alt)", padding: 3, borderRadius: "var(--radius-sm)", display: "inline-flex" }}>
          <button className={`seg-btn ${tool === "polygon" ? "active" : ""}`} onClick={() => setTool("polygon")}>
            <MousePointer size={12} style={{ display: "inline", marginRight: 4 }} />Polygon
          </button>
          <button className={`seg-btn ${tool === "sam" ? "active" : ""}`} onClick={() => setTool("sam")}>
            <Crosshair size={12} style={{ display: "inline", marginRight: 4 }} />SAM
          </button>
        </div>

        <button className="btn btn-secondary" style={{ padding: "5px 10px" }} onClick={handleUndo} title="Undo">
          <Undo2 size={13} />
        </button>
        <button className="btn btn-secondary" style={{ padding: "5px 10px", color: "var(--danger)" }} onClick={handleClear} title="Clear all">
          <Trash2 size={13} />
        </button>

        {/* SAM "Done with this polygon" button */}
        {hasSamInProgress && (
          <button
            className="btn btn-secondary"
            style={{ padding: "5px 14px", color: "var(--success)", borderColor: "var(--success-border)" }}
            onClick={doneSamPolygon}
          >
            <CheckCircle size={13} />
            Done with this polygon
          </button>
        )}

        <div style={{ flex: 1 }} />

        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          {polygons.length} polygon{polygons.length !== 1 ? "s" : ""}
          {hasSamInProgress && ` · 1 in progress`}
          {samLoading && " · SAM running…"}
        </span>

        {onClose && (
          <button className="btn btn-secondary" style={{ padding: "5px 12px" }} onClick={onClose}>Cancel</button>
        )}
        <button className="btn btn-primary" style={{ padding: "5px 12px" }} onClick={handleSave} disabled={saving}>
          <Save size={13} />
          {saving ? "Saving…" : "Save"}
        </button>
      </div>

      {error && <p style={{ fontSize: 12, color: "var(--danger)", margin: 0 }}>{error}</p>}

      {/* Legend for SAM mode */}
      {tool === "sam" && (
        <div className="flex gap-4" style={{ fontSize: 11, color: "var(--text-muted)" }}>
          <span><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#22C55E", marginRight: 4 }} />Click outside → extend polygon</span>
          <span><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#EF4444", marginRight: 4 }} />Click inside → reduce polygon</span>
        </div>
      )}

      {/* Canvas */}
      <div
        ref={containerRef}
        style={{
          flex: 1, position: "relative", overflow: "hidden",
          background: "var(--surface-alt)", borderRadius: "var(--radius-md)",
          border: "1px solid var(--border)", minHeight: 400,
          cursor: tool === "sam" ? "crosshair" : "crosshair",
        }}
      >
        <canvas
          ref={canvasRef}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
          onClick={handleCanvasClick}
          onDoubleClick={handleCanvasDblClick}
        />
        {tool === "polygon" && (
          <div style={{
            position: "absolute", bottom: 8, left: "50%", transform: "translateX(-50%)",
            fontSize: 11, color: "var(--text-muted)", background: "var(--surface)",
            padding: "3px 10px", borderRadius: 99, border: "1px solid var(--border)", pointerEvents: "none",
          }}>
            Click to add points · Double-click or click first point to close
          </div>
        )}
        {tool === "sam" && !hasSamInProgress && (
          <div style={{
            position: "absolute", bottom: 8, left: "50%", transform: "translateX(-50%)",
            fontSize: 11, color: "var(--text-muted)", background: "var(--surface)",
            padding: "3px 10px", borderRadius: 99, border: "1px solid var(--border)", pointerEvents: "none",
          }}>
            Click on the object to auto-segment
          </div>
        )}
        {hasSamInProgress && (
          <div style={{
            position: "absolute", bottom: 8, left: "50%", transform: "translateX(-50%)",
            fontSize: 11, color: "var(--text-muted)", background: "var(--surface)",
            padding: "3px 10px", borderRadius: 99, border: "1px solid var(--border)", pointerEvents: "none",
          }}>
            Click outside to extend · Click inside to reduce · "Done" to finalize
          </div>
        )}
      </div>
    </div>
  )
}

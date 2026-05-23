"use client"

import { useRef, useState, useEffect, useCallback } from "react"
import { Camera, CameraOff, Zap, AlertTriangle, CheckCircle } from "lucide-react"
import { Detection, ModelConfig } from "@/lib/api"
import { HistoryEntry } from "@/hooks/useHistory"

interface Props {
  config: ModelConfig
  onDetection: (entry: Omit<HistoryEntry, "id" | "timestamp">) => void
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

function sleep(ms: number) { return new Promise<void>((r) => setTimeout(r, ms)) }
function toBlob(c: HTMLCanvasElement, q: number): Promise<Blob | null> {
  return new Promise((r) => c.toBlob(r, "image/jpeg", q))
}

export default function WebcamDetector({ config, onDetection }: Props) {
  const videoRef   = useRef<HTMLVideoElement>(null)
  const captureRef = useRef<HTMLCanvasElement>(null)
  const streamRef  = useRef<MediaStream | null>(null)
  const runningRef = useRef(false)
  const pendingRef = useRef<Blob | null>(null)
  const sessionAnomalyCountRef = useRef(0)
  const sessionClassesRef = useRef<Set<string>>(new Set())

  const [isStreaming,  setIsStreaming]  = useState(false)
  const [annotatedSrc, setAnnotatedSrc] = useState<string | null>(null)
  const [detections,   setDetections]   = useState<Detection[]>([])
  const [isAnomaly,    setIsAnomaly]    = useState(false)
  const [inferMs,      setInferMs]      = useState(0)
  const [fps,          setFps]          = useState(0)
  const [error,        setError]        = useState<string | null>(null)
  const [rtImgsz,      setRtImgsz]      = useState(320)

  useEffect(() => () => {
    runningRef.current = false
    streamRef.current?.getTracks().forEach((t) => t.stop())
  }, [])

  const startCamera = useCallback(async () => {
    setError(null)
    sessionAnomalyCountRef.current = 0
    sessionClassesRef.current = new Set()
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
        audio: false,
      })
      streamRef.current = stream
      videoRef.current!.srcObject = stream
      await videoRef.current!.play()
      setIsStreaming(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Camera access denied")
    }
  }, [])

  const stopCamera = useCallback(() => {
    runningRef.current = false
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    pendingRef.current = null
    if (sessionAnomalyCountRef.current > 0) {
      onDetection({
        type: "camera",
        filename: "Camera session",
        is_anomaly: true,
        detections_count: sessionAnomalyCountRef.current,
        defect_classes: [...sessionClassesRef.current],
      })
    }
    setIsStreaming(false); setAnnotatedSrc(null); setDetections([])
    setIsAnomaly(false); setFps(0); setInferMs(0)
  }, [onDetection])

  // Two concurrent async loops: capture at ~15fps, infer as fast as backend allows
  useEffect(() => {
    if (!isStreaming) return
    runningRef.current = true
    pendingRef.current = null

    const captureLoop = async () => {
      const canvas = captureRef.current!
      while (runningRef.current) {
        const video = videoRef.current
        if (!video || video.readyState < 2 || video.videoWidth === 0) {
          await sleep(80); continue
        }
        const scale = Math.min(rtImgsz / video.videoWidth, rtImgsz / video.videoHeight, 1)
        canvas.width  = Math.max(1, Math.round(video.videoWidth  * scale))
        canvas.height = Math.max(1, Math.round(video.videoHeight * scale))
        canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height)
        const blob = await toBlob(canvas, 0.75)
        if (blob) pendingRef.current = blob
        await sleep(67)
      }
    }

    let frameCount = 0, fpsTs = performance.now()

    const inferLoop = async () => {
      while (runningRef.current) {
        if (!pendingRef.current) { await sleep(20); continue }

        const blob = pendingRef.current
        pendingRef.current = null

        const form = new FormData()
        form.append("file",            new File([blob], "frame.jpg", { type: "image/jpeg" }))
        form.append("weights",          config.weights)
        form.append("conf",             config.conf.toString())
        form.append("iou",              config.iou.toString())
        form.append("imgsz",            rtImgsz.toString())
        form.append("isolate",          config.isolate.toString())
        form.append("mask_overlap_thr", config.mask_overlap_thr.toString())
        form.append("return_original",  "false")

        const t0 = performance.now()
        try {
          const res = await fetch(`${API_URL}/api/predict/image`, { method: "POST", body: form })
          if (!res.ok) { await sleep(200); continue }
          const data = await res.json()
          const elapsed = Math.round(performance.now() - t0)
          if (!runningRef.current) break

          setAnnotatedSrc(`data:image/jpeg;base64,${data.annotated_b64}`)
          setDetections(data.detections ?? [])
          setIsAnomaly(data.is_anomaly ?? false)
          setInferMs(elapsed)
          if (data.is_anomaly) {
            sessionAnomalyCountRef.current += (data.detections ?? []).length
            ;(data.detections ?? []).forEach((d: { class: string }) => sessionClassesRef.current.add(d.class))
          }

          frameCount++
          const now = performance.now()
          if (now - fpsTs >= 1000) {
            setFps(Math.round((frameCount * 1000) / (now - fpsTs)))
            frameCount = 0; fpsTs = now
          }
        } catch { await sleep(300) }
      }
    }

    captureLoop()
    inferLoop()
    return () => { runningRef.current = false }
  }, [isStreaming, config, rtImgsz])

  const grouped = detections.reduce<Record<string, { count: number; maxConf: number }>>((acc, d) => {
    if (!acc[d.class]) acc[d.class] = { count: 0, maxConf: 0 }
    acc[d.class].count++
    acc[d.class].maxConf = Math.max(acc[d.class].maxConf, d.conf)
    return acc
  }, {})

  return (
    <div className="space-y-4">
      <canvas ref={captureRef} className="hidden" />

      {/* Single camera panel */}
      <div
        className="relative overflow-hidden"
        style={{
          borderRadius: "var(--radius-md)",
          border: `1px solid ${isStreaming && isAnomaly ? "var(--danger-border)" : "var(--border)"}`,
          background: "var(--surface-alt)",
          minHeight: 260,
          aspectRatio: "4/3",
          transition: "border-color 0.2s",
        }}
      >
        {/* Raw video feed (always rendered while streaming) */}
        <video
          ref={videoRef} playsInline muted
          className="w-full h-full object-cover"
          style={{ display: isStreaming ? "block" : "none" }}
        />

        {/* Annotated frame overlaid on top of video */}
        {isStreaming && annotatedSrc && (
          <img
            src={annotatedSrc}
            alt="detection"
            className="absolute inset-0 w-full h-full object-cover"
          />
        )}

        {/* Idle placeholder */}
        {!isStreaming && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <div
              className="w-12 h-12 rounded flex items-center justify-center"
              style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
            >
              <Camera size={22} style={{ color: "var(--text-muted)" }} />
            </div>
            <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Press Start to begin</p>
          </div>
        )}

        {/* LIVE badge */}
        {isStreaming && (
          <div className="absolute bottom-2 left-2">
            <span
              className="flex items-center gap-1.5"
              style={{
                fontSize: 11, fontWeight: 700, color: "#fff",
                background: "rgba(0,0,0,0.55)", padding: "3px 8px",
                borderRadius: "var(--radius-sm)", letterSpacing: "0.5px",
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
              LIVE
            </span>
          </div>
        )}

        {/* Status badge (top-left) */}
        {isStreaming && (
          <div className="absolute top-2 left-2">
            <span
              className="flex items-center gap-1.5"
              style={{
                fontSize: 11, fontWeight: 700, color: "#fff",
                background: isAnomaly ? "rgba(220,38,38,0.82)" : "rgba(22,163,74,0.82)",
                padding: "3px 8px", borderRadius: "var(--radius-sm)",
              }}
            >
              {isAnomaly ? <AlertTriangle size={10} /> : <CheckCircle size={10} />}
              {isAnomaly ? "Defect" : "OK"}
            </span>
          </div>
        )}

        {/* Perf metrics (top-right) */}
        {isStreaming && (
          <div className="absolute top-2 right-2 flex flex-col gap-1 items-end">
            {[`${inferMs} ms`, `${fps} fps`].map((label) => (
              <span
                key={label}
                style={{
                  fontSize: 10, fontFamily: "monospace", fontWeight: 600,
                  color: "rgba(255,255,255,0.85)",
                  background: "rgba(0,0,0,0.5)", padding: "2px 7px",
                  borderRadius: "var(--radius-sm)",
                }}
              >
                {label}
              </span>
            ))}
          </div>
        )}

        {/* Detection labels (bottom-right) */}
        {isStreaming && Object.entries(grouped).length > 0 && (
          <div className="absolute bottom-2 right-2 flex flex-col gap-1 items-end">
            {Object.entries(grouped).map(([cls, info]) => (
              <span
                key={cls}
                style={{
                  fontSize: 11, fontWeight: 700, color: "#fff",
                  background: "rgba(220,38,38,0.82)", padding: "3px 9px",
                  borderRadius: "var(--radius-sm)",
                }}
              >
                {cls} {Math.round(info.maxConf * 100)}%
                {info.count > 1 && <span style={{ opacity: 0.75, marginLeft: 4 }}>×{info.count}</span>}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3 items-center">
        {!isStreaming ? (
          <button className="btn btn-primary" onClick={startCamera}>
            <Camera size={14} /> Start Detection
          </button>
        ) : (
          <button className="btn btn-secondary" onClick={stopCamera}>
            <CameraOff size={14} /> Stop
          </button>
        )}

        <div className="flex items-center gap-2">
          <span className="section-label">RT size</span>
          <div className="flex gap-1">
            {([320, 480, 640] as const).map((s) => (
              <button
                key={s}
                className={`seg-btn${rtImgsz === s ? " active" : ""}`}
                onClick={() => setRtImgsz(s)}
              >
                {s}{s === 320 ? " ⚡" : ""}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Guidance */}
      {!isStreaming && !error && (
        <div className="panel" style={{ padding: "14px 16px" }}>
          <p style={{
            fontSize: 11, fontWeight: 700, letterSpacing: "0.7px",
            textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 8,
          }}>
            Performance tips
          </p>
          <ul style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 2 }}>
            <li>· Use <strong style={{ color: "var(--text-primary)" }}>320 px</strong> for fastest throughput (recommended)</li>
            <li>· Disable <strong style={{ color: "var(--text-primary)" }}>Product boundary</strong> isolation to skip the segmentation step</li>
            <li>· A GPU backend brings inference below 80 ms per frame</li>
          </ul>
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
    </div>
  )
}

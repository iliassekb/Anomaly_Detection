const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export interface Detection {
  class: string
  class_id: number
  conf: number
}

export interface ImageResult {
  original_b64: string
  annotated_b64: string
  detections: Detection[]
  inference_ms: number
  is_anomaly: boolean
  class_names: string[]
}

export interface VideoStats {
  total_frames: number
  anomaly_frames: number
  anomaly_rate: number
  total_detections: number
  detections_by_class: Record<string, number>
}

export interface ModelConfig {
  weights: string
  conf: number
  iou: number
  imgsz: number
  isolate: boolean
  mask_overlap_thr: number
}

export async function apiHealth(): Promise<{ status: string; device: string }> {
  const res = await fetch(`${API_URL}/api/health`, { signal: AbortSignal.timeout(5000) })
  return res.json()
}

export async function apiModels(): Promise<string[]> {
  const res = await fetch(`${API_URL}/api/models`)
  const data = await res.json()
  return data.models
}

export async function predictImage(file: File, cfg: ModelConfig): Promise<ImageResult> {
  const form = new FormData()
  form.append("file", file)
  form.append("weights", cfg.weights)
  form.append("conf", cfg.conf.toString())
  form.append("iou", cfg.iou.toString())
  form.append("imgsz", cfg.imgsz.toString())
  form.append("isolate", cfg.isolate.toString())
  form.append("mask_overlap_thr", cfg.mask_overlap_thr.toString())
  const res = await fetch(`${API_URL}/api/predict/image`, { method: "POST", body: form })
  if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function predictVideo(
  file: File,
  cfg: ModelConfig,
  step: number
): Promise<{ blob: Blob; stats: VideoStats }> {
  const form = new FormData()
  form.append("file", file)
  form.append("weights", cfg.weights)
  form.append("conf", cfg.conf.toString())
  form.append("iou", cfg.iou.toString())
  form.append("imgsz", cfg.imgsz.toString())
  form.append("step", step.toString())
  const res = await fetch(`${API_URL}/api/predict/video`, { method: "POST", body: form })
  if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`)
  const statsHeader = res.headers.get("X-Stats")
  const stats: VideoStats = statsHeader
    ? JSON.parse(statsHeader)
    : { total_frames: 0, anomaly_frames: 0, anomaly_rate: 0, total_detections: 0, detections_by_class: {} }
  const blob = await res.blob()
  return { blob, stats }
}

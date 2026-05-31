const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

// ── Types ─────────────────────────────────────────────────────────

export interface ClassInfo {
  name: string
  id: number
  annotated: number
  pending: number
}

export interface ImageEntry {
  filename: string
  status: "annotated" | "pending"
  key: string
}

export interface Polygon {
  points: [number, number][]  // pixel coords
  class_name?: string
}

export interface TrainingStatus {
  running: boolean
  current_epoch: number
  total_epochs: number
  run_id: string | null
  error: string | null
  stop_requested: boolean
  current_metrics: Record<string, number>
}

export interface TrainingRun {
  run_id: string
  status: string
  start_time: number
  end_time: number | null
  metrics: Record<string, number>
  params: Record<string, string>
  tags: Record<string, string>
}

// ── Class endpoints ───────────────────────────────────────────────

export async function getClasses(): Promise<ClassInfo[]> {
  const res = await fetch(`${API_URL}/api/dataset/classes`, {
    signal: AbortSignal.timeout(10_000),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return data.classes
}

export async function createClass(name: string): Promise<ClassInfo> {
  const res = await fetch(`${API_URL}/api/dataset/classes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
    signal: AbortSignal.timeout(10_000),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function deleteClass(name: string): Promise<void> {
  const res = await fetch(`${API_URL}/api/dataset/classes/${encodeURIComponent(name)}`, {
    method: "DELETE",
    signal: AbortSignal.timeout(10_000),
  })
  if (!res.ok) throw new Error(await res.text())
}

// ── Image listing ─────────────────────────────────────────────────

export async function getImages(className: string): Promise<ImageEntry[]> {
  const res = await fetch(
    `${API_URL}/api/dataset/images/${encodeURIComponent(className)}`,
    { signal: AbortSignal.timeout(15_000) },
  )
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return data.images
}

export function imageUrl(className: string, filename: string): string {
  return `${API_URL}/api/dataset/image/${encodeURIComponent(className)}/${encodeURIComponent(filename)}`
}

// ── Upload endpoints ──────────────────────────────────────────────

export async function uploadRaw(className: string, file: File): Promise<{ filename: string; status: string }> {
  const form = new FormData()
  form.append("class_name", className)
  form.append("file", file)
  const res = await fetch(`${API_URL}/api/dataset/upload/raw`, {
    method: "POST", body: form,
    signal: AbortSignal.timeout(30_000),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function uploadAnnotated(
  className: string,
  image: File,
  annotation: File,
): Promise<{ filename: string; status: string }> {
  const form = new FormData()
  form.append("class_name", className)
  form.append("image", image)
  form.append("annotation", annotation)
  const res = await fetch(`${API_URL}/api/dataset/upload/annotated`, {
    method: "POST", body: form,
    signal: AbortSignal.timeout(30_000),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// ── Annotation endpoints ──────────────────────────────────────────

export async function getAnnotation(className: string, filename: string): Promise<string> {
  const res = await fetch(
    `${API_URL}/api/dataset/annotation/${encodeURIComponent(className)}/${encodeURIComponent(filename)}`,
    { signal: AbortSignal.timeout(10_000) },
  )
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.text()
}

export async function saveAnnotation(
  className: string,
  filename: string,
  polygons: Polygon[],
  imgW: number,
  imgH: number,
  sourceKey?: string,
): Promise<void> {
  const res = await fetch(
    `${API_URL}/api/dataset/annotation/${encodeURIComponent(className)}/${encodeURIComponent(filename)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygons, img_w: imgW, img_h: imgH, source_key: sourceKey }),
      signal: AbortSignal.timeout(15_000),
    },
  )
  if (!res.ok) throw new Error(await res.text())
}

// ── SAM endpoint ──────────────────────────────────────────────────

export async function samSegment(
  imageB64: string,
  points: { x: number; y: number; label: number }[],
): Promise<{ polygon: [number, number][]; mask_b64: string }> {
  const res = await fetch(`${API_URL}/api/sam/segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_b64: imageB64, points }),
    signal: AbortSignal.timeout(30_000),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// ── Training endpoints ────────────────────────────────────────────

export async function getTrainingStatus(): Promise<TrainingStatus> {
  const res = await fetch(`${API_URL}/api/training/status`, {
    signal: AbortSignal.timeout(5_000),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function triggerTraining(): Promise<void> {
  const res = await fetch(`${API_URL}/api/training/trigger`, {
    method: "POST",
    signal: AbortSignal.timeout(10_000),
  })
  if (!res.ok) throw new Error(await res.text())
}

export async function stopTraining(): Promise<void> {
  const res = await fetch(`${API_URL}/api/training/stop`, {
    method: "POST",
    signal: AbortSignal.timeout(10_000),
  })
  if (!res.ok) throw new Error(await res.text())
}

export async function getTrainingRuns(): Promise<TrainingRun[]> {
  const res = await fetch(`${API_URL}/api/training/runs`, {
    signal: AbortSignal.timeout(10_000),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return data.runs
}

export async function getRunMetricHistory(
  runId: string,
): Promise<{ run_id: string; history: Record<string, { step: number; value: number }[]> }> {
  const res = await fetch(`${API_URL}/api/training/runs/${runId}/metric-history`, {
    signal: AbortSignal.timeout(15_000),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function activateRun(runId: string): Promise<void> {
  const res = await fetch(`${API_URL}/api/training/runs/${runId}/activate`, {
    method: "POST",
    signal: AbortSignal.timeout(60_000),
  })
  if (!res.ok) throw new Error(await res.text())
}

// ── Feedback endpoints ────────────────────────────────────────────

export async function confirmFeedback(
  imageB64: string,
  filename: string,
  className: string,
  detections: Array<{ mask_b64?: string; polygon?: [number, number][]; label?: string }>,
): Promise<{ saved: string; feedback_id: string }> {
  const res = await fetch(`${API_URL}/api/feedback/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_b64: imageB64, filename, class_name: className, detections }),
    signal: AbortSignal.timeout(30_000),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function correctFeedback(
  imageB64: string,
  filename: string,
  className: string,
  polygons: Polygon[],
  imgW: number,
  imgH: number,
): Promise<{ saved: string; feedback_id: string }> {
  const res = await fetch(`${API_URL}/api/feedback/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_b64: imageB64,
      filename,
      class_name: className,
      polygons,
      img_w: imgW,
      img_h: imgH,
    }),
    signal: AbortSignal.timeout(30_000),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

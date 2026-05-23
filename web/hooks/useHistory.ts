import { useState, useCallback, useEffect } from "react"

export interface HistoryDetection {
  class: string
  class_id: number
  conf: number
}

export interface HistoryVideoStats {
  total_frames: number
  anomaly_frames: number
  anomaly_rate: number
  total_detections: number
  detections_by_class: Record<string, number>
}

export interface HistoryEntry {
  id: string
  timestamp: number
  type: "image" | "video" | "camera"
  filename: string
  is_anomaly: boolean
  detections_count: number
  defect_classes: string[]
  inference_ms?: number
  anomaly_rate?: number
  // Session-only — stripped before localStorage persist
  _imageData?: {
    original_b64: string
    annotated_b64: string
    detections: HistoryDetection[]
  }
  _videoStats?: HistoryVideoStats
}

const KEY = "dv_history_v1"
const MAX = 100

function stripSessionData(entries: HistoryEntry[]): HistoryEntry[] {
  return entries.map(({ _imageData: _i, _videoStats: _v, ...rest }) => rest)
}

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>([])

  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY)
      if (raw) setEntries(JSON.parse(raw))
    } catch {}
  }, [])

  const persist = (updated: HistoryEntry[]) => {
    try { localStorage.setItem(KEY, JSON.stringify(stripSessionData(updated))) } catch {}
  }

  const addEntry = useCallback((entry: Omit<HistoryEntry, "id" | "timestamp">) => {
    const item: HistoryEntry = {
      ...entry,
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      timestamp: Date.now(),
    }
    setEntries((prev) => {
      const updated = [item, ...prev].slice(0, MAX)
      persist(updated)
      return updated
    })
    return item.id
  }, [])

  const removeEntry = useCallback((id: string) => {
    setEntries((prev) => {
      const updated = prev.filter((e) => e.id !== id)
      persist(updated)
      return updated
    })
  }, [])

  const clearAll = useCallback(() => {
    setEntries([])
    try { localStorage.removeItem(KEY) } catch {}
  }, [])

  return { entries, addEntry, removeEntry, clearAll }
}

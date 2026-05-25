"use client"

import { useState } from "react"
import { Trash2, Plus, RefreshCw } from "lucide-react"
import { ClassInfo, getClasses, createClass, deleteClass } from "@/lib/dataset-api"

interface Props {
  classes: ClassInfo[]
  onRefresh: () => void
  loading: boolean
}

export default function ClassManager({ classes, onRefresh, loading }: Props) {
  const [newName, setNewName] = useState("")
  const [creating, setCreating] = useState(false)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleCreate() {
    const name = newName.trim()
    if (!name) return
    setCreating(true)
    setError(null)
    try {
      await createClass(name)
      setNewName("")
      onRefresh()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setCreating(false)
    }
  }

  async function handleDelete(name: string) {
    if (!confirm(`Delete class "${name}" and all its images?`)) return
    setDeleting(name)
    try {
      await deleteClass(name)
      onRefresh()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setDeleting(null)
    }
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="section-label">Classes</span>
        <button className="btn btn-secondary" style={{ padding: "6px 10px" }} onClick={onRefresh} disabled={loading}>
          <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {/* New class input */}
      <div className="flex gap-2">
        <input
          value={newName}
          onChange={e => setNewName(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleCreate()}
          placeholder="New class name…"
          style={{
            flex: 1, background: "var(--input-bg)", border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)", padding: "8px 10px", fontSize: 13,
            color: "var(--text-primary)", outline: "none", fontFamily: "inherit",
          }}
        />
        <button className="btn btn-primary" onClick={handleCreate} disabled={creating || !newName.trim()}>
          <Plus size={14} />
          {creating ? "Adding…" : "Add"}
        </button>
      </div>

      {error && (
        <p style={{ fontSize: 12, color: "var(--danger)" }}>{error}</p>
      )}

      {/* Class cards */}
      {classes.length === 0 && !loading && (
        <p style={{ fontSize: 13, color: "var(--text-muted)", textAlign: "center", padding: "24px 0" }}>
          No classes yet. Add one above.
        </p>
      )}

      <div className="grid grid-cols-2 gap-3" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))" }}>
        {classes.map(cls => (
          <div key={cls.name} className="panel" style={{ padding: "14px 16px" }}>
            <div className="flex items-start justify-between gap-2">
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", wordBreak: "break-all" }}>
                  {cls.name}
                </div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                  ID {cls.id}
                </div>
              </div>
              <button
                onClick={() => handleDelete(cls.name)}
                disabled={deleting === cls.name}
                style={{
                  background: "none", border: "none", cursor: "pointer", padding: 2,
                  color: "var(--text-muted)", flexShrink: 0,
                }}
              >
                <Trash2 size={14} />
              </button>
            </div>
            <div className="flex gap-3 mt-3">
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.7px", textTransform: "uppercase", color: "var(--text-muted)" }}>
                  Annotated
                </div>
                <div style={{ fontSize: 20, fontWeight: 700, color: "var(--success)", lineHeight: 1.2 }}>
                  {cls.annotated}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.7px", textTransform: "uppercase", color: "var(--text-muted)" }}>
                  Pending
                </div>
                <div style={{ fontSize: 20, fontWeight: 700, color: "var(--accent)", lineHeight: 1.2 }}>
                  {cls.pending}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

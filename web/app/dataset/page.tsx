"use client"

import { useState, useEffect, useCallback } from "react"
import Link from "next/link"
import { ArrowLeft, Database, Upload, Image, Activity } from "lucide-react"
import ClassManager from "@/components/dataset/ClassManager"
import ImageUploader from "@/components/dataset/ImageUploader"
import ImageGallery from "@/components/dataset/ImageGallery"
import TrainingPanel from "@/components/dataset/TrainingPanel"
import { ClassInfo, ImageEntry, getClasses, getImages } from "@/lib/dataset-api"

type Tab = "classes" | "upload" | "gallery" | "training"

export default function DatasetPage() {
  const [tab, setTab] = useState<Tab>("classes")
  const [classes, setClasses] = useState<ClassInfo[]>([])
  const [selectedClass, setSelectedClass] = useState<string>("")
  const [images, setImages] = useState<ImageEntry[]>([])
  const [classesLoading, setClassesLoading] = useState(true)
  const [imagesLoading, setImagesLoading] = useState(false)

  const loadClasses = useCallback(async () => {
    setClassesLoading(true)
    try {
      const cls = await getClasses()
      setClasses(cls)
      if (!selectedClass && cls.length > 0) setSelectedClass(cls[0].name)
    } catch {}
    finally { setClassesLoading(false) }
  }, [selectedClass])

  const loadImages = useCallback(async () => {
    if (!selectedClass) return
    setImagesLoading(true)
    try { setImages(await getImages(selectedClass)) } catch {}
    finally { setImagesLoading(false) }
  }, [selectedClass])

  useEffect(() => { loadClasses() }, [])
  useEffect(() => { if (selectedClass) loadImages() }, [selectedClass])

  const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: "classes", label: "Classes", icon: <Database size={14} /> },
    { key: "upload", label: "Upload", icon: <Upload size={14} /> },
    { key: "gallery", label: "Gallery", icon: <Image size={14} /> },
    { key: "training", label: "Training", icon: <Activity size={14} /> },
  ]

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)" }}>
      {/* Header */}
      <header style={{
        background: "var(--header-bg)", borderBottom: "1px solid var(--border)",
        padding: "0 24px", height: 56, display: "flex", alignItems: "center", gap: 16,
        position: "sticky", top: 0, zIndex: 40,
      }}>
        <Link href="/" style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--text-muted)", textDecoration: "none", fontSize: 13 }}>
          <ArrowLeft size={15} />
          Back
        </Link>
        <div style={{ width: 1, height: 20, background: "var(--border)" }} />
        <span style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)" }}>
          Dataset Manager
        </span>
      </header>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 24px" }}>
        {/* Tab bar */}
        <div className="flex gap-1" style={{
          background: "var(--surface-alt)", padding: 4, borderRadius: "var(--radius-md)",
          display: "inline-flex", marginBottom: 28,
        }}>
          {tabs.map(t => (
            <button
              key={t.key}
              className={`seg-btn ${tab === t.key ? "active" : ""}`}
              style={{ padding: "7px 14px", display: "flex", alignItems: "center", gap: 6 }}
              onClick={() => setTab(t.key)}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="panel" style={{ padding: "24px 28px" }}>
          {tab === "classes" && (
            <ClassManager
              classes={classes}
              onRefresh={loadClasses}
              loading={classesLoading}
            />
          )}

          {tab === "upload" && (
            classes.length === 0
              ? <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Create at least one class before uploading.</p>
              : <ImageUploader classes={classes} onUploaded={() => { loadClasses(); loadImages() }} />
          )}

          {tab === "gallery" && (
            <ImageGallery
              classes={classes}
              selectedClass={selectedClass}
              onSelectClass={name => { setSelectedClass(name); setImages([]) }}
              images={images}
              onRefresh={loadImages}
            />
          )}

          {tab === "training" && <TrainingPanel />}
        </div>
      </div>
    </div>
  )
}

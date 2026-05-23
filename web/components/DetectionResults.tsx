"use client"

import { Detection } from "@/lib/api"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts"

interface Props {
  detections: Detection[]
  inference_ms: number
  is_anomaly: boolean
}

export default function DetectionResults({ detections, inference_ms, is_anomaly }: Props) {
  /* Group by class */
  const grouped: Record<string, { count: number; maxConf: number }> = {}
  detections.forEach((d) => {
    if (!grouped[d.class]) grouped[d.class] = { count: 0, maxConf: 0 }
    grouped[d.class].count++
    grouped[d.class].maxConf = Math.max(grouped[d.class].maxConf, d.conf)
  })
  const entries = Object.entries(grouped)
  const chartData = entries.map(([name, g]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    value: Math.round(g.maxConf * 100),
  }))

  return (
    <div className="space-y-3">

      {/* Status row */}
      <div
        className="flex items-center justify-between px-4 py-3 rounded"
        style={{
          background: is_anomaly ? "var(--danger-surface)" : "var(--success-surface)",
          border: `1px solid ${is_anomaly ? "var(--danger-border)" : "var(--success-border)"}`,
          borderRadius: "var(--radius-sm)",
        }}
      >
        <div className="flex items-center gap-2.5">
          <div className="w-2 h-2 rounded-full flex-shrink-0"
            style={{ background: is_anomaly ? "var(--danger)" : "var(--success)" }} />
          <span style={{ fontSize: 13, fontWeight: 600, color: is_anomaly ? "var(--danger)" : "var(--success)" }}>
            {is_anomaly
              ? `Defect detected — ${entries.map(([c, g]) => `${c} ×${g.count}`).join(", ")}`
              : "No defects detected"}
          </span>
        </div>
        <span style={{ fontSize: 11, fontFamily: "monospace", color: "var(--text-muted)" }}>
          {inference_ms} ms
        </span>
      </div>

      {/* Metrics + chart */}
      {entries.length > 0 && (
        <div className="panel" style={{ padding: "16px 18px" }}>

          {/* Metric row */}
          <div className="flex gap-6 pb-4 mb-4" style={{ borderBottom: "1px solid var(--border)" }}>
            {entries.map(([cls, g]) => (
              <div key={cls}>
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 4 }}>
                  {cls}
                </div>
                <div className="flex items-end gap-2">
                  <span style={{ fontSize: 26, fontWeight: 700, lineHeight: 1, color: "var(--text-primary)", fontVariantNumeric: "tabular-nums" }}>
                    {g.count}
                  </span>
                  <span style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 2 }}>
                    instance{g.count > 1 ? "s" : ""}
                  </span>
                </div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                  Max confidence: <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>
                    {Math.round(g.maxConf * 100)}%
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Bar chart */}
          <div>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 10 }}>
              Confidence by class
            </div>
            <ResponsiveContainer width="100%" height={110}>
              <BarChart data={chartData} barSize={28} margin={{ top: 4, right: 4, left: -10, bottom: 0 }}>
                <XAxis dataKey="name"
                  tick={{ fill: "var(--text-muted)", fontSize: 11, fontFamily: "DM Sans, sans-serif" }}
                  axisLine={{ stroke: "var(--border)" }} tickLine={false}
                />
                <YAxis domain={[0, 100]} unit="%"
                  tick={{ fill: "var(--text-muted)", fontSize: 10, fontFamily: "DM Sans, sans-serif" }}
                  axisLine={false} tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface)", border: "1px solid var(--border)",
                    borderRadius: "var(--radius-sm)", fontSize: 12, color: "var(--text-primary)",
                    boxShadow: "var(--shadow-md)", fontFamily: "DM Sans, sans-serif",
                  }}
                  formatter={(v: number) => [`${v}%`, "Confidence"]}
                  cursor={{ fill: "var(--surface-alt)" }}
                />
                <Bar dataKey="value" radius={[2, 2, 0, 0]}>
                  {chartData.map((_, i) => (
                    <Cell key={i} fill="var(--danger)" opacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}

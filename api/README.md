---
title: DefectVision API
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# DefectVision API

FastAPI inference backend for DefectVision Pro.

**Endpoints:**
- `GET /api/health` — health check + device info
- `GET /api/models` — list available weight files
- `POST /api/predict/image` — run YOLO on an uploaded image
- `POST /api/predict/video` — run YOLO on an uploaded video

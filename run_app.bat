@echo off
cd /d "%~dp0"
echo Starting DefectVision Pro (Docker)...
echo.
echo Services:
echo   Frontend  →  http://localhost:3000
echo   API       →  http://localhost:8000
echo   MinIO UI  →  http://localhost:9001  (user: minioadmin / minioadmin)
echo   MLflow UI →  http://localhost:5000
echo.
docker compose up --build

# Deployment Guide

## Architecture
- Frontend (Next.js): Vercel
- Backend (FastAPI + YOLO): Render (Docker)

## 1) Deploy Backend to Render
1. Push this project to a Git repository.
2. In Render, create a new **Blueprint** deployment and select the repo.
3. Render will use `render.yaml` and build `backend/Dockerfile`.
4. Set backend environment variables:
   - `CORS_ORIGINS=https://<your-vercel-domain>`
   - `YOLO_DEVICE=cpu`
5. Deploy. After success, copy backend URL, for example:
   - `https://thermal-app-gps-backend.onrender.com`

### Important
- Current DJI SDK binary in this repo is Windows-only (`dji_irp.exe`).
- On Render (Linux), absolute DJI temperature mode via `dji_irp.exe` is unavailable.
- The API still works and falls back to available thermal modes.

## 2) Deploy Frontend to Vercel
1. Import `frontend` project into Vercel.
2. Set environment variable in Vercel Project Settings:
   - `NEXT_PUBLIC_BACKEND_URL=https://<your-render-backend-url>`
3. Deploy.

## 3) Verify
1. Open frontend URL.
2. Upload an image.
3. Confirm API request is sent to Render backend and returns detections.

## Local Run (unchanged)
- Backend: `uvicorn main:app --reload --host 127.0.0.1 --port 8000`
- Frontend: `npm run dev`

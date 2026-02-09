# Deployment Guide (Render Only)

## Architecture
- Backend: Render Web Service (Docker) from `backend/`
- Frontend: Render Web Service (Node) from `frontend/`
- Both are created by `render.yaml` (Blueprint)

## 1) Deploy with Blueprint
1. Push code to GitHub.
2. In Render: `New` -> `Blueprint`.
3. Select repo `SadBoySoyBad/thermal-app-test`.
4. Render reads `render.yaml` and creates 2 services:
   - `thermal-app-gps-backend`
   - `thermal-app-gps-frontend`
5. Click `Apply` / `Deploy`.

## 2) Wait Until Both Services Are Live
1. Backend should expose URL like:
   - `https://thermal-app-gps-backend.onrender.com`
2. Frontend should expose URL like:
   - `https://thermal-app-gps-frontend.onrender.com`

## 3) Verify
1. Open frontend URL.
2. Upload an image.
3. Confirm response is returned and result is shown.

## 4) Security Tightening (After First Successful Test)
Current blueprint starts backend with `CORS_ALLOW_ALL=true` for easier first deploy.
After frontend is live:
1. Open backend service -> `Environment`.
2. Set:
   - `CORS_ALLOW_ALL=false`
   - `CORS_ORIGINS=https://<your-frontend-render-url>`
3. Save and redeploy backend.

## Important Notes
- This repo now includes DJI IRP binaries for both Windows and Linux.
- Development on Windows uses `.exe` binaries automatically.
- Production on Render (Linux) uses `/app/tools/dji-tsdk/utility/bin/linux/release_x64/dji_irp`.
- If you override `DJI_IRP_PATH`, make sure it matches the OS.

## Local Run
- Backend: `uvicorn main:app --reload --host 127.0.0.1 --port 8000`
- Frontend: `npm run dev`

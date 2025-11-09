# BetterGrade OpenCV OMR Backend (FastAPI)

Production-ready microservice that scans A4 bubble sheets (20/50/100 items) using OpenCV.

## Quick start (local)
```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8080
```
Health: http://localhost:8080/health

## Docker
```bash
docker build -t bettergrade-omr .
docker run -p 8080:8080 bettergrade-omr
```

## API
POST /scan
```
{
  "image_b64": "<base64>",
  "sheet_version": "50",
  "threshold": 0.4,
  "return_debug": true
}
```
Returns `answers`, `confidence`, and optional debug images (base64).

## Deploy (Render Docker)
1) Push this folder to a GitHub repo.
2) Render → New → Web Service → pick the repo.
3) Exposes port 8080; deploy. Copy the public URL.

## Connect from Expo/Rork
```ts
import * as FileSystem from 'expo-file-system';

export async function scanWithOpenCV(uri: string, version: '20'|'50'|'100') {
  const b64 = await FileSystem.readAsStringAsync(uri, { encoding: (FileSystem as any).EncodingType.Base64 });
  const url = (process.env.EXPO_PUBLIC_OMR_URL || '').replace(/\/$/, '');
  const r = await fetch(url + '/scan', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ image_b64: b64, sheet_version: version, threshold: 0.40 })
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
```

## Tuning
- If marks are missed → lower threshold to 0.35.
- If double-marked → adjust multi-mark delta (0.05) in `server.py`.
- Ensure printed corner squares are large/dark; keep margins ~12–15 mm.

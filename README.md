# Beginner OMR Backend (Flask + OpenCV)
- GET /health -> {"ok": true}
- POST /api/scan with JSON:
  {
    "image": "<base64>",
    "testId": "20x5",
    "answerKey": ["A","B","C","D", ...],
    "threshold": 0.45,
    "returnOverlay": true
  }
Returns answers, optional score, per-question confidence, and overlay image (base64 JPG).

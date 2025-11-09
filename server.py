from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple
import base64, json, os
import numpy as np
import cv2

app = FastAPI(title="BetterGrade OpenCV OMR", version="0.1.0")

class ScanRequest(BaseModel):
    image_b64: str
    sheet_version: str
    threshold: float = 0.40
    return_debug: bool = False

class ScanResponse(BaseModel):
    sheet_version: str
    answers: Dict[int, Optional[str]]
    confidence: Dict[int, float]
    debug: Optional[Dict] = None

def load_template(ver: str) -> dict:
    here = os.path.dirname(__file__)
    path = os.path.join(here, "templates", f"{ver}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def decode_image(image_b64: str):
    try:
        raw = base64.b64decode(image_b64)
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {e}")

def to_b64_img(img):
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    return base64.b64encode(enc.tobytes()).decode("utf-8")

def order_corners(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4,2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # TL
    ordered[2] = pts[np.argmax(s)]  # BR
    ordered[1] = pts[np.argmin(diff)] # TR
    ordered[3] = pts[np.argmax(diff)] # BL
    return ordered

def find_marker_squares(binary: np.ndarray) -> List[np.ndarray]:
    contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    H, W = binary.shape[:2]
    area_min = 0.002 * (H*W)
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            candidates.append(approx.reshape(4,2))
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda pts: -cv2.contourArea(pts.reshape(-1,1,2)))
    pts = np.array([np.mean(q, axis=0) for q in candidates[:10]], dtype=np.float32)
    tl = pts[np.argmin(pts.sum(axis=1))]
    br = pts[np.argmax(pts.sum(axis=1))]
    tr = pts[np.argmin(np.diff(pts, axis=1))]
    bl = pts[np.argmax(np.diff(pts, axis=1))]
    return [tl, tr, br, bl]

def warp_to_canvas(img, corners, canvas_size):
    W, H = canvas_size
    dst = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    src = np.float32(corners)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H))
    return warped, M

def sample_fill_ratio(bin_img, cx, cy, r):
    H, W = bin_img.shape[:2]
    mask = np.zeros((H,W), dtype=np.uint8)
    cv2.circle(mask, (int(cx),int(cy)), int(r), 255, -1)
    filled = cv2.mean(bin_img, mask=mask)[0] / 255.0
    return float(filled)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    tpl = load_template(req.sheet_version)
    img = decode_image(req.image_b64)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    bin_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 35, 10)

    corners = find_marker_squares(bin_inv)
    if len(corners) != 4:
        edges = cv2.Canny(blur, 60, 160)
        kernel = np.ones((3,3), np.uint8)
        dil = cv2.dilate(edges, kernel, iterations=1)
        corners = find_marker_squares(dil)
        if len(corners) != 4:
            raise HTTPException(status_code=422, detail="Corner markers not found.")

    ordered = order_corners(np.array(corners, dtype=np.float32))

    W, H = tpl["sheetSize"]
    warped, M = warp_to_canvas(gray, ordered, (W, H))
    warped_blur = cv2.GaussianBlur(warped, (3,3), 0)
    warped_bin = cv2.adaptiveThreshold(warped_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 35, 10)
    kernel = np.ones((3,3), np.uint8)
    warped_bin = cv2.morphologyEx(warped_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

    answers = {}
    confidence = {}
    for q in tpl["questions"]:
        qnum = int(q["q"])
        best_letter, best_score = None, -1.0
        scores = {}
        for letter, (cx, cy, r) in q["choices"].items():
            score = sample_fill_ratio(warped_bin, cx, cy, r)
            scores[letter] = score
            if score > best_score:
                best_score, best_letter = score, letter
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ambiguous = len(sorted_scores)>1 and (sorted_scores[0][1]-sorted_scores[1][1])<=0.05
        answers[qnum] = best_letter if (best_score>=req.threshold and not ambiguous) else None
        confidence[qnum] = float(best_score)

    debug = None
    if req.return_debug:
        debug = {
            "corners": np.array(ordered).tolist(),
            "warped_bin_jpg_b64": to_b64_img(warped_bin),
            "warped_gray_jpg_b64": to_b64_img(warped),
        }

    return ScanResponse(sheet_version=req.sheet_version, answers=answers, confidence=confidence, debug=debug)

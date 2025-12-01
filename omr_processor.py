import cv2
import numpy as np
import json, os, base64

def load_template(ver: str) -> dict:
    here = os.path.dirname(__file__)
    path = os.path.join(here, "templates", f"{ver}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def decode_image_b64(image_b64: str):
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None")
    return img

def order_corners(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4,2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # TL
    ordered[2] = pts[np.argmax(s)]      # BR
    ordered[1] = pts[np.argmin(diff)]   # TR
    ordered[3] = pts[np.argmax(diff)]   # BL
    return ordered

def find_markers(binary: np.ndarray):
    contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = binary.shape[:2]
    area_min = 0.002 * (H*W)
    quads = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < area_min:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04*peri, True)
        if len(approx) == 4:
            quads.append(approx.reshape(4,2))
    if not quads:
        return []
    quads = sorted(quads, key=lambda q: -cv2.contourArea(q.reshape(-1,1,2)))
    centers = np.array([np.mean(q, axis=0) for q in quads[:10]], dtype=np.float32)
    tl = centers[np.argmin(centers.sum(axis=1))]
    br = centers[np.argmax(centers.sum(axis=1))]
    tr = centers[np.argmin(np.diff(centers, axis=1))]
    bl = centers[np.argmax(np.diff(centers, axis=1))]
    return [tl, tr, br, bl]

def warp_canvas(img, corners, canvas_size):
    W, H = canvas_size
    dst = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    src = np.float32(corners)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H))
    return warped

def sample_fill_ratio(bin_img, cx, cy, r):
    H, W = bin_img.shape[:2]
    mask = np.zeros((H,W), dtype=np.uint8)
    cv2.circle(mask, (int(cx),int(cy)), int(r), 255, -1)
    return cv2.mean(bin_img, mask=mask)[0] / 255.0

def to_b64_jpg(img):
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok: return None
    return base64.b64encode(enc.tobytes()).decode("utf-8")

def draw_overlay(warped_gray, tpl, answers: dict, answer_key):
    overlay = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
    GREY=(180,180,180); GREEN=(0,200,0); RED=(0,0,220); YELL=(0,220,220)
    for q in tpl["questions"]:
        for (_, (cx,cy,r)) in q["choices"].items():
            cv2.circle(overlay, (int(cx),int(cy)), int(r), GREY, 1)
    key_map = {}
    if isinstance(answer_key, list):
        key_map = {i+1: ch for i,ch in enumerate(answer_key)}
    for q in tpl["questions"]:
        qn = int(q["q"])
        det = answers.get(qn)
        corr = key_map.get(qn) if key_map else None
        if det is None:
            if corr and corr in q["choices"]:
                cx,cy,r = q["choices"][corr]
                cv2.circle(overlay, (int(cx),int(cy)), int(r)+3, YELL, 2)
            continue
        if det in q["choices"]:
            dx,dy,dr = q["choices"][det]
            ok = (corr is not None and det == corr)
            cv2.circle(overlay, (int(dx),int(dy)), int(dr)+4, GREEN if ok else RED, 3)
            if (not ok) and corr and corr in q["choices"]:
                cx,cy,r = q["choices"][corr]
                cv2.circle(overlay, (int(cx),int(cy)), int(r)+2, GREEN, 2)
    return to_b64_jpg(overlay)

def process(image_b64: str, tpl_version: str, threshold: float=0.45):
    tpl = load_template(tpl_version)
    img = decode_image_b64(image_b64)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    bin_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 35, 10)

    corners = find_markers(bin_inv)
    if len(corners) != 4:
        edges = cv2.Canny(blur, 60, 160)
        kernel = np.ones((3,3), np.uint8)
        dil = cv2.dilate(edges, kernel, iterations=1)
        corners = find_markers(dil)
        if len(corners) != 4:
            raise ValueError("Corner markers not found.")

    import numpy as np
    ordered = order_corners(np.array(corners, dtype=np.float32))
    W,H = tpl["sheetSize"]
    warped = warp_canvas(gray, ordered, (W,H))
    w_blur = cv2.GaussianBlur(warped, (3,3), 0)
    w_bin  = cv2.adaptiveThreshold(w_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 10)
    kernel = np.ones((3,3), np.uint8)
    w_bin  = cv2.morphologyEx(w_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

    answers = {}
    confidence = {}
    for q in tpl["questions"]:
        qn = int(q["q"])
        best = None; best_score = -1.0
        scores = {}
        for ch,(cx,cy,r) in q["choices"].items():
            s = float(sample_fill_ratio(w_bin, cx, cy, r))
            scores[ch] = s
            if s > best_score:
                best_score, best = s, ch
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ambiguous = len(top)>1 and (top[0][1]-top[1][1]) <= 0.05
        answers[qn] = best if (best_score >= threshold and not ambiguous) else None
        confidence[qn] = float(best_score)

    return answers, confidence, warped

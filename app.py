from flask import Flask, request, jsonify
from omr_processor import process, load_template, draw_overlay

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/api/scan")
def scan():
    data = request.get_json(force=True, silent=True) or {}
    image_b64 = data.get("image")
    test_id = data.get("testId", "20x5")
    answer_key = data.get("answerKey")
    threshold = float(data.get("threshold", 0.45))
    return_overlay = bool(data.get("returnOverlay", True))

    if not image_b64:
        return jsonify({"error": "Missing 'image' (base64) in body"}), 400

    try:
        answers, confidence, warped = process(image_b64, test_id, threshold)
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    score = None
    if isinstance(answer_key, list):
        score = sum(1 for i,ch in enumerate(answer_key, start=1) if answers.get(i) == ch)

    overlay_b64 = None
    if return_overlay:
        tpl = load_template(test_id)
        overlay_b64 = draw_overlay(warped, tpl, answers, answer_key)

    ordered = [answers.get(i) for i in range(1, len(answers)+1)]
    return jsonify({
        "testId": test_id,
        "answers": ordered,
        "score": score,
        "confidence": confidence,
        "overlay_jpg_b64": overlay_b64
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

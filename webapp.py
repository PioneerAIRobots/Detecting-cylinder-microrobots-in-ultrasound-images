"""
USMicroMagSet — YOLOv5 Inference Backend
Flask server that loads a YOLOv5 model and serves predictions.

Usage:
    1. Install deps:  pip install flask flask-cors torch torchvision pillow
    2. Put your trained weights at:  weights/cylinder_best.pt
       (or change MODEL_PATH below)
    3. Run: python app.py
    4. Open index.html in your browser (or serve it from Flask)
"""

import os
import io
import base64
import time
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# Source - https://stackoverflow.com/a/64200241
# Posted by KumaTea, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-23, License - CC BY-SA 4.0

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# ─────────────────────────────────────────
#  CONFIG — edit these paths as needed
# ─────────────────────────────────────────
MODEL_PATH  = "weights/cylinder_best.pt"   # Path to your trained .pt weights
YOLOV5_REPO = "ultralytics/yolov5"         # torch.hub source (uses cached if offline)
CONF_THRESH = 0.25                          # Confidence threshold
IOU_THRESH  = 0.45                          # NMS IoU threshold
IMG_SIZE    = 640                           # Inference image size
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["cylinder"]                  # Single class — adjust if multi-class

# Accent colour for bounding boxes
BOX_COLOR   = (0, 255, 157)   # --accent2 green

app = Flask(__name__, static_folder=".")
CORS(app)  # Allow requests from the HTML frontend

# ─────────────────────────────────────────
#  MODEL LOAD  (once at startup)
# ─────────────────────────────────────────
print(f"[USMicroMag] Loading model from {MODEL_PATH} on {DEVICE} …")
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            # Load custom weights from local file via torch.hub
            model = torch.hub.load(
                YOLOV5_REPO, "custom",
                path=MODEL_PATH,
                source="github",
                force_reload=False,
                trust_repo=True
            )
        else:
            # Fallback: load pretrained YOLOv5s for demo (no custom weights found)
            print(f"[WARN] {MODEL_PATH} not found — loading demo yolov5s weights.")
            model = torch.hub.load(YOLOV5_REPO, "yolov5s", trust_repo=True)

        model.conf = CONF_THRESH
        model.iou  = IOU_THRESH
        model.to(DEVICE)
        model.eval()
        print(f"[USMicroMag] Model loaded ✓  Device: {DEVICE}")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        model = None

load_model()

# ─────────────────────────────────────────
#  HELPER — draw boxes on PIL image
# ─────────────────────────────────────────
def draw_predictions(img: Image.Image, detections) -> Image.Image:
    """Draw YOLOv5 bounding boxes onto a PIL image."""
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Try to load a small font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for *xyxy, conf, cls_id in detections:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cls_name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else f"cls{int(cls_id)}"
        label = f"{cls_name}  {conf:.2f}"

        # Box + thick corners
        lw = max(2, int((W + H) / 500))
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=lw)

        # Corner brackets
        cs = 14
        for px, py, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            draw.line([(px, py),(px+sx*cs, py)], fill=BOX_COLOR, width=lw+1)
            draw.line([(px, py),(px, py+sy*cs)], fill=BOX_COLOR, width=lw+1)

        # Label background
        tw = len(label) * 8 + 8
        th = 18
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=BOX_COLOR)
        draw.text((x1 + 4, y1 - th + 2), label, fill=(0, 0, 0), font=font)

    return img


# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend HTML."""
    return send_from_directory(".", "usmicromagset.html")


@app.route("/status")
def status():
    """Health check — lets the frontend know if the model is ready."""
    return jsonify({
        "model_loaded": model is not None,
        "device": DEVICE,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "conf_thresh": CONF_THRESH,
        "iou_thresh": IOU_THRESH,
        "classes": CLASS_NAMES
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart/form-data POST with field 'image'.
    Returns JSON with detections and base64-encoded annotated image.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image field in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = img.size

        t0 = time.time()

        # Run inference
        results = model(img, size=IMG_SIZE)
        inference_ms = round((time.time() - t0) * 1000, 1)

        # Parse detections — results.xyxy[0] shape: (N, 6) [x1,y1,x2,y2,conf,cls]
        dets = results.xyxy[0].cpu().numpy()

        detection_list = []
        for *xyxy, conf, cls_id in dets:
            x1, y1, x2, y2 = [round(float(v), 1) for v in xyxy]
            detection_list.append({
                "class_id":   int(cls_id),
                "class_name": CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else f"cls{int(cls_id)}",
                "confidence": round(float(conf), 4),
                "bbox":       [x1, y1, x2, y2],
                "bbox_norm":  [round(x1/orig_w,4), round(y1/orig_h,4),
                               round(x2/orig_w,4), round(y2/orig_h,4)]
            })

        # Draw boxes on image
        annotated = draw_predictions(img.copy(), dets)

        # Encode to base64
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=92)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Also encode original for comparison
        buf2 = io.BytesIO()
        img.save(buf2, format="JPEG", quality=90)
        orig_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

        return jsonify({
            "success":       True,
            "inference_ms":  inference_ms,
            "device":        DEVICE,
            "image_size":    [orig_w, orig_h],
            "num_detections": len(detection_list),
            "detections":    detection_list,
            "annotated_image": f"data:image/jpeg;base64,{img_b64}",
            "original_image":  f"data:image/jpeg;base64,{orig_b64}"
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("─" * 55)
    print(" USMicroMagSet Backend  —  http://localhost:5000")
    print("─" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
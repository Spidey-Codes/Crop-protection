cat > /mnt/user-data/outputs/main.py << 'EOF'
"""
Smart Crop Predator Detection — Railway Cloud Server
=====================================================
ESP32-CAM POSTs JPEG frames to /detect endpoint.
Server runs YOLOv8 → pushes to Firebase → DevKit triggers deterrents.

Railway Variables needed:
  FIREBASE_CREDENTIALS  — paste your firebase_cred.json as single line JSON
  FIREBASE_URL          — your Firebase database URL

Google Sheets logging is handled by a separate Railway project.
"""

import os
import json
import tempfile
import time
import numpy as np
import cv2
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db as firebase_db

app = FastAPI()

# ── Config ────────────────────────────────────────────────────
MODEL_PATH       = "best.pt"
CONFIDENCE       = 0.55
COOLDOWN_SECONDS = 20
CLASS_NAMES      = ["Human", "Locust", "Monkey", "Wildboar"]

FIREBASE_URL = os.environ.get(
    "FIREBASE_URL",
    "https://crop-protection-6f8ca-default-rtdb.asia-southeast1.firebasedatabase.app/"
)

# ── State ─────────────────────────────────────────────────────
model               = None
firebase_ok         = False
last_detection_time = 0
total_detections    = 0
server_start_time   = datetime.now()


# ── Get Firebase credentials ──────────────────────────────────
def get_cred_path():
    cred_json_str = os.environ.get("FIREBASE_CREDENTIALS")

    if cred_json_str:
        try:
            cred_dict = json.loads(cred_json_str)
            tmp = tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            )
            json.dump(cred_dict, tmp)
            tmp.close()
            print("Credentials: loaded from FIREBASE_CREDENTIALS env variable")
            return tmp.name
        except Exception as e:
            print(f"Credentials: failed to parse env variable — {e}")
            return None

    elif os.path.exists("firebase_cred.json"):
        print("Credentials: loaded from firebase_cred.json file")
        return "firebase_cred.json"

    else:
        print("Credentials: NOT FOUND — set FIREBASE_CREDENTIALS in Railway Variables")
        return None


# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global model, firebase_ok

    print("=" * 50)
    print("  Crop Predator Detection — Railway Server")
    print("=" * 50)

    # Load YOLO model
    try:
        model = YOLO(MODEL_PATH)
        model.fuse()
        print(f"Model: loaded {MODEL_PATH}")
        print(f"Classes: {model.names}")
    except Exception as e:
        print(f"Model ERROR: {e}")

    # Get credentials
    cred_path = get_cred_path()
    if not cred_path:
        print("WARNING: No credentials — Firebase disabled")
        print("Server will still run inference but won't log to Firebase")
        return

    # Firebase init
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
        firebase_ok = True
        print(f"Firebase: connected OK")

        # Log server online status to Firebase
        firebase_db.reference("/crop_predator/server").set({
            "status":     "online",
            "started_at": datetime.now().isoformat(),
            "model":      MODEL_PATH,
            "classes":    CLASS_NAMES,
            "url":        "https://crop-protection-api-production.up.railway.app"
        })
    except Exception as e:
        print(f"Firebase: error — {e}")
    print("\nServer ready — waiting for frames from ESP32-CAM\n")


# ── Health check ──────────────────────────────────────────────
@app.get("/")
def health():
    uptime = str(datetime.now() - server_start_time).split(".")[0]
    return {
        "status":           "online",
        "model":            MODEL_PATH,
        "classes":          CLASS_NAMES,
        "firebase":         firebase_ok,
        "sheets":           "handled by separate Railway project",
        "confidence":       CONFIDENCE,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "total_detections": total_detections,
        "uptime":           uptime,
        "firebase_url":     FIREBASE_URL
    }


# ── Detection endpoint ────────────────────────────────────────
@app.post("/detect")
async def detect(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    global last_detection_time, total_detections

    if model is None:
        return JSONResponse(
            {"error": "model not loaded — check Railway logs"},
            status_code=503
        )

    # Decode uploaded JPEG
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    # Run YOLOv8 inference
    results    = model(frame, verbose=False, conf=CONFIDENCE)
    best_label = None
    best_conf  = 0.0

    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        conf  = float(box.conf[0])
        if conf > best_conf:
            best_label = label
            best_conf  = conf

    now           = time.time()
    cooldown_left = max(0, COOLDOWN_SECONDS - (now - last_detection_time))

    if best_label and cooldown_left == 0:
        last_detection_time  = now
        total_detections    += 1

        print(f"\n{'='*40}")
        print(f"DETECTED: {best_label} ({best_conf*100:.1f}%)")
        print(f"Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total:    {total_detections}")
        print(f"{'='*40}")

        # Push to Firebase in background
        background_tasks.add_task(push_to_firebase, best_label, bes
        return JSONResponse({
            "detected":         True,
            "class":            best_label,
            "confidence":       round(best_conf, 3),
            "timestamp":        datetime.now().isoformat(),
            "total_detections": total_detections,
            "message":          "Firebase updated — DevKit will trigger deterrent"
        })

    return JSONResponse({
        "detected":     False,
        "class":        best_label,
        "confidence":   round(best_conf, 3) if best_conf else 0,
        "cooldown_left": round(cooldown_left, 1),
        "timestamp":    datetime.now().isoformat()
    })


# ── Firebase push ─────────────────────────────────────────────
def push_to_firebase(label, confidence):
    if not firebase_ok:
        return
    try:
        # /latest — DevKit polls this every 2 seconds
        firebase_db.reference("/crop_predator/latest").set({
            "class":      label,
            "confidence": round(float(confidence), 3),
            "timestamp":  datetime.now().isoformat(),
            "unix":       int(time.time()),
            "action":     "TRIGGER"
        })
        # /events — permanent log
        firebase_db.reference("/crop_predator/events").push({
            "class":      label,
            "confidence": round(float(confidence), 3),
            "timestamp":  datetime.now().isoformat(),
            "unix":       int(time.time()),
        })
        print(f"Firebase: pushed {label}")
    except Exception as e:
        print(f"Firebase error: {e}")

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

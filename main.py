"""
Smart Crop Predator Detection — Railway Cloud Server (Final)
=============================================================
Pre-filled with your Railway URL and Firebase project.
Deploy this to Railway — it runs YOLOv8 24/7.

ESP32-CAM POSTs JPEG frames to /detect endpoint.
Server runs YOLOv8 → pushes to Firebase → DevKit triggers deterrents.
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db as firebase_db
import gspread
from google.oauth2.service_account import Credentials as GCredentials

app = FastAPI()

# ── Config ────────────────────────────────────────────────────
MODEL_PATH       = "best.pt"
CONFIDENCE       = 0.6
COOLDOWN_SECONDS = 30
FIREBASE_CRED    = "firebase_cred.json"
SHEETS_NAME      = "Crop Predator Log"

# Pre-filled Firebase URL
firebase_json = json.loads(os.environ["FIREBASE_CREDENTIALS"])

cred = credentials.Certificate(firebase_json)

firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": DATABASE_URL
    }
)
# Your model's class names exactly as trained
CLASS_NAMES = ["Human", "Locust", "Monkey", "Wildboar"]

# ── State ─────────────────────────────────────────────────────
model               = None
firebase_ok         = False
sheets_ok           = False
sheets_client       = None
last_detection_time = 0
total_detections    = 0
server_start_time   = datetime.now()


# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global model, firebase_ok, sheets_ok, sheets_client

    print("="*50)
    print("  Crop Predator Detection — Railway Server")
    print("="*50)

    # Load YOLO
    try:
        model = YOLO(MODEL_PATH)
        model.fuse()   # reduces memory usage on Railway free tier
        print(f"Model: loaded {MODEL_PATH}")
        print(f"Classes: {model.names}")
    except Exception as e:
        print(f"Model load ERROR: {e}")

    # Firebase
    try:
        cred = credentials.Certificate(FIREBASE_CRED)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
        firebase_ok = True
        print("Firebase: connected OK")
        # Log server start
        firebase_db.reference("/crop_predator/server").set({
            "status":     "online",
            "started_at": datetime.now().isoformat(),
            "model":      MODEL_PATH,
            "classes":    CLASS_NAMES,
            "url":        "https://crop-protection-api-production.up.railway.app"
        })
    except Exception as e:
        print(f"Firebase: disabled ({e})")

    # Google Sheets
    try:
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = GCredentials.from_service_account_file(FIREBASE_CRED, scopes=scopes)
        sheets_client = gspread.authorize(creds)
        sheet = sheets_client.open(SHEETS_NAME).sheet1
        # Check headers
        if not sheet.get_all_values():
            sheet.insert_row(
                ["Timestamp", "Class", "Confidence", "Temperature", "Humidity", "Firebase Key"],
                index=1
            )
        sheets_ok = True
        print("Google Sheets: connected OK")
    except Exception as e:
        print(f"Google Sheets: disabled ({e})")

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
        "sheets":           sheets_ok,
        "total_detections": total_detections,
        "uptime":           uptime,
        "url":              "https://crop-protection-api-production.up.railway.app"
    }


# ── Detection endpoint ────────────────────────────────────────
@app.post("/detect")
async def detect(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    global last_detection_time, total_detections

    if model is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)

    # Decode image
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

    # Only trigger if detection found and cooldown elapsed
    if best_label and cooldown_left == 0:
        last_detection_time  = now
        total_detections    += 1

        print(f"\n{'='*40}")
        print(f"DETECTED: {best_label} ({best_conf*100:.1f}%)")
        print(f"time":datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")")
        print(f"Total detections: {total_detections}")
        print(f"{'='*40}")

        # Push to Firebase + Sheets in background (non-blocking)
        background_tasks.add_task(push_to_firebase, best_label, best_conf)
        background_tasks.add_task(push_to_sheets,   best_label, best_conf)

        return JSONResponse({
            "detected":         True,
            "class":            best_label,
            "confidence":       round(best_conf, 3),
            "timestamp":        datetime.now().isoformat(),
            "total_detections": total_detections,
            "message":          f"Firebase updated → DevKit will trigger deterrent"
        })

    # No detection or in cooldown
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
        # /latest — DevKit polls this
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


# ── Sheets push ───────────────────────────────────────────────
def push_to_sheets(label, confidence):
    if not sheets_ok or sheets_client is None:
        return
    try:
        sheet = sheets_client.open(SHEETS_NAME).sheet1
        sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            round(float(confidence), 3),
            "", ""   # temp/hum not available on cloud server
        ])
        print(f"Sheets: logged {label}")
    except Exception as e:
        print(f"Sheets error: {e}")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

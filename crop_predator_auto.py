"""
Smart Crop Predator Detection — Fully Automated Pipeline
=========================================================
Flow:
  ESP32-CAM stream / USB camera
        ↓
  YOLOv8 inference on every frame
        ↓
  Detection → push to Firebase Realtime Database
        ↓
  ESP32 DevKit polls Firebase → triggers deterrent

Install requirements:
  pip install ultralytics firebase-admin opencv-python pyserial gspread google-auth
"""

import cv2
import time
import json
import serial
import serial.tools.list_ports
import threading
import gspread
from google.oauth2.service_account import Credentials as GCredentials
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

try:
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("WARNING: firebase-admin not installed. Run: pip install firebase-admin")

# =============================================================================
# ── CONFIGURATION — fill these in ────────────────────────────────────────────
# =============================================================================

MODEL_PATH            = r"C:\Users\ankit\Music\Project\best.pt"
ESP32_CAM_URL         = "http://10.87.231.27:81/stream"
USB_CAMERA_INDEX      = 0
FIREBASE_CREDENTIALS  = r"C:\Users\ankit\Music\Project\firebase_cred.json"
FIREBASE_DATABASE_URL = "https://crop-protection-6f8ca-default-rtdb.asia-southeast1.firebasedatabase.app/"
CONFIDENCE_THRESHOLD  = 0.70
COOLDOWN_SECONDS      = 20
SHOW_PREVIEW          = True
CLASS_NAMES           = ["human", "locust", "monkey", "wild_boar"]

# =============================================================================


# ── Firebase ──────────────────────────────────────────────────────────────────
def init_firebase():
    if not FIREBASE_AVAILABLE:
        return False
    if not Path(FIREBASE_CREDENTIALS).exists():
        print(f"Firebase: credentials not found at {FIREBASE_CREDENTIALS}")
        return False
    try:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        print("Firebase: connected OK")
        return True
    except Exception as e:
        print(f"Firebase init error: {e}")
        return False


def push_detection(label, confidence, sensor_data=None):
    try:
        firebase_db.reference("/crop_predator/latest").set({
            "class":      label,
            "confidence": round(float(confidence), 3),
            "timestamp":  datetime.now().isoformat(),
            "unix":       int(time.time()),
            "action":     "TRIGGER"
        })
        firebase_db.reference("/crop_predator/events").push({
            "class":      label,
            "confidence": round(float(confidence), 3),
            "timestamp":  datetime.now().isoformat(),
            "unix":       int(time.time()),
            **(sensor_data or {})
        })
        print(f"Firebase: pushed {label} ({confidence*100:.1f}%)")
        return True
    except Exception as e:
        print(f"Firebase push error: {e}")
        return False


# ── Google Sheets ─────────────────────────────────────────────────────────────
sheets_client = None                          # FIX 1: kept at module level (correct)

def init_sheets():
    global sheets_client
    try:
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = GCredentials.from_service_account_file(
            FIREBASE_CREDENTIALS, scopes=scopes
        )
        sheets_client = gspread.authorize(creds)
        sheet = sheets_client.open("Crop Predator Log").sheet1
        # Write headers if sheet is empty
        if sheet.row_count == 0 or sheet.cell(1, 1).value != "Timestamp":
            sheet.insert_row(
                ["Timestamp", "Class", "Confidence", "Temperature", "Humidity"],
                index=1
            )
        print("Google Sheets: connected OK")
        return True
    except Exception as e:
        print(f"Google Sheets: disabled ({e})")
        return False


def log_to_sheets(label, confidence, temp=None, hum=None):
    global sheets_client
    if sheets_client is None:
        return
    try:
        sheet = sheets_client.open("Crop Predator Log").sheet1
        sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            round(float(confidence), 3),
            temp if temp is not None else "",
            hum  if hum  is not None else ""
        ])
        print(f"Sheets: logged {label}")
    except Exception as e:
        print(f"Sheets error: {e}")


# ── Serial ────────────────────────────────────────────────────────────────────
def find_devkit_port():
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if any(x in desc for x in ["cp210", "ch340", "uart", "esp32", "usb serial"]):
            return p.device
    return None


def open_serial(port, baud=115200):
    try:
        s = serial.Serial(port, baud, timeout=0.5)
        print(f"Serial: {port} connected at {baud} baud")
        return s
    except Exception as e:
        print(f"Serial: could not open {port} — {e}")
        return None


# ── Camera ────────────────────────────────────────────────────────────────────
def open_camera():
    if ESP32_CAM_URL:
        cap = cv2.VideoCapture(ESP32_CAM_URL)
        if cap.isOpened():
            print(f"Camera: ESP32-CAM stream connected ({ESP32_CAM_URL})")
            return cap
        print("Camera: ESP32-CAM stream failed — falling back to USB camera")

    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(USB_CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"Camera: USB camera {USB_CAMERA_INDEX} opened")
        return cap

    print("Camera: ERROR — no camera found")
    return None


# ── Serial listener thread ────────────────────────────────────────────────────
def serial_listener(ser):
    """Reads JSON sensor data from DevKit in background."""
    while True:
        try:
            if ser and ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    print(f"Sensors: {data}")
                    if firebase_ok:
                        firebase_db.reference("/crop_predator/sensors").set({
                            **data,
                            "timestamp": datetime.now().isoformat()
                        })
        except Exception:
            pass
        time.sleep(0.05)


# ── Overlay ───────────────────────────────────────────────────────────────────
COLOR_MAP = {
    "monkey":    (0, 0, 255),
    "wild_boar": (0, 80, 255),
    "locust":    (0, 165, 255),
    "human":     (0, 200, 50),
}

def draw_status(frame, label=None, conf=None, cooldown_left=0):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.putText(frame, "Smart Crop Predator Detection",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 180), 1)
    if label and conf is not None:
        color = COLOR_MAP.get(label, (255, 255, 255))
        text  = f"{label.upper()}  {conf*100:.0f}%"
        cv2.rectangle(frame, (0, h-50), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, text, (12, h-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if cooldown_left > 0:
            cv2.putText(frame, f"cooldown {cooldown_left:.0f}s",
                        (w-160, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (10, h-56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global firebase_ok, sheets_ok

    print("=" * 55)
    print("  Smart Crop Predator Detection — Auto Pipeline")
    print("=" * 55)

    # FIX 2: both inits at correct indentation inside main()
    firebase_ok = init_firebase()
    sheets_ok   = init_sheets()

    model = YOLO(MODEL_PATH)
    print(f"Model: loaded {MODEL_PATH}")
    print(f"Classes: {model.names}")

    cap = open_camera()
    if cap is None:
        print("ERROR: No camera. Exiting.")
        return

    port = find_devkit_port()
    ser  = open_serial(port) if port else None
    if not ser:
        print("Serial: DevKit not found — detections will still go to Firebase")

    if ser:
        t = threading.Thread(target=serial_listener, args=(ser,), daemon=True)
        t.start()

    last_detection_time = 0
    last_label = None
    last_conf  = None

    # FIX 3: store latest sensor data so Sheets logging gets temp/hum
    latest_sensors = {}

    print("\nRunning — press Q to quit, SPACE to force-send last detection\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        results   = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        annotated = results[0].plot()

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
            last_detection_time = now
            last_label = best_label
            last_conf  = best_conf

            print(f"\n{'='*40}")
            print(f"DETECTED: {best_label.upper()} ({best_conf*100:.1f}%)")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*40}")

            # 1. Push to Firebase
            if firebase_ok:
                push_detection(best_label, best_conf, latest_sensors or None)

            # 2. Log to Google Sheets     FIX 4: moved here so auto-detections log too
            if sheets_ok:
                log_to_sheets(
                    best_label,
                    best_conf,
                    temp=latest_sensors.get("temp"),
                    hum=latest_sensors.get("hum")
                )

            # 3. Send over Serial as backup
            if ser:
                try:
                    ser.write((best_label.upper() + "\n").encode())
                    print(f"Serial: sent {best_label.upper()} to DevKit")
                except Exception as e:
                    print(f"Serial send error: {e}")

        annotated = draw_status(annotated, last_label, last_conf, cooldown_left)

        if SHOW_PREVIEW:
            cv2.imshow("Crop Predator Detection", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # FIX 5: SPACE uses last_label/last_conf not best_label/best_conf
            # (best_label may be None if no detection in current frame)
            if key == ord(' ') and last_label:
                print(f"Manual trigger: {last_label}")
                if firebase_ok:
                    push_detection(last_label, last_conf or 0.9)
                if sheets_ok:
                    log_to_sheets(last_label, last_conf or 0.9)
                if ser:
                    ser.write((last_label.upper() + "\n").encode())
        else:
            time.sleep(0.01)

    cap.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()

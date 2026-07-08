"""
Smart Crop Predator Detection — Windows PC Inference Script
============================================================
What this does:
  1. Listens on USB Serial for "MOTION" trigger from ESP32 DevKit (PIR fired)
  2. Captures a frame from USB camera via OpenCV
  3. Resizes to 96x96, runs TFLite INT8 inference
  4. Sends detected class label back to DevKit over Serial
  5. Logs event to Firebase Realtime Database
  6. Shows live camera preview window with detection overlay

Requirements (install once):
  pip install opencv-python tflite-runtime pyserial firebase-admin numpy

Usage:
  python crop_predator_inference.py

Make sure:
  - ESP32 DevKit is connected via USB
  - USB camera is plugged in
  - crop_predator_model.tflite is in the same folder as this script
  - Fill in your config values below
"""

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import threading
import json
from datetime import datetime
from pathlib import Path

# ── Try importing TFLite runtime ─────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite_runtime")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("Using tensorflow.lite")
    except ImportError:
        print("ERROR: Install tflite_runtime:  pip install tflite-runtime")
        exit(1)

# ── Try importing Firebase ────────────────────────────────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db
    FIREBASE_AVAILABLE = True
except ImportError:
    print("WARNING: firebase-admin not installed. Firebase logging disabled.")
    print("Install with:  pip install firebase-admin")
    FIREBASE_AVAILABLE = False

# =============================================================================
# CONFIGURATION — fill these in
# =============================================================================

# Serial port of your ESP32 DevKit
# Windows: "COM3", "COM4" etc. — check Device Manager
# To auto-detect, set to None and the script will find it
SERIAL_PORT = "COM3"           # e.g. "COM4"
SERIAL_BAUD = 115200

# USB camera index (0 = first camera, 1 = second, etc.)
CAMERA_INDEX = 0

# Path to your TFLite model (downloaded from Colab)
MODEL_PATH = "crop_predator_model.tflite"

# Image size — must match what you trained with (96 or 128)
IMG_SIZE = 96

# Minimum confidence to send a detection (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.70

# Firebase config — leave blank strings to disable Firebase
FIREBASE_CREDENTIALS_JSON = "firebase_credentials.json"   # Download from Firebase console
FIREBASE_DATABASE_URL      = "https://YOUR_PROJECT.firebaseio.com"

# Class names — must match your Roboflow/Colab training order (alphabetical)
CLASS_NAMES = ["bird", "deer", "human", "monkey", "wild_boar"]

# Cooldown between detections (seconds) — prevents spam
COOLDOWN_SECONDS = 30

# Show live preview window (set False if running headless / no monitor)
SHOW_PREVIEW = True

# =============================================================================

# ── State ─────────────────────────────────────────────────────────────────────
last_detection_time = 0
detection_result    = None   # Shared between threads
lock = threading.Lock()

# ── Colors for preview overlay (BGR) ─────────────────────────────────────────
CLASS_COLORS = {
    "bird":      (0, 200, 100),
    "deer":      (0, 160, 255),
    "human":     (100, 100, 255),
    "monkey":    (0, 80, 255),
    "wild_boar": (0, 40, 200),
}

# =============================================================================
# FIREBASE SETUP
# =============================================================================

def init_firebase():
    if not FIREBASE_AVAILABLE:
        return False
    if not FIREBASE_DATABASE_URL or "YOUR_PROJECT" in FIREBASE_DATABASE_URL:
        print("Firebase: database URL not configured — skipping")
        return False
    cred_path = Path(FIREBASE_CREDENTIALS_JSON)
    if not cred_path.exists():
        print(f"Firebase: credentials file not found at {cred_path} — skipping")
        print("Download it from Firebase Console → Project Settings → Service Accounts")
        return False
    try:
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        print("Firebase: connected OK")
        return True
    except Exception as e:
        print(f"Firebase init failed: {e}")
        return False


def log_to_firebase(class_name, confidence, sensor_data=None):
    """Push a detection event to Firebase Realtime Database."""
    try:
        ref  = firebase_db.reference("/crop_predator/events")
        data = {
            "class":      class_name,
            "confidence": round(float(confidence), 3),
            "timestamp":  datetime.now().isoformat(),
            "unix_time":  int(time.time()),
        }
        if sensor_data:
            data.update(sensor_data)
        ref.push(data)
        print(f"Firebase: logged {class_name} ({confidence*100:.1f}%)")
    except Exception as e:
        print(f"Firebase log error: {e}")

# =============================================================================
# SERIAL SETUP
# =============================================================================

def find_esp32_port():
    """Auto-detect ESP32 DevKit COM port by looking for CP210x or CH340."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        if any(x in desc for x in ["cp210", "ch340", "uart", "usb serial", "esp32"]):
            print(f"Auto-detected ESP32 on {p.device} ({p.description})")
            return p.device
    # Fallback: just pick the first available port
    if ports:
        print(f"No ESP32 found by name — using first port: {ports[0].device}")
        return ports[0].device
    return None


def open_serial(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        print(f"Serial: opened {port} at {baud} baud")
        return ser
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return None

# =============================================================================
# TFLITE MODEL
# =============================================================================

def load_model(model_path):
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Download crop_predator_model.tflite from your Colab notebook (Cell 13)")
        return None, None, None
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"Model loaded: {model_path}")
    print(f"  Input  shape : {input_details['shape']}  dtype: {input_details['dtype']}")
    print(f"  Output shape : {output_details['shape']} dtype: {output_details['dtype']}")
    return interpreter, input_details, output_details


def run_inference(interpreter, input_details, output_details, frame_bgr):
    """
    Takes a BGR frame from OpenCV, resizes, quantizes, runs inference.
    Returns (class_name, confidence, all_scores_dict)
    """
    # Resize to model input size
    img = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))

    # Convert BGR → RGB (model was trained on RGB images)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0.0, 1.0]
    img_float = img.astype(np.float32) / 255.0

    # Quantize float32 → int8 using model's scale/zero_point
    scale      = input_details["quantization_parameters"]["scales"][0]
    zero_point = input_details["quantization_parameters"]["zero_points"][0]

    if input_details["dtype"] == np.int8:
        img_q = (img_float / scale + zero_point).astype(np.int8)
    elif input_details["dtype"] == np.uint8:
        img_q = (img_float / scale + zero_point).astype(np.uint8)
    else:
        # Float model (no quantization)
        img_q = img_float

    # Add batch dimension: (96, 96, 3) → (1, 96, 96, 3)
    input_data = np.expand_dims(img_q, axis=0)

    # Run inference
    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()

    # Read output
    output_data = interpreter.get_tensor(output_details["index"])[0]

    # Dequantize output if INT8
    if output_details["dtype"] == np.int8:
        out_scale = output_details["quantization_parameters"]["scales"][0]
        out_zp    = output_details["quantization_parameters"]["zero_points"][0]
        scores = (output_data.astype(np.float32) - out_zp) * out_scale
    else:
        scores = output_data.astype(np.float32)

    # Get best class
    best_idx  = int(np.argmax(scores))
    best_conf = float(scores[best_idx])
    best_name = CLASS_NAMES[best_idx] if best_idx < len(CLASS_NAMES) else "unknown"

    all_scores = {CLASS_NAMES[i]: float(scores[i]) for i in range(min(len(CLASS_NAMES), len(scores)))}

    return best_name, best_conf, all_scores

# =============================================================================
# CAMERA
# =============================================================================

def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # CAP_DSHOW faster on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)               # Fallback without DSHOW
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {index}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Camera {index} opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap

# =============================================================================
# PREVIEW OVERLAY
# =============================================================================

def draw_overlay(frame, class_name=None, confidence=None, all_scores=None, status="Monitoring..."):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top status bar
    cv2.rectangle(overlay, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(overlay, f"Crop Predator Detection  |  {status}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 200), 1)

    # Detection result
    if class_name and confidence:
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        label = f"{class_name.upper()}  {confidence*100:.1f}%"
        cv2.rectangle(overlay, (0, h-60), (w, h), (20, 20, 20), -1)
        cv2.putText(overlay, label, (12, h-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # Confidence bar
        bar_w = int((w - 24) * confidence)
        cv2.rectangle(overlay, (12, h-16), (w-12, h-8), (60, 60, 60), -1)
        cv2.rectangle(overlay, (12, h-16), (12+bar_w, h-8), color, -1)

    # Score sidebar (small, top-right)
    if all_scores:
        x0, y0 = w - 160, 46
        cv2.rectangle(overlay, (x0-4, y0-4), (w-4, y0 + len(all_scores)*22 + 4), (20, 20, 20), -1)
        for i, (cls, score) in enumerate(sorted(all_scores.items(), key=lambda x: -x[1])):
            y = y0 + i * 22
            bar_len = int(140 * score)
            bar_color = CLASS_COLORS.get(cls, (120, 120, 120))
            cv2.rectangle(overlay, (x0, y+4), (x0+140, y+16), (50, 50, 50), -1)
            cv2.rectangle(overlay, (x0, y+4), (x0+bar_len, y+16), bar_color, -1)
            cv2.putText(overlay, f"{cls[:10]} {score*100:.0f}%",
                        (x0, y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(overlay, ts, (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    return cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)

# =============================================================================
# SERIAL LISTENER THREAD
# =============================================================================

def serial_listener(ser, cap, interpreter, input_details, output_details, firebase_ok):
    """
    Runs in a background thread.
    Waits for "MOTION" from DevKit → captures frame → runs inference → sends result.
    """
    global last_detection_time, detection_result

    print("Serial listener started — waiting for PIR trigger from DevKit...")

    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                print(f"[DevKit] {line}")

                # DevKit sends "MOTION" when PIR fires
                if line.upper() == "MOTION":
                    now = time.time()
                    cooldown_remaining = COOLDOWN_SECONDS - (now - last_detection_time)

                    if cooldown_remaining > 0:
                        print(f"In cooldown — {cooldown_remaining:.0f}s remaining")
                        ser.write(b"COOLDOWN\n")
                        continue

                    print("PIR triggered — capturing frame...")
                    ret, frame = cap.read()
                    if not ret:
                        print("Camera capture failed")
                        ser.write(b"NO_DETECTION\n")
                        continue

                    # Run inference
                    t0 = time.time()
                    class_name, confidence, all_scores = run_inference(
                        interpreter, input_details, output_details, frame
                    )
                    infer_ms = (time.time() - t0) * 1000

                    print(f"Inference: {infer_ms:.0f}ms")
                    for cls, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                        marker = " ◄" if cls == class_name else ""
                        print(f"  {cls:<12} {score*100:.1f}%{marker}")

                    if confidence >= CONFIDENCE_THRESHOLD:
                        print(f"DETECTED: {class_name} ({confidence*100:.1f}%)")

                        # Send result to DevKit
                        result_str = f"{class_name}\n"
                        ser.write(result_str.encode("utf-8"))

                        # Update shared state for preview overlay
                        with lock:
                            detection_result = (class_name, confidence, all_scores, frame.copy())

                        # Log to Firebase
                        if firebase_ok:
                            log_to_firebase(class_name, confidence)

                        last_detection_time = time.time()

                    else:
                        print(f"Low confidence ({confidence*100:.1f}%) — no detection sent")
                        ser.write(b"NO_DETECTION\n")

                # DevKit can also send sensor data as JSON
                # e.g. {"temp":28.5,"humidity":72,"distance":1.2,"soil":45}
                elif line.startswith("{"):
                    try:
                        sensor_data = json.loads(line)
                        print(f"Sensor data: {sensor_data}")
                        # Update Firebase with latest sensor reading
                        if firebase_ok:
                            firebase_db.reference("/crop_predator/sensors").set({
                                **sensor_data,
                                "timestamp": datetime.now().isoformat()
                            })
                    except json.JSONDecodeError:
                        pass

        except serial.SerialException as e:
            print(f"Serial disconnected: {e}")
            break
        except Exception as e:
            print(f"Listener error: {e}")

        time.sleep(0.02)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 55)
    print("  Smart Crop Predator Detection — Inference Engine")
    print("=" * 55)

    # Firebase
    firebase_ok = init_firebase()

    # Model
    interpreter, input_details, output_details = load_model(MODEL_PATH)
    if interpreter is None:
        return

    # Camera
    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        return

    # Serial
    port = SERIAL_PORT or find_esp32_port()
    if port is None:
        print("ERROR: No serial port found. Check DevKit is connected and drivers installed.")
        cap.release()
        return

    ser = open_serial(port, SERIAL_BAUD)
    if ser is None:
        cap.release()
        return

    print("\nSystem ready. Press Q in preview window to quit.\n")

    # Start serial listener in background thread
    t = threading.Thread(
        target=serial_listener,
        args=(ser, cap, interpreter, input_details, output_details, firebase_ok),
        daemon=True
    )
    t.start()

    # Main thread: preview window
    global detection_result
    last_result  = None
    result_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        # Check for new detection result
        with lock:
            if detection_result is not None:
                last_result  = detection_result
                result_timer = time.time()
                detection_result = None

        # Show result for 10 seconds then clear
        if last_result and (time.time() - result_timer) < 10:
            cls, conf, scores, _ = last_result
            status = f"DETECTED: {cls.upper()}"
            disp = draw_overlay(frame, cls, conf, scores, status)
        else:
            last_result = None
            disp = draw_overlay(frame, status="Monitoring — waiting for motion...")

        if SHOW_PREVIEW:
            cv2.imshow("Crop Predator Detection", disp)
            key = cv2.waitKey(1) & 0xFF

            # Press Q to quit
            if key == ord("q"):
                print("Quitting...")
                break

            # Press SPACE to manually trigger inference (for testing without PIR)
            if key == ord(" "):
                print("Manual trigger (SPACE key)...")
                cls, conf, scores = run_inference(interpreter, input_details, output_details, frame)
                print(f"Manual result: {cls} ({conf*100:.1f}%)")
                with lock:
                    detection_result = (cls, conf, scores, frame.copy())
                if conf >= CONFIDENCE_THRESHOLD and firebase_ok:
                    log_to_firebase(cls, conf)

        else:
            time.sleep(0.03)

    # Cleanup
    cap.release()
    ser.close()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()

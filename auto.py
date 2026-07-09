from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
import json


# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------

MODEL_PATH = "best.pt"

DATABASE_URL = "https://crop-protection-6f8ca-default-rtdb.asia-southeast1.firebasedatabase.app/"

firebase_json = json.loads(os.environ["FIREBASE_CREDENTIALS"])

cred = credentials.Certificate(firebase_json)

firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": DATABASE_URL
    }
)
# ---------------------------------------------------

model = None

@app.on_event("startup")
def load_model():
    global model
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("Model Loaded.")
print(model.names)

app = FastAPI(title="Crop Protection API")

# ---------------------------------------------------
# HOME
# ---------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "Running",
        "model": MODEL_PATH,
        "classes": model.names
    }

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    suffix = "." + file.filename.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:

        temp.write(await file.read())

        temp_path = temp.name

    results = model.predict(
        temp_path,
        conf=0.70,
        verbose=False
    )

    os.remove(temp_path)

    best_class = None
    best_conf = 0

    detections = []

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            label = model.names[cls]

            detections.append(
                {
                    "class": label,
                    "confidence": round(conf,3)
                }
            )

            if conf > best_conf:

                best_conf = conf

                best_class = label

    # ---------------------------------------
    # Upload to Firebase
    # ---------------------------------------

    if best_class is not None:

        db.reference("/crop_predator/latest").set(

            {

                "class": best_class,

                "confidence": round(best_conf,3),

                "action":"TRIGGER",

                "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            }

        )

    else:

        db.reference("/crop_predator/latest").set(

            {

                "class":"None",

                "confidence":0,

                "action":"NONE",

                "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            }

        )

    return {

        "success":True,

        "best_class":best_class,

        "confidence":round(best_conf,3),

        "detections":detections

    }

# ---------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------

@app.get("/health")

def health():

    return {

        "status":"healthy"

    }

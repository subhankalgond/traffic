import os
import time
import threading
import base64
from datetime import datetime

import cv2
import requests
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "LntDwq5p2x7JvDG7GM2d").strip()
# Exact model ID for Roboflow Deploy: project/version
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "ambulance-wmxl5-6rgov/1").strip()
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.25"))
SIGNAL_HOLD_SECONDS = int(os.getenv("SIGNAL_HOLD_SECONDS", "12"))
INFERENCE_INTERVAL_SECONDS = float(os.getenv("INFERENCE_INTERVAL_SECONDS", "0.5"))

lock = threading.Lock()
state = {
    "detected": False,
    "signal": "RED",
    "source": "Waiting for camera",
    "last_event": "System started",
    "last_seen": None,
    "mode": "Roboflow hosted AI",
    "camera_ok": False,
    "model_ok": False,
    "confidence": None,
}

camera = None
last_detection_time = 0
manual_trigger_until = 0
last_api_error_time = 0


def log_event(message):
    with lock:
        state["last_event"] = (
            f"{datetime.now().strftime('%H:%M:%S')} — {message}"
        )


def load_model():
    if ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID:
        with lock:
            state["model_ok"] = True
            state["mode"] = f"Roboflow hosted AI configured: {ROBOFLOW_MODEL_ID}"
        log_event("Roboflow settings found; hosted AI is configured")
    else:
        missing = []
        if not ROBOFLOW_API_KEY:
            missing.append("ROBOFLOW_API_KEY")
        if not ROBOFLOW_MODEL_ID:
            missing.append("ROBOFLOW_MODEL_ID")
        with lock:
            state["model_ok"] = False
            state["mode"] = "Missing setting: " + ", ".join(missing)
        log_event("Set the missing Roboflow setting(s) in Command Prompt")


def get_camera():
    global camera
    if camera is None:
        try:
            if os.name == "nt":
                camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
                if not camera.isOpened():
                    camera = cv2.VideoCapture(CAMERA_INDEX)
            else:
                camera = cv2.VideoCapture(CAMERA_INDEX)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception as e:
            log_event(f"Local camera init exception: {e}")
            camera = None
    return camera


def detect_ambulance_from_b64(image_base64, width=640, height=360, conf=None):
    """Send base64-encoded image to Roboflow hosted model and return matching bounding boxes."""
    global last_api_error_time

    if not ROBOFLOW_API_KEY or not ROBOFLOW_MODEL_ID:
        return []

    active_conf = conf if conf is not None else CONFIDENCE
    threshold = max(0.10, min(0.95, float(active_conf)))
    url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"

    try:
        response = requests.post(
            url,
            params={
                "api_key": ROBOFLOW_API_KEY,
                "confidence": int(threshold * 100),
            },
            data=image_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=8,
        )
        response.raise_for_status()
        result = response.json()
        predictions = result.get("predictions", [])

        matches = []
        for prediction in predictions:
            label = str(prediction.get("class", "")).lower()
            confidence = float(prediction.get("confidence", 0))
            if "ambulance" not in label or confidence < threshold:
                continue

            # Roboflow object-detection responses use center x/y and width/height.
            x = float(prediction.get("x", 0))
            y = float(prediction.get("y", 0))
            w_box = float(prediction.get("width", 0))
            h_box = float(prediction.get("height", 0))
            matches.append({
                "label": label,
                "confidence": confidence,
                "x1": max(0, int(x - w_box / 2)),
                "y1": max(0, int(y - h_box / 2)),
                "x2": min(width - 1, int(x + w_box / 2)),
                "y2": min(height - 1, int(y + h_box / 2)),
            })
        return matches

    except (requests.RequestException, ValueError) as exc:
        now = time.time()
        # Avoid filling the dashboard event field with the same error every frame.
        if now - last_api_error_time >= 10:
            last_api_error_time = now
            log_event(f"Roboflow request failed: {exc}")
        return []


def detect_ambulance_api(frame):
    """Helper for opencv video loop: encodes frame and invokes detect_ambulance_from_b64."""
    ok, jpg = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )
    if not ok:
        return []

    image_base64 = base64.b64encode(jpg.tobytes()).decode("utf-8")
    h, w = frame.shape[:2]
    return detect_ambulance_from_b64(image_base64, width=w, height=h)



def update_signal(detected, source="Camera"):
    global last_detection_time
    now = time.time()
    with lock:
        previous = state["detected"]
        state["detected"] = bool(detected)
        if detected:
            last_detection_time = now
            state["signal"] = "GREEN"
            state["source"] = source
            state["last_seen"] = datetime.now().strftime("%H:%M:%S")
            if not previous:
                state["last_event"] = (
                    f"{state['last_seen']} — Ambulance detection triggered"
                )
        elif (
            now - last_detection_time > SIGNAL_HOLD_SECONDS
            and now > manual_trigger_until
        ):
            state["signal"] = "RED"
            state["source"] = "No ambulance detected"
        # This is only a website demonstration, not a real road traffic controller.


def generate_frames():
    cap = get_camera()
    last_inference_time = 0
    cached_predictions = []

    while True:
        ok, frame = (False, None) if cap is None else cap.read()
        if not ok:
            with lock:
                state["camera_ok"] = False
            import numpy as np
            blank = np.full((360, 640, 3), 255, dtype="uint8")
            cv2.putText(
                blank,
                "Camera unavailable - check webcam permissions",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 180),
                2,
            )
            ok2, encoded = cv2.imencode(".jpg", blank)
            if ok2:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + encoded.tobytes() + b"\r\n"
                )
            time.sleep(0.5)
            continue

        with lock:
            state["camera_ok"] = True

        now = time.time()
        if now - last_inference_time >= INFERENCE_INTERVAL_SECONDS:
            cached_predictions = detect_ambulance_api(frame)
            last_inference_time = now

        detected = bool(cached_predictions)
        best_conf = None

        for prediction in cached_predictions:
            best_conf = max(best_conf or 0, prediction["confidence"])
            x1, y1, x2, y2 = (
                prediction["x1"], prediction["y1"],
                prediction["x2"], prediction["y2"],
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 3)
            cv2.putText(
                frame,
                f"AMBULANCE {prediction['confidence']:.2f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 0),
                2,
            )

        if time.time() < manual_trigger_until:
            detected = True
            cv2.putText(
                frame, "DEMO TRIGGER ACTIVE", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3
            )

        if detected and cached_predictions:
            source = "Roboflow AI camera detection"
        elif detected:
            source = "Manual demo trigger"
        else:
            source = "Roboflow camera check"

        update_signal(detected, source)

        with lock:
            state["confidence"] = round(best_conf, 3) if best_conf is not None else None
            signal = state["signal"]

        color = (0, 200, 0) if signal == "GREEN" else (0, 0, 220)
        cv2.putText(
            frame, f"DEMO SIGNAL: {signal}", (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3
        )

        ok2, encoded = cv2.imencode(".jpg", frame)
        if ok2:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes() + b"\r\n"
            )


SAMPLE_AMBULANCE_URL = "https://source.roboflow.com/S9LZkfoGB7UBir4iHIIWFIqDU203/aOkWy9z2wqwfmYgT139Y/original.jpg"
SAMPLE_CIVILIAN_URL = "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?w=800&auto=format&fit=crop&q=80"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def status():
    with lock:
        return jsonify(dict(state))


@app.route("/api/detect_frame", methods=["POST"])
def detect_frame():
    """Receives image base64 from browser webcam or upload, runs AI inference, and updates signal state."""
    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get("image", "")
        source_label = data.get("source", "Browser Optical Sensor")
        width = int(data.get("width", 640))
        height = int(data.get("height", 360))

        if not image_data:
            return jsonify({"ok": False, "error": "No image data provided"}), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        client_conf = float(data.get("confidence", CONFIDENCE))
        matches = detect_ambulance_from_b64(
            image_data, width=width, height=height, conf=client_conf
        )
        detected = bool(matches)
        best_conf = max([m["confidence"] for m in matches], default=None)

        update_signal(detected, f"{source_label} (Roboflow AI)" if detected else source_label)
        with lock:
            state["confidence"] = round(best_conf, 3) if best_conf is not None else None
            signal = state["signal"]

        return jsonify({
            "ok": True,
            "detected": detected,
            "predictions": matches,
            "signal": signal,
            "confidence": state["confidence"],
            "source": state["source"]
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sample_data/<kind>")
def get_sample_data(kind):
    """Provides bundled test images for 1-click cloud demo testing without a physical camera."""
    try:
        url = SAMPLE_AMBULANCE_URL if kind == "ambulance" else SAMPLE_CIVILIAN_URL
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            b64 = base64.b64encode(resp.content).decode("utf-8")
            return jsonify({
                "ok": True,
                "image": f"data:image/jpeg;base64,{b64}",
                "kind": kind
            })
        return jsonify({"ok": False, "error": "Could not fetch sample"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/demo_trigger", methods=["POST"])
def demo_trigger():
    global manual_trigger_until
    manual_trigger_until = time.time() + 10
    log_event("Manual demo trigger activated for 10 seconds")
    return jsonify({"ok": True, "message": "Demo trigger active for 10 seconds"})


@app.route("/api/reset", methods=["POST"])
def reset():
    global manual_trigger_until
    manual_trigger_until = 0
    update_signal(False, "Manual reset")
    log_event("Manual demo trigger reset")
    return jsonify({"ok": True})


# Initialize model on module import (required for gunicorn on Render)
load_model()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    print(f"\nAmbulance Traffic Dispatcher: http://{host}:{port}")
    print(f"Roboflow Model ID: {ROBOFLOW_MODEL_ID}")
    app.run(host=host, port=port, debug=False, threaded=True)


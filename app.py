import atexit
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template

from gesture_recognition import process_frame


app = Flask(__name__)

camera = cv2.VideoCapture(0)
state_lock = threading.Lock()
latest_gesture = "Waiting for camera"
latest_frame = None
camera_running = True


def release_camera():
    global camera_running

    camera_running = False
    if camera.isOpened():
        camera.release()


atexit.register(release_camera)


def camera_worker():
    global latest_frame, latest_gesture

    while camera_running:
        success, frame = camera.read()
        if not success:
            with state_lock:
                latest_gesture = "Camera not available"
            time.sleep(0.25)
            continue

        try:
            result = process_frame(frame)
        except Exception:
            # Match the original script's behavior: ignore bad frames and keep reading.
            time.sleep(0.1)
            continue

        success, encoded_frame = cv2.imencode(".jpg", result.frame)
        with state_lock:
            latest_gesture = result.label
            if success:
                latest_frame = encoded_frame.tobytes()

        time.sleep(0.03)


camera_thread = threading.Thread(target=camera_worker, daemon=True)
camera_thread.start()


def generate_frames():
    while True:
        with state_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/gesture")
def gesture():
    with state_lock:
        gesture_label = latest_gesture
    return jsonify({"gesture": gesture_label})


if __name__ == "__main__":
    if not camera.isOpened():
        print("Warning: could not open webcam. Check camera permissions or try another camera index.")

    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False, threaded=True)

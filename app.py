from flask import Flask, render_template, Response
import face_recognition
import cv2
import pickle
import time
import threading
from playsound import playsound

app = Flask(__name__)

# Load trained SVM model once
a = open("face_recognition_model.pkl", "rb")
clf = pickle.load(a)
a.close()

# Buzzer settings (we'll skip this for now)
BUZZER_SOUND = "Buzzer_alert.mp3"
last_buzz_time = 0
buzz_interval = 3  # seconds

def buzz_alert():
    threading.Thread(target=playsound, args=(BUZZER_SOUND,), daemon=True).start()

# Configuration
USE_IP_CAMERA = False  # Set this to True if you want to use IP camera
IP_CAMERA_URL = "rtsp://<IP_ADDRESS>/stream"  # Replace with actual IP camera URL
VIDEO_SOURCE = 0 if not USE_IP_CAMERA else IP_CAMERA_URL  # Choose webcam or IP camera

# Initialize camera (either webcam or IP camera)
video_capture = cv2.VideoCapture(VIDEO_SOURCE)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame generator with optimized detection
def gen_frames():
    scale = 0.5               # downscale factor for detection
    process_every_n = 2       # process every 2nd frame
    frame_count = 0
    face_locations = []
    face_names = []
    face_confidences = []

    while True:
        success, frame = video_capture.read()
        if not success:
            continue

        frame_count += 1
        # Perform heavy detection only on selected frames
        if frame_count % process_every_n == 0:
            # Downscale frame for faster detection
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces & encodings on the small frame
            face_locations = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, face_locations)

            face_names = []
            face_confidences = []
            for enc in encodings:
                proba = clf.predict_proba([enc])[0]
                conf = proba.max()
                name = clf.classes_[proba.argmax()] if conf > 0.7 else "Unknown"
                face_names.append(name)
                face_confidences.append(conf)

        # Draw boxes on original frame (scale coords back)
        for (top, right, bottom, left), name, conf in zip(face_locations, face_names, face_confidences):
            top = int(top / scale)
            right = int(right / scale)
            bottom = int(bottom / scale)
            left = int(left / scale)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} ({conf:.2f})",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    video_capture.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

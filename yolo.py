import cv2
import torch
import face_recognition
import pickle
import numpy as np
from ultralytics import YOLO  # Correct YOLO import

# Load YOLOv8 model
model = YOLO("D:/FaceRecognition/runs/detect/train4/weights/best.pt")  # Correct model loading

# Load trained SVM model
with open("face_recognition_model.pkl", "rb") as f:
    clf = pickle.load(f)

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 🔹 Step 1: Use YOLO to Detect Faces
    results = model(frame)  # YOLO inference
    faces = results[0].boxes.data.cpu().numpy() if results else []  # Extract boxes correctly

    for face in faces:
        x1, y1, x2, y2, conf, cls = face
        if conf < 0.6:  # Ignore low-confidence detections
            continue

        # Convert YOLO box format to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Extract face ROI for encoding
        face_encodings = face_recognition.face_encodings(rgb_frame, [(y1, x2, y2, x1)])

        if face_encodings:
            encoding = face_encodings[0]
            name = "Unknown"

            # 🔹 Step 2: Predict using SVM model
            prediction = clf.predict([encoding])
            confidence = clf.predict_proba([encoding]).max()

            if confidence > 0.7:
                name = prediction[0]

            # Draw bounding box and name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO + SVM Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

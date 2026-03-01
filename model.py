import face_recognition
import cv2
import pickle
import numpy as np

# Load trained SVM model
with open("face_recognition_model.pkl", "rb") as f:
    clf = pickle.load(f)

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        name = "Unknown"

        # Predict using trained SVM model
        prediction = clf.predict([face_encoding])
        confidence = clf.predict_proba([face_encoding]).max()

        if confidence > 0.7:  # Confidence threshold
            name = prediction[0]

        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

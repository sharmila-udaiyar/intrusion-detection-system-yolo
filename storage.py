import os
import face_recognition
import pickle
import numpy as np
from sklearn.svm import SVC  

# Path to dataset folder
dataset_path = r"C:/PROJECTS/FaceRecognition KP/FR dataset/dataset/Classified"

encodings = []
labels = []

# Loop through dataset
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if os.path.isdir(person_folder):  # Check if it's a folder
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = face_recognition.load_image_file(image_path)

            # Extract face encodings
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:  # If face detected
                encodings.append(face_encodings[0])
                labels.append(person_name)

# Train an SVM classifier
clf = SVC(kernel="rbf ", probability=True)
clf.fit(encodings, labels)

# Save model
with open("face_recognition_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Face recognition model trained and saved!")























import os
import cv2
import pickle
import face_recognition
from sklearn.svm import SVC

dataset_path = r"C:/PROJECTS/FaceRecognition KP/IDS_Final/guards dataset"
encodings = []
labels = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if os.path.isdir(person_folder):
        print(f"🔍 Processing: {person_name}")
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {image_path}")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)

            if not face_locations:
                print(f"❌ No face detected in {image_name}")
                continue

            enc = face_recognition.face_encodings(rgb_img, known_face_locations=face_locations)
            if enc:
                encodings.append(enc[0])
                labels.append(person_name)
            else:
                print(f"❌ Encoding failed for {image_name}")

print(f"\n✅ Total encodings: {len(encodings)}")

# Train SVM
if encodings:
    clf = SVC(kernel="rbf", probability=True)
    clf.fit(encodings, labels)

    with open("face_recognition_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("\n🎉 SVM model trained and saved as face_recognition_model.pkl")
else:
    print("❌ No encodings available. Model not trained.")
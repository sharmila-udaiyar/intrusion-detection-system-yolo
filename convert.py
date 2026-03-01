from PIL import Image
import os

folder_path = "D:\FaceRecognition\dataset\Classified\Person5_Swapnil"

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        file_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(file_path)
            img.save(file_path.replace(".jpg", ".png").replace(".jpeg", ".png"))
            os.remove(file_path)  # Remove original corrupt file
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

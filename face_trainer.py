import os
import cv2
import numpy as np
from PIL import Image


class FaceTrainer:
    DATASET_DIR = "dataset"
    TRAINER_DIR = "trainer"
    MODEL_PATH = os.path.join("trainer", "trainer.yml")

    def get_images_and_labels(self):
        image_paths = [
            os.path.join(self.DATASET_DIR, f)
            for f in os.listdir(self.DATASET_DIR)
            if f.lower().endswith(".jpg")
        ]

        face_samples = []
        ids = []

        for image_path in image_paths:
            pil_img = Image.open(image_path).convert("L")
            img_array = np.array(pil_img, dtype="uint8")

            # Filename format: User.<id>.<sample>.jpg
            parts = os.path.basename(image_path).split(".")
            try:
                student_id = int(parts[1])
            except (IndexError, ValueError):
                continue

            face_samples.append(img_array)
            ids.append(student_id)

        return face_samples, np.array(ids)

    def train(self) -> int:
        os.makedirs(self.TRAINER_DIR, exist_ok=True)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = self.get_images_and_labels()

        if len(faces) == 0:
            raise ValueError("No face images found in dataset.")

        recognizer.train(faces, ids)
        recognizer.write(self.MODEL_PATH)
        return len(set(ids))

    def model_exists(self) -> bool:
        return os.path.exists(self.MODEL_PATH)

    def load_recognizer(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.MODEL_PATH)
        return recognizer

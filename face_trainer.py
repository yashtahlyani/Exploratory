import os
import cv2
import numpy as np
from PIL import Image

FACE_SIZE = (100, 100)   # resize all faces to this before training/predicting


class FaceTrainer:
    DATASET_DIR = "dataset"
    TRAINER_DIR = "trainer"
    MODEL_PATH  = os.path.join("trainer", "trainer.yml")

    def get_images_and_labels(self):
        image_paths = [
            os.path.join(self.DATASET_DIR, f)
            for f in os.listdir(self.DATASET_DIR)
            if f.lower().endswith(".jpg")
        ]

        face_samples, ids = [], []

        for path in image_paths:
            parts = os.path.basename(path).split(".")
            try:
                student_id = int(parts[1])
            except (IndexError, ValueError):
                continue

            pil_img   = Image.open(path).convert("L")
            resized   = pil_img.resize(FACE_SIZE, Image.LANCZOS)
            img_array = np.array(resized, dtype="uint8")

            face_samples.append(img_array)
            ids.append(student_id)

        return face_samples, np.array(ids)

    def train(self) -> int:
        os.makedirs(self.TRAINER_DIR, exist_ok=True)
        faces, ids = self.get_images_and_labels()

        if len(faces) == 0:
            raise ValueError("No face images found in the dataset folder.")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, ids)
        recognizer.write(self.MODEL_PATH)
        return len(set(ids))

    def model_exists(self) -> bool:
        return os.path.exists(self.MODEL_PATH)

    def load_recognizer(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.MODEL_PATH)
        return recognizer

    @staticmethod
    def prepare_face(gray_roi) -> np.ndarray:
        """Resize a face ROI to the training size before prediction."""
        resized = cv2.resize(gray_roi, FACE_SIZE, interpolation=cv2.INTER_LANCZOS4)
        return resized

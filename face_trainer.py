import os
import cv2
import numpy as np
from PIL import Image

from config import FACE_SIZE, DATASET_DIR, TRAINER_DIR, MODEL_PATH


class FaceTrainer:
    DATASET_DIR = DATASET_DIR
    TRAINER_DIR = TRAINER_DIR
    MODEL_PATH  = MODEL_PATH

    def get_images_and_labels(self) -> tuple[list[np.ndarray], np.ndarray]:
        image_paths = [
            os.path.join(self.DATASET_DIR, f)
            for f in os.listdir(self.DATASET_DIR)
            if f.lower().endswith(".jpg")
        ]

        face_samples: list[np.ndarray] = []
        ids: list[int] = []

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
        """Train the LBPH model and persist it; returns the number of unique students."""
        os.makedirs(self.TRAINER_DIR, exist_ok=True)
        faces, ids = self.get_images_and_labels()

        if len(faces) == 0:
            raise ValueError("No face images found in the dataset folder.")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, ids)
        recognizer.write(self.MODEL_PATH)
        return len(set(ids.tolist()))

    def model_exists(self) -> bool:
        return os.path.exists(self.MODEL_PATH)

    def load_recognizer(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.MODEL_PATH)
        return recognizer

    @staticmethod
    def prepare_face(gray_roi: np.ndarray) -> np.ndarray:
        """Resize a face ROI to the training size before prediction."""
        return cv2.resize(gray_roi, FACE_SIZE, interpolation=cv2.INTER_LANCZOS4)

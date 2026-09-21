# src/utils/yolo_uniform_detector.py
import cv2
import numpy as np

from src.config import Config

model = None


def _get_model():
    global model
    if model is None:
        import os
        if not os.path.isfile(Config.YOLO_PERSON_MODEL):
            raise FileNotFoundError(f"YOLO person model not found: {Config.YOLO_PERSON_MODEL}")
        from ultralytics import YOLO
        model = YOLO(Config.YOLO_PERSON_MODEL)
    return model

def is_wearing_blue_uniform(frame):
    """
    Uses YOLO to detect people and checks if their clothing is blue.
    Returns True if uniform detected, else False.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    detector = _get_model()
    results = detector.predict(frame, conf=0.5, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = detector.names[cls]
            if label == "person":
                # Get bounding box of the detected person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = frame[y1:y2, x1:x2]

                # Convert to HSV for color detection
                if person_crop.size == 0:
                    continue
                hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
                lower_blue = np.array([95, 30, 100])
                upper_blue = np.array([115, 100, 200])
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                blue_ratio = np.sum(mask > 0) / mask.size

                if blue_ratio > 0.05:
                    return True  # Wearing blue uniform
    return False  # No blue uniform detected
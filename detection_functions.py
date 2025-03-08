import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load YOLO model for weapon detection
weapon_model = YOLO("models/best1.pt")

# Load MobileNetV2 model for violence detection
violence_model = load_model("models/ModelWeights.weights.h5")


def detect_weapons(frame):
    """Runs YOLOv8 weapon detection on a frame and returns detection status."""
    results = weapon_model(frame)
    weapon_detected = False  

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = result.names[int(box.cls[0])]

            if confidence > 0.5:  # High-confidence detections
                weapon_detected = True  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return weapon_detected, frame


def detect_violence(frame):
    """Runs MobileNetV2-based violence detection and returns detection status."""
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    prediction = violence_model.predict(input_frame)[0][0]
    violence_detected = prediction > 0.5  

    # Draw label on the frame
    color = (0, 0, 255) if violence_detected else (0, 255, 0)
    label = "Violence" if violence_detected else "Non-Violence"
    cv2.putText(frame, f"{label} ({prediction:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return violence_detected, frame

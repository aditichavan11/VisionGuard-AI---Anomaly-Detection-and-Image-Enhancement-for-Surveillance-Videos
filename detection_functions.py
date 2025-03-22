import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load YOLO model for weapon detection
weapon_model = YOLO("models/best1.pt")

# Load MobileNetV2 model for violence detection
violence_model = load_model("models/ModelWeights.weights.h5")

def detect_weapons(frame):
    """
    Runs YOLOv8 weapon detection on a frame.
    Returns:
      weapon_detected (bool): Whether a weapon was detected.
      processed_frame (np.array): Frame with bounding boxes drawn.
      max_conf (float): Highest confidence score among detected weapons.
    """
    results = weapon_model(frame)
    weapon_detected = False
    max_conf = 0.0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label_index = int(box.cls[0])
            label = result.names[label_index]

            if confidence > 0.5:  # detection threshold
                weapon_detected = True
                if confidence > max_conf:
                    max_conf = confidence

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

    return weapon_detected, frame, max_conf

def detect_violence(frame):
    """
    Runs MobileNetV2-based violence detection on a frame.
    Returns:
      violence_detected (bool): Whether violence was detected.
      processed_frame (np.array): Frame with label drawn.
      prediction (float): Probability of violence (0.0 to 1.0).
    """
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    prediction = violence_model.predict(input_frame)[0][0]  # Probability
    violence_detected = prediction > 0.5

    label = "Violence" if violence_detected else "Non-Violence"
    color = (0, 0, 255) if violence_detected else (0, 255, 0)

    cv2.putText(frame, f"{label} ({prediction:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2)

    return violence_detected, frame, prediction

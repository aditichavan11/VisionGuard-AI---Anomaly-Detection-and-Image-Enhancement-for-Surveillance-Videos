import os
import cv2
from datetime import datetime
from detection_functions import detect_weapons, detect_violence  # Import detection functions

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_path = os.path.join("static/processed", os.path.basename(video_path))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    weapon_frames = []
    violence_frames = []
    timestamps = []
    detected_labels = []

    report_folder = os.path.join("static/reports", os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(report_folder, exist_ok=True)  # Ensure directory exists

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Run Weapon Detection
        yolo_frame = cv2.resize(frame, (640, 640))
        weapon_detected, yolo_processed_frame = detect_weapons(yolo_frame)

        # Run Violence Detection
        mobilenet_frame = cv2.resize(frame, (128, 128))
        violence_detected, violence_processed_frame = detect_violence(mobilenet_frame)

        # Determine Anomaly Type
        label = "No Anomaly"
        if weapon_detected and violence_detected:
            label = "Both Weapon & Violence Detected"
        elif weapon_detected:
            label = "Weapon Detected"
        elif violence_detected:
            label = "Violence Detected"

        # Save a maximum of 3 frames for each type
        if weapon_detected and len(weapon_frames) < 3:
            frame_filename = f"weapon_{len(weapon_frames)}.jpg"
            frame_path = os.path.join(report_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            weapon_frames.append(f"reports/{os.path.basename(report_folder)}/{frame_filename}")
            timestamps.append(detection_time)
            detected_labels.append("Weapon Detected")

        if violence_detected and len(violence_frames) < 3:
            frame_filename = f"violence_{len(violence_frames)}.jpg"
            frame_path = os.path.join(report_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            violence_frames.append(f"reports/{os.path.basename(report_folder)}/{frame_filename}")
            timestamps.append(detection_time)
            detected_labels.append("Violence Detected")

        out.write(yolo_processed_frame)

    cap.release()
    out.release()

    detected_frames = weapon_frames + violence_frames

    print(f"✅ Detected Frames: {detected_frames}")
    print(f"✅ Detection Timestamps: {timestamps}")
    print(f"✅ Detected Labels: {detected_labels}")

    return output_path, detected_frames, timestamps, detected_labels

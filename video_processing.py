# video_processing.py

import os
import cv2
from datetime import datetime
from detection_functions import detect_weapons, detect_violence

def process_video(video_path):
    """
    Processes the uploaded video using both weapon and violence detection on each frame.
    Saves the detection frames as original images (without direct enhancement),
    so that users can request enhancement on demand.
    
    Returns:
      output_path: Path to the processed video.
      saved_weapon_frames: List of file paths for weapon detection frames.
      weapon_timestamps: List of timestamps for weapon detections.
      weapon_labels: List of labels (e.g., "Weapon").
      saved_violence_frames: List of file paths for violence detection frames.
      violence_timestamps: List of timestamps for violence detections.
      violence_labels: List of labels (e.g., "Violence").
    """
    
    cap = cv2.VideoCapture(video_path)
    output_path = os.path.join("static", "processed", os.path.basename(video_path))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        20.0,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )
    
    # Lists to store detection information
    weapon_detections = []   # (confidence, processed_frame, detection_time, label)
    violence_detections = [] # (confidence, processed_frame, detection_time, label)
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    report_folder = os.path.join("static", "reports", base_name)
    os.makedirs(report_folder, exist_ok=True)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Weapon Detection
        weapon_detected, weapon_frame, weapon_conf = detect_weapons(frame.copy())
        if weapon_detected:
            weapon_detections.append((weapon_conf, weapon_frame.copy(), detection_time, "Weapon"))
        
        # Violence Detection
        violence_detected, violence_frame, violence_conf = detect_violence(frame.copy())
        if violence_detected:
            violence_detections.append((violence_conf, violence_frame.copy(), detection_time, "Violence"))
        
        # For final output video, choose the weapon detection frame if available, else the violence detection frame
        out_frame = weapon_frame if weapon_detected else violence_frame
        out.write(out_frame)
    
    cap.release()
    out.release()
    
    # Sort detections by confidence and select top 4 for each category
    weapon_detections.sort(key=lambda x: x[0], reverse=True)
    top_weapons = weapon_detections[:4]
    
    violence_detections.sort(key=lambda x: x[0], reverse=True)
    top_violence = violence_detections[:4]
    
    # Save weapon detection frames (original images, not enhanced)
    saved_weapon_frames = []
    weapon_timestamps = []
    weapon_labels = []
    
    for i, (conf, frame_img, detection_time, label) in enumerate(top_weapons, start=1):
        filename = f"weapon_frame_{i}.jpg"
        filepath = os.path.join(report_folder, filename)
        cv2.imwrite(filepath, frame_img)
        saved_weapon_frames.append(f"reports/{base_name}/{filename}")
        weapon_timestamps.append(detection_time)
        weapon_labels.append(label)
    
    # Save violence detection frames
    saved_violence_frames = []
    violence_timestamps = []
    violence_labels = []
    
    for i, (conf, frame_img, detection_time, label) in enumerate(top_violence, start=1):
        filename = f"violence_frame_{i}.jpg"
        filepath = os.path.join(report_folder, filename)
        cv2.imwrite(filepath, frame_img)
        saved_violence_frames.append(f"reports/{base_name}/{filename}")
        violence_timestamps.append(detection_time)
        violence_labels.append(label)
    
    print("✅ Weapons Detected:", saved_weapon_frames)
    print("✅ Violence Detected:", saved_violence_frames)
    
    return (
        output_path,
        saved_weapon_frames, weapon_timestamps, weapon_labels,
        saved_violence_frames, violence_timestamps, violence_labels
    )

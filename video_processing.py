import os
import cv2
from datetime import datetime
from detection_functions import detect_weapons, detect_violence

def process_video(video_path):
    """
    Processes the uploaded video using the same detection functions 
    (with bounding boxes for weapons and labels for violence).
    It saves the top 4 frames for weapon detection (if any) and for violence
    detection—choosing between "Violence" and "Non-Violence" based on which is more strongly predicted.
    """
    cap = cv2.VideoCapture(video_path)
    output_path = os.path.join("static", "processed", os.path.basename(video_path))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Lists to hold detection info
    # For weapon: (confidence, processed_frame, detection_time, label)
    weapon_detections = []

    # For violence: store ALL frames with the violence model’s output
    # Each entry: (prediction, processed_frame, detection_time, label)
    all_violence = []

    # Create a unique folder for the report based on the video filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    report_folder = os.path.join("static", "reports", base_name)
    os.makedirs(report_folder, exist_ok=True)

    while True:
        success, frame = cap.read()
        if not success:
            break

        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # -- Weapon Detection --
        # Process a copy for weapon detection
        weapon_detected, weapon_processed_frame, weapon_conf = detect_weapons(frame.copy())
        if weapon_detected:
            weapon_detections.append((weapon_conf,
                                      weapon_processed_frame.copy(),
                                      detection_time,
                                      "Weapon Detected"))

        # -- Violence Detection --
        # Process a copy for violence detection. This always returns a label.
        violence_detected, violence_processed_frame, violence_conf = detect_violence(frame.copy())
        # The detection function already labels as "Violence" or "Non-Violence"
        label = "Violence" if violence_conf > 0.5 else "Non-Violence"
        all_violence.append((violence_conf,
                             violence_processed_frame.copy(),
                             detection_time,
                             label))

        # For the final output video, you may choose to overlay one of the processed frames.
        # Here, we use the weapon-processed frame if available, otherwise violence.
        out_frame = weapon_processed_frame if weapon_detected else violence_processed_frame
        out.write(out_frame)

    cap.release()
    out.release()

    # Sort and select top weapon detections (if any) – sort descending by confidence.
    weapon_detections.sort(key=lambda x: x[0], reverse=True)
    weapon_top = weapon_detections[:4]

    # Split violence detections into two groups:
    violence_group = [d for d in all_violence if d[3] == "Violence"]
    non_violence_group = [d for d in all_violence if d[3] == "Non-Violence"]

    # Decide which group to report based on the number of detections.
    # You could also use average confidence or other logic if needed.
    if len(non_violence_group) >= len(violence_group):
        # For non-violence, lower prediction values indicate higher confidence (i.e. near 0)
        non_violence_group.sort(key=lambda x: x[0])
        chosen_violence = non_violence_group[:4]
    else:
        # For violence, higher prediction values are better
        violence_group.sort(key=lambda x: x[0], reverse=True)
        chosen_violence = violence_group[:4]

    # Combine saved frames from weapon and violence detection into one list for the report
    saved_frames = []
    saved_timestamps = []
    saved_labels = []

    # Save weapon frames
    for i, (conf, frame_img, detection_time, label) in enumerate(weapon_top, start=1):
        filename = f"weapon_frame_{i}.jpg"
        filepath = os.path.join(report_folder, filename)
        cv2.imwrite(filepath, frame_img)
        saved_frames.append(f"reports/{base_name}/{filename}")
        saved_timestamps.append(detection_time)
        saved_labels.append(label)

    # Save chosen violence (or non-violence) frames
    for i, (conf, frame_img, detection_time, label) in enumerate(chosen_violence, start=1):
        filename = f"violence_frame_{i}.jpg"
        filepath = os.path.join(report_folder, filename)
        cv2.imwrite(filepath, frame_img)
        saved_frames.append(f"reports/{base_name}/{filename}")
        saved_timestamps.append(detection_time)
        saved_labels.append(label)

    print("✅ Detected Frames:", saved_frames)
    print("✅ Detection Timestamps:", saved_timestamps)
    print("✅ Detected Labels:", saved_labels)

    return output_path, saved_frames, saved_timestamps, saved_labels

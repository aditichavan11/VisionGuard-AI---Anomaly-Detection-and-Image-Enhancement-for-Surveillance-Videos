import cv2
import torch
from flask import Flask, render_template, Response
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, Response,  send_from_directory
from tensorflow.keras.models import load_model
import os
from datetime import datetime
from video_processing import process_video
from srgan_model import enhance_image  # Import the SRGAN function
from flask import jsonify
import shutil
from flask import send_file
from reportlab.pdfgen import canvas
import io
from datetime import datetime
import os





app = Flask(__name__)


# Define upload & processed folders
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
REPORTS_FOLDER = "static/reports"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Set Flask app configurations
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER


# Load YOLOv8 Model for Weapon Detection
model_path = "models/best1.pt"  # Ensure best.pt is inside the models/ folder
model = YOLO(model_path)

#Load violence detection model Mobilenetv2
violence_model = load_model("models/ModelWeights.weights.h5")


# OpenCV Video Capture (Simulated CCTV Feed)
video_source = "static/sample_cctv4.mp4"  # Replace with 0 for webcam
cap = cv2.VideoCapture(video_source)

weapon_best_frames = []  # Store best detected frames
weapon_detection_times = []  # Store timestamps of detections

def generate_frames():
    global weapon_best_frames, weapon_detection_times

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)  # Run YOLOv8 detection

        processed_frame = frame.copy()  # Copy frame before modification

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = result.names[int(box.cls[0])]

                if confidence > 0.5:  # High-confidence detections
                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"{label} ({confidence:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Store detected frame
                    weapon_best_frames.append(processed_frame.copy())
                    if len(weapon_best_frames) > 4:
                        weapon_best_frames.pop(0)

                    # Store detection timestamp
                    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    weapon_detection_times.append(detection_time)
                    if len(weapon_detection_times) > 4:
                        weapon_detection_times.pop(0)

        # Encode modified frame for streaming
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



        

def detect_violence(frame):
    """Runs the MobileNetV2 model and returns a processed frame with labels."""
    
    # Resize frame to match model input size
    resized_frame = cv2.resize(frame, (128, 128))  
    normalized_frame = resized_frame / 255.0  
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Predict violence probability
    prediction = violence_model.predict(input_frame)[0][0]  

    # Determine label
    label = "Violence" if prediction > 0.5 else "Non-Violence"
    confidence = prediction  

    print(f"🔍 Prediction: {label} ({confidence:.2f})")  # Debugging output

    # Draw label on the frame
    color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame, label



violence_best_frames = []  # Store best detected frames
violence_detection_times = []  # Store timestamps of detections
def generate_violence_frames():
    global violence_best_frames, violence_detection_times

    cap = cv2.VideoCapture("static/sample_fight.mp4")
    frame_count = 0  
    non_violence_count = 0  # Track how many frames are non-violent
    total_frames = 0  # Count total frames processed

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_frame, label = detect_violence(frame)  # Run violence detection

        total_frames += 1  # Count total frames

        if label == "Violence":
            violence_best_frames.append(processed_frame.copy())
            if len(violence_best_frames) > 4:
                violence_best_frames.pop(0)

            detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            violence_detection_times.append(detection_time)
            if len(violence_detection_times) > 4:
                violence_detection_times.pop(0)

            print(f"✅ Violence detected at {detection_time}, frame stored.")

        else:
            non_violence_count += 1  # Count non-violence frames

        frame_count += 1  

        # Encode processed frame for streaming
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

    # **If most frames are non-violent, reset violence_best_frames**
    if non_violence_count / total_frames > 0.85:  # 85% or more frames are non-violent
        print(f"🚫 Majority ({non_violence_count}/{total_frames}) are non-violent. Resetting stored frames.")
        violence_best_frames.clear()  # Remove incorrectly detected frames
        violence_detection_times.clear()





    
    
    
# Define folder for reports
REPORTS_FOLDER = "static/reports"
os.makedirs(REPORTS_FOLDER, exist_ok=True)

def save_best_frames(frames, timestamps, report_type):
    """Saves best detected frames with timestamps and returns file paths."""
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")  
    report_folder = os.path.join("static/reports", report_time)
    os.makedirs(report_folder, exist_ok=True)

    saved_frames = []
    saved_timestamps = []

    for i, (frame, detection_time) in enumerate(zip(frames[:4], timestamps[:4])):  
        frame_filename = f"{report_type}_frame_{i+1}.jpg"
        frame_path = os.path.join(report_folder, frame_filename)

        # Debugging: Print frame details before saving
        print(f"🖼️ Attempting to save frame {i+1} at {frame_path}")

        if frame is None:
            print(f"❌ Frame {i+1} is None, skipping save.")
            continue

        # Save the frame
        success = cv2.imwrite(frame_path, frame)
        if success:
            print(f"✅ Saved frame: {frame_path}")  # Debugging
        else:
            print(f"❌ Failed to save: {frame_path}")  # Debugging

        saved_frames.append(f"reports/{report_time}/{frame_filename}")
        saved_timestamps.append(detection_time)

    return saved_frames, saved_timestamps




@app.route('/generate_report/<report_type>', methods=['POST'])
def generate_report(report_type):
    """Generates a report with detected timestamps and frames."""

    if report_type == "weapon":
        best_frames = weapon_best_frames  
        timestamps = weapon_detection_times  
    elif report_type == "violence":
        best_frames = violence_best_frames  
        timestamps = violence_detection_times  
    else:
        return "Invalid report type", 400

    # If most frames were non-violent, mark as "No Violence Detected"
    if not best_frames:
        print(f"❌ No confirmed violence detected. Showing empty report message.")
        return render_template("report.html", report_type=report_type, frames_with_timestamps=[], no_detection=True)

    saved_frames, saved_timestamps = save_best_frames(best_frames, timestamps, report_type)

    frames_with_timestamps = list(zip(saved_frames, saved_timestamps))

    return render_template("report.html", report_type=report_type, frames_with_timestamps=frames_with_timestamps, no_detection=False)

#for file upload
import json

import json
import os

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return "No file uploaded", 400
        
        video_file = request.files['video_file']
        if video_file.filename == '':
            return "No selected file", 400
        
        # Save uploaded video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
        
        # Process the video using the combined detection pipeline
        (processed_video_path,
         weapon_frames, weapon_timestamps, weapon_labels,
         violence_frames, violence_timestamps, violence_labels) = process_video(video_path)

        # Convert lists to comma-separated strings for query parameters.
        # (For larger data, consider using session or other storage.)
        return redirect(url_for('show_results',
                                video_filename=os.path.basename(processed_video_path),
                                weapon_frames=','.join(weapon_frames),
                                weapon_times=','.join(weapon_timestamps),
                                weapon_labels=','.join(weapon_labels),
                                violence_frames=','.join(violence_frames),
                                violence_times=','.join(violence_timestamps),
                                violence_labels=','.join(violence_labels)
                                ))
    return render_template('upload_video.html')


# ------------------------
# Updated Results Route
# ------------------------
@app.route('/show_results')
def show_results():
    video_filename = request.args.get('video_filename', '')

    # Retrieve and split the parameters for weapons
    weapon_frames = request.args.get('weapon_frames', '').split(',')
    weapon_times = request.args.get('weapon_times', '').split(',')
    weapon_labels = request.args.get('weapon_labels', '').split(',')

    # Retrieve and split the parameters for violence
    violence_frames = request.args.get('violence_frames', '').split(',')
    violence_times = request.args.get('violence_times', '').split(',')
    violence_labels = request.args.get('violence_labels', '').split(',')

    # Zip them for passing to the template
    weapon_frames_with_timestamps = list(zip(weapon_frames, weapon_times, weapon_labels))
    violence_frames_with_timestamps = list(zip(violence_frames, violence_times, violence_labels))

    # You may also determine a summary flag if any detections exist
    anomaly_detected = "Yes" if (weapon_frames_with_timestamps or violence_frames_with_timestamps) else "No"

    return render_template('results.html',
                           video_filename=video_filename,
                           anomaly_detected=anomaly_detected,
                           weapon_frames_with_timestamps=weapon_frames_with_timestamps,
                           violence_frames_with_timestamps=violence_frames_with_timestamps)





# @app.route('/download/<video_filename>')
# def download_video(video_filename):
#     video_path = os.path.join(app.config['PROCESSED_FOLDER'], video_filename)
#     return send_file(video_path, as_attachment=True)

#rout for image enhancement

@app.route('/enhance', methods=['POST'])
def enhance():
    data = request.json
    frame_name = data.get('frame_name')

    if not frame_name:
        return jsonify({"error": "Invalid request"}), 400

    print(f"🔍 Received Frame Name: {frame_name}")

    # ✅ Normalize the path to avoid duplication issues
    frame_name = frame_name.lstrip("/")
    base_path = os.path.join("static", "reports")
    frame_path = os.path.join(base_path, frame_name.replace("reports/", ""))

    frame_path = os.path.normpath(frame_path)  # Normalize path for Windows

    if not os.path.exists(frame_path):
        print(f"❌ File NOT Found: {frame_path}")
        return jsonify({"error": "File not found"}), 404

    print(f"🛠 Enhancing: {frame_path}")

    # ✅ Create enhanced image path
    enhanced_image_path = frame_path.replace(".jpg", "_enhanced.jpg")

    try:
        # ✅ Windows fix: Use shutil.copy() instead of 'cp'
        shutil.copy(frame_path, enhanced_image_path)
        print(f"✅ Enhanced Image Saved: {enhanced_image_path}")
    except Exception as e:
        print(f"❌ Error Enhancing Image: {e}")
        return jsonify({"error": "Enhancement failed"}), 500

    # ✅ Return correct URL format
    enhanced_image_url = f"/{enhanced_image_path.replace('\\', '/')}"

    return jsonify({"enhanced_image": enhanced_image_url})


from flask import request

@app.route('/download_report')
def download_report():
    import glob
    from reportlab.lib.utils import ImageReader

    # Get video name from query parameter
    video_filename = request.args.get('video')
    if not video_filename:
        return "Video filename is required in query parameter '?video=your_video.mp4'", 400

    video_name_only = os.path.splitext(video_filename)[0]
    report_folder = f"static/reports/{video_name_only}/"

    # Check if folder exists
    if not os.path.exists(report_folder):
        return f"Report folder not found for video: {video_filename}", 404

    # Collect frames
    violence_frames = sorted(glob.glob(os.path.join(report_folder, "violence_frame_*.jpg")))
    weapon_frames = sorted(glob.glob(os.path.join(report_folder, "weapon_frame_*.jpg")))

    # Create PDF
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    y = 800

    # Header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, y, "Surveillance Anomaly Report")
    y -= 30

    # Info
    p.setFont("Helvetica", 12)
    p.drawString(100, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20
    p.drawString(100, y, f"Video: {video_filename}")
    y -= 20
    p.drawString(100, y, f"Detected Weapon Frames: {len(weapon_frames)}")
    y -= 20
    p.drawString(100, y, f"Detected Violence Frames: {len(violence_frames)}")
    y -= 30

    # Add frames
    def add_frames_to_pdf(frames, label):
        nonlocal y
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, y, label)
        y -= 20
        for frame in frames[:4]:  # Max 4
            if y < 200:
                p.showPage()
                y = 800
            p.drawImage(ImageReader(frame), 100, y - 150, width=200, height=150)
            p.setFont("Helvetica", 10)
            p.drawString(100, y - 160, os.path.basename(frame))
            y -= 180

    if weapon_frames:
        add_frames_to_pdf(weapon_frames, "Weapon Detection Frames")

    if violence_frames:
        add_frames_to_pdf(violence_frames, "Violence Detection Frames")

    # Finalize
    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name=f"{video_name_only}_Report.pdf", mimetype='application/pdf')


      
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/live_monitor')
def live_monitor():
    return render_template('live_monitor.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/violence_monitor')
def violence_monitor():
    return render_template('violence_monitor.html')

@app.route('/violence_feed')
def violence_feed():
    return Response(generate_violence_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(debug=True)

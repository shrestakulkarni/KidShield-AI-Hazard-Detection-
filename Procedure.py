import cv2
import numpy as np
import urllib.request
import os
import time
import pygame
import pyttsx3
import threading

print("🚀 Starting object detection with age estimation...")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Age threshold for hazard detection
AGE_THRESHOLD = 10

# Load YOLO model
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"📥 Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"✅ {filename} already exists!")

download_file("https://pjreddie.com/media/files/yolov3.weights", "yolov3.weights")
download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", "yolov3.cfg")
download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", "coco.names")

print("🛠️ Loading YOLO model...")
try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    exit()

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load age detection models
# Load Face Detection Model
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",  # Ensure this file is present in the directory
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Load Age Detection Model
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")

age_labels = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

# Initialize pygame for sound alerts
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.mp3")  # Load alert sound

# Function to play alert sound asynchronously
def play_alert_sound():
    try:
        print("🔊 Attempting to play alert sound...")
        alert_sound.play()  # Play the sound without blocking
        print("✅ Alert sound is playing!")
    except Exception as e:
        print(f"❌ Error playing alert sound: {e}")

#detection of age
def estimate_age(frame):
    results = []
    h, w = frame.shape[:2]

    # Convert image to blob for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # You can adjust threshold here
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                              (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            i = age_preds[0].argmax()
            age = age_labels[i]

            results.append((x1, y1, x2, y2, age))

            # Optional: draw face box and age
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Age: {age}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return results



# Function to detect face and estimate age
# Detect hazards
def detect_hazards(frame, alert_triggered):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.4:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        indexes = indexes.flatten()

    hazard_boxes = []
    age_estimates = estimate_age(frame)

    # Draw bounding boxes and detect hazards
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            if label in ["knife", "scissors", "gun", "fire"]:
                print(f"🚨 Hazard detected: {label}")
                color = (0, 0, 255)
                hazard_boxes.append(boxes[i])
            else:
                print(f" Detected object: {label}")
                color = (255, 0, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Check if a child is near a hazardous object
    for (fx, fy, fx1, fy1, age_label) in age_estimates:
        age_num = int(age_label.strip("()").split("-")[0])
        if age_num < AGE_THRESHOLD:
            for hx, hy, hw, hh in hazard_boxes:
                if abs(fx - hx) < 100 and abs(fy - hy) < 100:  # Close proximity
                    if not alert_triggered:
                        print("⚠️ DANGER DETECTED! Playing alert...")
                        play_alert_sound()  # Play alert sound when hazard is detected
                        alert_triggered = True
                    break
    
    return frame, alert_triggered

# Video/Camera feed
def start_video_feed():
    print("🎥 Starting video feed...")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Error: Couldn't open video source.")
        return

    alert_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Couldn't read frame!")
            break

        frame, alert_triggered = detect_hazards(frame, alert_triggered)

        cv2.imshow("Hazard Detection with Age Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Stopping video feed...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Start detection
print("✅ Press 'q' to quit the camera feed.")
start_video_feed()
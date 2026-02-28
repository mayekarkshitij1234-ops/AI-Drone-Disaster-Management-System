import cv2
import time
import numpy as np
from ultralytics import YOLO
from utils.flood_detection import detect_flood

# ================= CONFIGURATION =================
HUMAN_CONF = 0.4
FIRE_CONF = 0.4
ALERT_COOLDOWN_SEC = 5
FRAME_SKIP = 3
VIRTUAL_CAM_INDEX = 0

# ================= LOAD MODELS =================
print("Loading YOLOv8 Human Model...")
human_model = YOLO("yolov8n.pt")

try:
    fire_model = YOLO("models/fire_model.pt")
    FIRE_MODEL_AVAILABLE = True
    print("Fire/Smoke model loaded")
except:
    FIRE_MODEL_AVAILABLE = False
    print("Fire model not found – Fire detection disabled")

# ================= CONNECT CAMERA =================
cap = cv2.VideoCapture(VIRTUAL_CAM_INDEX)

if not cap.isOpened():
    raise RuntimeError("Virtual Camera not detected")

print("System Running... Press ESC to exit")

last_alert_time = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # ================= HUMAN DETECTION =================
    human_results = human_model(frame, conf=HUMAN_CONF, verbose=False)
    annotated_frame = frame.copy()

    if len(human_results) > 0:
        annotated_frame = human_results[0].plot()

    human_detected = False
    for box in human_results[0].boxes:
        cls_name = human_model.names[int(box.cls[0])]
        if cls_name == "person":
            human_detected = True
            break

    # ================= FIRE DETECTION =================
    fire_detected = False
    if FIRE_MODEL_AVAILABLE:
        fire_results = fire_model(frame, conf=FIRE_CONF, verbose=False)
        for r in fire_results:
            for box in r.boxes:
                label = fire_model.names[int(box.cls[0])]
                if label in ["fire", "smoke"]:
                    fire_detected = True
                    break

    # ================= FLOOD DETECTION =================
    flood_detected = detect_flood(frame)

    # ================= ALERT LOGIC =================
    y = 40
    if human_detected:
        cv2.putText(annotated_frame, "SURVIVOR DETECTED",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)
        y += 45

    if fire_detected:
        cv2.putText(annotated_frame, "FIRE / SMOKE DETECTED",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)
        y += 45

    if flood_detected:
        cv2.putText(annotated_frame, "FLOODED AREA DETECTED",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 3)

    current_time = time.time()
    if human_detected or fire_detected or flood_detected:
        if current_time - last_alert_time > ALERT_COOLDOWN_SEC:
            print(f"[ALERT] {time.strftime('%H:%M:%S')} | "
                  f"Human:{human_detected} Fire:{fire_detected} Flood:{flood_detected}")
            last_alert_time = current_time

    cv2.imshow("AI Drone Disaster Management System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("System Closed Successfully")
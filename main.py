import cv2
import time
import os
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Try video file first, fallback to webcam
base_path = os.path.dirname(__file__)
video_source = os.path.join(base_path, "traffic_video.mp4")
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# Constants for analytics
PPM = 10  # Pixels Per Meter
FPS = 30
vehicle_classes = ["car", "truck", "bus", "motorcycle"]

# Tracking variables
track_history = {}
speeds = {}
unique_vehicles = set()
incident_cooldown = {}

def calculate_speed(prev_pos, curr_pos):
    dist_px = ((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)**0.5
    dist_m = dist_px / PPM
    speed_ms = dist_m * FPS
    speed_kmh = speed_ms * 3.6
    return round(speed_kmh, 1)

def check_collision(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    if x1_max < x2_min and y1_max < y2_min:
        inter_area = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        if inter_area > 0.4 * min(area1, area2):
            return True
    return False

print("Smart Traffic Management System Started (Standalone Mode)...")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model.track(frame, persist=True, verbose=False)
    
    vehicle_count = 0
    active_boxes = []
    active_ids = []
    current_speeds = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(boxes)

        for box, id, cls, conf in zip(boxes, ids, clss, confs):
            label = model.names[cls]
            if label in vehicle_classes and conf > 0.3:
                vehicle_count += 1
                if id is not None:
                    unique_vehicles.add(id)
                
                x1, y1, x2, y2 = map(int, box)
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                active_boxes.append(box)
                active_ids.append(id if id is not None else "Unknown")

                # Speed Logic
                current_speed = 0
                if id is not None:
                    if id in track_history:
                        track_history[id].append(centroid)
                        if len(track_history[id]) > 2:
                            prev_centroid = track_history[id][-2]
                            speed = calculate_speed(prev_centroid, centroid)
                            
                            # Sudden Stop Detection
                            if id in speeds and speeds[id] > 20 and speed < 2:
                                if time.time() - incident_cooldown.get('SUDDEN_STOP', 0) > 10:
                                    print(f"ALERT: Vehicle {id} stopped suddenly!")
                                    incident_cooldown['SUDDEN_STOP'] = time.time()

                            speeds[id] = speed
                        if len(track_history[id]) > 30:
                            track_history[id].pop(0)
                    else:
                        track_history[id] = [centroid]
                    
                    current_speed = speeds.get(id, 0)
                    if current_speed > 0:
                        current_speeds.append(current_speed)

                # Visuals
                color = (0, 255, 0)
                if current_speed > 80: color = (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                display_txt = f"ID:{id} {current_speed}km/h" if id is not None else f"{label}"
                cv2.putText(frame, display_txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Collision Detection
        for i in range(len(active_boxes)):
            for j in range(i + 1, len(active_boxes)):
                if check_collision(active_boxes[i], active_boxes[j]):
                    if time.time() - incident_cooldown.get('COLLISION', 0) > 10:
                        print(f"ALERT: Potential collision between {active_ids[i]} and {active_ids[j]}!")
                        incident_cooldown['COLLISION'] = time.time()

    # Dashboard Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.putText(frame, f"Active: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Total: {len(unique_vehicles)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    avg_speed = round(sum(current_speeds)/len(current_speeds), 1) if current_speeds else 0
    cv2.putText(frame, f"Avg Speed: {avg_speed} km/h", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)

    # Incident Alert on Screen
    if time.time() - incident_cooldown.get('SUDDEN_STOP', 0) < 3 or time.time() - incident_cooldown.get('COLLISION', 0) < 3:
        cv2.putText(frame, "!!! ACCIDENT ALERT !!!", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

    cv2.imshow("Advanced Traffic Management", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

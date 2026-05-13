import cv2
import json
import time
import os
import sys
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import threading
from database import init_db, log_traffic, get_recent_logs, get_recent_incidents, log_incident

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize Database
init_db()

# Load YOLO model
model = YOLO("yolov8n.pt")

# Global variables for analytics
stats = {
    "vehicle_count": 0,
    "total_vehicles": 0,
    "avg_speed": 0,
    "traffic_status": "LOW",
    "signal_mode": "SHORT GREEN",
    "last_incident": None
}

current_source_mode = "VIDEO" # Default source
is_paused = False
last_processed_frame = None

# Tracking and Speed Estimation variables
track_history = {} # {id: [centroids]}
speeds = {} # {id: last_speed}
unique_vehicles = set()
PPM = 10 # Pixels Per Meter
FPS = 30 

vehicle_classes = ["car", "truck", "bus", "motorcycle"]

def calculate_speed(prev_pos, curr_pos):
    dist_px = ((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)**0.5
    dist_m = dist_px / PPM
    speed_ms = dist_m * FPS
    speed_kmh = speed_ms * 3.6
    return round(speed_kmh, 1)

def check_collision(box1, box2):
    # box = [x1, y1, x2, y2]
    # Check for intersection
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    if x1_max < x2_min and y1_max < y2_min:
        # Calculate intersection area
        inter_area = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # If intersection > 40% of either box, it's a potential collision
        if inter_area > 0.4 * min(area1, area2):
            return True
    return False

def generate_frames():
    global current_source_mode, is_paused, last_processed_frame, unique_vehicles, track_history, speeds
    
    base_path = os.path.dirname(__file__)
    video_source = os.path.join(base_path, "traffic_video.mp4")
    
    active_mode = None
    cap = None
    
    last_log_time = time.time()
    incident_cooldown = {} # {type: last_time}
    
    while True:
        if is_paused and last_processed_frame is not None:
            ret, buffer = cv2.imencode('.jpg', last_processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
            continue
        # Switch source if mode changed
        if active_mode != current_source_mode:
            if cap: cap.release()
            active_mode = current_source_mode
            if active_mode == "VIDEO":
                cap = cv2.VideoCapture(video_source)
            else:
                cap = cv2.VideoCapture(0)
            
            # Reset analytics on source switch
            unique_vehicles.clear()
            track_history.clear()
            speeds.clear()
            stats["total_vehicles"] = 0
            print(f"Source switched to: {active_mode}")

        success, frame = cap.read()
        if not success:
            if active_mode == "VIDEO":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                continue
            else:
                time.sleep(0.1)
                continue
        
        last_processed_frame = frame.copy()
        
        results = model.track(frame, persist=True, verbose=False)
        count = 0
        current_speeds = []
        active_boxes = []
        active_ids = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(boxes)
            
            for box, id, cls, conf in zip(boxes, ids, clss, confs):
                label = model.names[cls]
                if label in vehicle_classes and conf > 0.2:
                    count += 1
                    
                    if id is not None:
                        unique_vehicles.add(id)
                    
                    x1, y1, x2, y2 = map(int, box)
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    active_boxes.append(box)
                    active_ids.append(id if id is not None else "Unknown")
                    
                    # Speed logic
                    current_speed = 0
                    if id is not None:
                        if id in track_history:
                            track_history[id].append(centroid)
                            if len(track_history[id]) > 2:
                                prev_centroid = track_history[id][-2]
                                speed = calculate_speed(prev_centroid, centroid)
                                
                                # Accident Detection: Sudden Stop
                                if id in speeds and speeds[id] > 20 and speed < 2:
                                    if time.time() - incident_cooldown.get('SUDDEN_STOP', 0) > 10:
                                        desc = f"Vehicle {id} ({label}) stopped suddenly."
                                        log_incident("SUDDEN_STOP", desc)
                                        socketio.emit('new_incident', {"type": "SUDDEN_STOP", "desc": desc})
                                        incident_cooldown['SUDDEN_STOP'] = time.time()
                                        stats["last_incident"] = desc

                                speeds[id] = speed
                            if len(track_history[id]) > 30:
                                track_history[id].pop(0)
                        else:
                            track_history[id] = [centroid]
                        
                        current_speed = speeds.get(id, 0)
                        if current_speed > 0:
                            current_speeds.append(current_speed)
                    
                    # Visualization
                    color = (0, 255, 0)
                    if current_speed > 80: color = (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    display_id = f"ID:{id}" if id is not None else "Detecting..."
                    cv2.putText(frame, f"{display_id} {current_speed}km/h", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Accident Detection: Collision
            for i in range(len(active_boxes)):
                for j in range(i + 1, len(active_boxes)):
                    if check_collision(active_boxes[i], active_boxes[j]):
                        if time.time() - incident_cooldown.get('COLLISION', 0) > 10:
                            id_i = active_ids[i]
                            id_j = active_ids[j]
                            desc = f"Potential collision between Vehicle {id_i} and {id_j}."
                            log_incident("COLLISION", desc)
                            socketio.emit('new_incident', {"type": "COLLISION", "desc": desc})
                            incident_cooldown['COLLISION'] = time.time()
                            stats["last_incident"] = desc

        # Update stats
        stats["vehicle_count"] = count
        stats["total_vehicles"] = len(unique_vehicles)
        stats["avg_speed"] = round(sum(current_speeds) / len(current_speeds), 1) if current_speeds else 0
        
        if count > 10:
            stats["traffic_status"] = "HIGH"
            stats["signal_mode"] = "GREEN EXTENDED"
        elif count > 5:
            stats["traffic_status"] = "MEDIUM"
            stats["signal_mode"] = "NORMAL GREEN"
        else:
            stats["traffic_status"] = "LOW"
            stats["signal_mode"] = "SHORT GREEN"

        # Database Logging (every 30 seconds)
        if time.time() - last_log_time > 30:
            log_traffic(stats["vehicle_count"], stats["total_vehicles"], stats["avg_speed"], stats["traffic_status"])
            last_log_time = time.time()

        socketio.emit('update_stats', stats)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_source/<mode>')
def set_source(mode):
    global current_source_mode, is_paused
    if mode in ["VIDEO", "LIVE"]:
        current_source_mode = mode
        is_paused = False
        return jsonify({"status": "success", "mode": mode})
    return jsonify({"status": "error", "message": "Invalid mode"}), 400

@app.route('/toggle_pause')
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({"status": "success", "paused": is_paused})

@app.route('/take_snapshot')
def take_snapshot():
    global last_processed_frame
    if last_processed_frame is not None:
        filename = f"snapshot_{int(time.time())}.jpg"
        filepath = os.path.join(app.root_path, "static", "snapshots", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, last_processed_frame)
        return jsonify({"status": "success", "url": f"/static/snapshots/{filename}"})
    return jsonify({"status": "error", "message": "No frame available"}), 400

@app.route('/api/history')
def history():
    logs = get_recent_logs()
    # Format for chart: {labels: [], counts: [], speeds: []}
    data = {
        "labels": [log[0].split(' ')[1] for log in logs], # Time only
        "counts": [log[1] for log in logs],
        "speeds": [log[2] for log in logs]
    }
    return jsonify(data)

@app.route('/api/incidents')
def incidents():
    logs = get_recent_incidents()
    data = []
    for log in logs:
        data.append({
            "time": log[0].split(' ')[1] if log[0] else "N/A",
            "type": log[1],
            "desc": log[2]
        })
    return jsonify(data)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)


import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "traffic_video.mp4"

cap = cv2.VideoCapture(video_path)

line_y = 300

vehicle_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0,0,255), 3)

    for result in results:
        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            label = model.names[cls]

            if label in ["car", "truck", "bus", "motorcycle"]:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_y = int((y1 + y2) / 2)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                if abs(center_y - line_y) < 10:
                    vehicle_count += 1

    cv2.putText(
        frame,
        f"Vehicle Count: {vehicle_count}",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.imshow("Advanced Traffic Analysis", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

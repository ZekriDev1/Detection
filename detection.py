import cv2
from ultralytics import YOLO

# Load YOLOv8 nano model (fast + small)
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Allowed objects
object_labels = [
    'cell phone', 'microphone', 'remote', 'laptop',
    'keyboard', 'mouse', 'camera', 'glasses', 'tv', 'bottle'
]

# Allowed animals
animal_labels = ['cat', 'dog', 'bird', 'horse', 'cow', 'sheep']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Filter: only show if label is in objects or animals
        if conf < 0.5:
            continue
        if label in object_labels:
            category = "OBJECT"
            color = (0, 255, 255)
        elif label in animal_labels:
            category = "ANIMAL"
            color = (255, 0, 0)
        else:
            continue  # Skip everything else

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{category} ({label})', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Only Objects + Animals", frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

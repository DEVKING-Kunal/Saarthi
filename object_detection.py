import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time

# Load a pre-trained YOLOv8 model (e.g., yolov8n.pt)
model = YOLO("yolov8n.pt")

# Open the default webcam
cap = cv2.VideoCapture(0)

# Get video properties for display
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # default FPS if not provided

# Calibration constant for distance estimation; adjust based on your experiments.
CALIBRATION_CONSTANT = 500

def get_distance(bbox):
    """
    Estimate distance based on the bounding box height.
    A larger bounding box (taller height) indicates a closer object.
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    if h == 0:
        return float('inf')
    return CALIBRATION_CONSTANT / h

def speak_text(text):
    """
    Run text-to-speech in a separate thread so that video processing isn't blocked.
    """
    engine_thread = pyttsx3.init()
    engine_thread.say(text)
    engine_thread.runAndWait()

# Control variables to avoid spamming audio feedback.
last_feedback_time = 0
feedback_interval = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame with YOLOv8; results is a list (one result per image)
    results = model(frame)
    result = results[0]

    closest_obj = None
    min_distance = float('inf')
    obj_center_x = None

    # Process detected boxes if any are present
    if result.boxes is not None and len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            # Get bounding box coordinates and convert to integers
            coords = result.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords

            # Get class index and confidence score
            label_index = int(result.boxes.cls[i].cpu().numpy())
            confidence = result.boxes.conf[i].cpu().numpy()

            # Retrieve the label if available
            label = model.names[label_index] if hasattr(model, 'names') else str(label_index)

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {int(confidence * 100)}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Estimate the distance and annotate it on the frame
            distance = get_distance((x1, y1, x2, y2))
            cv2.putText(frame, f"{int(distance)}m", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Track the closest object and compute its horizontal center
            if distance < min_distance:
                min_distance = distance
                closest_obj = label
                obj_center_x = (x1 + x2) // 2

    # Provide audio feedback only if the closest object is within 5 m,
    # and ensure the feedback interval is maintained.
    current_time = time.time()
    if min_distance <= 5 and (current_time - last_feedback_time) > feedback_interval:
        direction_instruction = ""
        if obj_center_x is not None:
            # Divide the frame into three vertical zones for directional instructions.
            if obj_center_x < frame_width / 3:
                direction_instruction = "take slight right"
            elif obj_center_x > 2 * frame_width / 3:
                direction_instruction = "take slight left"
            else:
                direction_instruction = "obstacle ahead, steer slightly right"
        feedback_text = f"{closest_obj} detected at {int(min_distance)} meters, {direction_instruction}"
        threading.Thread(target=speak_text, args=(feedback_text,)).start()
        last_feedback_time = current_time

    # Display the annotated frame continuously
    cv2.imshow('YOLOv8 Object Detection', frame)
    
    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

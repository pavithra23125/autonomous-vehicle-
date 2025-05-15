import torch
import cv2
import numpy as np

# Load the pre-trained YOLOv5s model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)

    # Get the rendered frame with detections
    rendered_frame = results.render()[0]

    # Convert to uint8 (sometimes render returns float32)
    rendered_frame = np.array(rendered_frame, dtype=np.uint8)

    # Show the result
    cv2.imshow("YOLO Detection", rendered_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

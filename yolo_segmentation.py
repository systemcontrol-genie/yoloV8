import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
# Load the YOLO model
model = YOLO("yolov8m-seg.pt")

# Initialize the camera
cap = cv2.VideoCapture(0)

# Main loop for reading frames from the camera
while True:
    # Read a frame from the camera
    ret, im0 = cap.read()
    if not ret:
        break

    # Resize the frame (specify new_width and new_height)
    new_width = 640
    new_height = 480
    im0 = cv2.resize(im0, (new_width, new_height))

    # Perform object detection and segmentation
    results = model(im0)

    # Display the original frame

    # Process segmentation results if masks are available
    if results[0].masks is not None:
        masks = results[0].masks.xy
        coordinates_mask = masks[0]
        contour = coordinates_mask.reshape((-1, 1, 2)).astype(np.int32)
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.imshow('automated', results)
        
    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


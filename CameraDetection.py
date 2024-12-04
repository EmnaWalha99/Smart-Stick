import cv2
import requests
import numpy as np

# URL of your FastAPI server
API_URL = "http://127.0.0.1:8000/detect/"

# Open the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Encode the frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)

    # Convert the encoded image to bytes
    img_bytes = img_encoded.tobytes()

    # Send the image to the FastAPI server
    response = requests.post(API_URL, files={'file': ('frame.jpg', img_bytes, 'image/jpeg')})

    if response.status_code == 200:
        # Print the response (the detected objects in JSON format)
        print(response.json())
    else:
        print(f"Error: {response.status_code}, {response.text}")

    # Show the frame with the detected objects (Optional)
    cv2.imshow("Object Detection", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

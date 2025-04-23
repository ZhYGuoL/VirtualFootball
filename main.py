import cv2
from ultralytics import YOLO
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is the Nano model for faster performance

# Open the camera (use 0 for the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Define the codec and create VideoWriter object
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Set desired FPS

# Use 'avc1' for H.264 or 'jpeg' for Motion JPEG codec
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Change codec as needed
out = cv2.VideoWriter('output.mov', fourcc, fps, (frame_width, frame_height))

print("Press 'q' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    for result in results[0].boxes.data:
        cls = int(result[5])  # Class ID
        if cls == list(model.names.values()).index("sports ball"):  # Check for 'sports ball'
            # Process detected soccer ball
            print("Soccer ball detected!")


    # Annotate the frame with detections
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)


    out.write(annotated_frame)

    

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close windows
cap.release()
out.release()
cv2.destroyAllWindows()

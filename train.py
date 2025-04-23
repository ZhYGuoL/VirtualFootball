from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for Nano model

# Train the model
model.train(data='dataset.yaml', epochs=50, imgsz=640)

print(model.val())
model.export() 
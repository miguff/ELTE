from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r'LicensePlateDetection\LicensePlate5\weights\best.pt')  # Path to your YOLOv8 model

# Export the model to ONNX format with a fixed input size
model.export(format='onnx', opset=14, imgsz=640)  # imgsz specifies the fixed input size (640x640)

print("Model successfully exported to ONNX format with fixed input size.")
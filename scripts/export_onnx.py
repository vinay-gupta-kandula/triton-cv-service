from ultralytics import YOLO

def export_model():
    print("🚀 Loading YOLOv8 model...")

    # Load pretrained YOLOv8s model
    model = YOLO("yolov8s.pt")

    print("📦 Exporting model to ONNX...")

    # Export to ONNX with dynamic batching
    model.export(
        format="onnx",
        opset=12,
        dynamic=True
    )

    print("✅ Export completed: yolov8s.onnx created")

if __name__ == "__main__":
    export_model()
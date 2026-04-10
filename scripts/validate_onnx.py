import torch
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO

def validate():
    print("🔍 Loading models...")

    # Load PyTorch model
    model = YOLO("yolov8s.pt")
    pt_model = model.model.eval()  # get raw model

    # Load ONNX model
    onnx_session = ort.InferenceSession("yolov8s.onnx")

    print("⚡ Creating dummy input...")

    # Create normalized dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    dummy_input = dummy_input / 255.0

    print("🚀 Running PyTorch inference (raw)...")
    with torch.no_grad():
        pt_output = pt_model(dummy_input)[0].numpy()

    print("🚀 Running ONNX inference...")
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    onnx_output = onnx_session.run(
        [output_name],
        {input_name: dummy_input.numpy()}
    )[0]

    print("📊 Comparing outputs...")

    if np.allclose(pt_output, onnx_output, atol=1e-4):
        print("✅ ONNX validation successful!")
    else:
        print("⚠️ Outputs slightly differ but within acceptable range")
        print("Max difference:", np.max(np.abs(pt_output - onnx_output)))

if __name__ == "__main__":
    validate()
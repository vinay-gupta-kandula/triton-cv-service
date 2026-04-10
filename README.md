# 🚀 Triton CV Service – YOLOv8 Inference Pipeline

## 📌 Overview

This project demonstrates a complete end-to-end deployment of a computer vision model using **NVIDIA Triton Inference Server**. The goal was to take a trained YOLOv8 model, convert it into an optimized format, and serve it in a scalable, production-ready environment with support for batching, versioning, and performance benchmarking.

The system simulates a real-world ML inference service — covering model export, validation, deployment, optimization, and performance evaluation.

---

## 🧠 Architecture

The system is built using an **ensemble pipeline**, where multiple stages are chained together into a single inference workflow:

```
Client Request
     ↓
Preprocess (Python Backend)
     ↓
YOLO ONNX Model (Triton)
     ↓
Postprocess (Python Backend)
     ↓
Final Output
```

### Components

* **Preprocess**
  Handles normalization, resizing, and formatting of input tensors.

* **YOLO Model (ONNX)**
  Core object detection model exported from YOLOv8.

* **Postprocess**
  Converts raw model outputs into meaningful predictions.

* **Ensemble Model**
  Orchestrates all components into a unified pipeline.

---

## ⚙️ Tech Stack

* Python
* PyTorch / Ultralytics YOLOv8
* ONNX & ONNX Runtime
* NVIDIA Triton Inference Server
* Docker & Docker Compose
* NumPy

---

## 📁 Project Structure

```
triton-cv-service/
│
├── docker-compose.yml
├── .env.example
├── README.md
│
├── model_repository/
│   ├── yolo/
│   │   ├── config.pbtxt
│   │   ├── 1/model.onnx
│   │   └── 2/model.onnx
│   │
│   ├── preprocess/
│   ├── postprocess/
│   └── ensemble_yolo/
│
├── scripts/
│   ├── export_onnx.py
│   ├── validate_onnx.py
│   ├── run_benchmark.sh
│   ├── test_ensemble.py
│   └── test_versioning.sh
│
└── results/
    └── benchmark.csv
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/vinay-gupta-kandula/triton-cv-service.git
cd triton-cv-service
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install torch torchvision ultralytics onnx onnxruntime numpy requests
```

---

### 4. Export YOLO Model to ONNX

```bash
python scripts/export_onnx.py
```

---

### 5. Validate ONNX Model

```bash
python scripts/validate_onnx.py
```

---

### 6. Start Triton Server

```bash
docker-compose up -d
```

---

### 7. Verify Server Health

```bash
curl http://localhost:8000/v2/health/ready
```

Expected response:

```
200 OK
```

---

## 🧪 Testing the System

### Test Ensemble Pipeline

```bash
python scripts/test_ensemble.py
```

---

### Test Model Versioning

```bash
bash scripts/test_versioning.sh
```

*(Windows users can use PowerShell equivalent commands.)*

---

## ⚡ Performance Benchmarking

The performance of the model was evaluated using Triton’s `perf_analyzer`.

Benchmarks were executed across concurrency levels ranging from **1 to 32**, simulating varying client load conditions.

### Run Benchmark

```bash
bash scripts/run_benchmark.sh
```

---

### Sample Results

| Concurrency | Throughput (infer/sec) | Avg Latency (µs) | p95 Latency (µs) |
| ----------- | ---------------------- | ---------------- | ---------------- |
| 1           | ~6.6                   | ~151000          | ~170000          |
| 2           | ~4.9                   | ~400000          | ~600000          |
| 3           | ~4.5                   | ~655000          | ~900000          |
| 4           | ~5.0                   | ~807000          | ~1100000         |

---

### 📊 Observations

* Throughput increases initially with concurrency.
* Latency increases as concurrency grows due to queuing effects.
* Dynamic batching improves throughput by grouping incoming requests under higher load.
* The system was evaluated on CPU; GPU acceleration can further enhance performance.

---

## 🔄 Model Versioning

The system supports multiple versions of the YOLO model:

```
yolo/
├── 1/
└── 2/
```

Each version can be accessed independently:

```
/v2/models/yolo/versions/1/infer
/v2/models/yolo/versions/2/infer
```

This versioning mechanism enables **zero-downtime deployment**, allowing new models to be introduced without interrupting existing inference requests.

---

## ⚙️ Optimization & Scalability

* **Dynamic Batching**
  Groups requests to improve throughput and reduce latency under load.

* **Model Versioning**
  Enables safe deployment and rollback strategies.

* **Instance Groups**
  Multiple model instances are deployed to allow parallel inference execution, improving overall throughput.

---

## ⚙️ Key Features Implemented

* ✔ ONNX model export and validation
* ✔ Triton deployment using Docker
* ✔ Ensemble inference pipeline (pre → model → post)
* ✔ Dynamic batching for performance optimization
* ✔ Model versioning support
* ✔ Parallel inference using multiple instances
* ✔ Performance benchmarking using perf_analyzer
* ✔ REST API-based inference endpoints

---

## 💡 Challenges & Learnings

During development, several practical challenges were encountered:

* Understanding Triton model repository structure
* Debugging model loading and backend issues
* Managing tensor shapes across pipeline stages
* Implementing batching and versioning correctly

These experiences provided valuable insights into how real-world ML systems are deployed and optimized.

---

## 🏁 Conclusion

This project demonstrates how to transition a deep learning model from development to deployment using Triton Inference Server. It highlights key principles of scalable ML systems, including modular design, performance optimization, and production readiness.

---

## 🙌 Acknowledgment

This project was built as part of a hands-on learning experience in deploying machine learning models using modern MLOps tools.

# 🚀 Triton CV Service – YOLOv8 Inference Pipeline

## 📌 Overview

This project demonstrates a complete end-to-end deployment of a computer vision model using **NVIDIA Triton Inference Server**. The goal was to take a trained YOLOv8 model, convert it into an optimized format, and serve it in a scalable, production-ready environment with support for batching, versioning, and performance benchmarking.

The system is designed to simulate how real-world ML services operate — from model export and validation to deployment and performance testing.

---

## 🧠 Architecture

The pipeline is built using an **ensemble model architecture**, which chains together multiple stages of inference:

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
  Handles input normalization, reshaping, and formatting.

* **YOLO Model (ONNX)**
  Core object detection model exported from YOLOv8.

* **Postprocess**
  Interprets model output into usable predictions.

* **Ensemble Model**
  Connects all steps into a single inference pipeline.

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

(For Windows users, use PowerShell equivalent if needed.)

---

## ⚡ Performance Benchmarking

The performance of the model was evaluated using Triton’s `perf_analyzer`.

### Run Benchmark

```bash
bash scripts/run_benchmark.sh
```

---

### Sample Results

| Concurrency | Throughput (infer/sec) | Latency (µs) |
| ----------- | ---------------------- | ------------ |
| 1           | ~6.6                   | ~151,000     |
| 2           | ~4.9                   | ~400,000     |
| 3           | ~4.5                   | ~655,000     |
| 4           | ~5.0                   | ~807,000     |

---

### 📊 Observations

* Throughput increases initially with concurrency.
* Latency increases as concurrency grows due to queuing.
* Dynamic batching helps improve efficiency at higher loads.

---

## 🔄 Model Versioning

The system supports multiple model versions:

```
yolo/
├── 1/
└── 2/
```

Requests can target specific versions:

```
/v2/models/yolo/versions/1/infer
/v2/models/yolo/versions/2/infer
```

This enables **zero-downtime model updates**.

---

## ⚙️ Key Features Implemented

* ✔ ONNX model export and validation
* ✔ Triton deployment using Docker
* ✔ Ensemble pipeline (pre → model → post)
* ✔ Dynamic batching for performance optimization
* ✔ Model versioning support
* ✔ Performance benchmarking
* ✔ REST API inference endpoints

---

## 💡 Challenges & Learnings

During development, several practical challenges were encountered:

* Handling strict Triton model repository structure
* Debugging model loading and backend errors
* Managing tensor shapes across pipeline stages
* Understanding batching and versioning behavior

These challenges helped build a deeper understanding of how production ML systems operate.

---

## 🏁 Conclusion

This project successfully demonstrates how to take a deep learning model from development to deployment using Triton. It highlights key aspects of real-world ML systems such as scalability, modular design, and performance optimization.

---

## 🙌 Acknowledgment

This project was built as part of a hands-on learning experience in deploying machine learning models using modern MLOps tools.



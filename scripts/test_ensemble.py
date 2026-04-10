import requests
import numpy as np

# Dummy input (1 image, 3x640x640)
data = np.random.rand(1, 3, 640, 640).astype(np.float32)

url = "http://localhost:8000/v2/models/ensemble_yolo/infer"

payload = {
    "inputs": [
        {
            "name": "INPUT",
            "shape": [1, 3, 640, 640],
            "datatype": "FP32",
            "data": data.flatten().tolist()
        }
    ]
}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Response:", response.text)
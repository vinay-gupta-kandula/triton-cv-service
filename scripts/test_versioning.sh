#!/bin/bash

echo "Testing YOLO version 1..."
curl -X POST http://localhost:8000/v2/models/yolo/versions/1/infer

echo "Testing YOLO version 2..."
curl -X POST http://localhost:8000/v2/models/yolo/versions/2/infer

echo "Versioning test completed."
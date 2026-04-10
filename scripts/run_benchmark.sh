#!/bin/bash

# Create results folder
mkdir -p results

echo "Running Triton Benchmark..."

docker run --rm --net=host nvcr.io/nvidia/tritonserver:23.12-py3-sdk \
    perf_analyzer \
    -m yolo \
    -u localhost:8000 \
    --concurrency-range 1:32:3 \
    --input-data=zero \
    -f results/benchmark.csv

echo "Benchmark completed. Results saved to results/benchmark.csv"
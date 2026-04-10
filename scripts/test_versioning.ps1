Write-Host "Testing YOLO version 1..."
Invoke-WebRequest -Uri http://localhost:8000/v2/models/yolo/versions/1/infer -Method Post

Write-Host "Testing YOLO version 2..."
Invoke-WebRequest -Uri http://localhost:8000/v2/models/yolo/versions/2/infer -Method Post

Write-Host "Versioning test completed."
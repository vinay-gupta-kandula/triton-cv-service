import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get input tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            img = input_tensor.as_numpy()

            # Ensure float32
            img = img.astype(np.float32)

            # Normalize (0–1)
            img = img / 255.0

            # Handle input shapes properly
            # Case 1: (H, W, C) → add batch + transpose
            if img.ndim == 3:
                img = np.transpose(img, (2, 0, 1))  # (C, H, W)
                img = np.expand_dims(img, axis=0)   # (1, C, H, W)

            # Case 2: (B, H, W, C) → transpose
            elif img.ndim == 4 and img.shape[-1] == 3:
                img = np.transpose(img, (0, 3, 1, 2))  # (B, C, H, W)

            # Case 3: already (B, C, H, W) → do nothing

            # Ensure final shape is (B, 3, 640, 640)
            # Minimal safe resize (avoid distortion-heavy ops)
            batch_size = img.shape[0]
            if img.shape[1:] != (3, 640, 640):
                resized = []
                for i in range(batch_size):
                    resized.append(np.resize(img[i], (3, 640, 640)))
                img = np.stack(resized, axis=0)

            # Output tensor (must match yolo input name)
            output_tensor = pb_utils.Tensor("images", img)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output_tensor])
            )

        return responses
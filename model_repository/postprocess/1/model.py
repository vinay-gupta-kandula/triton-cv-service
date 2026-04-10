import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            output = pb_utils.get_input_tensor_by_name(request, "output0")
            data = output.as_numpy()

            # Simple processing (just pass-through or basic formatting)
            result = data.astype(np.float32)

            out_tensor = pb_utils.Tensor("FINAL_OUTPUT", result)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_tensor])
            )

        return responses
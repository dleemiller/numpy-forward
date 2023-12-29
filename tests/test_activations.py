import unittest
import numpy as np
import torch
import inference.activations as activations


class TestActivations(unittest.TestCase):
    def test_relu(self):
        x_np = np.random.randn(10, 10)
        x_torch = torch.from_numpy(x_np)

        result_np = activations.relu(x_np)
        result_torch = torch.relu(x_torch).numpy()

        np.testing.assert_allclose(result_np, result_torch, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

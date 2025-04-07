import unittest
import numpy as np
from model import LogisticRegression  # Adjust import path if needed

class TestLogisticRegression(unittest.TestCase):
    def test_model_input(self):
        model = LogisticRegression()
        sample_input = np.random.rand(10, 5)
        try:
            model.predict(sample_input)
        except Exception as e:
            self.fail(f"Model failed to accept input: {e}")

    def test_model_output_shape(self):
        model = LogisticRegression()
        sample_input = np.random.rand(10, 5)
        predictions = model.predict(sample_input)
        self.assertEqual(predictions.shape, (10,))

if __name__ == '__main__':
    unittest.main()
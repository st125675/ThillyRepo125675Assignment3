import unittest
import numpy as np
from LogisticRegresson import LogisticRegression  # Adjust import path if needed

class TestLogisticRegression(unittest.TestCase):
    def test_model_input(self):
        model = LogisticRegression()
        sample_input = np.random.rand(10, 5)
        # Train the model before predicting
        sample_labels = np.random.randint(0, 2, 10)  # Example labels (assuming binary classification)
        model.fit(sample_input, sample_labels)
        try:
            model.predict(sample_input)
        except Exception as e:
            self.fail(f"Model failed to accept input after training: {e}")

    def test_model_output_shape(self):
        model = LogisticRegression()
        sample_input = np.random.rand(10, 5)
        # Train the model before predicting
        sample_labels = np.random.randint(0, 2, 10)  # Example labels
        model.fit(sample_input, sample_labels)
        predictions = model.predict(sample_input)
        self.assertEqual(predictions.shape, (10,))

if __name__ == '__main__':
    unittest.main()
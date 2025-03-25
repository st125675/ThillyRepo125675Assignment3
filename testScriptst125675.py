import unittest
import numpy as np
import pandas as pd

class CustomLogisticRegression:
    """
    A basic logistic regression model with L2 regularization.
    """
    def __init__(self, penalty='l2', C=1.0):
        self.penalty = penalty
        self.C = C
        self.coef_ = None
        self.intercept_ = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Simple gradient descent (placeholder, replace with actual training logic)
        self.coef_ = np.zeros((X.shape[1],))
        self.intercept_ = 0.0
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert DataFrame to NumPy array if needed
        if X.shape[1] != len(self.coef_):
            raise ValueError("Incorrect input shape")
        logits = np.dot(X, self.coef_) + self.intercept_
        return (self.sigmoid(logits) > 0.5).astype(int)

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.X_sample = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_sample = np.array([0, 1, 1, 0])
        self.model = CustomLogisticRegression(penalty='l2', C=1.0)
        self.model.fit(self.X_sample, self.y_sample)

    def test_model_input(self):
        self.assertTrue(self.model.predict(self.X_sample).shape == self.y_sample.shape)

    def test_model_input_pandas(self):
        X_sample_pd = pd.DataFrame(self.X_sample)
        self.assertTrue(self.model.predict(X_sample_pd).shape == self.y_sample.shape)

    def test_output_shape(self):
        y_pred = self.model.predict(self.X_sample)
        self.assertEqual(y_pred.shape, self.y_sample.shape)

    def test_output_data_type(self):
        y_pred = self.model.predict(self.X_sample)
        self.assertIsInstance(y_pred, np.ndarray)

    def test_model_invalid_input_shape(self):
        X_invalid = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            self.model.predict(X_invalid)
    
    def test_model_coefficients_shape(self):
        self.assertEqual(self.model.coef_.shape, (self.X_sample.shape[1],))
        self.assertIsInstance(self.model.intercept_, float)

    def test_prediction_probability_range(self):
        logits = np.dot(self.X_sample, self.model.coef_) + self.model.intercept_
        probs = self.model.sigmoid(logits)
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        # Number of training examples and features
        rows, cols = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(cols)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.iterations):
            # Predicted values
            Y_pred = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (1 / rows) * np.dot(X.T, (Y_pred - Y))
            db = (1 / rows) * np.sum(Y_pred - Y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
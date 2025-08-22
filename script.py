import numpy as np
import matplotlib.pyplot as plt

# Simple 2D dataset (linearly separable)
X = np.array([
    [2, 1],
    [1, -1],
    [-1, -2],
    [-2, -1],
    [2, 2],
    [-2, -3],
])
y = np.array([1, 1, -1, -1, 1, -1])  # labels (+1 / -1)

# Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=10):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_epochs):
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = np.where(linear_output >= 0, 1, -1)
                if target != y_pred:
                    self.weights += self.lr * target * xi
                    self.bias += self.lr * target

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)

# Train perceptron
clf = Perceptron(learning_rate=0.1, n_epochs=10)
clf.fit(X, y)

# Get weights and bias
w, b = clf.weights, clf.bias

# Plot data points
plt.figure(figsize=(6,6))
for i, point in enumerate(X):
    if y[i] == 1:
        plt.scatter(point[0], point[1], color="blue", marker="o", label="+1" if i == 0 else "")
    else:
        plt.scatter(point[0], point[1], color="red", marker="x", label="-1" if i == 2 else "")

# Decision boundary: w1*x1 + w2*x2 + b = 0 -> x2 = -(w1*x1 + b)/w2
x1_vals = np.linspace(-3, 3, 100)
x2_vals = -(w[0]*x1_vals + b) / w[1]
plt.plot(x1_vals, x2_vals, 'k--', label="Decision boundary")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Perceptron Classification Example")
plt.grid(True)
plt.show()

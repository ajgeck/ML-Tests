import numpy as np
from keras.datasets import mnist

def vectorize_image(input_matrix):
    return np.insert(input_matrix.flatten(), 0, 1)

# Sigmoid Function f(u)
def sigmoid(u):
    return 1/(1 + np.exp(-u))

# Trains Data
def train(x_train, y_train, W, epochs, learning_rate):
    for epoch in range(epochs):
        for x, y in zip(x_train, y_train):
            x_vector  = vectorize_image(x)
            u         = np.dot(W, x_vector)
            y_hat     = sigmoid(u)
            y_true    = np.zeros(10)
            y_true[y] = 1

            # Error in each output node
            error = (y_hat - y_true) * y_hat * (1 - y_hat)

            # Update Weight Matrix
            for index in range(10):
                gradient  = 2 * error[index] * x_vector
                W[index] -= learning_rate * gradient
    return W

# Evaluate Model
def evaluate(x_test, y_test, W):
    correct = 0
    for x, y in zip(x_test, y_test):
        x_vector = vectorize_image(x)
        y_hat    = sigmoid(np.dot(W, x_vector))
        if np.argmax(y_hat) == y:
            correct += 1
    return correct/len(y_test)

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the Dataset
x_train, x_test = x_train/255.0, x_test/255.0

# Seed for Replicability
np.random.seed(436)

# Neural Network Parameters
W             = np.random.randn(10, 28*28 + 1)
epochs        = 10
learning_rate = 0.5

# Train the Model
W = train(x_train, y_train, W, epochs, learning_rate)

# Evaluate the Model
accuracy = evaluate(x_test, y_test, W)

# Print Results
print(f"Accuracy: {accuracy}, Learning Rate: {learning_rate}, Epochs: {epochs}")
"""
These are some quick implementations of neural networks for regression.

Training is via backprop

No bias because meh
"""
import numpy as np
import math

class ffnn(object):
    def __init__(self, learning_rate, input_dim):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        
        # xavier initialization for layer 1 (input x 8)
        self.W1 = np.random.randn(self.input_dim, 8) / np.sqrt(self.input_dim)

        # xavier for layer 2 (8 x 1)
        self.W2 = np.random.randn(8, 1) / np.sqrt(8)

    # activation function
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    # derivative of activation function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # absolute loss
    def loss(self, y_hat, y):
       return y - y_hat

    # forward pass:
    #    y = W^T X
    def forward_pass(self, X):
        h = self.sigmoid(np.dot(X, self.W1))
        y_hat = self.sigmoid(np.dot(h, self.W2))
        return h, y_hat

    # backprop:
    #    dE/dW  =  dE/dO * dO/dSigmoid * dSigmoid/dW1  (via chain rule)
    #           =  loss * sigmoid derivative
    def backward_pass(self, loss, h, y_hat, X):
        dW2 =  loss * self.sigmoid_derivative(y_hat)
        dh = np.dot(dW2, self.W2.T)
        dW1 = dh * self.sigmoid_derivative(h)
        return dW1, dW2

    # training step: forward and backward pass
    def train_step(self, X, Y):
        h, y_hat = self.forward_pass(X)
        loss = self.loss(y_hat, Y)
        dW1, dW2 = self.backward_pass(loss, h, y_hat, X)

        self.W2 += self.learning_rate * np.dot(h.T, dW2)
        self.W1 += self.learning_rate * np.dot(X.T, dW1)

        return loss, y_hat
                                               

# some test data of dim 3
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

nn = ffnn(0.9, 3)

for i in range(100000):
    l, y_hat = nn.train_step(X, y)
    if i % 10000 == 0:
        print str(np.mean(np.abs(l))), y_hat

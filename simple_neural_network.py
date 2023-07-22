# This code is incomplete.

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    # Defining Sigmoid activation function
    sig = 1 / (1 + np.exp(-x))
    return sig


def derivative_sigmoid(x):
    # Defining derivative of sigmoid activation function
    dsig = sigmoid(x) * (1 - sigmoid(x))
    return dsig


def graph(x, y):
    # Defining method to graph activation function (sigmoid)
    x_data = x
    y_data1 = y
    y_data2 = derivative_sigmoid(x)
    plt.figure(figsize=(5, 5))
    plt.plot(x_data, y_data1, color='blue', label='sigmoid(x)')
    plt.plot(x_data, y_data2, color='red', label="sigmoid'(x)")
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.xlim(-5, 5)
    plt.xticks(np.linspace(-5, 5, 5))
    plt.title('Graph of y(x)')
    plt.legend()
    plt.show()


def forward_pass(x, w1, b1, w2, b2):
    """
        Forward pass of a neural network with one hidden layer.

        Arguments:
        x -- Input data, shape: (number_of_samples, number_of_dimensions)
        w1 -- Weight matrix of the hidden layer, shape: (number_of_dimensions, hidden_units)
        b1 -- Bias vector of the hidden layer, shape: (hidden_units,)
        w2 -- Weight vector connecting the hidden layer to the output, shape: (hidden_units,)
        b2 -- Bias scalar for the output layer

        Returns:
        y_hat -- Predicted output, shape: (number_of_samples,)
        """
    # Calculation of activation function of hidden layer
    hidden_layer = np.dot(x, w1) + b1
    hidden_output = sigmoid(hidden_layer)

    # Calculation of output
    yhat = np.dot(hidden_output, np.transpose(w2)) + b2
    return yhat


def mse(y_true, y_pred):
    # Mean Squared Error Loss
    mse_loss = np.mean((y_pred - y_true) ** 2)
    return mse_loss


def derivative_mse(y_true, y_pred):
    # Derivative of MSE loss wrt y_pred
    mse_derivative = 2 * (y_pred - y_true) / len(y_true)
    return mse_derivative


def cal_gradient(x, y, y_hat, w1, b1, w2, b2):
    """
            Calculation of Gradient.

            Arguments:
            x -- Input data, shape: (number_of_samples, number_of_dimensions)
            y -- ground truth
            y_hat -- output of neural network
            w1 -- Weight matrix of the hidden layer, shape: (number_of_dimensions, hidden_units)
            b1 -- Bias vector of the hidden layer, shape: (hidden_units,)
            w2 -- Weight vector connecting the hidden layer to the output, shape: (hidden_units,)
            b2 -- Bias scalar for the output layer

            Returns:
            y_hat -- Predicted output, shape: (number_of_samples,)
            """
    N = len(y)
    # For ease of calculation
    hidden_output = sigmoid(x * w1 + b1)
    # Gradients of output layer
    dL_dw2 = np.dot(2 * (y_hat - y) / N, hidden_output.T)
    dL_db2 = 2 * (y_hat - y) / N


X = np.linspace(-5, 5, 100)
Y1 = sigmoid(X)
Xi = np.column_stack((len(X), X))
graph(X, Y1)

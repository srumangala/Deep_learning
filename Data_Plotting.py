# Data Plotting

# Importing packages
import numpy as np
import matplotlib.pyplot as plt

# Creating 1D Data
# A 1D dataset {(x1, y1),(x2, y2), . . . ,(x100, y100)}
# where x is evenly spaced w.r.t.the range [−1, 1], and y = 0.1 · x + x^2 + x^3
X = np.linspace(-1, 1, num=100)
Y = 0.1 * X + np.power(X, 2) + np.power(X, 3)
Z = list(zip(X, Y))

# Creating 2D Data. A 2D dataset {(x1, y1),(x2, y2), . . . ,(x100, y100)}, where each dimension of xi is sampled from
# a standard normal distribution. yi = 0 if ||xi||^2 (Squared Euclidean Norm) < 1, else yi = 1. The seed = 42
np.random.seed(42)  # Setting seed so that the same output is obtained every time
x = np.random.standard_normal((100, 2))
norms = np.linalg.norm(x, axis=1)  # Calculating Euclidean norms row-wise (||x||)
y = np.where(norms ** 2 < 1, 0, 1)  # Calculating value of y. if ||x||^2 < 1, y=0. if ||x||^2 > 1, y=1


def create_dataset(n_samples=100):
    # Creating a function to retrieve select number of samples from 1D dataset
    x1 = X[:n_samples]  # Slicing the np array
    y1 = Y[:n_samples]  # Slicing the np array
    z1 = list(zip(x1, y1))  # Joining x and y
    return z1


def create2d_dataset(n_samples=100):
    # Creating a function to retrieve select number of samples from 2D dataset
    x1 = x[:n_samples]
    y1 = y[:n_samples]
    z1 = np.column_stack((x, y))
    return z1


class Dataset:
    # Creating a class to get dataset objects
    def __init__(self, n_samples=100):
        self.n_samples = n_samples

    def load_data(self):
        # Function to create a 1D dataset
        return create_dataset(self.n_samples)

    def load2d_data(self):
        # Function to create a 2D dataset
        return create2d_dataset(self.n_samples)

    def graph1d_data(self):
        # Function to plot the 1D dataset
        dataset = self.load_data()
        x_data = [data[0] for data in dataset]
        y_data = [data[1] for data in dataset]
        plt.figure(num='A 1D Figure', figsize=(5, 5))
        plt.plot(x_data, y_data)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-1, 1)
        plt.xticks(np.linspace(-1, 1, 5))
        plt.title('1D Dataset')
        plt.show()

    def graph2d_data(self):
        # Function to plot the 2D dataset
        dataset = self.load2d_data()
        x_data = dataset[:, 0]
        y_data = dataset[:, 1]
        plt.figure(num='A 2D Figure', figsize=(5, 5))
        labels = dataset[:, 2]
        # Plot class 0 (red triangles)
        class0_indices = np.where(labels == 0)
        plt.scatter(x_data[class0_indices], y_data[class0_indices], marker='o', color='blue', label='Class 0')
        # Plot class 1 (blue circles)
        class1_indices = np.where(labels == 1)
        plt.scatter(x_data[class1_indices], y_data[class1_indices], marker='^', color='red', label='Class 1')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('2D Dataset')
        plt.legend()
        plt.show()


# Calling the object and plotting 1D figure
data_obj1 = Dataset(100)
data1 = data_obj1.load_data()
data_obj1.graph1d_data()

# Calling the object and plotting 2D figure
data_obj2 = Dataset(100)
data2 = data_obj2.load2d_data()
data_obj2.graph2d_data()

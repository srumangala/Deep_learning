#  Importing Numpy

import numpy as np

#  Creating a 4*4 matrix X1 and a 2*2 filter W1

X1 = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
W1 = np.ones((2, 2))

# Creating a 5*5 matrix X2 and a 3*3 filter W2
X2 = np.array([[0, 1, 2, 3, 4], [4, 5, 6, 7, 8], [8, 9, 10, 11, 0], [12, 13, 14, 15, 20], [12, 13, 14, 15, 20]])
W2 = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])


def convolve(x, w, s=1):
    # Function for convolution of square matrices
    matrix_size = len(x)
    kernel_size = len(w)
    stride = s
    output_size = (matrix_size - kernel_size) + 1
    output_size_actual = (matrix_size - kernel_size) // stride + 1
    output_matrix = np.zeros((output_size, output_size))  # matrix that has additional zeroes in rows and columns
    trimmed_matrix = np.zeros((output_size_actual, output_size_actual))  # final matrix without additional zeroes

    for i in range(0, output_size, stride):
        for j in range(0, output_size, stride):
            x_half = x[i:i + kernel_size, j:j + kernel_size]  # The small matrix to multiply with kernel function
            output_matrix[i // stride, j // stride] = np.sum(x_half * w)
            last_nonzero_row = np.max(np.nonzero(output_matrix.sum(axis=1)))
            last_nonzero_col = np.max(np.nonzero(output_matrix.sum(axis=0)))
            # Explanation about above lines of code : matrix.sum(axis=1): This computes the sum along each row of the
            # matrix. axis=1 indicates that the summation is performed horizontally.
            # np.nonzero(matrix.sum(axis=1)): This returns the indices where the sum computed in the previous step is
            # non-zero. The np.nonzero function finds the indices of the non-zero elements.
            # np.max(np.nonzero(matrix.sum(axis=1))): This takes the maximum value from the array of non-zero indices
            # obtained in the previous step. It represents the index of the last non-zero element along the rows.
            trimmed_matrix = output_matrix[:last_nonzero_row + 1, :last_nonzero_col + 1]
    print(trimmed_matrix)


convolve(X1, W1)
convolve(X2, W2, 2)

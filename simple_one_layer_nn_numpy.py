# Simple Neural Network with One Layer using NumPy
# This code demonstrates a single-layer neural network implemented with NumPy

import numpy as np

# Network parameters
input_size = 3
output_size = 2

# Initialize weights and biases randomly using NumPy
weights = np.random.uniform(-1, 1, (output_size, input_size))  # Shape: (2, 3)
biases = np.random.uniform(-1, 1, output_size)  # Shape: (2,)

def forward(inputs):
    inputs = np.array(inputs)
    outputs = np.dot(weights, inputs) + biases
    return outputs

# Example usage
if __name__ == "__main__":
    sample_input = [0.5, -0.2, 0.1]
    print("Sample Input:", sample_input)
    print("Weights shape:", weights.shape)
    print("Weights:\n", weights)
    print("Biases:", biases)
    
    output = forward(sample_input)
    print("Output:", output)
    print("Output shape:", output.shape)

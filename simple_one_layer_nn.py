import random

# Simple Neural Network with One Layer (No external packages)
# This code demonstrates a single-layer neural network (perceptron) for demonstration purposes.

input_size = 3
output_size = 2

# Initialize weights and biases randomly
weights = [random.uniform(-1, 1) for _ in range(input_size * output_size)]
biases = [random.uniform(-1, 1) for _ in range(output_size)]

def forward(inputs):
    outputs = []
    for j in range(output_size):
        weighted_sum = 0
        for i in range(input_size):
            weighted_sum += inputs[i] * weights[j * input_size + i]
        weighted_sum += biases[j]
        outputs.append(weighted_sum)
    return outputs

# Example usage
if __name__ == "__main__":
    sample_input = [0.5, -0.2, 0.1]
    output = forward(sample_input)
    print("Output:", output)

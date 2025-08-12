# Dense Layer Class - Simple Implementation for Multi-Layer Network
# This code demonstrates connecting multiple dense layers

import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.uniform(-1, 1, (input_size, output_size))
        self.biases = np.random.uniform(-1, 1, output_size)
    
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

# Example usage - 2-3 layers connected
if __name__ == "__main__":
    # Create layers
    layer1 = DenseLayer(input_size=4, output_size=6)  # 4 -> 6
    layer2 = DenseLayer(input_size=6, output_size=3)  # 6 -> 3
    layer3 = DenseLayer(input_size=3, output_size=2)  # 3 -> 2
    
    # Batch input (3 samples, 4 features each)
    batch_input = np.array([
        [0.5, -0.2, 0.1, 0.8],
        [0.3, 0.7, -0.4, 0.2],
        [-0.1, 0.2, 0.8, -0.5]
    ])
    
    print("Input shape:", batch_input.shape)
    
    # Forward pass through all layers
    output1 = layer1.forward(batch_input)
    print("Layer 1 output shape:", output1.shape)
    
    output2 = layer2.forward(output1)
    print("Layer 2 output shape:", output2.shape)
    
    output3 = layer3.forward(output2)
    print("Final output shape:", output3.shape)
    print("Final output:")
    print(output3)

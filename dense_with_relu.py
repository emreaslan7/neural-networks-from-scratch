import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.uniform(-1, 1, (input_size, output_size))
        self.biases = np.random.uniform(-1, 1, output_size)
    
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)

if __name__ == "__main__":
    # Create layers and activation
    layer1 = DenseLayer(input_size=4, output_size=6)
    relu1 = Activation_ReLU()
    
    layer2 = DenseLayer(input_size=6, output_size=3)
    relu2 = Activation_ReLU()
    
    layer3 = DenseLayer(input_size=3, output_size=2)
    
    # Batch input
    batch_input = np.array([
        [0.5, -0.2, 0.1, 0.8],
        [0.3, 0.7, -0.4, 0.2],
        [-0.1, 0.2, 0.8, -0.5]
    ])
    
    # Forward pass with ReLU activations
    output1 = layer1.forward(batch_input)
    output1_relu = relu1.forward(output1)
    
    output2 = layer2.forward(output1_relu)
    output2_relu = relu2.forward(output2)
    
    output3 = layer3.forward(output2_relu)
    
    print("Final output:")
    print(output3)

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

class Activation_Softmax:
    def forward(self, inputs):
        # Subtract max for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize to get probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

if __name__ == "__main__":
    # Create layers and activations
    layer1 = DenseLayer(input_size=4, output_size=8)
    relu1 = Activation_ReLU()
    
    layer2 = DenseLayer(input_size=8, output_size=6)
    relu2 = Activation_ReLU()
    
    layer3 = DenseLayer(input_size=6, output_size=4)
    relu3 = Activation_ReLU()
    
    # Output layer for classification (3 classes)
    output_layer = DenseLayer(input_size=4, output_size=3)
    softmax = Activation_Softmax()
    
    # Batch input
    batch_input = np.array([
        [0.1, -0.2, 0.1, 0.8],
        [0.3, 0.99, 0.4, 0.2],
        [-0.1, 0.2, -0.8, 0.1],
        [0.9, -0.3, 0.4, 0.99]
    ])
    
    print("Input shape:", batch_input.shape)
    print("Input data:")
    print(batch_input)
    print("\n" + "="*50)
    
    # Forward pass through the network
    # Hidden layers with ReLU
    output1 = layer1.forward(batch_input)
    output1_relu = relu1.forward(output1)
    print(f"After layer 1 + ReLU: shape {output1_relu.shape}")
    
    output2 = layer2.forward(output1_relu)
    output2_relu = relu2.forward(output2)
    print(f"After layer 2 + ReLU: shape {output2_relu.shape}")
    
    output3 = layer3.forward(output2_relu)
    output3_relu = relu3.forward(output3)
    print(f"After layer 3 + ReLU: shape {output3_relu.shape}")
    
    # Output layer with Softmax
    final_output = output_layer.forward(output3_relu)
    probabilities = softmax.forward(final_output)
    
    print(f"\nRaw output (before softmax): shape {final_output.shape}")
    print(final_output)
    
    print(f"\nFinal probabilities (after softmax): shape {probabilities.shape}")
    print(probabilities)
    
    print("\nPredicted classes (highest probability):")
    predicted_classes = np.argmax(probabilities, axis=1)
    print(predicted_classes)
    
    print("\nConfidence scores (max probability for each sample):")
    confidence_scores = np.max(probabilities, axis=1)
    print(confidence_scores)
    
    # Verify probabilities sum to 1
    print("\nProbability sums (should be ~1.0):")
    print(np.sum(probabilities, axis=1))

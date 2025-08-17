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

class CategoricalCrossEntropy:
    def forward(self, predictions, targets):
        # Clip predictions to prevent log(0)
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        
        # Calculate loss for each sample
        if targets.ndim == 1:
            # If targets are class indices (sparse)
            correct_confidences = predictions_clipped[range(len(predictions)), targets]
        else:
            # If targets are one-hot encoded
            correct_confidences = np.sum(predictions_clipped * targets, axis=1)
        
        # Calculate negative log likelihood
        negative_log_likelihoods = -np.log(correct_confidences)
        
        # Return mean loss
        return np.mean(negative_log_likelihoods)

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
    
    # Loss function
    loss_function = CategoricalCrossEntropy()
    
    # Batch input
    batch_input = np.array([
        [0.1, -0.2, 0.1, 0.8],
        [0.3, 0.99, 0.4, 0.2],
        [-0.1, 0.2, -0.8, 0.1],
        [0.9, -0.3, 0.4, 0.99]
    ])
    
    # Target classes (ground truth)
    targets = np.array([0, 1, 2, 1])  # Sparse format (class indices)
    
    print("Input shape:", batch_input.shape)
    print("Input data:")
    print(batch_input)
    print("\nTarget classes:", targets)
    print("\n" + "="*50)
    
    # Forward pass through the network
    output1 = layer1.forward(batch_input)
    output1_relu = relu1.forward(output1)
    
    output2 = layer2.forward(output1_relu)
    output2_relu = relu2.forward(output2)
    
    output3 = layer3.forward(output2_relu)
    output3_relu = relu3.forward(output3)
    
    # Output layer with Softmax
    final_output = output_layer.forward(output3_relu)
    probabilities = softmax.forward(final_output)
    
    print(f"Final probabilities: shape {probabilities.shape}")
    print(probabilities)
    
    # Calculate loss
    loss = loss_function.forward(probabilities, targets)
    print(f"\nCategorical Cross Entropy Loss: {loss:.4f}")
    
    # Predictions and analysis
    predicted_classes = np.argmax(probabilities, axis=1)
    print(f"\nPredicted classes: {predicted_classes}")
    print(f"Target classes:    {targets}")
    
    # Accuracy calculation
    accuracy = np.mean(predicted_classes == targets)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Detailed analysis for each sample
    print("\n" + "="*50)
    print("DETAILED ANALYSIS:")
    for i in range(len(batch_input)):
        print(f"\nSample {i+1}:")
        print(f"  Input: {batch_input[i]}")
        print(f"  Predicted probabilities: {probabilities[i]}")
        print(f"  Predicted class: {predicted_classes[i]}")
        print(f"  True class: {targets[i]}")
        print(f"  Confidence for true class: {probabilities[i][targets[i]]:.4f}")
        print(f"  Correct: {'✓' if predicted_classes[i] == targets[i] else '✗'}")

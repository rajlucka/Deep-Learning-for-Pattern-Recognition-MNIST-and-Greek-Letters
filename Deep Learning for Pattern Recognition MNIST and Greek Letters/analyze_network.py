#!/usr/bin/env python3
# Name: Raj Lucka
# Task 2: Analyze the trained network
# Part A: Visualize the weights of the first convolutional layer

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the CNN model architecture used for MNIST digit classification
class MyNetwork(nn.Module):
    """
    Convolutional Neural Network architecture used during training.
    Assumes input size of 28x28 and outputs predictions over 10 classes (digits 0–9).
    """
    def __init__(self):
        super(MyNetwork, self).__init__()

        # First convolutional layer: 1 input channel, 10 output filters of size 5x5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # Second convolutional layer: 10 input channels, 20 output filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # Dropout layer for regularization after conv layers
        self.dropout = nn.Dropout(0.3)

        # Max pooling layer to reduce spatial dimensions by half
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer: flatten 20 x 4 x 4 → 320 input features
        self.fc1 = nn.Linear(20 * 4 * 4, 50)

        # Output layer for 10 digit classes
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Apply conv1 → pool → ReLU
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)

        # Apply conv2 → dropout → pool → ReLU
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply fc1 → ReLU → fc2
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # Use log_softmax to match with NLLLoss
        return F.log_softmax(x, dim=1)

# Main function to load the model and visualize the first layer filters
def main(argv):
    model_path = "mnist_cnn.pth"  # Path to the trained model weights

    # Instantiate the model
    model = MyNetwork()

    # Load the model parameters from file (CPU-safe)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()  # Set to evaluation mode to disable dropout, etc.

    # Print model structure for reference
    print("Model Architecture:\n")
    print(model)

    # Extract weights from the first convolutional layer
    first_layer_weights = model.conv1.weight

    # Display the shape of the weight tensor: [10 filters, 1 input channel, 5x5]
    print("\nShape of conv1 weights:", first_layer_weights.shape)

    # Detach from computation graph and convert to NumPy for visualization
    weight_array = first_layer_weights.detach().cpu().numpy()

    # Create a 2x5 grid for plotting 10 filter kernels
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()

    # Loop through all 10 filters and plot each kernel as a grayscale image
    for i in range(10):
        # weight_array[i, 0, :, :] corresponds to the i-th filter (5x5)
        axes[i].imshow(weight_array[i, 0, :, :], cmap='gray')
        axes[i].set_title(f"Filter {i}")
        axes[i].axis('off')  # Hide axes for a cleaner look

    plt.tight_layout()  # Prevent overlap
    plt.show()          # Display the grid of filters

# Entry point
if __name__ == "__main__":
    main(sys.argv)

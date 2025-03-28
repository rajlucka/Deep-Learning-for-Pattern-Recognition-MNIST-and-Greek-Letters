#!/usr/bin/env python3
# Name: Raj Lucka
# Task 2B: Visualize the effect of each first-layer filter in the trained MNIST model on a sample image.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2  # Used for 2D filtering with custom kernels

class MyNetwork(nn.Module):
    """
    CNN architecture used for MNIST digit classification.
    Includes two convolutional layers, max pooling, dropout, and two FC layers.
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)      # 10 filters, input channel = 1
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)     # 20 filters after first conv layer
        self.dropout = nn.Dropout(0.3)                    # Dropout between conv and FC layers
        self.pool = nn.MaxPool2d(2, 2)                    # Downsampling with 2x2 window
        self.fc1 = nn.Linear(20 * 4 * 4, 50)              # Fully connected layer: 320 -> 50
        self.fc2 = nn.Linear(50, 10)                      # Output: 50 -> 10 digit classes

    def forward(self, x):
        x = self.conv1(x)         # First convolution
        x = self.pool(x)          # Downsampling
        x = F.relu(x)             # Non-linearity

        x = self.conv2(x)         # Second convolution
        x = self.dropout(x)       # Dropout regularization
        x = self.pool(x)          # Downsampling
        x = F.relu(x)             # Non-linearity

        x = x.view(x.size(0), -1) # Flatten feature maps to vector
        x = self.fc1(x)           # Fully connected hidden layer
        x = F.relu(x)             # Non-linearity
        x = self.fc2(x)           # Output layer
        return F.log_softmax(x, dim=1)  # Log-Softmax for classification

def main(argv):
    """
    Load pretrained model and MNIST data, apply each first-layer filter to an input image,
    and visualize both the kernel and the resulting filtered image.
    """
    # Load model and set it to evaluation mode
    model_path = "mnist_cnn.pth"
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Define image preprocessing: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load a single MNIST image (first image in training set)
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    first_batch = next(iter(train_loader))
    image_tensor, label = first_batch[0], first_batch[1]
    print(f"First training example label: {label.item()}")

    # Extract and convert the first conv layer's weights (10 filters)
    with torch.no_grad():
        filters = model.conv1.weight           # Shape: [10, 1, 5, 5]
        filters_np = filters.cpu().numpy()     # Convert to NumPy for OpenCV

    # Convert the input image to NumPy (single 28x28 grayscale image)
    image_np = image_tensor[0, 0].cpu().numpy().astype('float32')

    # Prepare a 5x4 grid of subplots: 2 columns per filter (kernel + output)
    fig, axes = plt.subplots(5, 4, figsize=(8, 10))
    axes = axes.flatten()

    # Loop through all 10 filters
    for i in range(10):
        kernel = filters_np[i, 0, :, :]  # Extract 5x5 kernel for filter i

        # Apply the filter to the input image using OpenCV
        filtered = cv2.filter2D(image_np, ddepth=-1, kernel=kernel)

        # Compute subplot indices: 2 columns per row (kernel and output)
        row = i // 2
        col_kernel = 2 * (i % 2)
        col_result = col_kernel + 1

        # Plot the kernel (weights)
        axes[row * 4 + col_kernel].imshow(kernel, cmap='gray')
        axes[row * 4 + col_kernel].set_title(f"Filter {i} Kernel")
        axes[row * 4 + col_kernel].axis('off')

        # Plot the result of applying the filter to the input image
        axes[row * 4 + col_result].imshow(filtered, cmap='gray')
        axes[row * 4 + col_result].set_title(f"Filter {i} Result")
        axes[row * 4 + col_result].axis('off')

    # Adjust layout and display all subplots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

#!/usr/bin/env python3
# Name: Raj Lucka
# Task 1F: Test the trained MNIST model on new, handwritten digit images.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps

class MyNetwork(nn.Module):
    """
    Convolutional Neural Network matching the training architecture.
    Designed for handwritten digit classification (MNIST).
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)    # First conv: 1 input channel, 10 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)   # Second conv: 10 input → 20 output channels
        self.dropout = nn.Dropout(0.3)                  # Dropout to reduce overfitting
        self.pool = nn.MaxPool2d(2, 2)                  # 2x2 Max pooling
        self.fc1 = nn.Linear(20 * 4 * 4, 50)            # Flattened conv output → hidden layer
        self.fc2 = nn.Linear(50, 10)                    # Hidden layer → 10 output classes

    def forward(self, x):
        x = self.conv1(x)           # Apply first convolution
        x = self.pool(x)            # Downsample via max pooling
        x = F.relu(x)               # Apply ReLU activation

        x = self.conv2(x)           # Apply second convolution
        x = self.dropout(x)         # Apply dropout
        x = self.pool(x)            # Another pooling step
        x = F.relu(x)               # ReLU activation

        x = x.view(x.size(0), -1)   # Flatten the feature maps for FC layer
        x = self.fc1(x)             # First fully connected layer
        x = F.relu(x)               # ReLU activation
        x = self.fc2(x)             # Output layer
        return F.log_softmax(x, dim=1)  # Log-Softmax for classification

def main(argv):
    """
    Loads a trained CNN model and evaluates it on a set of custom digit images.

    Expects:
      - Trained model saved as 'mnist_cnn.pth'
      - Input images: grayscale digits (28x28) in JPEG/PNG format
    """
    model_path = "mnist_cnn.pth"
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))  # Load pre-trained weights
    model.eval()  # Set model to inference mode (disables dropout, etc.)

    # Define transformation pipeline: convert to tensor and normalize like MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,))       # Normalize using MNIST stats
    ])

    # List of file paths to the handwritten digit images
    image_paths = [
        "digit0.jpg", "digit1.jpg", "digit2.jpg", "digit3.jpg", "digit4.jpg",
        "digit5.jpg", "digit6.jpg", "digit7.jpg", "digit8.jpg", "digit9.jpg"
    ]

    # Evaluate the model on each image
    for path in image_paths:
        # Load image using PIL and convert to grayscale (1 channel)
        image = Image.open(path).convert("L")

        # If your digits are black on white, invert colors to match MNIST format
        # image = ImageOps.invert(image)

        # Resize image to 28x28 (in case it's larger)
        image = image.resize((28, 28))

        # Apply transform and add batch dimension: [1, 1, 28, 28]
        input_tensor = transform(image).unsqueeze(0)

        # Perform forward pass through the trained model
        output = model(input_tensor)
        pred_digit = output.argmax(dim=1).item()  # Get the predicted class index

        # Print the result
        print(f"Image: {path} -> Predicted digit: {pred_digit}")

if __name__ == "__main__":
    main(sys.argv)

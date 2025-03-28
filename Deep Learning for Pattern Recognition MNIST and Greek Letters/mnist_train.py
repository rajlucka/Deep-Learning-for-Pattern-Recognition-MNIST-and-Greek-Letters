#!/usr/bin/env python3
# Name: Raj Lucka
# Task 1C: Train the MNIST network on GPU and visualize training/testing loss across epochs

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class MyNetwork(nn.Module):
    """
    Deep Convolutional Neural Network tailored for handwritten digit classification (MNIST).
    
    Model architecture:
      - Conv layer: 10 filters of size 5x5
      - Max Pooling (2x2) + ReLU
      - Conv layer: 20 filters of size 5x5
      - Dropout for regularization (30% dropout rate)
      - Max Pooling (2x2) + ReLU
      - Flatten operation
      - Fully connected (dense) layer: 320 -> 50 + ReLU
      - Output layer: 50 -> 10 + Log-Softmax activation
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # Input: [1,28,28] → Output: [10,24,24]
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # Input: [10,12,12] → Output: [20,8,8]
        self.dropout = nn.Dropout(0.3)                # 30% dropout after conv2
        self.pool = nn.MaxPool2d(2, 2)                # Reduces spatial size by half
        self.fc1 = nn.Linear(320, 50)                 # Fully connected: 20*4*4 = 320 → 50
        self.fc2 = nn.Linear(50, 10)                  # Output layer for 10 digit classes

    def forward(self, x):
        x = self.conv1(x)            # First conv layer
        x = self.pool(x)             # Max pooling (24x24 → 12x12)
        x = F.relu(x)                # Apply ReLU
        x = self.conv2(x)            # Second conv layer
        x = self.dropout(x)          # Apply dropout
        x = self.pool(x)             # Max pooling (8x8 → 4x4)
        x = F.relu(x)                # Apply ReLU
        x = x.view(x.size(0), -1)    # Flatten feature maps into a vector
        x = self.fc1(x)              # Dense layer: 320 → 50
        x = F.relu(x)                # Apply ReLU
        x = self.fc2(x)              # Output layer: 50 → 10
        return F.log_softmax(x, dim=1)  # Log-Softmax for classification (used with NLLLoss)

def load_data(batch_size=64):
    """
    Preprocess and load the MNIST dataset.
    
    Normalization is applied using the dataset's mean and standard deviation.
    Returns DataLoader objects for training and testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                            # Convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,))         # Normalize using MNIST stats
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # Shuffle for training
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # No shuffle for testing
    return train_loader, test_loader

def train_network(model, device, train_loader, optimizer, epoch):
    """
    Train the model for one epoch on the training dataset.
    
    Loss is computed using negative log-likelihood.
    Returns the average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  # Move input and labels to GPU (if available)

        optimizer.zero_grad()               # Clear previous gradients
        output = model(data)                # Forward pass
        loss = F.nll_loss(output, target)   # Compute NLL loss
        loss.backward()                     # Backward pass (compute gradients)
        optimizer.step()                    # Update weights using optimizer

        running_loss += loss.item() * data.size(0)  # Accumulate total loss over the batch

    avg_loss = running_loss / len(train_loader.dataset)  # Normalize by dataset size
    print(f"Epoch {epoch}: Training loss: {avg_loss:.4f}")
    return avg_loss

def test_network(model, device, test_loader, epoch):
    """
    Evaluate model performance on the test dataset.
    
    Computes and prints average loss and accuracy.
    """
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0

    with torch.no_grad():  # Disable gradient calculation for inference
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = F.nll_loss(output, target, reduction='sum')  # Sum loss over batch
            test_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)           # Get class prediction
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    avg_loss = test_loss / len(test_loader.dataset)  # Compute average loss
    accuracy = 100. * correct / len(test_loader.dataset)  # Compute accuracy
    print(f"Epoch {epoch}: Test loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss

def main(argv):
    """
    Entry point for model training and evaluation.
    
    Configures hyperparameters, trains the model for defined epochs,
    and visualizes loss progression across training and testing sets.
    """
    batch_size = 64
    epochs = 5
    learning_rate = 0.01
    momentum = 0.5

    use_cuda = torch.cuda.is_available()                     # Check for GPU availability
    device = torch.device("cuda" if use_cuda else "cpu")     # Set device
    print("Using device:", device)

    train_loader, test_loader = load_data(batch_size)        # Load MNIST dataset
    model = MyNetwork().to(device)                           # Initialize and move model to device
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # SGD optimizer

    training_losses = []
    testing_losses = []

    for epoch in range(1, epochs + 1):
        train_loss = train_network(model, device, train_loader, optimizer, epoch)  # Train for one epoch
        test_loss = test_network(model, device, test_loader, epoch)                # Evaluate on test set

        training_losses.append(train_loss)  # Track training loss
        testing_losses.append(test_loss)    # Track testing loss

    # Plot training and testing loss curves
    plt.plot(range(1, epochs + 1), training_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), testing_losses, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training and Testing Loss over Epochs")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

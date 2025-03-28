#!/usr/bin/env python3
# Name: Raj Lucka
# Task 1D: Train the MNIST network and save the trained model to a file

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
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture Overview:
      - Conv layer: 10 filters (5x5) → MaxPool → ReLU
      - Conv layer: 20 filters (5x5) → Dropout → MaxPool → ReLU
      - Fully connected: 320 → 50 → 10 (output)
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)      # Input: 1x28x28 → Output: 10x24x24
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)     # After pool: 10x12x12 → 20x8x8 → pool → 20x4x4
        self.dropout = nn.Dropout(0.3)                    # Dropout for regularization
        self.pool = nn.MaxPool2d(2, 2)                    # 2x2 Max pooling
        self.fc1 = nn.Linear(320, 50)                     # Flattened input: 20×4×4 = 320 → 50
        self.fc2 = nn.Linear(50, 10)                      # Output: 50 → 10 (digit classes)

    def forward(self, x):
        x = self.conv1(x)         # Apply first convolution
        x = self.pool(x)          # Downsample
        x = F.relu(x)             # Activation

        x = self.conv2(x)         # Apply second convolution
        x = self.dropout(x)       # Dropout layer
        x = self.pool(x)          # Downsample
        x = F.relu(x)             # Activation

        x = x.view(x.size(0), -1) # Flatten feature maps to vector
        x = self.fc1(x)           # Fully connected layer 1
        x = F.relu(x)             # Activation
        x = self.fc2(x)           # Final layer (logits)
        return F.log_softmax(x, dim=1)  # Log-Softmax for NLL loss

def load_data(batch_size=64):
    """
    Loads and normalizes the MNIST dataset using the known dataset statistics.

    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                              # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))           # Normalize using MNIST mean/std
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_network(model, device, train_loader, optimizer, epoch):
    """
    Train model for one full epoch on the training dataset.

    Args:
        model: the CNN to train
        device: torch.device (CPU or CUDA)
        train_loader: DataLoader for training samples
        optimizer: SGD/Adam/etc.
        epoch: current epoch number (for printing)

    Returns:
        avg_loss: average training loss for the epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()                 # Clear gradients
        output = model(data)                  # Forward pass
        loss = F.nll_loss(output, target)     # Compute loss
        loss.backward()                       # Backward pass
        optimizer.step()                      # Update model weights

        running_loss += loss.item() * data.size(0)  # Accumulate batch loss

    avg_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}: Training loss = {avg_loss:.4f}")
    return avg_loss

def test_network(model, device, test_loader, epoch):
    """
    Evaluate the model on test dataset and calculate loss and accuracy.

    Args:
        model: the trained CNN
        device: torch.device
        test_loader: DataLoader for test data
        epoch: current epoch number (for printing)

    Returns:
        avg_loss: average test loss
    """
    model.eval()  # Switch to evaluation mode
    test_loss = 0.0
    correct = 0

    with torch.no_grad():  # Disable gradient tracking
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sum loss over batch
            loss = F.nll_loss(output, target, reduction='sum')
            test_loss += loss.item()

            # Get predicted class and count correct predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Epoch {epoch}: Test loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
    return avg_loss

def save_network(model, filename="mnist_cnn.pth"):
    """
    Save the trained model to disk using PyTorch's state_dict.

    Args:
        model: the trained model
        filename: file path to save model weights
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def main(argv):
    """
    Main function that drives the training and evaluation process.

    - Initializes model, data, and optimizer
    - Runs training/testing loop
    - Saves the model
    - Plots loss curves
    """
    batch_size = 64
    epochs = 5
    learning_rate = 0.01
    momentum = 0.5

    # Set device (GPU if available)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # Load MNIST dataset
    train_loader, test_loader = load_data(batch_size)

    # Initialize model and optimizer
    model = MyNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    training_losses = []
    testing_losses = []

    # Run training and testing for each epoch
    for epoch in range(1, epochs + 1):
        train_loss = train_network(model, device, train_loader, optimizer, epoch)
        test_loss = test_network(model, device, test_loader, epoch)
        training_losses.append(train_loss)
        testing_losses.append(test_loss)

    # Save final trained model to file
    save_network(model, "mnist_cnn.pth")

    # Visualize training and testing loss progression
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs + 1), training_losses, label="Training Loss", marker='o')
    plt.plot(range(1, epochs + 1), testing_losses, label="Testing Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training and Testing Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

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
    Convolutional Neural Network for MNIST digit recognition.

    Architecture:
      - Conv1: 10 filters (5x5), ReLU, MaxPool(2x2)
      - Conv2: 20 filters (5x5), Dropout(30%), ReLU, MaxPool(2x2)
      - Flatten -> Fully Connected (320 -> 50) -> ReLU -> FC (50 -> 10) -> LogSoftmax
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)       # Input: 1x28x28 -> Output: 10x24x24
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)      # Input: 10x12x12 -> Output: 20x8x8
        self.dropout = nn.Dropout(0.3)                     # Dropout for regularization
        self.pool = nn.MaxPool2d(2, 2)                     # Downsampling by factor of 2
        self.fc1 = nn.Linear(320, 50)                      # Flattened size: 20*4*4 = 320
        self.fc2 = nn.Linear(50, 10)                       # 10 output classes (digits 0-9)

    def forward(self, x):
        x = self.conv1(x)           # Apply first conv layer: 1x28x28 → 10x24x24
        x = self.pool(x)            # After pooling: 10x12x12
        x = F.relu(x)               # ReLU activation

        x = self.conv2(x)           # Apply second conv: 10x12x12 → 20x8x8
        x = self.dropout(x)         # Apply dropout
        x = self.pool(x)            # After pooling: 20x4x4
        x = F.relu(x)               # ReLU activation

        x = x.view(x.size(0), -1)   # Flatten: (batch_size, 320)
        x = self.fc1(x)             # Fully connected layer: 320 → 50
        x = F.relu(x)               # ReLU activation
        x = self.fc2(x)             # Final output layer: 50 → 10
        return F.log_softmax(x, dim=1)  # Log-Softmax for classification

def load_data(batch_size=64):
    """
    Loads MNIST dataset and returns training and testing DataLoaders.

    Data is normalized using the dataset's mean and standard deviation.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                        # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))     # Normalize using MNIST mean & std
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # Shuffle training data
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # Don't shuffle test data
    return train_loader, test_loader

def train_network(model, device, train_loader, optimizer, epoch):
    """
    Trains the CNN model for one epoch.

    Returns:
        Average training loss for the epoch.
    """
    model.train()                     # Set model to training mode
    running_loss = 0.0               # Initialize running loss for this epoch

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  # Move data to GPU if available

        optimizer.zero_grad()         # Reset gradients from previous step
        output = model(data)          # Forward pass
        loss = F.nll_loss(output, target)  # Negative log likelihood loss
        loss.backward()               # Backward pass
        optimizer.step()              # Update model parameters

        running_loss += loss.item() * data.size(0)  # Multiply loss by batch size and accumulate

    avg_loss = running_loss / len(train_loader.dataset)  # Normalize by dataset size
    print(f"Epoch {epoch}: Training loss: {avg_loss:.4f}")
    return avg_loss

def test_network(model, device, test_loader, epoch):
    """
    Evaluates the trained model on the test set.

    Returns:
        Average test loss for the epoch.
    """
    model.eval()                     # Set model to evaluation mode
    test_loss = 0.0
    correct = 0                      # Track number of correct predictions

    with torch.no_grad():           # No gradient calculation needed for testing
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)    # Forward pass
            loss = F.nll_loss(output, target, reduction='sum')  # Sum loss over batch
            test_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)  # Get class with highest probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Epoch {epoch}: Test loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss

def main(argv):
    """
    Main routine to:
    - Initialize model and data
    - Train and evaluate for 5 epochs
    - Plot training and testing loss curves
    """
    batch_size = 64
    epochs = 5
    learning_rate = 0.01
    momentum = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    print("Using device:", device)

    train_loader, test_loader = load_data(batch_size)
    model = MyNetwork().to(device)                     # Instantiate model and move to device
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    training_losses = []   # Store training loss for each epoch
    testing_losses = []    # Store testing loss for each epoch

    for epoch in range(1, epochs + 1):
        train_loss = train_network(model, device, train_loader, optimizer, epoch)
        test_loss = test_network(model, device, test_loader, epoch)
        training_losses.append(train_loss)
        testing_losses.append(test_loss)

    # Plot the training and testing loss curves
    plt.plot(range(1, epochs + 1), training_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), testing_losses, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training and Testing Loss over Epochs")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

#!/usr/bin/env python3
# Name: Raj Lucka

"""
FashionMNIST Experiments with 32 Channels per Convolution Layer

This script evaluates how the number of convolutional layers, filter size,
and dropout affect CNN performance on FashionMNIST.

Each configuration is trained for 10 epochs. Accuracy and training time are recorded.
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import itertools
import pandas as pd
import matplotlib.pyplot as plt

# Select GPU if available for faster computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ParameterizedNet(nn.Module):
    """
    Configurable CNN:
    - Uses fixed 32 filters per convolutional layer
    - Number of conv layers, filter size, and dropout are all adjustable
    - Fully connected head has 50 hidden units, followed by 10-class output
    """
    def __init__(self, num_conv_layers=2, filter_size=5, dropout_rate=0.25):
        super(ParameterizedNet, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.filter_size = filter_size
        self.dropout_rate = dropout_rate

        layers = []
        in_channels = 1  # FashionMNIST has 1 grayscale input channel
        padding = filter_size // 2  # Pad to retain spatial dimensions before pooling

        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, 32, kernel_size=filter_size, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))  # Reduce feature map size by half
            in_channels = 32  # Set for next layer

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout_rate)

        # Dynamically calculate output size after conv layers using dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            conv_output = self.conv(dummy_input)
            flattened_size = conv_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 50)  # Intermediate dense layer
        self.fc2 = nn.Linear(50, 10)  # Final layer outputs 10 class scores

    def forward(self, x):
        x = self.conv(x)              # Feature extraction via conv layers
        x = self.dropout(x)           # Apply dropout to conv features
        x = x.view(x.size(0), -1)     # Flatten feature maps to 1D
        x = F.relu(self.fc1(x))       # Fully connected + ReLU
        x = self.fc2(x)               # Output logits
        return F.log_softmax(x, dim=1)  # Return log-probabilities for classification


# Normalize FashionMNIST inputs using dataset-specific statistics
transform_fashion = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Download and prepare FashionMNIST training and test datasets
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_fashion)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_fashion)

# Set up DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

def train_epoch(model, device, train_loader, optimizer):
    """
    Train model for one epoch.
    Returns average loss over all training examples.
    """
    model.train()
    running_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # Clear gradients
        output = model(data)  # Forward pass
        loss = F.nll_loss(output, target)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item() * data.size(0)  # Sum total loss over batch

    return running_loss / len(train_loader.dataset)  # Return average loss

def test_model(model, device, test_loader):
    """
    Evaluate model on test data.
    Returns average loss and accuracy percentage.
    """
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():  # Disable autograd for inference
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Total loss
            pred = output.argmax(dim=1, keepdim=True)  # Get predicted label
            correct += pred.eq(target.view_as(pred)).sum().item()  # Compare with true label

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# Fixed training settings
epochs = 10
lr = 0.01
momentum = 0.5

# Define search space for grid search
num_conv_layers_options = [2, 3, 4]
filter_size_options = [3, 5, 7]
dropout_rate_options = [0.0, 0.25, 0.5]

# Create list of all combinations of hyperparameters
param_grid = list(itertools.product(num_conv_layers_options, filter_size_options, dropout_rate_options))

results = []
experiment_num = 1

print(f"Starting experiments over {len(param_grid)} configurations...")

for num_layers, filter_size, dropout_rate in param_grid:
    # Each maxpool layer halves spatial resolution
    final_size = 28
    for _ in range(num_layers):
        final_size = final_size // 2

    # Skip configurations that collapse spatial dimensions completely
    if final_size <= 0:
        print(f"\nSkipping configuration: Layers={num_layers}, Filter={filter_size} → Final size invalid.")
        continue

    print(f"\nExperiment {experiment_num}: Layers={num_layers}, Filter={filter_size}, Dropout={dropout_rate}")

    # Instantiate model with given configuration
    model = ParameterizedNet(
        num_conv_layers=num_layers,
        filter_size=filter_size,
        dropout_rate=dropout_rate
    ).to(device)

    # Set optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Track training time for benchmarking
    start_time = time.time()

    # Run training and evaluation for fixed number of epochs
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, device, train_loader, optimizer)
        test_loss, test_accuracy = test_model(model, device, test_loader)
        print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Accuracy={test_accuracy:.2f}%")

    elapsed_time = time.time() - start_time

    # Store results from this experiment run
    results.append({
        'num_layers': num_layers,
        'filter_size': filter_size,
        'dropout_rate': dropout_rate,
        'final_accuracy': test_accuracy,
        'training_time': elapsed_time
    })

    experiment_num += 1

# Convert collected results into a DataFrame for analysis
df = pd.DataFrame(results)
print("\nSummary of Experiments:")
print(df)

# Plot final accuracy vs experiment index
plt.figure(figsize=(10, 6))
plt.scatter(range(len(df)), df['final_accuracy'], c='blue', label='Final Accuracy (%)')
plt.xlabel("Experiment Number")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy Across Network Configurations")
plt.legend()
plt.grid(True)
plt.show()

# Plot training time for each configuration
plt.figure(figsize=(10, 6))
plt.scatter(range(len(df)), df['training_time'], c='red', label='Training Time (s)')
plt.xlabel("Experiment Number")
plt.ylabel("Training Time (s)")
plt.title("Training Time Across Network Configurations")
plt.legend()
plt.grid(True)
plt.show()

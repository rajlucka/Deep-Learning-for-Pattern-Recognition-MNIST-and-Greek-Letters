#!/usr/bin/env python3
# Name: Raj Lucka
"""
Greek Letter Classification Experimentation

This script evaluates the impact of various hyperparameters on the classification accuracy
of a CNN trained to distinguish between three Greek letters: alpha, beta, and gamma.

Explored hyperparameters:
  - Fine-tuning: freeze convolutional base or train the entire network
  - Hidden units in fully connected layer: 50, 100, 200
  - Optimizer: SGD vs. Adam
  - Data augmentation: with/without random rotation
  - Number of epochs: 10 or 20
  - Batch size: 10 or 20

For hidden_units = 50, the convolutional layers and fc1 are initialized with pre-trained MNIST weights.
"""

import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns

# Automatically select GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to dataset: contains subfolders 'alpha', 'beta', 'gamma'
training_set_path = r"C:\Users\rajlu\OneDrive\Desktop\Work\Pattern Recognition\Projects\Video\Project1\Project1\greek_train"

# Define image transformation pipeline with optional augmentation
def get_transform(augmentation):
    if augmentation:
        return transforms.Compose([
            transforms.Resize((128, 128)),               # Resize input to standard size
            transforms.RandomRotation(10),               # Add slight variation (data augmentation)
            transforms.Grayscale(num_output_channels=1), # Convert RGB to grayscale
            transforms.ToTensor(),                       # Convert to tensor
            transforms.Resize((28, 28)),                 # Downscale to match MNIST input
            transforms.Normalize((0.1307,), (0.3081,))    # Normalize using MNIST mean and std
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

# Define CNN architecture — similar to the original MNIST network
class GreekCNN(nn.Module):
    def __init__(self, hidden_units=50):
        super(GreekCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)    # Output: 24x24
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)   # Output after pool: 4x4
        self.pool = nn.MaxPool2d(2)                     # Reduces spatial dimensions by half
        self.dropout1 = nn.Dropout(0.25)                # Prevents overfitting after flattening

        # Fully connected layers
        self.fc1 = nn.Linear(320, hidden_units)         # 320 = 20 * 4 * 4 (flattened conv output)
        self.dropout2 = nn.Dropout(0.5)                 # Applied before final classification
        self.fc2 = nn.Linear(hidden_units, 3)           # 3 output classes: alpha, beta, gamma

    def forward(self, x):
        x = F.relu(self.conv1(x))       # -> [batch, 10, 24, 24]
        x = self.pool(x)                # -> [batch, 10, 12, 12]
        x = F.relu(self.conv2(x))       # -> [batch, 20, 8, 8]
        x = self.pool(x)                # -> [batch, 20, 4, 4]
        x = torch.flatten(x, 1)         # Flatten to shape [batch, 320]
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Use log_softmax for NLLLoss compatibility

# Load pre-trained MNIST weights (for conv1, conv2, fc1) into the model
def load_pretrained(model, checkpoint_path="mnist_cnn.pth"):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.conv1.weight.data.copy_(checkpoint['conv1.weight'])
    model.conv1.bias.data.copy_(checkpoint['conv1.bias'])
    model.conv2.weight.data.copy_(checkpoint['conv2.weight'])
    model.conv2.bias.data.copy_(checkpoint['conv2.bias'])
    model.fc1.weight.data.copy_(checkpoint['fc1.weight'])
    model.fc1.bias.data.copy_(checkpoint['fc1.bias'])
    return model

# Freeze all layers except the classification head, unless full fine-tuning is enabled
def freeze_model(model, fine_tune):
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc1.parameters():
            param.requires_grad = True
        for param in model.fc2.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    return model

# Split the dataset into training and validation sets, return corresponding loaders
def get_greek_dataloaders(transform, batch_size):
    dataset = datasets.ImageFolder(root=training_set_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Perform one training epoch on the training set
def train_epoch(model, device, loader, optimizer):
    model.train()
    running_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)  # Accumulate weighted loss
    return running_loss / len(loader.dataset)

# Evaluate model on validation set and return loss and accuracy
def evaluate(model, device, loader):
    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    return loss, accuracy

# Define the hyperparameter grid
fine_tune_options = [False, True]
hidden_units_options = [50, 100, 200]
optimizer_options = ["SGD", "Adam"]
augmentation_options = [False, True]
epochs_options = [10, 20]
batch_size_options = [10, 20]

# Cartesian product: all possible combinations of hyperparameters
param_grid = list(itertools.product(
    fine_tune_options, hidden_units_options, optimizer_options,
    augmentation_options, epochs_options, batch_size_options
))

results = []
experiment_num = 1

print(f"Starting experiments over {len(param_grid)} configurations...")

# Iterate over every experiment configuration
for (fine_tune, hidden_units, opt_name, augmentation, num_epochs, batch_size) in param_grid:
    transform_choice = get_transform(augmentation)
    train_loader, val_loader = get_greek_dataloaders(transform_choice, batch_size)

    model = GreekCNN(hidden_units=hidden_units).to(device)

    # Load pre-trained weights only when hidden_units = 50 (to match dimensions)
    if hidden_units == 50:
        try:
            model = load_pretrained(model)
        except Exception as e:
            print("Could not load pre-trained weights:", e)

    model = freeze_model(model, fine_tune)

    # Choose optimizer based on configuration
    if opt_name == "SGD":
        lr = 0.01
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    else:
        lr = 0.001
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Decrease learning rate after every 5 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train and evaluate the model
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, device, train_loader, optimizer)
        val_loss, val_accuracy = evaluate(model, device, val_loader)
        scheduler.step()
    elapsed_time = time.time() - start_time

    # Save results for this configuration
    results.append({
        'fine_tune': fine_tune,
        'hidden_units': hidden_units,
        'optimizer': opt_name,
        'augmentation': augmentation,
        'epochs': num_epochs,
        'batch_size': batch_size,
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'training_time': elapsed_time
    })

    print(f"Experiment {experiment_num}: fine_tune={fine_tune}, hidden_units={hidden_units}, "
          f"optimizer={opt_name}, augmentation={augmentation}, epochs={num_epochs}, "
          f"batch_size={batch_size} => Val Acc={val_accuracy:.2f}%, Time={elapsed_time:.2f}s")
    experiment_num += 1

# Convert results to DataFrame and save to CSV
df = pd.DataFrame(results)
print("\nSummary of Experiments:")
print(df)
df.to_csv("greek_letter_experiments_results.csv", index=False)

# Plot validation accuracy across different configurations
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=df,
    x="batch_size",
    y="val_accuracy",
    hue="hidden_units",
    style="optimizer",
    size="epochs"
)
plt.xlabel("Batch Size")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy Across Configurations")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

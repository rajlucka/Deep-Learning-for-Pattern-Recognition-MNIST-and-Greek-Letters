#!/usr/bin/env python3
# Name: Raj Lucka
"""
Task 3: Fine-tune a pre-trained MNIST CNN for classifying Greek letters: alpha, beta, and gamma.

This script:
1. Defines a custom transform to convert RGB Greek images into MNIST-style inputs.
2. Loads the dataset using ImageFolder with subfolders for each class.
3. Loads a pre-trained MNIST model and freezes all layers.
4. Replaces the final fully connected layer to classify 3 classes.
5. Trains only the new output layer on the Greek letter dataset.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Custom image transform for Greek letter classification
class GreekTransform:
    def __call__(self, x):
        # Convert the input RGB image (3 channels) to grayscale (1 channel)
        x = TF.rgb_to_grayscale(x)

        # Apply an affine transformation to scale down the size of the character,
        # mimicking the scale of MNIST digits
        x = TF.affine(
            x,
            angle=0,                      # No rotation
            translate=(0, 0),             # No translation
            scale=36/128,                 # Scale image down to ~MNIST digit size
            shear=0                       # No shear
        )

        # Center-crop the scaled image to 28x28 (MNIST input size)
        x = TF.center_crop(x, (28, 28))

        # Invert intensities so that the character is white on black background,
        # matching the MNIST style
        return TF.invert(x)

# Path to the training dataset folder (with subfolders: alpha/, beta/, gamma/)
training_set_path = r"C:\Users\rajlu\OneDrive\Desktop\Work\Pattern Recognition\Projects\Video\Project1\Project1\greek_train"

# Compose full transformation pipeline
greek_transform = T.Compose([
    T.ToTensor(),                      # Convert PIL image to tensor [0,1]
    GreekTransform(),                 # Apply custom scaling/cropping/inversion
    T.Normalize((0.1307,), (0.3081,)) # Normalize using MNIST stats
])

# Load Greek letter dataset using ImageFolder (labelled automatically from subfolders)
greek_dataset = torchvision.datasets.ImageFolder(
    root=training_set_path,
    transform=greek_transform
)

# Wrap the dataset in a DataLoader for mini-batch training
greek_train = DataLoader(greek_dataset, batch_size=5, shuffle=True)

# Define the CNN model architecture (same as original MNIST model)
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 10 filters of size 5x5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Second convolutional layer: 10 input channels, 20 output channels
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer: input = flattened feature map, output = 50 hidden units
        self.fc1 = nn.Linear(320, 50)  # 20 channels * 4 * 4 from pooled conv output
        self.fc2 = nn.Linear(50, 10)   # Original MNIST had 10 output classes

    def forward(self, x):
        # First convolution -> pool -> ReLU
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)

        # Second convolution -> dropout -> pool -> ReLU
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(x)

        # Flatten the 4D tensor (batch, channels, height, width) to 2D (batch, features)
        x = x.view(x.size(0), -1)

        # Pass through first fully connected layer and activate
        x = self.fc1(x)
        x = F.relu(x)

        # Final classification layer (to be replaced for Greek classification)
        x = self.fc2(x)

        # Apply log-softmax for use with NLLLoss
        return F.log_softmax(x, dim=1)

# Load pre-trained MNIST model and adapt it to Greek classification
def prepare_model():
    model = MyNetwork()

    # Load trained MNIST weights from file (must match architecture)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
    model.eval()  # Set to evaluation mode before freezing

    # Freeze all layers so only the new output layer gets trained
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer (10 classes) with one for 3 Greek letters
    model.fc2 = nn.Linear(50, 3)  # Output classes: alpha, beta, gamma

    return model

# Train only the new final layer using Greek dataset
def train_greek_model(model, device, train_loader, epochs=5):
    model.train()  # Enable training mode

    # Use SGD optimizer only on the final layer's parameters
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.NLLLoss()  # Negative log-likelihood loss for log_softmax output

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Move data to CPU or GPU
            images, labels = images.to(device), labels.to(device)

            # Reset gradients from previous step
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and weight update
            loss.backward()
            optimizer.step()

            # Accumulate loss and correct predictions for accuracy
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Compute average loss and accuracy for this epoch
        epoch_loss = running_loss / total
        accuracy = 100. * correct / total
        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.2f}%")

    return model

# Entry point of the script
def main(argv):
    # Automatically select GPU if available, else fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Prepare the modified model (pre-trained and reconfigured)
    model = prepare_model()
    model.to(device)

    # Train model on the Greek dataset (only final layer is updated)
    print("Training on Greek letters dataset...")
    model = train_greek_model(model, device, greek_train, epochs=5)

    # Save the fine-tuned model to disk
    torch.save(model.state_dict(), "greek_letters_model.pth")
    print("Training complete. Model saved as 'greek_letters_model.pth'.")

# Trigger training when script is run directly
if __name__ == "__main__":
    main(sys.argv)

#!/usr/bin/env python3
# Name: Raj Lucka
# Task 3 (continued): Test the trained Greek letters model on your own images

import sys
import os
import glob
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

class GreekTransform:
    """
    Custom transformation pipeline for processing handwritten Greek letter images.

    Steps:
      - Convert RGB to grayscale
      - Apply affine scaling to adjust letter size
      - Center-crop to 28x28
      - Invert intensities to match MNIST-style input (black letters on white)
    """
    def __init__(self):
        pass

    def __call__(self, x):
        x = TF.rgb_to_grayscale(x)                                # Convert 3-channel RGB to single-channel grayscale
        x = TF.affine(x, angle=0, translate=(0, 0), scale=36/128, shear=0)  # Scale down proportionally to 28x28 crop size
        x = TF.center_crop(x, (28, 28))                            # Crop to final 28x28 shape
        return TF.invert(x)                                        # Invert so letter is black on white, like MNIST

# Compose the full transformation: convert to tensor, apply GreekTransform, normalize
transform = T.Compose([
    T.ToTensor(),                                                 # Convert to tensor [C, H, W]
    GreekTransform(),                                             # Custom affine + crop + invert
    T.Normalize((0.1307,), (0.3081,))                              # Normalize using MNIST mean and std
])

class MyNetwork(torch.nn.Module):
    """
    Convolutional Neural Network for Greek letter classification.
    Architecture is based on the MNIST model with the final layer adapted for 3 classes.
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)        # Conv1: 10 filters of size 5x5
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)       # Conv2: 20 filters of size 5x5
        self.dropout = torch.nn.Dropout(0.3)                      # Dropout with 30% rate
        self.pool = torch.nn.MaxPool2d(2, 2)                      # 2x2 Max pooling
        self.fc1 = torch.nn.Linear(320, 50)                       # Fully connected layer: 320 → 50
        self.fc2 = torch.nn.Linear(50, 3)                         # Final output layer: 50 → 3 (for alpha, beta, gamma)

    def forward(self, x):
        x = self.conv1(x)                                         # Apply first convolution
        x = self.pool(x)                                          # Downsample via max pool
        x = torch.nn.functional.relu(x)                           # Apply ReLU activation

        x = self.conv2(x)                                         # Second convolution
        x = self.dropout(x)                                       # Apply dropout
        x = self.pool(x)                                          # Max pool again
        x = torch.nn.functional.relu(x)                           # ReLU activation

        x = x.view(x.size(0), -1)                                 # Flatten for fully connected layer
        x = self.fc1(x)                                           # First FC layer
        x = torch.nn.functional.relu(x)                           # ReLU activation
        x = self.fc2(x)                                           # Final layer (logits for 3 classes)
        return torch.nn.functional.log_softmax(x, dim=1)          # Log-probabilities (used with NLLLoss)

def main(argv):
    # Load the trained model from disk
    model = MyNetwork()
    model.load_state_dict(torch.load("greek_letters_model.pth", map_location="cpu"))
    model.eval()  # Set to evaluation mode (disables dropout, etc.)

    # Directory containing your handwritten Greek letter images
    images_path = r"C:\Users\rajlu\OneDrive\Desktop\Work\Pattern Recognition\Projects\Video\Project1\Project1\own_greek_letters"
    image_files = glob.glob(os.path.join(images_path, "*.jpg"))  # Grab all .jpg images in that folder

    # Class mapping for output indices
    classes = ["alpha", "beta", "gamma"]

    # Loop through each image in the folder
    for img_file in image_files:
        image = Image.open(img_file).convert("RGB")               # Open and ensure RGB format
        print(f"Image '{os.path.basename(img_file)}' original size: {image.size}")

        # Resize to 128x128 if not already that size
        if image.size != (128, 128):
            image = image.resize((128, 128))                      # Resize to expected size before transform
            print(f"Resized image '{os.path.basename(img_file)}' to: {image.size}")

        # Apply transformation and add batch dimension for model input
        input_tensor = transform(image).unsqueeze(0)              # Shape becomes [1, 1, 28, 28]

        # Inference (no gradient tracking needed)
        with torch.no_grad():
            output = model(input_tensor)                          # Forward pass through model
            pred = output.argmax(dim=1).item()                    # Get predicted class index

        predicted_class = classes[pred]                           # Map index to class name
        print(f"Image '{os.path.basename(img_file)}' predicted as: {predicted_class}")

        # Display the image with its predicted class
        plt.imshow(image)                                         # Show original (resized) image
        plt.title(f"Predicted: {predicted_class}")                # Add prediction as title
        plt.axis("off")                                           # Hide axis ticks
        plt.show()

if __name__ == "__main__":
    main(sys.argv)

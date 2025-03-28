#!/usr/bin/env python3
# Name: Raj Lucka
# Task: Display the first six images from the MNIST test dataset in a 2x3 grid using matplotlib

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def show_first_six_mnist():
    """
    Loads the MNIST test dataset and visualizes the first six digit images.
    
    The images are displayed in a 2x3 grid along with their corresponding labels.
    """
    # Define transformation to convert images to PyTorch tensors (0–1 range)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the MNIST test dataset with transformations applied
    # Set train=False to load the test split (not training data)
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )

    # Set up a 2-row, 3-column grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    axes = axes.flatten()  # Flatten for easy iteration (from 2D array to list)

    # Display the first six images (indices 0 to 5)
    for i in range(6):
        image, label = test_dataset[i]   # Get image and corresponding digit label
        image_np = image.squeeze()       # Remove channel dimension (1x28x28 → 28x28)
        axes[i].imshow(image_np, cmap='gray')  # Display in grayscale
        axes[i].set_title(f"Label: {label}")   # Set subplot title
        axes[i].axis('off')              # Hide axis ticks

    # Automatically adjust subplot spacing for better layout
    plt.tight_layout()
    plt.show()

# Run the function if script is executed directly
if __name__ == "__main__":
    show_first_six_mnist()

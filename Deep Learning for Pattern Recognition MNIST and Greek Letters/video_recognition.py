#!/usr/bin/env python3
# Name: Raj Lucka
# Task: Real-time digit recognition from webcam input using a pre-trained CNN on MNIST

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

class CNN(nn.Module):
    """
    Convolutional Neural Network architecture trained on MNIST.
    Assumes 28x28 grayscale input images with no padding.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)       # First convolutional layer: 10 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)      # Second convolutional layer: 20 filters
        self.pool = nn.MaxPool2d(2)                        # Max pooling with 2x2 kernel
        self.dropout1 = nn.Dropout(0.25)                   # Dropout layer to prevent overfitting
        self.fc1 = nn.Linear(320, 50)                      # Fully connected layer: 20*4*4 = 320 inputs
        self.dropout2 = nn.Dropout(0.5)                    # Additional dropout before final layer
        self.fc2 = nn.Linear(50, 10)                       # Output layer: 10 classes for digits

    def forward(self, x):
        x = F.relu(self.conv1(x))                          # Conv1 → ReLU
        x = self.pool(x)                                   # Max pooling
        x = F.relu(self.conv2(x))                          # Conv2 → ReLU
        x = self.pool(x)                                   # Max pooling
        x = torch.flatten(x, 1)                            # Flatten to vector shape [batch_size, 320]
        x = self.dropout1(x)                               # Apply dropout
        x = F.relu(self.fc1(x))                            # Fully connected layer → ReLU
        x = self.dropout2(x)                               # Dropout again
        x = self.fc2(x)                                    # Output logits
        return F.log_softmax(x, dim=1)                     # Log-softmax for classification

# Load the pre-trained model weights from file
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=torch.device("cpu")))
model.eval()                                              # Set model to evaluation mode

# Define transformation: resize to 28x28, grayscale, normalize like MNIST
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Background subtractor to detect moving foreground (handwriting)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False
)

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction to isolate moving object (the digit)
    fg_mask = bg_subtractor.apply(gray)

    # Clean up the foreground mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours in the cleaned mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 5000:                                   # Ignore very small or large blobs
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 10                                             # Add padding around detected region
            digit_boxes.append((x - pad, y - pad, w + 2 * pad, h + 2 * pad))

    # Sort boxes from left to right (in case of multiple digits)
    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

    for (x, y, w, h) in digit_boxes:
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        # Crop region of interest from grayscale image
        roi = gray[y:y+h, x:x+w]

        # Invert colors: white digit on black background (MNIST style)
        roi = cv2.bitwise_not(roi)

        # Optional: smooth using Gaussian blur
        roi = cv2.GaussianBlur(roi, (3, 3), 0)

        try:
            # Apply transform to prepare image for model input
            input_tensor = transform(roi).unsqueeze(0)         # Add batch dimension → shape: [1, 1, 28, 28]
        except Exception as e:
            print("Error during transform:", e)
            continue

        # Predict digit using model
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()

        # Draw rectangle around digit and display prediction
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(pred), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

    # Show foreground mask and live prediction window
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Live Digit Recognition", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

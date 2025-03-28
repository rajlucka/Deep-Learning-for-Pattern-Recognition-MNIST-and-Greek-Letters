# Deep Learning for Pattern Recognition: MNIST and Greek Letters

This repository contains code for a master's project on deep learning for pattern recognition. It includes:

- **MNIST CNN Training:** A convolutional neural network (CNN) is built and trained on the MNIST dataset for handwritten digit recognition, with training and testing loss curves plotted to evaluate performance.
- **Transfer Learning for Greek Letters:** The pre-trained MNIST network is adapted for classifying Greek letters (alpha, beta, gamma) by freezing the initial layers and replacing the final fully connected layer.
- **Hyperparameter Experiments:** Automated experiments evaluate multiple dimensions such as fine-tuning strategies, hidden layer sizes, optimizer choices, data augmentation, number of epochs, and batch sizes to optimize network performance.
- **Live Video Digit Recognition:** A real-time application uses OpenCV to capture video, processes regions of interest (ROI) from the webcam, and uses the trained CNN to recognize digits on the fly.

## Libraries Used
- **Python 3:** Programming language.
- **PyTorch & Torchvision:** For building, training, and evaluating neural networks and handling datasets.
- **OpenCV:** For live video capture and image processing.
- **Matplotlib & Seaborn:** For plotting and visualizing training curves and experiment results.
- **Graphviz (optional):** For generating network architecture diagrams.

## Usage
1. **MNIST Training:**  
   Run `mnist_training.py` to train the CNN on MNIST and view loss curves.
2. **Greek Letter Experiments:**  
   Run `greek_letter_experiments.py` to adapt the network for Greek letter classification and perform hyperparameter evaluations.
3. **Live Video Recognition:**  
   Run `video_recognition.py` to launch the live digit recognition application.

## Environment
- Operating System: Windows 10/11  
- IDE: Visual Studio Code (or your preferred Python IDE)
- Ensure the required libraries are installed via pip.

This project demonstrates network design, transfer learning, hyperparameter tuning, and real-time inference, providing valuable insights into deep learning for pattern recognition.

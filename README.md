📘 MNIST CNN Classifier

A simple and efficient Convolutional Neural Network (CNN) built using TensorFlow/Keras to classify handwritten digits (0–9) from the popular MNIST dataset.

This project is perfect for beginners learning Deep Learning, and it demonstrates a complete end-to-end pipeline — from loading data to training, evaluation, and prediction.

🚀 Project Overview

This project covers:

Loading & preprocessing image data

Building a CNN architecture

Training and validating the model

Evaluating accuracy

Predicting digits on test images

Visualizing sample predictions

The model achieves ~99% accuracy using a lightweight CNN.

🧠 What is MNIST?

The MNIST dataset contains:

70,000 images of handwritten digits

Image size: 28 × 28 pixels

Grayscale (1 channel)

Clean, labeled data for digits 0–9

Widely known as the “Hello World of Deep Learning”

MNIST is the best dataset to understand CNN fundamentals.

🏗️ Model Architecture

The CNN architecture used:

Input: 28 × 28 × 1 (grayscale image)

Conv2D (32 filters, 3×3) + ReLU
MaxPooling2D (2×2)

Conv2D (64 filters, 3×3) + ReLU
MaxPooling2D (2×2)

Flatten
Dense (64 units) + ReLU
Dense (10 units) + Softmax


This network is lightweight, fast to train, and performs exceptionally well on MNIST.

📦 Installation

Install the necessary dependencies:

pip install tensorflow matplotlib numpy

▶️ How to Run

Run the Python file:

python cnn_mnist.py


What this script does:

Loads MNIST data

Builds the CNN

Trains for 5 epochs

Evaluates on test data

Shows test accuracy

Displays a sample image

Predicts the digit

📊 Results

Typical output:

Test Accuracy: 99.12%
Predicted digit: 7


The model learns to classify digits with high confidence.

🖼️ Sample Digits

Sample MNIST digits:

🧩 Project Structure
mnist-cnn-classifier/
│
├── cnn_mnist.py        # Main model training + prediction script
└── README.md           # Project documentation

💡 What You Learn from This Project

How CNNs work

Convolutions, pooling, flattening, softmax

How to preprocess image datasets

How to train and evaluate ML models

How to structure ML code

How to visualize predictions

A strong foundation before moving to larger datasets like CIFAR-10 or real-world image classification tasks.

🚀 Future Improvements

You can extend this project by adding:

✔️ Confusion matrix

✔️ Accuracy/loss visualization

✔️ Dropout for regularization

✔️ Batch Normalization

✔️ Saving/loading model (.h5)

✔️ Training on larger datasets

✔️ Deploying as a web app

If you want, I can help you implement any of these.

🤝 Contributions

Feel free to open issues or submit pull requests to improve the model or add new features.

⭐ Show Support

If you found this project helpful, please consider starring the repository ✨

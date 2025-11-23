📘 MNIST CNN Classifier

A simple and efficient Convolutional Neural Network (CNN) built using TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset.
This project is perfect for learning image classification and understanding how CNNs work.

🚀 Project Overview

This project demonstrates:

How to load and preprocess image datasets

How to build CNN architectures

How to train models and evaluate accuracy

How to make predictions on new images

Basic end-to-end Deep Learning workflow

The model achieves ~99% accuracy on MNIST using just a few layers.

🧠 What is MNIST?

MNIST is a classic dataset containing:

70,000 handwritten digit images

Image size: 28×28 pixels

Grayscale (1 channel)

Clean, labeled images from 0 to 9

It’s widely known as the "Hello World of Deep Learning".

🏗️ Model Architecture

The CNN used in this project:

Conv2D (32 filters, 3×3) + ReLU  
MaxPooling2D (2×2)

Conv2D (64 filters, 3×3) + ReLU  
MaxPooling2D (2×2)

Flatten  
Dense (64 units) + ReLU  
Dense (10 units) + Softmax


This simple architecture is enough to reach high accuracy with fast training.

📦 Installation

Install dependencies:

pip install tensorflow numpy matplotlib

▶️ How to Run

Run the Python file:

python cnn_mnist.py


This will:

Train the CNN

Evaluate on test data

Show accuracy

Display a test image

Predict the digit

📊 Results

A typical output:

Test Accuracy: 99.12%
Predicted digit: 4

🖼️ Sample Prediction

The model takes a test image like:

And predicts the correct digit with high confidence.

🧩 Project Structure
mnist-cnn-classifier/
│
├── cnn_mnist.py        # Main training + prediction file
├── README.md           # Project documentation
└── requirements.txt    # (Optional) dependencies

💡 What You Learn from This Project

How CNNs work

How to use Conv2D, MaxPooling, Dense layers

How to normalize image data

How to train a deep learning model

How to evaluate and predict results

How to build a real ML project end-to-end

📈 Future Improvements

You can expand this project by adding:

Confusion Matrix

Model accuracy/loss graphs

Dropout layers

Batch Normalization

Saving/loading the model

Trying bigger datasets like CIFAR-10

🤝 Contributions

Pull requests are welcome!
Feel free to improve the model or add new features.

⭐ If you found this helpful, give the repo a star! ⭐
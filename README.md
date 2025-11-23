# 🧠 MNIST CNN Classifier

A simple and efficient **Convolutional Neural Network (CNN)** built using **TensorFlow/Keras** to classify handwritten digits (0–9) from the popular MNIST dataset.

This project is perfect for beginners learning Deep Learning.  
It demonstrates a complete end-to-end pipeline — from loading data to training, evaluation, and prediction.

---

## 🚀 Project Overview

This project includes:

- 🔄 Loading & preprocessing image data  
- 🧱 Building a CNN architecture  
- 🎓 Training and validating the model  
- 📊 Evaluating accuracy  
- 🔢 Predicting digits on test images  

---

## ✨ Features

- 🧠 Lightweight CNN with ~99% accuracy  
- 🖼 Handles grayscale image preprocessing  
- ⚡ Fast training (no GPU required)  
- 📈 Shows model accuracy  
- 🧪 Predicts digits on unseen images  
- 🧩 Great starter deep learning project  

---

## 📦 Installation

Install dependencies:
▶️ How to Run
python cnn_mnist.py


This will:

Load the MNIST dataset

Build the CNN

Train the model

Evaluate accuracy

Display a sample test image

Predict the digit

🏗️ Model Architecture
Input: 28 × 28 × 1

Conv2D (32 filters, 3×3) + ReLU  
MaxPooling2D (2×2)

Conv2D (64 filters, 3×3) + ReLU  
MaxPooling2D (2×2)

Flatten  
Dense (64 units) + ReLU  
Dense (10 units) + Softmax

📊 Sample Output
Test Accuracy: 99.12%
Predicted digit: 4

🖼 Sample MNIST Digits

📁 Project Structure
mnist-cnn-classifier/
│
├── cnn_mnist.py        # Main CNN training & prediction script
└── README.md           # Project documentation

🧠 Technologies Used

TensorFlow / Keras

NumPy

Matplotlib

Python

🚀 Future Enhancements

📉 Add accuracy & loss visualization

🧪 Add confusion matrix

🧱 Add BatchNorm / Dropout

💾 Save & load model

⚡ Train deeper CNN models

🌐 Deploy using Streamlit or Flask

🤝 Contributing

Contributions are welcome!

Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a pull request

👨‍💻 Author

Ketan
GitHub: @ketann13

```bash
pip install tensorflow numpy matplotlib

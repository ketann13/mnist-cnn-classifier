# 🧠 MNIST CNN Classifier

A simple and efficient **Convolutional Neural Network (CNN)** built using **TensorFlow/Keras** to classify handwritten digits (0–9) from the popular MNIST dataset.

This project is ideal for beginners in Deep Learning and demonstrates a complete end-to-end machine learning workflow — from loading data to training, evaluation, and prediction.

---

## 🚀 Project Overview

This project covers the following steps:

- 🔄 Loading & preprocessing the MNIST dataset  
- 🧱 Building a CNN architecture  
- 🎓 Training and validating the model  
- 📊 Checking model accuracy  
- 🔢 Predicting digits on new test images  

---

## ✨ Features

- 🧠 Lightweight CNN achieving **~99% accuracy**  
- 🖼 Handles grayscale image preprocessing  
- ⚡ Fast training (CPU-friendly, no GPU required)  
- 📈 Shows evaluation metrics  
- 🧪 Makes predictions on unseen MNIST digits  
- 🧩 Great starter project for ML and DL portfolios  

---

## 📦 Installation

Install the required dependencies:

```bash
pip install tensorflow numpy matplotlib
```

---

## ▶️ How to Run the Project

Run the script:

python cnn_mnist.py

The script will:

Load MNIST data

Preprocess images

Build the CNN

Train the model

Evaluate the model

Display a test digit

Predict the digit

---

## 🏗️ Model Architecture
Input: 28 × 28 × 1

Conv2D (32 filters, 3×3) + ReLU  
MaxPooling2D (2×2)

Conv2D (64 filters, 3×3) + ReLU  
MaxPooling2D (2×2)

Flatten  
Dense (64 units) + ReLU  
Dense (10 units) + Softmax


This simple architecture performs exceptionally well on MNIST with minimal computation.

---

## 📊 Sample Output
Test Accuracy: 99.12%
Predicted Digit: 7

---

## 📁 Project Structure
mnist-cnn-classifier/
│
├── cnn_mnist.py        # Main model training & prediction script
└── README.md           # Project documentation

---

## 🧠 Technologies Used

🐍 Python

🔶 TensorFlow / Keras

📘 NumPy

🎨 Matplotlib

---

 ## 🚀 Future Enhancements

Potential improvements:

📉 Plot training accuracy & loss curves

🧪 Add confusion matrix

🧱 Add dropout & batch normalization

💾 Save/load trained models (model.h5)

🌐 Deploy with Streamlit or Flask

⚡ Experiment with deeper CNNs or transfer learning

---

 ## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repository and submit a pull request.

 project, please star ⭐ the repository — it helps a lot!

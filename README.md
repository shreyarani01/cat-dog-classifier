# 🐾 Cat vs. Dog Image Classifier

An end-to-end Computer Vision web application built with **PyTorch** and **Streamlit** that classifies uploaded images into **Cat** or **Dog** categories in real-time with confidence scoring.

---

## 📌 Project Overview
* **Model Architecture:** Custom modified AlexNet Convolutional Neural Network (CNN).
* **Frameworks & Tools:** PyTorch, torchvision, Streamlit.
* **Input Spatial Resolution:** $128 \times 128$ RGB images.
* **Key Features:** Real-time inference, image preprocessing pipelines, and dynamic confidence score visualization.

---

## ⚙️ Model Architecture & Preprocessing

1. **Preprocessing Pipeline:**
   * Images are resized to $128 \times 128$ pixels.
   * Converted to PyTorch Tensors (`ToTensor()`).
   * Standardized using ImageNet channel mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).

2. **Feature Extractor (`features`):**
   * AlexNet backbone with 5 Convolutional layers, ReLU activations, and Max-Pooling blocks.

3. **Classifier Head (`classifier`):**
   * Dense classification network consisting of custom Dropout and Linear layers mapping extracted features to 2 output classes.

---

## 🛠️ Local Setup & Running

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/shreyarani01/cat-dog-classifier.git](https://github.com/shreyarani01/cat-dog-classifier.git)
   cd cat-dog-classifier

## 📸 Application Demo

<img width="1440" height="900" alt="Screenshot 2026-10-24 at 9 59 59 PM" src="https://github.com/user-attachments/assets/3ab83605-ea63-4291-ade9-6efc0a47e581" />

<img width="1440" height="900" alt="Screenshot 2026-10-24 at 10 00 07 PM" src="https://github.com/user-attachments/assets/c87f277a-7253-4434-9e11-286452f085dc" />



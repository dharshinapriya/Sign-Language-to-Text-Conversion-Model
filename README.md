# Sign-Language-to-Text-Conversion-Model
Sign Language to Text Conversion Model, Break the Barriers Of Communication

A deep learning-based real-time American Sign Language (ASL) recognition system that converts hand gestures captured via a webcam into readable English text. The model achieves high accuracy without the need for specialized hardware.

---

## 📅 Duration
**December 2024 – April 2025**

---

## 📌 Key Features
- 🔤 Translates ASL alphabet (A–Z) gestures to English text in real time.
- 📷 Works with standard webcams using OpenCV—no special sensors needed.
- 🧠 Achieved **98.0% model accuracy** using a custom CNN architecture.
- 📈 Enhanced model precision by **30%** via custom data augmentation.
- 🖥️ GUI powered by Tkinter for ease of use and real-time feedback.
- ✅ Integrated Hunspell for improved spelling correction in output text.

---

## 🧰 Tech Stack
- **Programming Language**: Python  
- **Libraries/Tools**: OpenCV, NumPy, Hunspell, TensorFlow, Keras, Tkinter  
- **IDE**: Visual Studio Code  

---

## 🧠 Model Architecture
- Multi-layer Convolutional Neural Network (CNN)
- Activation: ReLU and Softmax
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Accuracy: **98.0%** on validation set

---

## 🗂️ Dataset
- **Custom-built dataset** with **26,000+ images** of ASL gestures (A-Z)
- Captured using OpenCV and standard webcam
- Preprocessing: grayscale, normalization, resizing, and augmentation

---

## 💻 How to Run the Project

### 🔧 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sign-language-to-text.git
   cd sign-language-to-text

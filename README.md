# 🧥 Fashion-MNIST Image Classification (ANN + Streamlit)

This project is an end-to-end **Image Classification** pipeline built using a **Dense Neural Network (ANN)** trained on the **Fashion-MNIST dataset**.  
It also includes a **Streamlit web app** for real-time predictions with probability visualization.

---

## 📌 Project Highlights
✅ Built and trained a Dense Neural Network (ANN) using TensorFlow/Keras  
✅ Fashion-MNIST dataset (10 clothing categories, 28×28 grayscale images)  
✅ Model evaluation using accuracy and class probabilities  
✅ Deployed as an interactive Streamlit web application  
✅ Includes a **Random Fashion-MNIST Sample button** for correct demo predictions

---

## 📂 Dataset Info (Fashion-MNIST)
Fashion-MNIST consists of **70,000 grayscale clothing images** of size **28×28** across **10 classes**:

0. T-shirt/top  
1. Trouser  
2. Pullover  
3. Dress  
4. Coat  
5. Sandal  
6. Shirt  
7. Sneaker  
8. Bag  
9. Ankle boot  

---

## 🧠 Model Architecture (ANN)
- Input: 28×28 image → Flattened into 784 features  
- Hidden Layers: Dense layers with ReLU + Dropout  
- Output: Dense(10) with Softmax  

✅ Dense networks work well on Fashion-MNIST, but they lose spatial features due to flattening, which is why CNNs usually perform better.

---

## 🚀 Streamlit App Features
The Streamlit app supports:

✅ Image upload (PNG/JPG/JPEG)  
✅ Converts image → 28×28 grayscale → normalized → flattened  
✅ Shows **Predicted class + Confidence score**  
✅ Shows **Top-3 predictions**  
✅ Displays **Class probability chart**  
✅ Random Fashion-MNIST sample demo button  

⚠️ **Note:** This model is trained on Fashion-MNIST images, so real-world clothing photos may produce incorrect predictions due to dataset mismatch.

---

https://image-classification-nidhim-soni.streamlit.app/

click on the link to use the app



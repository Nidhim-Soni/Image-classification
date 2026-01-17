import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Fashion-MNIST Classifier",
    page_icon="🧥",
    layout="wide"
)

st.title("🧥 Fashion-MNIST Image Classifier (ANN)")
st.caption("Built using TensorFlow/Keras + Streamlit | Dense Neural Network on Fashion-MNIST")

st.warning(
    "⚠️ Note: This model is trained on Fashion-MNIST (28×28 grayscale clothing images). "
    "Real-world photos may produce incorrect predictions."
)

# -------------------------
# Label Names
# -------------------------
label_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_fashion_model():
    # ✅ Use .keras model format (recommended)
    return keras.models.load_model("fashion_mnist_ann.keras", compile=False)

model = load_fashion_model()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("⚙️ Settings")
invert = st.sidebar.checkbox("Invert Colors", value=False)
show_prob_table = st.sidebar.checkbox("Show Probability Table", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("✅ **Tip:** If predictions look wrong, try enabling **Invert Colors**.")

# -------------------------
# Main UI Layout
# -------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Upload an image (PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    st.markdown("✅ Recommended: Use Fashion-MNIST-like images (simple clothing on plain background).")

with col2:
    st.subheader("📌 Model Info")
    st.write("**Model Type:** Dense Neural Network (ANN)")
    st.write("**Input Format:** 28×28 grayscale → Flattened to 784")
    st.write("**Output:** 10 Fashion Classes")

# -------------------------
# Prediction Logic
# -------------------------
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("🖼️ Image Preview + Prediction")

    # Read image
    img = Image.open(uploaded_file).convert("L")  # grayscale
    img_resized = img.resize((28, 28))

    img_array = np.array(img_resized) / 255.0

    if invert:
        img_array = 1 - img_array

    # Prepare for ANN (Flatten)
    img_final = img_array.reshape(1, 784)

    # Predict
    pred_prob = model.predict(img_final, verbose=0)[0]  # shape (10,)
    pred_class = int(np.argmax(pred_prob))
    confidence = float(np.max(pred_prob))

    # Top 3 predictions
    top3_idx = np.argsort(pred_prob)[::-1][:3]
    top3 = [(label_names[i], float(pred_prob[i])) for i in top3_idx]

    # Layout
    left, right = st.columns([1, 1])

    with left:
        st.image(img, caption="Uploaded Image", use_container_width=True)

        st.subheader("✅ Prediction Result")
        st.success(f"Predicted Class: **{label_names[pred_class]}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        st.subheader("🏆 Top-3 Predictions")
        for name, prob in top3:
            st.write(f"**{name}** → `{prob:.4f}`")

    with right:
        st.subheader("📊 Class Probability Chart")

        prob_df = pd.DataFrame({
            "Class": label_names,
            "Probability": pred_prob
        }).sort_values(by="Probability", ascending=False)

        # Bar chart (Streamlit)
        st.bar_chart(prob_df.set_index("Class"))

        if show_prob_table:
            st.subheader("📋 Probability Table")
            st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)

else:
    st.info("👆 Upload an image to get prediction.")

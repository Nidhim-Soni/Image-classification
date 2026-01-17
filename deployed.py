import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow import keras

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Fashion-MNIST Classifier",
    page_icon="🧥",
    layout="wide"
)

# -------------------------
# Title
# -------------------------
st.title("🧥 Fashion-MNIST Image Classifier (ANN)")
st.caption("Built using TensorFlow/Keras + Streamlit | Dense Neural Network on Fashion-MNIST")

st.warning(
    "⚠️ Note: This model is trained on Fashion-MNIST (28×28 grayscale clothing silhouettes). "
    "Real-world photos (from Google/camera) may give incorrect predictions due to dataset mismatch."
)

# -------------------------
# Label Names (Fashion-MNIST)
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
    # Recommended format: .keras
    return keras.models.load_model("fashion_mnist_ann.keras", compile=False)

model = load_fashion_model()

# -------------------------
# Sidebar Settings
# -------------------------
st.sidebar.header("⚙️ Settings")
invert_manual = st.sidebar.checkbox("Manual Invert Colors", value=False)
use_threshold = st.sidebar.checkbox("Apply Thresholding (Fashion-like)", value=True)
show_prob_table = st.sidebar.checkbox("Show Probability Table", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("✅ Tip: Use the **Random Fashion-MNIST Sample** button for best demo results.")

# -------------------------
# Helper Function
# -------------------------
def preprocess_image(img_pil, invert=False, threshold=True):
    """
    Converts input image to Fashion-MNIST-like format:
    - grayscale
    - resize to 28x28
    - normalize
    - optional thresholding
    - auto invert if needed
    """
    img_gray = img_pil.convert("L")
    img_resized = img_gray.resize((28, 28))

    img_array = np.array(img_resized).astype("float32") / 255.0

    # Thresholding makes it more similar to dataset silhouettes
    if threshold:
        img_array = (img_array > 0.5).astype("float32")

    # Auto-invert if background looks white (common in real images)
    if img_array.mean() > 0.5:
        img_array = 1 - img_array

    # Manual invert toggle
    if invert:
        img_array = 1 - img_array

    # ANN input expects (1, 784)
    img_final = img_array.reshape(1, 784)
    return img_array, img_final

def predict_and_display(img_array_28x28, img_final_784):
    pred_prob = model.predict(img_final_784, verbose=0)[0]
    pred_class = int(np.argmax(pred_prob))
    confidence = float(np.max(pred_prob))

    top3_idx = np.argsort(pred_prob)[::-1][:3]
    top3 = [(label_names[i], float(pred_prob[i])) for i in top3_idx]

    # Show results
    st.subheader("✅ Prediction Result")
    st.success(f"Predicted Class: **{label_names[pred_class]}**")
    st.write(f"Confidence: **{confidence:.2f}**")

    st.subheader("🏆 Top-3 Predictions")
    for name, prob in top3:
        st.write(f"**{name}** → `{prob:.4f}`")

    # Probabilities dataframe
    prob_df = pd.DataFrame({
        "Class": label_names,
        "Probability": pred_prob
    }).sort_values(by="Probability", ascending=False)

    st.subheader("📊 Class Probability Chart")
    st.bar_chart(prob_df.set_index("Class"))

    if show_prob_table:
        st.subheader("📋 Probability Table")
        st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)

# -------------------------
# Layout
# -------------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("🎲 Best Demo Option (Recommended)")
    if st.button("🎯 Try a Random Fashion-MNIST Sample"):
        (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
        idx = np.random.randint(0, len(x_train))

        sample_img = x_train[idx].astype("float32") / 255.0
        sample_label = int(y_train[idx])

        st.image(sample_img, caption=f"True Label: {label_names[sample_label]}", use_container_width=True)

        sample_final = sample_img.reshape(1, 784)

        st.markdown("---")
        st.subheader("📌 Model Output (Random Sample)")
        predict_and_display(sample_img, sample_final)

with right_col:
    st.subheader("📤 Upload Your Own Image")
    uploaded_file = st.file_uploader(
        "Upload an image (PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    st.info("✅ Best results occur when the uploaded image looks similar to Fashion-MNIST samples.")

# -------------------------
# Uploaded Image Prediction
# -------------------------
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("🖼️ Uploaded Image Prediction")

    img = Image.open(uploaded_file)

    preview_col, pred_col = st.columns([1, 1])

    with preview_col:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess + Predict
    img_array_28x28, img_final_784 = preprocess_image(
        img,
        invert=invert_manual,
        threshold=use_threshold
    )

    with pred_col:
        st.image(img_array_28x28, caption="Processed 28×28 Image (Model Input)", use_container_width=True)

    st.markdown("---")
    predict_and_display(img_array_28x28, img_final_784)
else:
    st.info("👆 Upload an image OR try the Random Fashion-MNIST Sample button.")

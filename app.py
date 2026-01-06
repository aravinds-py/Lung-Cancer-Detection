import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd

# ==========================
# Settings & Model Loading
# ==========================

st.set_page_config(
    page_title="Lung Cancer Classifier",
    page_icon="🫁",
    layout="centered"
)

MODEL_PATH = "lung_classifier.keras"  
IMG_SIZE = 256
CLASS_NAMES = ['Adenocarcinoma', 'Squamous Cell Carcinoma', 'Normal']  # Adjust to your model

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def preprocess_image(image: Image.Image):
    image = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ==========================
# Sidebar Contents
# ==========================
with st.sidebar:
    st.image('.\healthcare.png', width=96)
    st.title("About")
    st.write("""
        **Lung Cancer Image Classifier**  
        Upload an image from a lung histology dataset to predict its type.
        - Supported formats: JPG, PNG
        - Model: Trained with TensorFlow/Keras
        - Categories: Adenocarcinoma, Squamous Cell Carcinoma, Normal
      
        _For educational & demo purposes only._
    """)
    st.markdown("---")
    st.write("Created with ❤️ by Aravind")

# ==========================
# Main Layout
# ==========================

st.markdown(
    "<h1 style='text-align: center; color: #319EDA;'>🫁 Lung Cancer Image Classification</h1>",
    unsafe_allow_html=True
)
st.write("Upload a lung image, and get a prediction with confidence scores.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="fileUploader")

if uploaded_file is not None:
    cols = st.columns([1,2,1])
    with cols[1]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=350, use_column_width="auto")
    
    if st.button("🔎 Predict"):
        with st.spinner("Running inference..."):
            input_img = preprocess_image(image)
            predictions = model.predict(input_img)
            idx = int(np.argmax(predictions))
            confidence = float(np.max(predictions))
            pred_label = CLASS_NAMES[idx]

            # Create a DataFrame for confidence scores
            conf_df = pd.DataFrame({"Class": CLASS_NAMES, "Confidence": predictions[0]})
            conf_df = conf_df.sort_values(by="Confidence", ascending=False)

            st.success(f"### 🩺 Prediction: {pred_label}")
            st.write(f"**Confidence:** {confidence:.2%}")

            # Custom horizontal bar chart for class probabilities
            st.markdown("#### Confidence by Class")
            st.bar_chart(
                conf_df.set_index("Class"),
                use_container_width=True,
                height=250,
            )

            st.progress(int(confidence * 100), text=f"{pred_label}: {confidence:.2%} confidence")

        st.info(
            "Model predictions are a statistical outcome and cannot replace expert medical advice. For professional use, consult a healthcare specialist!"
        )
else:
    st.warning("👈 Please upload a lung histology image to get started.")

# ==========================
# Custom Footer
# ==========================
st.markdown(
    """
    <hr style="border: 1px solid #319EDA;">
    <small>
    Made with <span style="color: #e25555;">&#10084;</span> using Streamlit & TensorFlow.<br>
    Demo only, not for clinical use.
    </small>
    """,
    unsafe_allow_html=True,
)
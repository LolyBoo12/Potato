import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model from .keras file
model_path = 'E:/Model_kopi/kopi_model.keras'
model = keras.models.load_model(model_path)

# Preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to match model input
    img_array = np.array(img.convert('RGB'))  # Convert to RGB
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize the image

# Classify the image
def classify_image(img):
    preprocessed_img = preprocess_image(img)
    prediction = model.predict(preprocessed_img)
    classes = ['Arabika', 'Excelsa', 'Robusta']
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * -100
    return predicted_class, confidence

# Streamlit app
st.title("Uji Biji Kopi")

st.write("""Upload Gambar""")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class, confidence = classify_image(img)
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Display the image with prediction
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%", fontsize=14)
    st.pyplot(fig)
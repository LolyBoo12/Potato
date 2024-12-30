import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Definisikan lapisan kustom LayerScale
class LayerScale(tf.keras.layers.Layer):
    def __init__(self, init_values=1e-6, projection_dim=128, **kwargs):
        super(LayerScale, self).__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
        self.scale = self.add_weight(
            name='scale',
            shape=(self.projection_dim,),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True
        )

    def call(self, inputs):
        return inputs * self.scale

# Muat model dengan custom_objects
model = tf.keras.models.load_model('E:/Model Kentang/kentag84%.h5', custom_objects={'LayerScale': LayerScale})

# Fungsi untuk praproses gambar
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalisasi
    return img_array

# Fungsi untuk memprediksi gambar
def predict_image(model, image, class_names=['Early Blight', 'Late Blight', 'Sehat']):
    img = preprocess_image(image)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_names[predicted_class[0]]
    confidence = predictions[0][predicted_class[0]] * 100

    return predicted_label, confidence, predictions

# Aplikasi Streamlit
st.title('Potato Leaf Disease Classification')
st.write('Unggah gambar daun kentang untuk memprediksi jenis penyakit.')

uploaded_file = st.file_uploader("Pilih gambar daun kentang...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
    st.write("Klasifikasi...")

    predicted_label, confidence, predictions = predict_image(model, image)

    # Display the image with prediction
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    plt.figtext(0.5, 0.9, f"Predicted Class: {predicted_label}\nConfidence: {confidence:.2f}%", 
                fontsize=14, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6))
    st.pyplot(fig)

    st.write('Confidence Scores:')
    for i, class_name in enumerate(['Early Blight', 'Late Blight', 'Sehat']):
        st.write(f'{class_name}: {predictions[0][i] * 100:.2f}%')

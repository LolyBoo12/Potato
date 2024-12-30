import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import time

# Fungsi untuk memuat model berdasarkan pilihan pengguna
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "MobileNet": 'E:/Model Kentang/potato_disease_model_kyknya_bagus.h5',  # Ganti path model
        "ConvNextBase": 'E:/Model Kentang/pp.keras',
        "VGG": 'E:/Model Kentang/my_modelpp_mnet64.keras'
    }
    model_path = model_paths.get(model_name)
    if model_path:
        return keras.models.load_model(model_path)
    else:
        st.error("Model tidak ditemukan.")
        return None

# Fungsi preprocessing gambar
def preprocess_image(img):
    img = img.resize((256, 256))  # Ubah ukuran gambar
    img_array = np.array(img.convert('RGB'))  # Konversi ke RGB
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
    return img_array / 255.0  # Normalisasi

# Fungsi klasifikasi gambar
def classify_image(img, model_choice):
    start_time = time.time()
    
    model = load_model(model_choice)
    if not model:
        return {"Error": "Model tidak ditemukan."}, 0.0
    
    preprocessed_img = preprocess_image(img)
    prediction = model.predict(preprocessed_img)[0]
    classes = ['Early Blight', 'Sehat', 'Late Blight']
    results = {classes[i]: float(prediction[i]) * 100 for i in range(len(classes))}
    prediction_time = time.time() - start_time

    return results, prediction_time

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Klasifikasi Penyakit Daun Kentang", layout="wide")
st.title("🥔 Klasifikasi Penyakit Daun Kentang")
st.write("Unggah gambar daun kentang untuk mendeteksi penyakitnya dengan bantuan model CNN.")

# Layout Streamlit
col1, col2 = st.columns([1, 1])

# Kolom Kiri: Upload dan Input
with col1:
    st.header("Input Gambar dan Model")
    uploaded_file = st.file_uploader("Unggah gambar daun kentang:", type=["jpg", "jpeg", "png"])
    model_choice = st.selectbox("Pilih Model CNN:", ["MobileNet", "ConvNextBase", "VGG"])
    submit_button = st.button("Submit")

# Kolom Kanan: Output Prediksi
with col2:
    st.header("Hasil Prediksi")
    if submit_button and uploaded_file:
        # Proses Gambar
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_column_width=True)
        
        # Proses Prediksi
        with st.spinner("Menganalisis gambar..."):
            results, prediction_time = classify_image(img, model_choice)
        
        # Tampilkan hasil prediksi
        st.success("Analisis selesai!")
        st.write("### Tingkat Confidence (Kepercayaan)")
        for kelas, confidence in results.items():
            st.progress(confidence / 100)
            st.write(f"**{kelas}:** {confidence:.2f}%")

        # Waktu prediksi
        st.write("### Waktu Prediksi")
        st.write(f"{prediction_time:.2f} detik")

# Footer
st.markdown("---")
st.markdown("""
**Aplikasi Deteksi Penyakit Daun Kentang**  
Dikembangkan untuk membantu petani dalam mendeteksi penyakit Early Blight dan Late Blight.  
📧 Kontak: [email@example.com](mailto:email@example.com)
""")

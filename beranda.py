import streamlit as st
import base64
import os
from pathlib import Path
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers
import cloudinary
import cloudinary.uploader
import gspread
from google.oauth2.service_account import Credentials

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(page_title="Klasifikasi Daun Tebu", layout="wide")

# ==============================
# Konfigurasi Cloudinary
# ==============================
cloudinary.config(
    cloud_name=st.secrets["cloudinary"]["cloud_name"],
    api_key=st.secrets["cloudinary"]["api_key"],
    api_secret=st.secrets["cloudinary"]["api_secret"]
)

def upload_to_cloudinary(file_path: str) -> str:
    """Upload file ke Cloudinary dan mengembalikan URL."""
    try:
        result = cloudinary.uploader.upload(file_path)
        return result["secure_url"]
    except Exception as e:
        st.error(f"Gagal upload ke Cloudinary: {e}")
        return ""

# ==============================
# Konfigurasi Google Sheets
# ==============================
scope = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=scope
)
client = gspread.authorize(credentials)
sheet = client.open_by_url(st.secrets["gspread"]["sheet_url"]).sheet1

def simpan_hasil(url_gambar, pred_label, confidence):
    """Simpan hasil prediksi ke Google Sheets."""
    try:
        sheet.append_row([url_gambar, pred_label, f"{confidence*100:.2f}%"])
    except Exception as e:
        st.warning(f"Gagal menyimpan ke Google Sheet: {e}")

# ==============================
# Load Feature Extractor (EfficientNetB7)
# ==============================
@st.cache_resource
def load_feature_extractor_rgb():
    input_tensor = layers.Input(shape=(224, 224, 3))
    base_model = EfficientNetB7(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_tensor=input_tensor
    )
    base_model.trainable = False
    return base_model

feature_extractor = load_feature_extractor_rgb()

# ==============================
# Load SVM Model
# ==============================
@st.cache_resource(show_spinner=False)
def load_svm_model(model_path: str):
    return joblib.load(model_path)

svm_model = load_svm_model("svm_K3_fold2_C10_Gamma0.01_iter1.pkl")

# Kalau mau, ini hanya untuk referensi:
class_labels = ["Mosaic", "RedRot", "Rust", "Yellow", "Healthy"]

# ==============================
# Preprocessing Image
# ==============================
def preprocess_image(image: Image.Image):
    """Resize dan crop sesuai pipeline training."""
    image = image.convert("RGB")
    resized = image.resize((448, 448), Image.Resampling.LANCZOS)
    w, h = resized.size
    crop_w, crop_h = w // 2, h // 2
    start_x, start_y = (w - crop_w) // 2, (h - crop_h) // 2
    cropped = resized.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))
    final_img = cropped.resize((224, 224), Image.Resampling.LANCZOS)
    final_array = np.array(final_img).astype("float32")
    return np.expand_dims(final_array, axis=0)

def extract_features(img_array, model):
    return model.predict(img_array, verbose=0)

# ==============================
# Upload & Prediksi
# ==============================
st.title("🌱 Klasifikasi Penyakit Daun Tebu")

uploaded_file = st.file_uploader("Pilih File Gambar", type=["jpg", "jpeg", "png"])

# variabel penampung sementara
if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None
if "uploaded_url" not in st.session_state:
    st.session_state.uploaded_url = None

# Step 1: tombol unggah gambar
if uploaded_file:
    st.image(uploaded_file, width=224)

    if st.button("Unggah Gambar"):
        UPLOAD_FOLDER = "uploads"
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Upload ke Cloudinary
        url_gambar = upload_to_cloudinary(file_path)

        st.session_state.uploaded_path = file_path
        st.session_state.uploaded_url = url_gambar

        st.success("✅ Gambar berhasil diunggah. Sekarang klik 'Lihat Hasil Klasifikasi'")

# Step 2: tombol klasifikasi setelah ada file yang diunggah
if st.session_state.uploaded_path and st.button("Lihat Hasil Klasifikasi"):
    with st.spinner("Memproses gambar..."):
        # Prediksi
        img = Image.open(st.session_state.uploaded_path)
        proc_img = preprocess_image(img)
        features = extract_features(proc_img, feature_extractor)

        pred_label = svm_model.predict(features)[0]
        confidence = float(np.max(svm_model.predict_proba(features)[0]))

        # Simpan ke Google Sheets
        if st.session_state.uploaded_url:
            simpan_hasil(st.session_state.uploaded_url, pred_label, confidence)

        # Tampilkan hasil -> 2 baris
        st.success(
f"""🌾 **Prediksi: {pred_label}**  
📊 **Tingkat Keyakinan: {confidence*100:.2f}%**"""
        )

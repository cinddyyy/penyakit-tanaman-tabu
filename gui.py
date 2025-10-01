import streamlit as st
import mysql.connector
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(page_title="Klasifikasi Penyakit Tebu", layout="centered")

# ===============================
# Koneksi Database
# ===============================
def create_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        database=st.secrets["mysql"]["database"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"]
    )

# ===============================
# Generate ID Gambar (G001, G002, ...)
# ===============================
def generate_id_gambar():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id_gambar FROM gambar ORDER BY id_gambar DESC LIMIT 1")
    result = cursor.fetchone()
    conn.close()

    if result:
        last_id = result[0]   # contoh "G005"
        last_num = int(last_id[1:])  # ambil angka 005 -> 5
        new_num = last_num + 1
    else:
        new_num = 1  # kalau tabel kosong

    return f"G{new_num:03d}"

# ===============================
# Simpan path gambar ke DB
# ===============================
def insert_gambar(file_path):
    conn = create_connection()
    cursor = conn.cursor()

    id_gambar = generate_id_gambar()
    query = "INSERT INTO gambar (id_gambar, file_path_gambar) VALUES (%s, %s)"
    cursor.execute(query, (id_gambar, file_path))
    conn.commit()
    conn.close()

    return id_gambar

# ===============================
# Simpan hasil klasifikasi ke DB
# ===============================
def insert_hasil_klasifikasi(id_gambar, nama_penyakit, tingkat_kepercayaan: float):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id_hasil_klasifikasi FROM hasil_klasifikasi ORDER BY id_hasil_klasifikasi DESC LIMIT 1")
    result = cursor.fetchone()
    if result:
        last_id = result[0]
        last_num = int(last_id[1:])
        new_num = last_num + 1
    else:
        new_num = 1
    id_hasil = f"H{new_num:03d}"

    query = """
        INSERT INTO hasil_klasifikasi 
        (id_hasil_klasifikasi, id_gambar, nama_penyakit, tingkat_kepercayaan) 
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(query, (id_hasil, id_gambar, nama_penyakit, tingkat_kepercayaan))
    conn.commit()
    conn.close()

    return id_hasil

# ===============================
# Load data gabungan
# ===============================
def load_data():
    conn = create_connection()
    query = """
    SELECT g.id_gambar, g.file_path_gambar, h.id_hasil_klasifikasi, h.nama_penyakit, h.tingkat_kepercayaan
    FROM gambar g
    LEFT JOIN hasil_klasifikasi h ON g.id_gambar = h.id_gambar
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# =========================
# Load Feature Extractor (cached)
# =========================
@st.cache_resource(show_spinner=False)
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

# =========================
# Load SVM Model (cached)
# =========================
@st.cache_resource(show_spinner=False)
def load_svm_model(model_path: str):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model SVM: {e}")
        return None

svm_model = load_svm_model("svm_K3_fold2_C10_Gamma0.01_iter1.pkl")

# =========================
# Label Kelas
# =========================
class_labels = ["Mosaic", "Red Rot", "Rust", "Yellow", "Healthy"]

# =========================
# Preprocessing Image
# =========================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    img = np.array(image)
    resized = cv2.resize(img, (448, 448))
    h, w, _ = resized.shape
    crop_h, crop_w = h // 2, w // 2
    start_x, start_y = (w - crop_w) // 2, (h - crop_h) // 2
    cropped = resized[start_y:start_y + crop_h, start_x:start_x + crop_w]
    final_img = cv2.resize(cropped, (224, 224))
    final_img = final_img.astype("float32")
    final_img = np.expand_dims(final_img, axis=0)
    return final_img, cropped

# =========================
# Ekstraksi Fitur
# =========================
def extract_features(img_array, model):
    return model.predict(img_array, verbose=0)

# =========================
# Styling
# =========================
st.markdown("""
<style>
.stButton>button {
    width: 100% !important;
    background-color: #4CAF50;
    color: white;
    padding: 18px 20px;
    font-size: 20px;
    border-radius: 12px;
    border: none;
    margin-top: 8px;
}
.prediction-box {
    background-color: #f6fff6;
    border: 1px solid #d3f0d3;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    font-size: 18px;
}
.small-note { color: #666; font-size: 13px; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# =========================
# UI
# =========================
st.title("🌱 Aplikasi Klasifikasi Penyakit Tanaman Tebu (SVM + EfficientNetB7)")

# Folder untuk simpan gambar
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])

# Tombol Unggah Gambar
if st.button("📤 Unggah Gambar", use_container_width=True):
    if uploaded_file is None:
        st.warning("Silakan pilih file gambar terlebih dahulu sebelum mengunggah.")
    else:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Insert gambar ke DB
        new_id = insert_gambar(file_path)
        st.session_state["id_gambar"] = new_id

        img = Image.open(uploaded_file)
        st.session_state.image = img

        # Preprocess
        with st.spinner("Memproses gambar..."):
            proc_img, cropped = preprocess_image(img)
            st.session_state.processed_img = proc_img
            st.session_state.cropped_img = cropped
            if feature_extractor is not None:
                st.session_state.features = extract_features(proc_img, feature_extractor)

        st.success(f"Gambar berhasil disimpan ke database dengan ID {new_id}!")

# Selalu tampilkan gambar yang sudah ada di session_state
if "image" in st.session_state:
    st.image(st.session_state.image, caption=f"Gambar {st.session_state.get('id_gambar','')}", use_column_width=True)

    # Tombol klasifikasi
    if st.button("🔍 Lihat Hasil Klasifikasi", use_container_width=True):
        if svm_model is None:
            st.error("Model SVM belum dimuat.")
        else:
            with st.spinner("Melakukan prediksi..."):
                features = st.session_state.get("features")

                # Prediksi kelas
                pred_raw = svm_model.predict(features)[0]

                # Ambil confidence
                if hasattr(svm_model, "predict_proba"):
                    probs = svm_model.predict_proba(features)[0]
                    confidence = float(np.max(probs))  # ambil probabilitas tertinggi
                else:
                    confidence = None

                try:
                    idx = int(pred_raw)
                    pred_label = class_labels[idx] if 0 <= idx < len(class_labels) else str(pred_raw)
                except Exception:
                    pred_label = str(pred_raw)

                # Simpan hasil klasifikasi ke DB
                if "id_gambar" in st.session_state:
                    insert_hasil_klasifikasi(
                        st.session_state["id_gambar"],
                        pred_label,
                        confidence
                    )

                st.session_state["pred_label"] = pred_label
                st.session_state["confidence"] = confidence

# Jika sudah ada prediksi, tampilkan
if "pred_label" in st.session_state:
    conf = st.session_state.get("confidence")
    conf_text = f"<b>📊 Tingkat Keyakinan:</b> {conf*100:.2f}%" if conf is not None else ""
    st.markdown(
        f'<div class="prediction-box">🌾<b>Prediksi:</b> {st.session_state["pred_label"]}<br>{conf_text}</div>',
        unsafe_allow_html=True
    )


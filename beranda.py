import streamlit as st
import base64
import mysql.connector
import os
from pathlib import Path
import cv2
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(page_title="Klasifikasi Daun Tebu", layout="wide")

# ==============================
# Koneksi Database
# ==============================
def create_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        database=st.secrets["mysql"]["database"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"]
    )

def generate_id_gambar():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id_gambar FROM gambar ORDER BY id_gambar DESC LIMIT 1")
    result = cursor.fetchone()
    conn.close()
    if result:
        last_id = result[0]   # contoh "G005"
        new_num = int(last_id[1:]) + 1
    else:
        new_num = 1
    return f"G{new_num:03d}"

def insert_gambar(file_path):
    conn = create_connection()
    cursor = conn.cursor()
    id_gambar = generate_id_gambar()
    cursor.execute(
        "INSERT INTO gambar (id_gambar, file_path_gambar) VALUES (%s, %s)",
        (id_gambar, file_path)
    )
    conn.commit()
    conn.close()
    return id_gambar

def insert_hasil_klasifikasi(id_gambar, nama_penyakit, tingkat_kepercayaan: float):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id_hasil_klasifikasi FROM hasil_klasifikasi ORDER BY id_hasil_klasifikasi DESC LIMIT 1")
    result = cursor.fetchone()
    if result:
        new_num = int(result[0][1:]) + 1
    else:
        new_num = 1
    id_hasil = f"H{new_num:03d}"
    cursor.execute(
        """
        INSERT INTO hasil_klasifikasi 
        (id_hasil_klasifikasi, id_gambar, nama_penyakit, tingkat_kepercayaan) 
        VALUES (%s, %s, %s, %s)
        """,
        (id_hasil, id_gambar, nama_penyakit, tingkat_kepercayaan)
    )
    conn.commit()
    conn.close()
    return id_hasil

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
class_labels = ["Mosaic", "RedRot", "Rust", "Yellow", "Healthy"]

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
    final_img = cv2.resize(cropped, (224, 224)).astype("float32")
    return np.expand_dims(final_img, axis=0)

def extract_features(img_array, model):
    return model.predict(img_array, verbose=0)

# ==============================
# Fungsi untuk convert gambar ke base64
# ==============================
def get_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ==============================
# Assets
# ==============================
hero_bg = get_base64("images/bg.png")
ig_icon   = get_base64("images/ig.png")
gh_icon   = get_base64("images/github.png")
ln_icon   = get_base64("images/linkedln.png")
user_icon = get_base64("images/user.png")

penyakit_list = [
    {"nama":"Healthy","deskripsi":"Daun tebu sehat ...","gambar":get_base64("images/sehat bg.png")},
    {"nama":"Mosaic","deskripsi":"Gejala mosaik ...","gambar":get_base64("images/mosaic bg.png")},
    {"nama":"Rust","deskripsi":"Penyakit ini ...","gambar":get_base64("images/rust bg.png")},
    {"nama":"Red Rot","deskripsi":"Penyakit red rot ...","gambar":get_base64("images/redrot bg.png")},
    {"nama":"Yellow","deskripsi":"Penyakit yellow ...","gambar":get_base64("images/yellow bg.png")},
]

# ==============================
# CSS satu kali
# ==============================
style_path = Path("style.css")
if style_path.exists():
    style = style_path.read_text(encoding='utf-8')
else:
    style = ""

style += f"""
.hero {{
    background: url("data:image/png;base64,{hero_bg}") no-repeat center center/cover;
    height: 400px;
    width: 100%;
    color: white;
    padding: 2rem;
}}
"""
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

# ==============================
# Hero Section
# ==============================
with st.container():
    st.markdown("## Klasifikasi Penyakit Daun Tebu Menggunakan EfficientNetB7-SVM")
    st.image(f"data:image/png;base64,{hero_bg}", use_column_width=True)
    st.write("""
    Penyakit pada daun tebu menjadi ancaman serius ...  
    Penelitian ini menggunakan pendekatan hybrid, yaitu model CNN dengan arsitektur
    EfficientNet-B7 sebagai ekstraksi fitur dan Support Vector Machine (SVM) sebagai model klasifikasi.
    """)

# ==============================
# Bagian Contoh Penyakit
# ==============================
st.markdown("### Contoh Penyakit")
cols = st.columns(5)
for idx, penyakit in enumerate(penyakit_list):
    with cols[idx % 5]:
        st.image(f"data:image/png;base64,{penyakit['gambar']}", caption=penyakit['nama'])
        st.write(penyakit['deskripsi'])

# ==============================
# Upload & Prediksi
# ==============================
st.markdown("### Klasifikasi Penyakit Citra Daun Tebu")

centered_col = st.columns([1, 2, 1])[1]

if "upload_clicked" not in st.session_state:
    st.session_state.upload_clicked = False
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "removed_image" not in st.session_state:
    st.session_state.removed_image = False

with centered_col:
    uploaded_file = st.file_uploader("Pilih File", type=["jpg","jpeg","png"])
    upload_pressed = st.button("Unggah Gambar", use_container_width=True)

    if uploaded_file is None and st.session_state.uploaded_image is not None:
        st.session_state.uploaded_image = None
        st.session_state.removed_image = True

if upload_pressed:
    st.session_state.upload_clicked = True
    if uploaded_file is None:
        if not st.session_state.removed_image:
            st.warning("⚠ Silakan unggah gambar terlebih dahulu")
        st.session_state.removed_image = False
    else:
        UPLOAD_FOLDER = "uploads"
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        new_id = insert_gambar(file_path)
        st.session_state.id_gambar = new_id
        st.session_state.uploaded_image = uploaded_file
        st.session_state.removed_image = False

if st.session_state.uploaded_image:
    with centered_col:
        st.image(st.session_state.uploaded_image, width=224)
        if st.button("Lihat Hasil Klasifikasi", use_container_width=True):
            with st.spinner("Memproses gambar..."):
                img = Image.open(uploaded_file)
                proc_img = preprocess_image(img)
                features = extract_features(proc_img, feature_extractor)
                pred_raw = svm_model.predict(features)[0]
                confidence = None
                if hasattr(svm_model,"predict_proba"):
                    probs = svm_model.predict_proba(features)[0]
                    confidence = float(np.max(probs))
                try:
                    idx = int(pred_raw)
                    pred_label = class_labels[idx] if 0 <= idx < len(class_labels) else str(pred_raw)
                except Exception:
                    pred_label = str(pred_raw)
                if "id_gambar" in st.session_state:
                    insert_hasil_klasifikasi(st.session_state["id_gambar"], pred_label, confidence)
                conf_text = f"Tingkat Keyakinan: {confidence*100:.2f}%" if confidence else ""
                st.success(f"🌾 Prediksi: {pred_label} \n\n{conf_text}")

# ==============================
# Footer
# ==============================
with st.container():
    c1,c2 = st.columns([2,1])
    with c1:
        st.write("**Universitas Sanata Dharma Yogyakarta**")
        st.write("Fakultas Sains & Teknologi")
        st.write("Kampus III, Paingan, Maguwoharjo, Kec. Depok\nDaerah Istimewa Yogyakarta")
    with c2:
        st.markdown(f"[![Instagram](data:image/png;base64,{ig_icon})](https://www.instagram.com/patrisia__cindy)")
        st.markdown(f"[![GitHub](data:image/png;base64,{gh_icon})](https://github.com/cinddyyy)")
        st.markdown(f"[![LinkedIn](data:image/png;base64,{ln_icon})](https://www.linkedin.com/in/patrisia-cindy-paskariana-043629238/)")

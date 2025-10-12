import streamlit as st
import os
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers
import cloudinary
import cloudinary.uploader
import gspread
from google.oauth2.service_account import Credentials
import base64
from streamlit_cropper import st_cropper

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(page_title="Klasifikasi Penyakit Citra Daun Tebu", layout="wide")

# ==============================
# Konfigurasi Cloudinary
# ==============================
cloudinary.config(
    cloud_name=st.secrets["cloudinary"]["cloud_name"],
    api_key=st.secrets["cloudinary"]["api_key"],
    api_secret=st.secrets["cloudinary"]["api_secret"]
)

def upload_to_cloudinary(file_path: str) -> str:
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

svm_model = load_svm_model("svm_rbf_K7_fold7_C10_Gamma0.001.pkl")

class_labels = ["Mosaic", "RedRot", "Rust", "Yellow", "Healthy"]

# ==============================
# Preprocessing Image
# ==============================
def preprocess_image(image: Image.Image):
    """Resize ke 224x224 dan convert ke array"""
    image = image.convert("RGB")
    resized = image.resize((224, 224), Image.Resampling.LANCZOS)
    final_array = np.array(resized).astype("float32")
    return np.expand_dims(final_array, axis=0)

def center_crop(image: Image.Image):
    """Resize ke 448x448 lalu crop tengah jadi 224x224"""
    image = image.convert("RGB")
    resized = image.resize((448, 448), Image.Resampling.LANCZOS)
    w, h = resized.size
    crop_w, crop_h = w // 2, h // 2
    start_x, start_y = (w - crop_w) // 2, (h - crop_h) // 2
    cropped = resized.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))
    final_img = cropped.resize((224, 224), Image.Resampling.LANCZOS)
    return final_img

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
    {"nama":"Healthy", "deskripsi":"Daun tebu sehat ditandai dengan warna hijau segar "
    "dan pertumbuhan normal. Tidak ditemukan bercak, lesi, ataupun perubahan warna pada daun."
    "Kondisi ini menunjukkan tanaman bebas dari serangan hama maupun penyakit.","gambar":get_base64("images/sehat bg.png")},
    {"nama":"Mosaic", "deskripsi":"Gejala penyakit mosaik ditandai dengan bercak atau belang hijau dan kuning pada daun "
    "semakin jelas saat terkena sinar matahari. Jika infeksi sudah parah, daun dapat mengalami klorosis total serta nekrosis berbentuk bintik merah.","gambar":get_base64("images/mosaic bg.png")},
    {"nama":"Rust", "deskripsi":"Penyakit ini menyebar melalui spora yang terbawa angin, sehingga dapat dengan cepat"
    " menginfeksi lahan tebu yang luas. Gejala ditandai dengan bintik hijau muda di daun kemudian perlahan berubah menjadi bercak cokelat.","gambar":get_base64("images/rust bg.png")},
    {"nama":"Red Rot", "deskripsi":"Penyakit red rot dikenal sebagai kanker tebu disebabkan oleh jamur Colletotrichum falcatum. "
    "Penyakit ini menyebabkan daun layu dan perubahan warna, terutama pada tulang daun menunjukkan lesi merah dengan tepi gelap.","gambar":get_base64("images/redrot bg.png")},
    {"nama":"Yellow", "deskripsi":"Penyakit yellow disebabkan oleh Sugarcane yellow leaf virus (ScYLV) yang ditularkan kutu daun. "
    "Gejalanya berupa penguningan pada tulang daun yang menyebar ke helaian daun, lebih parah di daerah dengan curah hujan tinggi.","gambar":get_base64("images/yellow bg.png")},
]

# ==============================
# Muat style.css
# ==============================
style_path = Path("style.css")
style = ""
if style_path.exists():
    style = style_path.read_text(encoding='utf-8')
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)
else:
    st.warning("File style.css tidak ditemukan!")

style += f"""
.hero {{
    background: url("data:image/png;base64,{hero_bg}") no-repeat center center/cover;
    height: 100vh;
    width: 100%;
    color: white;
}}
"""
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f9fff9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# Hero + Navbar
# ==============================
st.markdown(f"""
<div class="hero">
    <div class="navbar" id="beranda">
        <div class="navbar-left">
            <img src="data:image/png;base64,{user_icon}" alt="User">
            <span>Patrisia Cindy</span>
        </div>
        <div class="navbar-right">
            <a href="#beranda">Beranda</a>
            <a href="#klasifikasi">Klasifikasi</a>
        </div>
    </div>
    <div class="hero-text" >
        <h2>Klasifikasi Penyakit Citra Daun Tebu Menggunakan EfficientNetB7-SVM</h2>
        <p>
            Penyakit pada daun tebu menjadi ancaman serius bagi industri tebu,
            karena dapat menyebabkan kerusakan besar pada tanaman yang terinfeksi,
            menurunkan hasil panen, serta menimbulkan kerugian finansial bagi para petani.
            Deteksi dini terhadap penyakit ini melalui teknologi machine learning dapat mencegah
            kerugian dan kerusakan yang lebih besar.
            Penelitian ini menggunakan pendekatan hybrid, yaitu model CNN dengan arsitektur
            EfficientNet-B7 sebagai ekstraksi fitur dan Support Vector Machine (SVM) sebagai model klasifikasi.  
        </p>
        <br>
        <p> 
            <strong>Catatan:</strong> Aplikasi ini merupakan bagian dari penelitian akademik dan tidak untuk tujuan komersial.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# Bagian Contoh Penyakit
# ==============================
st.markdown('<div></div>', unsafe_allow_html=True)
cols = st.columns(5)
for idx, penyakit in enumerate(penyakit_list):
    with cols[idx % 5]:
        st.markdown(f"""
            <div style="display:flex; justify-content:center;">
                <div class="card overlay-card">
                    <img src="data:image/png;base64,{penyakit['gambar']}" class="card-img" alt="{penyakit['nama']}">
                    <div class="card-overlay">
                        <h3>{penyakit['nama']}</h3>
                        <p>{penyakit['deskripsi']}</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ==============================
# Upload & Prediksi
# ==============================
st.markdown('<div class="section-judul" id="klasifikasi"><h3 class="cc">Klasifikasi Penyakit Citra Daun Tebu</h3></div>', unsafe_allow_html=True)

centered_col = st.columns([1, 2, 1])[1]

# Session state init
if "upload_clicked" not in st.session_state:
    st.session_state.upload_clicked = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None
if "uploaded_url" not in st.session_state:
    st.session_state.uploaded_url = None

# ==============================
# Upload
# ==============================
with centered_col:
    uploaded_file = st.file_uploader("Pilih File", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    upload_pressed = st.button("Unggah Gambar", use_container_width=True)

    # Jika user klik ❌
    if uploaded_file is None and st.session_state.uploaded_file is not None:
        st.session_state.uploaded_file = None
        st.session_state.uploaded_path = None
        st.session_state.uploaded_url = None

# Tombol unggah ditekan
if upload_pressed:
    st.session_state.upload_clicked = True
    if uploaded_file is None:
        st.markdown("<div class='custom-warning'>⚠ Silakan unggah gambar terlebih dahulu</div>", unsafe_allow_html=True)
    else:
        UPLOAD_FOLDER = "uploads"
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        url_gambar = upload_to_cloudinary(file_path)

        st.session_state.uploaded_file = uploaded_file
        st.session_state.uploaded_path = file_path
        st.session_state.uploaded_url = url_gambar

# ==============================
# Crop & Prediksi
# ==============================
if st.session_state.uploaded_file and st.session_state.uploaded_path:
    with centered_col:
        img = Image.open(st.session_state.uploaded_path)

        # Pilihan crop
        col1, col2, col3 = st.columns([1,4,1])
        with col2:
            st.markdown("<p class='crop-title'>✂️ Pilih metode crop sebelum klasifikasi</b>", unsafe_allow_html=True)
            crop_option = st.radio(
                "Metode Crop:",
                ["Tanpa Crop", "Crop Manual"],
                index=0,
                horizontal=True
            )
        final_img = None

        if crop_option == "Tanpa Crop":
            final_img = center_crop(img)
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <img src="data:image/png;base64,{get_base64(st.session_state.uploaded_path)}" width="224">
                </div>
                """,
                unsafe_allow_html=True
            )
            

        elif crop_option == "Crop Manual":
            cropped_img = st_cropper(img, aspect_ratio=None)
            if cropped_img is not None:
                final_img = cropped_img
                # Simpan hasil crop ke temporary path
                temp_path = "temp_crop.png"
                final_img.save(temp_path)
                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <img src="data:image/png;base64,{get_base64(temp_path)}" width="224">
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ==============================
        # Prediksi
        # ==============================
        if final_img is not None:
            if st.button("Lihat Hasil Klasifikasi", use_container_width=True):
                with st.spinner("Memproses gambar..."):
                    proc_img = preprocess_image(final_img)
                    features = extract_features(proc_img, feature_extractor)

                    pred_raw = svm_model.predict(features)[0]
                    confidence = float(np.max(svm_model.predict_proba(features)[0]))

                    try:
                        idx = int(pred_raw)
                        pred_label = class_labels[idx] if 0 <= idx < len(class_labels) else str(pred_raw)
                    except Exception:
                        pred_label = str(pred_raw)

                    if st.session_state.uploaded_url:
                        simpan_hasil(st.session_state.uploaded_url, pred_label, confidence)

                    st.markdown(
                        f"""
                        <div class="prediction-box" style="text-align:center;">
                            🌾 <b>Prediksi:</b> {pred_label}<br>
                            📊 Tingkat Keyakinan: {confidence*100:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# ==============================
# Footer
# ==============================
st.markdown(f"""
    <footer>
        <div class="footer-container">
            <div class="footer-left">
                <p class="fw-bold">Patrisia Cindy Paskariana</p>
                <p class="fw-bold">Informatika, Fakultas Sains & Teknologi</p>
                <p class="fw-bold">Universitas Sanata Dharma Yogyakarta</p>
                <p>2025</p>
            </div>
            <div class="footer-right">
                <a href="https://www.instagram.com/patrisia__cindy?igsh=YmE5ZTdzZzF0cDFs">
                    <img src="data:image/png;base64,{ig_icon}" alt="Instagram">
                </a>
                <a href="https://github.com/cinddyyy">
                    <img src="data:image/png;base64,{gh_icon}" alt="GitHub">
                </a>
                <a href="https://www.linkedin.com/in/patrisia-cindy-paskariana-043629238/">
                    <img src="data:image/png;base64,{ln_icon}" alt="LinkedIn">
                </a>
            </div>
        </div>
    </footer>
""", unsafe_allow_html=True)
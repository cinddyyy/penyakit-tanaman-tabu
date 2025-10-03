import streamlit as st
import base64
import mysql.connector
import os
from pathlib import Path
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

# Generate ID gambar
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

# Simpan gambar ke DB
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

# Simpan hasil klasifikasi ke DB
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
st.cache_resource.clear()

# =========================
# Load Feature Extractor (EfficientNetB7)
# REVISI INI MENGGUNAKAN input_tensor UNTUK MEMAKSA 3 CHANNEL
# =========================
@st.cache_resource
def load_feature_extractor_rgb():
    # 1. Buat input tensor 3-channel secara eksplisit
    input_tensor = layers.Input(shape=(224, 224, 3)) 
    
    # 2. Gunakan input_tensor saat membangun EfficientNetB7
    base_model = EfficientNetB7(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_tensor=input_tensor # Menggantikan input_shape
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

svm_model = load_svm_model("svm_K3_fold2_C10_Gamma0.01_iter1")

# =========================
# Label Kelas
# =========================
class_labels = ["Mosaic", "RedRot", "Rust", "Yellow", "Healthy"]

# =========================
# Preprocessing Image
# =========================
def preprocess_image(image: Image.Image):
    # Pastikan RGB
    image = image.convert("RGB")

    # Resize awal ke 448x448
    resized = image.resize((448, 448), Image.Resampling.LANCZOS)

    # Crop bagian tengah (setengah ukuran)
    w, h = resized.size
    crop_w, crop_h = w // 2, h // 2
    start_x, start_y = (w - crop_w) // 2, (h - crop_h) // 2
    cropped = resized.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))

    # Resize hasil crop ke 224x224
    final_img = cropped.resize((224, 224), Image.Resampling.LANCZOS)

    # Convert ke NumPy array float32
    final_array = np.array(final_img).astype("float32")

    # Tambahkan dimensi batch
    return np.expand_dims(final_array, axis=0)

# =========================
# Ekstraksi Fitur
# =========================
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
st.markdown(
    """
    <div class="section-judul" id="klasifikasi">
        <h3 class="cc">Klasifikasi Penyakit Citra Daun Tebu</h3>
    </div>
    """,
    unsafe_allow_html=True
)

centered_col = st.columns([1, 2, 1])[1]

if "upload_clicked" not in st.session_state:
    st.session_state.upload_clicked = False
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "removed_image" not in st.session_state:
    st.session_state.removed_image = False

with centered_col:
    uploaded_file = st.file_uploader("Pilih File", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    upload_pressed = st.button("Unggah Gambar", use_container_width=True)

    if uploaded_file is None and st.session_state.uploaded_image is not None:
        st.session_state.uploaded_image = None
        st.session_state.removed_image = True

if upload_pressed:
    st.session_state.upload_clicked = True
    if uploaded_file is None:
        if not st.session_state.removed_image:
            st.markdown("<div class='custom-warning'>⚠ Silakan unggah gambar terlebih dahulu</div>", unsafe_allow_html=True)
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
        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="data:image/png;base64,{base64.b64encode(st.session_state.uploaded_image.getvalue()).decode()}" width="224">
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Lihat Hasil Klasifikasi", use_container_width=True):
            with st.spinner("Memproses gambar..."):
                img = Image.open(uploaded_file)
                proc_img = preprocess_image(img)
                features = extract_features(proc_img, feature_extractor)
                pred_raw = svm_model.predict(features)[0]
                confidence = None
                if hasattr(svm_model, "predict_proba"):
                    probs = svm_model.predict_proba(features)[0]
                    confidence = float(np.max(probs))
                try:
                    idx = int(pred_raw)
                    pred_label = class_labels[idx] if 0 <= idx < len(class_labels) else str(pred_raw)
                except Exception:
                    pred_label = str(pred_raw)
                if "id_gambar" in st.session_state:
                    insert_hasil_klasifikasi(st.session_state["id_gambar"], pred_label, confidence)
                conf_text = f"<b>📊 Tingkat Keyakinan:</b> {confidence*100:.2f}%" if confidence else ""
                st.markdown(
                    f'<div class="prediction-box">🌾 <b>Prediksi:</b> {pred_label}<br>{conf_text}</div>',
                    unsafe_allow_html=True
                )

# ==============================
# Footer
# ==============================
st.markdown(f"""
    <footer>
        <div class="footer-container">
            <div class="footer-left">
                <p class="fw-bold">Universitas Sanata Dharma Yogyakarta</p>
                <p class="fw-bold">Fakultas Sains & Teknologi</p>
                <p>
                    Kampus III, Paingan, Maguwoharjo, Kec. Depok <br>
                    Daerah Istimewa Yogyakarta
                </p>
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

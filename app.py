import streamlit as st
from pathlib import Path

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Klasifikasi Penyakit Tebu",
    layout="wide",
)

# --- Styling Custom ---
st.markdown("""
    <style>
    body {
        font-family: 'Poppins', sans-serif;
    }
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 30px;
        position: fixed;
        width: 100%;
        top: 0;
        z-index: 1000;
        background-color: transparent;
    }
    .navbar a {
        color: #F6F6F6;
        text-decoration: none;
        font-weight: 700;
        margin-left: 20px;
        border-radius: 8px;
        padding: 6px 12px;
        transition: all 0.3s ease-in-out;
    }
    .navbar a:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    .navbar-brand {
        color: #F6F6F6;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .jumbotron {
        background-image: url('images/bg.png');
        background-size: cover;
        background-position: center;
        color: white;
        height: 100vh;
        padding-top: 100px;
    }
    .hero-text {
        background: rgba(0,0,0,0.5);
        padding: 20px;
        border-radius: 12px;
    }
    .card img {
        width: 100%;
        border-radius: 10px 10px 0 0;
    }
    footer {
        background-color: #36964F;
        color: white;
        padding: 20px;
        margin-top: 50px;
    }
    footer img {
        width: 40px;
        height: 40px;
    }
    </style>
""", unsafe_allow_html=True)


# --- Navbar ---
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <img src="images/user.png" style="width:40px; height:40px; border-radius:50%;">
        Patrisia Cindy
    </div>
    <div>
        <a href="#beranda">Beranda</a>
        <a href="#klasifikasi">Klasifikasi</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="jumbotron d-flex align-items-center" id="beranda">
    <div class="hero-text">
        <h1><b>Klasifikasi Penyakit Citra Daun Tebu</b><br>
        Menggunakan EfficientNetB7–SVM</h1>
        <p>
            Penyakit pada daun tebu menjadi ancaman serius bagi industri tebu, karena dapat menyebabkan kerusakan besar pada tanaman yang terinfeksi,
            menurunkan hasil panen, serta menimbulkan kerugian finansial bagi para petani.
            Deteksi dini melalui teknologi <em>machine learning</em> dapat mencegah kerugian yang lebih besar.
            Penelitian ini menggunakan pendekatan <em>hybrid</em>, yaitu EfficientNet-B7 sebagai ekstraksi fitur dan SVM sebagai model klasifikasi.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Data Penyakit ---
penyakit_list = [
    {"nama": "Sehat", "gambar": "images/sehat bg.png", "deskripsi": "Tanaman tebu sehat, daun hijau cerah..."},
    {"nama": "Mosaic", "gambar": "images/mosaic bg.png", "deskripsi": "Bercak hijau dan kuning pada daun..."},
    {"nama": "Brown Rust", "gambar": "images/rust bg.png", "deskripsi": "Bercak berwarna coklat pada daun..."},
    {"nama": "Red Rot", "gambar": "images/redrot bg.png", "deskripsi": "Merusak batang tebu..."},
    {"nama": "Yellow Leaf", "gambar": "images/yellow bg.png", "deskripsi": "Daun menguning sebagai gejala awal..."},
]

# --- Cards Section ---
st.markdown("<h2 id='klasifikasi'>Klasifikasi Penyakit</h2>", unsafe_allow_html=True)

cols = st.columns(5)
for col, penyakit in zip(cols, penyakit_list):
    with col:
        st.image(penyakit["gambar"], use_container_width=True)
        st.markdown(f"### {penyakit['nama']}")
        st.caption(penyakit["deskripsi"])

# --- Footer ---
st.markdown("""
<footer>
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <p class="fw-bold">Universitas Sanata Dharma Yogyakarta</p>
            <p class="fw-bold">Fakultas Sains & Teknologi</p>
            <p>Kampus III, Paingan, Maguwoharjo, Kec. Depok <br> Daerah Istimewa Yogyakarta</p>
        </div>
        <div style="display:flex; gap:20px;">
            <a href="#"><img src="images/ig.png"></a>
            <a href="#"><img src="images/github.png"></a>
            <a href="#"><img src="images/linkedln.png"></a>
        </div>
    </div>
</footer>
""", unsafe_allow_html=True)

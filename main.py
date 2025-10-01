import streamlit as st
import mysql.connector
import pandas as pd
import os

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
# Load data dari DB (tanpa tingkat_keyakinan)
# ===============================
def load_data():
    conn = create_connection()
    query = """
    SELECT g.id_gambar, g.file_path_gambar, h.id_hasil_klasifikasi, h.nama_penyakit
    FROM gambar g
    LEFT JOIN hasil_klasifikasi h ON g.id_gambar = h.id_gambar
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ===============================
# GUI Streamlit
# ===============================
st.title("🌿 Unggah & Tampilkan Gambar Tanaman")

# Folder untuk simpan gambar
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

    # Simpan file fisik
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tombol unggah ke database
    if st.button("Unggah Gambar ke Database"):
        new_id = insert_gambar(file_path)
        st.success(f"✅ Gambar berhasil disimpan ke database dengan ID {new_id}!")

        # Tampilkan gambar
        st.image(file_path, caption=f"Gambar {new_id}", use_container_width=True)

st.subheader("📊 Data dari Database")
data = load_data()
st.dataframe(data)

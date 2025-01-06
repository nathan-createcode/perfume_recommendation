import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder
import ast
import logging
import os
import subprocess
from dotenv import load_dotenv
from PIL import Image

# Pengecekan keberadaan file database
if not os.path.exists('perfume_recommendation.db'):
    st.error("Database tidak ditemukan. Pastikan file `perfume_recommendation.db` ada di direktori.")
else:
    try:
        conn = sqlite3.connect('perfume_recommendation.db')
        cursor = conn.cursor()
    except sqlite3.Error as e:
        st.error(f"Terjadi kesalahan saat menghubungkan ke database: {e}")

def clean_price_column(df):
    try:
        df['Harga'] = df['Harga'].str.replace('Rp', '', regex=False).str.replace('.', '', regex=False)
        df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')  # Ubah invalid value menjadi NaN
        df = df.dropna(subset=['Harga'])  # Hapus baris dengan Harga NaN
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membersihkan kolom 'Harga': {e}")
        return df

def check_installed_packages():
    try:
        installed = subprocess.check_output(['pip', 'list'])
        print(installed.decode('utf-8'))
    except Exception as e:
        print(f"Error checking installed packages: {e}")

check_installed_packages()

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Fungsi untuk membaca data dari database
def get_perfume_data():
    try:
        query = "SELECT * FROM perfumes"
        df = pd.read_sql_query(query, conn)
        required_columns = ['Nama Parfum', 'Brand atau Produsen', 'Kategori Aroma', 'Top Notes', 'Middle Notes', 'Base Notes', 'Gender', 'Harga']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Kolom yang diperlukan tidak ditemukan dalam database: {', '.join(missing_columns)}")
            return pd.DataFrame()
        return df
    except pd.io.sql.DatabaseError as e:
        logging.error(f"Error reading data from database: {e}")
        st.error("Terjadi kesalahan saat membaca data dari database.")
        return pd.DataFrame()

# Fungsi untuk membersihkan data
def clean_data(df):
    def safe_eval(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return x
        return x

    for col in ['Top Notes', 'Middle Notes', 'Base Notes']:
        df[col] = df[col].apply(safe_eval)
    return df

# Fungsi untuk visualisasi data
def visualize_data(df):
    st.subheader("Visualisasi Data Parfum")

    # Visualisasi berdasarkan Kategori Aroma
    st.write("Distribusi Parfum berdasarkan Kategori Aroma")
    aroma_counts = df['Kategori Aroma'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=aroma_counts.index, y=aroma_counts.values, ax=ax, palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # Visualisasi berdasarkan Gender
    st.write("Distribusi Parfum berdasarkan Gender")
    gender_counts = df['Gender'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax, palette="pastel")
    plt.tight_layout()
    st.pyplot(fig)

    # Visualisasi berdasarkan Brand
    st.write("Top 10 Brand Parfum")
    brand_counts = df['Brand atau Produsen'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=brand_counts.values, y=brand_counts.index, ax=ax, palette="Set2")
    plt.tight_layout()
    st.pyplot(fig)

# Fungsi untuk memodelkan AI (Regresi atau Klasifikasi)
def ai_model(df, model_type="regression"):
    # Preprocessing untuk model
    df = clean_price_column(df)
    df = df.dropna(subset=['Harga'])  # Hapus baris dengan Harga NaN

    # Encoding untuk variabel kategorikal
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'].astype(str))
    df['Kategori Aroma'] = le.fit_transform(df['Kategori Aroma'].astype(str))

    # Pilih fitur dan target
    X = df[['Gender', 'Kategori Aroma']]
    y = df['Harga']

    # Split data menjadi training dan testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pilih model berdasarkan tipe
    if model_type == "regression":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Latih model
    model.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)

    # Metrik evaluasi
    if model_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.2f}")
    elif model_type == "classification":
        report = classification_report(y_test, y_pred)
        st.text(report)

    # Prediksi harga atau kategori untuk data baru
    new_data = st.text_input("Masukkan data untuk prediksi (contoh: Gender=1, Kategori Aroma=2):")
    if new_data:
        try:
            gender, kategori_aroma = map(int, new_data.split(","))
            prediction = model.predict([[gender, kategori_aroma]])
            st.write(f"Prediksi Harga: Rp {prediction[0]:,.0f}")
        except ValueError:
            st.error("Input data tidak valid. Harap masukkan data dalam format yang benar.")

# Fungsi untuk menambahkan parfum baru
def add_new_perfume(data):
    query = """
    INSERT INTO perfumes (
        "Nama Parfum", "Brand atau Produsen", "Jenis", "Kategori Aroma",
        "Top Notes", "Middle Notes", "Base Notes", "Kekuatan Aroma",
        "Daya Tahan", "Musim atau Cuaca", "Harga", "Ukuran Botol",
        "image_path", "Gender"
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        cursor.execute(query, tuple(data.values()))
        conn.commit()
        return True
    except sqlite3.Error as e:
        logging.error(f"Error adding new perfume: {e}")
        return False

# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Rekomendasi Parfum")

    menu = ["Home", "Search Perfume", "Add New Perfume", "AI Model"]
    choice = st.sidebar.selectbox("Menu", menu, key="menu_selectbox")

    df = get_perfume_data()
    if not df.empty:
        df = clean_data(df)

    if choice == "Home":
        st.write("Selamat datang di Aplikasi Rekomendasi Parfum!")

        # Bagian Edukasi tentang Parfum
        st.subheader("Edukasi tentang Parfum")
        st.write("""
        🌸 **Kategori Aroma Parfum**

        Parfum memiliki berbagai kategori aroma yang unik, seperti:
        - **Floral**: Aroma bunga-bungaan yang lembut dan feminin.
        - **Woody**: Aroma kayu-kayuan yang hangat dan maskulin.
        - **Oriental**: Aroma rempah-rempah eksotis yang sensual.
        - **Fresh**: Aroma segar seperti citrus atau air laut.
        - **Gourmand**: Aroma manis seperti vanila atau karamel.

        🎵 **Struktur Aroma Parfum**

        Parfum memiliki tiga lapisan aroma yang berbeda:
        1. **Top Notes**: Aroma pertama yang tercium, biasanya ringan dan segar.
        2. **Middle Notes**: Aroma yang muncul setelah top notes menghilang, membentuk "jantung" parfum.
        3. **Base Notes**: Aroma yang bertahan paling lama, memberikan kedalaman pada parfum.

        💧 **Konsentrasi Parfum**

        - **Parfum (P)**: Konsentrasi tertinggi (20-30%), bertahan 6-8 jam.
        - **Eau de Parfum (EDP)**: Konsentrasi 15-20%, bertahan 4-5 jam.
        - **Eau de Toilette (EDT)**: Konsentrasi 5-15%, bertahan 2-3 jam.
        - **Eau de Cologne (EDC)**: Konsentrasi 2-4%, bertahan 2 jam.
        - **Eau Fraiche**: Konsentrasi terendah (1-3%), bertahan 1 jam.

        Semakin tinggi konsentrasinya, semakin kuat dan tahan lama aromanya!
        """)

        # Tampilkan Visualisasi Data
        visualize_data(df)

    if choice == "Add New Perfume":
        st.subheader("Tambah Parfum Baru")
        nama = st.text_input("Nama Parfum")
        brand = st.text_input("Brand atau Produsen")
        jenis = st.text_input("Jenis Parfum (Eau de Parfum, dll.)")
        kategori = st.text_input("Kategori Aroma")
        top_notes = st.text_area("Top Notes (pisahkan dengan koma)")
        middle_notes = st.text_area("Middle Notes (pisahkan dengan koma)")
        base_notes = st.text_area("Base Notes (pisahkan dengan koma)")
        kekuatan = st.text_input("Kekuatan Aroma")
        daya_tahan = st.text_input("Daya Tahan (dalam jam)")
        musim = st.text_input("Musim atau Cuaca")
        harga = st.text_input("Harga (format: Rp X.XXX.XXX)")
        ukuran = st.text_input("Ukuran Botol")
        gender = st.selectbox("Gender", ["Male", "Female", "Unisex"])
        image = st.file_uploader("Upload Gambar Parfum", type=['png', 'jpg', 'jpeg'])

        if st.button("Tambah Parfum"):
            if nama and brand:  # Minimal nama dan brand harus diisi
                if image:
                    image_path = os.path.join("img", image.name)
                    with open(image_path, "wb") as f:
                        f.write(image.getbuffer())
                else:
                    image_path = ""

                new_perfume = {
                    "NamaParfum": nama,
                    "Brand atau Produsen": brand,
                    "Jenis": jenis,
                    "Kategori Aroma": kategori,
                    "Top Notes": str(top_notes.split(',')),
                    "Middle Notes": str(middle_notes.split(',')),
                    "Base Notes": str(base_notes.split(',')),
                    "Kekuatan Aroma": kekuatan,
                    "Daya Tahan": daya_tahan,
                    "Musim atau Cuaca": musim,
                    "Harga": harga,
                    "Ukuran Botol": ukuran,
                    "image_path": image_path,
                    "Gender": gender
                }

                if add_new_perfume(new_perfume):
                    st.success("Parfum baru berhasil ditambahkan!")
                else:
                    st.error("Terjadi kesalahan saat menambahkan parfum baru.")

    if choice == "AI Model":
        st.subheader("Pemodelan AI untuk Prediksi")
        model_type = st.radio("Pilih Tipe Model", ("regression", "classification"))
        ai_model(df, model_type)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        st.error("Aplikasi mengalami error. Silakan cek log untuk detailnya.")
    finally:
        # Pastikan koneksi database ditutup setelah aplikasi selesai
        if 'conn' in globals() and conn:
            conn.close()

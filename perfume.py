import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import logging
import os
from PIL import Image
from sqlite3 import Error
import ast
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Konfigurasi logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk mendapatkan koneksi database
def get_db_connection():
    try:
        conn = sqlite3.connect('perfume_recommendation.db', check_same_thread=False)
        conn.row_factory = sqlite3.Row
        logging.info("Database connection established successfully")
        return conn
    except Error as e:
        logging.error(f"Error connecting to database: {e}")
        return None

# Fungsi untuk menutup koneksi database
def close_db_connection(conn):
    if conn:
        conn.close()
        logging.info("Database connection closed.")

# Fungsi untuk membaca data dari database
def get_perfume_data():
    conn = get_db_connection()
    if not conn:
        st.error("Tidak dapat terhubung ke database.")
        return pd.DataFrame()

    try:
        query = "SELECT * FROM perfumes"
        df = pd.read_sql_query(query, conn)
        return df
    except pd.io.sql.DatabaseError as e:
        logging.error(f"Error reading data from database: {e}")
        st.error("Terjadi kesalahan saat membaca data dari database.")
        return pd.DataFrame()
    finally:
        close_db_connection(conn)

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

# Content-Based Filtering
def create_feature_matrix(df):
    features = ['Kategori Aroma', 'Top Notes', 'Middle Notes', 'Base Notes', 'Gender']
    df['combined_features'] = df.apply(lambda row: ' '.join([str(row[feature]) for feature in features]), axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    scaler = MinMaxScaler()
    df['normalized_price'] = scaler.fit_transform(df[['Harga']])
    feature_matrix = np.hstack((tfidf_matrix.toarray(), df['normalized_price'].values.reshape(-1, 1)))
    return feature_matrix

def get_recommendations(perfume_name, df, feature_matrix):
    idx = df.index[df['Nama Parfum'] == perfume_name].tolist()[0]
    cosine_sim = cosine_similarity(feature_matrix[idx:idx+1], feature_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar perfumes
    perfume_indices = [i[0] for i in sim_scores]
    return df.iloc[perfume_indices]

# NLP untuk Edukasi Parfum
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words]
    return tokens

def extract_perfume_info(perfume):
    info = {
        'nama': perfume['Nama Parfum'],
        'brand': perfume['Brand atau Produsen'],
        'kategori': perfume['Kategori Aroma'],
        'top_notes': perfume['Top Notes'],
        'middle_notes': perfume['Middle Notes'],
        'base_notes': perfume['Base Notes'],
        'gender': perfume['Gender']
    }
    return info

def generate_perfume_description(perfume_info, level='pemula'):
    if level == 'pemula':
        description = f"{perfume_info['nama']} adalah parfum {perfume_info['kategori']} dari {perfume_info['brand']}. "
        description += f"Parfum ini cocok untuk {perfume_info['gender']}. "
        description += f"Aroma utamanya adalah {perfume_info['top_notes'][0] if perfume_info['top_notes'] else 'tidak diketahui'}."
    else:  # expert
        description = f"{perfume_info['nama']} adalah kreasi {perfume_info['brand']} dalam kategori {perfume_info['kategori']}. "
        description += f"Didesain untuk {perfume_info['gender']}, parfum ini memiliki profil aroma yang kompleks. "
        description += f"Top notes: {', '.join(perfume_info['top_notes'])}. "
        description += f"Middle notes: {', '.join(perfume_info['middle_notes'])}. "
        description += f"Base notes: {', '.join(perfume_info['base_notes'])}."
    return description

def answer_perfume_question(question, perfume_info):
    tokens = preprocess_text(question)
    if 'apa' in tokens and 'top' in tokens and 'notes' in tokens:
        return f"Top notes dari {perfume_info['nama']} adalah {', '.join(perfume_info['top_notes'])}."
    elif 'siapa' in tokens and 'pembuat' in tokens:
        return f"{perfume_info['nama']} dibuat oleh {perfume_info['brand']}."
    elif 'kategori' in tokens:
        return f"{perfume_info['nama']} termasuk dalam kategori {perfume_info['kategori']}."
    else:
        return "Maaf, saya tidak dapat menjawab pertanyaan tersebut. Coba tanyakan tentang top notes, pembuat, atau kategori parfum."

# Fungsi untuk visualisasi data
def visualize_data(df):
    st.subheader("Visualisasi Data Parfum")
    st.write("Distribusi Parfum berdasarkan Kategori Aroma")
    aroma_counts = df['Kategori Aroma'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=aroma_counts.index, y=aroma_counts.values, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.write("Distribusi Parfum berdasarkan Gender")
    gender_counts = df['Gender'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("Top 10 Brand Parfum")
    brand_counts = df['Brand atau Produsen'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=brand_counts.values, y=brand_counts.index, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

# Fungsi untuk menambahkan parfum baru
def add_new_perfume(data):
    conn = get_db_connection()
    if not conn:
        st.error("Tidak dapat terhubung ke database.")
        return False

    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO perfumes (
            "Nama Parfum", "Brand atau Produsen", "Jenis", "Kategori Aroma",
            "Top Notes", "Middle Notes", "Base Notes", "Kekuatan Aroma",
            "Daya Tahan", "Musim atau Cuaca", "Harga", "Ukuran Botol",
            "image_path", "Gender"
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, tuple(data.values()))
        conn.commit()
        return True
    except sqlite3.Error as e:
        logging.error(f"Error dalam transaksi database: {e}")
        return False
    finally:
        close_db_connection(conn)

# Fungsi untuk mencari parfum
def search_perfume(query, filters):
    conn = get_db_connection()
    if not conn:
        st.error("Tidak dapat terhubung ke database.")
        return pd.DataFrame()

    try:
        sql_query = "SELECT * FROM perfumes WHERE 1=1"
        params = []

        if filters['nama_parfum']:
            sql_query += ' AND "Nama Parfum" LIKE ?'
            params.append(f"%{filters['nama_parfum']}%")
        if filters['brand']:
            sql_query += ' AND "Brand atau Produsen" LIKE ?'
            params.append(f"%{filters['brand']}%")
        if filters['gender']:
            sql_query += ' AND Gender = ?'
            params.append(filters['gender'])
        if filters['jenis']:
            sql_query += ' AND Jenis = ?'
            params.append(filters['jenis'])
        if filters['kekuatan_aroma']:
            sql_query += ' AND "Kekuatan Aroma" = ?'
            params.append(filters['kekuatan_aroma'])
        if filters['daya_tahan']:
            sql_query += ' AND "Daya Tahan" = ?'
            params.append(filters['daya_tahan'])
        if filters['musim']:
            sql_query += ' AND "Musim atau Cuaca" = ?'
            params.append(filters['musim'])
        if filters['max_harga']:
            sql_query += ' AND CAST(REPLACE(REPLACE(Harga, "Rp", ""), ".", "") AS INTEGER) <= ?'
            params.append(int(filters['max_harga'].replace('Rp', '').replace('.', '')))

        df = pd.read_sql_query(sql_query, conn, params=params)
        return df
    except sqlite3.Error as e:
        logging.error(f"Error searching perfume: {e}")
        return pd.DataFrame()
    finally:
        close_db_connection(conn)

def main():
    st.title("Aplikasi Rekomendasi dan Edukasi Parfum")

    menu = ["Home", "Rekomendasi Parfum", "Edukasi Parfum", "Search Perfume", "Add New Perfume", "Visualisasi Data"]
    choice = st.sidebar.selectbox("Menu", menu)

    df = get_perfume_data()
    if not df.empty:
        df = clean_data(df)

    if choice == "Home":
        st.write("Selamat datang di Aplikasi Rekomendasi dan Edukasi Parfum!")
        st.write("Pilih menu di sidebar untuk memulai.")

    elif choice == "Rekomendasi Parfum":
        st.subheader("Rekomendasi Parfum")
        perfume_name = st.selectbox("Pilih parfum yang Anda sukai:", df['Nama Parfum'].tolist())

        if st.button("Dapatkan Rekomendasi"):
            feature_matrix = create_feature_matrix(df)
            recommendations = get_recommendations(perfume_name, df, feature_matrix)

            st.write("Rekomendasi parfum untuk Anda:")
            for idx, row in recommendations.iterrows():
                st.write(f"{row['Nama Parfum']} - {row['Brand atau Produsen']}")

    elif choice == "Edukasi Parfum":
        st.subheader("Edukasi Parfum")
        perfume_name = st.selectbox("Pilih parfum untuk dipelajari:", df['Nama Parfum'].tolist())
        level = st.radio("Pilih tingkat penjelasan:", ('Pemula', 'Expert'))

        if st.button("Pelajari Parfum"):
            perfume = df[df['Nama Parfum'] == perfume_name].iloc[0]
            perfume_info = extract_perfume_info(perfume)
            description = generate_perfume_description(perfume_info, level.lower())
            st.write(description)

            question = st.text_input("Tanyakan sesuatu tentang parfum ini:")
            if question:
                answer = answer_perfume_question(question, perfume_info)
                st.write(answer)

    elif choice == "Search Perfume":
        st.subheader("Cari Parfum")

        col1, col2 = st.columns(2)
        with col1:
            nama_parfum = st.text_input("Nama Parfum")
            brand = st.text_input("Brand atau Produsen")
            gender = st.selectbox("Gender", ["", "Female", "Male", "Unisex"])
            jenis = st.selectbox("Jenis", ["", "EDP", "EDT", "EDC", "Perfume", "Extrait de Parfum", "Parfum Cologne"])

        with col2:
            kekuatan_aroma = st.selectbox("Kekuatan Aroma", ["", "Ringan", "Sedang", "Kuat", "Sangat Kuat"])
            daya_tahan = st.selectbox("Daya Tahan", ["", "Pendek", "Sedang", "Lama", "Sangat Lama"])
            musim = st.selectbox("Musim atau Cuaca", ["", "Semua Musim", "Musim Panas", "Musim Dingin", "Musim Semi", "Musim Gugur", "Malam Hari"])
            harga = st.text_input("Batas Harga (contoh: Rp7.000.000)", "")

        filters = {
            'nama_parfum': nama_parfum,
            'brand': brand,
            'gender': gender,
            'jenis': jenis,
            'kekuatan_aroma': kekuatan_aroma,
            'daya_tahan': daya_tahan,
            'musim': musim,
            'max_harga': harga
        }

        if st.button("Cari"):
            results = search_perfume("", filters)
            if not results.empty:
                st.write(f"Ditemukan {len(results)} hasil:")
                for index, row in results.iterrows():
                    st.write(f"### {row['Nama Parfum']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'image_path' in row and row['image_path']:
                            image = Image.open(row['image_path'])
                            st.image(image, caption=row['Nama Parfum'], use_column_width=True)
                    with col2:
                        for column in results.columns:
                            if column not in ['Nama Parfum', 'image_path']:
                                st.write(f"**{column}:** {row[column]}")
                    st.write("---")
            else:
                st.write("Tidak ada hasil yang ditemukan.")

    elif choice == "Add New Perfume":
        st.subheader("Tambah Parfum Baru")
        nama = st.text_input("Nama Parfum")
        brand = st.text_input("Brand atau Produsen")
        jenis = st.selectbox("Jenis Parfum", ["EDP", "EDT", "EDC", "Perfume", "Extrait de Parfum", "Parfum Cologne"])
        kategori = st.text_input("Kategori Aroma")
        top_notes = st.text_area("Top Notes (pisahkan dengan koma)")
        middle_notes = st.text_area("Middle Notes (pisahkan dengan koma)")
        base_notes = st.text_area("Base Notes (pisahkan dengan koma)")
        kekuatan = st.selectbox("Kekuatan Aroma", ["Ringan", "Sedang", "Kuat", "Sangat Kuat"])
        daya_tahan = st.selectbox("Daya Tahan", ["Pendek", "Sedang", "Lama", "Sangat Lama"])
        musim = st.selectbox("Musim atau Cuaca", ["Semua Musim", "Musim Panas", "Musim Dingin", "Musim Semi", "Musim Gugur", "Malam Hari"])
        harga = st.text_input("Harga (contoh: 7000000)")
        ukuran = st.text_input("Ukuran Botol")
        gender = st.selectbox("Gender", ["Female", "Male", "Unisex"])
        image = st.file_uploader("Upload Gambar Parfum", type=['png', 'jpg', 'jpeg'])

        if st.button("Tambah Parfum"):
            if nama and brand:
                if image:
                    image_path = os.path.join("img", image.name)
                    with open(image_path, "wb") as f:
                        f.write(image.getbuffer())
                else:
                    image_path = ""

                new_perfume = {
                    "Nama Parfum": nama,
                    "Brand atau Produsen": brand,
                    "Jenis": jenis,
                    "Kategori Aroma": kategori,
                    "Top Notes": str([x.strip() for x in top_notes.split(',')]),
                    "Middle Notes": str([x.strip() for x in middle_notes.split(',')]),
                    "Base Notes": str([x.strip() for x in base_notes.split(',')]),
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
            else:
                st.error("Nama Parfum dan Brand harus diisi.")

    elif choice == "Visualisasi Data":
        visualize_data(df)

if __name__ == "__main__":
    main()

print("Aplikasi Rekomendasi Parfum berhasil dijalankan!")


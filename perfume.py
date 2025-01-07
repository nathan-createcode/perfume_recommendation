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
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk mendapatkan koneksi database
def get_db_connection():
    try:
        conn = sqlite3.connect('perfume_recommendation.db', check_same_thread=False)
        conn.row_factory = sqlite3.Row

        # Nonaktifkan write-ahead logging
        conn.execute("PRAGMA journal_mode=DELETE")
        # Aktifkan synchronous mode
        conn.execute("PRAGMA synchronous=FULL")
        # Aktifkan foreign key constraints
        conn.execute("PRAGMA foreign_keys=ON")

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

# Fungsi untuk membersihkan kolom harga
def clean_price_column(df):
    try:
        df['Harga'] = df['Harga'].str.replace('Rp', '', regex=False).str.replace('.', '', regex=False)
        df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')
        df = df.dropna(subset=['Harga'])
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membersihkan kolom 'Harga': {e}")
        return df

# Fungsi untuk membaca data dari database
def get_perfume_data():
    conn = get_db_connection()
    if not conn:
        st.error("Tidak dapat terhubung ke database.")
        return pd.DataFrame()

    try:
        query = "SELECT * FROM perfumes"
        df = pd.read_sql_query(query, conn)
        required_columns = ['Nama Parfum', 'Brand atau Produsen', 'Kategori Aroma', 'Top Notes',
                          'Middle Notes', 'Base Notes', 'Gender', 'Harga']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Kolom yang diperlukan tidak ditemukan dalam database: {', '.join(missing_columns)}")
            return pd.DataFrame()
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

# Fungsi untuk sistem rekomendasi berbasis konten menggunakan TF-IDF
def create_tfidf_matrix(df):
    # Menggabungkan semua fitur teks menjadi satu string
    df['combined_features'] = df['Kategori Aroma'] + ' ' + df['Top Notes'].astype(str) + ' ' + \
                              df['Middle Notes'].astype(str) + ' ' + df['Base Notes'].astype(str) + ' ' + \
                              df['Gender']

    # Menggunakan TF-IDF untuk fitur teks
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    return tfidf_matrix

def get_recommendations(perfume_name, df, tfidf_matrix):
    idx = df.index[df['Nama Parfum'] == perfume_name].tolist()[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar perfumes
    perfume_indices = [i[0] for i in sim_scores]
    return df.iloc[perfume_indices]

# Fungsi untuk NLP dan edukasi parfum
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
        'top_notes': ast.literal_eval(perfume['Top Notes']),
        'middle_notes': ast.literal_eval(perfume['Middle Notes']),
        'base_notes': ast.literal_eval(perfume['Base Notes']),
        'gender': perfume['Gender'],
        'kekuatan': perfume['Kekuatan Aroma'],
        'daya_tahan': perfume['Daya Tahan'],
        'musim': perfume['Musim atau Cuaca'],
        'harga': perfume['Harga']
    }
    return info

def generate_perfume_description(perfume_info, level='pemula'):
    description = {
        'nama': f"Nama Parfum: {perfume_info['nama']}",
        'brand': f"Brand: {perfume_info['brand']}",
        'kategori': f"Kategori: {perfume_info['kategori']}",
        'gender': f"Gender: {perfume_info['gender']}",
    }

    if level == 'expert':
        description.update({
            'top_notes': f"Top Notes: {', '.join(perfume_info['top_notes'])}",
            'middle_notes': f"Middle Notes: {', '.join(perfume_info['middle_notes'])}",
            'base_notes': f"Base Notes: {', '.join(perfume_info['base_notes'])}",
            'kekuatan': f"Kekuatan Aroma: {perfume_info['kekuatan']}",
            'daya_tahan': f"Daya Tahan: {perfume_info['daya_tahan']}",
            'musim': f"Musim yang Cocok: {perfume_info['musim']}",
            'harga': f"Harga: {perfume_info['harga']}"
        })
    else:  # pemula
        description.update({
            'aroma_utama': f"Aroma Utama: {perfume_info['top_notes'][0] if perfume_info['top_notes'] else 'tidak diketahui'}",
            'kekuatan': f"Kekuatan Aroma: {perfume_info['kekuatan']}",
            'musim': f"Musim yang Cocok: {perfume_info['musim']}"
        })

    return description

def normalize_price(price_str):
    if not price_str:
        return None
    # Remove 'Rp', dots, spaces, and commas
    price_str = price_str.replace('Rp', '').replace('.', '').replace(' ', '').replace(',', '')
    try:
        # Convert to integer
        price = int(price_str)
        # Format back to standard format
        return f"Rp{price:,}".replace(',', '.')
    except ValueError:
        return None

def add_new_perfume(data):
    conn = get_db_connection()
    if not conn:
        st.error("Tidak dapat terhubung ke database.")
        return False

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM perfumes")
        count_before = cursor.fetchone()[0]
        logging.info(f"Jumlah parfum sebelum penambahan: {count_before}")

        query = """
        INSERT INTO perfumes (
            "Nama Parfum", "Brand atau Produsen", "Jenis", "Kategori Aroma",
            "Top Notes", "Middle Notes", "Base Notes", "Kekuatan Aroma",
            "Daya Tahan", "Musim atau Cuaca", "Harga", "Ukuran Botol",
            "image_path", "Gender"
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        logging.info(f"Memulai transaksi untuk menambah parfum baru: {data['Nama Parfum']}")

        # Mulai transaksi dengan IMMEDIATE untuk mengunci database
        conn.execute("BEGIN IMMEDIATE TRANSACTION")

        # Eksekusi query insert
        cursor.execute(query, tuple(data.values()))

        # Verifikasi penambahan
        cursor.execute("SELECT COUNT(*) FROM perfumes")
        count_after = cursor.fetchone()[0]
        logging.info(f"Jumlah parfum setelah penambahan: {count_after}")

        if count_after > count_before:
            # Verifikasi data yang baru ditambahkan
            cursor.execute(
                'SELECT * FROM perfumes WHERE "Nama Parfum" = ? AND "Brand atau Produsen" = ?',
                (data['Nama Parfum'], data['Brand atau Produsen'])
            )
            new_perfume = cursor.fetchone()

            if new_perfume:
                logging.info(f"Data parfum baru berhasil diverifikasi: {new_perfume}")
                conn.commit()

                # Paksa flush ke disk
                conn.execute("PRAGMA wal_checkpoint")
                conn.execute("VACUUM")

                logging.info("Transaksi berhasil di-commit dan di-flush ke disk")
                return True
            else:
                logging.error("Data parfum baru tidak ditemukan setelah insert")
                conn.rollback()
                return False
        else:
            logging.error(f"Jumlah parfum tidak bertambah: {count_before} -> {count_after}")
            conn.rollback()
            return False

    except sqlite3.Error as e:
        logging.error(f"Error dalam transaksi database: {e}")
        if conn:
            conn.rollback()
            logging.info("Transaksi di-rollback karena error")
        return False
    finally:
        if conn:
            try:
                conn.execute("PRAGMA integrity_check")
                logging.info("Database integrity check passed")
            except sqlite3.Error as e:
                logging.error(f"Database integrity check failed: {e}")
            finally:
                close_db_connection(conn)

def check_database_status():
    conn = get_db_connection()
    if not conn:
        st.error("Tidak dapat terhubung ke database.")
        return

    try:
        cursor = conn.cursor()

        # Periksa jumlah total parfum
        cursor.execute("SELECT COUNT(*) FROM perfumes")
        total_count = cursor.fetchone()[0]
        logging.info(f"Total parfum dalam database: {total_count}")

        # Periksa parfum terakhir yang ditambahkan
        cursor.execute("""
            SELECT "Nama Parfum", "Brand atau Produsen"
            FROM perfumes
            ORDER BY rowid DESC
            LIMIT 1
        """)
        last_perfume = cursor.fetchone()
        if last_perfume:
            logging.info(f"Parfum terakhir dalam database: {last_perfume}")

        # Periksa integritas database
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]
        logging.info(f"Database integrity check result: {integrity_result}")

        return {
            'total_count': total_count,
            'total_count': total_count,
            'last_perfume': last_perfume,
            'integrity': integrity_result
        }
    except sqlite3.Error as e:
        logging.error(f"Error saat memeriksa status database: {e}")
        return None
    finally:
        close_db_connection(conn)

# Tambahan fungsi untuk memastikan database tersimpan
def ensure_database_saved():
    conn = get_db_connection()
    if not conn:
        return False

    try:
        # Paksa semua perubahan tersimpan ke disk
        conn.execute("PRAGMA wal_checkpoint")
        conn.execute("VACUUM")
        conn.commit()
        return True
    except sqlite3.Error as e:
        logging.error(f"Error ensuring database saved: {e}")
        return False
    finally:
        close_db_connection(conn)

def search_perfume(query, filters):
    conn = get_db_connection()
    if not conn:
        st.error("Tidak dapat terhubung ke database.")
        return pd.DataFrame()

    try:
        sql_query = """
        SELECT * FROM perfumes
        WHERE 1=1
        """
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
            normalized_price = normalize_price(filters['max_harga'])
            if normalized_price:
                sql_query += ' AND CAST(REPLACE(REPLACE(Harga, "Rp", ""), ".", "") AS INTEGER) <= ?'
                params.append(int(normalized_price.replace('Rp', '').replace('.', '')))

        df = pd.read_sql_query(sql_query, conn, params=params)
        return df
    except sqlite3.Error as e:
        logging.error(f"Error searching perfume: {e}")
        return pd.DataFrame()
    finally:
        close_db_connection(conn)

def normalize_image_path(path):
    if not path:
        return ""
    # Ubah backslash menjadi forward slash
    normalized = path.replace('\\', '/')
    # Hapus awalan 'perfume_recommendation/' jika ada
    if normalized.startswith('perfume_recommendation/'):
        normalized = normalized[len('perfume_recommendation/'):]
    return normalized

def optimize_database():
    conn = get_db_connection()
    if not conn:
        st.error("Tidak dapat terhubung ke database.")
        return

    try:
        conn.execute("VACUUM")
        logging.info("Database optimized successfully")
    except sqlite3.Error as e:
        logging.error(f"Error optimizing database: {e}")
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
            tfidf_matrix = create_tfidf_matrix(df)
            recommendations = get_recommendations(perfume_name, df, tfidf_matrix)

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

            st.write("### Informasi Parfum")
            st.write(f"**Nama Parfum:** {perfume_info['nama']}")
            st.write(f"**Brand:** {perfume_info['brand']}")
            st.write(f"**Kategori:** {perfume_info['kategori']}")
            st.write(f"**Gender:** {perfume_info['gender']}")

            if level.lower() == 'expert':
                st.write(f"**Top Notes:** {', '.join(perfume_info['top_notes'])}")
                st.write(f"**Middle Notes:** {', '.join(perfume_info['middle_notes'])}")
                st.write(f"**Base Notes:** {', '.join(perfume_info['base_notes'])}")
            else:
                st.write(f"**Aroma Utama:** {perfume_info['top_notes'][0] if perfume_info['top_notes'] else 'tidak diketahui'}")

            st.write("\n### Panduan Penggunaan")
            st.write("1. Aplikasikan parfum pada titik nadi (pergelangan tangan, leher, belakang telinga)")
            st.write("2. Jangan menggosok parfum setelah diaplikasikan")
            st.write("3. Aplikasikan pada kulit yang bersih dan lembab")
            st.write("4. Simpan parfum di tempat yang sejuk dan terhindar dari sinar matahari langsung")

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
                            normalized_path = normalize_image_path(row['image_path'])
                            full_path = os.path.join(os.getcwd(), normalized_path)
                            if os.path.exists(full_path):
                                try:
                                    image = Image.open(full_path)
                                    st.image(image, caption=row['Nama Parfum'], use_column_width=True)
                                except Exception as e:
                                    st.error(f"Terjadi kesalahan saat menampilkan gambar: {e}")
                            else:
                                st.write("Gambar tidak ditemukan.")
                        else:
                            st.write("Path gambar tidak tersedia dalam data.")
                    with col2:
                        for column in results.columns:
                            if column not in ['Nama Parfum', 'image_path', 'created_at', 'updated_at']:
                                st.write(f"**{column}:** {row[column]}")
                    st.write("---")
            else:
                st.write("Tidak ada hasil yang ditemukan.")

    elif choice == "Add New Perfume":
        st.subheader("Tambah Parfum Baru")

        # Tampilkan status database sebelum penambahan
        st.write("Status Database Sebelum Penambahan:")
        initial_status = check_database_status()
        if initial_status:
            st.write(f"Total parfum: {initial_status['total_count']}")
            if initial_status['last_perfume']:
                st.write(f"Parfum terakhir: {initial_status['last_perfume'][0]}")

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
        harga = st.text_input("Harga (contoh: 7000000, 7.000.000, atau Rp7.000.000)")
        ukuran = st.text_input("Ukuran Botol")
        gender = st.selectbox("Gender", ["Female", "Male", "Unisex"])
        image = st.file_uploader("Upload Gambar Parfum", type=['png', 'jpg', 'jpeg'])

        if st.button("Tambah Parfum"):
            if nama and brand:
                if image:
                    # Save the image in the 'img' folder
                    os.makedirs("img", exist_ok=True)
                    image_path = os.path.join("img", image.name)
                    with open(image_path, "wb") as f:
                        f.write(image.getbuffer())
                else:
                    image_path = ""

                normalized_harga = normalize_price(harga)
                if not normalized_harga:
                    st.error("Format harga tidak valid. Harap masukkan harga dalam format yang benar.")
                else:
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
                        "Harga": normalized_harga,
                        "Ukuran Botol": ukuran,
                        "image_path": image_path,
                        "Gender": gender
                    }

                    if add_new_perfume(new_perfume):
                        if ensure_database_saved():
                            st.success("Parfum baru berhasil ditambahkan dan tersimpan ke database!")
                        else:
                            st.warning("Parfum berhasil ditambahkan tetapi mungkin belum tersimpan permanen. Silakan cek database.")

                        # Tampilkan status database setelah penambahan
                        st.write("Status Database Setelah Penambahan:")
                        final_status = check_database_status()
                        if final_status:
                            st.write(f"Total parfum: {final_status['total_count']}")
                            if final_status['last_perfume']:
                                st.write(f"Parfum terakhir: {final_status['last_perfume'][0]}")

                        if final_status and initial_status:
                            if final_status['total_count'] > initial_status['total_count']:
                                st.success(f"Jumlah parfum berhasil bertambah dari {initial_status['total_count']} menjadi {final_status['total_count']}")
                            else:
                                st.warning("Jumlah parfum tidak bertambah setelah penambahan")

                        logging.info(f"New perfume added: {new_perfume}")
                    else:
                        st.error("Terjadi kesalahan saat menambahkan parfum baru. Silakan cek log untuk detailnya.")
            else:
                st.error("Nama Parfum dan Brand harus diisi.")

    elif choice == "Visualisasi Data":
        visualize_data(df)

if __name__ == "__main__":
    try:
        optimize_database()
        main()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        st.error("Aplikasi mengalami error. Silakan cek log untuk detailnya.")

print("Aplikasi Rekomendasi Parfum berhasil dijalankan!")


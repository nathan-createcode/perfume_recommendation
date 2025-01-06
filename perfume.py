import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
from PIL import Image
import ast
import logging
from dotenv import load_dotenv
import subprocess
import traceback

# Pengecekan keberadaan file database sebelum koneksi dilakukan
def connect_to_database(db_path):
    try:
        if not os.path.exists(db_path):
            st.error("Database tidak ditemukan. Pastikan file `perfume_recommendation.db` ada di direktori.")
            return None
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        st.error(f"Terjadi kesalahan saat menghubungkan ke database: {e}")
        return None

# Pembersihan Kolom Harga
def clean_price_column(df):
    try:
        df['Harga'] = df['Harga'].replace({'Rp': '', '.': ''}, regex=True).astype(float)
        df = df.dropna(subset=['Harga'])  # Hapus baris dengan harga NaN
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membersihkan kolom 'Harga': {e}")
        return df

# Install matplotlib jika belum terinstal
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

# Pengecekan Zlib
from PIL import features
print("zlib available:", features.check('zlib'))

# Cek Paket yang Terinstal
def check_installed_packages():
    try:
        installed = subprocess.check_output(['pip', 'list'])
        print(installed.decode('utf-8'))
    except Exception as e:
        logging.error(f"Error checking installed packages: {e}")

check_installed_packages()

# Konfigurasi OpenAI API
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Koneksi ke database
conn = connect_to_database('perfume_recommendation.db')

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

# Fungsi untuk membersihkan data (berkaitan dengan literal_eval)
def clean_data(df):
    def safe_eval(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return x  # Jika gagal, kembalikan string asli
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

# Fungsi untuk mencari parfum dengan Cosine Similarity
def search_perfume_cosine(df, description, gender, max_price):
    df['combined_text'] = df['Nama Parfum'] + ' ' + df['Brand atau Produsen'] + ' ' + df['Kategori Aroma'] + ' ' + df['Top Notes'].astype(str) + ' ' + df['Middle Notes'].astype(str) + ' ' + df['Base Notes'].astype(str) + ' ' + df['Harga'].astype(str) + ' ' + df['Gender'].astype(str)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    user_vector = tfidf.transform([description])
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    df['similarity'] = cosine_similarities
    df = clean_price_column(df)
    filtered_df = df[df['Gender'].str.contains(gender, case=False, na=False)] if gender != 'All' else df
    filtered_df = filtered_df[filtered_df['Harga'] <= max_price]
    results = filtered_df.sort_values('similarity', ascending=False).head(10)
    return results

# Fungsi untuk konsultasi dengan ChatGPT
def consult_chatgpt(question):
    prompt = f"Pertanyaan tentang parfum: {question}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Anda adalah seorang ahli parfum yang menjawab pertanyaan seputar parfum."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content']
    except openai.error.OpenAIError as e:
        logging.error(f"Error in ChatGPT API call: {e}")
        return "Maaf, terjadi kesalahan saat berkomunikasi dengan ChatGPT."

# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Rekomendasi Parfum")
    menu = ["Home", "Search Perfume", "Consult with ChatGPT", "Add New Perfume"]
    choice = st.sidebar.selectbox("Menu", menu)

    df = get_perfume_data()
    if not df.empty:
        df = clean_data(df)

    if choice == "Home":
        st.write("Selamat datang di Aplikasi Rekomendasi Parfum!")
        if not df.empty:
            visualize_data(df)
        else:
            st.warning("Tidak ada data parfum yang tersedia.")

    if choice == "Search Perfume":
        description = st.text_area("Masukkan deskripsi parfum yang Anda inginkan:")
        gender = st.selectbox("Pilih Gender", ["All", "Male", "Female", "Unisex"])
        max_price = st.number_input("Harga Maksimum (dalam Rupiah)", min_value=0, max_value=7000000, value=1000000, step=100000)

        if st.button("Cari"):
            logging.info(f"Deskripsi: {description}, Gender: {gender}, Harga Maks: {max_price}")
            if not df.empty:
                cosine_results = search_perfume_cosine(df, description, gender, max_price)
                st.subheader("Hasil Pencarian dengan Cosine Similarity")
                for _, row in cosine_results.iterrows():
                    st.write(f"**{row['Nama Parfum']}** oleh {row['Brand atau Produsen']}")
                    st.write(f"Kategori: {row['Kategori Aroma']}")
                    st.write(f"Harga: Rp {int(row['Harga']):,}".replace(",", "."))
                    st.write(f"Top Notes: {', '.join(ast.literal_eval(row['Top Notes']))}")
                    st.write(f"Middle Notes: {', '.join(ast.literal_eval(row['Middle Notes']))}")
                    st.write(f"Base Notes: {', '.join(ast.literal_eval(row['Base Notes']))}")

                    if 'image_path' in row and row['image_path']:
                        try:
                            image = Image.open(row['image_path'])
                            st.image(image, caption=row['Nama Parfum'], width=200)
                        except FileNotFoundError:
                            st.warning(f"Gambar tidak ditemukan untuk {row['Nama Parfum']}")
                    st.write("---")
            else:
                st.warning("Tidak ada data parfum yang tersedia untuk pencarian.")

    elif choice == "Consult with ChatGPT":
        st.subheader("Konsultasi dengan ChatGPT")
        question = st.text_area("Ajukan pertanyaan tentang parfum:")
        if st.button("Tanya"):
            answer = consult_chatgpt(question)
            st.write("Jawaban:")
            st.write(answer)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        st.error("Aplikasi mengalami error. Silakan cek log untuk detailnya.")
        traceback.print_exc()
    finally:
        if 'conn' in globals() and conn:
            conn.close()

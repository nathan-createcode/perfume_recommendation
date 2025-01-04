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
from PIL import features
import ast
import logging
from dotenv import load_dotenv
import os
import subprocess

# Install matplotlib jika belum terinstal
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

print("zlib available:", features.check('zlib'))

def check_installed_packages():
    try:
        installed = subprocess.check_output(['pip', 'list'])
        print(installed.decode('utf-8'))
    except Exception as e:
        print(f"Error checking installed packages: {e}")

check_installed_packages()

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Konfigurasi OpenAI API
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Koneksi ke database
try:
    conn = sqlite3.connect('perfume_recommendation.db')
    cursor = conn.cursor()
except sqlite3.Error as e:
    logging.error(f"Error connecting to database: {e}")
    st.error("Terjadi kesalahan saat menghubungkan ke database.")

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
                # Jika gagal, kembalikan string asli
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

# Fungsi untuk mencari parfum dengan Cosine Similarity
def search_perfume_cosine(df, description, gender, max_price):
    # Gabungkan beberapa kolom yang relevan untuk pencarian
    df['combined_text'] = df['Nama Parfum'] + ' ' + df['Brand atau Produsen'] + ' ' + df['Kategori Aroma'] + ' ' + df['Top Notes'].astype(str) + ' ' + df['Middle Notes'].astype(str) + ' ' + df['Base Notes'].astype(str)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])

    user_vector = tfidf.transform([description])
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    df['similarity'] = cosine_similarities
    filtered_df = df[df['Gender'].str.contains(gender, case=False, na=False)] if gender != 'All' else df
    filtered_df = filtered_df[filtered_df['Harga'].str.replace('Rp', '').str.replace('.', '').astype(float) <= float(max_price)]

    results = filtered_df.sort_values('similarity', ascending=False).head(10)
    return results

# Fungsi untuk mencari parfum dengan ChatGPT
def search_perfume_chatgpt(description, gender, max_price):
    prompt = f"Berdasarkan deskripsi: '{description}', untuk gender: {gender}, dengan harga maksimum: Rp{max_price}, berikan rekomendasi parfum yang sesuai beserta penjelasannya."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Anda adalah seorang ahli parfum yang memberikan rekomendasi berdasarkan preferensi pelanggan."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except openai.error.OpenAIError as e:
        logging.error(f"Error in ChatGPT API call: {e}")
        return "Maaf, terjadi kesalahan saat berkomunikasi dengan ChatGPT."

# Fungsi untuk konsultasi dengan ChatGPT
def consult_chatgpt(question):
    prompt = f"Pertanyaan tentang parfum: {question}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Anda adalah seorang ahli parfum yang menjawab pertanyaan seputar parfum."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except openai.error.OpenAIError as e:
        logging.error(f"Error in ChatGPT API call: {e}")
        return "Maaf, terjadi kesalahan saat berkomunikasi dengan ChatGPT."

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

    elif choice == "Search Perfume":
        st.subheader("Cari Parfum")
        description = st.text_area("Masukkan deskripsi parfum yang Anda inginkan:")
        gender = st.selectbox("Pilih Gender", ["All", "Male", "Female", "Unisex"])
        max_price = st.number_input("Harga Maksimum (dalam Rupiah)", min_value=0, max_value=7000000, value=1000000, step=100000)

        if st.button("Cari"):
            if not df.empty:
                cosine_results = search_perfume_cosine(df, description, gender, max_price)
                chatgpt_results = search_perfume_chatgpt(description, gender, max_price)

                st.subheader("Hasil Pencarian dengan Cosine Similarity")
                for _, row in cosine_results.iterrows():
                    st.write(f"**{row['Nama Parfum']}** oleh {row['Brand atau Produsen']}")
                    st.write(f"Kategori: {row['Kategori Aroma']}")
                    st.write(f"Harga: {row['Harga']}")
                    st.write(f"Top Notes: {', '.join(eval(row['Top Notes']))}")
                    st.write(f"Middle Notes: {', '.join(eval(row['Middle Notes']))}")
                    st.write(f"Base Notes: {', '.join(eval(row['Base Notes']))}")
                    if 'image_path' in row and row['image_path']:
                        try:
                            image = Image.open(row['image_path'])
                            st.image(image, caption=row['Nama Parfum'], width=200)
                        except FileNotFoundError:
                            st.warning(f"Gambar tidak ditemukan untuk {row['Nama Parfum']}")
                    st.write("---")

                st.subheader("Rekomendasi dari ChatGPT")
                st.write(chatgpt_results)
            else:
                st.warning("Tidak ada data parfum yang tersedia untuk pencarian.")

    elif choice == "Consult with ChatGPT":
        st.subheader("Konsultasi dengan ChatGPT")
        question = st.text_area("Ajukan pertanyaan tentang parfum:")
        if st.button("Tanya"):
            answer = consult_chatgpt(question)
            st.write("Jawaban:")
            st.write(answer)

    elif choice == "Add New Perfume":
        st.subheader("Tambah Parfum Baru")
        nama = st.text_input("Nama Parfum")
        brand = st.text_input("Brand atau Produsen")
        jenis = st.text_input("Jenis")
        kategori = st.text_input("Kategori Aroma")
        top_notes = st.text_input("Top Notes (pisahkan dengan koma)")
        middle_notes = st.text_input("Middle Notes (pisahkan dengan koma)")
        base_notes = st.text_input("Base Notes (pisahkan dengan koma)")
        kekuatan = st.text_input("Kekuatan Aroma")
        daya_tahan = st.text_input("Daya Tahan")
        musim = st.text_input("Musim atau Cuaca")
        harga = st.text_input("Harga (format: Rp X.XXX.XXX)")
        ukuran = st.text_input("Ukuran Botol")
        gender = st.selectbox("Gender", ["Male", "Female", "Unisex"])
        image = st.file_uploader("Upload Gambar Parfum", type=['png', 'jpg', 'jpeg'])

        if st.button("Tambah Parfum"):
            if nama and brand:  # Minimal nama dan brand harus diisi
                if image:
                    image_path = os.path.join("img_upload", image.name)
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
            else:
                st.warning("Nama Parfum dan Brand atau Produsen harus diisi.")

if __name__ == "__main__":
    main()

# Tutup koneksi database saat aplikasi selesai
conn.close()
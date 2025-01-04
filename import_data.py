import pandas as pd
import sqlite3

# Nama file
csv_file = 'perfume_recommendation.csv'
db_file = 'perfume_recommendation.db'

# Membaca data dari file CSV
df = pd.read_csv(csv_file)

# Fungsi untuk membersihkan data
def clean_data(dataframe):
    """
    Membersihkan data:
    - Menghapus duplikat.
    - Mengisi nilai kosong dengan nilai default.
    - Membersihkan kolom tertentu jika ada format yang tidak sesuai.
    """
    dataframe = dataframe.drop_duplicates()  # Hapus duplikat
    dataframe = dataframe.fillna("Unknown")  # Isi nilai kosong dengan 'Unknown'
    return dataframe

# Membersihkan data
df_cleaned = clean_data(df)

# Tambahkan kolom yang tidak ada di CSV
if "created_at" not in df_cleaned.columns:
    df_cleaned["created_at"] = pd.Timestamp.now()  # Waktu saat ini

if "updated_at" not in df_cleaned.columns:
    df_cleaned["updated_at"] = pd.Timestamp.now()  # Waktu saat ini

if "image_path" not in df_cleaned.columns:
    df_cleaned["image_path"] = "Unknown"  # Isi default untuk kolom image_path

# Membuat koneksi ke database SQLite
conn = sqlite3.connect(db_file)

# Membuat tabel dengan struktur yang sesuai
create_table_query = """
CREATE TABLE IF NOT EXISTS perfumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    "Nama Parfum" TEXT,
    "Brand atau Produsen" TEXT,
    "Jenis" TEXT,
    "Kategori Aroma" TEXT,
    "Top Notes" TEXT,
    "Middle Notes" TEXT,
    "Base Notes" TEXT,
    "Kekuatan Aroma" TEXT,
    "Daya Tahan" TEXT,
    "Musim atau Cuaca" TEXT,
    "Harga" TEXT,
    "Ukuran Botol" TEXT,
    image_path TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
"""

# Eksekusi query untuk membuat tabel
cursor = conn.cursor()
cursor.execute(create_table_query)
conn.commit()

# Simpan data ke tabel SQLite
table_name = 'perfumes'
df_cleaned.to_sql(table_name, conn, if_exists='append', index=False)  # Gunakan append untuk menambahkan data

# Menutup koneksi
conn.close()

print("Data berhasil dimasukkan ke database dan dibersihkan.")

import os
import sqlite3
import glob

# Nama folder untuk menyimpan gambar
img_folder = 'img'
db_file = 'perfume_recommendation.db'

# Membuat koneksi ke database SQLite
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Query untuk mendapatkan semua parfum dari tabel
query = "SELECT id, `Nama Parfum` FROM perfumes"
cursor.execute(query)
perfumes = cursor.fetchall()  # Mendapatkan semua baris (id dan Nama Parfum)

# Loop melalui setiap baris dan perbarui kolom image_path
for perfume in perfumes:
    perfume_id = perfume[0]  # id dari parfum
    nama_parfum = perfume[1]  # Nama Parfum

    # Buat pola pencarian file berdasarkan nama parfum
    # Cari file gambar dengan nama yang sesuai dan ekstensi apa pun
    search_pattern = os.path.join(img_folder, f"{nama_parfum}.*")  # .* untuk semua ekstensi
    matching_files = glob.glob(search_pattern)  # Cari file yang cocok dengan pola

    if matching_files:
        # Jika ada file yang cocok, gunakan file pertama yang ditemukan
        img_path = matching_files[0]
        update_query = "UPDATE perfumes SET image_path = ? WHERE id = ?"
        cursor.execute(update_query, (img_path, perfume_id))
    else:
        # Jika tidak ada file yang cocok, beri nilai default
        update_query = "UPDATE perfumes SET image_path = ? WHERE id = ?"
        cursor.execute(update_query, ("Unknown", perfume_id))

# Commit perubahan ke database
conn.commit()

# Menutup koneksi
conn.close()

print("Path gambar berhasil ditambahkan ke database.")

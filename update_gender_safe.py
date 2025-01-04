import sqlite3
import pandas as pd

# Nama file database dan Excel
db_file_path = 'perfume_recommendation.db'
updated_excel_path = 'perfume_recommendation_updated.xlsx'

# Step 1: Menambahkan kolom Gender jika belum ada
def add_gender_column():
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE perfumes ADD COLUMN Gender TEXT;")
        print("Kolom Gender berhasil ditambahkan.")
    except sqlite3.OperationalError as e:
        print(f"Kolom Gender sudah ada atau terjadi kesalahan: {e}")
    finally:
        conn.commit()
        conn.close()

# Step 2: Memperbarui data Gender berdasarkan file Excel
def update_gender_data():
    # Load data dari Excel hanya kolom yang relevan
    updated_data = pd.read_excel(updated_excel_path, sheet_name='Sheet1')

    # Pastikan hanya menggunakan 'Nama Parfum' dan 'Gender'
    updated_data = updated_data[['Nama Parfum', 'Gender']]
    gender_mapping = dict(zip(updated_data['Nama Parfum'], updated_data['Gender']))

    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Update kolom Gender berdasarkan nama parfum
    update_query = """
    UPDATE perfumes
    SET Gender = ?
    WHERE "Nama Parfum" = ?;
    """
    for perfume_name, gender in gender_mapping.items():
        cursor.execute(update_query, (gender, perfume_name))

    conn.commit()
    conn.close()
    print("Data Gender berhasil diperbarui.")

# Eksekusi fungsi
add_gender_column()
update_gender_data()

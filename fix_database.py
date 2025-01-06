import sqlite3
import logging
from datetime import datetime

# Konfigurasi logging yang lebih detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_fix.log'),
        logging.StreamHandler()
    ]
)

def fix_database_sync():
    try:
        # Buka koneksi dengan isolation level yang lebih ketat
        conn = sqlite3.connect('perfume_recommendation.db', isolation_level='IMMEDIATE')
        cursor = conn.cursor()

        # 1. Backup data yang ada
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cursor.execute("CREATE TABLE IF NOT EXISTS perfumes_backup_{} AS SELECT * FROM perfumes".format(timestamp))
        logging.info(f"Backup created: perfumes_backup_{timestamp}")

        # 2. Perbaiki auto-increment
        cursor.execute("SELECT COUNT(*) FROM perfumes")
        actual_count = cursor.fetchone()[0]
        logging.info(f"Actual count in database: {actual_count}")

        # 3. Periksa dan hapus duplikat jika ada
        cursor.execute("""
            CREATE TEMPORARY TABLE temp_perfumes AS
            SELECT MIN(rowid) as rowid, *
            FROM perfumes
            GROUP BY "Nama Parfum", "Brand atau Produsen"
            HAVING COUNT(*) >= 1
        """)

        # Hapus tabel asli dan ganti dengan data yang sudah dibersihkan
        cursor.execute("DROP TABLE IF EXISTS perfumes_old")
        cursor.execute("ALTER TABLE perfumes RENAME TO perfumes_old")
        cursor.execute("""
            CREATE TABLE perfumes AS
            SELECT p.*
            FROM perfumes_old p
            INNER JOIN temp_perfumes t
            ON p.rowid = t.rowid
        """)

        # 4. Verifikasi hasil
        cursor.execute("SELECT COUNT(*) FROM perfumes")
        new_count = cursor.fetchone()[0]
        logging.info(f"Count after cleanup: {new_count}")

        # 5. Optimasi database
        cursor.execute("VACUUM")
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]
        logging.info(f"Integrity check result: {integrity_result}")

        # 6. Reset sequence
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='perfumes'")

        # Commit perubahan
        conn.commit()

        print("\nPerbaikan database selesai!")
        print(f"Jumlah record sebelum: {actual_count}")
        print(f"Jumlah record setelah pembersihan: {new_count}")
        print(f"Backup tersimpan dalam tabel: perfumes_backup_{timestamp}")
        print("\nLangkah selanjutnya:")
        print("1. Tutup aplikasi Streamlit")
        print("2. Tutup DB Browser for SQLite")
        print("3. Jalankan ulang aplikasi Streamlit")
        print("4. Buka kembali DB Browser dan refresh tampilan")

        return True

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

# Jalankan perbaikan
if __name__ == "__main__":
    fix_database_sync()
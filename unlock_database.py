import sqlite3
import logging
import time
import os
import signal
from pathlib import Path

# Konfigurasi logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_unlock.log'),
        logging.StreamHandler()
    ]
)

def check_lock_files():
    """Periksa dan hapus file lock database jika ada"""
    db_path = Path('perfume_recommendation.db')
    lock_files = [
        db_path.with_suffix('.db-journal'),
        db_path.with_suffix('.db-wal'),
        db_path.with_suffix('.db-shm')
    ]

    for lock_file in lock_files:
        if lock_file.exists():
            try:
                os.remove(lock_file)
                logging.info(f"Berhasil menghapus file lock: {lock_file}")
            except Exception as e:
                logging.error(f"Gagal menghapus file lock {lock_file}: {e}")

def unlock_database():
    db_path = Path('perfume_recommendation.db')
    if not db_path.exists():
        logging.error(f"File database tidak ditemukan di: {db_path}")
        return False

    # Coba beberapa kali untuk menangani kasus locked
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            logging.info(f"Percobaan {attempt + 1} dari {max_attempts}")

            # Hapus file lock jika ada
            check_lock_files()

            # Tunggu sebentar
            time.sleep(2)

            # Buka koneksi dengan timeout yang lebih lama
            conn = sqlite3.connect(
                'perfume_recommendation.db',
                isolation_level='IMMEDIATE',
                timeout=20.0
            )
            cursor = conn.cursor()

            # 1. Coba ambil exclusive lock
            cursor.execute("BEGIN EXCLUSIVE")

            # 2. Periksa dan perbaiki database
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            logging.info(f"Hasil pemeriksaan integritas: {integrity_result}")

            # 3. Reset journal mode
            cursor.execute("PRAGMA journal_mode=DELETE")
            cursor.execute("PRAGMA synchronous=FULL")

            # 4. Periksa jumlah data
            cursor.execute("SELECT COUNT(*) FROM perfumes")
            total_count = cursor.fetchone()[0]
            logging.info(f"Total data dalam database: {total_count}")

            # 5. Periksa record terakhir
            cursor.execute("""
                SELECT rowid, "Nama Parfum", "Brand atau Produsen"
                FROM perfumes
                ORDER BY rowid DESC
                LIMIT 1
            """)
            last_record = cursor.fetchone()
            logging.info(f"Record terakhir: {last_record}")

            # 6. Optimasi database
            cursor.execute("VACUUM")

            # Commit perubahan
            conn.commit()

            print("\nDatabase berhasil di-unlock!")
            print(f"Total records: {total_count}")
            print(f"Record terakhir: {last_record}")
            print("\nLangkah selanjutnya:")
            print("1. Tutup SEMUA aplikasi yang menggunakan database ini")
            print("2. Tutup DB Browser for SQLite")
            print("3. Tunggu 10 detik")
            print("4. Buka kembali DB Browser")
            print("5. Jalankan ulang aplikasi Streamlit")

            return True

        except sqlite3.Error as e:
            logging.error(f"Error database pada percobaan {attempt + 1}: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            time.sleep(2)  # Tunggu sebentar sebelum mencoba lagi

        finally:
            if 'conn' in locals() and conn:
                conn.close()
                logging.info("Koneksi database ditutup")

    return False

if __name__ == "__main__":
    print("Memulai proses unlock database...")
    print("Pastikan semua aplikasi yang menggunakan database sudah ditutup!")

    # Tunggu input user
    input("Tekan Enter untuk melanjutkan...")

    success = unlock_database()
    if success:
        print("\nProses unlock database selesai!")
        print("Silakan ikuti langkah-langkah di atas.")
    else:
        print("\nGagal melakukan unlock database.")
        print("Saran penyelesaian:")
        print("1. Tutup SEMUA aplikasi yang menggunakan database")
        print("2. Hapus file .db-journal, .db-wal, dan .db-shm jika ada")
        print("3. Restart komputer Anda")
        print("4. Coba jalankan script ini lagi")
import sqlite3
import logging
import time
from pathlib import Path

# Konfigurasi logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_sync.log'),
        logging.StreamHandler()
    ]
)

def force_sync_database():
    db_path = Path('perfume_recommendation.db')
    if not db_path.exists():
        logging.error(f"Database file not found at: {db_path}")
        return False

    conn = None
    try:
        # Tunggu sebentar untuk memastikan semua transaksi selesai
        time.sleep(1)

        # Buka koneksi dengan mode yang lebih ketat
        conn = sqlite3.connect('perfume_recommendation.db', isolation_level='IMMEDIATE')
        cursor = conn.cursor()

        # 1. Paksa commit semua transaksi yang tertunda
        cursor.execute("BEGIN IMMEDIATE")

        # 2. Periksa status database
        cursor.execute("SELECT COUNT(*) FROM perfumes")
        total_count = cursor.fetchone()[0]
        logging.info(f"Total records in database: {total_count}")

        # 3. Periksa record terakhir
        cursor.execute("""
            SELECT rowid, "Nama Parfum", "Brand atau Produsen"
            FROM perfumes
            ORDER BY rowid DESC
            LIMIT 1
        """)
        last_record = cursor.fetchone()
        logging.info(f"Last record: {last_record}")

        # 4. Paksa flush ke disk
        cursor.execute("PRAGMA wal_checkpoint")
        cursor.execute("PRAGMA journal_mode=DELETE")
        cursor.execute("PRAGMA synchronous=FULL")

        # 5. Optimasi database
        cursor.execute("VACUUM")

        # 6. Verifikasi integritas
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]
        logging.info(f"Integrity check result: {integrity_result}")

        # Commit final
        conn.commit()

        print("\nLangkah-langkah untuk melihat perubahan di DB Browser:")
        print("1. Tutup koneksi database di DB Browser (klik kanan -> Close Database)")
        print("2. Tutup aplikasi Streamlit")
        print("3. Tunggu 5 detik")
        print("4. Buka kembali database di DB Browser")
        print("5. Jalankan ulang aplikasi Streamlit")

        return True

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        return False

    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed")

# Jalankan fungsi sinkronisasi
if __name__ == "__main__":
    print("Memulai proses sinkronisasi database...")
    success = force_sync_database()
    if success:
        print("\nSinkronisasi database selesai!")
        print("Silakan ikuti langkah-langkah di atas untuk melihat perubahan.")
    else:
        print("\nGagal melakukan sinkronisasi database.")
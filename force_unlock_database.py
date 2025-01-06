import sqlite3
import os
import time
import logging
from pathlib import Path

# Konfigurasi logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_temp_files():
    """Membersihkan file temporary database"""
    base_path = Path('perfume_recommendation.db')
    temp_files = [
        base_path.with_suffix('.db-journal'),
        base_path.with_suffix('.db-wal'),
        base_path.with_suffix('.db-shm')
    ]

    for file in temp_files:
        if file.exists():
            try:
                file.unlink()
                logging.info(f"Berhasil menghapus file: {file}")
            except Exception as e:
                logging.error(f"Gagal menghapus file {file}: {e}")

def force_unlock_database():
    """Memaksa unlock database"""
    db_path = 'perfume_recommendation.db'

    print("Memulai proses force unlock database...")
    print("PERINGATAN: Pastikan semua aplikasi yang menggunakan database sudah ditutup!")
    input("Tekan Enter untuk melanjutkan...")

    # Langkah 1: Bersihkan file temporary
    print("\nMembersihkan file temporary...")
    clean_temp_files()

    try:
        # Langkah 2: Buka koneksi dengan pengaturan khusus
        conn = sqlite3.connect(db_path, timeout=20)
        cursor = conn.cursor()

        # Langkah 3: Reset pengaturan database
        commands = [
            "PRAGMA journal_mode=DELETE",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA busy_timeout=5000",
            "PRAGMA locking_mode=NORMAL",
            "BEGIN IMMEDIATE",
            "COMMIT",
            "VACUUM"
        ]

        for command in commands:
            try:
                cursor.execute(command)
                result = cursor.fetchone() if command != "VACUUM" else None
                logging.info(f"Eksekusi {command}: {result if result else 'Berhasil'}")
            except sqlite3.Error as e:
                logging.warning(f"Warning pada {command}: {e}")
                continue

        # Langkah 4: Verifikasi integritas
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]
        logging.info(f"Hasil pemeriksaan integritas: {integrity_result}")

        if integrity_result == 'ok':
            print("\nDatabase berhasil di-unlock dan diverifikasi!")
            print("\nLangkah selanjutnya:")
            print("1. Buka DB Browser for SQLite")
            print("2. Buka database Anda")
            print("3. Jika masih ada masalah, jalankan 'VACUUM' dari DB Browser")
            return True

    except sqlite3.Error as e:
        logging.error(f"Error saat unlock database: {e}")
        return False

    finally:
        try:
            if 'conn' in locals():
                conn.close()
                logging.info("Koneksi database ditutup")
        except Exception as e:
            logging.error(f"Error saat menutup koneksi: {e}")

def verify_database():
    """Verifikasi status database"""
    try:
        conn = sqlite3.connect('perfume_recommendation.db')
        cursor = conn.cursor()

        # Cek jumlah record
        cursor.execute("SELECT COUNT(*) FROM perfumes")
        count = cursor.fetchone()[0]
        print(f"\nJumlah record dalam database: {count}")

        # Cek record terakhir
        cursor.execute("""
            SELECT "Nama Parfum", "Brand atau Produsen"
            FROM perfumes
            ORDER BY rowid DESC
            LIMIT 1
        """)
        last_record = cursor.fetchone()
        print(f"Record terakhir: {last_record}")

        return True
    except sqlite3.Error as e:
        logging.error(f"Error saat verifikasi: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    try:
        if force_unlock_database():
            time.sleep(2)  # Tunggu sebentar
            verify_database()
    except Exception as e:
        logging.error(f"Error tidak terduga: {e}")
        print("\nTerjadi error. Silakan coba langkah-langkah berikut:")
        print("1. Restart komputer Anda")
        print("2. Backup file database Anda")
        print("3. Coba jalankan script ini lagi")
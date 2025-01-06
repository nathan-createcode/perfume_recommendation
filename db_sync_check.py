import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_database_state():
    try:
        # Buka koneksi dengan immediate transaction untuk memastikan konsistensi
        conn = sqlite3.connect('perfume_recommendation.db', isolation_level='IMMEDIATE')
        cursor = conn.cursor()

        # 1. Periksa jumlah total record
        cursor.execute("SELECT COUNT(*) FROM perfumes")
        total_count = cursor.fetchone()[0]
        logger.info(f"Total records in database: {total_count}")

        # 2. Periksa record terakhir
        cursor.execute("""
            SELECT rowid, "Nama Parfum", "Brand atau Produsen"
            FROM perfumes
            ORDER BY rowid DESC
            LIMIT 1
        """)
        last_record = cursor.fetchone()
        logger.info(f"Last record: {last_record}")

        # 3. Verifikasi integritas database
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]
        logger.info(f"Integrity check result: {integrity_result}")

        # 4. Force VACUUM untuk mengompres dan mengoptimalkan database
        cursor.execute("VACUUM")

        # 5. Periksa dan setel journal mode
        cursor.execute("PRAGMA journal_mode=DELETE")
        journal_mode = cursor.fetchone()[0]
        logger.info(f"Journal mode: {journal_mode}")

        # 6. Paksa sinkronisasi ke disk
        cursor.execute("PRAGMA synchronous=FULL")

        conn.commit()

        print("\nUntuk memperbaiki tampilan di DB Browser for SQLite:")
        print("1. Tutup koneksi database di DB Browser")
        print("2. Klik kanan pada database -> Refresh")
        print("3. Buka kembali tabel perfumes")

        return True

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return False
    finally:
        if conn:
            conn.close()

# Jalankan verifikasi
verify_database_state()
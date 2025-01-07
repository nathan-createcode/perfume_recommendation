import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Mengunduh data NLTK yang diperlukan...")

# Unduh semua paket yang diperlukan
packages = [
    'punkt',
    'punkt_tab',
    'stopwords',
    'averaged_perceptron_tagger',
    'wordnet'
]

for package in packages:
    try:
        print(f"\nMencoba mengunduh {package}...")
        nltk.download(package, quiet=True)
        print(f"Berhasil mengunduh {package}")
    except Exception as e:
        print(f"Gagal mengunduh {package}: {str(e)}")

print("\nMencoba verifikasi instalasi...")

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    # Coba gunakan fungsinya
    text = "Ini adalah test."
    tokens = word_tokenize(text)
    stop_words = stopwords.words('indonesian')

    print("\nHasil verifikasi:")
    print(f"Tokenisasi berhasil: {tokens}")
    print(f"Stop words tersedia: {len(stop_words)} kata")
    print("\nNLTK siap digunakan!")

except Exception as e:
    print(f"\nTerjadi kesalahan dalam verifikasi: {str(e)}")
    print("Saran troubleshooting:")
    print("1. Pastikan Python dan NLTK terinstal dengan benar")
    print("2. Coba jalankan 'pip install --upgrade nltk'")
    print("3. Jika menggunakan virtual environment, pastikan sudah diaktifkan")
    print("4. Coba jalankan script ini dengan hak administrator")

print("\nProses selesai!")
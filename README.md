
# Website Klasifikasi Berita (News Classification Website)

Proyek ini adalah aplikasi web full-stack yang dirancang untuk mengklasifikasikan artikel berita berbahasa Indonesia ke dalam lima kategori: **Politik, Olahraga, Teknologi, Hiburan, dan Ekonomi.**

Aplikasi ini menggunakan model Machine Learning (TF-IDF dan K-Nearest Neighbors) di backend dan antarmuka React.js di frontend.

## Fitur Utama

* **Klasifikasi Otomatis:** Memprediksi kategori berita secara otomatis berdasarkan judul dan isinya.
* **Pelatihan Real-time:** Model secara otomatis dilatih ulang dengan data baru setiap kali pengguna mengirimkan artikel baru, meningkatkan akurasinya seiring waktu.
* **Antarmuka Interaktif:** Frontend yang responsif (dibangun dengan React dan shadcn/ui) memungkinkan pengguna untuk:
  * Mengirim berita baru.
  * Melihat hasil klasifikasi dan skor kepercayaan (confidence score).
  * Menjelajahi semua berita yang ada, diurutkan berdasarkan kategori.
* **Dashboard Statistik:** Menampilkan jumlah total berita untuk setiap kategori secara real-time.

## Teknologi yang Digunakan (Technology Stack)

### Backend

* **Python 3.10+**
* **FastAPI:** Framework web untuk membangun API.
* **scikit-learn:** Untuk implementasi model `TfidfVectorizer` dan `KNeighborsClassifier`.
* **Sastrawi:** Library NLP untuk preprocessing Bahasa Indonesia (Stemming & Stopword Removal).
* **MongoDB (motor):** Database NoSQL untuk menyimpan artikel berita.
* **Uvicorn:** Server ASGI untuk FastAPI.

### Frontend

* **React.js**
* **Axios:** Untuk melakukan permintaan HTTP ke backend API.
* **shadcn/ui & tailwindcss:** Untuk komponen UI dan styling.
* **Sonner:** Untuk notifikasi (toast).

## Prasyarat

Sebelum memulai, pastikan telah menginstal perangkat lunak berikut:

* Python (versi 3.10 atau lebih baru)
* Node.js (versi 18.x atau lebih baru) & Yarn
* MongoDB (database server yang berjalan, baik lokal maupun di cloud seperti MongoDB Atlas)

## Langkah-langkah Menjalankan Lokal

### 1. Backend (FastAPI)

```bash
# 1. Buka terminal dan masuk ke direktori backend
cd website-klasifikasi-berita/backend

# 2. (Opsional tapi disarankan) Buat dan aktifkan virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Instal semua dependensi Python
pip install -r requirements.txt

# 4. Buat file .env di dalam direktori backend
#    Ganti nilainya sesuai dengan konfigurasi yang dimiliki
touch .env

# Isi file .env backend:
MONGO_URL=mongodb://localhost:27017/
DB_NAME=klasifikasi_berita
CORS_ORIGIN=http://localhost:3000

# 5. Jalankan server backend
uvicorn server:app --reload

# Server backend sekarang berjalan di http://127.0.0.1:8000.
```
### 2. Frontend (React)
```bash
# 1. Buka terminal BARU dan masuk ke direktori frontend
cd website-klasifikasi-berita/frontend

# 2. Instal semua dependensi Node.js menggunakan Yarn
yarn install

# 3. Buat file .env di dalam direktori frontend
touch .env

# Isi file .env frontend:
REACT_APP_BACKEND_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)

# 4. Jalankan aplikasi frontend
yarn start

# Aplikasi frontend sekarang berjalan di http://localhost:3000 dan akan terhubung ke backend.
```

## API ENDPOINTS
* ```GET /api/```:Pesan selamat datang.
* ```POST /api/news```: Mengirim berita baru (judul, isi) untuk diklasifikasi dan disimpan.
* ```GET /api/news```: Mengambil semua berita yang telah disimpan.
* ```GET /api/news/{kategori}```: Mengambil berita berdasarkan kategori tertentu.
* ```GET /api/categories/stats```: Mengambil statistik jumlah berita per kategori.
* ```POST /api/train```: Memicu pelatihan ulang model secara manual.
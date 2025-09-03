# Sistem Deteksi Api dan Percikan Api Real-time

## Deskripsi Proyek

Proyek ini adalah sebuah sistem deteksi api dan percikan api secara _real-time_ menggunakan model _object detection_ YOLOv8. Aplikasi ini dibangun dengan Python menggunakan framework Flask untuk backend dan web dashboard, serta OpenCV untuk pemrosesan video.

### Fitur Utama

- **Deteksi Real-time**: Mendeteksi objek 'api' dan 'percikan' dari stream video.
- **Dukungan Multi-Kamera**: Sistem dirancang untuk dapat memonitor beberapa kamera secara bersamaan.
- **Dashboard Web**: Antarmuka web untuk melihat stream video langsung dari setiap kamera, status deteksi, dan riwayat kejadian.
- **Penyimpanan Bukti**: Secara otomatis menyimpan gambar ketika api atau percikan terdeteksi ke dalam folder `assets/`.
- **Riwayat Deteksi**: Menyimpan dan menampilkan riwayat deteksi api dan percikan untuk setiap kamera.

## Dataset

Model deteksi pada proyek ini dilatih menggunakan dataset kustom. Anda dapat mengakses dan mengunduh dataset tersebut melalui tautan berikut:

[Dataset Deteksi Api & Percikan](https://drive.google.com/file/d/1oAD72VHvjBieVi3eg44bzW2aqZn-sXma/view?usp=sharing)

## Panduan Menjalankan Proyek

### 1. Persiapan Lingkungan

Pastikan Anda telah menginstal Python (versi 3.8 atau lebih baru).

### 2. Instalasi Dependensi

Install semua library Python yang dibutuhkan. Anda dapat membuat file `requirements.txt` dengan isi berikut dan menjalankannya.

Jalankan perintah berikut di terminal:

```bash
pip install -r requirements.txt
```

_Catatan: `ultralytics` akan otomatis menginstal `torch` dan `numpy` jika belum ada._

### 3. Unduh Model

Pastikan Anda memiliki file model `model.pt` yang telah dilatih dan letakkan di direktori utama proyek, sejajar dengan `dashboard_server.py`.

### 4. Konfigurasi Kamera

Buka file `dashboard_server.py` dan sesuaikan variabel `CAMERA_IDS` sesuai dengan indeks kamera yang terhubung ke komputer Anda. Secara default, sistem akan mencoba mengakses kamera dengan indeks `0` dan `1`.

```python
# Inisialisasi dua kamera: 0 (Ruang 1), 1 (Ruang 2)
CAMERA_IDS = ['0', '1']
# ...
```

### 5. Jalankan Server

Untuk memulai aplikasi, jalankan skrip `dashboard_server.py` dari terminal:

```bash
python dashboard_server.py
```

### 6. Akses Dashboard

Setelah server berjalan, buka browser web dan akses alamat berikut:

```
http://127.0.0.1:5000
```

Anda akan melihat dashboard yang menampilkan feed video dari kamera yang telah dikonfigurasi beserta status deteksinya.

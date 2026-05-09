# 🚢 Titanic Survival Predictor

Aplikasi web interaktif berbasis Machine Learning yang memprediksi probabilitas keselamatan penumpang kapal Titanic berdasarkan data demografis dan tiket. Aplikasi ini dibangun menggunakan model klasifikasi Machine Learning dan dibungkus dalam antarmuka dasbor analitik interaktif menggunakan **Streamlit**.

🌍 **Akses Aplikasi Langsung (Live App):** [Titanic Survival Predictor](https://advanced-assignments-4-5-fadiazahranzain.streamlit.app/)

---

## 🌟 Fitur Utama
- **Prediksi Cepat & Real-time:** Menghasilkan prediksi instan saat parameter input diubah.
- **Visualisasi Probabilitas Interaktif:** Menampilkan metrik keyakinan prediksi model melalui *gauge chart* sirkular yang responsif secara dinamis.
- **Analisis Pengaruh Data Input:** Memberikan wawasan transparan mengenai fitur apa yang paling berpengaruh terhadap keselamatan penumpang (seperti Jenis Kelamin, Kelas Tiket, dll).
- **Generator Boarding Pass:** Membuat replika UI *boarding pass* Titanic kustom secara *real-time* berdasarkan data input pengguna.
- **Desain UI Premium:** Dibangun dengan *custom CSS injection* yang memberikan tampilan modern bergaya dasbor analitik kelas atas (menggunakan tipografi *Outfit* dan komponen visual interaktif).

---

## 🛠️ Tech Stack
- **Bahasa Pemrograman:** Python 3.x
- **Framework Frontend/UI:** Streamlit & HTML/CSS (Custom Injection)
- **Machine Learning Library:** Scikit-Learn (Model Klasifikasi), Joblib (Model Serialization)
- **Manipulasi Data:** Pandas, NumPy
- **Deployment:** Streamlit Community Cloud

---

## 📂 Struktur Proyek
- `app.py`: File utama aplikasi Streamlit yang berisi logika interaksi antarmuka (UI/UX) dan pemanggilan prediksi model.
- `model.joblib`: Model Machine Learning pra-latih (pre-trained model) yang telah di-ekspor/disimpan.
- `requirements.txt`: Daftar dependensi pustaka Python yang dibutuhkan agar aplikasi dapat berjalan dengan baik.

---

## 🚀 Cara Menjalankan Aplikasi Secara Lokal (Local Run)

Ikuti langkah-langkah berikut jika Anda ingin menjalankan aplikasi ini di komputer Anda sendiri:

1. **Clone Repositori ini:**
   ```bash
   git clone https://github.com/fadiazahranzn/advanced-assignments-4-5.git
   cd advanced-assignments-4-5
   ```

2. **Buat Virtual Environment (Opsional tapi direkomendasikan):**
   ```bash
   python -m venv venv
   # Di Windows:
   venv\Scripts\activate
   # Di macOS/Linux:
   source venv/bin/activate
   ```

3. **Instal Dependensi:**
   Pastikan Anda menginstal semua *library* yang diperlukan melalui `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi:**
   Jalankan file utama melalui server Streamlit.
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan secara otomatis terbuka di peramban web (*browser*) default Anda pada alamat `http://localhost:8501`.

---

## 📝 Catatan Penilaian
Aplikasi ini dikembangkan untuk tujuan *Advanced Assignment* dengan fokus pada implementasi Machine Learning yang bukan hanya akurat secara komputasi, namun juga harus sangat intuitif dan profesional dari segi *User Interface* (UI) & *User Experience* (UX).

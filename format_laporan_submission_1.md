# Laporan Proyek Machine Learning - Julio Aldrin Purba 

## Domain Proyek

Prediksi cuaca merupakan salah satu tantangan penting dalam dunia meteorologi yang berpengaruh pada berbagai sektor kehidupan, seperti pertanian, transportasi, dan penanggulangan bencana. Informasi cuaca yang akurat dan tepat waktu sangat membantu dalam pengambilan keputusan yang lebih baik dan pengurangan risiko bencana.

Proyek ini berfokus pada pembuatan model machine learning untuk memprediksi cuaca berdasarkan berbagai fitur meteorologis, seperti suhu, kelembaban, tekanan, dan lain-lain.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Prakiraan cuaca manual sering kali kurang akurat dan memerlukan sumber daya besar.
- Prediksi otomatis dapat membantu sektor industri dan masyarakat dalam merencanakan aktivitas harian.
- Dampak cuaca ekstrem seperti banjir dan kekeringan dapat diminimalisir jika dapat diprediksi secara dini.
  
  Format Referensi: Weather Forecasting Using Machine Learning Algorithms ([https://scholar.google.com/](https://scholar.google.com/scholar?q=Weather+Forecasting+Using+Machine+Learning+Algorithms) 

## Business Understanding
### Problem Statements
Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara memprediksi cuaca berdasarkan data historis meteorologi?
- Algoritma machine learning apa yang paling efektif dalam melakukan prediksi tersebut?

### Goals
Menjelaskan tujuan dari pernyataan masalah:
- Membuat sistem prediksi cuaca menggunakan data historis.
- Meningkatkan akurasi prediksi menggunakan beberapa model machine learning seperti Random Forest, KNN, dan SVM.

    ### Solution statements
    - Menerapkan algoritma klasifikasi seperti:
      - Random Forest: model ensemble untuk menghindari overfitting.
      - K-Nearest Neighbors (KNN): model berbasis kemiripan data untuk prediksi.
      - Support Vector Machine (SVM): untuk memaksimalkan margin antar kelas
      - LSTM (Long Short-Term Memory) : untuk memprediksi curah hujan.
    - Membandingkan performa setiap model menggunakan MAE,RAE,MSE,R2

## Data Understanding

Dataset yang digunakan merupakan data cuaca yang memiliki fitur-fitur sebagai berikut:
- MinTemp: Suhu minimum harian
- MaxTemp: Suhu maksimum harian
- Rainfall: Curah hujan
- WindGustSpeed: Kecepatan angin kencang
- Humidity3pm: Kelembaban pada pukul 3 sore
- Pressure9am, Pressure3pm: Tekanan udara pada pukul 9 pagi dan 3 sore
- Temp3pm: Suhu pada pukul 3 sore
- RainTomorrow: Label target (apakah akan hujan besok atau tidak)

Dataset ini dibersihkan dan diambil subset-nya sebanyak 30.000 baris dari total 142.000 baris untuk efisiensi proses pelatihan model. Contoh: [Prediksi_Cuaca](https://www.kaggle.com/datasets/ratnasarii/prediksi-cuaca).

## Data Preparation
Langkah-langkah data preparation yang dilakukan:
- Handling Missing Values: Menghapus data dengan nilai kosong.
- Encoding: Mengubah data kategorikal menjadi numerik (misalnya, RainTomorrow diubah ke 0 dan 1).
- Feature Selection: Memilih fitur-fitur relevan untuk model.
- Feature Scaling: Normalisasi menggunakan StandardScaler untuk KNN dan SVM agar performa optimal.
Alasan dilakukan preparation ini adalah agar data siap digunakan untuk model ML yang sensitif terhadap skala dan tidak dapat menangani data kosong.

## Modeling
Model yang digunakan:

- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- LSTM (Long Short-Term Memory)
Setiap model dilatih menggunakan data train-test split (80:20), dan dilakukan evaluasi setelahnya.

Kelebihan dan Kekurangan:

Random Forest: Akurat, tahan terhadap overfitting, namun lebih lambat.
KNN: Sederhana namun sensitif terhadap skala dan outlier.
SVM: Baik untuk data dengan margin jelas, namun kurang efisien untuk dataset besar.
LTSM: Mampu Menangani Data Berurutan / Time Series, Butuh Banyak Data dan Waktu Latih.

## Evaluation
Hasil evaluasi:
Model	Akurasi	Precision	Recall	F1 Score
Random Forest	0.85	0.89	0.82	0.85
KNN	0.77	0.81	0.74	0.77
SVM	0.78	0.83	0.75	0.78
Model terbaik yang dipilih adalah Random Forest karena memberikan skor terbaik di semua metrik.



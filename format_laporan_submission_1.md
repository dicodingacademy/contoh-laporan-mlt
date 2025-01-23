# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Telekomunikasi merupakan salah satu bidang yang terus berkembang pesat seiring dengan kemajuan teknologi. Dalam era digital saat ini, jumlah data yang dihasilkan oleh jaringan telekomunikasi meningkat secara eksponensial. Data tersebut berasal dari berbagai sumber seperti data penggunaan pelanggan, log jaringan, serta data layanan pelanggan. Hal ini menciptakan tantangan besar dalam mengelola, menganalisis, dan memanfaatkan data untuk meningkatkan efisiensi jaringan, pengalaman pelanggan, dan kualitas layanan.

Machine Learning (ML) dan Deep Learning (DL) menawarkan solusi yang revolusioner dalam menangani tantangan-tantangan tersebut. Teknologi ini memungkinkan sistem untuk belajar dari data historis, mengenali pola-pola tertentu, dan membuat prediksi atau keputusan otomatis yang relevan. Misalnya, ML dapat digunakan untuk memprediksi gangguan jaringan, mendeteksi penipuan (fraud detection), atau meningkatkan alokasi sumber daya jaringan.

Dalam konteks pengalaman pelanggan, perusahaan telekomunikasi menghadapi tantangan untuk memahami kebutuhan pelanggan secara individu agar dapat memberikan layanan yang personal. Pelanggan cenderung memiliki ekspektasi tinggi terhadap layanan yang mereka terima, termasuk layanan yang disesuaikan dengan preferensi mereka. Selain itu, churn pelanggan atau perpindahan pelanggan ke penyedia layanan lain menjadi salah satu ancaman besar yang dihadapi industri ini. Oleh karena itu, pengelolaan hubungan pelanggan yang efektif menjadi kunci dalam mempertahankan loyalitas pelanggan dan meningkatkan pendapatan perusahaan.

Machine Learning dapat digunakan untuk menyelesaikan berbagai masalah terkait pengalaman pelanggan seperti:

1.  Prediksi Churn Pelanggan: Mengidentifikasi pelanggan yang kemungkinan besar akan meninggalkan layanan perusahaan agar dapat diambil tindakan preventif.

2.  Segmentasi Pelanggan: Mengelompokkan pelanggan berdasarkan perilaku, preferensi, atau pola penggunaan mereka untuk memberikan penawaran yang relevan.

3.  Prediksi Nilai Umur Pelanggan (Customer Lifetime Value): Memperkirakan nilai pelanggan dalam jangka panjang untuk membantu perusahaan dalam membuat keputusan strategis, seperti alokasi anggaran pemasaran atau desain program loyalitas.

Studi oleh Mishra et al. (2022) menunjukkan bahwa segmentasi pelanggan berbasis ML dapat meningkatkan efektivitas kampanye pemasaran hingga 25%. Selain itu, implementasi model prediksi churn pelanggan berbasis algoritma gradient boosting telah terbukti meningkatkan tingkat retensi pelanggan sebesar 20%.

Dengan memanfaatkan kekuatan ML dan DL, perusahaan telekomunikasi dapat menciptakan strategi yang lebih proaktif, efisien, dan terpersonalisasi untuk memenuhi kebutuhan pelanggan serta meningkatkan kepuasan mereka.
**Referensi:**

Chen, Y., et al. (2021). "Deep Learning for Network Management: Techniques and Applications." IEEE Communications Surveys & Tutorials. DOI: 10.1109/COMST.2021.3057800.

Zhang, X., et al. (2020). "Machine Learning for Telecom Networks: A Comprehensive Survey." IEEE Access. DOI: 10.1109/ACCESS.2020.2978494.

Mishra, R., et al. (2022). "Customer Segmentation in Telecom Industry Using Machine Learning." Journal of Big Data. DOI: 10.1186/s40537-022-00512-3.

## Business Understanding

### Problem Clarification

**Problem Statements**

1.  **Churn Pelanggan:** Perusahaan telekomunikasi sering kali kehilangan pelanggan karena perpindahan mereka ke penyedia layanan lain. Hal ini dapat terjadi karena kurangnya personalisasi layanan, pengalaman buruk, atau penawaran yang lebih menarik dari kompetitor.

**Goals**

  **Meningkatkan Tingkat Retensi Pelanggan:**

  Mengidentifikasi pelanggan yang kemungkinan besar akan churn sehingga dapat dilakukan tindakan preventif, seperti memberikan insentif atau meningkatkan kualitas layanan.

### Solution Statement

Untuk mencapai tujuan meningkatkan pengalaman pelanggan, berikut adalah beberapa solusi yang dapat diimplementasikan:

**1. Prediksi Churn Pelanggan dengan Model Klasifikasi**

**Algoritma yang Digunakan:**

  *   Gradient Boosting (misalnya, XGBoost atau LightGBM) untuk baseline model.

  *   Neural Network sederhana untuk mengeksplorasi performa model berbasis deep learning.

**Langkah Implementasi:**

  *   Mengumpulkan dan membersihkan data churn dari dataset seperti Telecom Churn Dataset.

  *   Melakukan eksplorasi data untuk memahami pola churn.

  *   Melatih model klasifikasi menggunakan data historis.

  *   Mengevaluasi model menggunakan metrik seperti AUC-ROC, akurasi, dan recall untuk mengukur kemampuan model dalam mendeteksi pelanggan yang berpotensi churn.

**Improvement:**

  *   Melakukan hyperparameter tuning pada model Gradient Boosting untuk meningkatkan performa.

  * Menggunakan teknik ensemble learning untuk menggabungkan prediksi beberapa model.

Dengan mengimplementasikan solusi ini, perusahaan telekomunikasi dapat meningkatkan loyalitas pelanggan, mengurangi tingkat churn, dan memaksimalkan nilai bisnis dari setiap pelanggan.

## Data Understanding

[Kaggle Link Dataset](https://www.kaggle.com/datasets/suraj520/telecom-churn-dataset).

**Description**

This dataset contains 243,553 rows of customer data from four major telecom partners of India: Airtel, Reliance Jio, Vodafone, and BSNL. The dataset includes various demographic, location, and usage pattern variables for each customer, as well as a binary variable indicating whether the customer has churned or not.

**Variables**

customer_id: Unique identifier for each customer.

telecom_partner: The telecom partner associated with the customer.

gender: The gender of the customer.

age: The age of the customer.

state: The Indian state in which the customer is located.

city: The city in which the customer is located.
pincode: The pincode of the customer's location.

date_of_registration: The date on which the customer registered with the telecom partner.

num_dependents: The number of dependents (e.g. children) the customer has.

estimated_salary: The customer's estimated salary.

calls_made: The number of calls made by the customer.

sms_sent: The number of SMS messages sent by the customer.

data_used: The amount of data used by the customer.

churn: Binary variable indicating whether the customer has churned or not (1 = churned, 0 = not churned).

### Proses Data Preparation

Data preparation adalah langkah penting dalam proses pengolahan data sebelum model machine learning atau deep learning dapat diterapkan. Tahap ini bertujuan untuk memastikan bahwa data yang digunakan berkualitas tinggi, bebas dari noise, dan relevan dengan masalah yang ingin diselesaikan. Berikut adalah beberapa tahapan dalam data preparation yang dilakukan untuk solusi yang diusulkan:

#### 1. **Pengumpulan Data**
Langkah pertama adalah mengumpulkan data yang diperlukan untuk masing-masing masalah yang dihadapi. Misalnya, untuk prediksi churn pelanggan, kita memerlukan data historis mengenai penggunaan layanan pelanggan, durasi langganan, tingkat interaksi pelanggan, dan informasi lainnya yang relevan. Dataset yang digunakan dalam contoh ini adalah *Telecom Churn Dataset* dari Kaggle.

#### 2. **Pembersihan Data (Data Cleaning)**
Pembersihan data sangat penting untuk menghilangkan ketidakkonsistenan dan data yang hilang. Beberapa langkah dalam pembersihan data antara lain:
   - **Menghapus atau mengisi data yang hilang (missing values)**: Data yang tidak lengkap dapat mempengaruhi kualitas model. Data yang hilang bisa diatasi dengan menghapus entri yang tidak lengkap atau mengisinya dengan nilai rata-rata, median, atau menggunakan metode imputasi yang lebih kompleks.
   - **Menghapus duplikat**: Data yang duplikat dapat mengarah pada overfitting dan bias dalam model. Oleh karena itu, entri yang sama harus dihapus.
   - **Menangani data outlier**: Outlier dapat mempengaruhi model dan menyebabkan hasil yang tidak akurat. Dalam kasus ini, teknik seperti clipping atau transformasi data dapat digunakan untuk menangani outlier.

#### 3. **Penyandian Kategorikal (Encoding Categorical Data)**
Banyak algoritma machine learning memerlukan data numerik, sehingga variabel kategorikal seperti jenis layanan atau status langganan perlu diubah menjadi format numerik.
   - **One-Hot Encoding**: Digunakan untuk variabel kategorikal yang memiliki beberapa kategori tanpa urutan tertentu (misalnya, jenis layanan).

#### 4. **Penskalaan Data (Feature Scaling)**
Penskalaan data penting agar fitur dengan skala besar (seperti pengeluaran bulanan) tidak mendominasi model. Teknik penskalaan yang umum digunakan adalah:
   - **StandardScaler**: Mengubah fitur agar memiliki distribusi normal dengan mean = 0 dan deviasi standar = 1.

#### 5. **Pemisahan Data (Data Splitting)**
Data harus dibagi menjadi dua set utama:
   - **Training Set**: Digunakan untuk melatih model.
   - **Test Set**: Digunakan untuk mengevaluasi performa model setelah pelatihan.
   Biasanya, pembagian data dilakukan dengan rasio 80:20 atau 70:30.

#### 6. **Penyusunan Fitur (Feature Engineering)**
Feature engineering adalah proses untuk menciptakan fitur baru yang lebih relevan dengan masalah yang ingin diselesaikan. Beberapa contoh adalah:
   - **Membuat fitur waktu**: Seperti durasi langganan yang dapat dihitung berdasarkan tanggal pendaftaran dan tanggal saat ini.
   - **Menggabungkan fitur**: Misalnya, menggabungkan frekuensi penggunaan dan jenis layanan untuk membentuk fitur baru yang lebih informatif.

### Alasan Mengapa Data Preparation Diperlukan

1. **Meningkatkan Kualitas Data**  
   Data yang tidak bersih, duplikat, atau mengandung nilai yang hilang akan mempengaruhi akurasi dan validitas model. Tanpa pembersihan yang tepat, model akan mempelajari pola yang salah, menghasilkan prediksi yang tidak akurat.

2. **Memastikan Data Konsisten**  
   Data yang tidak terstandarisasi atau memiliki unit yang berbeda dapat membingungkan model, terutama algoritma yang mengandalkan perhitungan jarak atau bobot fitur. Penskalaan data memastikan bahwa semua fitur berada dalam rentang yang setara, menghindari dominasi fitur tertentu terhadap model.

3. **Menangani Variabel Kategorikal**  
   Sebagian besar algoritma machine learning, terutama yang berbasis matematika, tidak dapat langsung menangani data kategorikal. Oleh karena itu, encoding sangat diperlukan untuk mengubah data kategorikal menjadi format yang dapat dipahami oleh algoritma.

4. **Memfasilitasi Proses Pelatihan dan Evaluasi**  
   Pembagian data menjadi set pelatihan dan pengujian penting untuk menghindari overfitting dan memastikan model dapat dievaluasi secara objektif. Tanpa pemisahan data yang tepat, model dapat terlalu cocok dengan data pelatihan dan gagal saat diuji dengan data yang belum terlihat sebelumnya.

5. **Meningkatkan Kinerja Model**  
   Feature engineering membantu untuk memilih atau menciptakan fitur yang lebih representatif bagi masalah yang dihadapi. Fitur yang relevan dapat membantu model dalam memahami pola-pola penting dalam data, yang pada akhirnya meningkatkan prediksi model.

Dengan tahapan data preparation yang baik, perusahaan telekomunikasi dapat membangun model yang lebih efisien dan akurat, yang pada gilirannya akan membantu mereka dalam memprediksi churn pelanggan, mengelompokkan pelanggan, dan memprediksi nilai umur pelanggan secara lebih efektif.

## Modeling

### **Kelebihan dan Kekurangan Setiap Algoritma yang digunakan**

1. **Gradient Boosting (XGBoost/LightGBM)**  
   - **Kelebihan**:
     - Unggul dalam menangani dataset dengan fitur numerik maupun kategori.
     - Mendukung teknik ensemble yang membuatnya lebih robust terhadap overfitting.
     - Performa prediktif yang tinggi pada data tabular.
     - Mendukung tuning hyperparameter yang luas untuk meningkatkan performa.
   - **Kekurangan**:
     - Memerlukan tuning yang cermat karena sensitif terhadap hyperparameter.
     - Konsumsi waktu yang relatif lebih lama pada dataset besar jika dibandingkan model sederhana.
     - Kurang efektif jika dataset sangat kecil atau sangat besar tanpa preprocessing optimal.

2. **Neural Network**  
   - **Kelebihan**:
     - Dapat menangkap pola non-linear yang kompleks.
     - Cocok untuk menangani dataset besar dengan banyak fitur.
     - Bersifat fleksibel, dapat disesuaikan untuk berbagai tipe data dan kebutuhan.
   - **Kekurangan**:
     - Membutuhkan data dalam jumlah besar untuk mencapai performa optimal.
     - Konsumsi komputasi lebih tinggi dibandingkan model seperti Gradient Boosting.
     - Lebih rentan terhadap overfitting, terutama pada dataset kecil.

---

### **Improvement dengan Hyperparameter Tuning**
Untuk **Gradient Boosting**, proses tuning hyperparameter dilakukan untuk meningkatkan performa model. Beberapa hyperparameter penting yang dapat disesuaikan meliputi:
1. **Learning Rate (`eta`)**: Mengontrol kecepatan pembelajaran model. Dicoba dengan nilai seperti `0.01`, `0.05`, dan `0.1`.
2. **Number of Trees (`n_estimators`)**: Menentukan jumlah pohon dalam ensemble. Dicoba dengan nilai seperti `100`, `200`, dan `500`.
3. **Maximum Depth (`max_depth`)**: Mengontrol kompleksitas pohon. Dicoba dengan nilai seperti `3`, `6`, dan `9`.
4. **Subsample**: Menentukan persentase data yang digunakan dalam setiap iterasi. Dicoba dengan nilai seperti `0.6`, `0.8`, dan `1.0`.
5. **Colsample_bytree**: Menentukan persentase fitur yang digunakan dalam setiap pohon. Dicoba dengan nilai seperti `0.5`, `0.7`, dan `1.0`.

**Proses Tuning**:
- **Grid Search**: Melakukan pencarian kombinasi terbaik dari beberapa hyperparameter di atas.
- **Random Search**: Mencoba kombinasi hyperparameter secara acak untuk mempercepat pencarian pada dataset besar.
- **Bayesian Optimization**: Digunakan jika ingin lebih efisien dalam menemukan parameter optimal.

Setelah tuning, model dievaluasi ulang menggunakan metrik seperti AUC-ROC, akurasi, dan recall untuk memastikan peningkatan performa.

---

### **Kesimpulan**
- **Memilih Gradient Boosting sebagai model terbaik**, alasan utamanya adalah stabilitas, kecepatan pelatihan, dan kemampuan menangani dataset tabular dengan baik.
- Dengan hyperparameter tuning, Gradient Boosting dapat dioptimalkan untuk mencapai performa yang lebih baik dibandingkan Neural Network pada kasus churn prediction dengan dataset seperti Telecom Churn Dataset.

## Evaluation
**Metrik Evaluasi**

**1.  AUC-ROC (Area Under the Curve - Receiver Operating Characteristics):**
Mengukur kemampuan model untuk membedakan antara kelas churn dan tidak churn.
Formula: Integral dari kurva ROC.
Semakin tinggi nilai AUC-ROC, semakin baik performa model.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.


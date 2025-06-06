# Laporan Proyek Machine Learning - Berlian Ishma Zhafira Sujana

## Domain Proyek

Customer churn merupakan tantangan kritis di industri telekomunikasi dengan dampak finansial signifikan. Menurut penelitian Reichheld, F.F. dan Sasser, W.E. (1990), peningkatan 5% dalam customer retention dapat meningkatkan profitabilitas sebesar 25-95%. Dataset Telco Customer Churn dari Kaggle ini merekam karakteristik 7,043 pelanggan dengan 21 fitur yang mencakup demografi, layanan yang digunakan, dan riwayat pembayaran.

## Business Understanding
### Problem Statements
1. Tingkat churn pelanggan mencapai 26.5% dari total dataset
2. Perusahaan kesulitan mengidentifikasi pelanggan berisiko churn secara dini
3. Ketidakseimbangan kelas (class imbalance) antara churn dan non-churn

### Goals
1. Membangun model prediktif dengan recall >80% untuk kelas churn
2. Mengidentifikasi faktor dominan penyebab churn
3. Mengurangi false negative untuk meminimalkan pelanggan berisiko yang terlewat

### Solution statements
1. Mengimplementasikan Random Forest dengan class weighting
2. Mengevaluasi performa XGBoost dengan SMOTE untuk handling imbalance
3. Hyperparameter tuning menggunakan GridSearchCV

## Data Understanding
Sumber Dataset: [Telco Customer Churn] - (https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)

Fitur:
- Demografi: gender, SeniorCitizen, Partner, Dependents
- Layanan: PhoneService, MultipleLines, InternetService
- Pembayaran: Contract, PaperlessBilling, PaymentMethod
- Riwayat: tenure, MonthlyCharges, TotalCharges
- Target: Churn (Yes/No)

Penjelasan Fitur:
Berikut adalah deskripsi singkat fitur:
- `customerID`: Kode unik untuk mengidentifikasi pelanggan 
- `gender`: Jenis kelamin pelanggan
- `SeniorCitizen`: Apakah pelanggan termasuk warga senior (1) atau tidak (0)
- `Partner`: Apakah pelanggan memiliki pasangan
- `Dependents`: Apakah pelanggan memiliki tanggungan
- `PhoneService`: Apakah pelanggan memiliki layanan telepon atau tidak
- `MultipleLines`: Apakah pelanggan memiliki beberapa jalur atau tidak
- `InternetService`: Apakah pelanggan memiliki layanan internet atau tidak
- `OnlineSecurity`: Apakah pelanggan memiliki keamanan online atau tidak
- `OnlineBackup`: Apakah pelanggan memiliki cadangan online atau tidak
- `DeviceProtection`: Apakah pelanggan memiliki perlindungan perangkat atau tidak
- `TechSupport`: Apakah pelanggan memiliki dukungan teknis atau tidak
- `StreamingTV`: Apakah pelanggan memiliki TV streaming atau tidak
- `StreamingMovies`: Apakah pelanggan memiliki film streaming atau tidak
- `Contract`: Jangka waktu kontrak pelanggan
- `PaymentMethod`: Tata cara pembayaran
- `PaperlessBilling`: Apakah pelanggan memiliki tagihan tanpa kertas atau tidak
- `tenure`: Lama berlangganan (dalam bulan)
- `MonthlyCharges`: Biaya bulanan
- `TotalCharges`: Total biaya selama menjadi pelanggan
- `Churn`: Target (Yes/No) — apakah pelanggan berhenti berlangganan

### 🔍 Exploratory Data Analysis (EDA):

* **Tidak ada nilai kosong (missing values)** pada sebagian besar kolom berdasarkan hasil `df.isnull().sum()`. Namun, kolom `TotalCharges` bertipe `object` meskipun berisi angka. Setelah dikonversi ke `float`, ditemukan **11 nilai NaN**, yang akan ditangani pada tahap data preparation.
* **Tidak terdapat duplikat** dalam data.
* Kolom `customerID` merupakan identifier unik yang tidak memiliki kontribusi terhadap prediksi dan akan dihapus.
* **Distribusi target (`Churn`) menunjukkan ketidakseimbangan kelas**:

  * Sekitar **73%** data berada pada kelas `No` (tidak churn)
  * Sekitar **27%** data berada pada kelas `Yes` (churn)
    Hal ini menandakan bahwa dataset bersifat **imbalanced**, yang perlu menjadi perhatian khusus dalam proses pelatihan model.

* Beberapa fitur seperti `Contract`, `PaymentMethod`, `InternetService`, dan `OnlineBackup` memiliki **kategori yang berbeda-beda**, yang perlu diubah menjadi representasi numerik menggunakan encoding pada tahap selanjutnya.


## Data Preparation
1. Handling Missing Values: Isi missing values dengan median
2. Feature Engineering: Buat fitur baru TenureYears dari tenure
3. Encoding Kategorikal: Lakukan one-hot encoding untuk kolom kategorikal
4. Train-Test Split: Pisahkan 80% training dan 20% test data



## Modeling

Model yang digunakan dalam proyek ini adalah **Random Forest Classifier**, yaitu algoritma ensemble berbasis decision tree yang sangat populer dalam klasifikasi dan regresi.

### 🔍 Cara Kerja Algoritma Random Forest:

Random Forest bekerja dengan membangun **banyak pohon keputusan (decision trees)** pada subset acak dari data pelatihan, lalu menggabungkan hasil prediksi dari masing-masing pohon (melalui voting mayoritas untuk klasifikasi) untuk menghasilkan prediksi akhir. Proses ini dikenal sebagai **bagging (Bootstrap Aggregating)**, di mana setiap pohon:

* Dilatih dengan data yang di-*sampling* secara acak dengan pengembalian (bootstrapping).
* Pada setiap node, hanya subset acak dari fitur yang dipertimbangkan untuk split, sehingga meningkatkan keberagaman antar pohon.

**Keuntungan utama** Random Forest adalah:

* Lebih tahan terhadap **overfitting** dibanding single decision tree.
* Mampu menangani fitur kategorikal dan numerik secara bersamaan.
* Memberikan estimasi **feature importance**, yang berguna untuk interpretasi model.

---

### ⚙️ Parameter Model yang Digunakan

Model dibangun menggunakan **`RandomForestClassifier` dari library `sklearn.ensemble`** dengan parameter sebagai berikut:

* `random_state=42`: Digunakan untuk memastikan hasil model dapat **direproduksi**. Nilai 42 dipilih secara arbitrer namun umum digunakan sebagai nilai acuan.
* Parameter lainnya seperti `n_estimators`, `max_depth`, dan sebagainya menggunakan nilai **default** dari scikit-learn.

```python
model = RandomForestClassifier(random_state=42)
```

Model dilatih pada data pelatihan (`X_train`, `y_train`) menggunakan:

```python
model.fit(X_train, y_train)
```

Setelah pelatihan, model digunakan untuk memprediksi data uji (`X_test`) dan dilakukan evaluasi performa menggunakan metrik klasifikasi seperti **accuracy**, **precision**, **recall**, dan **f1-score**.

---

### ✨ Catatan Tambahan:

Random Forest juga memberikan nilai **feature importance** yang dapat digunakan untuk mengetahui fitur mana yang paling berpengaruh terhadap prediksi churn. Hasil menunjukkan bahwa fitur-fitur seperti `Contract`, `tenure`, `PaymentMethod`, dan `MonthlyCharges` termasuk yang paling berkontribusi dalam memprediksi churn.


## Evaluation
A. Metrik Evaluasi yang Digunakan
Dalam proyek prediksi customer churn ini, kami menggunakan empat metrik evaluasi utama untuk mengukur performa model:

1. **Precision**  
   - *Definisi*: Rasio prediksi positif yang benar (True Positive) terhadap seluruh prediksi positif (True Positive + False Positive).  
   - *Relevansi*: Mengukur seberapa akurat model dalam memprediksi churn (menghindari false alarm).  

2. **Recall (Sensitivity)**  
   - *Definisi*: Rasio prediksi positif yang benar (True Positive) terhadap seluruh kasus aktual positif (True Positive + False Negative).  
   - *Relevansi*: Mengukur kemampuan model menemukan pelanggan yang benar-benar berisiko churn (menghindari missed detection).  

3. **F1-Score**  
   - *Definisi*: Rata-rata harmonik (harmonic mean) dari precision dan recall.  
   - *Relevansi*: Memberikan balance antara precision dan recall, terutama penting untuk data tidak seimbang.  

4. **Accuracy**  
   - *Definisi*: Rasio prediksi benar (True Positive + True Negative) terhadap total sampel.  
   - *Relevansi*: Mengukur performa keseluruhan model, tetapi kurang informatif untuk data imbalance.  



B. Analisis Hasil Evaluasi

| Kelas        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.82      | 0.86   | 0.84     | 1036    |
| 1            | 0.54      | 0.47   | 0.50     | 373     |
| Akurasi      | -         | -      | -        | 0.75    |
| Macro Avg    | 0.68      | 0.66   | 0.67     | -       |  

C. Interpretasi:  
1. **Kelas 0 (No Churn)**  
   - Precision 82%: Dari semua prediksi "No Churn", 82% benar.  
   - Recall 86%: Model berhasil mengidentifikasi 86% pelanggan yang tidak churn.  
   - Performa baik karena kelas ini dominan (73.5% dataset).  

2. **Kelas 1 (Churn)**  
   - Precision 54%: Dari semua prediksi "Churn", hanya 54% yang benar (banyak false positive).  
   - Recall 47%: Model hanya mendeteksi 47% kasus churn aktual (banyak false negative).  
   - Masalah utama: Model kesulitan memprediksi kelas minoritas (churn).  


D. Rekomendasi Perbaikan
1. **Handling Class Imbalance**  
   - Gunakan teknik oversampling (SMOTE) atau class weighting (`class_weight='balanced'`).  
2. **Optimasi Threshold**  
   - Turunkan threshold prediksi churn untuk meningkatkan recall (misalnya, dari 0.5 ke 0.3).  
3. **Eksperimen Model Lain**  
   - Coba algoritma seperti **XGBoost** atau **LightGBM** yang lebih robust terhadap imbalance.  
4. **Feature Engineering**  
   - Tambahkan fitur interaksi (contoh: `MonthlyCharges/tenure`) untuk meningkatkan sinyal churn.  

E. Kesimpulan:  
Model saat ini cukup baik dalam memprediksi "No Churn" tetapi kurang optimal untuk deteksi dini churn. Fokus perbaikan harus pada peningkatan recall kelas 1 tanpa mengorbankan precision secara signifikan.


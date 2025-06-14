# Laporan Proyek Machine Learning - Berlian Ishma Zhafira Sujana

## Domain Proyek

Customer churn merupakan tantangan kritis di industri telekomunikasi dengan dampak finansial signifikan. Menurut penelitian Reichheld, F.F. dan Sasser, W.E. (1990), peningkatan 5% dalam customer retention dapat meningkatkan profitabilitas sebesar 25-95%. Dataset Telco Customer Churn dari Kaggle ini merekam karakteristik 7,043 pelanggan dengan 21 fitur yang mencakup demografi, layanan yang digunakan, dan riwayat pembayaran.

---

## Business Understanding

### Problem Statements

1. **Tingkat churn pelanggan mencapai 26.5%** dari total pelanggan, yang berdampak signifikan terhadap pendapatan dan pertumbuhan perusahaan.
2. **Data bersifat imbalanced**, dengan distribusi kelas `Churn = No` sebanyak \~73% dan `Churn = Yes` sebanyak \~27%, yang menyebabkan risiko bias pada model terhadap kelas mayoritas.

---

### Goals

1. **Membangun model prediksi churn pelanggan** yang dapat mengklasifikasikan apakah pelanggan berpotensi churn atau tidak.
2. **Mengidentifikasi fitur-fitur utama** yang paling berpengaruh terhadap kemungkinan pelanggan melakukan churn, seperti `Contract`, `tenure`, `MonthlyCharges`, dan `PaymentMethod`.

---

### Solution Statements

1.  **Menggunakan algoritma Random Forest Classifier** sebagai model utama dalam prediksi customer churn.

   * Random Forest dipilih karena mampu menangani data dengan kombinasi fitur numerik dan kategorikal (setelah dilakukan encoding).

2.  **Melakukan feature selection dan encoding** terhadap fitur-fitur penting dari dataset.

   * Fitur yang memuat numerik dikonversi ke tipe data numerik (float/integer).
   * Kolom kategorikal  `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, dan `PaymentMethod` diencode menggunakan **categorical encoding**.

3.  **Mengukur kontribusi setiap fitur terhadap prediksi churn menggunakan feature importance dari Random Forest**.

   * Hasil feature importance divisualisasikan dalam bentuk bar chart.

---

###  Metrik Evaluasi yang Digunakan

Model dievaluasi menggunakan metrik berikut:

* **Precision**: Menilai seberapa akurat model dalam memprediksi pelanggan churn.
* **Recall**: Fokus utama karena ingin mendeteksi sebanyak mungkin pelanggan yang berisiko churn.
* **F1-score**: Untuk mengukur keseimbangan antara precision dan recall.
* **Accuracy**: Metrik keseluruhan, namun tidak dijadikan tolok ukur utama karena data bersifat imbalanced.


## Data Understanding
Sumber Dataset: [Telco Customer Churn] - (https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)

Dataset terdiri dari **7043 baris dan 21 kolom**, yang mencakup berbagai aspek pelanggan seperti:

* **Identitas pelanggan**: `customerID`
* **Karakteristik demografi**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Jenis layanan yang digunakan**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
* **Informasi langganan dan pembayaran**: `Contract`, `PaymentMethod`, `PaperlessBilling`, `MonthlyCharges`, `TotalCharges`, `tenure`
* **Kolom target**: `Churn`, yang bernilai `Yes` jika pelanggan churn, dan `No` jika tidak.

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

###  Exploratory Data Analysis (EDA):
Lampiran Grafik EDA:
![Screenshot 2025-05-26 223042](https://github.com/user-attachments/assets/13f6f57b-e23b-4ca4-9342-feccb2b68fec)

* **Tidak ada nilai kosong (missing values)** pada sebagian besar kolom berdasarkan hasil `df.isnull().sum()`. Namun, kolom `TotalCharges` bertipe `object` meskipun berisi angka. Setelah dikonversi ke `float`, ditemukan **11 nilai NaN**, yang akan ditangani pada tahap data preparation.
* **Tidak terdapat duplikat** dalam data.
* Kolom `customerID` merupakan identifier unik yang tidak memiliki kontribusi terhadap prediksi dan akan dihapus.
* **Distribusi target (`Churn`) menunjukkan ketidakseimbangan kelas**:

  * Sekitar **73%** data berada pada kelas `No` (tidak churn)
  * Sekitar **27%** data berada pada kelas `Yes` (churn)
    Hal ini menandakan bahwa dataset bersifat **imbalanced**, yang perlu menjadi perhatian khusus dalam proses pelatihan model.

* Beberapa fitur seperti `Contract`, `PaymentMethod`, `InternetService`, dan `OnlineBackup` memiliki **kategori yang berbeda-beda**, yang perlu diubah menjadi representasi numerik menggunakan encoding pada tahap selanjutnya.

### **Correlation Matrix**
Lampiran Correlation Matrix:
![image](https://github.com/user-attachments/assets/f2bc62dd-e116-42c6-8245-8ab284a64263)

* Setelah proses encoding, dilakukan analisis **matriks korelasi (correlation matrix)** untuk mengetahui hubungan antar fitur. Visualisasi seperti heatmap digunakan untuk mengidentifikasi fitur-fitur yang paling berpengaruh, serta untuk mendeteksi multikolinearitas antar fitur.

Berikut adalah detail dari korelasi antar fitur:
- `gender`: Tingkat korelasi rendah dengan semua fitur
- `SeniorCitizen`: Tingkat korelasi rendah dengan semua fitur
- `Partner`: Tingkat korelasi sebesar 0.45 dengan fitur 'Dependents'
- `Dependents`: Tingkat korelasi sebesar 0.45 dengan fitur 'Partner'
- `PhoneService`: Tingkat korelasi sebesar 0.68 dengan fitur 'MultipleLines'
- `MultipleLines`: Tingkat korelasi sebesar 0.68 dengan fitur 'PhoneService'
- `InternetService`: Tingkat korelasi sebesar 0.72 dengan fitur 'TechSupport' dan 'OnlineSecurity'
- `OnlineSecurity`: Tingkat korelasi sebesar 0.74 dengan fitur 'TechSupport'
- `OnlineBackup`: Tingkat korelasi sebesar 0.71 dengan fitur 'TechSupport', 'DeviceProtection', 'MonthlyCharges' dan 'OnlineSecurity'
- `DeviceProtection`: Tingkat korelasi sebesar 0.75 dengan fitur 'StreamingTV' dan 'StreamingFilm'
- `TechSupport`: Tingkat korelasi sebesar 0.74 dengan fitur 'OnlineSecurity'
- `StreamingTV` & `StreamingMovies`: Tingkat korelasi sebesar 0.82 dengan fitur 'MonthlyCharges'
- `Contract`: Tingkat korelasi sebesar 0.67 dengan fitur 'tenure'
- `PaymentMethod`: Tingkat korelasi sebesar 0.35 dengan fitur 'Contract'
- `PaperlessBilling`: Tingkat korelasi sebesar 0.35 dengan fitur 'MonthlyCharges'
- `tenure`: Tingkat korelasi sebesar 0.83 dengan fitur 'TotalCharges'
- `MonthlyCharges`: Tingkat korelasi sebesar 0.82 dengan fitur `StreamingTV` & `StreamingMovies`
- `TotalCharges`: Tingkat korelasi sebesar 0.83 dengan fitur 'tenure'
- `Churn`: Tingkat korelasi sebesar 0.19 dengan fitur 'MonthlyCharges' dan 'PaperlessBilling'


## Data Preparation
Tahap ini mencakup proses transformasi data mentah menjadi bentuk yang siap digunakan untuk pelatihan model machine learning. Berikut adalah langkah-langkah yang dilakukan:

---

###  1. **Transform Tipe Data**

* Dilakukan konversi kolom `TotalCharges` dari tipe `object` ke `float` menggunakan `pd.to_numeric(errors='coerce')`.

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
```

---

###  2. **Handling Missing Values**

* Awalnya, dataset tidak memiliki missing value eksplisit (`df.isnull().sum()` menunjukkan 0 pada semua kolom).
* Namun, setelah dilakukan konversi kolom `TotalCharges`, sebanyak **11 nilai menjadi `NaN`** karena berisi spasi atau string kosong.
* Missing value ini kemudian ditangani dengan cara **mengisi nilai yang hilang menggunakan median** dari kolom `TotalCharges`. Penggunaan median dipilih karena lebih tahan terhadap outlier dibandingkan mean.

```python
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
```

---

###  3. **Drop Kolom Tidak Relevan**

* Kolom `customerID` dihapus karena merupakan **identifier unik** yang tidak memiliki nilai prediktif dalam klasifikasi churn.

```python
df.drop(columns=['customerID'], inplace=True)
```

---

###  4. **Encoding Kategorikal**

* Untuk dapat digunakan dalam pemodelan, fitur kategorikal perlu diubah ke format numerik.
* Encoding dilakukan dengan pendekatan **label encoding**, mengubah kategori ke nilai 0/1/2/dst. secara eksplisit, terutama untuk fitur-fitur biner seperti `gender`, `Partner`, `Dependents`, `PaperlessBilling`, dan lainnya.

```python
# Encode fitur kategorikal
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['MultipleLines'] = df['MultipleLines'].map({'Yes': 2, 'No': 1, 'No phone service': 0})
df['InternetService'] = df['InternetService'].map({'DSL': 2, 'Fiber optic': 1, 'No': 0})
df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['TechSupport'] = df['TechSupport'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['StreamingTV'] = df['StreamingTV'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 2, 'No': 1, 'No internet service': 0})
df['Contract'] = df['Contract'].map({'Two year': 2, 'One year': 1, 'Month-to-month': 0})
df['PaymentMethod'] = df['PaymentMethod'].map({'Bank transfer (automatic)':3, 'Credit card (automatic)': 2, 'Mailed check': 1, 'Electronic check': 0})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
```

---

###  5. **Feature Selection dan Final Dataset**

* Dataset akhir yang digunakan untuk pelatihan model terdiri atas kombinasi fitur numerik dan hasil encoding fitur kategorikal.
* Kolom target yang diprediksi adalah `Churn`.

---

### 6. Pembagian Dataset (Train-Test Split)

* Dataset dibagi menjadi data **pelatihan (training)** dan **pengujian (testing)** menggunakan fungsi `train_test_split` dari `sklearn.model_selection`.
* **Proporsi pembagian** adalah 80% data untuk pelatihan dan 20% untuk pengujian.
* Parameter `random_state=42` digunakan agar proses pembagian data bersifat **reproducible** (hasil selalu konsisten setiap dijalankan).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---



## Modeling

Model yang digunakan dalam proyek ini adalah **Random Forest Classifier**, yaitu algoritma ensemble berbasis decision tree yang sangat populer dalam klasifikasi dan regresi.

###  Cara Kerja Algoritma Random Forest:

Random Forest bekerja dengan membangun **banyak pohon keputusan (decision trees)** pada subset acak dari data pelatihan, lalu menggabungkan hasil prediksi dari masing-masing pohon (melalui voting mayoritas untuk klasifikasi) untuk menghasilkan prediksi akhir. Proses ini dikenal sebagai **bagging (Bootstrap Aggregating)**, di mana setiap pohon:

* Dilatih dengan data yang di-*sampling* secara acak dengan pengembalian (bootstrapping).
* Pada setiap node, hanya subset acak dari fitur yang dipertimbangkan untuk split, sehingga meningkatkan keberagaman antar pohon.

**Keuntungan utama** Random Forest adalah:

* Lebih tahan terhadap **overfitting** dibanding single decision tree.
* Mampu menangani fitur kategorikal dan numerik secara bersamaan.
* Memberikan estimasi **feature importance**, yang berguna untuk interpretasi model.

---

###  Parameter Model yang Digunakan

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

###  Feature Importance:
Lampiran Grafik Feature Importance:
![image](https://github.com/user-attachments/assets/8cb02cfd-f076-4558-8e77-7eafa7830107)
* Setelah model dilatih, kita dapat mengevaluasi **seberapa besar kontribusi masing-masing fitur** terhadap prediksi churn dengan melihat `feature_importances_`.
* Fitur dengan nilai importance lebih tinggi memiliki **pengaruh lebih besar** terhadap keputusan model.
* Hasil ini dirangkum ke dalam dataframe dan diurutkan menurun, kemudian divisualisasikan dalam bentuk horizontal bar chart.
* Fitur seperti `TotalCharges`, `PaymentMethod`, `Contract`, `tenure`, dan `MonthlyCharges` berada di posisi atas, menunjukkan bahwa:

  * **Biaya total (TotalCharges)** dan **Biaya bulanan (MonthlyCharges)** yang tinggi bisa menjadi faktor risiko churn.
  * **Jenis kontrak** pelanggan (bulanan, tahunan) sangat memengaruhi kemungkinan churn.
  * **Lama berlangganan (tenure)** umumnya memiliki korelasi negatif dengan churn (semakin lama, semakin kecil kemungkinan churn).
  * **Metode pembayaran (PaymentMethod)** model prediksi churn karena metode pembayaran pelanggan dapat merefleksikan preferensi, kebiasaan, dan kemungkinan loyalitas mereka terhadap layanan.

## Evaluation
### A. Metrik Evaluasi yang Digunakan
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



### B. Analisis Hasil Evaluasi

| Metrik    | Kelas 0 (No Churn) | Kelas 1 (Churn) |
| --------- | ------------------ | --------------- |
| Precision | 0.83               | 0.66            |
| Recall    | 0.91               | 0.47            |
| F1-Score  | 0.87               | 0.55            |
| Support   | 1036               | 373             |

### C. Interpretasi:  
####  **Kelas 0 (Tidak Churn):**

* **Precision 0.83:** Dari seluruh prediksi "tidak churn", sebanyak 83% benar.
* **Recall 0.91:** Model berhasil menangkap 91% pelanggan yang memang tidak churn.
* **F1-score 0.87:** Performa keseluruhan untuk kelas mayoritas sangat baik.
* Ini wajar karena kelas 0 adalah **kelas dominan (sekitar 73%)**, sehingga model memiliki cukup data untuk belajar mengenalinya.

####  **Kelas 1 (Churn):**

* **Precision 0.66:** Dari semua prediksi "churn", hanya 66% yang benar.
* **Recall 0.47:** Model hanya berhasil menangkap 47% pelanggan yang benar-benar churn.
* **F1-score 0.55:** Menunjukkan bahwa model masih **kesulitan mengenali kelas minoritas (churn)** secara akurat.

####  **Akurasi dan Rata-Rata:**
* **Accuracy 0.79:** Model secara keseluruhan benar memprediksi 79% data.
* **Macro average:**

  * Rata-rata dari masing-masing kelas tanpa mempertimbangkan proporsi data.
  * **Recall macro hanya 0.69**, menunjukkan bahwa **kelas minoritas tidak dipelajari dengan baik**.
* **Weighted average:**

  * Rata-rata yang mempertimbangkan jumlah sampel per kelas.
  * Nilainya lebih tinggi karena dominasi kelas 0.


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


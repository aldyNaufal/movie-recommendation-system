
# 🎬 CineMatch: Personalized Movie Recommendations with Deep Learning

**Personalized Movie Suggestions Based on User Behavior**

---

![movie-banner](images/13067.jpg)

## 📌 1. Domain Proyek: Entertainment & Personalized Recommendation

Industri hiburan digital mengalami lonjakan konsumsi konten selama dekade terakhir, khususnya film dan serial yang tersedia di berbagai platform streaming seperti Netflix, Disney+, dan Amazon Prime. Dalam era banjir informasi ini, pengguna sering mengalami kesulitan dalam memilih tontonan yang sesuai dengan preferensi mereka.

Salah satu solusi untuk permasalahan ini adalah sistem rekomendasi film yang cerdas, yang dapat mempelajari kebiasaan pengguna dan menyarankan konten yang relevan. Menurut Ricci, Rokach, dan Shapira (2015), collaborative filtering terbukti menjadi salah satu pendekatan paling efektif dalam memahami pola minat pengguna, karena teknik ini dapat menghasilkan rekomendasi hanya berdasarkan interaksi pengguna seperti rating dan ulasan, tanpa memerlukan analisis konten film.

Lebih lanjut, penelitian oleh Koren, Bell, dan Volinsky (2009) menunjukkan bahwa pendekatan *matrix factorization* dalam collaborative filtering mampu secara signifikan meningkatkan akurasi prediksi sistem rekomendasi, terutama dalam konteks dataset skala besar seperti MovieLens.

Proyek ini bertujuan untuk membangun sistem rekomendasi film berbasis collaborative filtering menggunakan data dari Kaggle (Sayan0211, 2022). Dataset ini berisi rating dari pengguna terhadap berbagai film, yang kemudian digunakan untuk memprediksi preferensi pengguna lain terhadap film yang belum mereka tonton.



---

## 🎯 2. Business Understanding

### 🔍 Problem Statements

1. Bagaimana memberikan rekomendasi film yang relevan untuk pengguna baru maupun lama berdasarkan riwayat rating pengguna lain?
2. Bagaimana membangun model *collaborative filtering* berbasis deep learning untuk mengatasi masalah *sparsity* pada data rating film?
3. Bagaimana memanfaatkan informasi tambahan seperti tahun rilis dan rating film sebagai fitur untuk meningkatkan akurasi model?

---

### 🎯 Objectives (Revisi)

1. **Membangun sistem rekomendasi berbasis deep learning** yang dapat mempelajari representasi laten dari pengguna dan film menggunakan pendekatan *neural collaborative filtering (NCF)*.
2. **Mengintegrasikan embedding pengguna dan film dengan metadata film**, seperti rating rata-rata film (`movie_rating`) dan tahun rilis (`year`), untuk memperkaya fitur input model dan meningkatkan kualitas prediksi.
3. **Melakukan evaluasi menyeluruh terhadap model** menggunakan metrik RMSE, MAE, R², serta korelasi Pearson dan Spearman, untuk menilai seberapa akurat dan relevan rekomendasi yang dihasilkan.

---

### 💡 Solutions (Revisi)

1. **Merancang dan melatih model rekomendasi menggunakan TensorFlow/Keras**, dengan input multi-fitur: `user_id`, `movie_id`, `movie_rating`, dan `year`. Embedding digunakan untuk memetakan entitas ke dalam representasi vektor yang lebih informatif.
2. **Menggunakan teknik regularisasi dan callback selama pelatihan**, seperti `Dropout`, `L2 Regularization`, `EarlyStopping`, `ReduceLROnPlateau`, dan `ModelCheckpoint`, untuk mencegah overfitting dan meningkatkan generalisasi model.
3. **Mengevaluasi performa model menggunakan data uji**, dengan metrik: RMSE, MAE, R², Pearson correlation, Spearman correlation
4. **Mengombinasikan hasil embedding dengan interaksi non-linear (dot product dan element-wise multiplication)** serta metadata tambahan, lalu meneruskannya ke jaringan dense berlapis untuk mengatasi keterbatasan model klasik seperti matrix factorization yang bersifat linear.


---

## 📁 3. Dataset Overview

* **Sumber**: Kaggle - [Movie Recomendation pjct by sayan0211](https://www.kaggle.com/datasets/sayan0211/movie-recomendation-pjct)
* **Jumlah entri rating**: 100.836 baris
* **Jumlah film**: 9.729 judul unik
* **Penulis dataset**: sayan0211

---

## 📋 4. Fitur Dataset

### 📘 `ratings.csv`

| Kolom       | Tipe    | Deskripsi                                                        |
| ----------- | ------- | ---------------------------------------------------------------- |
| `userId`    | int64   | ID unik pengguna                                                 |
| `movieId`   | int64   | ID unik film (digunakan sebagai referensi ke tabel `movies.csv`) |
| `rating`    | float64 | Rating yang diberikan pengguna terhadap film (0.5 - 5.0)         |
| `timestamp` | int64   | Waktu rating diberikan (dalam UNIX timestamp)                    |

### 🎞 `movies.csv` 
| Kolom     | Tipe   | Deskripsi                                    |     |
| --------- | ------ | -------------------------------------------- | --- |
| `movieId` | int64  | ID unik film                                 |     |
| `title`   | object | Judul film                                   |     |
| `genres`  | object | Genre film (dipisahkan oleh pipe \`          | \`) | 

---

## 🔍 5. Data Understanding

### Statistik Umum:

* Dataset film terdiri dari **9.742 film**.
* Data rating berisi **100.818 rating** yang diberikan oleh pengguna terhadap film-film tersebut.
* Fitur utama film:

  * `title` yang sudah dibersihkan dan diubah menjadi huruf kecil tanpa tahun
  * `year` sebagai tahun rilis film (tipe data objek/string)
  * `genres` sebagai kategori genre film yang sudah dipisah dengan koma dan dalam huruf kecil
* Tahun rilis film (`year`) memiliki beberapa missing value awalnya (13 baris), namun sudah dihapus dalam pembersihan data.




---

### 🔎 Pemeriksaan Duplikasi dan Missing Value:

#### 1. ✅ **Duplikasi Data**

```python
data_movie[data_movie.duplicated()]
data_rating[data_rating.duplicated()]
```

* **Hasil**:

  * Tidak ditemukan baris duplikat pada dataset film maupun dataset rating.
  * Menunjukkan data sudah bersih dari pengulangan yang identik.

#### 2. 🚫 **Missing Value**

| Dataset          | Fitur  | Jumlah Missing |
| ---------------- | ------ | -------------- |
| **data\_movie**  | `year` | 13             |
| **data\_rating** | -      | 0              |

* Missing value hanya ditemukan di fitur `year` pada dataset film, dengan 13 baris yang hilang.

---


### 1. Distribusi Rating oleh User

![Distribusi Rating](images/distribusi_rating_user.png)

**Insight:**

* Rating yang diberikan oleh user cenderung terkonsentrasi pada nilai 3, 4, dan 5.
* Puncak tertinggi berada di rating 4, yang berarti banyak pengguna memberikan rating cukup tinggi ke film.
* Distribusi ini menunjukkan bahwa pengguna cenderung memberikan rating yang positif atau cukup puas terhadap film-film yang mereka tonton.
* Adanya kurva KDE (kernel density estimation) juga memperlihatkan variasi rating secara halus, mengindikasikan rating yang tidak terlalu tersebar merata di semua nilai.

---

### 2. Rata-rata Rating User per Tahun

![Rata-rata Rating User per Tahun](images/rata-rata_rating_user_per_tahun.png)

**Insight:**

* Rata-rata rating per tahun relatif stabil dengan fluktuasi kecil antara 3.3 sampai hampir 3.9.
* Terdapat beberapa tahun dengan kenaikan rata-rata rating, misalnya sekitar tahun 1998 dan 2013, yang mungkin menunjukkan tahun-tahun dengan film-film yang lebih disukai oleh pengguna.
* Penurunan pada beberapa tahun tertentu mengindikasikan mungkin ada film yang kurang memuaskan di tahun-tahun tersebut.
* Tren ini bisa membantu mengidentifikasi periode film dengan kualitas rating yang lebih baik atau kurang baik menurut pengguna.

---

### 3. Jumlah Film Dirilis per Tahun

![Jumlah Film Dirilis per Tahun](images/jumlah_film_per_tahun.png)

**Insight:**

* Jumlah film yang dirilis meningkat secara signifikan sejak tahun 1950-an hingga mencapai puncaknya sekitar tahun 2000-an.
* Setelah mencapai puncak sekitar tahun 2000-2010, terdapat penurunan jumlah film yang dirilis di tahun-tahun berikutnya.
* Penurunan drastis di akhir grafik bisa jadi karena data film terbaru yang belum lengkap atau ada perubahan tren produksi film.
* Grafik ini menunjukkan perkembangan industri film yang semakin berkembang pesat di abad ke-20, dengan puncak produksi film di awal abad ke-21.

---

### 4. 10 Genre Terbanyak

![10 Genre Terbanyak](images/genre_terbanyak.png)

**Insight:**

* Genre comedy dan drama mendominasi dengan jumlah film terbanyak dibanding genre lain.
* Kombinasi genre seperti comedy romance, comedy drama romance, dan comedy drama juga cukup banyak, menunjukkan kecenderungan film untuk menggabungkan beberapa genre populer.
* Genre action dengan tambahan subgenre seperti adventure dan thriller berada di posisi yang lebih rendah, tapi tetap signifikan.
* Informasi ini bisa berguna untuk memahami preferensi pasar dan tren genre film yang banyak diproduksi serta diminati.

---

## 🧹 6. Data Preparation

Setelah dilakukan eksplorasi dan penggabungan antara data `movies.csv` dan `ratings.csv`, dilakukan tahap **data preparation** untuk memastikan kualitas data siap untuk digunakan dalam proses pemodelan. Langkah-langkah ini dilakukan secara sistematis berdasarkan temuan eksplorasi data dan kebutuhan algoritma rekomendasi yang akan digunakan.

---

### 📌 Temuan Kunci dan Strategi Preprocessing

1. **Format Judul Film Tidak Konsisten:**
   Beberapa judul film mengandung tahun rilis dalam tanda kurung yang perlu dipisahkan agar kolom `title` bersih dan kolom `year` dapat digunakan sebagai fitur numerik. Regex digunakan untuk mengekstrak `year` dan membersihkan `title`.

2. **Genre Berbentuk String Gabungan:**
   Kolom `genres` awalnya berisi string panjang dengan delimiter `'|'`. Untuk mempermudah analisis konten dan kompatibilitas pemrosesan NLP atau representasi multi-label, karakter pemisah diubah menjadi koma, dan teks dijadikan huruf kecil.

3. **Missing Value pada Kolom Year:**
   Terdapat nilai kosong pada kolom `year` akibat film yang tidak mencantumkan tahun pada `title`. Baris-baris ini dihapus karena informasi tahun dianggap penting dan tidak dapat diimputasi secara akurat.

4. **Timestamp Rating:**
   Kolom `timestamp` dikonversi menjadi format waktu agar bisa dimanfaatkan lebih lanjut, seperti ekstraksi waktu menonton jika dibutuhkan (tidak digunakan dalam model saat ini).

5. **Rata-Rata Rating Film:**
   Untuk menambahkan informasi global tentang kualitas film, dihitung rata-rata rating (`movie_rating`) dari setiap `movieId`. Ini memberikan konteks tambahan pada film di luar penilaian individu pengguna.

---

### 📋 Rangkuman Langkah-Langkah Preprocessing:

| Langkah               | Penjelasan                                                                                                                                                                      |                                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Ekstraksi Tahun**   | Menggunakan **regex** untuk mengekstrak angka 4 digit dalam kurung pada `title`, lalu dipindahkan ke kolom baru `year`. Judul diformat ulang menjadi lowercase dan tanpa tahun. |                                                                                                    |
| **Pembersihan Genre** | Genre diformat menjadi **huruf kecil** dan delimiter \`                                                                                                                         | `diubah menjadi`,\` agar mudah diolah dan lebih kompatibel untuk representasi fitur berbasis teks. |
| **Missing Value**     | Baris dengan `year` kosong dihapus karena dianggap sebagai fitur penting dan sulit diimputasi secara akurat.                                                                    |                                                                                                    |
| **Gabung Dataset**    | Data `ratings` digabung dengan data film (`movies`) berdasarkan `movieId`. Hal ini dilakukan agar setiap rating memiliki metadata terkait film.                                 |                                                                                                    |
| **Rata-Rata Rating**  | Dihitung rata-rata rating per `movieId` sebagai representasi kualitas umum film, dan ditambahkan ke dalam dataset sebagai kolom `movie_rating`.                                 |                                                                                                    |
| **Encoding ID**       | Menggunakan **Label Encoding** untuk `userId` dan `movieId` karena akan digunakan sebagai input numerik dalam model rekomendasi berbasis embedding.                             |                                                                                                    |
| **Scaling**           | Kolom `movie_rating` dan `year` discale menggunakan **StandardScaler** karena keduanya akan dimasukkan ke dalam model sebagai fitur numerik.                                    |                                                                                                    |
| **Data Mapping**      | Dibuat kamus untuk memetakan `userId` dan `movieId` ke bentuk terenkode dan sebaliknya, untuk keperluan interpretasi hasil prediksi model.                                      |                                                                                                    |
| **Data Split**        | Dataset dibagi menjadi: **80% train+val**, **20% test**. Kemudian, dari train+val, diambil **10% untuk validasi**, menghasilkan rasio train\:val\:test sebesar **72%:8%:20%**.  |                                                                                                    |

---

### 🧭 Alasan Penggunaan Scaling dan Encoding

* **StandardScaler** digunakan untuk `year` dan `movie_rating` agar keduanya berada pada skala seragam dan tidak mendominasi perhitungan jarak dalam model. Karena nilai-nilai ini tidak mengandung outlier ekstrem, **StandardScaler** lebih cocok dibanding `RobustScaler`.

* **Label Encoding** pada `userId` dan `movieId` diperlukan agar data ID bisa digunakan dalam lapisan embedding model deep learning, tanpa menambah dimensi seperti pada One-Hot Encoding.

---

### ✅ Final Feature yang Digunakan:

| Fitur             | Tipe        | Keterangan                                |
| ----------------- | ----------- | ----------------------------------------- |
| `user_encoded`    | Kategorikal | ID user yang sudah di-label encode        |
| `movie_encoded`   | Kategorikal | ID film yang sudah di-label encode        |
| `movie_rating`    | Numerik     | Rata-rata rating film                     |
| `year`            | Numerik     | Tahun rilis film                          |
| `rating` (target) | Numerik     | Nilai rating yang diberikan oleh pengguna |

---

Dengan pendekatan **berbasis data dan domain knowledge** ini, data sudah disiapkan untuk memasuki tahap **modelling rekomendasi** menggunakan pendekatan pembelajaran terawasi maupun berbasis embedding.

---

Berikut adalah **penjelasan lengkap model custom Neural Network (NN)** yang Anda buat, dengan gaya yang **selaras** dengan struktur dokumentasi model sebelumnya:

---

## 7. Modelling
🔹 Model : **Custom Neural Network for Rating Prediction*
### ✅ Alasan Pemilihan:

Model ini dirancang khusus untuk menangani data **user-item interaction**, seperti sistem rekomendasi film berbasis rating. Model menggabungkan teknik **embedding**, **interaksi eksplisit (dot product dan element-wise multiply)**, serta **informasi tambahan** (seperti rating sebelumnya dan tahun rilis) dalam arsitektur neural network. Pendekatan ini jauh lebih fleksibel dan mampu **menangkap representasi latar belakang pengguna dan film** secara simultan.

---

### ⚙️ Cara Kerja:

Model ini bekerja melalui beberapa tahap:

1. **Embedding Layer**:

   * Mengubah `user_id` dan `movie_id` menjadi representasi vektor berdimensi rendah (`embedding_dim = 64`) yang dapat dipelajari.
   * Regularisasi L2 (`l2_reg`) digunakan untuk mencegah overfitting pada representasi pengguna dan film.

2. **Interaction Features**:

   * Menggunakan **dot product** dan **element-wise multiply** dari vektor user dan movie sebagai bentuk **interaksi eksplisit** antara keduanya.
   * Dikombinasikan dengan fitur tambahan seperti `movie_rating_input` dan `year_input`.

3. **Fully Connected Layers**:

   * Vektor hasil concatenation melewati beberapa lapisan dense dengan ukuran `[128, 64, 32]`, masing-masing dilengkapi dengan:

     * Aktivasi `ReLU`
     * Regularisasi L2
     * **Batch Normalization** untuk stabilitas pelatihan
     * **Dropout (0.3)** untuk mengurangi overfitting

4. **Output Layer**:

   * Lapisan akhir adalah neuron tunggal dengan aktivasi linear (`Dense(1, activation='linear')`) yang merepresentasikan **prediksi rating**.

---

### 🧩 Struktur Arsitektur:

```text
Input:
  - user_id (int)
  - movie_id (int)
  - movie_rating (float)
  - year (int)

Layers:
  - Embedding user_id → user_vec
  - Embedding movie_id → movie_vec
  - Dot(user_vec, movie_vec)
  - Multiply(user_vec, movie_vec)
  - Concatenate([user_vec, movie_vec, dot, multiply, movie_rating, year])
  - Dense(128) → BN → Dropout(0.3)
  - Dense(64) → BN → Dropout(0.3)
  - Dense(32) → BN → Dropout(0.3)
  - Dense(1) → Output rating
```

---

### ✅ Kelebihan:

* **Fleksibel dan powerful**, dapat mempelajari hubungan kompleks antar pengguna dan item.
* Menggunakan kombinasi representasi *latent* dan fitur numerik eksplisit untuk meningkatkan prediksi.
* **Mudah dikembangkan lebih lanjut**, misalnya menambah genre, lokasi, waktu tonton, dll.
* Sudah menggunakan teknik regularisasi modern seperti Dropout dan BatchNormalization.

---

### ❌ Kekurangan:

* **Butuh banyak data** untuk generalisasi yang baik.
* Interpretasi hasil lebih sulit dibanding model linear atau tree-based.
* Proses pelatihan lebih lambat, terutama jika data besar.

---

### ⚙️ Konfigurasi Model:

| Parameter       | Nilai                        |
| --------------- | ---------------------------- |
| `embedding_dim` | 64                           |
| `hidden_units`  | \[128, 64, 32]               |
| `dropout_rate`  | 0.3                          |
| `l2_reg`        | 0.001                        |
| `optimizer`     | Adam (default keras compile) |
| `loss`          | Mean Squared Error (MSE)     |

---

### 📌 Summary:

| Model            | Pola yang Bisa Ditangkap           | Tuning                | Interpretasi | Outlier Friendly | Hasil Akhir     |
| ---------------- | ---------------------------------- | --------------------- | ------------ | ---------------- | --------------- |
| Custom NN (Anda) | Non-linear, Representasi User-Item | ✅ (Manual & Flexible) | ✖️           | ✅                | **Sangat Baik** |


---

## 📊 Evaluasi Model

Dalam sistem rekomendasi berbasis prediksi rating, penting untuk mengetahui seberapa dekat prediksi model dengan rating aktual yang diberikan oleh pengguna. Oleh karena itu, digunakan beberapa metrik evaluasi untuk menilai performa model secara menyeluruh, baik dari segi kesalahan absolut, penyebaran prediksi, hingga kekuatan hubungan antara nilai prediksi dan aktual.

### 🎯 Alasan Penggunaan Metrik

Metrik evaluasi dipilih berdasarkan sifat prediksi regresi dari model rekomendasi ini (yaitu memprediksi skor rating numerik). Setiap metrik memberikan perspektif yang berbeda:

* **RMSE** dan **MAE** digunakan untuk mengukur seberapa besar kesalahan prediksi model secara absolut dan akar kuadrat rata-rata.
* **R² (Koefisien Determinasi)** mengukur proporsi variansi rating aktual yang dapat dijelaskan oleh prediksi model.
* **Pearson correlation** dan **Spearman correlation** mengukur kekuatan hubungan antara prediksi dan nilai aktual secara linier maupun ordinal.

---

### 📐 Rumus Metrik Evaluasi

Berikut adalah definisi matematis dari metrik yang digunakan:

#### 1. Root Mean Square Error (RMSE)

RMSE mengukur rata-rata kesalahan prediksi kuadrat dan sangat sensitif terhadap kesalahan besar.

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

> Di mana $y_i$ adalah rating aktual, dan $\hat{y}_i$ adalah rating prediksi ke-i.

#### 2. Mean Absolute Error (MAE)

MAE menghitung rata-rata dari semua kesalahan absolut antara prediksi dan nilai aktual.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

> MAE lebih tahan terhadap outlier dibanding RMSE.

#### 3. Coefficient of Determination (R²)

R² menunjukkan seberapa besar variansi dalam data target yang dapat dijelaskan oleh model.

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

> Nilai $R^2$ berkisar dari $-\infty$ hingga 1. Semakin mendekati 1, semakin baik model menjelaskan variansi data.

#### 4. Pearson Correlation Coefficient (ρ)

Pearson mengukur korelasi linier antara prediksi dan data aktual.

$$
\rho = \frac{\sum (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum (y_i - \bar{y})^2} \sqrt{\sum (\hat{y}_i - \bar{\hat{y}})^2}}
$$

> Nilai berkisar dari -1 hingga 1. Nilai positif tinggi menunjukkan hubungan linier yang kuat.

#### 5. Spearman Rank Correlation (ρₛ)

Spearman mengukur hubungan monoton antar dua variabel dengan melihat peringkat, bukan nilai asli.

$$
\rho_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

> Di mana $d_i$ adalah selisih antara peringkat prediksi dan aktual. Cocok ketika urutan lebih penting dari nilai absolut.

---

### 📌 Kesimpulan Pemilihan Metrik

Kombinasi metrik ini memungkinkan evaluasi dari berbagai sudut pandang:

* **RMSE** menangkap seberapa besar kesalahan besar berdampak ke model.
* **MAE** memberi gambaran langsung rata-rata kesalahan tanpa efek kuadrat.
* **R²** menunjukkan kemampuan model dalam menjelaskan data.
* **Pearson** dan **Spearman** menilai kekuatan hubungan prediksi dan target, penting ketika mempertimbangkan urutan dan hubungan linier rating.

Melalui hasil evaluasi berikut:

* **RMSE**: 0.8075
* **MAE**: 0.6163
* **R²**: 0.4044
* **Pearson**: 0.6389
* **Spearman**: 0.6297

dapat disimpulkan bahwa model berhasil membangun hubungan yang cukup kuat antara fitur input dengan output rating pengguna, serta cukup efektif dalam merepresentasikan pola preferensi pengguna dalam sistem rekomendasi.

---

### Kesimpulan Hasil Evaluasi 

#### 1. **Bagaimana membangun sistem rekomendasi film menggunakan pendekatan deep learning untuk menangkap pola preferensi pengguna dari data rating dan metadata film?**

Sistem rekomendasi dibangun menggunakan pendekatan deep learning berbasis **collaborative filtering**, di mana dua jenis embedding — untuk pengguna dan film — digunakan untuk menangkap hubungan laten antara pengguna dan preferensi film. Selain itu, informasi numerik tambahan seperti **rating film rata-rata** dan **tahun rilis film** digunakan sebagai fitur pendukung (metadata film) yang dikombinasikan melalui arsitektur neural network berlapis. Model ini dirancang untuk belajar dari pola rating historis yang diberikan pengguna terhadap film-film tertentu, sehingga mampu memprediksi rating baru yang kemungkinan besar akan diberikan pengguna terhadap film lainnya.

---

#### 2. **Bagaimana memanfaatkan embedding pengguna dan film serta fitur tambahan seperti rating film dan tahun rilis untuk meningkatkan akurasi prediksi rating?**

Embedding pengguna dan film digunakan untuk merepresentasikan identitas masing-masing dalam bentuk vektor berdimensi tetap, yang memungkinkan model mengenali hubungan kompleks antar entitas tersebut. Kombinasi dari **dot product** dan **element-wise multiplication** antara embedding pengguna dan film kemudian digabungkan dengan fitur numerik tambahan — yaitu **skor rata-rata film (`movie_rating_scaled`)** dan **tahun rilis (`year_scaled`)**. Seluruh fitur ini dikonsolidasikan ke dalam jaringan neural network berlapis (Dense Layer) dengan aktivasi ReLU dan regularisasi (dropout dan L2), sehingga meningkatkan kemampuan model dalam menangkap informasi relevan dari data heterogen, dan secara empiris membantu meningkatkan akurasi prediksi.

---

#### 3. **Sejauh mana pendekatan ini mampu memberikan rekomendasi yang akurat, dan bagaimana kualitas prediksi diukur menggunakan metrik evaluasi seperti RMSE, MAE, R², dan korelasi?**

Model yang dibangun menunjukkan performa yang cukup baik dalam melakukan prediksi rating. Berdasarkan hasil evaluasi terhadap data uji, diperoleh skor **RMSE sebesar 0.8075**, **MAE sebesar 0.6163**, dan **R² sebesar 0.4044**. Nilai RMSE dan MAE menunjukkan bahwa rata-rata kesalahan prediksi berada di bawah 1 poin skala rating, sedangkan nilai R² menunjukkan bahwa model mampu menjelaskan sekitar 40% variansi dalam data rating aktual. Di sisi lain, nilai korelasi **Pearson sebesar 0.6389** dan **Spearman sebesar 0.6297** menunjukkan adanya hubungan linier dan ranking yang cukup kuat antara prediksi model dan data aktual. Hasil ini menunjukkan bahwa pendekatan deep learning yang digunakan memiliki efektivitas yang cukup baik dalam menangkap pola preferensi pengguna dan memberikan rekomendasi yang relevan.




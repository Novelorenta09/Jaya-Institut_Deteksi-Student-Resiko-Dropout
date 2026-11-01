# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Permasalahan Bisnis
Tingginya jumlah siswa di Jaya Institut yang tidak menyelesaikan pendidikannya atau mengalami droput. Hal ini menjadi masalah besar bagi institusi pendidikan karena dapat mempengaruhi reputasi institusi dan kualitas kelulusan.

### Cakupan Proyek
- Proyek dikerjakan menggunakan tools seperti supabase, metabase (dashboard), python, Jupyternotebook, Streamlit (protoypye Model)
- Model dibuat menggunakan Machine Learning yakni  algoritma Gradient Boosting dengan metode LGBMClassifier
- Data bersumber dari Dicoding
- Permasalahan proyek terkait mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus

### Persiapan
- Versi metabese: 0.63.19.1.
- Versi Python: 3.14.7
- Versi Scikit-learn: 1.9.1
- email: root@mail.com
- pass: root123
- Akses: http://localhost:3000 
- Sumber data: "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/bce7a57a496d083716138922bc5839b5c30fa4ea/students_performance/data.csv"


Setup environment:
```
- pip install -r requirements.txt
- pipenv install jupyter
- pipenv shell
- jupyter notebook 
```

## Business Dashboard
Monitoring / Operational Dashboard dengan fokus Menjawab:
- Total keseluruhan siswa, Total dropout, Total aktif, Persentase dropout
- Isi chart: Visual dari status siswa berdasarkan kelas pagi atau siang, beasiswa berdasarkan status, jumlah siswa yang dropout per tiap jurusan, serta jumlah yang dropout berdasarkan jenis kelamin.
- Link Here (http://localhost:3000/public/dashboard/d3bb0f5b-92e3-4285-9275-8b5491c8a129)


## Menjalankan Sistem Machine Learning
Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.

```
- git clone https://github.com/Novelorenta09/jaya-institut_deteksi-student-resiko-dropout.git
- cd jaya-institut_deteksi-student-resiko-dropout
- pip install -r requirements.txt
- streamlit run data_deploy.py
- Link prototype:  https://jaya-institutdeteksi-student-resiko-dropout-hyphlrdpb7dphckagl.streamlit.app/
```

## Conclusion
Berdasarkan dashboard Metabase:
1. Tingkat Risiko Dropout yang Signifikan: Dari total 4.424 siswa, institusi menghadapi tantangan retensi yang cukup besar dengan angka dropout rate mencapai 32,12% (1.421 siswa putus sekolah), sementara siswa yang masih aktif (enrolled) berada di angka 794 orang.
2. Mayoritas besar aktivitas perkuliahan terkonsentrasi pada Kelas Siang (Daytime) sebesar 89,1% dibandingkan Kelas Malam (Evening) 10,9%.
3. Siswa yang berstatus dropout lebih mendominasi terhadap perempuan hingga mencapai 64% dibanding laki-laki yang hanya 35%
4. Siswa yang menerima beasiswa justru lebih banyak yang dropout yakni 168 orang dibanding dengan yang tidak menerima beasiswa yang hanya 17 orang.
5. Jurusan yang paling tinggi berdasarkan status dropout adalah diduduki jurusan "Management (evening attedence)" 136 orang, selanjutnya ada "Management" 134 orang, dan di posisi ke-3 adalah jurusan "Nursing" 118 orang.

Berdasarkan analisa Model GradientBoosting:
1. Model telah berhasil dilatih secara bersih dengan hanya menggunakan status "Dropout" & "Graduate" sebanyak 2.904 data latih. Berdasarkan evaluasi, model menunjukkan performa baik dengan tingkat akurasi, metrik seperti presisi, recall, f1-score yang optimal.
2. Insight Utama: Melalui pemanfaatan data siswa berstatus Enrolled (yang belum memiliki label akhir) ke dalam model, sistem berhasil melakukan scoring probabilitas risiko secara individu.
Analisis ini mendeteksi adanya sejumlah siswa aktif yang memiliki probabilitas dropout, yang sebagian besar terkonsentrasi pada nilai akademik, uang kuliah dan yang lain.


### Rekomendasi Action Items
1. Keringanan Pembayaran Uang Kuliah yang mana berdasarkan dari mdoel faktor keuangan (tuition fees up to date dan status debtor) merupakan salah satu pendorong  dari model LightGBM, pihak kampus  bagian keuangan perlu membuka program restrukturisasi cicilan atau bantuan subsidi/beasiswa darurat khusus bagi siswa aktif berisiko tinggi agar kendala finansial tidak berujung pada putus kuliah.

2. Pendampingan Akademik Terfokus (Academic Tutoring & Mentoring): Berdasarkan temuan variabel akademik (seperti nilai semester 1 dan semester 2), kampus harus mengaktifkan program bimbingan konseling dan tutoring wajib bagi siswa di jurusan-jurusan dengan tingkat dropout tertinggi untuk mendongkrak performa belajar mereka sejak awal semester.

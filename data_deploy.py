import streamlit as st
import pandas as pd
import joblib

# Load Model dan Template Data
model = joblib.load('model_dropout_lgbm.pkl')
template_df = joblib.load('X_template.pkl')

st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.write("Silakan isi form di bawah ini untuk memprediksi potensi kelulusan atau dropout mahasiswa.")

with st.form("prediction_form"):
    st.subheader("Formulir Data Mahasiswa")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia saat mendaftar (Age at enrollment)", min_value=15, max_value=70, value=20)
        
        # Jenis Kelamin 
        gender = st.selectbox("Jenis Kelamin (Gender)", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        # scholarship holder
        scholarship = st.selectbox("Penerima Beasiswa", options=[0, 1], format_func=lambda x: "Yes" if x == 0 else "No")
        
        # status Pernikahan dengan teks rapi
        marital_status = st.selectbox("Status Pernikahan", options=[1, 2, 3, 4, 5, 6], 
                                     format_func=lambda x: {
                                         1: 'Single', 2: 'Married', 3: 'Widower', 
                                         4: 'Divorced', 5: 'Facto Union', 6: 'Legally Separated'
                                     }[x])
    
    with col2:
        course = st.selectbox("Program Studi (Course)", options=[33, 171, 9003, 9070, 9119, 9147, 9254, 9500],
                             format_func=lambda x: {
                                 33: 'Biofuel Tech', 171: 'Animation', 9003: 'Agronomy', 
                                 9070: 'Design', 9119: 'Informatics', 9147: 'Management', 
                                 9254: 'Tourism', 9500: 'Nursing'
                             }[x])
        
        # Waktu Kuliah
        daytime_attendance = st.selectbox("Waktu Kuliah", options=[1, 0], format_func=lambda x: "Daytime (Siang)" if x == 1 else "Evening (Malam)")

        # Pembyaran uang kulaih
        Tuition_fees_up_to_date = st.selectbox("Pembayaran Uang Kuliah", options=[0, 1], format_func=lambda x: "Lunas" if x == 0 else "Belum Lunas")

        previous_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=120.0)

    submit_button = st.form_submit_button(label="Prediksi Risiko Mahasiswa")

if submit_button:
    # Salin template DataFrame asli dan tidak ada nilai NaN/inf
    input_data = template_df.copy().fillna(0)
    
    # nilai  form yang diinput user
    input_data.loc[0, 'Age_at_enrollment'] = age
    input_data.loc[0, 'Gender'] = gender
    input_data.loc[0, 'Marital_status'] = marital_status
    input_data.loc[0, 'Course'] = course
    input_data.loc[0, 'Daytime_evening_attendance'] = daytime_attendance
    input_data.loc[0, 'Previous_qualification_grade'] = previous_grade
    
    # Konversi tipe data dengan aman (mengabaikan error casting jika ada tipe khusus)
    for col in template_df.columns:
        if col in input_data.columns:
            try:
                # Cek jika tipe data aslinya integer
                if pd.api.types.is_integer_dtype(template_df[col]):
                    input_data[col] = input_data[col].fillna(0).astype('int64')
                else:
                    input_data[col] = input_data[col].astype(template_df[col].dtype)
            except Exception:
                pass # Lewati jika ada kolom yang gagal konversi agar tidak bikin aplikasi crash

    # lakukan Prediksi
    pred_class = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0] # [prob_graduate, prob_dropout]
    
    st.markdown("---")
    st.subheader("Hasil Prediksi Sistem")
    
    if pred_class == 1:
        st.error(f"**BERISIKO DROPOUT**")
        st.write(f"Probabilitas Potensi Dropout: **{pred_proba[1]*100:.2f}%**")
        st.write("*Rekomendasi:* Mahasiswa ini memerlukan perhatian khusus, konseling, atau bimbingan akademik dari pihak kampus.")
    else:
        st.success(f"✅ **POTENSI LULUS (GRADUATE)**")
        st.write(f"Probabilitas Kelulusan: **{pred_proba[0]*100:.2f}%**")
        st.write("*Rekomendasi:* Mahasiswa berada di jalur akademik yang aman.")
        st.success(f"✅ **POTENSI LULUS (GRADUATE)**")
        st.write(f"Probabilitas Kelulusan: **{pred_proba[0]*100:.2f}%**")
        st.write("*Rekomendasi:* Mahasiswa berada di jalur akademik yang aman.")

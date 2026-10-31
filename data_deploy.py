import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px




st.set_page_config(
    page_title="Dashboard Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide"  
)
st.title("Early Warning System - Jaya Jaya Institut🎓")
# muat model dan endocer
def load_artifacts():
    model = joblib.load('model_dropout_lgbm.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le
model,le = load_artifacts()

def load_data():
    url_csv = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/bce7a57a496d083716138922bc5839b5c30fa4ea/students_performance/data.csv"
    df = pd.read_csv(url_csv, sep=';')
    # df = pd.read_csv('data.csv')

    return df




try:
    df_raw = load_data()
    enrolled_df = df_raw[df_raw['Status'] == 'Enrolled'].copy()
    graduate_df = df_raw[df_raw['Status'] == 'Graduate'].copy()
    dropout_df = df_raw[df_raw['Status'] == 'Dropout'].copy()
    total_students =len(df_raw)

    # KPI
    col1, col2, col3, col4 = st.columns(4)
    pct_enrolled = (len(enrolled_df) / total_students) * 100
    pct_graduate = (len(graduate_df) / total_students) * 100
    pct_dropout = (len(dropout_df) / total_students) * 100
        
    with col1:
        st.metric(
            label="Total Siswa",
            value=f"{total_students:,}"
        )
        
    with col2:
        st.metric(
            label="Aktif (Enrolled)",
            value=f"{len(enrolled_df):,}",
            delta=f"{pct_enrolled:.1f}%"
        )
        
    with col3:
        st.metric(
            label="Lulus (Graduate)",
            value=f"{len(graduate_df):,}",
            delta=f"{pct_graduate:.1f}%"
        )
        
    with col4:
        st.metric(
            label="Putus (Dropout)",
            value=f"{len(dropout_df):,}",
            delta=f"{pct_dropout:.1f}%",
            delta_color="inverse" # Memberikan warna peringatan merah untuk dropout
        )
        
    st.markdown("---")
    

    # SIDEBAR
    st.sidebar.header("Filter Data")
    # st.sidebar.caption("Gunakan filter di bawah untuk mempersempit analisis mahasiswa *Enrolled*.")

   

    # filter course
    course_mapping = {
        33: 'Biofuel Production Technologies',
        171: 'Advertising and Marketing',
        8014: 'Social Service (Evening)',
        9070: 'Communication Design',
        9003: 'Agronomy 9070',
        9085: 'Veterinary Nursing',
        9119: 'Informatics Engineering',
        9130: 'Equinculture',
        9147: 'Management',
        9238: 'Social Service',
        9254: 'Tourism',
        9500: 'Nursing',
        9556: 'Oral Hygiene',
        9670: 'Advertising and Marketing Management',
        9773: 'Journalism and Communication',
        9853: 'Basic Education',
        9991: 'Management (evening attendance)'
    }
    enrolled_df['Course_Text'] = enrolled_df['Course'].map(course_mapping).fillna('Other Course')
    list_course = enrolled_df['Course_Text'].unique()
    selected_course_text = st.sidebar.selectbox("Pilih Program Studi:", list_course)


    # filter nationality
    nationality_mapping = {
        1: 'Portuguese',
        2: 'German',
        6: 'Spanish',
        11: 'Italian',
        13: 'Dutch',
        14: 'English',
        17: 'Lithuanian',
        21: 'Angolan',
        22: 'Cape Verdean',
        24: 'Guinean',
        25: 'Mozambican',
        26: 'Santomean',
        32: 'Turkish',
        41: 'Brazilian',
        62: 'Romanian',
        100: 'Moldovan',
        101: 'Mexican',
        103: 'Ukrainian',
        105: 'Russian',
        108: 'Cuban',
        109: 'Colombian'
    }

    enrolled_df['Nationality_Text'] = enrolled_df['Nacionality'].map(nationality_mapping).fillna('Other') 
    list_nationality = enrolled_df['Nationality_Text'].unique()
    selected_nationality_text = st.sidebar.selectbox("Pilih Kewarganegaraan:", list_nationality)

    filtered_df = enrolled_df[
        (enrolled_df['Course_Text'] == selected_course_text) & 
        (enrolled_df['Nationality_Text'] == selected_nationality_text)
    ].copy()

    if filtered_df.empty:
        st.warning("Tidak ada data mahasiswa yang cocok dengan filter tersebut.")
    else:
        # 6. Menyiapkan Fitur untuk Prediksi Model 
        # PENTING: Buang kolom tambahan teks buatan kita agar model tidak error saat diprediksi!
        X_input = filtered_df.drop(columns=['Status', 'Nationality_Text', 'Course_Text'], errors='ignore')

        # lakukan prediksi probabilitas
        y_prob = model.predict_proba(X_input)
        
        dropout_idx = list(le.classes_).index('Dropout')
        filtered_df['Dropout_Probability'] = y_prob[:, dropout_idx] * 100

        high_risk_filtered = filtered_df.sort_values(by='Dropout_Probability', ascending=False)


        gender_mapping = {1: 'Laki-laki', 0: 'Perempuan'}
        high_risk_filtered['Gender_Text'] = high_risk_filtered['Gender'].map(gender_mapping).fillna('Lainnya')

        scholarship_mapping = {1: 'Ya', 0: 'Tidak'}
        high_risk_filtered['Scholarship_Text'] = high_risk_filtered['Scholarship_holder'].map(scholarship_mapping).fillna('Lainnya')

        marital_mapping = {1: 'Belum Menikah', 2: 'Menikah', 3: 'Duda/Janda', 4: 'Cerai', 5: 'Kumpul Kebo/Nikah Siri', 6: 'Pisah Resmi'}
        high_risk_filtered['Marital_Text'] = high_risk_filtered['Marital_status'].map(marital_mapping).fillna('Lainnya')

        debtor_mapping = {1: 'Ya', 0: 'Tidak'}
        high_risk_filtered['Debtor_Text'] = high_risk_filtered['Debtor'].map(debtor_mapping).fillna('Lainnya')

        tuition_mapping = {1: 'Lunas', 0: 'Belum Lunas'}
        high_risk_filtered['Tuition_Text'] = high_risk_filtered['Tuition_fees_up_to_date'].map(tuition_mapping).fillna('Lainnya')

        # kolom yang ingin ditampilkan ke layar
        display_df = high_risk_filtered[[
            'Dropout_Probability', 
            'Gender_Text', 
            'Marital_Text', 
            'Age_at_enrollment', 
            'Curricular_units_1st_sem_grade',
            'Curricular_units_2nd_sem_grade',
            'Scholarship_Text',
            'Tuition_Text',
            'Debtor_Text',
        ]].copy()

        display_df = display_df.rename(columns={
            'Dropout_Probability': 'Risiko Dropout (%)',
            'Gender_Text': 'Jenis Kelamin',
            'Marital_Text': 'Status Pernikahan',
            'Age_at_enrollment': 'Usia Masuk',
            'Curricular_units_1st_sem_grade': 'Nilai Sem. 1',
            'Curricular_units_2nd_sem_grade': 'Nilai Sem 2',
            'Scholarship_Text': 'Beasiswa?',
            'Tuition_Text': 'Status Uang Kuliah',
            'Debtor_Text': 'Berutang'
            
        })

        st.write("### Daftar Mahasiswa yang Berisiko Dropout ")
        st.dataframe(display_df, use_container_width=True)




   


except FileNotFoundError:
    st.error("File CSV dataset tidak ditemukan! Pastikan file CSV Anda berada di folder yang sama dan namanya sudah benar.")
except Exception as e:
    st.error("Terjadi kesalahan saat memproses data:")
    st.exception(e) # Ini akan menampilkan detail error

st.sidebar.info(
    "📌 **Fokus Analisis:**\n"
    "Sistem ini memprediksi mahasiswa yg berisiko Dropout berdasarkan mahasiswa yang berstatus **Enrolled (Aktif)**."
)




st.caption("by Novelorenta 2026")

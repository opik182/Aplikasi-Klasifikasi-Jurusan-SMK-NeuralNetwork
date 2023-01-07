import pickle
import streamlit as st
import numpy as np

model_klasifikasi = pickle.load(open('ann_model_clasifier.sav', 'rb'))

st.title('Aplikasi Klasifikasi Jurusan SMK Menggunakan Neural Network')

MTK = st.number_input('Input Nilai Matematika', 0)
BahasaIndonesia = st.number_input('Input Nilai Bahasa Indonesia', 0)
BahasaInggris = st.number_input('Input Nilai Bahasa Inggris', 0)
IPA = st.number_input('Input Nilai IPA', 0)
IPS = st.number_input('Input Nilai IPS', 0)

# code untuk prediksi
klasifikasi = ''

# membuat tombol
if st.button('SUBMIT'):
    pred_jurusan = model_klasifikasi.predict(
        [[MTK, BahasaIndonesia, BahasaInggris, IPA, IPS]])
    pred_class = np.argmax(pred_jurusan)
    if pred_class == 0:
        klasifikasi = 'Anda Masuk Jurusan Keperawatan'
    elif pred_class == 1:
        klasifikasi = 'Anda Masuk Jurusan Farmasi'
    else:
        klasifikasi = 'Anda Masuk Jurusan TLM'

st.success(klasifikasi)

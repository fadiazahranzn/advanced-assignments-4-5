import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load("model.joblib")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load("model.joblib")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Centered Title and Subtitle
st.markdown("<h1 style='text-align: center;'>🚢 Titanic Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Masukkan data penumpang di bawah ini untuk melihat prediksi keselamatan.</p>", unsafe_allow_html=True)

st.divider()

# Use a form with border to match the card look
with st.form(key="prediction_form", border=True):
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox(
            "Kelas Tiket (Pclass)",
            options=[1, 2, 3],
            index=1,
            help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class"
        )
        
        sex = st.selectbox(
            "Jenis Kelamin",
            options=["male", "female"],
            index=1
        )
        
        sibsp = st.number_input(
            "Jumlah Saudara/Pasangan (SibSp)",
            min_value=0, max_value=8, value=1, step=1
        )

    with col2:
        parch = st.number_input(
            "Jumlah Orang Tua/Anak (Parch)",
            min_value=0, max_value=6, value=1, step=1
        )
        
        fare = st.number_input(
            "Tarif Tiket (Fare)",
            min_value=0.0, max_value=600.0, value=32.70, step=0.5, format="%.2f"
        )
        
        embarked = st.selectbox(
            "Pelabuhan Keberangkatan",
            options=["S", "C", "Q"]
        )

    submit_button = st.form_submit_button(label="Prediksi")

st.divider()

if submit_button:
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    survived_prob = probability[1] * 100
    not_survived_prob = probability[0] * 100

    if prediction == 1:
        st.success(f"**Prediksi: Selamat! 🎉** (Peluang: {survived_prob:.2f}%)")
    else:
        st.error(f"**Prediksi: Tidak Selamat 😔** (Peluang: {not_survived_prob:.2f}%)")

    # Placeholder for the image at the bottom
    # st.image("titanic_image.jpg", use_container_width=True)
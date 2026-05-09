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

st.title("🚢 Titanic Survival Predictor")
st.markdown("Masukkan data penumpang untuk memprediksi kemungkinan selamat.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Kelas Tiket (Pclass)",
        options=[1, 2, 3],
        format_func=lambda x: f"Kelas {x} ({'1st' if x == 1 else '2nd' if x == 2 else '3rd'})"
    )

    sex = st.selectbox(
        "Jenis Kelamin (Sex)",
        options=["male", "female"],
        format_func=lambda x: "Laki-laki" if x == "male" else "Perempuan"
    )

    embarked = st.selectbox(
        "Port Embarkasi (Embarked)",
        options=["S", "C", "Q"],
        format_func=lambda x: {
            "S": "S - Southampton",
            "C": "C - Cherbourg",
            "Q": "Q - Queenstown"
        }[x]
    )

with col2:
    sibsp = st.number_input(
        "Jumlah Saudara/Pasangan di Kapal (SibSp)",
        min_value=0, max_value=8, value=0, step=1
    )

    parch = st.number_input(
        "Jumlah Orang Tua/Anak di Kapal (Parch)",
        min_value=0, max_value=6, value=0, step=1
    )

    fare = st.number_input(
        "Harga Tiket (Fare) dalam USD",
        min_value=0.0, max_value=600.0, value=32.2, step=0.5
    )

st.divider()

if st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary"):
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

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.success(f"✅ **SELAMAT (Survived)** — Probabilitas: **{survived_prob:.1f}%**")
    else:
        st.error(f"❌ **TIDAK SELAMAT (Not Survived)** — Probabilitas: **{not_survived_prob:.1f}%**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probabilitas Selamat", f"{survived_prob:.1f}%")
    with col_b:
        st.metric("Probabilitas Tidak Selamat", f"{not_survived_prob:.1f}%")

    st.divider()
    with st.expander("🔎 Lihat Data Input"):
        st.dataframe(input_data, use_container_width=True)

st.markdown("---")
st.caption("Model: Random Forest + RandomizedSearchCV | Dataset: Titanic (Kaggle)")
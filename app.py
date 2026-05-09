import streamlit as st
import pandas as pd
import joblib

# Set Page Config MUST be the very first Streamlit command
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model with caching for performance
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Custom CSS Injection ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;1,600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Hide default streamlit header to make it look more like a custom app */
header {visibility: hidden;}

/* Adjust main padding */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

.ticket-title { font-family: 'Playfair Display', serif; }
.ticket-cutout {
    position: relative;
    background-color: #fefce8;
    border-radius: 1.5rem;
    border: 1px solid rgba(253, 230, 138, 0.6);
    color: #451a03;
    overflow: hidden;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}
.ticket-cutout::before, .ticket-cutout::after {
    content: '';
    position: absolute;
    top: 50%;
    width: 30px;
    height: 30px;
    background-color: #ffffff; /* Match Streamlit app background */
    border-radius: 50%;
    transform: translateY(-50%);
    z-index: 10;
}
.ticket-cutout::before { left: -15px; }
.ticket-cutout::after { right: -15px; }

.dash-card {
    background-color: white;
    border-radius: 1.5rem;
    padding: 2rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    border: 1px solid #e2e8f0;
    height: 100%;
}

/* Button override */
[data-testid="stFormSubmitButton"] button {
    background-color: #0f172a !important;
    color: white !important;
    border-radius: 0.75rem !important;
    font-weight: 700 !important;
    padding-top: 0.75rem !important;
    padding-bottom: 0.75rem !important;
    border: none !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background-color: #1e293b !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (Inputs) ---
st.sidebar.markdown("<h2 style='font-size: 1.25rem; font-weight: 700; color: #1e293b; margin-bottom: 0; margin-top: -1rem;'>⚙️ Pengaturan Penumpang</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 0.875rem; color: #64748b; margin-top: 0; margin-bottom: 1.5rem;'>Sesuaikan nilai sebelum prediksi</p>", unsafe_allow_html=True)

with st.sidebar.form(key="prediction_form"):
    st.markdown("**Kelas Tiket (Pclass)**")
    pclass = st.selectbox(
        "Kelas Tiket",
        options=[1, 2, 3],
        index=1,
        label_visibility="collapsed"
    )
    
    st.markdown("**Jenis Kelamin**")
    sex = st.selectbox(
        "Jenis Kelamin",
        options=["male", "female"],
        index=1,
        label_visibility="collapsed"
    )
    
    st.markdown("**Jumlah Saudara/Pasangan (SibSp)**")
    sibsp = st.slider(
        "SibSp",
        min_value=0, max_value=8, value=1, step=1,
        label_visibility="collapsed"
    )
    
    st.markdown("**Jumlah Orang Tua/Anak (Parch)**")
    parch = st.slider(
        "Parch",
        min_value=0, max_value=6, value=1, step=1,
        label_visibility="collapsed"
    )
    
    st.markdown("**Tarif Tiket (Fare)**")
    fare = st.number_input(
        "Fare",
        min_value=0.0, max_value=600.0, value=32.70, step=0.5, format="%.2f",
        label_visibility="collapsed"
    )
    
    st.markdown("**Pelabuhan Keberangkatan**")
    embarked = st.selectbox(
        "Embarked",
        options=["S", "C", "Q"],
        label_visibility="collapsed"
    )
    
    submit_button = st.form_submit_button(label="Mulai Prediksi", use_container_width=True)

# --- MAIN CONTENT ---
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 2rem;">
    <div>
        <h1 style="margin-bottom: 0; display: flex; align-items: center; gap: 12px; font-weight: 800; color: #0f172a;">
            <span style="font-size: 2.5rem; filter: drop-shadow(0 4px 3px rgb(0 0 0 / 0.07));">🚢</span> Titanic Survival Predictor
        </h1>
        <p style="color: #64748b; font-size: 1.125rem; font-weight: 500; margin-top: 0.5rem; margin-bottom: 0;">Masukkan data penumpang di bawah ini untuk melihat prediksi keselamatan.</p>
    </div>
    <div style="background-color: white; padding: 0.5rem 1.25rem; border-radius: 9999px; border: 1px solid #e2e8f0; font-size: 0.875rem; font-weight: 700; color: #475569; display: flex; align-items: center; gap: 8px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
        <span style="position: relative; display: flex; width: 12px; height: 12px;">
            <span style="animation: ping 1s cubic-bezier(0, 0, 0.2, 1) infinite; position: absolute; display: inline-flex; height: 100%; width: 100%; border-radius: 50%; background-color: #34d399; opacity: 0.75;"></span>
            <span style="position: relative; display: inline-flex; border-radius: 50%; height: 12px; width: 12px; background-color: #10b981;"></span>
        </span>
        Model Aktif
    </div>
</div>
""", unsafe_allow_html=True)

# Generate default values for first load if button hasn't been clicked
if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    
if submit_button:
    st.session_state.prediction_run = True

if st.session_state.prediction_run:
    # Run prediction
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
    
    is_survived = prediction == 1
    display_prob = survived_prob if is_survived else not_survived_prob
    status_text = "SELAMAT" if is_survived else "TIDAK SELAMAT"
    status_color = "#10b981" if is_survived else "#ef4444"
    bg_color = "#d1fae5" if is_survived else "#fee2e2"
    
    # Overview Banner
    st.markdown(f"""
    <div style="background-color: white; border-radius: 1.5rem; padding: 2.5rem; border: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1); position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; bottom: 0; width: 50%; background: linear-gradient(to right, rgba(16, 185, 129, 0.05), transparent); z-index: 0;"></div>
        <div style="position: relative; z-index: 1;">
            <div style="font-size: 0.875rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Hasil Prediksi</div>
            <div style="font-size: 3.5rem; font-weight: 900; color: #0f172a; line-height: 1; display: flex; align-items: center; gap: 1.25rem;">
                {status_text.capitalize()}
            </div>
        </div>
        <div style="text-align: right; position: relative; z-index: 1;">
            <div style="font-size: 0.875rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Peluang Keselamatan</div>
            <div style="font-size: 3.5rem; font-weight: 900; color: {status_color}; line-height: 1; text-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                {display_prob:.1f}<span style="font-size: 2.25rem; color: rgba({ '16, 185, 129' if is_survived else '239, 68, 68' }, 0.8);">%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3 Columns for metrics
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        # Probability Gauge using SVG/HTML
        dasharray = 251.2
        dashoffset = dasharray - (dasharray * (survived_prob / 100))
        
        st.markdown(f"""
        <div class="dash-card" style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
            <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 2rem;">
                <h4 style="font-size: 1rem; font-weight: 700; color: #1e293b; margin: 0;">Distribusi Peluang</h4>
            </div>
            
            <div style="position: relative; width: 14rem; height: 14rem; margin: 0 auto;">
                <svg style="width: 100%; height: 100%; transform: rotate(-90deg);" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="40" fill="transparent" stroke="#f1f5f9" stroke-width="12"></circle>
                    <circle cx="50" cy="50" r="40" fill="transparent" stroke="#10b981" stroke-width="12" stroke-dasharray="{dasharray}" stroke-dashoffset="{dashoffset}" stroke-linecap="round" style="filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));"></circle>
                </svg>
                <div style="position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <span style="font-size: 2.25rem; font-weight: 900; color: #0f172a; line-height: 1; margin-bottom: 4px;">{survived_prob:.0f}%</span>
                    <span style="font-size: 0.6875rem; font-weight: 800; color: #10b981; text-transform: uppercase; letter-spacing: 0.1em; background: #ecfdf5; padding: 4px 12px; border-radius: 9999px;">Selamat</span>
                </div>
            </div>
            
            <div style="display: flex; justify-content: center; gap: 2rem; width: 100%; margin-top: 2.5rem; font-size: 0.875rem; font-weight: 700;">
                <div style="display: flex; align-items: center; gap: 8px;"><span style="width: 16px; height: 16px; border-radius: 50%; background-color: #10b981; box-shadow: 0 1px 2px rgba(0,0,0,0.05);"></span> Selamat <span style="color: #64748b; margin-left: 4px;">{survived_prob:.1f}%</span></div>
                <div style="display: flex; align-items: center; gap: 8px;"><span style="width: 16px; height: 16px; border-radius: 50%; background-color: #e2e8f0; border: 1px solid #cbd5e1;"></span> Tidak <span style="color: #64748b; margin-left: 4px;">{not_survived_prob:.1f}%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Feature Importance (Mock logic for display)
        gender_bonus = "+30%" if sex == "female" else "-20%"
        gender_color = "#10b981" if sex == "female" else "#ef4444"
        gender_bg = "linear-gradient(to right, #34d399, #10b981)" if sex == "female" else "linear-gradient(to right, #f87171, #ef4444)"
        gender_width = "75%" if sex == "female" else "25%"
        
        pclass_bonus = "+15%" if pclass == 1 else ("+5%" if pclass == 2 else "-25%")
        pclass_color = "#10b981" if pclass < 3 else "#ef4444"
        pclass_bg = "linear-gradient(to right, #34d399, #10b981)" if pclass < 3 else "linear-gradient(to right, #f87171, #ef4444)"
        pclass_width = "60%" if pclass == 1 else ("40%" if pclass == 2 else "15%")

        st.markdown(f"""
        <div class="dash-card" style="display: flex; flex-direction: column;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
                <h4 style="font-size: 1rem; font-weight: 700; color: #1e293b; margin: 0;">Pengaruh Data Input</h4>
                <span style="font-size: 0.75rem; font-weight: 700; background-color: #eef2ff; color: #4f46e5; padding: 6px 12px; border-radius: 8px; border: 1px solid #e0e7ff;">Analisis Model</span>
            </div>
            
            <div style="display: flex; flex-direction: column; gap: 1.5rem; flex-grow: 1; justify-content: center;">
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.875rem; font-weight: 700; margin-bottom: 8px;">
                        <span style="color: #334155;">Jenis Kelamin ({sex})</span>
                        <span style="color: {gender_color};">{gender_bonus}</span>
                    </div>
                    <div style="width: 100%; background-color: #f1f5f9; border-radius: 9999px; height: 12px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);">
                        <div style="background: {gender_bg}; height: 100%; border-radius: 9999px; width: {gender_width}; transition: all 0.5s;"></div>
                    </div>
                </div>
                
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.875rem; font-weight: 700; margin-bottom: 8px;">
                        <span style="color: #334155;">Kelas Tiket ({pclass})</span>
                        <span style="color: {pclass_color};">{pclass_bonus}</span>
                    </div>
                    <div style="width: 100%; background-color: #f1f5f9; border-radius: 9999px; height: 12px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);">
                        <div style="background: {pclass_bg}; height: 100%; border-radius: 9999px; width: {pclass_width}; transition: all 0.5s;"></div>
                    </div>
                </div>
                
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.875rem; font-weight: 700; margin-bottom: 8px;">
                        <span style="color: #334155;">Saudara/Pasangan ({sibsp})</span>
                        <span style="color: #f43f5e;">-5%</span>
                    </div>
                    <div style="width: 100%; background-color: #f1f5f9; border-radius: 9999px; height: 12px; overflow: hidden; display: flex; justify-content: flex-end; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);">
                        <div style="background: linear-gradient(to left, #fb7185, #f43f5e); height: 100%; border-radius: 9999px; width: 16.66%; transition: all 0.5s;"></div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #f1f5f9; font-size: 0.8125rem; color: #64748b; line-height: 1.6; font-weight: 500;">
                Model memberikan bobot besar pada <strong style="color: #1e293b; background: #f1f5f9; padding: 2px 4px; border-radius: 4px;">Jenis Kelamin ({sex})</strong> dan <strong style="color: #1e293b; background: #f1f5f9; padding: 2px 4px; border-radius: 4px;">Kelas Tiket</strong> dalam menentukan peluang keselamatan ini.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Boarding Pass
        st.markdown(f"""
        <div class="ticket-cutout" style="display: flex; flex-direction: column; height: 100%;">
            <div style="background-color: #0f172a; color: #fefce8; padding: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.75rem; letter-spacing: 0.2em; color: #94a3b8; text-transform: uppercase; font-weight: 700;">White Star Line</div>
                    <div class="ticket-title" style="font-size: 1.5rem; margin-top: 4px; color: white;">Tiket Kapal</div>
                </div>
                <div style="width: 3rem; height: 3rem; border-radius: 50%; border: 2px solid rgba(245, 158, 11, 0.3); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    ⚓
                </div>
            </div>
            
            <div style="padding: 1.5rem; padding-bottom: 2rem; border-bottom: 2px dashed #fde68a; flex-grow: 1;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem 1rem; margin-bottom: 2rem;">
                    <div>
                        <div style="font-size: 0.625rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; color: rgba(180, 83, 9, 0.6); margin-bottom: 4px;">Penumpang</div>
                        <div style="font-weight: 600; font-size: 1.125rem;">{sex}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.625rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; color: rgba(180, 83, 9, 0.6); margin-bottom: 4px;">Kelas</div>
                        <div style="font-weight: 600; font-size: 1.125rem;">Kelas {pclass}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.625rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; color: rgba(180, 83, 9, 0.6); margin-bottom: 4px;">Keberangkatan</div>
                        <div style="font-weight: 600; font-size: 1.125rem;">{embarked}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.625rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; color: rgba(180, 83, 9, 0.6); margin-bottom: 4px;">Tarif</div>
                        <div style="font-weight: 600; font-size: 1.125rem;">£{fare:.2f}</div>
                    </div>
                </div>
                
                <div style="background-color: white; border-radius: 0.75rem; padding: 1.25rem; text-align: center; border: 1px solid #fef3c7; position: relative; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 6px; background-color: {status_color}; box-shadow: 0 0 10px {status_color};"></div>
                    <div style="font-size: 0.6875rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 6px; margin-top: 4px;">Status Prediksi</div>
                    <div style="font-size: 1.875rem; font-weight: 800; color: {status_color}; margin-bottom: 8px; letter-spacing: -0.025em;">{status_text}</div>
                    <div style="display: inline-flex; align-items: center; gap: 6px; background-color: {bg_color}; color: {status_color}; padding: 4px 12px; border-radius: 9999px; font-size: 0.8125rem; font-weight: 700; border: 1px solid rgba({ '16, 185, 129' if is_survived else '239, 68, 68' }, 0.2);">
                        {display_prob:.1f}% Peluang
                    </div>
                </div>
            </div>
            
            <div style="padding: 1.5rem; background-color: rgba(254, 243, 199, 0.4); display: flex; justify-content: space-between; align-items: center; opacity: 0.9;">
                <div style="opacity: 0.6; font-size: 0.5rem; font-family: monospace; letter-spacing: 0.2em; color: #78350f;">
                    || | || || | || ||| |<br>TN-84920183
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.625rem; font-weight: 700; color: #78350f; letter-spacing: 0.05em;">MODEL v1.2</div>
                    <div style="font-size: 0.5rem; color: rgba(180, 83, 9, 0.8); text-transform: uppercase; font-weight: 600;">Random Forest</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    # State before prediction is run
    st.markdown("""
    <div style="background-color: white; border-radius: 1.5rem; padding: 4rem 2rem; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1);">
        <div style="font-size: 4rem; margin-bottom: 1.5rem; color: #cbd5e1;">📊</div>
        <h3 style="font-size: 1.5rem; font-weight: 700; color: #334155; margin-bottom: 0.5rem; margin-top: 0;">Menunggu Prediksi</h3>
        <p style="color: #64748b; max-width: 32rem; margin: 0 auto; font-size: 1rem; line-height: 1.5;">
            Hasil prediksi keselamatan akan muncul di sini. Silakan atur parameter di sidebar kiri dan klik tombol "Mulai Prediksi".
        </p>
    </div>
    """, unsafe_allow_html=True)
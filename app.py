import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
from statistics import mode
from scipy.signal import butter, filtfilt, iirnotch
import plotly.graph_objects as go

# ==========================================
# 0. PROFESSONAL MEDITSINA VA SIGNAL UI (CSS)
# ==========================================
st.set_page_config(page_title="Aqlli EMG Monitoring", page_icon="🧬", layout="wide")

custom_css = """
<style>
/* 1. Asosiy fon rasmi */
[data-testid="stAppViewContainer"] {
    background-color: #e3effa; 
    background-image: url("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=2000&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-blend-mode: overlay; 
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 20px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.15);
    margin-top: 15px;
}
.stButton>button {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white; border: none; border-radius: 12px;
    padding: 10px 24px; font-size: 16px; font-weight: 500;
    transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stButton>button:hover {
    transform: translateY(-2px); box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2); color: #e0f7fa;
}
h1, h2, h3 { color: #1a252f !important; font-family: 'Inter', 'Segoe UI', sans-serif; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 1. SUN'IY INTELLEKT (AI) MODELLARINI YUKLASH
# ==========================================
@st.cache_resource
def load_ai_models():
    try:
        model = joblib.load('emg_rf_model.pkl')
        scaler = joblib.load('emg_scaler.pkl')
        return model, scaler, True
    except:
        return None, None, False

ai_model, ai_scaler, models_loaded = load_ai_models()

# ==========================================
# 2. KOGNITIV AGENT VA FORMULALAR
# ==========================================
class EMGAgent:
    def __init__(self):
        self.dataset_file = "emg_dataset.csv"

    def filter_signal(self, data, lowcut=20.0, highcut=450.0, fs=1000.0, order=4):
        notch_freq = 50.0
        quality_factor = 30.0
        b_notch, a_notch = iirnotch(notch_freq, quality_factor, fs)
        notched_data = filtfilt(b_notch, a_notch, data)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, notched_data)

    def find_active_zones(self, data, threshold, min_distance=100):
        abs_data = np.abs(data)
        active_points = np.where(abs_data > threshold)[0]
        if len(active_points) == 0: return []
        splits = np.where(np.diff(active_points) > min_distance)[0] + 1
        zones = np.split(active_points, splits)
        return [zone for zone in zones if len(zone) > 50]

    def extract_9_features(self, segment):
        segment = np.array(segment, dtype=np.float64)
        N = len(segment)
        if N <= 1: return {k: 0 for k in ["SSI", "4POW", "MAV", "WL", "ACC", "IEMG", "RMS", "VAR", "DASDV"]}
        return {
            "SSI": np.sum(np.abs(segment)**2),
            "4POW": np.sum(segment**4),
            "MAV": (1/N) * np.sum(np.abs(segment)),
            "WL": np.sum(np.abs(np.diff(segment))),
            "ACC": (1/(N-1)) * np.sum(np.abs(np.diff(segment))),
            "IEMG": np.sum(np.abs(segment)),
            "RMS": np.sqrt((1/N) * np.sum(segment**2)),
            "VAR": (1/(N-1)) * np.sum(segment**2),
            "DASDV": np.sqrt((1/(N-1)) * np.sum((np.diff(segment))**2))
        }

    def save_to_dataset(self, features_list, filename):
        df = pd.DataFrame(features_list)
        match = re.search(r'class\s*(\d+)', filename, re.IGNORECASE)
        class_label = f"Class {match.group(1)}" if match else "Noma'lum"
        df['Target_Class'] = class_label 
        if not os.path.isfile(self.dataset_file):
            df.to_csv(self.dataset_file, index=False, sep=';', decimal=',')
        else:
            df.to_csv(self.dataset_file, mode='a', header=False, index=False, sep=';', decimal=',')
        return len(df)

# ==========================================
# 3. VEB-SAHIFA INTERFEYSI (Streamlit)
# ==========================================
st.title("📈 Gibrid Agent: Aqlli EMG Signal Monitoring Tizimi")
st.markdown("**EMG signallarini filtrlash, ma'lumotlar bazasini shakllantirish va AI klassifikatsiya qilish**")
st.divider()

agent = EMGAgent()
uploaded_file = st.file_uploader("📂 EMG signal faylini yuklang (.txt yoki .csv formatida)", type=["txt", "csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file, sep=r'\s+', header=None, on_bad_lines='skip', engine='python')
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    
    ustunlar = df_raw.columns.tolist()
    default_index = ustunlar.index(5) if 5 in ustunlar else 0
    tanlangan_ustun = st.selectbox("🎯 Tahlil qilinadigan signal ustunini tanlang:", ustunlar, index=default_index)
    
    raw_signal = df_raw[tanlangan_ustun].dropna().values
    time_axis = list(range(len(raw_signal))) 
    
    if len(raw_signal) > 0:
        # YANGI 5-TAB QO'SHILDI
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 1. Xom Signal", "🧬 2. Filtrlash", "⚡ 3. Aktiv Zonalar", "💾 4. Dataset", "🤖 5. AI Klassifikatsiya"
        ])
        
        filtered_signal = agent.filter_signal(raw_signal - np.mean(raw_signal))
        threshold = st.slider("Sezuvchanlik chegarasi (Threshold)", 0.0, float(np.max(np.abs(filtered_signal))), float(np.max(np.abs(filtered_signal)) * 0.15), 0.1)
        active_zones = agent.find_active_zones(filtered_signal, threshold)

        with tab1:
            st.subheader("O'zgarishsiz olingan dastlabki raqamli signal")
            fig1 = go.Figure()
            fig1.add_trace(go.Scattergl(x=time_axis, y=raw_signal, mode='lines', line=dict(color='#2c3e50', width=0.5)))
            fig1.update_layout(height=400, template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            st.subheader("Shovqinlardan tozalangan signal (Notch + Bandpass Filter)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scattergl(x=time_axis, y=filtered_signal, mode='lines', line=dict(color='#0984e3', width=0.5)))
            fig2.update_layout(height=400, template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.subheader("Mushak qisqarish (Aktiv) zonalarini avtomatik aniqlash")
            if len(active_zones) > 0:
                report_text = f"Topilgan segmentlar: {len(active_zones)}\n"
                for i, zone in enumerate(active_zones):
                    report_text += f"{i+1}\t{zone[0]}\t{zone[-1]}\tlen= {len(zone)}\n"
                st.code(report_text, language="text")
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scattergl(x=time_axis, y=filtered_signal, mode='lines', line=dict(color='#b2bec3', width=0.4), showlegend=False))
            fig3.add_hline(y=threshold, line_dash="dash", line_color="#0984e3", line_width=1.5)
            fig3.add_hline(y=-threshold, line_dash="dash", line_color="#0984e3", line_width=1.5)
            
            for i, zone in enumerate(active_zones):
                fig3.add_trace(go.Scattergl(x=zone, y=filtered_signal[zone], mode='lines', line=dict(color='#2ecc71', width=0.8), showlegend=False))
                fig3.add_vrect(x0=zone[0], x1=zone[-1], fillcolor="#74b9ff", opacity=0.15, layer="below", line_width=0)
                
            fig3.update_layout(height=500, template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig3, use_container_width=True)

        with tab4:
            st.subheader("Ilmiy jadval asosidagi 9 ta matematik xususiyat")
            if len(active_zones) > 0:
                features_list = [agent.extract_9_features(filtered_signal[zone]) for zone in active_zones]
                st.dataframe(pd.DataFrame(features_list), use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True) 
                # TUGMA QISMI YANGILANDI
                if st.button("📥 Tahlil natijalarini Datasetga saqlash", key="save_data_btn"):
                    try:
                        agent.save_to_dataset(features_list, uploaded_file.name)
                        st.balloons() # Ekranda sharlar uchadi
                        st.toast("✅ Ma'lumotlar datasetga muvaffaqiyatli saqlandi!", icon="💾") # Pastki burchakda xabar
                        st.success(f"Ajoyib! {len(active_zones)} ta qisqarish zonasi 'emg_dataset.csv' fayliga qo'shildi.")
                    except Exception as e:
                        st.error(f"Saqlashda xatolik yuz berdi: {e}")
                
                st.markdown("---")
                if os.path.isfile(agent.dataset_file):
                    st.markdown("#### 📂 Joriy umumiy dataset holati (Oxirgi 5 ta qator):")
                    # Saqlangan ma'lumotlarni ko'rsatish
                    st.dataframe(pd.read_csv(agent.dataset_file, sep=';').tail(), use_container_width=True)
            else:
                st.warning("Aktiv zonalar topilmadi. Threshold ni pasaytirib ko'ring.")

        with tab5:
            st.subheader("🤖 Sun'iy Intellekt Kognitiv Tahlili va Bashorati")
            
            if not models_loaded:
                st.error("⚠️ Sun'iy intellekt modellari (emg_rf_model.pkl) topilmadi! Iltimos, avval modelni o'rgatuvchi skriptni ishga tushiring.")
            else:
                if len(active_zones) > 0:
                    features_list = [agent.extract_9_features(filtered_signal[zone]) for zone in active_zones]
                    df_features = pd.DataFrame(features_list)
                    
                    # Normalizatsiya va Klassifikatsiya
                    scaled_features = ai_scaler.transform(df_features)
                    predictions = ai_model.predict(scaled_features)
                    
                    # Ishonch ehtimolligini hisoblash (Probability)
                    probabilities = ai_model.predict_proba(scaled_features)
                    avg_probabilities = np.mean(probabilities, axis=0) * 100
                    
                    # Eng yuqori ehtimollikka ega sinfni aniqlash
                    best_class_idx = np.argmax(avg_probabilities)
                    final_decision = ai_model.classes_[best_class_idx]
                    confidence_score = avg_probabilities[best_class_idx]
                    
                    # --- 1. YAKUNIY XULOSA (METRIC CARDS) ---
                    st.markdown("### 🎯 Modelning Yakuniy Qarori")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Bashorat qilingan harakat", value=str(final_decision))
                    with col2:
                        st.metric(label="Modelning ishonch darajasi", value=f"{confidence_score:.1f}%")
                    with col3:
                        st.metric(label="Tahlil qilingan zonalar soni", value=str(len(active_zones)))
                        
                    st.divider()

                    # --- 2. EHTIMOLLIKLAR TAQSIMOTI GRAFIGI ---
                    st.markdown("#### 📊 Harakat sinflari bo'yicha ehtimollik taqsimoti")
                    
                    prob_df = pd.DataFrame({
                        'Sinf (Class)': ai_model.classes_,
                        'Ehtimollik (%)': avg_probabilities
                    }).sort_values(by='Ehtimollik (%)', ascending=True)

                    fig_prob = go.Figure(go.Bar(
                        x=prob_df['Ehtimollik (%)'],
                        y=prob_df['Sinf (Class)'],
                        orientation='h',
                        marker=dict(
                            color=prob_df['Ehtimollik (%)'],
                            colorscale='Viridis',
                            showscale=False
                        ),
                        text=prob_df['Ehtimollik (%)'].apply(lambda x: f"{x:.1f}%"),
                        textposition='auto'
                    ))
                    
                    fig_prob.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=30, b=20),
                        template="plotly_white",
                        xaxis_title="Ishonch darajasi (%)",
                        yaxis_title="",
                        font=dict(size=14)
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

                    # --- 3. BATAFSIL ILMIY HISOBOT (EXPANDER) ---
                    with st.expander("🔬 Har bir qisqarish bo'yicha batafsil matematik tahlil (Ochiluvchi oyna)"):
                        df_features['Bashorat'] = predictions
                        # Faqat eng muhim xususiyatlarni ko'rsatamiz
                        display_df = df_features[['Bashorat', 'SSI', 'RMS', 'MAV', 'WL', 'VAR']]
                        st.dataframe(display_df.style.highlight_max(axis=0, color='#dff9fb'), use_container_width=True)
                        st.caption("Eslatma: Ushbu jadval qisqarish zonalari bo'yicha olingan vaqt-domen (time-domain) xususiyatlarini aks ettiradi.")
                        
                else:
                    st.info("💡 Kognitiv tahlilni boshlash uchun, iltimos, 3-bo'limda threshold sezuvchanligini sozlab, mushak faolligi zonalarini aniqlang.")
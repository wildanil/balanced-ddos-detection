import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="ML Attack Classifier - Balanced Dataset", layout="wide")
st.title("🔐 Deteksi Serangan DDoS pada Dataset UNSW-NB15")
st.markdown("### *Balanced Machine Learning Model (Decision Tree, Naive Bayes, Random Forest, XGBoost)*")
# Load preprocessing tools
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")
feature_names = encoders['feature_names']
label_encoder_y = joblib.load("models/encoder_class.pkl")

# Load model helper
model_map = {
    "Decision Tree": "dt",
    "Naive Bayes": "nb",
    "Random Forest": "rf",
    "XGBoost": "xgb"
}
model_choice = st.sidebar.selectbox("Pilih Model", list(model_map.keys()))
model = joblib.load(f"models/model_{model_map[model_choice]}.pkl")
        
# Upload CSV
uploaded_file = st.file_uploader("📤 Unggah File CSV untuk Prediksi", type="csv")

# Sidebar Info Developer
st.sidebar.markdown("## 👨‍💻 Pengembang Aplikasi")
st.sidebar.markdown("""
- Wildanil Ghozi  
- Fauzi Adi Rafrastara  
- Ramadhan Rakhmat Sani
""")

st.sidebar.markdown("### ℹ️ Tentang Aplikasi")
st.sidebar.markdown("Aplikasi Simulasi Deteksi Serangan DDoS pada Dataset UNSW-NB15 menggunakan model terlatih yang telah diseimbangkan.")


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # ========================
    # 🔧 DATA PREPARATION
    # ========================

    # Perbaikan kolom ct_ftp_cmd
    df['ct_ftp_cmd'] = df['ct_ftp_cmd'].replace({'[^0-9.]': ''}, regex=True)
    df['ct_ftp_cmd'] = df['ct_ftp_cmd'].replace({'': np.nan})
    df['ct_ftp_cmd'] = df['ct_ftp_cmd'].astype(float)

    # Perbaikan attack_cat
    if 'attack_cat' in df.columns:
        df['attack_cat'] = np.where(df['attack_cat'] == " Fuzzers", "Fuzzers", df['attack_cat'])
        df['attack_cat'] = np.where(df['attack_cat'] == " Fuzzers ", "Fuzzers", df['attack_cat'])
        df['attack_cat'] = np.where(df['attack_cat'] == " Reconnaissance ", "Reconnaissance", df['attack_cat'])
        df['attack_cat'] = np.where(df['attack_cat'] == "Backdoors", "Backdoor", df['attack_cat'])
        df['attack_cat'] = np.where(df['attack_cat'] == " Shellcode ", "Shellcode", df['attack_cat'])
    
    try:
        
        # Imputasi missing value dengan nilai 0
        df = df.fillna(0)
        
        # Pastikan kolom kategorikal ada
        for col in ['proto', 'state', 'service']:
            if col not in df.columns:
                st.error(f"Kolom '{col}' tidak ditemukan dalam file.")
                st.stop()
            df[col] = encoders[col].transform(df[col])

        # Pastikan kolom fitur tersedia
        if not set(feature_names).issubset(df.columns):
            st.error("Beberapa fitur tidak tersedia dalam data.")
            st.stop()

        
        # Ambil target jika ada
        y_true = df['attack_cat'] if 'attack_cat' in df.columns else None
        X = df[feature_names]
        X_scaled = scaler.transform(X)
        
        # Label encode untuk class jika model XGBoost
        if model_choice == "XGBoost":
            if 'attack_cat' in df.columns:
                df['attack_cat'] = label_encoder_y.transform(df['attack_cat'])
                y_true = label_encoder_y.transform(y_true)
        
        # Prediksi
        y_pred = model.predict(X_scaled)
        df['Prediksi'] = y_pred

        # Tampilkan hasil
        st.subheader("📋 Hasil Prediksi")
        st.dataframe(df[['Prediksi'] + (['attack_cat'] if y_true is not None else [])])

        # Evaluasi jika ground truth tersedia
        if y_true is not None:
            st.subheader("📊 Evaluasi Model")
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            # ===========================
            # 📊 Chart Perbandingan Metrik
            # ===========================
            st.subheader("📉 Grafik Perbandingan Metrik")

            # Ambil metrik untuk class 'weighted avg'
            metrics = report_df.loc['weighted avg', ['precision', 'recall', 'f1-score']]
            accuracy = report['accuracy']

            plot_df = pd.DataFrame({
                'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                'Skor': [accuracy, metrics['precision'], metrics['recall'], metrics['f1-score']]
            })

            import altair as alt
            bar_chart = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X('Metrik', sort=None),
                y='Skor',
                color='Metrik'
            ).properties(width=600)

            st.altair_chart(bar_chart)

            # ===========================
            # Confusion Matrix
            # ===========================
            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)


        # Download hasil
        csv_result = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Prediksi",
            data=csv_result,
            file_name='hasil_prediksi.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Gagal memproses file: {e}")


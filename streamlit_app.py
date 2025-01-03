import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Fungsi Styling ---
def set_styles_with_thick_shadow(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({url});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        h1 {{
            font-size: 36px;
            color: #00008B;
            text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.5);
        }}
        h2, h3, h4, h5, h6 {{
            color: #00008B;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
        }}
        pre, code, .stMarkdown, .stDataFrame {{
            text-shadow: none;
            color: #00008B;
        }}
        .highlight-text {{
            background-color: #FFFFE0;
            padding: 4px;
            border-radius: 4px;
        }}
        .green-text {{
            color: #006400;
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Fungsi Visualisasi ---
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

def plot_heatmap(data):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Heatmap Korelasi")
    st.pyplot(fig)

def plot_accuracies(accuracies):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(accuracies.keys(), accuracies.values(), color='skyblue')
    ax.set_ylim(0, 1)
    ax.set_title("Perbandingan Akurasi Model")
    ax.set_ylabel("Akurasi")
    ax.set_xlabel("Model")
    st.pyplot(fig)

# --- Load Dataset ---
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/M-Imaduddin-A/Bengkel-Koding-DS/main/data/water_potability.csv"
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return None

# --- Aplikasi Utama ---
set_styles_with_thick_shadow("https://images7.alphacoders.com/926/926408.png")

st.markdown("# **Evaluasi Model Klasifikasi Kualitas Air**", unsafe_allow_html=True)

data = load_data()
if data is not None:
    st.write(f"**<span class='highlight-text'>Jumlah data: {data.shape[0]} baris, {data.shape[1]} kolom</span>**", unsafe_allow_html=True)
    st.write("**<span class='highlight-text'>Pratinjau dataset:</span>**", unsafe_allow_html=True)
    st.dataframe(data.head())

    # Preprocessing
    data.fillna(data.mean(), inplace=True)
    X = data.drop(columns=["Potability"])
    y = data["Potability"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalisasi Opsional
    normalize = st.checkbox("<span class='green-text'>Gunakan Normalisasi</span>", unsafe_allow_html=True)
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Model
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', random_state=42)
    }
    
    st.write("**<span class='highlight-text'>Evaluasi Model</span>**", unsafe_allow_html=True)
    accuracies = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[model_name] = acc
        
        st.write(f"**<span class='highlight-text'>{model_name}</span>**", unsafe_allow_html=True)
        st.write(f"Akurasi: {acc:.2f}")
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ["Not Potable", "Potable"])
    
    st.write("**<span class='highlight-text'>Perbandingan Akurasi</span>**", unsafe_allow_html=True)
    plot_accuracies(accuracies)
    
    st.write("**<span class='highlight-text'>Kesimpulan</span>**", unsafe_allow_html=True)
    best_model = max(accuracies, key=accuracies.get)
    st.write(f"Model dengan akurasi tertinggi adalah **<span class='highlight-text'>{best_model}</span>** dengan akurasi **{accuracies[best_model]:.2f}**.", unsafe_allow_html=True)
else:
    st.warning("Dataset tidak dapat dimuat. Silakan unggah file CSV.")
    uploaded_file = st.file_uploader("Upload dataset CSV Anda", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset berhasil dimuat!")

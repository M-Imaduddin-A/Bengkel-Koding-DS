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

# Set halaman
st.title("Evaluasi Model Klasifikasi Kualitas Air")

# Fungsi untuk menampilkan confusion matrix
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# Memuat dataset
@st.cache_data
def load_data_from_github():
    url = "https://raw.githubusercontent.com/M-Imaduddin-A/Bengkel-Koding-DS/main/data/water_potability.csv"
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Gagal memuat dataset dari GitHub: {e}")
        return None

# Load data dari GitHub
data = load_data_from_github()

# Fallback jika data dari GitHub gagal
if data is None:
    uploaded_file = st.file_uploader("Upload dataset CSV Anda", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset berhasil dimuat!")
    else:
        st.warning("Silakan upload file dataset atau pastikan koneksi internet tersedia.")
else:
    st.write("Dataset berhasil dimuat dari GitHub!")

# Handling missing values
data.fillna(data.mean(), inplace=True)

# Menampilkan informasi dataset
st.write(f"Jumlah data: {data.shape[0]} baris, {data.shape[1]} kolom")
st.write("Kolom:", list(data.columns))

# Split data
X = data.drop(columns=["Potability"])
y = data["Potability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pilihan normalisasi
normalize = st.checkbox("Gunakan Normalisasi")

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

# Evaluasi model
st.write("### Evaluasi Model")
accuracies = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[model_name] = acc
    
    st.write(f"**{model_name}**")
    st.write(f"Akurasi: {acc:.2f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ["Not Potable", "Potable"])

# Visualisasi Akurasi
st.write("### Perbandingan Akurasi")
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.Paired.colors[:len(accuracies)]
ax.bar(accuracies.keys(), accuracies.values(), color=colors)
ax.set_ylim(0, 1)
ax.set_title("Perbandingan Akurasi Model")
ax.set_ylabel("Akurasi")
ax.set_xlabel("Model")
st.pyplot(fig)

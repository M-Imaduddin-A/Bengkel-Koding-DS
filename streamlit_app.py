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
@st.cache
def load_data():
    # Contoh data dummy (gunakan dataset Anda di sini)
    data = pd.read_csv("water_quality.csv")  # Ganti dengan path dataset Anda
    return data

data = load_data()

# Menampilkan data
st.write("### Dataset")
st.write(data.head())

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
ax.bar(accuracies.keys(), accuracies.values(), color=["skyblue", "orange", "red"])
ax.set_ylim(0, 1)
ax.set_title("Perbandingan Akurasi Model")
ax.set_ylabel("Akurasi")
ax.set_xlabel("Model")
st.pyplot(fig)

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

# Tambahkan CSS untuk mengganti background
# Tambahkan CSS untuk mengganti background
def set_background(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({url});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Tambahkan CSS untuk mengganti background dan warna font
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
        .stApp * {{
            color: #fdfcfc; /* Warna teks putih terang */
            text-shadow: 4px 4px 6px rgba(0, 0, 0, 0.8); /* Bayangan tebal hitam */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Background custom dengan URL
set_background("https://images2.alphacoders.com/686/686406.jpg")
set_styles_with_thick_shadow("https://images2.alphacoders.com/686/686406.jpg")

# Judul aplikasi
st.title("Evaluasi Model Klasifikasi Kualitas Air")
st.write("Selamat datang di aplikasi analisis kualitas air!")


# Fungsi untuk menampilkan confusion matrix
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# Fungsi untuk menampilkan heatmap
def plot_heatmap(data):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Heatmap Korelasi")
    st.pyplot(fig)

# Fungsi untuk menampilkan distribusi data
def plot_distribution(data):
    fig, ax = plt.subplots(figsize=(12, 8))
    data.hist(bins=20, color="skyblue", edgecolor="black", ax=ax)
    plt.tight_layout()
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

# Tombol untuk menampilkan heatmap korelasi
if st.button("Tampilkan Heatmap Korelasi"):
    plot_heatmap(data)

# Tombol untuk menampilkan distribusi data
if st.button("Tampilkan Distribusi Data"):
    st.write("### Distribusi Data")
    for column in data.columns[:-1]:  # Exclude Potability
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data[column], kde=True, bins=20, color="blue", ax=ax)
        plt.title(f"Distribusi {column}")
        st.pyplot(fig)

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

# Kesimpulan
st.write("### Kesimpulan")
best_model = max(accuracies, key=accuracies.get)
st.write(f"Model dengan akurasi tertinggi adalah **{best_model}** dengan akurasi **{accuracies[best_model]:.2f}**.")
st.write("""
- **Keunggulan Random Forest**: Mampu menangani dataset dengan banyak fitur dan mengurangi overfitting.
- **Keterbatasan Logistic Regression**: Tidak cocok untuk dataset yang tidak linier.
- **Kelebihan SVM**: Cocok untuk dataset kecil dengan pemisahan linier.
""")
st.write(f"Rekomendasi: Gunakan **{best_model}** untuk kasus ini karena memberikan akurasi terbaik.")

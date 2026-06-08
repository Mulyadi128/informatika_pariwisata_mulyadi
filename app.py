import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("model_svm.pkl")
tfidf = joblib.load("tfidf.pkl")

# SIDEBAR
st.sidebar.header("Informasi Model")
st.sidebar.write("Algoritma : SVM")
st.sidebar.write("Dataset : Hotel Review Indonesia")
st.sidebar.write("Akurasi : 94.48%")
df = pd.read_csv("hotel_review.csv")

if st.checkbox("📈 Tampilkan Distribusi Sentimen"):

    fig, ax = plt.subplots()

    df["labels"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Distribusi Sentimen")
    ax.set_xlabel("Label")
    ax.set_ylabel("Jumlah Data")

    st.pyplot(fig)

# JUDUL
st.title("🏨 Klasifikasi Kepuasan Wisatawan")

st.write("""
Aplikasi ini digunakan untuk mengklasifikasikan kepuasan wisatawan
berdasarkan ulasan hotel menggunakan Support Vector Machine (SVM).
""")

# CONTOH INPUT
st.info("""
Contoh:

✅ hotelnya sangat nyaman dan bersih

✅ pelayanan ramah dan cepat

❌ kamar kotor dan bau

❌ AC rusak dan pelayanan buruk
""")

# INPUT
review = st.text_area(
    "Masukkan Ulasan Hotel"
)

# PREDIKSI
if st.button("Prediksi"):

    data = tfidf.transform([review])

    hasil = model.predict(data)[0]

    if hasil == 1:
        st.success("""
😊 Wisatawan Puas

Ulasan menunjukkan sentimen positif terhadap hotel.
""")
    else:
        st.error("""
😞 Wisatawan Tidak Puas

Ulasan menunjukkan sentimen negatif terhadap hotel.
""")

st.markdown("---")
st.caption("Dataset Hotel Review Indonesia | SVM | Akurasi 94.48%")
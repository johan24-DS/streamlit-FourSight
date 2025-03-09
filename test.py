import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data hasil clustering
df = pd.read_excel("Hasil Clustering KMeans.xlsx")  # Pastikan file ini adalah hasil dari K-Means

# Streamlit UI
st.title("🏡 Airbnb Clustering & Recommendation System")

# Checkbox untuk menampilkan data mentah
if st.checkbox("Show raw data"):
    st.write(df.head())

# Dropdown untuk memilih cluster tertentu
cluster_option = st.selectbox("Select Cluster", sorted(df["cluster"].unique()))

# Pastikan kolom 'price' dalam format numerik
df["price"] = df["price"].astype(str)  # Pastikan semua dalam bentuk string
df["price"] = df["price"].str.replace("[$,]", "", regex=True)  # Hapus simbol $ dan koma
df["price"] = pd.to_numeric(df["price"])  # Konversi ke numerik

# Slider untuk filter harga dan rating
price_range = st.slider("Select Price Range", int(df["price"].min()), int(df["price"].max()), (50, 200))
rating_range = st.slider("Select Review Scores Rating", int(df["review_scores_rating"].min()), 
                         int(df["review_scores_rating"].max()), (80, 100))

# Filter data berdasarkan input pengguna
filtered_df = df[(df["cluster"] == cluster_option) & 
                 (df["price"] >= price_range[0]) & (df["price"] <= price_range[1]) &
                 (df["review_scores_rating"] >= rating_range[0]) & (df["review_scores_rating"] <= rating_range[1])]

st.write(f"Showing {len(filtered_df)} properties matching your criteria:")
st.write(filtered_df[["room_type", "price", "review_scores_rating", "bedrooms", "bathrooms", "accommodates"]])

# Scatter plot Harga vs. Rating dengan warna berdasarkan cluster
st.subheader("📊 Price vs. Review Scores Rating (Clustered)")
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x=df["price"], y=df["review_scores_rating"], hue=df["cluster"], palette="coolwarm", ax=ax)
ax.set_xlabel("Price")
ax.set_ylabel("Review Scores Rating")
st.pyplot(fig)

# Pie Chart distribusi tipe kamar dalam cluster yang dipilih
st.subheader("🏠 Room Type Distribution in Selected Cluster")
fig, ax = plt.subplots()
filtered_df["room_type"].value_counts().plot.pie(autopct="%1.1f%%", startangle=140, cmap="coolwarm", ax=ax)
st.pyplot(fig)
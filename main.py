import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("airbnb_listings.csv")

# Encode categorical columns
label_encoders = {}
categorical_cols = ["room_type", "bed_type", "city_normalized"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features for clustering
features = [
    "price", "review_scores_rating", "accommodates", "bedrooms", "beds", 
    "bathrooms", "availability_365", "number_of_reviews", "distance_to_city",
    "room_type", "bed_type", "city_normalized"
]
X = df[features]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Gaussian Mixture Model Clustering
gmm = GaussianMixture(n_components=3, random_state=42)
df["cluster"] = gmm.fit_predict(X_scaled)

# Streamlit UI
st.title("🏡 Airbnb Property Clustering")

## User input filters with number input
num_bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=int(df["bedrooms"].max()), value=1, step=1)
num_bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=int(df["bathrooms"].max()), value=1, step=1)
num_accommodates = st.number_input("Number of Accommodates", min_value=1, max_value=int(df["accommodates"].max()), value=2, step=1)

# Price range filter
min_price, max_price = st.slider(
    "Select Price Range",
    min_value=int(df["price"].min()),
    max_value=int(df["price"].max()),
    value=(int(df["price"].min()), int(df["price"].max()))
)

# Sorting option for rating
sort_order = st.radio("Sort by Review Scores Rating", ["Highest to Lowest", "Lowest to Highest"])

# Tombol Search untuk menampilkan hasil rekomendasi
if st.button("Search"):
    filtered_df = df[
        (df["bedrooms"] >= num_bedrooms) &
        (df["bathrooms"] >= num_bathrooms) &
        (df["accommodates"] >= num_accommodates) &
        (df["price"] >= min_price) & 
        (df["price"] <= max_price)
    ]

    # Sorting the filtered results
    ascending = True if sort_order == "Lowest to Highest" else False
    filtered_df = filtered_df.sort_values(by="review_scores_rating", ascending=ascending)

    st.write("### Recommended Properties")
    st.write(filtered_df)

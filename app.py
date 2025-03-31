import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import psutil


@st.cache_data
def load_data():
    path = r'C:\\Users\\Guru Aswini Dath\Downloads\BigBasket Products.csv'
    df = pd.read_csv(path)
    return df

df = load_data()
print(df.isnull().sum())
df = df.dropna().reset_index(drop=True)

# Text Normalization Functions
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9]', ', ', str(text)).lower().split(', ')

def normalize_text(text):
    return re.sub(r'\s+', '', str(text).lower())

# Preprocessing Data
df["category"] = df["category"].apply(clean_text)
df["sub_category"] = df["sub_category"].apply(clean_text)
df["type"] = df["type"].apply(clean_text)
df["brand"] = df["brand"].apply(clean_text)

df["category"] = df["category"].apply(lambda x: ' '.join(x))
df["sub_category"] = df["sub_category"].apply(lambda x: ' '.join(x))
df["type"] = df["type"].apply(lambda x: ' '.join(x))
df["brand"] = df["brand"].apply(lambda x: ' '.join(x))

df["normalized_product"] = df["product"].apply(normalize_text)
df["normalized_description"] = df["description"].apply(normalize_text)

# Combining text features for similarity calculation
df["combined_features"] = df["category"] + " " + df["sub_category"] + " " + df["type"] + " " + df["brand"] + " " + df["description"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(df["combined_features"])


from sklearn.metrics.pairwise import cosine_similarity

# Recommendation Function
def recommend_products(search_term, min_price=None, max_price=None, min_rating=None):
    search_term = normalize_text(search_term)

    matches = df[
        (df["normalized_product"].str.contains(search_term, na=False)) |
        (df["normalized_description"].str.contains(search_term, na=False))
    ]
    
    if matches.empty:
        return None

    product_index = matches.index[0]

    # Compute similarity 
    search_vector = vectorizer.transform([df.iloc[product_index]["combined_features"]])
    similarity_scores = cosine_similarity(search_vector, feature_matrix)

    # Get the sorted similarity scores
    sorted_scores = sorted(enumerate(similarity_scores[0]), key=lambda x: x[1], reverse=True)

    recommendations = []
    for index, score in sorted_scores:
        product = df.iloc[index]

        if min_price is not None and product["sale_price"] < min_price:
            continue
        if max_price is not None and product["sale_price"] > max_price:
            continue
        if min_rating is not None and product["rating"] < min_rating:
            continue

        recommendations.append((product["product"], product["sale_price"], product["rating"], score))

        if len(recommendations) >= 10:
            break

    return pd.DataFrame(recommendations, columns=["Product", "Price", "Rating", "Similarity Score"])


st.title("🔍 Product Recommendation System")

# User Input
search_term = st.text_input("Enter Product Name or Description:")

col1, col2, col3 = st.columns(3)
with col1:
    min_price = st.number_input("Min Price", min_value=0, step=10, value=0)

with col2:
    max_price = st.number_input("Max Price", min_value=0, step=10, value=0)
with col3:
    min_rating = st.slider("Minimum Rating", min_value=0.0, max_value=5.0, step=0.1, value=0.0)


if st.button("🔍 Find Recommendations"):
    if search_term.strip() == "":
        st.warning("Please enter a product name or description!")
    else:
        results = recommend_products(search_term, min_price, max_price, min_rating)
        if results is None:
            st.error("❌ No products found for your search.")
        else:
            st.success(f"✅ Top {len(results)} Recommendations:")
            st.dataframe(results)


st.cache_data.clear()

# Monitor Memory Usage (optional, can be used to debug)
print(f"Memory Usage: {psutil.virtual_memory().percent}%")

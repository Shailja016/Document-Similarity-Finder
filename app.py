
import streamlit as st
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Set Streamlit page config
st.set_page_config(page_title="Document Similarity Finder", layout="wide")

# Load SBERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Title
st.title("📄 Document Similarity Finder with SBERT")

# Upload Excel file
uploaded_file = st.file_uploader("Upload your Excel file containing abstracts", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if 'abstract' not in df.columns:
        st.error("The Excel file must contain a column named 'abstract'")
    else:
        abstracts = df['abstract'].astype(str).tolist()

        # Encode abstracts
        with st.spinner("Encoding abstracts using SBERT..."):
            embeddings = model.encode(abstracts, convert_to_tensor=True)

        # Cosine similarity matrix
        cosine_scores = util.cos_sim(embeddings, embeddings)
        cosine_df = pd.DataFrame(cosine_scores.cpu().numpy())

        # Show heatmap of top 10
        st.subheader("🔗 Cosine Similarity Heatmap (Top 10)")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cosine_df.iloc[:10, :10], annot=True, cmap='coolwarm', fmt=".2f", square=True)
        st.pyplot(fig)

        # Show top N similar pairs
        top_n = st.slider("Select number of top similar pairs to show:", 5, 20, 10)
        st.subheader(f"📊 Top {top_n} Most Similar Document Pairs")

        similarities = []
        for i in range(len(abstracts)):
            for j in range(i + 1, len(abstracts)):
                sim_score = cosine_scores[i][j].item()
                similarities.append((i, j, sim_score))

        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)

        for idx, (i, j, score) in enumerate(similarities[:top_n]):
            st.markdown(f"**Pair {idx + 1}:**")
            st.write(f"**Document {i+1}**: {abstracts[i][:200]}...")
            st.write(f"**Document {j+1}**: {abstracts[j][:200]}...")
            st.write(f"**Cosine Similarity:** {score:.4f}")
            st.markdown("---")

        # Clustering
        st.subheader("📌 Clustering with KMeans + TSNE Visualization")
        num_clusters = st.slider("Select number of clusters:", 2, 10, 5)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings.cpu().numpy())

        perplexity = min(30, len(abstracts) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)

        reduced = tsne.fit_transform(embeddings.detach().cpu().numpy())


        cluster_df = pd.DataFrame(reduced, columns=["x", "y"])
        cluster_df['Cluster'] = cluster_labels

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=cluster_df, x='x', y='y', hue='Cluster', palette='tab10', s=100)
        plt.title("TSNE Plot of Abstracts Clustering")
        st.pyplot(fig2)

        # Search interface
        st.subheader("🔍 Find Similar Documents to a Query")
        query = st.text_input("Enter your query to find similar abstracts")
        if query:
            query_embedding = model.encode(query, convert_to_tensor=True)
            query_scores = util.cos_sim(query_embedding, embeddings)[0]
            top_results = torch.topk(query_scores, k=5)

            st.markdown("### 🔎 Top 5 Matches:")
            for idx, score in zip(top_results.indices, top_results.values):
                st.write(f"**Abstract {idx.item() + 1}:** {abstracts[idx.item()][:250]}...")
                st.write(f"**Similarity Score:** {score.item():.4f}")
                st.markdown("---")
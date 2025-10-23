# music_clustering_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Amazon Music Clustering",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cluster-card {
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1DB954;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🎵 Amazon Music Clustering Explorer</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
app_section = st.sidebar.radio("Go to:", [
    "📊 Project Overview", 
    "🎯 Cluster Explorer", 
    "📈 Feature Analysis",
    "🎧 Song Recommender"
])

# Load your data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('amazon_music_with_clusters.csv')
        return df
    except:
        st.error("Please make sure 'amazon_music_with_clusters.csv' is in the same directory")
        return None

df = load_data()

if df is None:
    st.stop()

# Section 1: Project Overview
if app_section == "📊 Project Overview":
    st.header("Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Songs", f"{len(df):,}")
    
    with col2:
        st.metric("Number of Clusters", df['cluster'].nunique())
    
    with col3:
        st.metric("Audio Features", "10")
    
    # Cluster distribution
    st.subheader("Cluster Distribution")
    cluster_counts = df['cluster'].value_counts().sort_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar chart
    cluster_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Songs per Cluster')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Number of Songs')
    ax1.tick_params(axis='x', rotation=0)
    
    # Pie chart
    ax2.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Cluster Proportions')
    
    st.pyplot(fig)
    
    # Key insights
    st.subheader("Key Insights")
    st.info("""
    🎯 **What this app shows:**
    - Songs are automatically grouped by their audio characteristics
    - Each cluster represents a different musical style or mood
    - Explore clusters to discover new music recommendations
    """)

# Section 2: Cluster Explorer
elif app_section == "🎯 Cluster Explorer":
    st.header("Cluster Explorer")
    
    # Cluster selector
    selected_cluster = st.selectbox(
        "Select a cluster to explore:",
        sorted(df['cluster'].unique())
    )
    
    # Filter data for selected cluster
    cluster_data = df[df['cluster'] == selected_cluster]
    cluster_size = len(cluster_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cluster Size", cluster_size)
    
    with col2:
        st.metric("Percentage", f"{(cluster_size/len(df)*100):.1f}%")
    
    with col3:
        # Show cluster interpretation if available
        if 'cluster_interpretation' in cluster_data.columns:
            interpretation = cluster_data['cluster_interpretation'].iloc[0]
            st.metric("Cluster Type", interpretation.split('|')[0].strip())
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    
    # Calculate average features for this cluster
    audio_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo', 'loudness']
    cluster_means = cluster_data[audio_features].mean()
    overall_means = df[audio_features].mean()
    
    # Create comparison chart
    comparison_df = pd.DataFrame({
        'This Cluster': cluster_means,
        'Overall Average': overall_means
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(kind='bar', ax=ax, color=['#1DB954', 'lightgray'])
    ax.set_title(f'Cluster {selected_cluster} vs Overall Average')
    ax.set_ylabel('Feature Value')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Sample songs from this cluster
    st.subheader("Sample Songs from This Cluster")
    
    sample_songs = cluster_data[['name_song', 'name_artists', 'genres']].head(10)
    st.dataframe(sample_songs, use_container_width=True)

# Section 3: Feature Analysis
elif app_section == "📈 Feature Analysis":
    st.header("Feature Analysis")
    
    # Feature selector
    selected_feature = st.selectbox(
        "Select a feature to analyze:",
        ['danceability', 'energy', 'acousticness', 'valence', 'tempo', 'loudness', 'speechiness']
    )
    
    # Distribution by cluster
    st.subheader(f"{selected_feature.title()} Distribution by Cluster")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df.boxplot(column=selected_feature, by='cluster', ax=ax)
    ax.set_title(f'{selected_feature.title()} Distribution Across Clusters')
    ax.set_ylabel(selected_feature)
    ax.set_xlabel('Cluster')
    st.pyplot(fig)
    
    # Feature correlations
    st.subheader("Feature Correlations")
    
    corr_features = ['danceability', 'energy', 'acousticness', 'valence', 'loudness']
    correlation_matrix = df[corr_features].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Audio Feature Correlations')
    st.pyplot(fig)

# Section 4: Song Recommender
elif app_section == "🎧 Song Recommender":
    st.header("Song Recommender")
    
    # Song selector
    song_options = df[['name_song', 'name_artists']].drop_duplicates()
    song_options['display'] = song_options['name_song'] + ' - ' + song_options['name_artists']
    
    selected_song_display = st.selectbox(
        "Select a song:",
        song_options['display'].values,
        index=0
    )
    
    # Get selected song details
    selected_song_name = selected_song_display.split(' - ')[0]
    selected_song_artist = selected_song_display.split(' - ')[1]
    
    selected_song = df[(df['name_song'] == selected_song_name) & 
                       (df['name_artists'] == selected_song_artist)].iloc[0]
    
    # Display selected song info
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Selected Song")
        st.write(f"**Song:** {selected_song['name_song']}")
        st.write(f"**Artist:** {selected_song['name_artists']}")
        st.write(f"**Cluster:** {selected_song['cluster']}")
        if 'cluster_interpretation' in selected_song:
            st.write(f"**Style:** {selected_song['cluster_interpretation']}")
    
    # Find similar songs (same cluster)
    similar_songs = df[df['cluster'] == selected_song['cluster']]
    similar_songs = similar_songs[similar_songs['name_song'] != selected_song['name_song']]
    
    with col2:
        st.subheader(f"Similar Songs (Cluster {selected_song['cluster']})")
        display_similar = similar_songs[['name_song', 'name_artists', 'genres']].head(10)
        st.dataframe(display_similar, use_container_width=True)
    
    # Audio features of selected song vs cluster average
    st.subheader("Audio Profile Comparison")
    
    audio_features = ['danceability', 'energy', 'acousticness', 'valence']
    song_features = selected_song[audio_features]
    cluster_features = similar_songs[audio_features].mean()
    
    comparison_data = pd.DataFrame({
        'Selected Song': song_features,
        'Cluster Average': cluster_features
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_data.T.plot(kind='bar', ax=ax, color=['#1DB954', 'orange'])
    ax.set_title('Selected Song vs Cluster Average')
    ax.set_ylabel('Feature Value')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎵 Amazon Music Clustering Project • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
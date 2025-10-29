import os
import glob
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
import json

def load_legal_texts():
    """Load all corrected legal texts from the results directories."""
    texts = []
    metadata = []
    
    # Define the base directory and state folders
    base_dir = "ocr_ai_results"
    state_dirs = ["al_results", "nc_results", "tn_results"]
    
    for state_dir in state_dirs:
        dir_path = os.path.join(base_dir, state_dir)
        if not os.path.exists(dir_path):
            continue
            
        # Find all corrected text files
        files = glob.glob(os.path.join(dir_path, "*_corrected.txt"))
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract metadata from filename
                filename = os.path.basename(file_path)
                state = state_dir.split('_')[0].upper()
                
                texts.append(content)
                metadata.append({
                    'file': filename,
                    'state': state,
                    'path': file_path
                })
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
    
    return texts, metadata

def process_texts(texts):
    """Process texts using sentence transformers."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def cluster_documents(embeddings):
    """Cluster documents using DBSCAN."""
    # Using a larger eps value and smaller min_samples for more inclusive clustering
    clusterer = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    clusters = clusterer.fit_predict(embeddings)
    return clusters

def reduce_dimensions(embeddings):
    """Reduce dimensions using UMAP for visualization."""
    reducer = UMAP(n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def create_visualizations(reduced_embeddings, clusters, metadata):
    """Create interactive visualizations using plotly."""
    df = pd.DataFrame({
        'UMAP1': reduced_embeddings[:, 0],
        'UMAP2': reduced_embeddings[:, 1],
        'Cluster': clusters,
        'State': [m['state'] for m in metadata],
        'File': [m['file'] for m in metadata]
    })
    
    # Create cluster visualization
    fig_clusters = px.scatter(
        df,
        x='UMAP1',
        y='UMAP2',
        color='Cluster',
        hover_data=['File', 'State'],
        title='Legal Codes Clustered by Theme'
    )
    
    # Create state-based visualization
    fig_states = px.scatter(
        df,
        x='UMAP1',
        y='UMAP2',
        color='State',
        hover_data=['File', 'Cluster'],
        title='Legal Codes by State'
    )
    
    # Save the visualizations
    fig_clusters.write_html("visualizations_clusters.html")
    fig_states.write_html("visualizations_states.html")
    
    # Save the clustering results
    results = {
        'clusters': clusters.tolist(),
        'metadata': metadata,
        'coordinates': reduced_embeddings.tolist()
    }
    
    with open('clustering_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def main():
    print("Loading legal texts...")
    texts, metadata = load_legal_texts()
    
    if not texts:
        print("No legal texts found!")
        return
    
    print(f"Processing {len(texts)} documents...")
    embeddings = process_texts(texts)
    
    print("Clustering documents...")
    clusters = cluster_documents(embeddings)
    
    print("Reducing dimensions for visualization...")
    reduced_embeddings = reduce_dimensions(embeddings)
    
    print("Creating visualizations...")
    create_visualizations(reduced_embeddings, clusters, metadata)
    
    print("\nAnalysis complete!")
    print("- Visualizations saved as 'visualizations_clusters.html' and 'visualizations_states.html'")
    print("- Results saved in 'clustering_results.json'")
    
    # Print cluster statistics
    unique_clusters = np.unique(clusters)
    print(f"\nFound {len(unique_clusters)} clusters:")
    for cluster in sorted(unique_clusters):
        if cluster == -1:
            label = "Unclustered"
        else:
            label = f"Cluster {cluster}"
        count = np.sum(clusters == cluster)
        print(f"{label}: {count} documents")

if __name__ == "__main__":
    main()
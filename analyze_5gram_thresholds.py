import os
import itertools
from collections import defaultdict
import json
from typing import List, Dict, Set, Tuple
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

def load_and_preprocess_text(file_path: str) -> str:
    """Load and preprocess text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Convert to lowercase and normalize whitespace
    text = text.lower()
    return ' '.join(text.split())

def generate_ngrams(text: str, n: int = 5) -> Set[tuple]:
    """Generate n-grams from text."""
    # Download required NLTK data if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    # Split text into words
    words = text.split()
    
    # Generate n-grams
    ngrams = zip(*[words[i:] for i in range(n)])
    return set(ngrams)

def calculate_jaccard_similarity(set1: Set[tuple], set2: Set[tuple]) -> float:
    """Calculate Jaccard similarity between two sets of n-grams."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def analyze_text_reuse_thresholds(directory: str) -> Dict:
    """Analyze text reuse with 5-grams across different similarity thresholds."""
    print("Analyzing text reuse with 5-grams...")
    
    # Get all text files
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('_corrected.txt'):
                all_files.append(os.path.join(root, file))
    
    # Generate n-grams for all documents
    document_ngrams = {}
    for file_path in all_files:
        text = load_and_preprocess_text(file_path)
        document_ngrams[file_path] = generate_ngrams(text, 5)
    
    # Calculate similarities for all document pairs
    similarities = []
    for file1, file2 in itertools.combinations(all_files, 2):
        similarity = calculate_jaccard_similarity(
            document_ngrams[file1],
            document_ngrams[file2]
        )
        
        if similarity > 0:  # Only store non-zero similarities
            similarities.append({
                'document1': os.path.basename(file1),
                'document2': os.path.basename(file2),
                'similarity_score': similarity,
                'state1': file1.split('/')[-2].split('_')[0],
                'state2': file2.split('/')[-2].split('_')[0]
            })
    
    # Sort by similarity score
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return similarities

def analyze_similarity_distribution(similarities: List[Dict]) -> None:
    """Analyze and visualize the distribution of similarity scores."""
    # Convert to DataFrame for analysis
    df = pd.DataFrame(similarities)
    
    # Calculate distribution statistics
    scores = df['similarity_score']
    percentiles = [25, 50, 75, 90, 95, 99]
    stats = {
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        **{f'p{p}': np.percentile(scores, p) for p in percentiles}
    }
    
    # Create distribution plot
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=scores,
        name='Similarity Distribution',
        nbinsx=50,
        opacity=0.7
    ))
    
    # Add vertical lines for potential thresholds
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    threshold_percentiles = [90, 75, 50, 25, 10]
    
    for color, p in zip(colors, threshold_percentiles):
        threshold = np.percentile(scores, p)
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{p}th percentile: {threshold:.3f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title="Distribution of Similarity Scores (5-grams)",
        xaxis_title="Similarity Score",
        yaxis_title="Count",
        showlegend=False
    )
    
    # Save plot
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    fig.write_html(output_dir / "similarity_distribution.html")
    
    return stats

def analyze_threshold_impact(similarities: List[Dict]) -> None:
    """Analyze the impact of different similarity thresholds."""
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    threshold_stats = []
    
    for threshold in thresholds:
        matches = [s for s in similarities if s['similarity_score'] >= threshold]
        if matches:
            stats = {
                'threshold': threshold,
                'num_matches': len(matches),
                'avg_similarity': np.mean([m['similarity_score'] for m in matches]),
                'cross_state_matches': sum(1 for m in matches if m['state1'] != m['state2']),
                'top_similarity': matches[0]['similarity_score']
            }
            threshold_stats.append(stats)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(threshold_stats)
    return df

def main():
    # Directory containing the legal codes
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'actual_divorce_codes')
    
    # Analyze similarities
    similarities = analyze_text_reuse_thresholds(base_dir)
    
    # Analyze distribution
    print("\nAnalyzing similarity score distribution...")
    stats = analyze_similarity_distribution(similarities)
    
    print("\nSimilarity Score Statistics:")
    for metric, value in stats.items():
        print(f"{metric}: {value:.3f}")
    
    # Analyze threshold impact
    print("\nAnalyzing impact of different thresholds...")
    threshold_stats = analyze_threshold_impact(similarities)
    print("\nThreshold Impact Analysis:")
    print(threshold_stats.to_string(index=False))
    
    # Save detailed results
    results = {
        'similarity_scores': similarities,
        'distribution_stats': stats,
        'threshold_analysis': threshold_stats.to_dict('records')
    }
    
    with open('text_reuse_analysis_5gram.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDetailed results saved to text_reuse_analysis_5gram.json")
    print("Visualization saved to visualizations/similarity_distribution.html")

if __name__ == "__main__":
    main()
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
from plotly.subplots import make_subplots
import hashlib
import re

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

def generate_k_shingles(text: str, k: int = 5) -> Set[str]:
    """Generate k-shingles (character-level n-grams) from text."""
    # Remove extra whitespace and normalize
    normalized_text = re.sub(r'\s+', ' ', text.strip())
    
    # Generate character-level k-shingles
    shingles = set()
    for i in range(len(normalized_text) - k + 1):
        shingle = normalized_text[i:i+k]
        shingles.add(shingle)
    
    return shingles

def generate_word_k_shingles(text: str, k: int = 3) -> Set[str]:
    """Generate word-level k-shingles from text."""
    words = text.split()
    
    # Generate word-level k-shingles
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i+k])
        shingles.add(shingle)
    
    return shingles

def hash_shingles(shingles: Set[str], hash_size: int = 32) -> Set[int]:
    """Hash k-shingles to integers for memory efficiency."""
    hashed_shingles = set()
    for shingle in shingles:
        # Use MD5 hash and take first hash_size bits
        hash_obj = hashlib.md5(shingle.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest()[:hash_size//4], 16)
        hashed_shingles.add(hash_int)
    
    return hashed_shingles

def calculate_jaccard_similarity(set1: Set, set2: Set) -> float:
    """Calculate Jaccard similarity between two sets (n-grams or k-shingles)."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def calculate_combined_similarity(ngrams1: Set[tuple], ngrams2: Set[tuple], 
                                shingles1: Set, shingles2: Set, 
                                ngram_weight: float = 0.6, shingle_weight: float = 0.4) -> Dict[str, float]:
    """Calculate combined similarity using both n-grams and k-shingles."""
    ngram_similarity = calculate_jaccard_similarity(ngrams1, ngrams2)
    shingle_similarity = calculate_jaccard_similarity(shingles1, shingles2)
    
    # Weighted combination
    combined_similarity = (ngram_weight * ngram_similarity + 
                          shingle_weight * shingle_similarity)
    
    return {
        'ngram_similarity': ngram_similarity,
        'shingle_similarity': shingle_similarity,
        'combined_similarity': combined_similarity
    }

def analyze_text_reuse_thresholds(directory: str) -> Dict:
    """Analyze text reuse with 5-grams and k-shingles across different similarity thresholds."""
    print("Analyzing text reuse with 5-grams and k-shingles...")
    
    # Get all text files
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('_corrected.txt'):
                all_files.append(os.path.join(root, file))
    
    print(f"Processing {len(all_files)} documents...")
    
    # Generate n-grams and k-shingles for all documents
    document_features = {}
    for i, file_path in enumerate(all_files):
        if i % 10 == 0:
            print(f"Processing document {i+1}/{len(all_files)}")
        
        text = load_and_preprocess_text(file_path)
        
        # Generate different types of features
        ngrams_5 = generate_ngrams(text, n=5)
        char_shingles_5 = generate_k_shingles(text, k=5)
        char_shingles_10 = generate_k_shingles(text, k=10)
        word_shingles_3 = generate_word_k_shingles(text, k=3)
        
        # Hash character shingles for memory efficiency
        hashed_char_5 = hash_shingles(char_shingles_5)
        hashed_char_10 = hash_shingles(char_shingles_10)
        
        document_features[file_path] = {
            'ngrams_5': ngrams_5,
            'char_shingles_5': hashed_char_5,
            'char_shingles_10': hashed_char_10,
            'word_shingles_3': word_shingles_3
        }
    
    print(f"Calculating similarities for {len(list(itertools.combinations(all_files, 2)))} document pairs...")
    
    # Calculate similarities for all document pairs
    similarities = []
    pair_count = 0
    total_pairs = len(list(itertools.combinations(all_files, 2)))
    
    for file1, file2 in itertools.combinations(all_files, 2):
        pair_count += 1
        if pair_count % 100 == 0:
            print(f"Processing pair {pair_count}/{total_pairs}")
        
        # Get features for both documents
        features1 = document_features[file1]
        features2 = document_features[file2]
        
        # Calculate different similarity measures
        ngram_sim = calculate_jaccard_similarity(
            features1['ngrams_5'], features2['ngrams_5']
        )
        char5_sim = calculate_jaccard_similarity(
            features1['char_shingles_5'], features2['char_shingles_5']
        )
        char10_sim = calculate_jaccard_similarity(
            features1['char_shingles_10'], features2['char_shingles_10']
        )
        word3_sim = calculate_jaccard_similarity(
            features1['word_shingles_3'], features2['word_shingles_3']
        )
        
        # Combined similarity (weighted average)
        combined_sim = (0.4 * ngram_sim + 0.2 * char5_sim + 
                       0.2 * char10_sim + 0.2 * word3_sim)
        
        # Store results if any similarity > 0
        if max(ngram_sim, char5_sim, char10_sim, word3_sim) > 0:
            similarities.append({
                'document1': os.path.basename(file1),
                'document2': os.path.basename(file2),
                'ngram_similarity': ngram_sim,
                'char5_similarity': char5_sim,
                'char10_similarity': char10_sim,
                'word3_similarity': word3_sim,
                'combined_similarity': combined_sim,
                'similarity_score': combined_sim,  # For backward compatibility
                'state1': file1.split('/')[-2].split('_')[0],
                'state2': file2.split('/')[-2].split('_')[0]
            })
    
    # Sort by combined similarity score
    similarities.sort(key=lambda x: x['combined_similarity'], reverse=True)
    
    print(f"Found {len(similarities)} document pairs with similarity > 0")
    
    return similarities

def analyze_similarity_distribution(similarities: List[Dict]) -> Dict:
    """Analyze and visualize the distribution of similarity scores."""
    # Convert to DataFrame for analysis
    df = pd.DataFrame(similarities)
    
    # Calculate distribution statistics for different similarity types
    similarity_types = ['ngram_similarity', 'char5_similarity', 'char10_similarity', 
                       'word3_similarity', 'combined_similarity']
    
    all_stats = {}
    
    for sim_type in similarity_types:
        if sim_type in df.columns:
            scores = df[sim_type]
            percentiles = [25, 50, 75, 90, 95, 99]
            stats = {
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
                **{f'p{p}': np.percentile(scores, p) for p in percentiles}
            }
            all_stats[sim_type] = stats
    
    # Use combined similarity for main analysis (backward compatibility)
    main_scores = df['combined_similarity'] if 'combined_similarity' in df.columns else df['similarity_score']
    percentiles = [25, 50, 75, 90, 95, 99]
    main_stats = {
        'mean': main_scores.mean(),
        'std': main_scores.std(),
        'min': main_scores.min(),
        'max': main_scores.max(),
        **{f'p{p}': np.percentile(main_scores, p) for p in percentiles}
    }
    
    # Create comparative distribution plots
    # Create subplots for different similarity types
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('N-gram Similarity', 'Char-5 Shingles', 'Char-10 Shingles',
                       'Word-3 Shingles', 'Combined Similarity', 'Main Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot individual similarity distributions
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (sim_type, color) in enumerate(zip(similarity_types[:5], colors)):
        if sim_type in df.columns:
            row, col = positions[i]
            scores = df[sim_type]
            
            fig.add_trace(
                go.Histogram(x=scores, name=sim_type, nbinsx=30, 
                           marker_color=color, opacity=0.7),
                row=row, col=col
            )
    
    # Main distribution plot (combined similarity)
    fig.add_trace(
        go.Histogram(x=main_scores, name='Combined Distribution', 
                    nbinsx=50, marker_color='darkblue', opacity=0.7),
        row=2, col=3
    )
    
    # Add vertical lines for potential thresholds on main plot
    threshold_percentiles = [90, 75, 50, 25, 10]
    threshold_colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for color, p in zip(threshold_colors, threshold_percentiles):
        threshold = np.percentile(main_scores, p)
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{p}th percentile: {threshold:.4f}",
            annotation_position="top",
            row=2, col=3
        )
    
    fig.update_layout(
        title_text="Distribution of Similarity Scores (N-grams + K-shingles)",
        height=800,
        showlegend=False
    )
    
    # Save plot
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    fig.write_html(output_dir / "similarity_distribution_combined.html")
    
    # Also create a simple single plot for backward compatibility
    simple_fig = go.Figure()
    simple_fig.add_trace(go.Histogram(
        x=main_scores,
        name='Combined Similarity Distribution',
        nbinsx=50,
        opacity=0.7
    ))
    
    for color, p in zip(threshold_colors, threshold_percentiles):
        threshold = np.percentile(main_scores, p)
        simple_fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{p}th percentile: {threshold:.4f}",
            annotation_position="top"
        )
    
    simple_fig.update_layout(
        title="Combined Similarity Distribution (N-grams + K-shingles)",
        xaxis_title="Similarity Score",
        yaxis_title="Count",
        showlegend=False
    )
    
    simple_fig.write_html(output_dir / "similarity_distribution.html")
    
    return {'main_stats': main_stats, 'detailed_stats': all_stats}

def analyze_threshold_impact(similarities: List[Dict]) -> pd.DataFrame:
    """Analyze the impact of different similarity thresholds."""
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    threshold_stats = []
    
    # Use combined similarity as the main metric
    main_similarity_key = 'combined_similarity' if 'combined_similarity' in similarities[0] else 'similarity_score'
    
    for threshold in thresholds:
        matches = [s for s in similarities if s[main_similarity_key] >= threshold]
        
        if matches:
            # Calculate stats for different similarity types
            stats = {
                'threshold': threshold,
                'num_matches': len(matches),
                'avg_combined_similarity': np.mean([m.get('combined_similarity', m.get('similarity_score', 0)) for m in matches]),
                'cross_state_matches': sum(1 for m in matches if m['state1'] != m['state2']),
                'top_combined_similarity': matches[0].get('combined_similarity', matches[0].get('similarity_score', 0))
            }
            
            # Add stats for individual similarity types if available
            if 'ngram_similarity' in matches[0]:
                stats.update({
                    'avg_ngram_similarity': np.mean([m['ngram_similarity'] for m in matches]),
                    'avg_char5_similarity': np.mean([m['char5_similarity'] for m in matches]),
                    'avg_char10_similarity': np.mean([m['char10_similarity'] for m in matches]),
                    'avg_word3_similarity': np.mean([m['word3_similarity'] for m in matches])
                })
            
            threshold_stats.append(stats)
    
    # Convert to DataFrame
    df = pd.DataFrame(threshold_stats)
    return df

def print_top_similarities(similarities: List[Dict], n: int = 10):
    """Print top N similarities with detailed breakdown."""
    print(f"\nTop {n} Most Similar Document Pairs:")
    print("=" * 80)
    
    for i, sim in enumerate(similarities[:n], 1):
        print(f"{i}. {sim['document1']} <-> {sim['document2']}")
        print(f"   States: {sim['state1'].upper()} <-> {sim['state2'].upper()}")
        
        if 'ngram_similarity' in sim:
            print(f"   N-gram (5): {sim['ngram_similarity']:.4f}")
            print(f"   Char-5 shingles: {sim['char5_similarity']:.4f}")
            print(f"   Char-10 shingles: {sim['char10_similarity']:.4f}")
            print(f"   Word-3 shingles: {sim['word3_similarity']:.4f}")
            print(f"   Combined: {sim['combined_similarity']:.4f}")
        else:
            print(f"   Similarity: {sim['similarity_score']:.4f}")
        
        cross_state = "✓" if sim['state1'] != sim['state2'] else "✗"
        print(f"   Cross-state: {cross_state}")
        print()

def main():
    # Directory containing the legal codes
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'actual_divorce_codes')
    
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' not found!")
        return
    
    # Analyze similarities
    similarities = analyze_text_reuse_thresholds(base_dir)
    
    if not similarities:
        print("No similarities found!")
        return
    
    # Print top similarities
    print_top_similarities(similarities, 15)
    
    # Analyze distribution
    print("\nAnalyzing similarity score distribution...")
    stats_result = analyze_similarity_distribution(similarities)
    
    if isinstance(stats_result, dict) and 'main_stats' in stats_result:
        main_stats = stats_result['main_stats']
        detailed_stats = stats_result['detailed_stats']
        
        print("\nMain Similarity Score Statistics (Combined):")
        for metric, value in main_stats.items():
            print(f"  {metric}: {value:.6f}")
        
        print("\nDetailed Statistics by Similarity Type:")
        for sim_type, stats in detailed_stats.items():
            print(f"\n{sim_type.replace('_', ' ').title()}:")
            for metric, value in stats.items():
                print(f"  {metric}: {value:.6f}")
    else:
        main_stats = stats_result
        print("\nSimilarity Score Statistics:")
        for metric, value in main_stats.items():
            print(f"  {metric}: {value:.6f}")
    
    # Analyze threshold impact
    print("\nAnalyzing impact of different thresholds...")
    threshold_stats = analyze_threshold_impact(similarities)
    print("\nThreshold Impact Analysis:")
    print(threshold_stats.to_string(index=False))
    
    # Save detailed results
    results = {
        'metadata': {
            'analysis_type': '5-gram + k-shingles',
            'features_used': ['5-grams', 'char-5-shingles', 'char-10-shingles', 'word-3-shingles'],
            'total_documents': len(set([s['document1'] for s in similarities] + [s['document2'] for s in similarities])),
            'total_pairs': len(similarities)
        },
        'similarity_scores': similarities,
        'distribution_stats': stats_result,
        'threshold_analysis': threshold_stats.to_dict('records')
    }
    
    with open('text_reuse_analysis_5gram_shingles.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Detailed results saved to text_reuse_analysis_5gram_shingles.json")
    print(f"Visualizations saved to visualizations/similarity_distribution*.html")
    print(f"Found {len(similarities)} document pairs with measurable similarity")
    
    # Summary statistics
    cross_state_pairs = [s for s in similarities if s['state1'] != s['state2']]
    print(f"Cross-state pairs: {len(cross_state_pairs)} ({100*len(cross_state_pairs)/len(similarities):.1f}%)")
    
    if 'combined_similarity' in similarities[0]:
        high_similarity = [s for s in similarities if s['combined_similarity'] > 0.1]
        print(f"High similarity pairs (>0.1): {len(high_similarity)}")
        
        very_high_similarity = [s for s in similarities if s['combined_similarity'] > 0.5]
        print(f"Very high similarity pairs (>0.5): {len(very_high_similarity)}")
        
        if very_high_similarity:
            print("\nVery high similarity pairs likely indicate:")
            print("- Duplicate documents")
            print("- Direct copying of legal text")
            print("- Different versions of the same document")

if __name__ == "__main__":
    main()
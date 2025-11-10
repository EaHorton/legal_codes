import os
import itertools
from collections import defaultdict
import json
from typing import List, Dict, Set, Tuple
import nltk
from nltk.corpus import stopwords
import pandas as pd

def load_and_preprocess_text(file_path: str) -> str:
    """Load and preprocess text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Convert to lowercase and normalize whitespace
    text = text.lower()
    return ' '.join(text.split())

def generate_ngrams(text: str, n: int) -> Set[tuple]:
    """Generate n-grams from text."""
    # Download required NLTK data if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    # Split text into words (simple tokenization)
    words = text.split()
    
    # Generate n-grams
    ngrams = zip(*[words[i:] for i in range(n)])
    return set(ngrams)

def calculate_jaccard_similarity(set1: Set[tuple], set2: Set[tuple]) -> float:
    """Calculate Jaccard similarity between two sets of n-grams."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def analyze_text_reuse_multiple_n(directory: str, n_values: List[int], similarity_threshold: float = 0.1) -> Dict:
    """Analyze text reuse across all documents using different n-gram sizes."""
    results = {
        'metadata': {
            'n_gram_sizes': n_values,
            'similarity_threshold': similarity_threshold
        },
        'analysis': {}
    }
    
    # Get all text files
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('_corrected.txt'):
                all_files.append(os.path.join(root, file))
    
    # Analyze for each n-gram size
    for n in n_values:
        print(f"\nAnalyzing with {n}-grams...")
        n_results = []
        
        # Generate n-grams for all documents
        document_ngrams = {}
        for file_path in all_files:
            text = load_and_preprocess_text(file_path)
            document_ngrams[file_path] = generate_ngrams(text, n)
        
        # Compare all pairs of documents
        for file1, file2 in itertools.combinations(all_files, 2):
            similarity = calculate_jaccard_similarity(
                document_ngrams[file1],
                document_ngrams[file2]
            )
            
            if similarity >= similarity_threshold:
                n_results.append({
                    'document1': os.path.basename(file1),
                    'document2': os.path.basename(file2),
                    'similarity_score': similarity,
                    'state1': file1.split('/')[-2].split('_')[0],
                    'state2': file2.split('/')[-2].split('_')[0]
                })
        
        # Sort results by similarity score
        n_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        results['analysis'][n] = n_results
        
        # Print summary for this n-gram size
        print(f"Found {len(n_results)} similar document pairs above threshold {similarity_threshold}")
        if n_results:
            top_pair = n_results[0]
            print(f"Top similarity score: {top_pair['similarity_score']:.3f} between:")
            print(f"- {top_pair['document1']} ({top_pair['state1']})")
            print(f"- {top_pair['document2']} ({top_pair['state2']})")
    
    return results

def analyze_ngram_statistics(results: Dict) -> pd.DataFrame:
    """Analyze statistics for different n-gram sizes."""
    stats = []
    for n, n_results in results['analysis'].items():
        if n_results:
            similarities = [r['similarity_score'] for r in n_results]
            stats.append({
                'n_gram_size': n,
                'num_matches': len(n_results),
                'avg_similarity': sum(similarities) / len(similarities),
                'max_similarity': max(similarities),
                'min_similarity': min(similarities)
            })
    
    return pd.DataFrame(stats)

def main():
    # Directory containing the legal codes
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'actual_divorce_codes')
    
    # Parameters for analysis
    n_values = [3, 4, 5, 6, 7, 8]  # Different n-gram sizes to try
    similarity_threshold = 0.1
    
    print(f"Analyzing text reuse with different n-gram sizes...")
    
    # Perform analysis
    results = analyze_text_reuse_multiple_n(base_dir, n_values, similarity_threshold)
    
    # Save detailed results to JSON file
    output_file = 'text_reuse_analysis_ngram_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Analyze and display statistics
    stats_df = analyze_ngram_statistics(results)
    print("\nN-gram Size Statistics:")
    print(stats_df.to_string(index=False))
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
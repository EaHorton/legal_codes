import os
import itertools
from collections import defaultdict
import json
from typing import List, Dict, Set, Tuple
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

def load_and_preprocess_text(file_path: str) -> str:
    """Load and preprocess text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

def generate_ngrams(text: str, n: int) -> Set[tuple]:
    """Generate n-grams from text."""
    # Download all required NLTK data
    required_packages = ['punkt', 'stopwords', 'punkt_tab']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package.startswith('punkt') else f'corpora/{package}')
        except LookupError:
            nltk.download(package, quiet=True)
    
    # Split text into words (simple tokenization)
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in words if word not in stop_words]
    
    # Generate n-grams
    return set(ngrams(tokens, n))

def calculate_jaccard_similarity(set1: Set[tuple], set2: Set[tuple]) -> float:
    """Calculate Jaccard similarity between two sets of n-grams."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def analyze_text_reuse(directory: str, n: int = 4, similarity_threshold: float = 0.1) -> Dict:
    """Analyze text reuse across all documents in the directory and its subdirectories."""
    # Dictionary to store results
    results = {
        'metadata': {
            'n_gram_size': n,
            'similarity_threshold': similarity_threshold
        },
        'document_pairs': []
    }
    
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
        document_ngrams[file_path] = generate_ngrams(text, n)
    
    # Compare all pairs of documents
    for file1, file2 in itertools.combinations(all_files, 2):
        similarity = calculate_jaccard_similarity(
            document_ngrams[file1],
            document_ngrams[file2]
        )
        
        if similarity >= similarity_threshold:
            results['document_pairs'].append({
                'document1': os.path.basename(file1),
                'document2': os.path.basename(file2),
                'similarity_score': similarity,
                'state1': file1.split('/')[-2].split('_')[0],  # Extract state abbreviation
                'state2': file2.split('/')[-2].split('_')[0]
            })
    
    # Sort results by similarity score in descending order
    results['document_pairs'].sort(key=lambda x: x['similarity_score'], reverse=True)
    return results

def main():
    # Directory containing the legal codes
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'actual_divorce_codes')
    
    # Parameters for analysis
    n_gram_size = 4
    similarity_threshold = 0.1
    
    print(f"Analyzing text reuse with {n_gram_size}-grams and similarity threshold of {similarity_threshold}...")
    
    # Perform analysis
    results = analyze_text_reuse(base_dir, n_gram_size, similarity_threshold)
    
    # Save results to JSON file
    output_file = 'text_reuse_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
    
    # Print summary of findings
    print("\nTop 5 most similar document pairs:")
    for pair in results['document_pairs'][:5]:
        print(f"\n{pair['document1']} ({pair['state1']}) - {pair['document2']} ({pair['state2']})")
        print(f"Similarity score: {pair['similarity_score']:.3f}")

if __name__ == "__main__":
    main()
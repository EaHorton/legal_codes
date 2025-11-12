#!/usr/bin/env python3
"""
Updated text reuse analysis with 7-grams and appropriate threshold for legal text borrowing
"""

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

def load_and_preprocess_text(file_path: str) -> str:
    """Load and preprocess text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Convert to lowercase and normalize whitespace
    text = text.lower()
    return ' '.join(text.split())

def generate_ngrams(text: str, n: int = 7) -> Set[tuple]:
    """Generate n-grams from text."""
    # Download required NLTK data if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    # Split text into words
    words = text.split()
    
    # Generate n-grams (need at least n words)
    if len(words) < n:
        return set()
    
    ngrams = zip(*[words[i:] for i in range(n)])
    return set(ngrams)

def calculate_jaccard_similarity(set1: Set[tuple], set2: Set[tuple]) -> float:
    """Calculate Jaccard similarity between two sets of n-grams."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def extract_state_from_filename(filename: str) -> str:
    """Extract state abbreviation from filename."""
    filename_lower = filename.lower()
    
    if 'al_' in filename_lower or filename_lower.startswith('al'):
        return 'al'
    elif 'nc_' in filename_lower or 'north' in filename_lower:
        return 'nc'
    elif 'tn_' in filename_lower or 'tennessee' in filename_lower:
        return 'tn'
    else:
        # Fallback - try to extract from path
        path_parts = filename.split('/')
        for part in path_parts:
            if 'al_' in part or part == 'al':
                return 'al'
            elif 'nc_' in part or 'north' in part:
                return 'nc' 
            elif 'tn_' in part or 'tennessee' in part:
                return 'tn'
        return 'unknown'

def analyze_text_reuse_7gram(directory: str, similarity_threshold: float = 0.02) -> Dict:
    """
    Analyze text reuse using 7-grams with threshold appropriate for legal text borrowing.
    
    For 7-grams in legal text:
    - Higher n-grams are more specific, so we can use higher thresholds
    - Legal borrowing typically shows clear patterns when n=7
    - Threshold of 0.02-0.05 is appropriate for detecting meaningful legal text reuse
    """
    
    print(f"Analyzing text reuse with 7-grams and threshold {similarity_threshold}...")
    
    results = {
        'metadata': {
            'n_gram_size': 7,
            'similarity_threshold': similarity_threshold,
            'analysis_type': 'legal_text_borrowing'
        },
        'similarity_scores': []
    }
    
    # Get all text files
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    print(f"Found {len(all_files)} text files to analyze")
    
    # Generate n-grams for all documents
    print("Generating 7-grams for all documents...")
    document_ngrams = {}
    
    for i, file_path in enumerate(all_files):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_files)} files...")
        
        text = load_and_preprocess_text(file_path)
        document_ngrams[file_path] = generate_ngrams(text, 7)
    
    print("Calculating pairwise similarities...")
    
    # Calculate similarities between all pairs
    total_pairs = len(all_files) * (len(all_files) - 1) // 2
    processed_pairs = 0
    
    for i, file1 in enumerate(all_files):
        for file2 in all_files[i+1:]:
            processed_pairs += 1
            
            if processed_pairs % 1000 == 0:
                print(f"  Processed {processed_pairs}/{total_pairs} pairs...")
            
            similarity = calculate_jaccard_similarity(
                document_ngrams[file1],
                document_ngrams[file2]
            )
            
            # Extract states for both documents
            state1 = extract_state_from_filename(file1)
            state2 = extract_state_from_filename(file2)
            
            results['similarity_scores'].append({
                'document1': os.path.basename(file1),
                'document2': os.path.basename(file2),
                'similarity_score': similarity,
                'state1': state1,
                'state2': state2,
                'full_path1': file1,
                'full_path2': file2
            })
    
    print(f"Analysis complete! Processed {len(results['similarity_scores'])} document pairs")
    
    # Filter for meaningful similarities and report statistics
    meaningful_similarities = [s for s in results['similarity_scores'] if s['similarity_score'] >= similarity_threshold]
    cross_state_similarities = [s for s in meaningful_similarities if s['state1'] != s['state2']]
    
    print(f"\nResults Summary:")
    print(f"  Total document pairs: {len(results['similarity_scores'])}")
    print(f"  Pairs above threshold ({similarity_threshold}): {len(meaningful_similarities)}")
    print(f"  Cross-state pairs above threshold: {len(cross_state_similarities)}")
    
    if cross_state_similarities:
        max_similarity = max(cross_state_similarities, key=lambda x: x['similarity_score'])
        print(f"  Highest cross-state similarity: {max_similarity['similarity_score']:.6f}")
        print(f"    Between: {max_similarity['state1'].upper()} and {max_similarity['state2'].upper()}")
    
    return results

def determine_optimal_threshold_7gram(directory: str):
    """
    Determine optimal threshold for 7-gram analysis of legal text borrowing.
    """
    print("Determining optimal threshold for 7-gram legal text analysis...")
    
    # First run a quick analysis to see the distribution
    sample_results = analyze_text_reuse_7gram(directory, similarity_threshold=0.001)  # Very low threshold to capture all
    
    all_similarities = [s['similarity_score'] for s in sample_results['similarity_scores']]
    cross_state_similarities = [s['similarity_score'] for s in sample_results['similarity_scores'] 
                               if s['state1'] != s['state2']]
    
    print(f"\n7-Gram Similarity Distribution Analysis:")
    print(f"All similarities - Mean: {np.mean(all_similarities):.6f}, Max: {np.max(all_similarities):.6f}")
    
    if cross_state_similarities:
        print(f"Cross-state similarities:")
        print(f"  Mean: {np.mean(cross_state_similarities):.6f}")
        print(f"  Median: {np.median(cross_state_similarities):.6f}")
        print(f"  90th percentile: {np.percentile(cross_state_similarities, 90):.6f}")
        print(f"  95th percentile: {np.percentile(cross_state_similarities, 95):.6f}")
        print(f"  Max: {np.max(cross_state_similarities):.6f}")
        
        # For legal text borrowing with 7-grams, recommend 95th percentile or 0.02, whichever is higher
        recommended_threshold = max(np.percentile(cross_state_similarities, 95), 0.02)
        
        print(f"\nRecommended threshold for 7-gram legal text borrowing: {recommended_threshold:.6f}")
        print(f"This balances specificity of 7-grams with meaningful legal borrowing detection.")
        
        return recommended_threshold
    else:
        print("No cross-state similarities found. Using default threshold of 0.02")
        return 0.02

def main():
    """Main analysis function"""
    base_dir = "actual_divorce_codes"
    
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' not found!")
        return
    
    # Determine optimal threshold
    optimal_threshold = determine_optimal_threshold_7gram(base_dir)
    
    print(f"\nRunning full analysis with 7-grams and threshold {optimal_threshold:.6f}...")
    
    # Run the full analysis
    results = analyze_text_reuse_7gram(base_dir, optimal_threshold)
    
    # Save results
    output_file = 'text_reuse_analysis_7gram.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
    
    # Print top cross-state similarities for immediate review
    cross_state_pairs = [s for s in results['similarity_scores'] 
                        if s['state1'] != s['state2'] and s['similarity_score'] >= optimal_threshold]
    
    if cross_state_pairs:
        cross_state_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        print(f"\nTop 10 Cross-State Text Borrowing Candidates (7-grams):")
        for i, pair in enumerate(cross_state_pairs[:10], 1):
            print(f"  {i:2d}. {pair['state1'].upper()} - {pair['state2'].upper()}: "
                  f"{pair['similarity_score']:.6f}")
            print(f"      {pair['document1'][:60]}...")
            print(f"      {pair['document2'][:60]}...")
            print()

if __name__ == "__main__":
    main()
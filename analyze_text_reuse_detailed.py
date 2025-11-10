import os
import itertools
from collections import defaultdict
import json
from typing import List, Dict, Set, Tuple
import nltk
from nltk.corpus import stopwords
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

def load_and_preprocess_text(file_path: str) -> Tuple[str, List[str]]:
    """Load and preprocess text from a file, returning both processed text and original lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()
        text = ' '.join(line.strip() for line in original_lines)
    
    # Convert to lowercase for comparison
    processed_text = text.lower()
    return processed_text, original_lines

def generate_ngrams_with_positions(text: str, n: int = 5) -> Dict[tuple, List[int]]:
    """Generate n-grams from text and track their positions."""
    words = text.split()
    ngram_positions = defaultdict(list)
    
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngram_positions[ngram].append(i)
    
    return ngram_positions

def find_matching_passages(text1: str, text2: str, n: int = 5) -> List[Dict]:
    """Find matching passages between two texts using n-grams."""
    # Generate n-grams with positions for both texts
    ngrams1 = generate_ngrams_with_positions(text1, n)
    ngrams2 = generate_ngrams_with_positions(text2, n)
    
    # Find matching n-grams
    matching_passages = []
    processed = set()  # Track processed positions to avoid duplicates
    
    # Find common n-grams
    common_ngrams = set(ngrams1.keys()) & set(ngrams2.keys())
    
    for ngram in common_ngrams:
        for pos1 in ngrams1[ngram]:
            if pos1 in processed:
                continue
                
            for pos2 in ngrams2[ngram]:
                # Extend the match as far as possible
                current_pos1 = pos1
                current_pos2 = pos2
                words1 = text1.split()
                words2 = text2.split()
                
                # Look ahead for additional matching words
                while (current_pos1 + 1 < len(words1) and 
                       current_pos2 + 1 < len(words2) and 
                       words1[current_pos1 + 1] == words2[current_pos2 + 1]):
                    current_pos1 += 1
                    current_pos2 += 1
                
                # Calculate the length of the match
                match_length = current_pos1 - pos1 + 1
                
                if match_length >= n:  # Only include matches at least as long as n-gram size
                    matching_text = ' '.join(words1[pos1:current_pos1 + 1])
                    matching_passages.append({
                        'text': matching_text,
                        'length': match_length,
                        'position1': pos1,
                        'position2': pos2
                    })
                    
                    # Mark these positions as processed
                    for i in range(pos1, current_pos1 + 1):
                        processed.add(i)
                    
                    break  # Move to next position in text1
    
    return matching_passages

def calculate_similarity_metrics(matches: List[Dict], text1: str, text2: str) -> Dict:
    """Calculate various similarity metrics based on matching passages."""
    words1 = text1.split()
    words2 = text2.split()
    
    # Calculate total words in matching passages
    matching_words = sum(match['length'] for match in matches)
    
    return {
        'num_matches': len(matches),
        'total_matching_words': matching_words,
        'coverage_text1': matching_words / len(words1) if words1 else 0,
        'coverage_text2': matching_words / len(words2) if words2 else 0,
        'avg_match_length': sum(match['length'] for match in matches) / len(matches) if matches else 0
    }

def analyze_text_reuse(directory: str) -> Dict:
    """Analyze text reuse across all documents using 5-grams and identify matching passages."""
    results = {
        'metadata': {
            'n_gram_size': 5,
            'similarity_threshold': 0.10
        },
        'document_pairs': []
    }
    
    # Get all text files
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('_corrected.txt'):
                all_files.append(os.path.join(root, file))
    
    # Compare all pairs of documents
    total_comparisons = len(list(itertools.combinations(all_files, 2)))
    print(f"Analyzing {total_comparisons} document pairs...")
    
    for idx, (file1, file2) in enumerate(itertools.combinations(all_files, 2), 1):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{total_comparisons} pairs analyzed")
            
        # Load and preprocess texts
        text1, lines1 = load_and_preprocess_text(file1)
        text2, lines2 = load_and_preprocess_text(file2)
        
        # Find matching passages
        matches = find_matching_passages(text1, text2)
        
        # Calculate similarity metrics
        metrics = calculate_similarity_metrics(matches, text1, text2)
        
        # Only include pairs with significant matches
        if metrics['num_matches'] > 0 and (metrics['coverage_text1'] >= 0.10 or metrics['coverage_text2'] >= 0.10):
            # Sort matches by length in descending order
            matches.sort(key=lambda x: x['length'], reverse=True)
            
            results['document_pairs'].append({
                'document1': os.path.basename(file1),
                'document2': os.path.basename(file2),
                'state1': file1.split('/')[-2].split('_')[0],
                'state2': file2.split('/')[-2].split('_')[0],
                'metrics': metrics,
                'matching_passages': matches[:5]  # Include top 5 longest matching passages
            })
    
    # Sort results by coverage
    results['document_pairs'].sort(
        key=lambda x: max(x['metrics']['coverage_text1'], x['metrics']['coverage_text2']),
        reverse=True
    )
    
    return results

def generate_html_report(results: Dict):
    """Generate an HTML report visualizing the text reuse analysis."""
    html_content = """
    <html>
    <head>
        <title>Text Reuse Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .pair { margin-bottom: 30px; padding: 15px; border: 1px solid #ccc; }
            .metrics { margin: 10px 0; }
            .passage { margin: 10px 0; padding: 10px; background-color: #f0f0f0; }
            .highlight { background-color: #ffeb3b; }
        </style>
    </head>
    <body>
        <h1>Text Reuse Analysis Report</h1>
    """
    
    for pair in results['document_pairs']:
        html_content += f"""
        <div class='pair'>
            <h2>Document Pair</h2>
            <p><strong>Document 1:</strong> {pair['document1']} ({pair['state1']})<br>
            <strong>Document 2:</strong> {pair['document2']} ({pair['state2']})</p>
            
            <div class='metrics'>
                <h3>Similarity Metrics</h3>
                <ul>
                    <li>Number of matching passages: {pair['metrics']['num_matches']}</li>
                    <li>Total matching words: {pair['metrics']['total_matching_words']}</li>
                    <li>Coverage in Document 1: {pair['metrics']['coverage_text1']:.1%}</li>
                    <li>Coverage in Document 2: {pair['metrics']['coverage_text2']:.1%}</li>
                    <li>Average match length: {pair['metrics']['avg_match_length']:.1f} words</li>
                </ul>
            </div>
            
            <h3>Top Matching Passages</h3>
        """
        
        for idx, passage in enumerate(pair['matching_passages'], 1):
            html_content += f"""
            <div class='passage'>
                <p><strong>Match {idx}</strong> (Length: {passage['length']} words)</p>
                <p class='highlight'>{passage['text']}</p>
            </div>
            """
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content

def main():
    # Directory containing the legal codes
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'actual_divorce_codes')
    
    print("Analyzing text reuse with optimized parameters (5-grams)...")
    results = analyze_text_reuse(base_dir)
    
    # Save detailed results to JSON
    with open('text_reuse_detailed_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save HTML report
    html_content = generate_html_report(results)
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'text_reuse_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("\nAnalysis complete!")
    print(f"Found {len(results['document_pairs'])} document pairs with significant text reuse")
    print("\nTop matches:")
    
    for pair in results['document_pairs'][:5]:
        print(f"\n{pair['document1']} ({pair['state1']}) - {pair['document2']} ({pair['state2']})")
        print(f"Coverage: {pair['metrics']['coverage_text1']:.1%} / {pair['metrics']['coverage_text2']:.1%}")
        print(f"Number of matching passages: {pair['metrics']['num_matches']}")
        print(f"Longest matching passage ({pair['matching_passages'][0]['length']} words):")
        print(f"\"{pair['matching_passages'][0]['text']}\"")
    
    print("\nDetailed results saved to text_reuse_detailed_analysis.json")
    print("Interactive report saved to visualizations/text_reuse_report.html")

if __name__ == "__main__":
    main()
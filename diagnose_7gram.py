#!/usr/bin/env python3
"""
Diagnostic script to check state identification and cross-state pairs
"""

import os
import json

def extract_state_from_filename(filename):
    """Extract state from filename using directory path"""
    filename_lower = filename.lower()
    
    # Check directory structure first - most reliable
    if '/al_actual_divorce_codes/' in filename_lower or 'al_actual_divorce_codes' in filename_lower:
        return 'AL'
    elif '/nc_actual_divorce_codes/' in filename_lower or 'nc_actual_divorce_codes' in filename_lower:
        return 'NC'
    elif '/tn_actual_divorce_codes/' in filename_lower or 'tn_actual_divorce_codes' in filename_lower:
        return 'TN'
    
    # Check for state prefixes in filename
    filename_only = filename.split('/')[-1].lower()
    if filename_only.startswith('al_'):
        return 'AL'
    elif filename_only.startswith('nc_') or 'north carolina' in filename_lower:
        return 'NC'
    elif filename_only.startswith('tn_'):
        return 'TN'
    
    # Check for state names in content
    if 'alabama' in filename_lower:
        return 'AL'
    elif 'north carolina' in filename_lower or 'carolina' in filename_lower:
        return 'NC'
    elif 'tennessee' in filename_lower:
        return 'TN'
        
    return 'UNKNOWN'

def diagnose_state_identification():
    """Diagnose state identification issues"""
    
    base_dir = "actual_divorce_codes"
    
    print("DIAGNOSING STATE IDENTIFICATION")
    print("=" * 50)
    
    # Get all text files and their identified states
    all_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    print(f"Found {len(all_files)} text files")
    
    # Check state identification
    state_counts = {'al': 0, 'nc': 0, 'tn': 0, 'unknown': 0}
    state_files = {'al': [], 'nc': [], 'tn': [], 'unknown': []}
    
    for file_path in all_files:
        state = extract_state_from_filename(file_path)
        state_counts[state] += 1
        state_files[state].append(file_path)
    
    print(f"\nState distribution:")
    for state, count in state_counts.items():
        print(f"  {state.upper()}: {count} files")
    
    # Show sample files for each state
    print(f"\nSample files by state:")
    for state in ['al', 'nc', 'tn', 'unknown']:
        if state_files[state]:
            print(f"\n  {state.upper()} files (showing first 3):")
            for file_path in state_files[state][:3]:
                print(f"    {file_path}")
    
    # Calculate potential cross-state pairs
    total_cross_state_pairs = 0
    for i, state1 in enumerate(['al', 'nc', 'tn']):
        for state2 in ['al', 'nc', 'tn'][i+1:]:
            pairs = state_counts[state1] * state_counts[state2]
            total_cross_state_pairs += pairs
            print(f"\n  Potential {state1.upper()}-{state2.upper()} pairs: {pairs}")
    
    print(f"\nTotal potential cross-state pairs: {total_cross_state_pairs}")
    
    return state_files, state_counts

def rerun_7gram_with_lower_threshold(threshold=0.001):
    """Rerun 7-gram analysis with much lower threshold to catch cross-state similarities"""
    
    print(f"\n\nRERUNNING 7-GRAM ANALYSIS WITH THRESHOLD {threshold}")
    print("=" * 60)
    
    # Load existing results
    with open('text_reuse_analysis_7gram.json', 'r') as f:
        data = json.load(f)
    
    # Filter for cross-state pairs above the new threshold
    cross_state_pairs = []
    for pair in data['similarity_scores']:
        if pair['state1'] != pair['state2'] and pair['similarity_score'] >= threshold:
            cross_state_pairs.append(pair)
    
    print(f"Cross-state pairs found with threshold {threshold}: {len(cross_state_pairs)}")
    
    if cross_state_pairs:
        # Sort by similarity
        cross_state_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        print(f"\nTop 10 cross-state similarities:")
        for i, pair in enumerate(cross_state_pairs[:10], 1):
            print(f"  {i:2d}. {pair['state1'].upper()} - {pair['state2'].upper()}: "
                  f"{pair['similarity_score']:.6f}")
            print(f"      {pair['document1'][:50]}...")
            print(f"      {pair['document2'][:50]}...")
        
        # Statistics
        similarities = [pair['similarity_score'] for pair in cross_state_pairs]
        print(f"\nCross-state similarity statistics:")
        print(f"  Count: {len(similarities)}")
        print(f"  Mean: {np.mean(similarities):.6f}")
        print(f"  Max: {max(similarities):.6f}")
        print(f"  Min: {min(similarities):.6f}")
        
        # State pair breakdown
        from collections import defaultdict
        state_pair_counts = defaultdict(int)
        for pair in cross_state_pairs:
            key = tuple(sorted([pair['state1'], pair['state2']]))
            state_pair_counts[key] += 1
        
        print(f"\nCross-state pairs by state combination:")
        for state_pair, count in state_pair_counts.items():
            print(f"  {state_pair[0].upper()}-{state_pair[1].upper()}: {count} pairs")
        
        return cross_state_pairs
    else:
        print("No cross-state pairs found even with very low threshold")
        print("This suggests either:")
        print("1. No meaningful text reuse between states with 7-grams")
        print("2. State identification issues")
        print("3. 7-grams may be too specific for this corpus")
        
        return []

if __name__ == "__main__":
    import numpy as np
    
    # First diagnose state identification
    state_files, state_counts = diagnose_state_identification()
    
    # Then rerun analysis with lower threshold
    cross_state_pairs = rerun_7gram_with_lower_threshold(0.001)
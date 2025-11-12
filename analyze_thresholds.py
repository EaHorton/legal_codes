#!/usr/bin/env python3
"""
Analyze similarity distribution to determine appropriate threshold for legal text analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_similarity_distribution():
    # Load the data
    with open('text_reuse_analysis_5gram.json', 'r') as f:
        data = json.load(f)
    
    similarity_data = data['similarity_scores']
    
    # Separate cross-state and within-state similarities
    cross_state = [p['similarity_score'] for p in similarity_data if p['state1'] != p['state2']]
    within_state = [p['similarity_score'] for p in similarity_data if p['state1'] == p['state2']]
    
    print("=" * 60)
    print("SIMILARITY DISTRIBUTION ANALYSIS FOR THRESHOLD SETTING")
    print("=" * 60)
    
    print(f"\nCROSS-STATE SIMILARITIES:")
    print(f"  Total pairs: {len(cross_state)}")
    print(f"  Mean: {np.mean(cross_state):.6f}")
    print(f"  Median: {np.median(cross_state):.6f}")
    print(f"  Standard deviation: {np.std(cross_state):.6f}")
    print(f"  Min: {np.min(cross_state):.6f}")
    print(f"  Max: {np.max(cross_state):.6f}")
    
    # Calculate percentiles
    percentiles = [50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(cross_state, p)
        count_above = sum(1 for s in cross_state if s >= val)
        print(f"    {p}th percentile: {val:.6f} ({count_above} pairs above this)")
    
    print(f"\nWITHIN-STATE SIMILARITIES (for comparison):")
    print(f"  Total pairs: {len(within_state)}")
    print(f"  Mean: {np.mean(within_state):.6f}")
    print(f"  Median: {np.median(within_state):.6f}")
    print(f"  Max: {np.max(within_state):.6f}")
    
    # Analyze what different thresholds would capture
    print(f"\nTHRESHOLD ANALYSIS:")
    
    thresholds = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    for threshold in thresholds:
        cross_count = sum(1 for s in cross_state if s >= threshold)
        within_count = sum(1 for s in within_state if s >= threshold)
        
        if cross_count > 0:
            max_cross_at_threshold = max([s for s in cross_state if s >= threshold])
            print(f"  Threshold {threshold:.3f}: {cross_count} cross-state pairs, {within_count} within-state pairs")
            print(f"    Highest cross-state similarity: {max_cross_at_threshold:.6f}")
        else:
            print(f"  Threshold {threshold:.3f}: No cross-state pairs above this threshold")
    
    # Recommendations based on legal text analysis best practices
    print(f"\nRECOMMENDATIONS FOR LEGAL TEXT ANALYSIS:")
    
    # For legal borrowing, we want to focus on meaningful similarities
    # that are likely to represent actual borrowing rather than coincidence
    
    # Use 90th percentile as a reasonable threshold for meaningful borrowing
    recommended_threshold = np.percentile(cross_state, 90)
    pairs_at_recommended = sum(1 for s in cross_state if s >= recommended_threshold)
    
    print(f"  1. CONSERVATIVE (90th percentile): {recommended_threshold:.6f}")
    print(f"     This captures {pairs_at_recommended} cross-state pairs")
    print(f"     Best for: High-confidence borrowing identification")
    
    # Alternative: Mean + 1 standard deviation
    alt_threshold = np.mean(cross_state) + np.std(cross_state)
    pairs_at_alt = sum(1 for s in cross_state if s >= alt_threshold)
    
    print(f"  2. MODERATE (mean + 1 std): {alt_threshold:.6f}")
    print(f"     This captures {pairs_at_alt} cross-state pairs")
    print(f"     Best for: Balance of precision and recall")
    
    # For network visualization, we might want something in between
    viz_threshold = 0.01  # Based on the data, this seems reasonable
    pairs_at_viz = sum(1 for s in cross_state if s >= viz_threshold)
    
    print(f"  3. VISUALIZATION (0.01): {viz_threshold:.6f}")
    print(f"     This captures {pairs_at_viz} cross-state pairs")
    print(f"     Best for: Network visualization with meaningful connections")
    
    print(f"\nRECOMMENDED THRESHOLD: {recommended_threshold:.6f}")
    print(f"This represents the 90th percentile of cross-state similarities,")
    print(f"focusing on the most significant potential borrowing relationships.")
    
    return recommended_threshold

if __name__ == "__main__":
    recommended = analyze_similarity_distribution()
    print(f"\n" + "=" * 60)
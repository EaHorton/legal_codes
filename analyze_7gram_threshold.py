#!/usr/bin/env python3
"""
Analyze 7-gram distribution to determine appropriate threshold for legal text borrowing
"""

import json
import numpy as np

def analyze_7gram_distribution():
    """Analyze the distribution of 7-gram similarities to set appropriate threshold"""
    
    # Load the 7-gram results
    with open('text_reuse_analysis_7gram.json', 'r') as f:
        data = json.load(f)
    
    similarities = [pair['similarity_score'] for pair in data['similarity_scores']]
    cross_state = [pair for pair in data['similarity_scores'] if pair['state1'] != pair['state2']]
    within_state = [pair for pair in data['similarity_scores'] if pair['state1'] == pair['state2']]
    
    print("=" * 60)
    print("7-GRAM SIMILARITY ANALYSIS FOR LEGAL TEXT BORROWING")
    print("=" * 60)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total document pairs: {len(similarities)}")
    print(f"  Cross-state pairs: {len(cross_state)}")
    print(f"  Within-state pairs: {len(within_state)}")
    
    print(f"\nALL SIMILARITIES:")
    print(f"  Mean: {np.mean(similarities):.6f}")
    print(f"  Median: {np.median(similarities):.6f}")
    print(f"  Max: {np.max(similarities):.6f}")
    print(f"  Standard deviation: {np.std(similarities):.6f}")
    
    # Cross-state analysis
    if cross_state:
        cross_similarities = [pair['similarity_score'] for pair in cross_state]
        print(f"\nCROSS-STATE SIMILARITIES:")
        print(f"  Count: {len(cross_similarities)}")
        print(f"  Mean: {np.mean(cross_similarities):.6f}")
        print(f"  Median: {np.median(cross_similarities):.6f}")
        print(f"  Max: {np.max(cross_similarities):.6f}")
        print(f"  Min: {np.min(cross_similarities):.6f}")
        
        # Percentile analysis
        percentiles = [75, 85, 90, 95, 98, 99]
        print(f"\n  Cross-state percentiles:")
        for p in percentiles:
            val = np.percentile(cross_similarities, p)
            count_above = sum(1 for s in cross_similarities if s >= val)
            print(f"    {p}th: {val:.6f} ({count_above} pairs above)")
    else:
        print(f"\nCROSS-STATE SIMILARITIES:")
        print(f"  No cross-state similarities found with current threshold")
        
        # Let's look at the highest cross-state similarities regardless of threshold
        all_cross_state_similarities = [pair['similarity_score'] for pair in cross_state]
        if all_cross_state_similarities:
            print(f"  Highest cross-state similarity: {max(all_cross_state_similarities):.6f}")
            print(f"  Mean cross-state similarity: {np.mean(all_cross_state_similarities):.6f}")
    
    # Within-state analysis for comparison
    if within_state:
        within_similarities = [pair['similarity_score'] for pair in within_state]
        print(f"\nWITHIN-STATE SIMILARITIES (for comparison):")
        print(f"  Count: {len(within_similarities)}")
        print(f"  Mean: {np.mean(within_similarities):.6f}")
        print(f"  Max: {np.max(within_similarities):.6f}")
    
    # Recommend threshold based on analysis
    print(f"\nTHRESHOLD RECOMMENDATIONS FOR 7-GRAMS:")
    
    if cross_state:
        cross_similarities = [pair['similarity_score'] for pair in cross_state]
        
        # For legal text with 7-grams, even small similarities can be meaningful
        conservative_threshold = np.percentile(cross_similarities, 90) if len(cross_similarities) > 10 else max(cross_similarities) * 0.5
        moderate_threshold = np.percentile(cross_similarities, 75) if len(cross_similarities) > 4 else max(cross_similarities) * 0.3
        
        print(f"  Conservative (90th percentile): {conservative_threshold:.6f}")
        print(f"  Moderate (75th percentile): {moderate_threshold:.6f}")
        
        # Check how many pairs each threshold would capture
        conservative_count = sum(1 for s in cross_similarities if s >= conservative_threshold)
        moderate_count = sum(1 for s in cross_similarities if s >= moderate_threshold)
        
        print(f"    Conservative would capture {conservative_count} cross-state pairs")
        print(f"    Moderate would capture {moderate_count} cross-state pairs")
        
    else:
        # If no cross-state similarities found, the threshold is too high
        # Look at all cross-state pairs regardless of current threshold
        all_cross_state = [pair['similarity_score'] for pair in cross_state]
        
        if all_cross_state:
            max_cross = max(all_cross_state)
            mean_cross = np.mean(all_cross_state)
            
            print(f"  Current threshold (0.02) too high for 7-grams")
            print(f"  Max cross-state similarity: {max_cross:.6f}")
            print(f"  Suggested threshold: {max_cross * 0.5:.6f} (50% of max)")
            print(f"  Alternative threshold: {mean_cross + np.std(all_cross_state):.6f} (mean + 1 std)")
            
            return max_cross * 0.5
        else:
            print(f"  No cross-state similarities detected")
            print(f"  Consider using 5-grams instead, or check document preprocessing")
            return 0.001
    
    return conservative_threshold if cross_state else 0.001

if __name__ == "__main__":
    recommended_threshold = analyze_7gram_distribution()
    print(f"\nRecommended threshold for 7-gram analysis: {recommended_threshold:.6f}")
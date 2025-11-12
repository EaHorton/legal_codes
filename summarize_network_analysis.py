#!/usr/bin/env python3
"""
Summary of Legal Code Borrowing Network Analysis Results
"""

import json
import numpy as np
from collections import defaultdict

def summarize_results():
    print("=" * 60)
    print("LEGAL CODE BORROWING NETWORK ANALYSIS - SUMMARY REPORT")
    print("=" * 60)
    
    # Load the data
    with open('text_reuse_analysis_5gram.json', 'r') as f:
        data = json.load(f)
    
    similarity_data = data['similarity_scores']
    
    # Separate cross-state and within-state pairs
    cross_state = [p for p in similarity_data if p['state1'] != p['state2']]
    within_state = [p for p in similarity_data if p['state1'] == p['state2']]
    
    print(f"\nDATA OVERVIEW:")
    print(f"  Total document pairs analyzed: {len(similarity_data):,}")
    print(f"  Cross-state pairs: {len(cross_state):,}")
    print(f"  Within-state pairs: {len(within_state):,}")
    
    # Filter for meaningful similarities
    meaningful_cross_state = [p for p in cross_state if p['similarity_score'] >= 0.005]
    meaningful_within_state = [p for p in within_state if p['similarity_score'] >= 0.005]
    
    print(f"\nMEANINGFUL SIMILARITIES (≥0.005):")
    print(f"  Cross-state pairs: {len(meaningful_cross_state)}")
    print(f"  Within-state pairs: {len(meaningful_within_state)}")
    
    # Cross-state analysis by state pairs
    print(f"\nCROSS-STATE BORROWING PATTERNS:")
    
    state_pairs = defaultdict(list)
    for pair in meaningful_cross_state:
        key = tuple(sorted([pair['state1'], pair['state2']]))
        state_pairs[key].append(pair['similarity_score'])
    
    for state_pair, similarities in state_pairs.items():
        print(f"\n  {state_pair[0].upper()} ↔ {state_pair[1].upper()}:")
        print(f"    Document pairs: {len(similarities)}")
        print(f"    Average similarity: {np.mean(similarities):.6f}")
        print(f"    Maximum similarity: {np.max(similarities):.6f}")
        print(f"    Range: {np.min(similarities):.6f} - {np.max(similarities):.6f}")
    
    # Find the highest similarity pairs
    print(f"\nTOP 5 CROSS-STATE SIMILARITIES:")
    top_cross_state = sorted(meaningful_cross_state, key=lambda x: x['similarity_score'], reverse=True)[:5]
    
    for i, pair in enumerate(top_cross_state, 1):
        print(f"\n  {i}. Similarity: {pair['similarity_score']:.6f}")
        print(f"     {pair['state1'].upper()}: {pair['document1'][:50]}...")
        print(f"     {pair['state2'].upper()}: {pair['document2'][:50]}...")
    
    print(f"\nKEY INSIGHTS:")
    
    # Insight 1: Most active borrowing relationship
    most_active = max(state_pairs.items(), key=lambda x: len(x[1]))
    print(f"  • Most active borrowing relationship: {most_active[0][0].upper()} ↔ {most_active[0][1].upper()}")
    print(f"    ({len(most_active[1])} document pairs with meaningful similarity)")
    
    # Insight 2: Strongest individual similarity
    strongest = max(meaningful_cross_state, key=lambda x: x['similarity_score'])
    print(f"  • Strongest cross-state similarity: {strongest['similarity_score']:.6f}")
    print(f"    Between {strongest['state1'].upper()} and {strongest['state2'].upper()}")
    
    # Insight 3: Compare with within-state similarities
    if meaningful_within_state:
        within_state_avg = np.mean([p['similarity_score'] for p in meaningful_within_state])
        cross_state_avg = np.mean([p['similarity_score'] for p in meaningful_cross_state])
        print(f"  • Average within-state similarity: {within_state_avg:.6f}")
        print(f"  • Average cross-state similarity: {cross_state_avg:.6f}")
        
        if cross_state_avg > 0:
            ratio = within_state_avg / cross_state_avg
            print(f"  • Within-state similarity is {ratio:.1f}x higher than cross-state")
    
    print(f"\nVISUALIZATIONS CREATED:")
    print(f"  📊 state_network.html - Interactive state relationship network")
    print(f"  📈 similarity_heatmap.html - State-to-state similarity matrix") 
    print(f"  🔗 cross_state_document_network.html - Document-level connections")
    print(f"  📋 comprehensive_borrowing_report.html - Full analysis report")
    print(f"  📋 network_analysis_report.html - Basic network analysis")
    
    print(f"\nRECOMMENDATIONS FOR LEGAL HISTORIANS:")
    
    if len(state_pairs) > 0:
        # Focus on the most promising relationship
        top_relationship = max(state_pairs.items(), key=lambda x: (len(x[1]), np.max(x[1])))
        state1, state2 = top_relationship[0]
        
        print(f"  1. Focus detailed analysis on {state1.upper()}-{state2.upper()} relationship")
        print(f"     • {len(top_relationship[1])} document pairs show similarity")
        print(f"     • Maximum similarity reaches {np.max(top_relationship[1]):.6f}")
        
        print(f"  2. Examine the top 3-5 highest similarity document pairs manually")
        print(f"     • Look for direct copying vs. adaptation patterns")
        print(f"     • Consider historical context and timing")
        
        print(f"  3. Use network visualizations to identify:")
        print(f"     • Hub documents that connect to multiple other documents")
        print(f"     • Clusters of related documents within states")
        print(f"     • Potential chains of influence across multiple states")
    
    print(f"\n" + "=" * 60)
    print(f"Analysis complete. Open the HTML files in your browser to explore interactively.")
    print(f"=" * 60)

if __name__ == "__main__":
    summarize_results()
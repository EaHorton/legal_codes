#!/usr/bin/env python3
"""
Enhanced Cross-State Borrowing Analysis
This script provides detailed analysis of cross-state legal code borrowing patterns.
"""

import json
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict, Counter
import os
from datetime import datetime

def load_and_analyze_cross_state_borrowing():
    """Enhanced analysis focusing specifically on cross-state borrowing patterns"""
    
    # Load the 5-gram analysis data
    with open('text_reuse_analysis_5gram.json', 'r') as f:
        data = json.load(f)
    
    similarity_data = data['similarity_scores']
    
    # Filter for cross-state pairs only
    cross_state_pairs = [
        pair for pair in similarity_data 
        if pair['state1'] != pair['state2'] and pair['similarity_score'] >= 0.005
    ]
    
    print(f"Found {len(cross_state_pairs)} cross-state document pairs with similarity >= 0.005")
    
    # Group by state pairs
    state_pair_analysis = defaultdict(list)
    for pair in cross_state_pairs:
        state_pair_key = tuple(sorted([pair['state1'], pair['state2']]))
        state_pair_analysis[state_pair_key].append(pair)
    
    # Create detailed analysis report
    print("\n=== DETAILED CROSS-STATE BORROWING ANALYSIS ===")
    
    for state_pair, pairs in state_pair_analysis.items():
        print(f"\n{state_pair[0].upper()} ↔ {state_pair[1].upper()}:")
        print(f"  Total document pairs with similarity: {len(pairs)}")
        
        similarities = [p['similarity_score'] for p in pairs]
        print(f"  Average similarity: {np.mean(similarities):.6f}")
        print(f"  Maximum similarity: {np.max(similarities):.6f}")
        print(f"  Standard deviation: {np.std(similarities):.6f}")
        
        # Find most similar documents
        top_pairs = sorted(pairs, key=lambda x: x['similarity_score'], reverse=True)[:3]
        print(f"  Top 3 most similar document pairs:")
        for i, pair in enumerate(top_pairs, 1):
            print(f"    {i}. Similarity: {pair['similarity_score']:.6f}")
            print(f"       {pair['state1'].upper()}: {pair['document1'][:60]}...")
            print(f"       {pair['state2'].upper()}: {pair['document2'][:60]}...")
    
    return cross_state_pairs, state_pair_analysis

def create_enhanced_network_visualization(cross_state_pairs):
    """Create an enhanced network visualization with multiple views"""
    
    # Create document-level network for high-similarity pairs
    high_sim_pairs = [p for p in cross_state_pairs if p['similarity_score'] >= 0.01]
    
    G = nx.Graph()
    
    # Add nodes and edges
    for pair in high_sim_pairs:
        doc1, doc2 = pair['document1'], pair['document2']
        state1, state2 = pair['state1'], pair['state2']
        similarity = pair['similarity_score']
        
        # Create shorter labels for visualization
        doc1_short = doc1.replace('_corrected.txt', '').replace('_corrected', '')[:40]
        doc2_short = doc2.replace('_corrected.txt', '').replace('_corrected', '')[:40]
        
        G.add_node(doc1, state=state1, label=doc1_short, full_name=doc1)
        G.add_node(doc2, state=state2, label=doc2_short, full_name=doc2)
        G.add_edge(doc1, doc2, weight=similarity)
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Color mapping for states
    color_map = {'al': '#FF6B6B', 'nc': '#4ECDC4', 'tn': '#45B7D1'}
    
    # Prepare traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(125,125,125,0.3)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Node traces by state
    node_traces = []
    for state in ['al', 'nc', 'tn']:
        state_nodes = [node for node in G.nodes() if G.nodes[node]['state'] == state]
        if state_nodes:
            node_x = [pos[node][0] for node in state_nodes]
            node_y = [pos[node][1] for node in state_nodes]
            
            node_info = []
            for node in state_nodes:
                connections = list(G.neighbors(node))
                connected_states = set(G.nodes[neighbor]['state'] for neighbor in connections)
                
                node_info.append(
                    f"State: {state.upper()}<br>"
                    f"Document: {G.nodes[node]['full_name']}<br>"
                    f"Connections: {len(connections)}<br>"
                    f"Connected to states: {', '.join(s.upper() for s in connected_states)}"
                )
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                hovertext=node_info,
                marker=dict(
                    size=10,
                    color=color_map[state],
                    line=dict(width=1, color='white')
                ),
                name=state.upper(),
                showlegend=True
            )
            node_traces.append(node_trace)
    
    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=dict(
            text='Cross-State Legal Code Document Network<br>'
                 '<sub>Documents connected by text similarity (threshold ≥ 0.01)</sub>',
            font=dict(size=16)
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        showlegend=True,
        legend=dict(x=1, y=1, bgcolor='rgba(255,255,255,0.8)'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    fig.write_html('visualizations/cross_state_document_network.html')
    print("Cross-state document network saved to visualizations/cross_state_document_network.html")
    
    return fig

def create_borrowing_timeline_analysis(cross_state_pairs):
    """Analyze borrowing patterns over time based on document names"""
    
    # Extract years from document names where possible
    import re
    
    year_pattern = r'(\d{4})'
    temporal_data = []
    
    for pair in cross_state_pairs:
        # Try to extract years from both documents
        doc1_years = re.findall(year_pattern, pair['document1'])
        doc2_years = re.findall(year_pattern, pair['document2'])
        
        if doc1_years and doc2_years:
            year1 = int(doc1_years[0])
            year2 = int(doc2_years[0])
            
            temporal_data.append({
                'year1': year1,
                'year2': year2,
                'state1': pair['state1'],
                'state2': pair['state2'],
                'similarity': pair['similarity_score'],
                'year_diff': abs(year1 - year2)
            })
    
    if temporal_data:
        df = pd.DataFrame(temporal_data)
        
        # Create year difference vs similarity plot
        fig = go.Figure()
        
        state_pairs = df[['state1', 'state2']].apply(lambda x: '-'.join(sorted([x['state1'], x['state2']])), axis=1).unique()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, state_pair in enumerate(state_pairs):
            mask = df[['state1', 'state2']].apply(lambda x: '-'.join(sorted([x['state1'], x['state2']])), axis=1) == state_pair
            subset = df[mask]
            
            fig.add_trace(go.Scatter(
                x=subset['year_diff'],
                y=subset['similarity'],
                mode='markers',
                name=state_pair.upper().replace('-', ' - '),
                marker=dict(size=8, color=colors[i % len(colors)], opacity=0.7)
            ))
        
        fig.update_layout(
            title='Document Similarity vs. Year Difference<br><sub>Analysis of temporal patterns in legal code borrowing</sub>',
            xaxis_title='Year Difference Between Documents',
            yaxis_title='Similarity Score',
            hovermode='closest'
        )
        
        fig.write_html('visualizations/temporal_borrowing_analysis.html')
        print("Temporal borrowing analysis saved to visualizations/temporal_borrowing_analysis.html")
        
        # Print summary statistics
        print(f"\nTemporal Analysis Summary:")
        print(f"  Documents with extractable years: {len(temporal_data)}")
        print(f"  Average year difference: {df['year_diff'].mean():.1f} years")
        print(f"  Most common year differences: {df['year_diff'].value_counts().head()}")
        
        return fig
    else:
        print("No temporal data could be extracted from document names")
        return None

def create_comprehensive_report(cross_state_pairs, state_pair_analysis):
    """Create a comprehensive HTML report"""
    
    total_similarities = [pair['similarity_score'] for pair in cross_state_pairs]
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cross-State Legal Code Borrowing Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-left: 4px solid #007bff; }}
            .stat {{ display: inline-block; margin: 10px 20px; background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stat-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
            .stat-label {{ font-size: 0.9em; color: #666; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .highlight {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .visualization-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .viz-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .method-box {{ background: #e9ecef; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <h1>Cross-State Legal Code Borrowing Analysis</h1>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <p>This analysis examines text reuse patterns between legal codes from Alabama (AL), North Carolina (NC), and Tennessee (TN) using n-gram similarity analysis. The findings reveal evidence of legal code borrowing between states, with varying degrees of similarity indicating different types of textual relationships.</p>
        </div>
        
        <div style="display: flex; justify-content: space-around; margin: 30px 0;">
            <div class="stat">
                <div class="stat-value">{len(cross_state_pairs)}</div>
                <div class="stat-label">Cross-State Document Pairs</div>
            </div>
            <div class="stat">
                <div class="stat-value">{np.mean(total_similarities):.4f}</div>
                <div class="stat-label">Average Similarity</div>
            </div>
            <div class="stat">
                <div class="stat-value">{np.max(total_similarities):.4f}</div>
                <div class="stat-label">Maximum Similarity</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(state_pair_analysis)}</div>
                <div class="stat-label">State Pair Relationships</div>
            </div>
        </div>

        <div class="method-box">
            <h3>Methodology</h3>
            <p><strong>N-gram Analysis:</strong> We used 5-gram analysis with Jaccard similarity to compare documents. This method identifies shared sequences of 5 consecutive words, providing a robust measure of textual similarity that can detect both direct copying and paraphrasing.</p>
            <p><strong>Threshold:</strong> Cross-state pairs with similarity ≥ 0.005 were included in this analysis to focus on meaningful relationships while filtering out random similarities.</p>
        </div>

        <h2>Key Findings</h2>
        
        <div class="highlight">
            <strong>Primary Finding:</strong> Tennessee and North Carolina show the strongest borrowing relationship, with {len(state_pair_analysis.get(('nc', 'tn'), []))} document pairs showing similarity above the threshold.
        </div>

        <h3>State-by-State Analysis</h3>
        <table>
            <tr>
                <th>State Pair</th>
                <th>Document Pairs</th>
                <th>Avg Similarity</th>
                <th>Max Similarity</th>
                <th>Interpretation</th>
            </tr>
    """
    
    # Add state pair data to table
    for state_pair, pairs in state_pair_analysis.items():
        similarities = [p['similarity_score'] for p in pairs]
        max_sim = np.max(similarities)
        avg_sim = np.mean(similarities)
        
        # Interpret the relationship
        if max_sim > 0.01:
            interpretation = "Strong evidence of borrowing"
        elif max_sim > 0.008:
            interpretation = "Moderate similarity patterns"
        else:
            interpretation = "Weak similarity patterns"
        
        report_html += f"""
            <tr>
                <td>{state_pair[0].upper()} - {state_pair[1].upper()}</td>
                <td>{len(pairs)}</td>
                <td>{avg_sim:.6f}</td>
                <td>{max_sim:.6f}</td>
                <td>{interpretation}</td>
            </tr>
        """
    
    report_html += f"""
        </table>

        <h2>Interactive Visualizations</h2>
        <div class="visualization-grid">
            <div class="viz-card">
                <h4>State Network</h4>
                <p>Shows overall relationships between states based on text similarity.</p>
                <a href="state_network.html" target="_blank">→ View State Network</a>
            </div>
            <div class="viz-card">
                <h4>Document Network</h4>
                <p>Individual documents connected by similarity (high-similarity pairs only).</p>
                <a href="cross_state_document_network.html" target="_blank">→ View Document Network</a>
            </div>
            <div class="viz-card">
                <h4>Similarity Heatmap</h4>
                <p>Matrix view showing average similarities between all state pairs.</p>
                <a href="similarity_heatmap.html" target="_blank">→ View Heatmap</a>
            </div>
            <div class="viz-card">
                <h4>Temporal Analysis</h4>
                <p>Analysis of borrowing patterns over time (where date information is available).</p>
                <a href="temporal_borrowing_analysis.html" target="_blank">→ View Temporal Analysis</a>
            </div>
        </div>

        <h2>Implications for Legal History Research</h2>
        <ul>
            <li><strong>Evidence of Interstate Legal Influence:</strong> The similarity patterns suggest that states did reference and adapt each other's legal codes.</li>
            <li><strong>Regional Patterns:</strong> The stronger TN-NC relationship may reflect geographic, cultural, or political connections between these neighboring states.</li>
            <li><strong>Methodological Validation:</strong> The network analysis approach successfully identifies potential borrowing relationships that merit further qualitative examination.</li>
        </ul>

        <h2>Recommended Next Steps</h2>
        <ol>
            <li><strong>Qualitative Analysis:</strong> Manually examine the highest-similarity document pairs to confirm borrowing and understand context.</li>
            <li><strong>Historical Context:</strong> Research the political and legal context during periods of high similarity to understand motivation for borrowing.</li>
            <li><strong>Expanded Corpus:</strong> Include additional states and time periods to map broader patterns of legal code diffusion.</li>
            <li><strong>Topic Modeling:</strong> Identify which specific legal topics show the most cross-state similarity.</li>
        </ol>

        <hr>
        <p><small>Generated by Legal Code Network Analysis Tool | {datetime.now().strftime("%Y-%m-%d")}</small></p>
    </body>
    </html>
    """
    
    with open('visualizations/comprehensive_borrowing_report.html', 'w') as f:
        f.write(report_html)
    
    print("Comprehensive report saved to visualizations/comprehensive_borrowing_report.html")

def main():
    print("Starting Enhanced Cross-State Borrowing Analysis...")
    
    # Load and analyze data
    cross_state_pairs, state_pair_analysis = load_and_analyze_cross_state_borrowing()
    
    # Create enhanced visualizations
    create_enhanced_network_visualization(cross_state_pairs)
    create_borrowing_timeline_analysis(cross_state_pairs)
    
    # Create comprehensive report
    create_comprehensive_report(cross_state_pairs, state_pair_analysis)
    
    print("\n=== ENHANCED ANALYSIS COMPLETE ===")
    print("New visualizations created:")
    print("- cross_state_document_network.html: Document-level network")
    print("- temporal_borrowing_analysis.html: Timeline analysis")
    print("- comprehensive_borrowing_report.html: Full analysis report")

if __name__ == "__main__":
    main()
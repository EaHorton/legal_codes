#!/usr/bin/env python3
"""
Enhanced Visualization Suite for K-Shingles Text Reuse Analysis
Creates comprehensive visualizations for the enhanced n-gram + k-shingles analysis
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def load_enhanced_results(json_path: str) -> Dict:
    """Load enhanced analysis results with k-shingles data."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_similarity_dashboard(results: Dict, output_file: str):
    """Create a comprehensive dashboard showing all similarity measures."""
    pairs = results['similarity_scores']
    
    # Prepare data
    similarity_types = ['ngram_similarity', 'char5_similarity', 'char10_similarity', 
                       'word3_similarity', 'combined_similarity']
    
    # Create subplot layout
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Distribution by Similarity Type', 'Cross-State vs Within-State',
            'Correlation Matrix', 'Top Cross-State Pairs',
            'Similarity Trends', 'State Network Strength'
        ),
        specs=[
            [{"colspan": 1}, {"colspan": 1}],
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"colspan": 2}, None]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Distribution by similarity type (violin plot)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (sim_type, color) in enumerate(zip(similarity_types, colors)):
        scores = [p[sim_type] for p in pairs if sim_type in p]
        
        fig.add_trace(
            go.Violin(
                y=scores,
                name=sim_type.replace('_', ' ').title(),
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                opacity=0.6,
                x0=sim_type
            ),
            row=1, col=1
        )
    
    # 2. Cross-state vs within-state comparison
    cross_state = [p for p in pairs if p['state1'] != p['state2']]
    within_state = [p for p in pairs if p['state1'] == p['state2']]
    
    comparison_data = []
    for sim_type in similarity_types:
        cross_scores = [p[sim_type] for p in cross_state if sim_type in p]
        within_scores = [p[sim_type] for p in within_state if sim_type in p]
        
        comparison_data.extend([
            {'Type': sim_type.replace('_', ' ').title(), 'Category': 'Cross-State', 'Score': np.mean(cross_scores)},
            {'Type': sim_type.replace('_', ' ').title(), 'Category': 'Within-State', 'Score': np.mean(within_scores)}
        ])
    
    comp_df = pd.DataFrame(comparison_data)
    
    for category in ['Cross-State', 'Within-State']:
        subset = comp_df[comp_df['Category'] == category]
        fig.add_trace(
            go.Bar(
                x=subset['Type'],
                y=subset['Score'],
                name=category,
                marker_color='red' if category == 'Cross-State' else 'blue'
            ),
            row=1, col=2
        )
    
    # 3. Correlation matrix between similarity measures
    corr_data = []
    for pair in pairs[:1000]:  # Limit for performance
        row = [pair.get(sim_type, 0) for sim_type in similarity_types]
        corr_data.append(row)
    
    corr_df = pd.DataFrame(corr_data, columns=[s.replace('_', ' ').title() for s in similarity_types])
    corr_matrix = corr_df.corr()
    
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text:.2f}",
            showscale=True
        ),
        row=2, col=1
    )
    
    # 4. Top cross-state pairs
    top_cross_state = sorted(cross_state, key=lambda x: x['combined_similarity'], reverse=True)[:10]
    
    pair_names = [f"{p['state1'].upper()}-{p['state2'].upper()}" for p in top_cross_state]
    combined_scores = [p['combined_similarity'] for p in top_cross_state]
    
    fig.add_trace(
        go.Bar(
            x=pair_names,
            y=combined_scores,
            marker_color='darkred',
            name='Top Cross-State'
        ),
        row=2, col=2
    )
    
    # 5. Similarity trends (scatter plot matrix)
    sample_pairs = pairs[:500]  # Sample for performance
    
    for i, sim_type in enumerate(['ngram_similarity', 'char5_similarity', 'combined_similarity']):
        fig.add_trace(
            go.Scatter(
                x=[p['ngram_similarity'] for p in sample_pairs],
                y=[p[sim_type] for p in sample_pairs],
                mode='markers',
                marker=dict(
                    color=[1 if p['state1'] != p['state2'] else 0 for p in sample_pairs],
                    colorscale='RdYlBu',
                    size=4
                ),
                name=f'N-gram vs {sim_type.replace("_", " ").title()}'
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title_text="Enhanced Text Reuse Analysis Dashboard (N-grams + K-Shingles)",
        height=1200,
        width=1600,
        showlegend=True
    )
    
    # Update subplot titles and axes
    fig.update_xaxes(title_text="Similarity Type", row=1, col=1)
    fig.update_yaxes(title_text="Similarity Score", row=1, col=1)
    
    fig.update_xaxes(title_text="Measure Type", row=1, col=2)
    fig.update_yaxes(title_text="Average Score", row=1, col=2)
    
    fig.update_xaxes(title_text="State Pairs", row=2, col=2)
    fig.update_yaxes(title_text="Combined Similarity", row=2, col=2)
    
    fig.update_xaxes(title_text="N-gram Similarity", row=3, col=1)
    fig.update_yaxes(title_text="Other Similarities", row=3, col=1)
    
    fig.write_html(output_file)
    return len(pairs), len(cross_state), len(within_state)

def create_method_comparison_analysis(results: Dict, output_file: str):
    """Create detailed comparison of different similarity methods."""
    pairs = results['similarity_scores']
    
    # Calculate method effectiveness metrics
    similarity_types = ['ngram_similarity', 'char5_similarity', 'char10_similarity', 
                       'word3_similarity']
    
    metrics_data = []
    
    for sim_type in similarity_types:
        scores = [p[sim_type] for p in pairs if sim_type in p]
        cross_state_scores = [p[sim_type] for p in pairs if p['state1'] != p['state2'] and sim_type in p]
        
        metrics_data.append({
            'Method': sim_type.replace('_', ' ').title(),
            'Mean Score': np.mean(scores),
            'Std Dev': np.std(scores),
            'Max Score': np.max(scores),
            'Cross-State Mean': np.mean(cross_state_scores) if cross_state_scores else 0,
            'Discrimination Power': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
            'Cross-State Ratio': len(cross_state_scores) / len(scores) if scores else 0
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create comparison visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Method Effectiveness', 'Score Distributions',
            'Cross-State Detection Power', 'Method Correlations'
        ),
        specs=[
            [{"type": "bar"}, {"type": "box"}],
            [{"type": "scatter"}, {"type": "heatmap"}]
        ]
    )
    
    # 1. Method effectiveness radar chart (as bar chart)
    fig.add_trace(
        go.Bar(
            x=metrics_df['Method'],
            y=metrics_df['Discrimination Power'],
            name='Discrimination Power',
            marker_color='blue'
        ),
        row=1, col=1
    )
    
    # 2. Score distributions
    for sim_type in similarity_types:
        scores = [p[sim_type] for p in pairs if sim_type in p]
        fig.add_trace(
            go.Box(
                y=scores,
                name=sim_type.replace('_', ' ').title(),
                boxpoints='outliers'
            ),
            row=1, col=2
        )
    
    # 3. Cross-state detection power
    fig.add_trace(
        go.Scatter(
            x=metrics_df['Cross-State Mean'],
            y=metrics_df['Discrimination Power'],
            mode='markers+text',
            text=metrics_df['Method'],
            textposition='top center',
            marker=dict(size=12, color='red'),
            name='Methods'
        ),
        row=2, col=1
    )
    
    # 4. Method correlation heatmap
    method_corr_data = []
    for pair in pairs[:1000]:  # Limit for performance
        row = [pair.get(sim_type, 0) for sim_type in similarity_types]
        method_corr_data.append(row)
    
    method_corr_df = pd.DataFrame(method_corr_data, columns=[s.replace('_', ' ').title() for s in similarity_types])
    method_corr_matrix = method_corr_df.corr()
    
    fig.add_trace(
        go.Heatmap(
            z=method_corr_matrix.values,
            x=method_corr_matrix.columns,
            y=method_corr_matrix.index,
            colorscale='Viridis',
            text=method_corr_matrix.values,
            texttemplate="%{text:.2f}",
            showscale=True
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Similarity Method Comparison Analysis",
        height=1000,
        width=1400,
        showlegend=True
    )
    
    fig.write_html(output_file)

def create_enhanced_network_visualization(results: Dict, output_file: str, min_similarity: float = 0.01):
    """Create an enhanced network visualization with k-shingles data."""
    pairs = results['similarity_scores']
    
    # Create network
    G = nx.Graph()
    
    for pair in pairs:
        if pair['combined_similarity'] >= min_similarity:
            G.add_node(pair['document1'], state=pair['state1'])
            G.add_node(pair['document2'], state=pair['state2'])
            
            G.add_edge(
                pair['document1'], 
                pair['document2'],
                weight=pair['combined_similarity'],
                ngram_sim=pair['ngram_similarity'],
                char5_sim=pair['char5_similarity'],
                char10_sim=pair['char10_similarity'],
                word3_sim=pair['word3_similarity'],
                cross_state=pair['state1'] != pair['state2']
            )
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Separate cross-state and within-state edges
    cross_state_edges = [(u, v) for u, v, d in G.edges(data=True) if d['cross_state']]
    within_state_edges = [(u, v) for u, v, d in G.edges(data=True) if not d['cross_state']]
    
    fig = go.Figure()
    
    # Add within-state edges
    for edge in within_state_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1 + 3*weight, color='rgba(100,100,100,0.3)'),
                hoverinfo='skip',
                showlegend=False
            )
        )
    
    # Add cross-state edges (highlighted)
    for edge in cross_state_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_data = G.edges[edge]
        
        hover_text = (
            f"<b>Cross-State Connection</b><br>"
            f"Combined: {edge_data['weight']:.4f}<br>"
            f"N-gram: {edge_data['ngram_sim']:.4f}<br>"
            f"Char-5: {edge_data['char5_sim']:.4f}<br>"
            f"Char-10: {edge_data['char10_sim']:.4f}<br>"
            f"Word-3: {edge_data['word3_sim']:.4f}"
        )
        
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2 + 5*edge_data['weight'], color='red'),
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False
            )
        )
    
    # Add nodes
    states = set(nx.get_node_attributes(G, 'state').values())
    state_colors = {state: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] 
                   for i, state in enumerate(sorted(states))}
    
    for node in G.nodes():
        x, y = pos[node]
        state = G.nodes[node]['state']
        
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=state_colors[state],
                    line=dict(width=2, color='white')
                ),
                text=node[:10] + '...',
                textposition='bottom center',
                hovertext=f"Document: {node}<br>State: {state.upper()}",
                hoverinfo='text',
                showlegend=False
            )
        )
    
    # Add legend for states
    for state in sorted(states):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=state_colors[state]),
                name=state.upper(),
                showlegend=True
            )
        )
    
    fig.update_layout(
        title="Enhanced Legal Code Network (N-grams + K-Shingles)<br><sub>Red lines = Cross-state connections</sub>",
        showlegend=True,
        hovermode='closest',
        width=1200,
        height=800,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    fig.write_html(output_file)
    return len(cross_state_edges), len(within_state_edges)

def main():
    """Main function to create all enhanced visualizations."""
    
    # Load enhanced analysis results
    enhanced_file = "text_reuse_analysis_5gram_shingles.json"
    
    try:
        results = load_enhanced_results(enhanced_file)
        print(f"Loaded enhanced analysis results from {enhanced_file}")
    except FileNotFoundError:
        print(f"Enhanced results file {enhanced_file} not found!")
        print("Please run analyze_5gram_thresholds.py first to generate enhanced results.")
        return
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating enhanced visualizations...")
    
    # 1. Comprehensive dashboard
    total_pairs, cross_pairs, within_pairs = create_similarity_dashboard(
        results, output_dir / "enhanced_dashboard.html"
    )
    
    # 2. Method comparison analysis
    create_method_comparison_analysis(results, output_dir / "method_comparison.html")
    
    # 3. Enhanced network visualization
    cross_edges, within_edges = create_enhanced_network_visualization(
        results, output_dir / "enhanced_network.html"
    )
    
    print("\n" + "="*60)
    print("ENHANCED VISUALIZATION SUITE COMPLETE")
    print("="*60)
    print(f"Total document pairs analyzed: {total_pairs}")
    print(f"Cross-state pairs: {cross_pairs}")
    print(f"Within-state pairs: {within_pairs}")
    print(f"Cross-state network edges: {cross_edges}")
    print(f"Within-state network edges: {within_edges}")
    print("\nGenerated visualizations:")
    print("1. enhanced_dashboard.html - Comprehensive analysis dashboard")
    print("2. method_comparison.html - Detailed method effectiveness analysis")
    print("3. enhanced_network.html - Interactive network with k-shingles data")
    print("\nAll files saved in the 'visualizations' directory.")

if __name__ == "__main__":
    main()
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots

def load_detailed_results(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def detect_analysis_format(results):
    """Detect the format of analysis results."""
    if 'similarity_scores' in results:
        return 'enhanced'
    elif 'document_pairs' in results:
        return 'detailed'
    else:
        return 'unknown'

def filter_cross_state_pairs(results):
    """Filter cross-state pairs from both enhanced and detailed analysis formats."""
    format_type = detect_analysis_format(results)
    
    if format_type == 'enhanced':
        pairs = results['similarity_scores']
    elif format_type == 'detailed':
        pairs = results['document_pairs']
    else:
        print(f"Unknown analysis format")
        return []
    
    cross_state_pairs = []
    for pair in pairs:
        if pair['state1'] != pair['state2']:
            cross_state_pairs.append(pair)
    
    return cross_state_pairs

def create_cross_state_network(pairs, output_file):
    """Create cross-state network visualization supporting both analysis formats."""
    if not pairs:
        print("No cross-state matches found.")
        return
    
    # Determine if this is enhanced or detailed analysis format
    has_metrics = 'metrics' in pairs[0] if pairs else False
    has_similarities = 'combined_similarity' in pairs[0] if pairs else False
    
    if has_metrics:
        # Detailed analysis format
        df_data = []
        for p in pairs:
            df_data.append({
                'doc1': p['document1'],
                'doc2': p['document2'],
                'state1': p['state1'],
                'state2': p['state2'],
                'coverage1': p['metrics']['coverage_text1'],
                'coverage2': p['metrics']['coverage_text2'],
                'num_matches': p['metrics']['num_matches'],
                'avg_match_length': p['metrics']['avg_match_length'],
                'similarity': p.get('similarity_score', 0)
            })
    else:
        # Enhanced analysis format
        df_data = []
        for p in pairs:
            # Use similarity as a proxy for coverage
            similarity = p.get('combined_similarity', p.get('similarity_score', 0))
            df_data.append({
                'doc1': p['document1'],
                'doc2': p['document2'],
                'state1': p['state1'],
                'state2': p['state2'],
                'coverage1': similarity,  # Use similarity as coverage proxy
                'coverage2': similarity,
                'similarity': similarity,
                'ngram_sim': p.get('ngram_similarity', 0),
                'char5_sim': p.get('char5_similarity', 0),
                'char10_sim': p.get('char10_similarity', 0),
                'word3_sim': p.get('word3_similarity', 0),
                'combined_sim': p.get('combined_similarity', 0)
            })
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        print("No cross-state matches found.")
        return
    
    # Create enhanced network visualization
    fig = go.Figure()
    
    # Create state positions
    states = list(set(df['state1'].tolist() + df['state2'].tolist()))
    state_positions = {state: i for i, state in enumerate(sorted(states))}
    
    # Add edges with enhanced hover information
    for _, row in df.iterrows():
        x_coords = [state_positions[row['state1']], state_positions[row['state2']]]
        y_coords = [row['coverage1'], row['coverage2']]
        
        # Create hover text based on available data
        if has_similarities:
            hover_text = (
                f"<b>{row['doc1']}</b> ↔ <b>{row['doc2']}</b><br>"
                f"States: {row['state1'].upper()} ↔ {row['state2'].upper()}<br>"
                f"Combined Similarity: {row['combined_sim']:.4f}<br>"
                f"N-gram: {row['ngram_sim']:.4f}<br>"
                f"Char-5: {row['char5_sim']:.4f}<br>"
                f"Char-10: {row['char10_sim']:.4f}<br>"
                f"Word-3: {row['word3_sim']:.4f}"
            )
        else:
            hover_text = (
                f"<b>{row['doc1']}</b> ↔ <b>{row['doc2']}</b><br>"
                f"States: {row['state1'].upper()} ↔ {row['state2'].upper()}<br>"
                f"Similarity: {row['similarity']:.4f}<br>"
                f"Coverage 1: {row['coverage1']:.4f}<br>"
                f"Coverage 2: {row['coverage2']:.4f}"
            )
        
        # Line width based on similarity
        line_width = 1 + 5 * row['similarity']
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines+markers',
            marker=dict(size=8, color='red'),
            line=dict(width=line_width, color='rgba(0,100,200,0.6)'),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False
        ))
    
    # Add state labels
    for state, pos in state_positions.items():
        fig.add_trace(go.Scatter(
            x=[pos], y=[0],
            mode='markers+text',
            marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
            text=[state.upper()],
            textposition="middle center",
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    title = "Cross-State Legal Code Text Reuse"
    if has_similarities:
        title += " (Enhanced with K-Shingles)"
    
    fig.update_layout(
        title=title,
        xaxis_title="State",
        yaxis_title="Similarity/Coverage Score",
        showlegend=False,
        height=600,
        width=800,
        xaxis=dict(
            tickmode='array',
            tickvals=list(state_positions.values()),
            ticktext=[s.upper() for s in sorted(states)]
        )
    )
    
    fig.write_html(output_file)
    print(f"Cross-state visualization saved to {output_file}")

def create_cross_state_heatmap(pairs, output_file):
    """Create cross-state heatmap supporting both analysis formats."""
    if not pairs:
        print("No cross-state matches found for heatmap.")
        return
    
    # Determine analysis format and extract appropriate similarity measure
    has_metrics = 'metrics' in pairs[0] if pairs else False
    has_similarities = 'combined_similarity' in pairs[0] if pairs else False
    
    df_data = []
    for p in pairs:
        if has_metrics:
            # Use coverage from detailed analysis
            similarity_score = max(p['metrics']['coverage_text1'], p['metrics']['coverage_text2'])
        else:
            # Use combined similarity from enhanced analysis
            similarity_score = p.get('combined_similarity', p.get('similarity_score', 0))
        
        df_data.append({
            'state1': p['state1'],
            'state2': p['state2'],
            'similarity': similarity_score
        })
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        print("No cross-state matches found for heatmap.")
        return
    
    # Group by state pairs and aggregate
    state_similarity = df.groupby(['state1', 'state2'])['similarity'].agg(['max', 'mean', 'count']).reset_index()
    
    # Get all unique states
    all_states = sorted(set(df['state1'].tolist() + df['state2'].tolist()))
    
    # Create matrices for different metrics
    matrices = {}
    for metric in ['max', 'mean', 'count']:
        matrix = pd.DataFrame(0.0, index=all_states, columns=all_states)
        
        for _, row in state_similarity.iterrows():
            matrix.loc[row['state1'], row['state2']] = row[metric]
            matrix.loc[row['state2'], row['state1']] = row[metric]  # Make symmetric
        
        matrices[metric] = matrix
    
    # Create subplot with multiple heatmaps
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Maximum Similarity', 'Average Similarity', 'Number of Connections'),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # Add heatmaps
    metrics_info = [
        ('max', 'Blues', 'Max Similarity'),
        ('mean', 'Viridis', 'Avg Similarity'), 
        ('count', 'Reds', 'Connections')
    ]
    
    for i, (metric, colorscale, title) in enumerate(metrics_info, 1):
        matrix = matrices[metric]
        
        fig.add_trace(
            go.Heatmap(
                z=matrix.values,
                x=matrix.columns,
                y=matrix.index,
                colorscale=colorscale,
                showscale=True,
                text=matrix.values,
                texttemplate="%{text:.3f}" if metric != 'count' else "%{text:.0f}",
                textfont={"size": 10},
                hovertemplate=f'{title}: %{{z:.3f}}<br>%{{y}} ↔ %{{x}}<extra></extra>'
            ),
            row=1, col=i
        )
    
    # Update layout
    title_text = "Cross-State Text Reuse Analysis"
    if has_similarities:
        title_text += " (Enhanced with K-Shingles)"
    
    fig.update_layout(
        title_text=title_text,
        height=500,
        width=1400
    )
    
    fig.write_html(output_file)
    print(f"Cross-state heatmap saved to {output_file}")

def create_similarity_breakdown_chart(pairs, output_file):
    """Create a breakdown chart of different similarity measures for cross-state pairs."""
    if not pairs or 'combined_similarity' not in pairs[0]:
        print("Similarity breakdown requires enhanced analysis results.")
        return
    
    # Prepare data
    similarity_types = ['ngram_similarity', 'char5_similarity', 'char10_similarity', 
                       'word3_similarity', 'combined_similarity']
    
    data = []
    for pair in pairs[:20]:  # Top 20 cross-state pairs
        pair_name = f"{pair['state1'].upper()}-{pair['state2'].upper()}"
        for sim_type in similarity_types:
            if sim_type in pair:
                data.append({
                    'Pair': pair_name,
                    'Documents': f"{pair['document1'][:15]}...↔{pair['document2'][:15]}...",
                    'Similarity Type': sim_type.replace('_', ' ').title(),
                    'Score': pair[sim_type]
                })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    fig = px.bar(df, 
                 x='Pair', 
                 y='Score',
                 color='Similarity Type',
                 title='Cross-State Similarity Breakdown (Top 20 Pairs)',
                 barmode='group',
                 hover_data=['Documents'])
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        width=1200,
        showlegend=True
    )
    
    fig.write_html(output_file)
    print(f"Similarity breakdown chart saved to {output_file}")

def main():
    # Try enhanced analysis results first, then fall back to detailed analysis
    enhanced_file = "text_reuse_analysis_5gram_shingles.json"
    detailed_file = "text_reuse_detailed_analysis.json"
    
    results_file = enhanced_file
    try:
        results = load_detailed_results(enhanced_file)
        print(f"Loaded enhanced analysis results from {enhanced_file}")
    except FileNotFoundError:
        try:
            results = load_detailed_results(detailed_file)
            results_file = detailed_file
            print(f"Loaded detailed analysis results from {detailed_file}")
        except FileNotFoundError:
            print(f"Neither {enhanced_file} nor {detailed_file} found!")
            return
    
    cross_state_pairs = filter_cross_state_pairs(results)
    
    if not cross_state_pairs:
        print("No cross-state pairs found!")
        return
    
    print(f"Found {len(cross_state_pairs)} cross-state pairs")
    
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Create visualizations
    create_cross_state_network(cross_state_pairs, output_dir / 'cross_state_network_enhanced.html')
    create_cross_state_heatmap(cross_state_pairs, output_dir / 'cross_state_heatmap_enhanced.html')
    
    # Create similarity breakdown if enhanced data is available
    if 'combined_similarity' in cross_state_pairs[0]:
        create_similarity_breakdown_chart(cross_state_pairs, output_dir / 'cross_state_similarity_breakdown.html')
    
    print("\nEnhanced cross-state visualizations created:")
    print("1. cross_state_network_enhanced.html - Interactive cross-state network")
    print("2. cross_state_heatmap_enhanced.html - Multi-metric state-to-state heatmap")
    if 'combined_similarity' in cross_state_pairs[0]:
        print("3. cross_state_similarity_breakdown.html - Detailed similarity measure comparison")

if __name__ == "__main__":
    main()

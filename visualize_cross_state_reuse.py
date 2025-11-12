import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def load_detailed_results(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def filter_cross_state_pairs(results):
    cross_state_pairs = []
    for pair in results['document_pairs']:
        if pair['state1'] != pair['state2']:
            cross_state_pairs.append(pair)
    return cross_state_pairs

def create_cross_state_network(pairs, output_file):
    # Build a DataFrame for easier processing
    df = pd.DataFrame([
        {
            'doc1': p['document1'],
            'doc2': p['document2'],
            'state1': p['state1'],
            'state2': p['state2'],
            'coverage1': p['metrics']['coverage_text1'],
            'coverage2': p['metrics']['coverage_text2'],
            'num_matches': p['metrics']['num_matches'],
            'avg_match_length': p['metrics']['avg_match_length']
        }
        for p in pairs
    ])
    if df.empty:
        print("No cross-state matches found.")
        return
    # Create a network graph using Plotly
    fig = go.Figure()
    # Add edges
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['state1'], row['state2']],
            y=[row['coverage1'], row['coverage2']],
            mode='lines+markers+text',
            text=[row['doc1'], row['doc2']],
            marker=dict(size=10),
            line=dict(width=2, color='blue'),
            name=f"{row['state1']} - {row['state2']}"
        ))
    fig.update_layout(
        title="Cross-State Legal Code Text Reuse",
        xaxis_title="State",
        yaxis_title="Coverage (fraction of document)",
        showlegend=False
    )
    fig.write_html(output_file)
    print(f"Cross-state visualization saved to {output_file}")

def create_cross_state_heatmap(pairs, output_file):
    # Build a DataFrame for easier processing
    df = pd.DataFrame([
        {
            'state1': p['state1'],
            'state2': p['state2'],
            'coverage': max(p['metrics']['coverage_text1'], p['metrics']['coverage_text2'])
        }
        for p in pairs
    ])
    if df.empty:
        print("No cross-state matches found for heatmap.")
        return
    # Group by state pairs and take max coverage
    df['pair'] = df.apply(lambda r: f"{r['state1']} - {r['state2']}", axis=1)
    heatmap_df = df.groupby('pair')['coverage'].max().reset_index()
    # Split pairs for axes
    heatmap_df[['state1', 'state2']] = heatmap_df['pair'].str.split(' - ', expand=True)
    pivot = heatmap_df.pivot(index='state1', columns='state2', values='coverage').fillna(0)
    fig = px.imshow(pivot, text_auto=True, color_continuous_scale='Blues',
                    title="Maximum Cross-State Text Reuse Coverage",
                    labels=dict(x="State 2", y="State 1", color="Max Coverage"))
    fig.write_html(output_file)
    print(f"Cross-state heatmap saved to {output_file}")

def main():
    results = load_detailed_results('text_reuse_detailed_analysis.json')
    cross_state_pairs = filter_cross_state_pairs(results)
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    create_cross_state_network(cross_state_pairs, output_dir / 'cross_state_network.html')
    create_cross_state_heatmap(cross_state_pairs, output_dir / 'cross_state_heatmap.html')

if __name__ == "__main__":
    main()

import json
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import plotly.express as px

def load_analysis_results(file_path: str) -> dict:
    """Load the text reuse analysis results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def detect_analysis_type(results: dict) -> str:
    """Detect whether this is enhanced (k-shingles) or basic analysis."""
    if 'similarity_scores' in results and len(results['similarity_scores']) > 0:
        sample = results['similarity_scores'][0]
        if 'combined_similarity' in sample:
            return 'enhanced'
    
    if 'document_pairs' in results and len(results['document_pairs']) > 0:
        return 'basic'
    
    return 'unknown'

def get_similarity_score(pair: dict, similarity_type: str = 'auto') -> float:
    """Get similarity score based on analysis type."""
    if similarity_type == 'auto':
        # Auto-detect best similarity score
        if 'combined_similarity' in pair:
            return pair['combined_similarity']
        elif 'similarity_score' in pair:
            return pair['similarity_score']
        else:
            return 0.0
    
    return pair.get(similarity_type, 0.0)

def create_similarity_network(results: dict, min_similarity: float = 0.1, similarity_type: str = 'auto') -> nx.Graph:
    """Create a network graph from the similarity data."""
    G = nx.Graph()
    
    # Determine data source based on analysis type
    analysis_type = detect_analysis_type(results)
    
    if analysis_type == 'enhanced':
        pairs = results['similarity_scores']
    else:
        pairs = results.get('document_pairs', [])
    
    # Add nodes and edges
    for pair in pairs:
        similarity = get_similarity_score(pair, similarity_type)
        
        if similarity >= min_similarity:
            # Add nodes with state information
            G.add_node(pair['document1'], state=pair['state1'])
            G.add_node(pair['document2'], state=pair['state2'])
            
            # Add edge with similarity score and detailed attributes
            edge_attrs = {'weight': similarity, 'similarity_type': similarity_type}
            
            # Add enhanced similarity data if available
            if 'ngram_similarity' in pair:
                edge_attrs.update({
                    'ngram_similarity': pair['ngram_similarity'],
                    'char5_similarity': pair['char5_similarity'],
                    'char10_similarity': pair['char10_similarity'],
                    'word3_similarity': pair['word3_similarity'],
                    'combined_similarity': pair['combined_similarity']
                })
            
            G.add_edge(pair['document1'], pair['document2'], **edge_attrs)
    
    return G

def create_network_visualization(G: nx.Graph, output_file: str):
    """Create an interactive network visualization using plotly."""
    # Use spring layout for node positions
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    edge_widths = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        edge_data = G.edges[edge]
        similarity = edge_data['weight']
        
        # Create detailed hover text based on available data
        if 'ngram_similarity' in edge_data:
            hover_text = (
                f"<b>Similarity Breakdown:</b><br>"
                f"Combined: {edge_data.get('combined_similarity', 0):.4f}<br>"
                f"N-gram (5): {edge_data.get('ngram_similarity', 0):.4f}<br>"
                f"Char-5: {edge_data.get('char5_similarity', 0):.4f}<br>"
                f"Char-10: {edge_data.get('char10_similarity', 0):.4f}<br>"
                f"Word-3: {edge_data.get('word3_similarity', 0):.4f}<br>"
                f"Documents: {edge[0]} ↔ {edge[1]}"
            )
        else:
            hover_text = f"Similarity: {similarity:.4f}<br>Documents: {edge[0]} ↔ {edge[1]}"
        
        edge_text.append(hover_text)
        # Scale edge width based on similarity (1-5 range)
        edge_widths.append(1 + 4 * similarity)
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='rgba(125,125,125,0.5)'),
        hoverinfo='text',
        text=edge_text,
        mode='lines')
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    # Create color mapping for states
    states = set(nx.get_node_attributes(G, 'state').values())
    color_map = {state: i for i, state in enumerate(states)}
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        state = G.nodes[node]['state']
        node_text.append(f"Document: {node}<br>State: {state.upper()}")
        node_color.append(color_map[state])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=10,
            colorbar=dict(
                title='State',
                ticktext=list(states),
                tickvals=list(range(len(states)))
            )
        ))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Legal Code Text Reuse Network (Enhanced with K-Shingles)',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[
                           dict(
                               text="Hover over edges for detailed similarity breakdown<br>Node colors represent different states",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
                               font=dict(size=10, color='gray')
                           )
                       ],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    # Save to HTML
    fig.write_html(output_file)

def create_heatmap(results: dict, output_file: str, similarity_type: str = 'auto'):
    """Create a heatmap of document similarities."""
    # Determine data source based on analysis type
    analysis_type = detect_analysis_type(results)
    
    if analysis_type == 'enhanced':
        pairs = results['similarity_scores']
    else:
        pairs = results.get('document_pairs', [])
    
    # Create a dictionary to store similarities
    similarities = {}
    documents = set()
    
    # Collect all documents and their similarities
    for pair in pairs:
        documents.add(pair['document1'])
        documents.add(pair['document2'])
        similarity = get_similarity_score(pair, similarity_type)
        similarities[(pair['document1'], pair['document2'])] = similarity
        similarities[(pair['document2'], pair['document1'])] = similarity
    
    # Convert to list for ordered processing
    documents = sorted(list(documents))
    
    # Create similarity matrix
    matrix = []
    for doc1 in documents:
        row = []
        for doc2 in documents:
            if doc1 == doc2:
                row.append(1.0)  # Documents are identical to themselves
            else:
                row.append(similarities.get((doc1, doc2), 0.0))
        matrix.append(row)
    
    # Create heatmap
    similarity_name = similarity_type if similarity_type != 'auto' else 'Combined'
    fig = px.imshow(matrix,
                    labels=dict(x="Document", y="Document", color="Similarity"),
                    x=documents,
                    y=documents,
                    title=f"Document Similarity Heatmap ({similarity_name})",
                    color_continuous_scale="Viridis",
                    aspect="auto")
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_tickangle=-45,
        width=max(1200, len(documents) * 15),
        height=max(1200, len(documents) * 15),
        font=dict(size=10)
    )
    
    # Save to HTML
    fig.write_html(output_file)

def create_similarity_comparison_plot(results: dict, output_file: str):
    """Create a comparison plot of different similarity measures."""
    analysis_type = detect_analysis_type(results)
    
    if analysis_type != 'enhanced':
        print("Similarity comparison plot requires enhanced analysis results.")
        return
    
    pairs = results['similarity_scores']
    
    # Prepare data for plotting
    similarity_types = ['ngram_similarity', 'char5_similarity', 'char10_similarity', 
                       'word3_similarity', 'combined_similarity']
    
    plot_data = []
    for pair in pairs[:50]:  # Limit to top 50 for readability
        doc_pair = f"{pair['document1'][:20]}...{pair['document2'][:20]}"
        for sim_type in similarity_types:
            if sim_type in pair:
                plot_data.append({
                    'Document Pair': doc_pair,
                    'Similarity Type': sim_type.replace('_', ' ').title(),
                    'Similarity Score': pair[sim_type],
                    'Cross State': pair['state1'] != pair['state2']
                })
    
    df = pd.DataFrame(plot_data)
    
    # Create grouped bar chart
    fig = px.bar(df, 
                 x='Document Pair', 
                 y='Similarity Score',
                 color='Similarity Type',
                 title='Comparison of Different Similarity Measures (Top 50 Pairs)',
                 barmode='group')
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        width=1400,
        showlegend=True
    )
    
    fig.write_html(output_file)

def main():
    # Try enhanced analysis results first, then fall back to basic
    enhanced_file = "text_reuse_analysis_5gram_shingles.json"
    basic_file = "text_reuse_analysis.json"
    
    results_file = enhanced_file
    try:
        results = load_analysis_results(enhanced_file)
        print(f"Loaded enhanced analysis results from {enhanced_file}")
    except FileNotFoundError:
        try:
            results = load_analysis_results(basic_file)
            results_file = basic_file
            print(f"Loaded basic analysis results from {basic_file}")
        except FileNotFoundError:
            print(f"Neither {enhanced_file} nor {basic_file} found!")
            return
    
    analysis_type = detect_analysis_type(results)
    print(f"Detected analysis type: {analysis_type}")
    
    # Create output directory if it doesn't exist
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Create and save network visualization
    min_similarity = 0.01 if analysis_type == 'enhanced' else 0.1
    G = create_similarity_network(results, min_similarity=min_similarity)
    create_network_visualization(G, output_dir / "text_reuse_network_enhanced.html")
    
    # Create and save heatmap visualization
    create_heatmap(results, output_dir / "similarity_heatmap_enhanced.html")
    
    print("Enhanced visualizations have been created in the 'visualizations' directory:")
    print("1. text_reuse_network_enhanced.html - Interactive network visualization with k-shingles")
    print("2. similarity_heatmap_enhanced.html - Document similarity heatmap")
    
    # Create similarity comparison plot if enhanced data is available
    if analysis_type == 'enhanced':
        create_similarity_comparison_plot(results, output_dir / "similarity_comparison.html")
        print("3. similarity_comparison.html - Comparison of different similarity measures")
        
        # Create individual similarity type heatmaps
        similarity_types = ['ngram_similarity', 'char5_similarity', 'char10_similarity', 
                           'word3_similarity', 'combined_similarity']
        
        for sim_type in similarity_types:
            filename = f"heatmap_{sim_type}.html"
            create_heatmap(results, output_dir / filename, similarity_type=sim_type)
            print(f"4. {filename} - {sim_type.replace('_', ' ').title()} heatmap")
    
    print(f"\nAnalyzed {len(G.nodes())} documents with {len(G.edges())} similarity connections")

if __name__ == "__main__":
    main()
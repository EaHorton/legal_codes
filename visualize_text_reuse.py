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

def create_similarity_network(results: dict, min_similarity: float = 0.1) -> nx.Graph:
    """Create a network graph from the similarity data."""
    G = nx.Graph()
    
    # Add nodes and edges
    for pair in results['document_pairs']:
        if pair['similarity_score'] >= min_similarity:
            # Add nodes with state information
            G.add_node(pair['document1'], state=pair['state1'])
            G.add_node(pair['document2'], state=pair['state2'])
            
            # Add edge with similarity score
            G.add_edge(pair['document1'], pair['document2'], 
                      weight=pair['similarity_score'])
    
    return G

def create_network_visualization(G: nx.Graph, output_file: str):
    """Create an interactive network visualization using plotly."""
    # Use spring layout for node positions
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        similarity = G.edges[edge]['weight']
        edge_text.append(f"Similarity: {similarity:.3f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
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
                       title='Legal Code Text Reuse Network',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    # Save to HTML
    fig.write_html(output_file)

def create_heatmap(results: dict, output_file: str):
    """Create a heatmap of document similarities."""
    # Create a dictionary to store similarities
    similarities = {}
    documents = set()
    
    # Collect all documents and their similarities
    for pair in results['document_pairs']:
        documents.add(pair['document1'])
        documents.add(pair['document2'])
        similarities[(pair['document1'], pair['document2'])] = pair['similarity_score']
        similarities[(pair['document2'], pair['document1'])] = pair['similarity_score']
    
    # Convert to list for ordered processing
    documents = list(documents)
    
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
    fig = px.imshow(matrix,
                    labels=dict(x="Document", y="Document", color="Similarity"),
                    x=documents,
                    y=documents,
                    title="Document Similarity Heatmap",
                    color_continuous_scale="Viridis")
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_tickangle=-45,
        width=1200,
        height=1200
    )
    
    # Save to HTML
    fig.write_html(output_file)

def main():
    # Load analysis results
    results_file = "text_reuse_analysis.json"
    results = load_analysis_results(results_file)
    
    # Create output directory if it doesn't exist
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Create and save network visualization
    G = create_similarity_network(results)
    create_network_visualization(G, output_dir / "text_reuse_network.html")
    
    # Create and save heatmap visualization
    create_heatmap(results, output_dir / "similarity_heatmap.html")
    
    print("Visualizations have been created in the 'visualizations' directory:")
    print("1. text_reuse_network.html - Interactive network visualization")
    print("2. similarity_heatmap.html - Document similarity heatmap")

if __name__ == "__main__":
    main()
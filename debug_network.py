#!/usr/bin/env python3
"""
Debug and fix the state network visualization
"""

import json
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict

def create_debug_network():
    """Create a debug version of the state network with enhanced visibility"""
    
    # Load the data
    with open('text_reuse_analysis_5gram.json', 'r') as f:
        data = json.load(f)
    
    similarity_data = data['similarity_scores']
    
    # Group by state pairs
    state_pairs = defaultdict(list)
    for pair in similarity_data:
        if pair['similarity_score'] >= 0.005:  # Use low threshold
            state_pair = tuple(sorted([pair['state1'], pair['state2']]))
            state_pairs[state_pair].append(pair['similarity_score'])
    
    # Create networkx graph
    G = nx.Graph()
    
    # Add all states as nodes
    states = ['al', 'nc', 'tn']
    for state in states:
        G.add_node(state)
    
    # Add edges with detailed information
    print("State pair connections:")
    for state_pair, similarities in state_pairs.items():
        if len(similarities) > 0:
            avg_sim = np.mean(similarities)
            max_sim = np.max(similarities)
            count = len(similarities)
            
            print(f"  {state_pair[0].upper()} - {state_pair[1].upper()}: {count} connections, avg={avg_sim:.6f}, max={max_sim:.6f}")
            
            # Add edge to graph
            G.add_edge(state_pair[0], state_pair[1], 
                      weight=avg_sim * 1000,  # Scale for visibility
                      avg_similarity=avg_sim,
                      max_similarity=max_sim,
                      num_connections=count)
    
    print(f"\nGraph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Create visualization with fixed positions for clarity
    pos = {
        'al': (-1, -1),
        'nc': (1, 0),
        'tn': (0, 1)
    }
    
    # Prepare edge traces
    edge_traces = []
    
    # Create edges with varying thickness
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_data = G.edges[edge]
        avg_sim = edge_data['avg_similarity']
        max_sim = edge_data['max_similarity']
        connections = edge_data['num_connections']
        
        # Calculate edge width based on number of connections
        edge_width = max(3, connections / 20)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=edge_width, color='red'),
            hoverinfo='none',
            showlegend=False,
            name=f'{edge[0].upper()}-{edge[1].upper()}'
        )
        edge_traces.append(edge_trace)
        
        # Add edge label at midpoint
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        edge_label = go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode='markers+text',
            text=[f'{connections}'],
            textposition='middle center',
            marker=dict(size=20, color='yellow', line=dict(width=2, color='red')),
            hovertext=f'{edge[0].upper()} ↔ {edge[1].upper()}<br>'
                     f'Connections: {connections}<br>'
                     f'Avg Similarity: {avg_sim:.6f}<br>'
                     f'Max Similarity: {max_sim:.6f}',
            hoverinfo='text',
            showlegend=False,
            name=f'{edge[0].upper()}-{edge[1].upper()} details'
        )
        edge_traces.append(edge_label)
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_info = []
    node_sizes = []
    
    color_map = {'al': '#FF6B6B', 'nc': '#4ECDC4', 'tn': '#45B7D1'}
    
    for node in states:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node.upper())
        node_colors.append(color_map[node])
        
        # Calculate node statistics
        connections = list(G.neighbors(node))
        if connections:
            total_connections = sum(G.edges[node, neighbor]['num_connections'] for neighbor in connections)
            node_size = max(80, 80 + len(connections) * 30)
            
            connected_info = []
            for neighbor in connections:
                edge_data = G.edges[node, neighbor]
                connected_info.append(f"{neighbor.upper()}: {edge_data['num_connections']} docs")
            
            node_info.append(
                f"State: {node.upper()}<br>"
                f"Cross-state connections: {len(connections)}<br>"
                f"Total document pairs: {total_connections}<br>"
                f"Connected to:<br>" + "<br>".join(connected_info)
            )
        else:
            node_size = 80
            node_info.append(f"State: {node.upper()}<br>No cross-state connections found")
        
        node_sizes.append(node_size)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='middle center',
        textfont=dict(size=18, color='white', family='Arial Black'),
        hovertext=node_info,
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=3, color='white')
        ),
        showlegend=False,
        name='States'
    )
    
    # Create figure
    all_traces = edge_traces + [node_trace]
    
    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=dict(
            text='Legal Code Borrowing Network Between States (Enhanced)<br>'
                 '<sub>Red lines show connections, numbers show document count, larger nodes = more connections</sub>',
            font=dict(size=18),
            x=0.5
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=80),
        annotations=[
            dict(
                text="Each number on a red line shows how many document pairs share similar text between those states",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                xanchor='center', yanchor='top',
                font=dict(size=12, color="gray")
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Save the enhanced version
    fig.write_html('visualizations/state_network_enhanced.html')
    print("Enhanced state network saved to visualizations/state_network_enhanced.html")
    
    return fig

if __name__ == "__main__":
    create_debug_network()
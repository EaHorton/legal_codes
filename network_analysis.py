#!/usr/bin/env python3
"""
Network Analysis for Legal Code Text Reuse
This script creates network visualizations to show borrowing patterns between states
based on n-gram and Jaccard similarity analysis results.
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

class LegalCodeNetworkAnalysis:
    def __init__(self):
        self.similarity_data = []
        self.state_pairs = defaultdict(list)
        self.cross_state_pairs = []
        self.within_state_pairs = []
        
    def load_similarity_data(self, file_path):
        """Load similarity data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if 'similarity_scores' in data:
                self.similarity_data = data['similarity_scores']
            elif 'document_pairs' in data:
                self.similarity_data = data['document_pairs']
            else:
                print(f"Unknown JSON structure in {file_path}")
                return False
                
            print(f"Loaded {len(self.similarity_data)} similarity pairs from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return False
    
    def preprocess_data(self, min_similarity=0.01):
        """Preprocess similarity data and separate cross-state vs within-state pairs"""
        for item in self.similarity_data:
            if item['similarity_score'] >= min_similarity:
                state1, state2 = item['state1'], item['state2']
                
                # Add to state pairs dictionary
                state_pair = tuple(sorted([state1, state2]))
                self.state_pairs[state_pair].append(item['similarity_score'])
                
                if state1 != state2:
                    self.cross_state_pairs.append(item)
                else:
                    self.within_state_pairs.append(item)
        
        print(f"Found {len(self.cross_state_pairs)} cross-state pairs")
        print(f"Found {len(self.within_state_pairs)} within-state pairs")
    
    def create_state_network(self, similarity_threshold=0.008945):
        """Create a network graph of states based on text similarity"""
        G = nx.Graph()
        
        # Calculate average similarity between each pair of states
        state_similarities = {}
        for state_pair, similarities in self.state_pairs.items():
            if len(similarities) > 0:
                avg_similarity = np.mean(similarities)
                max_similarity = np.max(similarities)
                
                state_similarities[state_pair] = {
                    'avg_similarity': avg_similarity,
                    'max_similarity': max_similarity,
                    'num_connections': len(similarities),
                    'total_similarity': np.sum(similarities)
                }
        
        # Add all states as nodes first
        all_states = {'al', 'nc', 'tn'}  # Ensure we have all three states
        for state in all_states:
            G.add_node(state, label=state.upper())
        
        print(f"State similarities found: {len(state_similarities)}")
        for state_pair, metrics in state_similarities.items():
            print(f"  {state_pair[0].upper()}-{state_pair[1].upper()}: avg={metrics['avg_similarity']:.6f}, max={metrics['max_similarity']:.6f}, count={metrics['num_connections']}")
        
        # Add edges between states with similarity above threshold
        edges_added = 0
        for state_pair, metrics in state_similarities.items():
            if metrics['avg_similarity'] >= similarity_threshold:
                state1, state2 = state_pair
                # Scale the weight to make it more visible (multiply by 100)
                edge_weight = metrics['avg_similarity'] * 100
                G.add_edge(state1, state2, 
                          weight=edge_weight,
                          avg_similarity=metrics['avg_similarity'],
                          max_similarity=metrics['max_similarity'],
                          num_connections=metrics['num_connections'],
                          total_similarity=metrics['total_similarity'])
                edges_added += 1
        
        print(f"Added {edges_added} edges with threshold >= {similarity_threshold}")
        
        return G, state_similarities
    
    def create_document_network(self, similarity_threshold=0.1, max_documents=100):
        """Create a network of documents showing text reuse patterns"""
        G = nx.Graph()
        
        # Filter to most similar pairs and limit for visualization
        filtered_pairs = [pair for pair in self.similarity_data 
                         if pair['similarity_score'] >= similarity_threshold]
        
        # Sort by similarity and take top pairs
        filtered_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
        filtered_pairs = filtered_pairs[:max_documents]
        
        # Add nodes and edges
        for pair in filtered_pairs:
            doc1, doc2 = pair['document1'], pair['document2']
            state1, state2 = pair['state1'], pair['state2']
            similarity = pair['similarity_score']
            
            # Add nodes with state information
            G.add_node(doc1, state=state1, label=doc1[:50] + "...")
            G.add_node(doc2, state=state2, label=doc2[:50] + "...")
            
            # Add edge
            G.add_edge(doc1, doc2, weight=similarity, 
                      cross_state=(state1 != state2))
        
        return G
    
    def visualize_state_network(self, G, state_similarities, output_file="state_network.html"):
        """Create interactive visualization of state network"""
        print(f"Creating visualization for graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        if G.number_of_edges() == 0:
            print("Warning: No edges in graph. Using fixed positions.")
            # Create fixed positions for states if no edges exist
            pos = {'al': (-1, -0.5), 'nc': (1, 0.5), 'tn': (0, 0)}
        else:
            pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare edge traces with variable width based on similarity
        edge_x, edge_y = [], []
        edge_info = []
        edge_widths = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_data = G.edges[edge]
            # Use avg_similarity if available, otherwise use weight
            similarity = edge_data.get('avg_similarity', edge_data.get('weight', 0))
            max_sim = edge_data.get('max_similarity', similarity)
            connections = edge_data.get('num_connections', 1)
            
            edge_info.append(f"{edge[0].upper()} - {edge[1].upper()}: "
                           f"Avg Similarity: {similarity:.6f}, "
                           f"Max Similarity: {max_sim:.6f}, "
                           f"Document Pairs: {connections}")
            
            # Make edge width proportional to similarity (scale up for visibility)
            width = max(2, similarity * 1000)  # Scale up and ensure minimum width
            edge_widths.extend([width, width, width])  # For x0, x1, None pattern
        
        # Create multiple edge traces if we have edges
        edge_traces = []
        if G.number_of_edges() > 0:
            # Main edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=3, color='rgba(255,100,100,0.8)'),  # More visible red color
                hoverinfo='none',
                mode='lines',
                name='Connections'
            )
            edge_traces.append(edge_trace)
        
        # Add invisible edge trace for hover info
        if edge_info:
            edge_hover_x = []
            edge_hover_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                # Add midpoint for hover
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                edge_hover_x.append(mid_x)
                edge_hover_y.append(mid_y)
            
            edge_hover_trace = go.Scatter(
                x=edge_hover_x, y=edge_hover_y,
                mode='markers',
                marker=dict(size=15, color='rgba(255,100,100,0.3)', 
                          line=dict(width=2, color='red')),
                hoverinfo='text',
                hovertext=edge_info,
                name='Connection Details'
            )
            edge_traces.append(edge_hover_trace)
        
        # Node traces
        node_x, node_y = [], []
        node_text, node_info = [], []
        node_colors = []
        node_sizes = []
        
        color_map = {'al': '#FF6B6B', 'nc': '#4ECDC4', 'tn': '#45B7D1'}
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node.upper())
            node_colors.append(color_map.get(node, '#95A5A6'))
            
            # Calculate node statistics
            connections = list(G.neighbors(node))
            if connections:
                # Use avg_similarity if available for total calculation
                total_similarity = 0
                for neighbor in connections:
                    edge_data = G.edges[node, neighbor]
                    similarity = edge_data.get('avg_similarity', edge_data.get('weight', 0))
                    total_similarity += similarity
                
                # Make node size proportional to connections (minimum size 40)
                node_size = max(40, 40 + len(connections) * 20)
            else:
                total_similarity = 0
                node_size = 40
            
            node_sizes.append(node_size)
            
            if connections:
                connected_states = ', '.join([n.upper() for n in connections])
                node_info.append(f"State: {node.upper()}<br>"
                               f"Connections: {len(connections)}<br>"
                               f"Total Similarity: {total_similarity:.6f}<br>"
                               f"Connected to: {connected_states}")
            else:
                node_info.append(f"State: {node.upper()}<br>"
                               f"Connections: 0<br>"
                               f"No cross-state similarities found")
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                              mode='markers+text',
                              hoverinfo='text',
                              text=node_text,
                              textposition='middle center',
                              textfont=dict(size=16, color='white'),
                              hovertext=node_info,
                              marker=dict(size=node_sizes,
                                        color=node_colors,
                                        line=dict(width=3, color='white')),
                              name='States')
        
        # Create figure with all traces
        all_traces = edge_traces + [node_trace]
        fig = go.Figure(data=all_traces)
        fig.update_layout(
            title=dict(
                text='Legal Code Borrowing Network Between States<br>'
                     '<sub>Red lines show connections, node size represents connection strength</sub>',
                font=dict(size=16),
                x=0.5
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=80),
            annotations=[
                dict(
                    text="Network showing text reuse patterns between states based on n-gram analysis",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.1,
                    xanchor='center', yanchor='top',
                    font=dict(color="gray", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Save to file
        fig.write_html(output_file)
        print(f"State network visualization saved to {output_file}")
        
        return fig
    
    def create_similarity_heatmap(self, output_file="similarity_heatmap.html"):
        """Create a heatmap showing average similarity between states"""
        # Calculate average similarities between all state pairs
        states = ['al', 'nc', 'tn']
        similarity_matrix = np.zeros((len(states), len(states)))
        
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                if i == j:
                    # Within-state similarity
                    within_state = [pair['similarity_score'] for pair in self.similarity_data 
                                   if pair['state1'] == state1 and pair['state2'] == state2]
                    if within_state:
                        similarity_matrix[i][j] = np.mean(within_state)
                else:
                    # Cross-state similarity
                    cross_state = [pair['similarity_score'] for pair in self.similarity_data 
                                  if (pair['state1'] == state1 and pair['state2'] == state2) or
                                     (pair['state1'] == state2 and pair['state2'] == state1)]
                    if cross_state:
                        similarity_matrix[i][j] = np.mean(cross_state)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=[s.upper() for s in states],
            y=[s.upper() for s in states],
            colorscale='Viridis',
            text=[[f'{val:.4f}' for val in row] for row in similarity_matrix],
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverinfo='x+y+z'
        ))
        
        fig.update_layout(
            title='Average Text Similarity Between States<br>'
                  '<sub>Based on n-gram analysis of legal codes</sub>',
            xaxis_title='State',
            yaxis_title='State',
            width=600,
            height=500
        )
        
        fig.write_html(output_file)
        print(f"Similarity heatmap saved to {output_file}")
        
        return fig, similarity_matrix
    
    def analyze_borrowing_patterns(self):
        """Analyze and report borrowing patterns between states"""
        print("\n=== BORROWING PATTERN ANALYSIS ===")
        
        # Cross-state analysis
        cross_state_stats = defaultdict(list)
        for pair in self.cross_state_pairs:
            state_pair = tuple(sorted([pair['state1'], pair['state2']]))
            cross_state_stats[state_pair].append(pair['similarity_score'])
        
        print("\nCross-State Borrowing Summary:")
        for state_pair, similarities in cross_state_stats.items():
            if similarities:
                print(f"{state_pair[0].upper()} - {state_pair[1].upper()}:")
                print(f"  Number of similar documents: {len(similarities)}")
                print(f"  Average similarity: {np.mean(similarities):.4f}")
                print(f"  Maximum similarity: {np.max(similarities):.4f}")
                print(f"  Minimum similarity: {np.min(similarities):.4f}")
                print()
        
        # Identify highest similarity cross-state pairs
        print("Top 10 Cross-State Document Similarities:")
        cross_state_sorted = sorted(self.cross_state_pairs, 
                                   key=lambda x: x['similarity_score'], reverse=True)
        
        for i, pair in enumerate(cross_state_sorted[:10]):
            print(f"{i+1:2d}. {pair['state1'].upper()} - {pair['state2'].upper()}: "
                  f"{pair['similarity_score']:.4f}")
            print(f"    {pair['document1'][:60]}...")
            print(f"    {pair['document2'][:60]}...")
            print()
    
    def generate_report(self, output_file="network_analysis_report.html"):
        """Generate a comprehensive HTML report"""
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Legal Code Network Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; margin: 20px 0; }}
                .stat {{ display: inline-block; margin: 10px 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Legal Code Text Reuse Network Analysis</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h2>Analysis Summary</h2>
                <div class="stat"><strong>Total Document Pairs:</strong> {len(self.similarity_data)}</div>
                <div class="stat"><strong>Cross-State Pairs:</strong> {len(self.cross_state_pairs)}</div>
                <div class="stat"><strong>Within-State Pairs:</strong> {len(self.within_state_pairs)}</div>
            </div>
            
            <h2>Key Findings</h2>
            <ul>
                <li>Cross-state text reuse indicates potential legal code borrowing between states</li>
                <li>Network analysis reveals which states have the most similar legal language</li>
                <li>High similarity scores suggest direct copying or adaptation of legal text</li>
            </ul>
            
            <h2>Visualizations</h2>
            <p>The following interactive visualizations have been generated:</p>
            <ul>
                <li><a href="state_network.html">State Network Graph</a> - Shows connections between states based on text similarity</li>
                <li><a href="similarity_heatmap.html">Similarity Heatmap</a> - Matrix view of average similarities between states</li>
            </ul>
            
            <h2>Methodology</h2>
            <p>This analysis uses n-gram based text similarity (Jaccard similarity) to identify potential text reuse 
            between legal documents from different states. The network visualization helps identify patterns of 
            borrowing and shared legal language.</p>
            
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(report_html)
        
        print(f"Analysis report saved to {output_file}")

def main():
    # Initialize analyzer
    analyzer = LegalCodeNetworkAnalysis()
    
    # Load data from 5-gram analysis (most comprehensive)
    if analyzer.load_similarity_data('text_reuse_analysis_5gram.json'):
        # Preprocess data with appropriate threshold for legal text analysis
        analyzer.preprocess_data(min_similarity=0.008945)  # 90th percentile - focus on meaningful borrowing
        
        # Create state network with conservative threshold for legal borrowing analysis
        state_graph, state_similarities = analyzer.create_state_network(similarity_threshold=0.008945)
        
        # Create visualizations
        analyzer.visualize_state_network(state_graph, state_similarities, 
                                       "visualizations/state_network.html")
        analyzer.create_similarity_heatmap("visualizations/similarity_heatmap.html")
        
        # Analyze borrowing patterns
        analyzer.analyze_borrowing_patterns()
        
        # Generate report
        analyzer.generate_report("visualizations/network_analysis_report.html")
        
        print("\n=== Network Analysis Complete ===")
        print("Check the 'visualizations' folder for output files:")
        print("- state_network.html: Interactive network of state relationships")
        print("- similarity_heatmap.html: Heatmap of cross-state similarities")
        print("- network_analysis_report.html: Comprehensive analysis report")
    
    else:
        print("Failed to load similarity data. Please check that text_reuse_analysis_5gram.json exists.")

if __name__ == "__main__":
    main()
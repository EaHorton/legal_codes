#!/usr/bin/env python3
"""
Summary of Network Analysis Visualizations Created
"""

import os

def list_visualizations():
    viz_dir = "visualizations"
    
    print("=" * 70)
    print("LEGAL CODE NETWORK ANALYSIS - VISUALIZATION FILES")
    print("=" * 70)
    
    if os.path.exists(viz_dir):
        files = os.listdir(viz_dir)
        html_files = [f for f in files if f.endswith('.html')]
        
        print(f"\n📁 Found {len(html_files)} HTML visualizations in '{viz_dir}/':\n")
        
        file_descriptions = {
            'state_network.html': '🔗 State Network Graph - Interactive network showing connections between states',
            'state_network_enhanced.html': '🔗 Enhanced State Network - Same as above but with improved edge visibility and labels',
            'similarity_heatmap.html': '🔥 Similarity Heatmap - Matrix showing average similarities between state pairs',
            'cross_state_document_network.html': '📄 Document Network - Individual documents connected by similarity',
            'comprehensive_borrowing_report.html': '📊 Comprehensive Report - Full analysis with statistics and insights',
            'network_analysis_report.html': '📋 Network Analysis Report - Basic network statistics and methodology',
            'cross_state_heatmap.html': '🔥 Cross-State Heatmap - Alternative heatmap visualization',
            'cross_state_network.html': '🔗 Cross-State Network - Alternative network layout',
            'text_reuse_report.html': '📄 Text Reuse Report - Original text reuse analysis results',
            'text_reuse_network.html': '🔗 Text Reuse Network - Network from previous analysis',
            'similarity_distribution.html': '📈 Similarity Distribution - Statistical distribution plots',
            'temporal_borrowing_analysis.html': '📅 Temporal Analysis - Time-based borrowing patterns (if dates available)'
        }
        
        # Sort files by importance
        priority_files = [
            'state_network_enhanced.html',
            'state_network.html',
            'comprehensive_borrowing_report.html',
            'similarity_heatmap.html',
            'cross_state_document_network.html'
        ]
        
        other_files = [f for f in html_files if f not in priority_files]
        
        print("🌟 MAIN VISUALIZATIONS (Start here):")
        for i, file in enumerate(priority_files, 1):
            if file in html_files:
                desc = file_descriptions.get(file, 'Visualization file')
                print(f"   {i}. {file}")
                print(f"      {desc}")
                print()
        
        if other_files:
            print("📚 ADDITIONAL VISUALIZATIONS:")
            for file in sorted(other_files):
                if file in file_descriptions:
                    desc = file_descriptions[file]
                    print(f"   • {file}")
                    print(f"     {desc}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Start with 'state_network_enhanced.html' - shows clearest view of state relationships")
        print(f"   2. Read 'comprehensive_borrowing_report.html' for detailed analysis and insights")
        print(f"   3. Use 'similarity_heatmap.html' for quick overview of similarity patterns")
        print(f"   4. Explore 'cross_state_document_network.html' to see specific document connections")
        
        print(f"\n🔍 KEY FINDINGS:")
        print(f"   • Tennessee ↔ North Carolina: 136 document pairs show text similarity")
        print(f"   • Alabama ↔ North Carolina: 2 document pairs show text similarity") 
        print(f"   • No Alabama ↔ Tennessee connections found above threshold")
        print(f"   • This suggests TN-NC had significant legal code borrowing relationship")
        
        print(f"\n📖 HOW TO VIEW:")
        print(f"   • Open any .html file in your web browser")
        print(f"   • Files are interactive - hover over elements for details")
        print(f"   • Use browser zoom and pan to explore visualizations")
        
        print(f"\n" + "=" * 70)
    else:
        print(f"Error: '{viz_dir}' directory not found!")

if __name__ == "__main__":
    list_visualizations()
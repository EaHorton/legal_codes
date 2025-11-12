# Legal Code Text Reuse Analysis Project

This repository contains a comprehensive analysis of text reuse patterns in 19th-century legal codes from Alabama (AL), North Carolina (NC), and Tennessee (TN). The project examines how legal language and concepts were borrowed and adapted between states using advanced text analysis techniques.

## Project Overview

The research investigates legal text borrowing patterns through multiple analytical approaches:
- **OCR Processing**: Converting historical legal documents from images to machine-readable text
- **AI-Enhanced Text Correction**: Using OpenAI GPT models to improve OCR accuracy
- **N-gram Analysis**: Detecting text reuse using various n-gram sizes (3-8 grams)
- **Network Analysis**: Visualizing borrowing patterns between states
- **Clustering Analysis**: Grouping similar legal documents
- **Similarity Threshold Optimization**: Finding optimal parameters for detecting meaningful text reuse

## Repository Structure

### Core Analysis Scripts

#### Text Preprocessing and OCR
- **`convert_pdfs.py`** - Converts PDF legal documents to JPEG images for OCR processing
- **`process_ocr_ai.py`** - Main OCR processing pipeline using Tesseract + OpenAI GPT correction
- **`process_ocr_ai_with_resume.py`** - Enhanced version with resume capability for large datasets
- **`separate_ocr_and_corrected.py`** - Separates raw OCR output from AI-corrected text

#### Text Reuse Analysis
- **`analyze_text_reuse.py`** - Basic n-gram analysis with 4-grams (original implementation)
- **`analyze_text_reuse_detailed.py`** - Enhanced 5-gram analysis with detailed matching passages
- **`analyze_text_reuse_7gram.py`** - Specialized 7-gram analysis for higher precision
- **`analyze_ngram_variations.py`** - Comparative analysis across multiple n-gram sizes (3-8)

#### Threshold Optimization
- **`analyze_5gram_thresholds.py`** - Threshold analysis specifically for 5-gram similarity
- **`analyze_7gram_threshold.py`** - Threshold optimization for 7-gram analysis
- **`analyze_thresholds.py`** - General threshold analysis across different parameters
- **`threshold_summary.py`** - Summarizes threshold analysis results

#### Advanced Analysis
- **`analyze_legal_codes.py`** - Semantic similarity using sentence transformers and clustering
- **`network_analysis.py`** - Network analysis of text reuse patterns between states
- **`enhanced_borrowing_analysis.py`** - Advanced borrowing pattern analysis

#### Visualization
- **`visualize_text_reuse.py`** - Creates network visualizations of text reuse
- **`visualize_cross_state_reuse.py`** - Specialized cross-state borrowing visualizations
- **`list_visualizations.py`** - Lists and describes all generated visualizations

#### Utilities and Debugging
- **`debug_network.py`** - Debugging tools for network analysis
- **`diagnose_7gram.py`** - Diagnostic tools for 7-gram analysis
- **`summarize_network_analysis.py`** - Summarizes network analysis results
- **`copy_jpeg_files.py`**, **`move_jpeg_files.py`** - File management utilities

### Data Structure

```
legal_codes/
├── actual_divorce_codes/           # Clean, processed legal texts by state
│   ├── al_actual_divorce_codes/    # Alabama legal codes (corrected)
│   ├── nc_actual_divorce_codes/    # North Carolina legal codes (corrected)
│   └── tn_actual_divorce_codes/    # Tennessee legal codes (corrected)
├── ocr_ai_results/                 # OCR processing results
│   ├── al_results/                 # Alabama OCR + AI correction results
│   ├── nc_results/                 # North Carolina OCR + AI correction results
│   ├── tn_results/                 # Tennessee OCR + AI correction results
│   ├── separated_ocr_texts/        # Raw OCR output (before AI correction)
│   └── separated_corrected_texts/  # AI-corrected texts
├── visualizations/                 # Generated analysis visualizations
├── divorce_codes_jpg/              # JPEG images from PDF conversion
└── [state]_divorce_codes/          # Original PDF documents by state
```

### Analysis Results

The project generates several JSON files with analysis results:
- **`text_reuse_analysis.json`** - Basic 4-gram analysis results
- **`text_reuse_analysis_5gram.json`** - 5-gram analysis with threshold optimization
- **`text_reuse_analysis_7gram.json`** - High-precision 7-gram analysis
- **`text_reuse_analysis_ngram_comparison.json`** - Comparative n-gram analysis
- **`text_reuse_detailed_analysis.json`** - Detailed analysis with matching passages
- **`clustering_results.json`** - Document clustering results

## Methodology

### 1. Document Preprocessing
1. **PDF to Image Conversion**: Historical legal documents converted to JPEG format
2. **OCR Processing**: Tesseract OCR extracts text from images
3. **AI Correction**: OpenAI GPT models correct OCR errors and improve readability
4. **Text Normalization**: Standardization of legal text format and structure

### 2. Text Reuse Detection
The project employs multiple approaches to detect text reuse:

#### N-gram Analysis
- **3-grams**: Broad similarity detection
- **4-grams**: Balanced precision/recall (original baseline)
- **5-grams**: Optimized for legal text analysis (recommended)
- **6-grams**: Higher precision, lower recall
- **7-grams**: Highest precision for detecting direct copying
- **8-grams**: Experimental ultra-high precision

#### Similarity Metrics
- **Jaccard Similarity**: Primary metric for n-gram overlap
- **Cosine Similarity**: Used in semantic analysis
- **Threshold Optimization**: Statistical analysis to determine optimal similarity thresholds

### 3. Network Analysis
- **State-Level Networks**: Borrowing patterns between AL, NC, and TN
- **Document-Level Networks**: Individual document relationships
- **Cross-State Analysis**: Focus on inter-state text borrowing
- **Temporal Analysis**: Evolution of legal borrowing over time

### 4. Clustering and Classification
- **DBSCAN Clustering**: Groups similar legal documents
- **Sentence Transformers**: Semantic similarity using pre-trained models
- **UMAP Dimensionality Reduction**: Visualization of document relationships

## Key Findings

### Cross-State Text Reuse Patterns
The analysis reveals significant evidence of legal code borrowing between states:
- **High Similarity Scores**: Some document pairs show 90%+ similarity
- **Borrowing Networks**: Clear patterns of legal concept transmission
- **Regional Clusters**: Geographic proximity influences borrowing patterns
- **Temporal Evolution**: Legal borrowing patterns change over time

### Optimal Analysis Parameters
Through systematic threshold analysis:
- **5-grams** provide the best balance for legal text analysis
- **Similarity threshold of 0.008945** (90th percentile) captures meaningful borrowing
- **7-grams** offer higher precision for detecting direct copying
- **Cross-state pairs** show lower but still significant similarity patterns

## Visualizations

The project generates comprehensive HTML visualizations:

### Network Visualizations
- **`state_network.html`** - Interactive network of state-to-state borrowing
- **`cross_state_network.html`** - Focus on inter-state relationships
- **`text_reuse_network.html`** - Document-level borrowing networks

### Statistical Analysis
- **`similarity_heatmap.html`** - Heatmap of similarity scores between states
- **`similarity_distribution.html`** - Distribution analysis of similarity scores
- **`cross_state_heatmap.html`** - Cross-state borrowing intensity

### Comprehensive Reports
- **`network_analysis_report.html`** - Complete network analysis findings
- **`comprehensive_borrowing_report.html`** - Executive summary with key findings
- **`text_reuse_report.html`** - Detailed text reuse analysis with examples

## Usage

### Quick Start
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Download NLTK data**: Run Python and execute `nltk.download('stopwords')` and `nltk.download('punkt')`
3. **Run basic analysis**: `python analyze_text_reuse_detailed.py`
4. **Generate visualizations**: `python network_analysis.py`

### Custom Analysis
```python
# Example: Run 5-gram analysis with custom threshold
from analyze_text_reuse_detailed import analyze_text_reuse

results = analyze_text_reuse("actual_divorce_codes")
# Results saved to text_reuse_detailed_analysis.json
```

### Comparative N-gram Analysis
```python
# Compare different n-gram sizes
python analyze_ngram_variations.py
# Results saved to text_reuse_analysis_ngram_comparison.json
```

### Network Analysis
```python
# Generate complete network analysis
python network_analysis.py
# Creates visualizations in visualizations/ directory
```

## Configuration

### OCR Processing
- Configure OpenAI API key in `.env` file
- Adjust OCR confidence levels in `process_ocr_ai.py`
- Modify prompt templates for AI correction

### Analysis Parameters
- **N-gram size**: Modify `n` parameter in analysis scripts
- **Similarity thresholds**: Adjust based on threshold analysis results
- **Preprocessing**: Customize text cleaning in utility functions

## Research Applications

This methodology can be applied to:
- **Legal History Research**: Understanding legal code evolution
- **Comparative Law Studies**: Cross-jurisdictional legal analysis
- **Digital Humanities**: Large-scale historical text analysis
- **Plagiarism Detection**: Academic and legal document analysis
- **Policy Diffusion Studies**: How policies spread between jurisdictions

## Technical Notes

### Performance Optimization
- **Parallel Processing**: OCR pipeline supports multiprocessing
- **Memory Management**: Large datasets processed in chunks
- **Resume Capability**: Long-running analyses can be resumed
- **Caching**: Intermediate results cached to avoid recomputation

### Accuracy Considerations
- **OCR Quality**: Historical documents may have variable OCR accuracy
- **AI Correction**: GPT models improve accuracy but may introduce subtle changes
- **Threshold Selection**: Critical for meaningful vs. noise detection
- **Manual Validation**: Sample results manually validated for accuracy

## Future Enhancements

- **Temporal Analysis**: Incorporate document dating for chronological analysis
- **Geographic Expansion**: Include additional states and regions
- **Advanced NLP**: Implement transformer-based similarity detection
- **Statistical Testing**: Add significance testing for borrowing patterns
- **Interactive Dashboard**: Web-based interface for exploring results

## Citation

If you use this code or methodology in your research, please cite:
```
[Author]. (2025). Legal Code Text Reuse Analysis: A Computational Approach to 
Historical Legal Borrowing Patterns. [Repository/Publication details]
```

## License

[Add appropriate license information]

## Contact

[Add contact information for questions and collaboration]
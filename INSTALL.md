# Installation and Setup Guide

## Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)
- Tesseract OCR (for OCR processing)

## System Dependencies

### macOS
```bash
# Install Tesseract OCR
brew install tesseract

# Install poppler for PDF processing
brew install poppler
```

### Ubuntu/Debian
```bash
# Install Tesseract OCR
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Install poppler for PDF processing
sudo apt-get install poppler-utils
```

### Windows
1. Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Add Tesseract to your PATH environment variable
3. Install poppler from: https://blog.alivate.com.au/poppler-windows/

## Python Environment Setup

### Option 1: Using pip
```bash
# Clone the repository
git clone [repository-url]
cd legal_codes

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Option 2: Using conda
```bash
# Create conda environment
conda create -n legal_codes python=3.8
conda activate legal_codes

# Install dependencies
pip install -r requirements.txt

# Or install some packages via conda
conda install numpy pandas matplotlib scikit-learn nltk
pip install sentence-transformers plotly networkx umap-learn pytesseract openai

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Configuration

### OpenAI API Setup (for AI text correction)
1. Create a `.env` file in the project root
2. Add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### Tesseract Configuration
If Tesseract is not in your PATH, you may need to specify its location in the OCR scripts:
```python
import pytesseract
# For Windows:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For macOS (if not in PATH):
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
```

## Quick Start Test

Test your installation:
```python
python -c "
import nltk
import pandas as pd
import plotly
import networkx as nx
from sentence_transformers import SentenceTransformer
print('All dependencies installed successfully!')
"
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Ensure Tesseract is installed and in your PATH
2. **OpenAI API errors**: Check your API key and internet connection  
3. **Memory issues**: For large datasets, consider processing in smaller batches
4. **NLTK data missing**: Run the NLTK download commands above
5. **PDF conversion errors**: Ensure poppler is installed for pdf2image

### Performance Tips

1. **Use SSD storage** for faster file I/O with large datasets
2. **Increase memory** if processing many documents simultaneously
3. **Use virtual environments** to avoid dependency conflicts
4. **Monitor API usage** when using OpenAI for text correction

## Verification

Run a simple analysis to verify everything works:
```bash
# Test basic text analysis (if you have data)
python analyze_text_reuse.py

# Test visualization generation
python visualize_text_reuse.py
```

## Next Steps

1. Review the main README.md for detailed usage instructions
2. Examine the sample data structure in `actual_divorce_codes/`
3. Start with basic analysis scripts before advanced features
4. Check the `visualizations/` folder for output examples
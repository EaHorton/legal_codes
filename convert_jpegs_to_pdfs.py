#!/usr/bin/env python3
"""
Convert JPEG files to PDF format
This script converts all JPEG files in al_divorce_codes and tn_divorce_codes 
directories to PDF format and stores them in corresponding PDF directories.
"""

import os
import sys
from PIL import Image
from pathlib import Path

def create_output_directories():
    """Create output directories for PDF files."""
    output_dirs = ["al_divorce_codes_pdfs", "tn_divorce_codes_pdfs"]
    
    for dir_name in output_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")
        else:
            print(f"Directory already exists: {dir_name}")
    
    return output_dirs

def convert_jpeg_to_pdf(jpeg_path, pdf_path):
    """Convert a single JPEG file to PDF."""
    try:
        # Open the JPEG image
        with Image.open(jpeg_path) as image:
            # Convert to RGB if necessary (PDFs work better with RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as PDF
            image.save(pdf_path, "PDF", resolution=100.0)
            
        return True
        
    except Exception as e:
        print(f"Error converting {jpeg_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir):
    """Process all JPEG files in a directory and convert them to PDF."""
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return 0
    
    # Find all JPEG files
    jpeg_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
    jpeg_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.endswith(ext) for ext in jpeg_extensions):
                jpeg_files.append(os.path.join(root, file))
    
    if not jpeg_files:
        print(f"No JPEG files found in {input_dir}")
        return 0
    
    print(f"Found {len(jpeg_files)} JPEG files in {input_dir}")
    
    # Convert each JPEG to PDF
    successful_conversions = 0
    
    for jpeg_file in jpeg_files:
        # Create relative path structure in output directory
        rel_path = os.path.relpath(jpeg_file, input_dir)
        
        # Change extension to .pdf
        pdf_filename = os.path.splitext(rel_path)[0] + '.pdf'
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        # Create subdirectories if they exist in the source
        pdf_dir = os.path.dirname(pdf_path)
        if pdf_dir and not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
        
        # Convert the file
        if convert_jpeg_to_pdf(jpeg_file, pdf_path):
            successful_conversions += 1
            print(f"Converted: {os.path.basename(jpeg_file)} -> {os.path.basename(pdf_path)}")
        
        # Progress indicator
        if successful_conversions % 10 == 0 and successful_conversions > 0:
            print(f"Progress: {successful_conversions}/{len(jpeg_files)} files converted")
    
    print(f"Completed {input_dir}: {successful_conversions}/{len(jpeg_files)} files successfully converted")
    return successful_conversions

def main():
    """Main function to convert JPEG files to PDF."""
    print("JPEG to PDF Converter")
    print("=" * 30)
    
    # Check if PIL is available
    try:
        from PIL import Image
    except ImportError:
        print("Error: PIL (Pillow) is required for this script.")
        print("Install it with: pip install Pillow")
        sys.exit(1)
    
    # Create output directories
    output_dirs = create_output_directories()
    
    # Define input-output directory mappings
    conversions = [
        ("al_divorce_codes", "al_divorce_codes_pdfs"),
        ("tn_divorce_codes", "tn_divorce_codes_pdfs")
    ]
    
    total_converted = 0
    
    # Process each directory
    for input_dir, output_dir in conversions:
        print(f"\nProcessing {input_dir} -> {output_dir}")
        print("-" * 40)
        
        converted = process_directory(input_dir, output_dir)
        total_converted += converted
    
    print(f"\n{'='*50}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*50}")
    print(f"Total files converted: {total_converted}")
    print(f"Output directories:")
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            pdf_count = len([f for f in os.listdir(output_dir) if f.endswith('.pdf')])
            print(f"  {output_dir}: {pdf_count} PDF files")

if __name__ == "__main__":
    main()
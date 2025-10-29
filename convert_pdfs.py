import os
from pdf2image import convert_from_path
from PIL import Image

def create_directory_structure():
    # Create main output directory
    output_dir = "divorce_codes_jpg"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subdirectories for each state
    states = ["al", "nc", "tn"]
    for state in states:
        state_dir = os.path.join(output_dir, f"{state}_divorce_codes_jpg")
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
    
    return output_dir

def convert_pdf_to_jpg(pdf_path, output_dir):
    try:
        # Get the filename without extension and state code
        filename = os.path.basename(pdf_path)
        state_code = pdf_path.split(os.sep)[0][:2]  # Get state code (al, nc, or tn)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Define output path
        output_subdir = os.path.join(output_dir, f"{state_code}_divorce_codes_jpg")
        
        # Convert PDF to images
        pages = convert_from_path(pdf_path)
        
        # If there's only one page, save it directly
        if len(pages) == 1:
            output_path = os.path.join(output_subdir, f"{name_without_ext}.jpg")
            pages[0].save(output_path, 'JPEG')
        else:
            # If there are multiple pages, save them with page numbers
            for i, page in enumerate(pages):
                output_path = os.path.join(output_subdir, f"{name_without_ext}_page_{i+1}.jpg")
                page.save(output_path, 'JPEG')
                
        print(f"Successfully converted {filename}")
        
    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")

def main():
    # Create output directory structure
    output_dir = create_directory_structure()
    
    # Process each state directory
    state_dirs = ["al_divorce_codes", "nc_divorce_codes", "tn_divorce_codes"]
    
    for state_dir in state_dirs:
        if os.path.exists(state_dir):
            # Get all PDF files in the directory
            pdf_files = [f for f in os.listdir(state_dir) if f.lower().endswith('.pdf')]
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(state_dir, pdf_file)
                convert_pdf_to_jpg(pdf_path, output_dir)

if __name__ == "__main__":
    main()
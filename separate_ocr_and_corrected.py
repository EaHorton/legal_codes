import os
import shutil

# Base directory containing the results folders
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_ai_results')

# Create directories for separated files
ocr_output_dir = os.path.join(base_dir, 'separated_ocr_texts')
corrected_output_dir = os.path.join(base_dir, 'separated_corrected_texts')

# Create the output directories if they don't exist
os.makedirs(ocr_output_dir, exist_ok=True)
os.makedirs(corrected_output_dir, exist_ok=True)

# State folders to process
state_folders = ['al_results', 'nc_results', 'tn_results']

def process_state_folder(state_folder):
    state_path = os.path.join(base_dir, state_folder)
    if not os.path.exists(state_path):
        print(f"Warning: {state_folder} does not exist")
        return
    
    # Create state-specific subdirectories in output folders
    state_ocr_dir = os.path.join(ocr_output_dir, state_folder)
    state_corrected_dir = os.path.join(corrected_output_dir, state_folder)
    os.makedirs(state_ocr_dir, exist_ok=True)
    os.makedirs(state_corrected_dir, exist_ok=True)
    
    # Process all files in the state folder
    for filename in os.listdir(state_path):
        if filename.endswith('_ocr.txt'):
            # Copy OCR file to OCR directory
            src = os.path.join(state_path, filename)
            dst = os.path.join(state_ocr_dir, filename)
            shutil.copy2(src, dst)
            print(f"Copied OCR file: {filename}")
            
        elif filename.endswith('_corrected.txt'):
            # Copy corrected file to corrected directory
            src = os.path.join(state_path, filename)
            dst = os.path.join(state_corrected_dir, filename)
            shutil.copy2(src, dst)
            print(f"Copied corrected file: {filename}")

def main():
    print("Starting to separate OCR and corrected texts...")
    
    for state_folder in state_folders:
        print(f"\nProcessing {state_folder}...")
        process_state_folder(state_folder)
    
    print("\nProcess completed!")
    print(f"OCR texts are in: {ocr_output_dir}")
    print(f"Corrected texts are in: {corrected_output_dir}")

if __name__ == "__main__":
    main()
import os
import json
import time
from datetime import datetime
import pytesseract
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import tiktoken

# Load environment variables
load_dotenv()

# Check if API key is available
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables!")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Create a tiktoken encoding for token counting
encoding = tiktoken.encoding_for_model("gpt-4")

class OCRProcessor:
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0
        self.processing_stats = []
        self.processed_files = set()
        
        # Create output directory
        self.output_dir = "ocr_ai_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
            
        # Create subdirectories for each state
        for state in ['al', 'nc', 'tn']:
            state_dir = os.path.join(self.output_dir, f"{state}_results")
            if not os.path.exists(state_dir):
                os.makedirs(state_dir)
                print(f"Created state directory: {state_dir}")

        # Load progress if exists
        self.progress_file = os.path.join(self.output_dir, "progress.json")
        if os.path.exists(self.progress_file):
            print(f"Loading progress from: {self.progress_file}")
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
                self.processed_files = set(progress_data.get('processed_files', []))
                self.total_tokens = progress_data.get('total_tokens', 0)
                self.total_cost = progress_data.get('total_cost', 0)
                self.processing_stats = progress_data.get('processing_stats', [])
            print(f"Loaded {len(self.processed_files)} previously processed files")

    def save_progress(self):
        """Save current progress to a JSON file"""
        progress_data = {
            'processed_files': list(self.processed_files),
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'processing_stats': self.processing_stats,
            'last_update': datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        print(f"Progress saved to: {self.progress_file}")

    def process_image(self, image_path):
        """Process a single image through OCR and AI correction"""
        if image_path in self.processed_files:
            print(f"\nSkipping already processed file: {image_path}")
            return True

        print(f"\nProcessing: {image_path}")
        
        # Extract state code from path
        state_code = os.path.basename(os.path.dirname(image_path)).split('_')[0]
        print(f"State code: {state_code}")
        
        # Perform OCR
        try:
            image = Image.open(image_path)
            print(f"Image opened successfully: {image.size}")
            ocr_text = pytesseract.image_to_string(image)
            print(f"OCR completed. Text length: {len(ocr_text)}")
        except Exception as e:
            print(f"Error performing OCR on {image_path}: {str(e)}")
            return None

        # Save original OCR text
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        ocr_filename = os.path.join(self.output_dir, f"{state_code}_results", f"{base_filename}_ocr.txt")
        with open(ocr_filename, 'w', encoding='utf-8') as f:
            f.write(ocr_text)
        print(f"OCR text saved to: {ocr_filename}")

        # Process with OpenAI with retry logic
        max_retries = 5
        retry_delay = 20  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"\nAttempting OpenAI correction (attempt {attempt + 1}/{max_retries})")
                corrected_text = self.correct_with_openai(ocr_text)
                
                # Save corrected text
                corrected_filename = os.path.join(self.output_dir, f"{state_code}_results", f"{base_filename}_corrected.txt")
                with open(corrected_filename, 'w', encoding='utf-8') as f:
                    f.write(corrected_text)
                print(f"Corrected text saved to: {corrected_filename}")
                
                # Mark file as processed
                self.processed_files.add(image_path)
                self.save_progress()
                
                return True
                
            except Exception as e:
                print(f"\nError details: {str(e)}")
                if "insufficient_quota" in str(e):
                    print(f"\nError: OpenAI API quota exceeded. Please check your billing details.")
                    return None
                elif attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    print(f"\nFailed to process after {max_retries} attempts: {str(e)}")
                    return None

    def correct_with_openai(self, text):
        """Send text to OpenAI for correction"""
        print("Preparing OpenAI request...")
        prompt = """Please correct this historical legal text. Fix OCR errors, punctuation, and formatting while preserving the original meaning and historical context. Rules:
1. Fix obvious OCR errors (like '0' for 'O', '1' for 'l')
2. Add appropriate punctuation and capitalization
3. Fix spacing and line breaks
4. Preserve original meaning and historical context
5. Make best guesses for unclear words based on context
6. Do not add or delete any content unless correcting clear errors
7. Process the ENTIRE text - do not truncate or summarize

Original text:
"""

        messages = [
            {"role": "system", "content": "You are a historical document transcription expert specializing in legal texts."},
            {"role": "user", "content": prompt + text}
        ]

        # Count tokens
        prompt_tokens = len(encoding.encode(prompt + text))
        print(f"Request contains {prompt_tokens} prompt tokens")
        
        # Add delay to respect rate limits
        time.sleep(3)  # Wait 3 seconds between API calls
        
        # Make API call
        print("Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3,
            max_tokens=4000
        )
        print("Received response from OpenAI API")
        
        # Update token counts and costs
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        # Calculate cost (GPT-4 pricing: $0.03/1K prompt tokens, $0.06/1K completion tokens)
        prompt_cost = (prompt_tokens / 1000) * 0.03
        completion_cost = (completion_tokens / 1000) * 0.06
        
        self.total_tokens += total_tokens
        self.total_cost += (prompt_cost + completion_cost)
        
        # Store processing stats
        self.processing_stats.append({
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'cost': prompt_cost + completion_cost
        })

        print(f"Processing complete. Total tokens: {total_tokens}, Cost: ${prompt_cost + completion_cost:.4f}")
        return response.choices[0].message.content

    def save_processing_stats(self):
        """Save processing statistics to a JSON file"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'detailed_stats': self.processing_stats
        }
        
        stats_file = os.path.join(self.output_dir, 'processing_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Processing stats saved to: {stats_file}")

def main():
    # Check Tesseract installation
    try:
        pytesseract.get_tesseract_version()
        print("Tesseract is properly installed")
    except Exception as e:
        print(f"Error: Tesseract is not properly installed: {str(e)}")
        return

    processor = OCRProcessor()
    
    # Get all image files from the jpg directories
    image_files = []
    base_dir = "divorce_codes_jpg"
    
    for state_dir in ['al_divorce_codes_jpg', 'nc_divorce_codes_jpg', 'tn_divorce_codes_jpg']:
        dir_path = os.path.join(base_dir, state_dir)
        if os.path.exists(dir_path):
            print(f"\nScanning directory: {dir_path}")
            for file in os.listdir(dir_path):
                if file.lower().endswith(('.jpg', '.jpeg')):
                    image_files.append(os.path.join(dir_path, file))
                    print(f"Found image: {file}")

    print(f"\nTotal images found: {len(image_files)}")

    # Remove already processed files from the list
    remaining_files = [f for f in image_files if f not in processor.processed_files]
    print(f"Remaining files to process: {len(remaining_files)}")
    
    # Process all images with progress bar
    with tqdm(total=len(remaining_files), desc="Processing Images") as pbar:
        for image_path in remaining_files:
            result = processor.process_image(image_path)
            if result is None:  # If processing failed due to quota
                print("\nStopping due to API quota limit")
                break
            pbar.update(1)

    # Save final statistics
    processor.save_processing_stats()
    
    # Print summary
    print("\nProcessing Complete!")
    print(f"Total Files Processed: {len(processor.processed_files)}")
    print(f"Total Tokens Used: {processor.total_tokens:,}")
    print(f"Total Estimated Cost: ${processor.total_cost:.2f}")
    print(f"Results saved in: {processor.output_dir}")

if __name__ == "__main__":
    main()
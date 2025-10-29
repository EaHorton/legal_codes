import os
import shutil

def move_jpeg_files():
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source directories
    al_source = os.path.join(base_dir, 'al_divorce_codes')
    tn_source = os.path.join(base_dir, 'tn_divorce_codes')
    
    # Destination directories
    al_dest = os.path.join(base_dir, 'divorce_codes_jpg', 'al_divorce_codes_jpg')
    tn_dest = os.path.join(base_dir, 'divorce_codes_jpg', 'tn_divorce_codes_jpg')
    
    # Create destination directories if they don't exist
    os.makedirs(al_dest, exist_ok=True)
    os.makedirs(tn_dest, exist_ok=True)
    
    # Move Alabama files
    print("Moving Alabama files...")
    for filename in os.listdir(al_source):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            src_path = os.path.join(al_source, filename)
            dst_path = os.path.join(al_dest, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {filename}")
    
    # Move Tennessee files
    print("\nMoving Tennessee files...")
    for filename in os.listdir(tn_source):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            src_path = os.path.join(tn_source, filename)
            dst_path = os.path.join(tn_dest, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {filename}")
    
    print("\nFile movement complete!")

if __name__ == "__main__":
    move_jpeg_files()
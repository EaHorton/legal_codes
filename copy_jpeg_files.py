import os
import shutil

def copy_jpeg_files():
    # Create directories if they don't exist
    for state in ['al', 'tn']:
        jpg_dir = f"{state}_divorce_codes_jpg"
        if not os.path.exists(jpg_dir):
            os.makedirs(jpg_dir)
    
    # Copy Alabama files
    al_src_dir = "al_divorce_codes"
    al_dst_dir = "al_divorce_codes_jpg"
    
    if os.path.exists(al_src_dir):
        for file in os.listdir(al_src_dir):
            if file.lower().endswith(('.jpg', '.jpeg')):
                src_path = os.path.join(al_src_dir, file)
                dst_path = os.path.join(al_dst_dir, file)
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {file}")
    
    # Copy Tennessee files
    tn_src_dir = "tn_divorce_codes"
    tn_dst_dir = "tn_divorce_codes_jpg"
    
    if os.path.exists(tn_src_dir):
        for file in os.listdir(tn_src_dir):
            if file.lower().endswith(('.jpg', '.jpeg')):
                src_path = os.path.join(tn_src_dir, file)
                dst_path = os.path.join(tn_dst_dir, file)
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {file}")

if __name__ == "__main__":
    copy_jpeg_files()
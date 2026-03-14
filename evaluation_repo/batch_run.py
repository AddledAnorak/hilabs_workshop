import os
import glob
import subprocess

TEST_DATA_DIR = "../workshop_test_data"
OUTPUT_DIR = "output"

def run_evaluation():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    folders = glob.glob(os.path.join(TEST_DATA_DIR, "*"))
    
    for folder in folders:
        if os.path.isdir(folder):
            json_files = glob.glob(os.path.join(folder, "*.json"))
            for jf in json_files:
                filename = os.path.basename(jf)
                output_file = os.path.join(OUTPUT_DIR, filename)
                print(f"Processing {filename}...")
                subprocess.run(["python3", "test.py", jf, output_file])

if __name__ == "__main__":
    run_evaluation()
    print("Batch processing complete.")

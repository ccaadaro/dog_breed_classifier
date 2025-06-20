import os
import zipfile
import kaggle

DATASET = 'competitions/dog-breed-identification'
OUTPUT_DIR = './data/dog-breed-identification'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading dataset from Kaggle...")
kaggle.api.competition_download_files(DATASET, path=OUTPUT_DIR)

zip_path = os.path.join(OUTPUT_DIR, 'dog-breed-identification.zip')
print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(OUTPUT_DIR)

os.remove(zip_path)
print("Done.")

#!/usr/bin/env python3
"""
Download datasets for BPE tokenization training.
"""
import os
import requests
from pathlib import Path
import argparse
import gzip
import shutil

# Dataset URLs
DATASETS = {
    "tinystories_train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "tinystories_valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "owt_train": "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz",
    "owt_valid": "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz",
}

def download_file(url: str, filepath: str, chunk_size: int = 8192):
    """Download a file from URL with progress indication."""
    print(f"Downloading {url}")
    print(f"Saving to {filepath}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="", flush=True)
    
    print(f"\nCompleted downloading {filepath}")

def extract_gzip(gz_filepath: str, output_filepath: str):
    """Extract a gzip file."""
    print(f"Extracting {gz_filepath} to {output_filepath}")
    with gzip.open(gz_filepath, 'rb') as f_in:
        with open(output_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extraction completed: {output_filepath}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for BPE tokenization")
    parser.add_argument("--data-dir", 
                       default="/Users/ishanpatil1994/stanford-cs336-repos/assignment1-basics/data",
                       help="Directory to save datasets (default: ../data)")
    parser.add_argument("--dataset", 
                       choices=list(DATASETS.keys()) + ["all"],
                       default="all",
                       help="Which dataset to download (default: all)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == "all":
        datasets_to_download = DATASETS
    else:
        datasets_to_download = {args.dataset: DATASETS[args.dataset]}
    
    for name, url in datasets_to_download.items():
        # Extract filename from URL
        filename = url.split("/")[-1]
        filepath = data_dir / filename
        
        if filepath.exists():
            print(f"File {filepath} already exists, skipping...")
            continue
            
        try:
            download_file(url, str(filepath))
            
            # Extract gzip files
            if filepath.suffix == '.gz':
                extracted_filepath = filepath.with_suffix('')
                if not extracted_filepath.exists():
                    extract_gzip(str(filepath), str(extracted_filepath))
                else:
                    print(f"Extracted file {extracted_filepath} already exists, skipping extraction...")
                    
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    
    print("\nDataset download completed!")
    print(f"Files saved to: {data_dir}")

if __name__ == "__main__":
    main()
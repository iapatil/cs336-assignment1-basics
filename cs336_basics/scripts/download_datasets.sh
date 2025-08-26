#!/bin/bash

# Download datasets for BPE tokenization training
# Usage: ./download_datasets.sh [data_directory]

# Default data directory
DATA_DIR="${1:-/Users/ishanpatil1994/stanford-cs336-repos/assignment1-basics/data}"

echo "Creating data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading TinyStories datasets..."
if [ ! -f "TinyStoriesV2-GPT4-train.txt" ]; then
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
else
    echo "TinyStoriesV2-GPT4-train.txt already exists, skipping..."
fi

if [ ! -f "TinyStoriesV2-GPT4-valid.txt" ]; then
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
else
    echo "TinyStoriesV2-GPT4-valid.txt already exists, skipping..."
fi

echo "Downloading OpenWebText datasets..."
if [ ! -f "owt_train.txt" ]; then
    if [ ! -f "owt_train.txt.gz" ]; then
        wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
    fi
    gunzip owt_train.txt.gz
else
    echo "owt_train.txt already exists, skipping..."
fi

if [ ! -f "owt_valid.txt" ]; then
    if [ ! -f "owt_valid.txt.gz" ]; then
        wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
    fi
    gunzip owt_valid.txt.gz
else
    echo "owt_valid.txt already exists, skipping..."
fi

echo "Dataset download completed!"
echo "Files saved to: $DATA_DIR"
ls -la "$DATA_DIR"
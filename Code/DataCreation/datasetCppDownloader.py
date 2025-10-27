"""
=======================================================
Module: datasetCppDownloader

Description:
    This script is specifically designed to download and prepare
    a dataset for the Cpp language from the "HuggingFaceTB/stack-edu"
    collection. Unlike other languages, Cpp required a separate
    handling because the generated files were often either too small
    or too large, making the general pipeline unsuitable.

    The script performs the following steps:
        1. Loads and shuffles the original Cpp dataset.
        2. Filters files by length (≈3000 bytes ±500).
        3. Selects a subset that accumulates to a maximum of 100MB.
        4. Downloads the corresponding file contents from the
           "softwareheritage" S3 bucket (decompressed via gzip).
        5. Stores the final dataset as a CSV in the Dataset folder.

Usage:
    Run this script directly to generate the processed Cpp dataset:
        $ python datasetCppDownloader.py

Output:
    Dataset/Cpp_100MB.csv
=======================================================
"""

import boto3
import gzip
import os
import time
from datasets import load_dataset
from botocore.exceptions import ClientError
from botocore import UNSIGNED
from botocore.config import Config

# ====== Download Contents ====== #
def download_contents(blob_id: str) -> dict[str, object]:
    '''
    Download and decompress the code content associated with a given blob ID from the S3 bucket.

    Input:
    - blob_id (str): The unique identifier of the content to be downloaded (used as the S3 key).

    Return:
    - dict: A dictionary containing:
        - "text" (str): The decompressed source code as a UTF-8 string (empty if not found).
        - "download_success" (bool): True if download was successful, False otherwise.
    '''
    key = f"content/{blob_id}"
    retries = 3
    delay = 5  # seconds between attempts

    for attempt in range(retries):
        try:
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            with gzip.GzipFile(fileobj=obj['Body']) as fin:
                content = fin.read().decode("utf-8", errors="ignore")
            return {"text": content, "download_success": True}

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"[{blob_id}] File not found: {key}")
                return {"text": "", "download_success": False}
            else:
                print(f"[{blob_id}] AWS error (attempt {attempt+1}/{retries}): {e}")
        except Exception as e:
            print(f"[{blob_id}] General error (attempt {attempt+1}/{retries}): {e}")

        time.sleep(delay)

    return {"text": "", "download_success": False}


# ====== Main ====== #
if __name__ == "__main__":
    
    # ----------------------------------------------------------------------
    # Settings: language, limits, and size-target parameters
    # ----------------------------------------------------------------------
    
    language = "Cpp"  # Python, C, Cpp, Java, JavaScript, CSharp
    max_total_bytes = 100 * 1024 * 1024  # 100 MB limit

    # Target average size per file (bytes) and tolerance
    target_size = 3000
    tolerance = 500  # accept files between 2500 and 3500 bytes

    # ----------------------------------------------------------------------
    # Load and shuffle dataset
    # ----------------------------------------------------------------------
    
    ds = load_dataset("HuggingFaceTB/stack-edu", language, split="train", num_proc=1)
    print("Original dataset: ", ds)
    ds_shuffled = ds.shuffle(seed=42)

    # ----------------------------------------------------------------------
    # Filter files by length_bytes within tolerance of target_size
    # ----------------------------------------------------------------------
    
    def is_within_target_size(example):
        return abs(example["length_bytes"] - target_size) <= tolerance

    ds_filtered = ds_shuffled.filter(is_within_target_size)
    print(f"Filtered to files within ±{tolerance} bytes of {target_size}: {len(ds_filtered)} examples")

    # ----------------------------------------------------------------------
    # Select subset accumulating up to max_total_bytes
    # ----------------------------------------------------------------------
    
    selected_indices = []
    total_size = 0
    for i, example in enumerate(ds_filtered):
        size_i = example["length_bytes"]
        if total_size + size_i > max_total_bytes:
            break
        selected_indices.append(i)
        total_size += size_i

    ds_small = ds_filtered.select(selected_indices)
    print("Dataset small version: ", ds_small)
    print(f"Total selected byte size: {total_size}")
    print(f"Number of files: {len(ds_small)}")
    print(f"Average size: {total_size / len(ds_small):.2f} bytes")

    # ----------------------------------------------------------------------
    # Download the selected files from S3
    # ----------------------------------------------------------------------
    
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = "softwareheritage"

    ds_small = ds_small.map(download_contents, input_columns="blob_id", num_proc=1)
    ds_small = ds_small.filter(lambda x: x['download_success'])
    print("Final dataset after download: ", ds_small)

    # ----------------------------------------------------------------------
    # Save to CSV
    # ----------------------------------------------------------------------
    
    os.makedirs("Dataset", exist_ok=True)
    output_path = f"Dataset/{language}_100MB.csv"
    ds_small.to_csv(output_path)
    print(f"Saved CSV to {output_path}")

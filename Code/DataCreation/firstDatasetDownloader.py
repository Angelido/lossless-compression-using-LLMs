"""
=======================================================
Module: firstDatasetDownloader

Description:
    This script was used to create an initial small dataset
    (~10 MB) for testing and experimentation purposes.
    It downloads a random subset of files for a chosen language
    from the "HuggingFaceTB/stack-edu" collection, retrieves the
    corresponding contents from the "softwareheritage" S3 bucket,
    and stores the results in a CSV file.

    In the original workflow, this script was used to download
    datasets separately for each language, and the resulting
    CSV files were later combined into a unified dataset using
    createFirstDataset.py.

Steps performed:
    1. Load the dataset for the chosen language (example: CSharp).
    2. Shuffle the dataset to ensure random sampling.
    3. Select a subset of files until reaching a total of ~10 MB.
    4. Download and decompress the file contents from S3.
    5. Keep only successfully downloaded files.
    6. Save the resulting dataset to a CSV file.

Usage:
    Set the language directly in the load_dataset call and run:

        $ python firstDatasetDownloader.py

Output:
    Dataset/<Language>.csv  (e.g., Dataset/C-Sharp.csv)
=======================================================
"""

import boto3
import gzip
import os
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
    try:
        # Try to get the object from the S3 bucket
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        
        # Decompress the .gz file and decode as UTF-8
        with gzip.GzipFile(fileobj=obj['Body']) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        
        return {"text": content, "download_success": True}
    
    except ClientError as e:
        # If the file is not found in the bucket
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"File not found: {key}")
            return {"text": "", "download_success": False}
        else:
            raise



# ====== Main ====== #
if __name__ == "__main__":
    
    language = "C-Sharp" # Set the language that you want

    # Load the Stack-Edu dataset
    ds = load_dataset("HuggingFaceTB/stack-edu", language, split="train", num_proc=1)
    print("Original dataset: \n", ds)

    # Set a maximum total size limit of 10 MB
    max_total_bytes = 10 * 1024 * 1024  

    # Shuffle the dataset for random sampling
    ds_shuffled = ds.shuffle(seed=42)

    # Select rows until the total size exceeds the limit
    selected_indices = []
    total_size = 0

    for i, example in enumerate(ds_shuffled):
        if total_size + example["length_bytes"] > max_total_bytes:
            break
        selected_indices.append(i)
        total_size += example["length_bytes"]

    # Create a new subset dataset with the selected rows
    ds_small = ds_shuffled.select(selected_indices)

    print("Dataset small version: \n", ds_small)
    print(f"Total selected byte size: {total_size}")

    # Create an unsigned S3 client for public access
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = "softwareheritage"

    # Download code content for each blob_id in the dataset from Software Heritage
    ds_small = ds_small.map(download_contents, input_columns="blob_id", num_proc=1)

    # Keep only successfully downloaded files
    ds_small = ds_small.filter(lambda x: x['download_success'])

    print("Final dataset: \n", ds_small)

    # Save the final dataset to a CSV file
    os.makedirs("Dataset", exist_ok=True)
    ds_small.to_csv(f"Dataset/{language}.csv")

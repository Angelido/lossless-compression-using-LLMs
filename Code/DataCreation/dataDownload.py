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
    
    # Define the number of retries and delay between attempts in seconds
    retries = 3
    delay = 5  

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
    
    # ==========================================================================
    # Download the dataset from Software Heritage using Hugging Face Datasets
    # Use as language: Python, C, Cpp, Java, JavaScript, CSharp
    # ==========================================================================

    language = "Cpp"  # Set the desired programming language here
    
    # Load the Stack-Edu dataset (set the language that you want)
    ds = load_dataset("HuggingFaceTB/stack-edu", language, split="train", num_proc=1)
    print("Original dataset: \n", ds)

    # Set a maximum total size limit of 100 MB
    max_total_bytes = 100 * 1024 * 1024  

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
    ds_small.to_csv(f"Dataset/{language}100MB.csv")

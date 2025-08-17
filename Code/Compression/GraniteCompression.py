import numpy as np
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
import os

# Read also files from the parent folder (utility, dataLoader, computeRank)   
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataLoader import create_chunk_dataloader, preprocess_dataset_fast_old
from computeRank import compute_token_ranks_fast_old
from utility import (
    count_nonpad_tokens_per_row, 
    sort_chunks_by_length, 
    save_info_to_csv,
    compress_and_save
)

# ATTENTION: change binary, compression, e file name ad every running 

# =========================
# Global variables, tokenizer e load dataset
# =========================

# Model name
model_name = "PrunaAI/ibm-granite-granite-3b-code-base-bnb-4bit-smashed"
language = "Python"  
batch_size = 32
max_length = 512

# If zstd is True use level=(3, 12, 22), else use level=(3, 9)
binary = True
use_zstd = True  
compression_level = 3
filename_prefix = "Granite_rank_list"

# Configuration for the quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # Also "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Model tokenizer
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3b-code-base")

# Set the pad token to the end of the sequence
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_TOKEN_ID = tokenizer.pad_token_id  

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    quantization_config=bnb_config
)

# Set the device to cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("device=", device)

# Read dataset and save information
df = pd.read_csv(f"Dataset/{language}100MB.csv")
input_texts = df["text"].tolist()
total_bytes = df["length_bytes"].sum()

# =========================
# Create dataloader e with chuncked lists
# =========================

start_create_dataloader = time.perf_counter()

# Preprocessing and chunking
input_id_list, mapping = preprocess_dataset_fast_old(
    input_texts,
    tokenizer,
    max_length=max_length,
    stride=0,           
)
input_id_list, mapping = sort_chunks_by_length(
    input_id_list,
    mapping,
    pad_token_id=PAD_TOKEN_ID,
    descending=True
)
dataloader = create_chunk_dataloader(
    input_id_list,
    batch_size=batch_size
)

end_create_dataloader = time.perf_counter()
time_dataloader = end_create_dataloader - start_create_dataloader

print("After dataloader")


# =========================
# Compute the ranks and reconstruct lists
# =========================

start_compute_ranks = time.perf_counter()

# Compute the rank list using the DataLoader
rank_list = compute_token_ranks_fast_old(
     dataloader,
     model,
     pad_token_id=PAD_TOKEN_ID,
     device=device
)

end_compute_ranks = time.perf_counter()
time_compute_ranks = end_compute_ranks - start_compute_ranks

print("Finished computing rank list")

start_reconstructing = time.perf_counter()

# Reconstruct the rank list using the mapping
reconstructed_rank_list = [
    [rank for chunk_idx in mapping[row_idx] for rank in rank_list[chunk_idx]]
    for row_idx in range(len(input_texts))
]

end_reconstructing = time.perf_counter()
reconstructing_time = end_reconstructing - start_reconstructing

print("Finished recostruncing rank list")

# =========================
# Compression and save results
# =========================
results_dir = "Results/CompressedFiles"

# Esempio di chiamata:
outfile_path, compressed_size_bytes, compression_time = compress_and_save(
    reconstructed_rank_list,
    results_dir,
    binary=binary,
    use_zstd=use_zstd,
    compression_level=compression_level,
    filename_prefix=filename_prefix
)

# Print final summary
print(f"File saved in: {outfile_path}")
print(f"Compressed file size: {compressed_size_bytes} byte")

# =========================
# Save execution summary to CSV (create or append)
# =========================

total_time = time_dataloader + time_compute_ranks + reconstructing_time + compression_time
info_dir = "Results"
csv_file   = "Compression_info.csv"

# Create a dictionary with the information
row_dict = {
    "model": model_name,
    "batch_size": batch_size,
    "max_length": max_length,
    "language": language,
    "binary": binary,
    "compression": f"zstd{compression_level}" if use_zstd else f"bzip2-{compression_level}",
    "output_file": outfile_path,
    "dataloader_time_s": round(time_dataloader, 4),
    "compute_ranks_time_s": round(time_compute_ranks, 4),
    "reconstructing_time_s": round(reconstructing_time, 4),
    "compression_time_s": round(compression_time, 4),
    "total_time_s": round(total_time, 4),
    "original_size_bytes": total_bytes,
    "compressed_size_bytes": compressed_size_bytes,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

# Save row_dict on the csv
save_info_to_csv(info_dir, csv_file, row_dict)

# Print results on screen
print("=== End-of-execution information ===")
for key, value in row_dict.items():
    print(f"{key:25s}: {value}")
print("=======================================\n")






import numpy as np
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

import zstandard as zstd
import bz2
import pickle
import io

# Read also files from the parent folder (utility, dataLoader, computeRank)   
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataLoader import create_chunk_dataloader, preprocess_dataset_fast_single
from computeRank import compute_entropy, compute_cross_entropy
from utility import (
    count_nonpad_tokens_per_row, 
    sort_chunks_by_length, 
    save_rank_list_to_file,
    save_info_to_csv,
    compress_and_save
)

# ATTENTION: change binary, compression, e file name ad every running 

# =========================
# Global variables, tokenizer e load dataset
# =========================

# Model name
model_name = "microsoft/phi-2"
language = "JavaScript"
cross_entropy = True  
batch_size = 32
max_length = 512

# Model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)

# Set the pad token to the end of the sequence
tokenizer.pad_token = tokenizer.eos_token 
PAD_TOKEN_ID = tokenizer.pad_token_id  

# Set the device to cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("device =", device)

df = pd.read_csv(f"Dataset/{language}100MB.csv")
input_texts = df["text"].tolist()

# =========================
# Create dataloader e with chuncked lists
# =========================

start_create_dataloader = time.perf_counter()

# NEW VERSION
input_id_list, mapping = preprocess_dataset_fast_single(
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

if cross_entropy:
    # Compute the rank list using the DataLoader
    total_bits, total_tokens, timings = compute_cross_entropy(
        dataloader, 
        model, 
        PAD_TOKEN_ID, 
        device, 
        show_progress=True, 
        clamp_min_log=-1e6, 
        debug=True
    )
else:
    total_bits, total_tokens, timings = compute_entropy(
        dataloader, 
        model, 
        PAD_TOKEN_ID, 
        device, 
        show_progress=True, 
        clamp_min_log=-1e6, 
        debug=True
    )

end_compute_ranks = time.perf_counter()
time_compute_ranks = end_compute_ranks - start_compute_ranks

print("Finished computing rank list")

information_to_save = {
    "model_name": model_name,
    "language": language,
    "batch_size": batch_size,
    "max_length": max_length,
    "total_tokens": total_tokens,
    "entropy_bits": total_bits,
    "entropy_bytes": total_bits / 8.0,
    "entropy_mb": (total_bits / 8.0) / (1024 * 1024),
    "bits_per_token": total_bits / total_tokens
}

folder_path = "Results"

if cross_entropy:
    csv_name = "Cross_Entropy_info.csv"
else:
    csv_name = "Entropy_info.csv"

save_info_to_csv(
    folder_path,
    csv_name,
    information_to_save
)

print("========== Information of end execution ==========")
for key, value in information_to_save.items():
    print(f"{key}: {value}")
print("=================================================")
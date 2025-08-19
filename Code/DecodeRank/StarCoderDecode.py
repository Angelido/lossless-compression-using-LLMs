import numpy as np
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

# Read also files from the parent folder (utility, dataLoader, computeRank)   
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataLoader import (
    create_chunk_dataloader, 
    preprocess_dataset_fast,
    get_token_info   
)
from computeRank import (
    compute_token_ranks_fast, 
    decode_token_ids_from_ranks
)
from utility import (
    count_nonpad_tokens_per_row, 
    check_bos_token_in_chunks,
    sort_chunks_by_length, 
    verify_reconstruction
)

# =========================
# Global variables, tokenizer e load dataset
# =========================

# Model name
model_name = "bigcode/starcoder2-3b"
language = "Python"  
batch_size = 16
max_length = 256

# Model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # Use bfloat16 for computation
    #torch_dtype = torch.float32,
    device_map="auto"
)

PAD_TOKEN_ID = tokenizer.pad_token_id  

# Print information about special token on screen
info=get_token_info(tokenizer)
for key, value in info.items():
    print(f"{key:25s}: {value}")

# Set the device to cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("device=", device)

# Read dataset and save information
df = pd.read_csv(f"Dataset/{language}100MB.csv")
input_texts = df["text"].tolist()

# =========================
# Create dataloader e with chuncked lists
# =========================

start_create_dataloader = time.perf_counter()

input_id_list, mapping = preprocess_dataset_fast(
    input_texts,
    tokenizer,
    max_length=max_length,
    stride=0,           
)

# Check if the tokenization add BOS tokens correctly
check_bos_token_in_chunks(input_id_list, tokenizer)

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
rank_list, mapping_time = compute_token_ranks_fast(
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

print("Reconstruction completed")

# SECOND PART: RECONSTRUCTION

reconstructed_tokens = decode_token_ids_from_ranks(
    rank_sequences=reconstructed_rank_list,
    model=model,
    tokenizer=tokenizer,
    max_length=max_length,
    pad_token_id=tokenizer.pad_token_id,
    device="cuda",
    debug=False
)

print("Verifying reconstructed tokens...")

ok = verify_reconstruction(
    input_texts, 
    reconstructed_tokens, 
    tokenizer)

if ok:
    print("🎉 Tutte le ricostruzioni sono corrette!")
else:
    print("⚠️ Alcune ricostruzioni non corrispondono.")

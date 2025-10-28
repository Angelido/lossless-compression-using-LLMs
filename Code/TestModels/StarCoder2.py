"""
=======================================================
Module: StarCoder2.py

Description:
    This script is part of the first phase of experimentation.
    It computes token rank lists from code samples using
    non-quantized versions of the StarCoder2 model.

    Two configurations are supported:
        - Full precision (float32)
        - Mixed precision (bfloat16)

    The pipeline follows these steps:
        1. Input         (load the dataset of code samples).
        2. Tokenization  (convert code into token IDs).
        3. Context       (preprocess, chunk, and build a DataLoader).
        4. ComputeRanks  (use the selected StarCoder2 configuration
                         to compute token rank positions).
        5. ListOfRanks   (reconstruct results and save them to file).

Output:
    TextInformation/StarCoder2_float32_rank_list.txt if full precision or
    TextInformation/StarCoder2_bfloat16_rank_list.txt for mixed precision
    (rank lists with execution time and model info)
=======================================================
"""

import pandas as pd
import time
import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Read also files from the parent folder (utility, dataLoader, computeRank)   
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from computeRank import compute_token_ranks_fast_old
from dataLoader import create_chunk_dataloader, preprocess_dataset_fast_old
from utility import ( 
    save_rank_list_to_file,
    count_nonpad_tokens_per_row, 
    sort_chunks_by_length, 
)

# Model name
model_name = "bigcode/starcoder2-3b"

# Model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # Change here the precision 
    device_map="auto" 
)

batch_size = 32
max_length = 512
PAD_TOKEN_ID = tokenizer.pad_token_id  

# Set the device to cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
print("device=", device)

df = pd.read_csv("Dataset/CodeDataset.csv")
input_texts = df["text"].tolist()

# Preprocessing and chunking
input_id_list, mapping = preprocess_dataset_fast_old(
    input_texts,
    tokenizer,
    max_length=max_length,
    stride=0           
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

print("After dataloader")

# # For choose the right max_length
# row_lengths = count_nonpad_tokens_per_row(input_id_list, mapping, PAD_TOKEN_ID)
# # Show the distribution of token lengths
# lengths = list(row_lengths.values())
# for p in [50, 75, 90, 95, 99]:
#     print(f"{p}° percentile token count: {np.percentile(lengths, p):.0f}")

# Timer start
start_time = time.perf_counter()

# Compute the rank list using the DataLoader
rank_list = compute_token_ranks_fast_old(
     dataloader,
     model,
     pad_token_id=PAD_TOKEN_ID,
     device=device
)

# Timer end
end_time = time.perf_counter()
execution_time = end_time - start_time

print("Finished computing rank list")

# Reconstruct the rank list using the mapping
reconstructed_rank_list = [
    [rank for chunk_idx in mapping[row_idx] for rank in rank_list[chunk_idx]]
    for row_idx in range(len(input_texts))
]

print("Reconstructed rank list")

if model.config.torch_dtype == torch.bfloat16:
    precision_str = "bfloat16"
else:
    precision_str = "float32"

# Save the rank list to a file
save_rank_list_to_file(
    rank_list=reconstructed_rank_list,
    file_path=f"TextInformation/StarCoder2_{precision_str}_rank_list.txt",
    execution_time=execution_time,
    model_name=model_name  
)

print("Saved rank list to file")

# Freeing the GPU memory cache after the operation
torch.cuda.empty_cache()
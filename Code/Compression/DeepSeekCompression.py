import numpy as np
import pandas as pd
import time
import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
import sys
import os

import zstandard as zstd
import bz2
import pickle
import io

# Read also files from the parent folder (utility, dataLoader, computeRank)   
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataLoader import create_chunk_dataloader, preprocess_dataset_fast
from computeRank import compute_token_ranks_fast
from utility import (
    count_nonpad_tokens_per_row, 
    sort_chunks_by_length, 
    save_rank_list_to_file,
    save_info_to_csv
)

# ATTENTION: change binary, compression, e file name ad every running 

# =========================
# Global variables, tokenizer e load dataset
# =========================

# Model name
model_name = "TheBloke/deepseek-coder-1.3b-base-AWQ"
language = "Python"  
batch_size = 32
max_length = 512

binary = False
use_zstd = True  
compression_level = 12

# Model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoAWQForCausalLM.from_pretrained(
        model_name,     
        device_map="auto",            # pass the model to the GPU
        torch_dtype=torch.float16,    # mantain the model in float16 where needed
        low_cpu_mem_usage=True,       # use less CPU memory
)

PAD_TOKEN_ID = tokenizer.pad_token_id

# Set the device to cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("device=", device)

df = pd.read_csv(f"Dataset/{language}100MB.csv")
input_texts = df["text"].tolist()
total_bytes = df["length_bytes"].sum()

# =========================
# Create dataloader e with chuncked lists
# =========================

start_create_dataloader = time.perf_counter()

# NEW VERSION
input_id_list, mapping = preprocess_dataset_fast(
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
rank_list = compute_token_ranks_fast(
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
os.makedirs(results_dir, exist_ok=True)

compress_start = time.perf_counter()

if binary:
    # Convert to NumPy arrays and write to .npy buffer
    dtype = np.uint16
    lengths = np.array([len(lst) for lst in reconstructed_rank_list], dtype=np.int32)
    flat_array = np.concatenate(
        [np.array(lst, dtype=dtype) for lst in reconstructed_rank_list]
    )
    buffer = io.BytesIO()
    np.save(buffer, lengths, allow_pickle=False)
    np.save(buffer, flat_array, allow_pickle=False)
    data_blob = buffer.getvalue()

else:
    # Serialize using pickle
    data_blob = pickle.dumps(reconstructed_rank_list)

# Compress with either Zstandard or bz2, based on use_zstd
if use_zstd:
    
    cctx = zstd.ZstdCompressor(level=compression_level)
    compressed_data = cctx.compress(data_blob)
    ext = "zst"
    compressor_name = f"zstd{compression_level}"
else:
    
    compressed_data = bz2.compress(data_blob, compresslevel=compression_level)
    ext = "bz2"
    compressor_name = f"bzip2-{compression_level}"

# Write on the file, using the name based on compressor
if binary:
    outfile_path = os.path.join(
        results_dir,
        f"DeepSeek_rank_list_binary_{compressor_name}.{ext}"
    )
else:
    outfile_path = os.path.join(
        results_dir,
        f"DeepSeek_rank_list_pickle_{compressor_name}.{ext}"
    )

# Write compressed data to disk
with open(outfile_path, "wb") as f_out:
    f_out.write(compressed_data)

compress_end = time.perf_counter()
compression_time = compress_end - compress_start
compressed_size_bytes = len(compressed_data)

# Print final summary
print(f"File salvato in: {outfile_path}")
print(f"Dimensione del file compresso: {compressed_size_bytes} byte")

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
print("=== Informazioni di fine esecuzione ===")
for key, value in row_dict.items():
    print(f"{key:25s}: {value}")
print("=======================================\n")






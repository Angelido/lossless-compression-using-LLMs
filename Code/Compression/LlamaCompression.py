"""
End-to-end pipeline for:
  1) tokenizing + chunking a language dataset,
  2) computing per-token ranks (top-k or full-vocab),
  3) reconstructing per-row rank/exception streams,
  4) compressing the results with multiple codecs, and
  5) logging timing/size summaries to CSV.

The code is intentionally deterministic (CUDA/cuDNN configs and RNG seeds)
to make timing and compression results reproducible across runs/hardware.
"""

import numpy as np
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import random
import os

# Read also files from the parent folder (utility, dataLoader, computeRank)   
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataLoader import (
    create_chunk_dataloader, 
    preprocess_dataset_fast
)
from computeRank import (
    compute_token_ranks_topk_fast, 
    compute_token_ranks_fullrank_fast
)
from utility import (
    count_nonpad_tokens_per_row,  
    sort_chunks_by_length, 
    save_info_to_csv
)
from compressionUtility import (
    compress_and_save,  
    compress_and_save_bz3
)

# =========================
# Reproducibility controls
# =========================

# cuBLAS workspace config + deterministic flags eliminate non-determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Force deterministic algorithms in PyTorch/cuDNN (may reduce peak performance)
torch.use_deterministic_algorithms(True)       
torch.backends.cudnn.deterministic = True      
torch.backends.cudnn.benchmark = False     

# Disable TF32 to avoid tiny numeric differences across GPUs/drivers
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# =========================
#  Global configuration
# =========================

language = "Java"         # dataset language tag (drives the CSV file name)
batch_size = 32           # mini-batch size for rank computation
max_length = 512          # per-chunk max sequence length (tokenizer truncation window)
use_topk = True           # True → top-k path; False → full-vocabulary ranks
topk = 100                # only used when use_topk=True

# =========================
# Model and tokenizer setup
# =========================

# Model card; file naming uses this for traceability
model_name = "unsloth/Llama-3.2-1B-bnb-4bit"
filename_prefix = "Llama_rank_list"

# Tokenizer: ensure a valid PAD token (fallback to EOS if needed)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantized model (bitsandbytes); compute dtype kept in float32 for stability
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
)

PAD_TOKEN_ID = tokenizer.pad_token_id

# =========================
# Device selection and data ingest
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("device=", device)

# Load dataset and basic stats
df = pd.read_csv(f"Dataset/{language}100MB.csv")
input_texts = df["text"].tolist()
total_bytes = df["length_bytes"].sum()

# =========================
# Chunking + DataLoader
# =========================

start_create_dataloader = time.perf_counter()

# 1) Tokenize and window the dataset into fixed-length chunks (no overlap here: stride=0)
input_id_list, mapping = preprocess_dataset_fast(
    input_texts,
    tokenizer,
    max_length=max_length,
    stride=0,           
)

end_tokenization = time.perf_counter()
time_tokenization = end_tokenization - start_create_dataloader

# 2) Sort chunks by decreasing number of non-PAD tokens to improve GPU utilization
input_id_list, mapping = sort_chunks_by_length(
    input_id_list,
    mapping,
    pad_token_id=PAD_TOKEN_ID,
    descending=True
)

end_sorting = time.perf_counter()
time_sorting = end_sorting - end_tokenization

# 3) Build a DataLoader over chunked tensors
dataloader = create_chunk_dataloader(
    input_id_list,
    batch_size=batch_size
)

end_create_dataloader = time.perf_counter()
time_create_dataloader = end_create_dataloader - end_sorting
total_time_dataloader = end_create_dataloader - start_create_dataloader

print("After dataloader")

# =========================
# Rank computation
# =========================

start_compute_ranks = time.perf_counter()

if use_topk:
    # Stable top-k ranks with soft-tie handling (epsilon threshold)
    rank_list, exc_list, mapping_time = compute_token_ranks_topk_fast(
        dataloader=dataloader,
        model=model,
        pad_token_id=PAD_TOKEN_ID,
        device=device,
        topk=topk,
        tie_as_exception=True,
        tie_eps_abs=3e-2
    )
else:
    # Full-vocabulary rank (O(V)) with stable id tie-break and soft-tie handling
    rank_list, exc_list, mapping_time = compute_token_ranks_fullrank_fast(
        dataloader=dataloader,
        model=model,
        pad_token_id=PAD_TOKEN_ID,
        device=device,
        tie_as_exception=True,
        tie_eps_abs=3e-2
    )

end_compute_ranks = time.perf_counter()
time_compute_ranks = end_compute_ranks - start_compute_ranks

print("Finished computing rank list")

# =========================
# Reconstruct per-row sequences
# =========================

start_reconstructing = time.perf_counter()

# Remap from chunk order back to original row order; concatenate chunk results
reconstructed_rank_list = [
    [rank for chunk_idx in mapping[row_idx] for rank in rank_list[chunk_idx]]
    for row_idx in range(len(input_texts))
]
reconstructed_exception_list = [
    [token for chunk_idx in mapping[row_idx] for token in exc_list[chunk_idx]]
    for row_idx in range(len(input_texts))
]

end_reconstructing = time.perf_counter()
reconstructing_time = end_reconstructing - start_reconstructing

print("Reconstruction completed")

# =========================
# Compression benchmarks
# =========================

results_dir = "Results/CompressedFiles"
info_dir = "Results"
csv_file_compression = "Compression_Exception_List_Info.csv"

# Time spent up to (but excluding) compression
partial_time = (total_time_dataloader + time_compute_ranks + reconstructing_time)

# Label top-k vs full-rank in the logs
if not use_topk:
    topk = "FullRank"

# ----------
# bzip2 -9
# ----------

print("Starting compression and saving with bzip2-9...")

outfile_path, compressed_size_bytes_bzip2_9, compression_time_bzip2_9 = compress_and_save (
    reconstructed_rank_list= reconstructed_rank_list,
    exception_list= reconstructed_exception_list,
    max_length=max_length,
    batch_size=batch_size,
    results_dir=results_dir,
    use_zstd=False,
    compression_level=9,
    filename_prefix=filename_prefix
)

# Print final summary
print(f"File salvato in: {outfile_path}")
print(f"Dimensione del file compresso: {compressed_size_bytes_bzip2_9} byte")

total_time = partial_time + compression_time_bzip2_9
compression_time = compression_time_bzip2_9
compressed_size_bytes = compressed_size_bytes_bzip2_9

# Create a dictionary with the information
row_dict = {
    "model": model_name,
    "language": language,
    "batch_size": batch_size,
    "max_length": max_length,
    "top_k": topk,
    "compression": "bzip2-9",
    "output_file": outfile_path,
    "dataloader_time_s": round(total_time_dataloader, 4),
    "compute_ranks_time_s": round(time_compute_ranks, 4),
    "reconstructing_time_s": round(reconstructing_time, 4),
    "compression_time_s": round(compression_time, 4),
    "total_time_s": round(total_time, 4),
    "original_size_bytes": total_bytes,
    "compressed_size_bytes": compressed_size_bytes,
    "throughput_MB_per_s": round((total_bytes / (1024 * 1024)) / total_time, 4),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

# Save row_dict on the csv
save_info_to_csv(info_dir, csv_file_compression, row_dict)

# Print results on screen
print("=== End-of-execution information ===")
for key, value in row_dict.items():
    print(f"{key:25s}: {value}")
print("=======================================\n")

# ----------
# bzip2 -3
# ----------

print("Starting compression and saving with bzip2-3...")

outfile_path, compressed_size_bytes_bzip2_3, compression_time_bzip2_3 = compress_and_save (
    reconstructed_rank_list= reconstructed_rank_list,
    exception_list= reconstructed_exception_list,
    max_length=max_length,
    batch_size=batch_size,
    results_dir=results_dir,
    use_zstd=False,
    compression_level=3,
    filename_prefix=filename_prefix
)

# Print final summary
print(f"File salvato in: {outfile_path}")
print(f"Dimensione del file compresso: {compressed_size_bytes_bzip2_3} byte")

total_time = partial_time + compression_time_bzip2_3
compression_time = compression_time_bzip2_3
compressed_size_bytes = compressed_size_bytes_bzip2_3

# Create a dictionary with the information
row_dict = {
    "model": model_name,
    "language": language,
    "batch_size": batch_size,
    "max_length": max_length,
    "top_k": topk,
    "compression": "bzip2-3",
    "output_file": outfile_path,
    "dataloader_time_s": round(total_time_dataloader, 4),
    "compute_ranks_time_s": round(time_compute_ranks, 4),
    "reconstructing_time_s": round(reconstructing_time, 4),
    "compression_time_s": round(compression_time, 4),
    "total_time_s": round(total_time, 4),
    "original_size_bytes": total_bytes,
    "compressed_size_bytes": compressed_size_bytes,
    "throughput_MB_per_s": round((total_bytes / (1024 * 1024)) / total_time, 4),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

# Save row_dict on the csv
save_info_to_csv(info_dir, csv_file_compression, row_dict)

# Print results on screen
print("=== End-of-execution information ===")
for key, value in row_dict.items():
    print(f"{key:25s}: {value}")
print("=======================================\n")

# ----------
# zstd -3
# ----------

print("Starting compression and saving with zstd3...")

outfile_path, compressed_size_bytes_zstd3, compression_time_zstd3 = compress_and_save (
    reconstructed_rank_list= reconstructed_rank_list,
    exception_list= reconstructed_exception_list,
    max_length=max_length,
    batch_size=batch_size,
    results_dir=results_dir,
    use_zstd=True,
    compression_level=3,
    filename_prefix=filename_prefix
)

# Print final summary
print(f"File salvato in: {outfile_path}")
print(f"Dimensione del file compresso: {compressed_size_bytes_zstd3} byte")


total_time = partial_time + compression_time_zstd3
compression_time = compression_time_zstd3
compressed_size_bytes = compressed_size_bytes_zstd3

# Create a dictionary with the information
row_dict = {
    "model": model_name,
    "language": language,
    "batch_size": batch_size,
    "max_length": max_length,
    "top_k": topk,
    "compression": "zstd3",
    "output_file": outfile_path,
    "dataloader_time_s": round(total_time_dataloader, 4),
    "compute_ranks_time_s": round(time_compute_ranks, 4),
    "reconstructing_time_s": round(reconstructing_time, 4),
    "compression_time_s": round(compression_time, 4),
    "total_time_s": round(total_time, 4),
    "original_size_bytes": total_bytes,
    "compressed_size_bytes": compressed_size_bytes,
    "throughput_MB_per_s": round((total_bytes / (1024 * 1024)) / total_time, 4),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

# Save row_dict on the csv
save_info_to_csv(info_dir, csv_file_compression, row_dict)

# Print results on screen
print("=== End-of-execution information ===")
for key, value in row_dict.items():
    print(f"{key:25s}: {value}")
print("=======================================\n")

# ----------
# zstd -12
# ----------

print("Starting compression and saving with zstd12...")

outfile_path, compressed_size_bytes_zstd12, compression_time_zstd12 = compress_and_save (
    reconstructed_rank_list= reconstructed_rank_list,
    exception_list= reconstructed_exception_list,
    max_length=max_length,
    batch_size=batch_size,
    results_dir=results_dir,
    use_zstd=True,
    compression_level=12,
    filename_prefix=filename_prefix
)

# Print final summary
print(f"File salvato in: {outfile_path}")
print(f"Dimensione del file compresso: {compressed_size_bytes_zstd12} byte")

total_time = partial_time + compression_time_zstd12
compression_time = compression_time_zstd12
compressed_size_bytes = compressed_size_bytes_zstd12

# Create a dictionary with the information
row_dict = {
    "model": model_name,
    "language": language,
    "batch_size": batch_size,
    "max_length": max_length,
    "top_k": topk,
    "compression": "zstd12",
    "output_file": outfile_path,
    "dataloader_time_s": round(total_time_dataloader, 4),
    "compute_ranks_time_s": round(time_compute_ranks, 4),
    "reconstructing_time_s": round(reconstructing_time, 4),
    "compression_time_s": round(compression_time, 4),
    "total_time_s": round(total_time, 4),
    "original_size_bytes": total_bytes,
    "compressed_size_bytes": compressed_size_bytes,
    "throughput_MB_per_s": round((total_bytes / (1024 * 1024)) / total_time, 4),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

# Save row_dict on the csv
save_info_to_csv(info_dir, csv_file_compression, row_dict)

# Print results on screen
print("=== End-of-execution information ===")
for key, value in row_dict.items():
    print(f"{key:25s}: {value}")
print("=======================================\n")

# ----------
# bzip3 (external CLI)
# ----------

print("Starting compression and saving with bzip3...")

outfile_path, compressed_size_bytes_bzip3, compression_time_bzip3 = compress_and_save_bz3 (
    reconstructed_rank_list= reconstructed_rank_list,
    exception_list= reconstructed_exception_list,
    max_length=max_length,
    batch_size=batch_size,
    results_dir=results_dir,
    filename_prefix=filename_prefix
)

# Print final summary
print(f"File salvato in: {outfile_path}")
print(f"Dimensione del file compresso: {compressed_size_bytes_bzip3} byte")

total_time = partial_time + compression_time_bzip3
compression_time = compression_time_bzip3
compressed_size_bytes = compressed_size_bytes_bzip3

row_dict_bz3 = {
    "model": model_name,
    "language": language,
    "batch_size": batch_size,
    "max_length": max_length,
    "top_k": topk,
    "compression": "bzip3_default",
    "output_file": outfile_path,
    "dataloader_time_s": round(total_time_dataloader, 4),
    "compute_ranks_time_s": round(time_compute_ranks, 4),
    "reconstructing_time_s": round(reconstructing_time, 4),
    "compression_time_s": round(compression_time, 4),
    "total_time_s": round(total_time, 4),
    "original_size_bytes": total_bytes,
    "compressed_size_bytes": compressed_size_bytes,
    "throughput_MB_per_s": round((total_bytes / (1024 * 1024)) / total_time, 4),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

# Save row_dict on the csv
save_info_to_csv(info_dir, csv_file_compression, row_dict_bz3)

# Print results on screen
print("=== End-of-execution information ===")
for key, value in row_dict_bz3.items():
    print(f"{key:25s}: {value}")
print("=======================================\n")

# =========================
# Timing summary → CSV
# =========================

time_dict = {
    "model": model_name,
    "language": language,
    "batch_size": batch_size,
    "max_length": max_length,
    "top_k": topk,
    "tokenization_time_s": round(time_tokenization, 4),
    "sorting_time_s": round(time_sorting, 4),
    "dataloader_time_s": round(time_create_dataloader, 4),
    "total_dataloader_time_s": round(total_time_dataloader, 4),
    "data_to_device_time_s": round(mapping_time["data_to_device"], 4),
    "forward_pass_time_s": round(mapping_time["forward_pass"], 4),
    "compute_target_logits_time_s": round(mapping_time["compute_target_logits"], 4),
    "topk_and_ranks_time_s": round(mapping_time["topk_and_ranks"], 4),
    "filter_and_split_time_s": round(mapping_time["filter_and_split"], 4),
    "total_compute_ranks_time_s": round(time_compute_ranks, 4),
    "total_reconstructing_time_s": round(reconstructing_time, 4),
}

save_info_to_csv(info_dir, "Time_Info.csv", time_dict)

print("All tasks completed.")
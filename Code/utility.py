import pandas as pd
import numpy as np
import torch
import os
import time
from typing import List, Tuple, Dict
from transformers import PreTrainedTokenizer
import io

import pickle
import bz2
import zstandard as zstd


# =====================================================
# This file contains general utility functions used across
# all the codes. These functions provide support for tasks 
# such as saving information, checking conditions, and 
# performing other simple operations.
# =====================================================


# ===== sort_chunks_by_length ===== #
def sort_chunks_by_length(
    input_id_list: List[torch.Tensor],
    mapping: Dict[int, List[int]],
    pad_token_id: int,
    descending: bool = True
) -> Tuple[List[torch.Tensor], Dict[int, List[int]]]:
    """
    Sort a list of input IDs based on the length of each tensor, without counting padding tokens.
    The mapping is updated to reflect the new order of the chunks. 

    Input:
    - input_id_list: List of tensors containing input IDs.
    - mapping: Dictionary mapping original row indices to chunk indices.
    - pad_token_id: ID of the padding token.
    - descending: If True, sort in descending order (longest first). If False, sort in ascending order (shortest first). 

    Return:
    - sorted_input_ids: List of tensors sorted by length.
    - new_mapping: Dictionary mapping original row indices to the new order of chunk indices.
    """
    # Compute the lengths of each tensor, ignoring padding tokens
    lengths = [int((ids != pad_token_id).sum().item()) for ids in input_id_list]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=descending)
    sorted_input_ids = [input_id_list[i] for i in order]

    # Mapping from original indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}

    # Update the mapping to reflect the new order
    new_mapping: Dict[int, List[int]] = {}
    for orig_idx, chunk_idxs in mapping.items():
        new_idxs = [old_to_new[ci] for ci in chunk_idxs]
        new_mapping[orig_idx] = sorted(new_idxs)

    return sorted_input_ids, new_mapping


# =============================================
# ======= FUNCTIONS FOR CHECK CONDITIONS ======
# =============================================


# ===== check_bos_token_in_chunks ===== #
def check_bos_token_in_chunks(
    chunks: List[torch.Tensor], 
    tokenizer: PreTrainedTokenizer
) -> bool:
    """
    Check if each chunk starts with the tokenizer's BOS (begin-of-sequence) token.
    
    Input:
    - chunks: list of torch.Tensor, each representing a tokenized chunk.
    - tokenizer: HuggingFace tokenizer to retrieve the bos_token_id from.
    
    Return:
    - True if all chunks start with bos_token_id, False otherwise.
    Prints information about chunks that do not conform.
    """
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError("The tokenizer does not have a defined bos_token_id.")

    all_correct = True
    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            print(f"⚠️ Empty chunk at index {i}.")
            all_correct = False
        elif chunk[0].item() != bos_id:
            print(f"❌ Chunk {i} does not start with bos_token_id ({bos_id}): first token = {chunk[0].item()}")
            all_correct = False
    if all_correct:
        print("✅ All chunks correctly start with the BOS token.")
    return all_correct



# ===== count_nonpad_tokens_per_row ===== #
def count_nonpad_tokens_per_row(
    input_id_list: List[torch.Tensor],
    mapping: Dict[int, List[int]],
    pad_token_id: int,
    bos_token_id: int,
) -> Dict[int, int]:
    """
    Count the number of non-padding tokens in each row of a list of input IDs. 
    Use the mapping to determine which chunks belong to which original row. 
    
    Input:
    - input_id_list: List of tensors containing input IDs.
    - mapping: Dictionary mapping original row indices to chunk indices.
    - pad_token_id: ID of the padding token. 
    - bos_token_id: ID of the beginning-of-sequence token.
    
    Return:
    - row_token_counts: Dictionary mapping original row indices to the count of non-padding tokens.
    """
    row_token_counts: Dict[int, int] = {}
    for orig_idx, chunk_indices in mapping.items():
        total = 0
        for ci in chunk_indices:
            ids: torch.Tensor = input_id_list[ci]
            # Count only tokens != pad AND != bos
            mask = (ids != pad_token_id) & (ids != bos_token_id)
            nonpad_nobos = int(mask.sum().item())
            total += nonpad_nobos
        row_token_counts[orig_idx] = total
    return row_token_counts



# ====== verify_reconstruction ====== #
def verify_reconstruction(
    input_texts: List[str], 
    reconstructed_tokens: Dict[int, List[int]], 
    tokenizer: torch.nn.Module, 
    debug: bool = True
)-> bool:
    """
    Verify that the reconstructed token IDs match those obtained by directly 
    tokenizing the original input texts. Special tokens (e.g., <pad>, <bos>) are ignored by disabling 
    `add_special_tokens` during reference tokenization.

    Input:
    - input_texts (List[str]): List of original text sequences.
    - reconstructed_tokens (Dict[int, List[int]]): Mapping from sequence index 
        to a list of reconstructed token IDs.
    - tokenizer: Tokenizer instance used for encoding and decoding.
    - debug (bool): If True, print detailed information about mismatches.

    Return:
    - success (bool): True if all reconstructed sequences match the expected 
        token IDs, False otherwise.
    """
    success = True
    for idx, token_ids in reconstructed_tokens.items():
        original_text = input_texts[idx]

        # Tokenize the original text without adding special tokens
        encoded = tokenizer(original_text, add_special_tokens=False)
        expected_ids = encoded["input_ids"]

        if token_ids != expected_ids:
            success = False
            if debug:
                print(f"\n❌ Mismatch at index {idx}")
                print(f"Text       : {original_text}")
                print(f"Expected   : {expected_ids}")
                print(f"Reconstructed: {token_ids}")
                print(f"Expected tokens   : {tokenizer.convert_ids_to_tokens(expected_ids)}")
                print(f"Reconstructed toks: {tokenizer.convert_ids_to_tokens(token_ids)}")
        else:
            if debug:
                print(f"\n✅ Match at index {idx}")

    return success


# =============================================
# ======= FUNCTIONS FOR SAVE INFORMATION ======
# =============================================


# ====== save_rank_list_to_file ====== #
def save_rank_list_to_file(rank_list: List[List[int]], file_path: str, 
                        execution_time: float, model_name: str) -> None:
    """
    Save the rank list to a file, including execution time.
    Each sublist in the rank list is saved in a separate line. 
    The execution time is appended at the end of the file.
    
    Input:
    - rank_list: List of lists containing ranks.
    - file_path: Path to the output file.
    - execution_time: Time taken to compute the rank list (in seconds).
    - model_name: Name of the model used for processing.
    """
    with open(file_path, 'w') as file:
        file.write(f"Execution time ({model_name} processing): {execution_time:.4f} seconds\n")
        for sublist in rank_list:
            file.write(f"[{' '.join(map(str, sublist))}]\n")
            


# ====== save_info_to_csv ====== #            
def save_info_to_csv(
    folder_path: str, 
    csv_filename: str, 
    row_dict: dict
) -> None:
    """
    Save or append a row to a CSV file in the specified folder.
    The row contains information about the model used, compression time, and other relevant metrics.
    If the CSV file does not exist, it will be created with headers.
    If it exists, the new row will be appended without headers. 

    Input:
    - folder_path (str): Path to the folder where the CSV will be saved or searched.
    - csv_filename (str): Name of the CSV file (e.g., "DeepSeek_rank_list_info.csv").
    - row_dict (dict): Dictionary containing the columns and values to insert (e.g., {"model": ..., "inference_time_s": ..., ...}).
    """
    # Check if the folder exists, create it if not
    os.makedirs(folder_path, exist_ok=True)

    csv_path = os.path.join(folder_path, csv_filename)
    df_row = pd.DataFrame([row_dict])

    # If the CSV file does not exist, create it with headers
    if not os.path.isfile(csv_path):
        df_row.to_csv(csv_path, mode="w", header=True, index=False)
        print(f"CSV creato con header: {csv_path}")
    # If the file exists, append the new row without headers
    else:
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
        print(f"Riga aggiunta al CSV esistente: {csv_path}")
        
        

# ====== compress_and_save ====== #
def compress_and_save(
    reconstructed_rank_list: list[list[int]],
    results_dir: str,
    *,
    binary: bool,
    use_zstd: bool,
    compression_level: int,
    filename_prefix: str
) -> tuple[str, int, float]:
    """
    Serialize and compress 'reconstructed_rank_list', then save the compressed file into 'results_dir'.

    Input:
    - reconstructed_rank_list: a list of lists of integers to compress.
    - results_dir: destination folder (will be created if it doesn’t exist).
    - binary: if True, convert the data into NumPy → .npy; if False, use pickle.dumps.
    - use_zstd: if True, use Zstandard; if False, use bzip2.
    - compression_level: compression level (for Zstd: 1-22 typically, for bzip2: 1-9).
    - filename_prefix: prefix for the output filename, e.g. "DeepSeek_rank_list".

    Return:
    - outfile_path: full path to the saved file,
    - compressed_size_bytes: size of the compressed file in bytes,
    - compression_time: time taken to perform compression (in seconds).
    """
    
    # Ensure that the output directory exists (create it if necessary)
    os.makedirs(results_dir, exist_ok=True)

    start = time.perf_counter()

    # Serialize into a single byte blob (data_blob)
    if binary:
        # Convert the list of lists into two NumPy arrays and write them to an in-memory .npy buffer
        max_value = max(max(lst) for lst in reconstructed_rank_list if lst)
        dtype = np.uint16 if max_value <= np.iinfo(np.uint16).max else np.uint32
        lengths = np.array([len(lst) for lst in reconstructed_rank_list], dtype=np.int32)
        flat_array = np.concatenate(
            [np.array(lst, dtype=dtype) for lst in reconstructed_rank_list]
        )
        buf = io.BytesIO()
        np.save(buf, lengths, allow_pickle=False)      # save lengths array header + data
        np.save(buf, flat_array, allow_pickle=False)   # save flattened array header + data
        data_blob = buf.getvalue()
    else:
        # Directly serialize the Python list of lists using pickle
        data_blob = pickle.dumps(reconstructed_rank_list)

    # Compress the serialized byte blob
    if use_zstd:
        compressor = zstd.ZstdCompressor(level=compression_level)
        compressed_data = compressor.compress(data_blob)
        ext = "zst"
        compressor_name = f"zstd{compression_level}"
    else:
        compressed_data = bz2.compress(data_blob, compresslevel=compression_level)
        ext = "bz2"
        compressor_name = f"bzip2-{compression_level}"
        
    end = time.perf_counter()
    compression_time = end - start

    # Build the output filename based on chosen mode and compressor
    mode = "binary" if binary else "pickle"
    filename = f"{filename_prefix}_{mode}_{compressor_name}.{ext}"
    outfile_path = os.path.join(results_dir, filename)
    #save_path = os.path.join(full_results_dir, filename)

    # Write the compressed bytes to disk
    with open(outfile_path, "wb") as f_out:
        f_out.write(compressed_data)

    compressed_size_bytes = len(compressed_data)
    return outfile_path, compressed_size_bytes, compression_time
        

 
# ===================================================================================================
# ================================ OLD FUNCTIONS - NOT USED =========================================
# ===================================================================================================



# ====== sort_rank_lists_by_length ====== #
def sort_rank_lists_by_length(
    rank_lists: List[List[int]]
) -> Tuple[List[List[int]], Dict[int, int]]:
    """
    Sorts a collection of rank lists by sequence length (descending order),
    while preserving a mapping to the original indices.

    Input:
    - rank_lists (List[List[int]]): A list of rank lists, one per input sequence.

    Return:
    - sorted_lists (List[List[int]]): The rank lists sorted by decreasing length.
    - index_map (Dict[int, int]): A mapping from the new index to the original index.
    """
    # Build metadata tuples: (length, original_index, list)
    meta = [(len(lst), idx, lst) for idx, lst in enumerate(rank_lists)]
    
    # Sort by length in descending order
    meta_sorted = sorted(meta, key=lambda x: x[0], reverse=True)
    # Extract sorted lists and create mapping new_idx -> original_idx
    sorted_lists = [item[2] for item in meta_sorted]
    index_map = {new_idx: orig_idx for new_idx, (_, orig_idx, _) in enumerate(meta_sorted)}
    
    return sorted_lists, index_map


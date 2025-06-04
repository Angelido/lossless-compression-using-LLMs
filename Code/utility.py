import pandas as pd
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from transformers import AutoModelForSeq2SeqLM
from unixcoder import UniXcoder
from typing import List, Tuple, Dict



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
    If it exists, the new row will be appended without headers. \n

    Input:
    - folder_path: Path to the folder where the CSV will be saved or searched.
    - csv_filename: Name of the CSV file (e.g., "DeepSeek_rank_list_info.csv").
    - row_dict: Dictionary containing the columns and values to insert (e.g., {"model": ..., "inference_time_s": ..., ...}).
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



# ====== sort_rank_lists_by_length ====== #
def sort_rank_lists_by_length(
    rank_lists: List[List[int]]
) -> Tuple[List[List[int]], Dict[int, int]]:
    """
    Ordina le liste di rank per lunghezze simili (descending), mantenendo un mapping all'indice originale.

    Args:
        rank_lists: lista di liste di rank (una per riga di testo).
    Returns:
        sorted_lists: liste ordinate per lunghezza decrescente.
        index_map: dizionario che mappa nuovo indice -> indice originale.
    """
    # crea lista di (len, original_idx, ranks)
    meta = [(len(lst), idx, lst) for idx, lst in enumerate(rank_lists)]
    # ordina per lunghezza decrescente
    meta_sorted = sorted(meta, key=lambda x: x[0], reverse=True)
    sorted_lists = [item[2] for item in meta_sorted]
    index_map = {new_idx: orig_idx for new_idx, (_, orig_idx, _) in enumerate(meta_sorted)}
    return sorted_lists, index_map

 
# ===================================================================================================
# ================================ OLD FUNCTIONS - NOT USED =========================================
# ===================================================================================================


# # ====== parse_code_blocks ====== #
# def parse_code_blocks(file_path: str) -> pd.DataFrame:
#     """
#     Parse a file containing code blocks separated by headers.
#     The headers are lines starting with '#'. \n
#     Input:
#     - file_path: Path to the input file. \n
#     Return:
#     - a DataFrame with a column 'Code Block' containing the code blocks.
 
#     """
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     blocks = []
#     current_block = []

#     for line in lines:
#         stripped = line.strip()
        
#         if stripped.startswith("#"):  # New block
#             if current_block:  
#                 blocks.append("\n".join(current_block))
#             current_block = [stripped]  
#         elif stripped:  
#             current_block.append(" " + stripped)  

#     # Add the last block
#     if current_block:
#         blocks.append("\n".join(current_block))

#     return pd.DataFrame({"Code Block": blocks})



# # ====== predict_next_token ====== #
# def predict_next_token(input_ids: torch.Tensor, model: torch.nn.Module, 
#                     tokenizer: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Predict the next token given a sequence of input tokens. \n
#     Input:
#     - input_ids: Tensor of input token IDs.
#     - model: Pre-trained language model.
#     - tokenizer: Tokenizer used to encode the input. \n
#     Return:
#     - top_k_probs: Tensor of probabilities for the top k tokens.
#     - top_k_tokens: Tensor of the top k token IDs.
#     - probs: Tensor of probabilities for all tokens in the vocabulary.
#     """
#     # Compute the logits of the next token
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids)
#         logits = outputs.logits[:, -1, :]  
#         # Compute the probabilities
#         probs = torch.softmax(logits, dim=-1)


#     # Get the top 5 most probable tokens
#     top_k_probs, top_k_tokens = torch.topk(probs, k=5, dim=-1)

#     return top_k_probs, top_k_tokens, probs



# # ====== compute_token_ranks ====== #
# def compute_token_ranks(input_ids: torch.Tensor, model: torch.nn.Module,
#                         tokenizer: torch.nn.Module, device: str,
#                         pad_token_id: int) -> List[List[int]]: 
#     """
#     Compute the rank of each token in the input sequences. 
#     Stops processing a sequence when encountering the padding token. \n
#     Input:
#     - input_ids: Tensor of input token IDs.
#     - model: Pre-trained language model.
#     - tokenizer: Tokenizer used to encode the input.
#     - device: Device to run the model on (e.g., 'cuda' or 'cpu').
#     - pad_token_id: ID of the padding token. \n
#     Return:
#     - rank_list: List of ranks for each sequence.
#     """
#     rank_list = [[] for _ in range(input_ids.shape[0])]

#     for seq_idx in range(input_ids.shape[0]):  
#         for i in range(len(input_ids[seq_idx]) - 1):  
            
#             # Stop if the padding token is reached
#             if input_ids[seq_idx, i].item() == pad_token_id:
#                 break  

#             token_prefix = input_ids[seq_idx, :i+1].unsqueeze(0)  

#             # Predict next token probabilities
#             _, _, probs = predict_next_token(token_prefix, model, tokenizer)

#             actual_token_id = input_ids[seq_idx, i+1].item()  

#             # Sort all tokens by probability
#             sorted_indices = torch.argsort(probs, descending=True)
            
#             # Get the rank of the actual token
#             actual_token_rank = (sorted_indices == actual_token_id).nonzero(as_tuple=True)[1].item()  
#             rank_list[seq_idx].append(actual_token_rank)

#     return rank_list



# # ====== compute_token_ranks_parallel ====== #
# def compute_token_ranks_parallel(input_ids: torch.Tensor, model: torch.nn.Module, 
#                                  tokenizer: torch.nn.Module, device: str, 
#                                  pad_token_id: int) -> List[List[int]]:
#     """
#     Compute the rank of each token in the input sequences for the entire batch simultaneously.
#     Stops processing a sequence when encountering the padding token. \n
#     Input:
#     - input_ids: Tensor of input token IDs.
#     - model: Pre-trained language model.
#     - tokenizer: Tokenizer used to encode the input.   
#     - device: Device to run the model on (e.g., 'cuda' or 'cpu').
#     - pad_token_id: ID of the padding token. \n
#     Return:
#     - rank_list: List of ranks for each sequence.
#     """
#     batch_size, seq_len = input_ids.shape
#     rank_list = [[] for _ in range(batch_size)]
    
#     # Mask to track which sequences are still active (1 = active, 0 = stopped due to padding)
#     active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

#     for i in range(seq_len - 1):  
#         if not active_mask.any():
#             break  # Stop if all sequences reached padding
        
#         # Get only the active sequences (those that haven't reached padding yet)
#         active_indices = active_mask.nonzero(as_tuple=True)[0]
#         token_prefixes = input_ids[active_indices, :i+1]  # Keep growing the prefix
        
#         # Predict next token probabilities for all active sequences
#         _, _, probs = predict_next_token(token_prefixes, model, tokenizer)

#         # Compute ranks for each active sequence
#         sorted_indices = torch.argsort(probs, descending=True)  # Get sorted token indices
#         actual_token_ids = input_ids[active_indices, i+1]  # Get the actual next token
        
#         for idx, seq_idx in enumerate(active_indices):
#             # Get rank of the actual token
#             actual_token_rank = (sorted_indices[idx] == actual_token_ids[idx]).nonzero(as_tuple=True)[0].item()
#             rank_list[seq_idx].append(actual_token_rank)

#             # Stop processing this sequence if we reach padding
#             if actual_token_ids[idx].item() == pad_token_id:
#                 active_mask[seq_idx] = False  

#     return rank_list



# # ====== regenerate_texts ====== #
# def compute_regenerate_texts(rank_list: List[List[int]], model: torch.nn.Module, 
#                      tokenizer: torch.nn.Module, device: str) -> List[str]:
#     """
#     Regenerate texts based on the rank list. \n
#     Input:
#     - rank_list: List of ranks for each sequence.
#     - model: Pre-trained language model.
#     - tokenizer: Tokenizer used to encode the input.
#     - device: Device to run the model on (e.g., 'cuda' or 'cpu'). \n
#     Return:
#     - generated_texts: List of regenerated texts.
#     """
#     generated_texts = []  
#      # Get the start token
#     start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id 

#     for seq_idx in range(len(rank_list)): 
#         generated_tokens = []  
#         token_prefix = torch.tensor([[start_token_id]], device=device)  # Start with the first token

#         for rank in rank_list[seq_idx]:  
#             # Predict the next token's probabilities
#             _, _, probs = predict_next_token(token_prefix, model, tokenizer)

#             # Sort token probabilities in descending order
#             sorted_probs, sorted_indices = torch.sort(probs, descending=True)

#             # Select the token based on the stored rank
#             generated_token_id = sorted_indices[:, rank].item()
#             generated_tokens.append(generated_token_id)  

#             # Update the prefix by adding the newly generated token
#             token_prefix = torch.cat((token_prefix, torch.tensor([[generated_token_id]], device=device)), dim=1)

#             # Stop if the generated token is the end-of-sequence token
#             if generated_token_id == tokenizer.eos_token_id:
#                 break

#         # Decode the generated token IDs into a text string
#         generated_texts.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))

#     return generated_texts 
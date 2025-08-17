import pandas as pd
import torch
from typing import List, Tuple, Dict



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



# ====== parse_code_blocks ====== #
def parse_code_blocks(file_path: str) -> pd.DataFrame:
    """
    Parse a file containing code blocks separated by headers.
    The headers are lines starting with '#'. \n
    Input:
    - file_path: Path to the input file. \n
    Return:
    - a DataFrame with a column 'Code Block' containing the code blocks.
 
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    blocks = []
    current_block = []

    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith("#"):  # New block
            if current_block:  
                blocks.append("\n".join(current_block))
            current_block = [stripped]  
        elif stripped:  
            current_block.append(" " + stripped)  

    # Add the last block
    if current_block:
        blocks.append("\n".join(current_block))

    return pd.DataFrame({"Code Block": blocks})



# ====== predict_next_token ====== #
def predict_next_token(input_ids: torch.Tensor, model: torch.nn.Module, 
                    tokenizer: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predict the next token given a sequence of input tokens. \n
    Input:
    - input_ids: Tensor of input token IDs.
    - model: Pre-trained language model.
    - tokenizer: Tokenizer used to encode the input. \n
    Return:
    - top_k_probs: Tensor of probabilities for the top k tokens.
    - top_k_tokens: Tensor of the top k token IDs.
    - probs: Tensor of probabilities for all tokens in the vocabulary.
    """
    # Compute the logits of the next token
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]  
        # Compute the probabilities
        probs = torch.softmax(logits, dim=-1)


    # Get the top 5 most probable tokens
    top_k_probs, top_k_tokens = torch.topk(probs, k=5, dim=-1)

    return top_k_probs, top_k_tokens, probs



# ====== compute_token_ranks ====== #
def compute_token_ranks(input_ids: torch.Tensor, model: torch.nn.Module,
                        tokenizer: torch.nn.Module, device: str,
                        pad_token_id: int) -> List[List[int]]: 
    """
    Compute the rank of each token in the input sequences. 
    Stops processing a sequence when encountering the padding token. \n
    Input:
    - input_ids: Tensor of input token IDs.
    - model: Pre-trained language model.
    - tokenizer: Tokenizer used to encode the input.
    - device: Device to run the model on (e.g., 'cuda' or 'cpu').
    - pad_token_id: ID of the padding token. \n
    Return:
    - rank_list: List of ranks for each sequence.
    """
    rank_list = [[] for _ in range(input_ids.shape[0])]

    for seq_idx in range(input_ids.shape[0]):  
        for i in range(len(input_ids[seq_idx]) - 1):  
            
            # Stop if the padding token is reached
            if input_ids[seq_idx, i].item() == pad_token_id:
                break  

            token_prefix = input_ids[seq_idx, :i+1].unsqueeze(0)  

            # Predict next token probabilities
            _, _, probs = predict_next_token(token_prefix, model, tokenizer)

            actual_token_id = input_ids[seq_idx, i+1].item()  

            # Sort all tokens by probability
            sorted_indices = torch.argsort(probs, descending=True)
            
            # Get the rank of the actual token
            actual_token_rank = (sorted_indices == actual_token_id).nonzero(as_tuple=True)[1].item()  
            rank_list[seq_idx].append(actual_token_rank)

    return rank_list



# ====== compute_token_ranks_parallel ====== #
def compute_token_ranks_parallel(input_ids: torch.Tensor, model: torch.nn.Module, 
                                 tokenizer: torch.nn.Module, device: str, 
                                 pad_token_id: int) -> List[List[int]]:
    """
    Compute the rank of each token in the input sequences for the entire batch simultaneously.
    Stops processing a sequence when encountering the padding token. \n
    Input:
    - input_ids: Tensor of input token IDs.
    - model: Pre-trained language model.
    - tokenizer: Tokenizer used to encode the input.   
    - device: Device to run the model on (e.g., 'cuda' or 'cpu').
    - pad_token_id: ID of the padding token. \n
    Return:
    - rank_list: List of ranks for each sequence.
    """
    batch_size, seq_len = input_ids.shape
    rank_list = [[] for _ in range(batch_size)]
    
    # Mask to track which sequences are still active (1 = active, 0 = stopped due to padding)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    for i in range(seq_len - 1):  
        if not active_mask.any():
            break  # Stop if all sequences reached padding
        
        # Get only the active sequences (those that haven't reached padding yet)
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        token_prefixes = input_ids[active_indices, :i+1]  # Keep growing the prefix
        
        # Predict next token probabilities for all active sequences
        _, _, probs = predict_next_token(token_prefixes, model, tokenizer)

        # Compute ranks for each active sequence
        sorted_indices = torch.argsort(probs, descending=True)  # Get sorted token indices
        actual_token_ids = input_ids[active_indices, i+1]  # Get the actual next token
        
        for idx, seq_idx in enumerate(active_indices):
            # Get rank of the actual token
            actual_token_rank = (sorted_indices[idx] == actual_token_ids[idx]).nonzero(as_tuple=True)[0].item()
            rank_list[seq_idx].append(actual_token_rank)

            # Stop processing this sequence if we reach padding
            if actual_token_ids[idx].item() == pad_token_id:
                active_mask[seq_idx] = False  

    return rank_list



# ====== regenerate_texts ====== #
def compute_regenerate_texts(rank_list: List[List[int]], model: torch.nn.Module, 
                     tokenizer: torch.nn.Module, device: str) -> List[str]:
    """
    Regenerate texts based on the rank list. \n
    Input:
    - rank_list: List of ranks for each sequence.
    - model: Pre-trained language model.
    - tokenizer: Tokenizer used to encode the input.
    - device: Device to run the model on (e.g., 'cuda' or 'cpu'). \n
    Return:
    - generated_texts: List of regenerated texts.
    """
    generated_texts = []  
     # Get the start token
    start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id 

    for seq_idx in range(len(rank_list)): 
        generated_tokens = []  
        token_prefix = torch.tensor([[start_token_id]], device=device)  # Start with the first token

        for rank in rank_list[seq_idx]:  
            # Predict the next token's probabilities
            _, _, probs = predict_next_token(token_prefix, model, tokenizer)

            # Sort token probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            # Select the token based on the stored rank
            generated_token_id = sorted_indices[:, rank].item()
            generated_tokens.append(generated_token_id)  

            # Update the prefix by adding the newly generated token
            token_prefix = torch.cat((token_prefix, torch.tensor([[generated_token_id]], device=device)), dim=1)

            # Stop if the generated token is the end-of-sequence token
            if generated_token_id == tokenizer.eos_token_id:
                break

        # Decode the generated token IDs into a text string
        generated_texts.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))

    return generated_texts 
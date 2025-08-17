from torch.utils.data import DataLoader, Dataset
import torch
from unixcoder import UniXcoder
from typing import List, Dict, Tuple


# =====================================================
# This file contains functions to create the DataLoader 
# for processing data in batch. In particular, it handles 
# splitting lines into chunks and prepending a BOS token 
# at the beginning of each line.
# Chunking is based on batch_size and max_length.
# =====================================================


# ====== ChunkDataset ====== #
class ChunkDataset(Dataset):
    '''
    Custom dataset class for handling chunks of tokenized input IDs. 
    
    Input:
    - input_id_list (List[torch.Tensor]): List of tokenized input IDs.
    '''
    def __init__(self, input_id_list: List[torch.Tensor]):
        self.input_ids = input_id_list

    def __len__(self):
        '''
        Returns the total number of chunks in the dataset.
        '''
        return len(self.input_ids)

    def __getitem__(self, idx):
        '''
        Returns the tokenized input IDs for the chunk at the given index.
        '''
        return self.input_ids[idx]



# ====== create_chunk_dataloader ====== #
def create_chunk_dataloader(
    input_id_list: List[torch.Tensor], 
    batch_size: int = 8
) -> DataLoader:
    '''
    Create a DataLoader for the given list of tokenized input IDs. 
    
    Input:
    - input_id_list (List[torch.Tensor]): List of tokenized input IDs.
    - batch_size (int): Batch size for the DataLoader (default=8). 
    
    Return:
    - dataloader (DataLoader): DataLoader for the tokenized input IDs.    
    '''
    dataset = ChunkDataset(input_id_list)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)



# ====== preprocess_dataset_fast ====== #
def preprocess_dataset_fast(
    input_texts: List[str],
    tokenizer: torch.nn.Module,
    max_length: int,
    stride: int = 0
) -> Tuple[List[torch.Tensor], Dict[int, List[int]]]:
    """
    Use the tokenizer with return_overflowing_tokens to create content chunks
    (each of length `max_length`), then manually insert BOS at the beginning
    of every chunk. Returns tensors of length max_length + 1.

    Input:
    - input_texts (List[str]): list of raw text strings to be chunked.
    - tokenizer (torch.nn.Module): the Hugging Face tokenizer to use.
    - max_length (int): number of content tokens per chunk (excluding BOS).
    - stride (int): tokenizer stride (overlap) applied to content tokens.

    Return:
    - all_input_ids (List[torch.Tensor]): list of torch.Tensor, each tensor has length max_length + 1 (BOS + content)
    - mapping (Dict[int, List[int]]): dict mapping original sample index -> list of chunk indices in all_input_ids
    """
    tokenizer.padding_side = "right"

    # Tokenize with overflow but without special tokens (we'll add BOS afterwards)
    enc = tokenizer(
        input_texts,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_attention_mask=False,
        return_token_type_ids=False,
        padding="max_length",
        add_special_tokens=False
    )

    # Determine BOS id (fallback to eos/cls if necessary)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_id = getattr(tokenizer, "eos_token_id", None)
    if bos_id is None:
        bos_id = getattr(tokenizer, "cls_token_id", None)
    if bos_id is None:
        raise ValueError("Tokenizer does not have a bos/eos/cls token id defined.")

    # Convert to tensors and prepend BOS
    raw_input_ids = enc["input_ids"]  # each has length `max_length` (padded)
    all_input_ids: List[torch.Tensor] = []
    for ids in raw_input_ids:
        new_ids = [bos_id] + ids  # resulting length = max_length + 1
        all_input_ids.append(torch.tensor(new_ids, dtype=torch.long))

    # Build mapping 
    mapping: Dict[int, List[int]] = {}
    for chunk_idx, orig_idx in enumerate(enc["overflow_to_sample_mapping"]):
        mapping.setdefault(orig_idx, []).append(chunk_idx)

    return all_input_ids, mapping



# ====== preprocess_dataset_fast_unixcoder ====== #
def preprocess_dataset_fast_unixcoder(
    input_texts: List[str],
    ux: UniXcoder,
    max_length: int,
    stride: int = 0
) -> tuple[List[torch.Tensor], Dict[int, List[int]]]:
    """
    Preprocess a list of texts into chunks of token IDs using the UniXcoder tokenizer.
    
    Input:
    - input_texts (List[str]): List of raw text inputs (e.g., from a dataset).
    - ux (UniXcoder): UniXcoder tokenizer object.
    - max_length (int): Maximum number of tokens per chunk.
    - stride (int): Stride for overlapping chunks (default=0).
    
    Return:
    - all_input_ids (List[torch.Tensor]): List of tokenized input IDs for each chunk.
    - mapping (Dict[int, List[int]]): Mapping from original row index to indices of its chunks
      in `all_input_ids`.
    """

    assert max_length < 1024, "UniXcoder supports up to 1024 tokens"
    assert stride < max_length, "Stride must be smaller than max_length"
    
    tokenizer = ux.tokenizer
    tokenizer.padding_side = "right"
    prefix = [tokenizer.cls_token]
    prefix_len = 1
    prefix_ids = tokenizer.convert_tokens_to_ids(prefix)
    available_len = max_length - prefix_len
    
    all_input_ids = []
    mapping: Dict[int, List[int]] = {}
    chunk_counter = 0
    
    for idx, text in enumerate(input_texts):
        # Tokenize the entire input (no truncation)
        full_tokens = tokenizer.tokenize(text)
        full_ids = tokenizer.convert_tokens_to_ids(full_tokens)

        start = 0
        while start < len(full_ids):
            end = min(start + available_len, len(full_ids))
            chunk_ids = prefix_ids + full_ids[start:end]
            # Pad if needed
            if len(chunk_ids) < max_length:
                pad_id = ux.config.pad_token_id
                chunk_ids += [pad_id] * (max_length - len(chunk_ids))
            # Append result
            all_input_ids.append(torch.tensor(chunk_ids, dtype=torch.long))
            mapping.setdefault(idx, []).append(chunk_counter)
            chunk_counter += 1
            if end == len(full_ids):
                break
            start += (available_len - stride)
    
    return all_input_ids, mapping



def get_token_info(tokenizer):
    """
    Retrieve basic special-token IDs and their textual representations from a Hugging Face tokenizer.

    Returns a dictionary with:
      - 'bos_id'   : integer id of BOS token or None
      - 'eos_id'   : integer id of EOS token or None
      - 'cls_id'   : integer id of CLS token or None
      - 'pad_id'   : integer id of PAD token or None
      - 'bos_token': string representation of the BOS token (or None)
      - 'pad_token': string representation of the PAD token (or None)

    The function uses `getattr` to safely query token attributes and `convert_ids_to_tokens`
    to map an id back to a textual token. Any exceptions during conversion are caught and
    result in None for the textual token.
    """
    info = {}
    # Try to get common special token ids; return None if not present
    info['bos_id'] = getattr(tokenizer, "bos_token_id", None)
    info['eos_id'] = getattr(tokenizer, "eos_token_id", None)
    info['cls_id'] = getattr(tokenizer, "cls_token_id", None)
    info['pad_id'] = getattr(tokenizer, "pad_token_id", None)

    # Helper: convert an id to its string token safely
    def id_to_token(tid):
        if tid is None:
            return None
        try:
            return tokenizer.convert_ids_to_tokens(tid)
        except Exception:
            # If conversion fails for any reason, return None instead of crashing
            return None

    # Convert bos and pad ids (if any) to token strings
    info['bos_token'] = id_to_token(info['bos_id'])
    info['pad_token'] = id_to_token(info['pad_id'])
    return info



# ===================================================================================================
# ================================ OLD FUNCTIONS - NOT USED =========================================
# ===================================================================================================



# ====== preprocess_dataset_fast_old ====== #
def preprocess_dataset_fast_old(
    input_texts: List[str], 
    tokenizer: torch.nn.Module, 
    max_length: int, 
    stride: int = 0
) -> tuple[List[torch.Tensor], Dict[int, List[int]]]:
    '''
    Split a list of texts into chunks of token IDs using the tokenizer.
    
    Input:
    - input_texts (List[str]): List of raw text inputs (e.g., from a dataset).
    - tokenizer: Tokenizer used for text processing.
    - max_length (int): Maximum number of tokens per chunk.
    - stride (int): Stride for overlapping chunks (default=0). 
    
    Return:
    - all_input_ids (List[torch.Tensor]): List of tokenized input IDs for each chunk.
    - mapping (Dict[int, List[int]]): Mapping from original row index to indices of its chunks
      in `all_input_ids`.
    '''
    
    tokenizer.padding_side = "right"
    
    enc = tokenizer(
        input_texts,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_attention_mask=False,
        return_token_type_ids=False,
        padding="max_length"
    )
    # Convert the input IDs to tensors
    all_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in enc["input_ids"]]

    # Create a mapping from original row index to chunk indices
    mapping: Dict[int, List[int]] = {}
    for chunk_idx, orig_idx in enumerate(enc["overflow_to_sample_mapping"]):
        mapping.setdefault(orig_idx, []).append(chunk_idx)

    return all_input_ids, mapping



# ====== preprocess_dataset_fast_single ====== #
def preprocess_dataset_fast_single(
    input_texts: List[str], 
    tokenizer: torch.nn.Module, 
    max_length: int, 
    stride: int = 0
) -> tuple[List[torch.Tensor], Dict[int, List[int]]]:
    """
    Tokenizes a list of texts one by one, splitting each text into 
    chunks of at most `max_length` tokens (with optional overlapping via `stride`). 
    Each chunk is padded to `max_length` tokens.

    Input:
    - input_texts (List[str]): List of raw text inputs.
    - tokenizer (torch.nn.Module): HuggingFace tokenizer.
    - max_length (int): Maximum number of tokens per chunk.
    - stride (int, optional): Overlap between chunks. Default = 0.

    Return:
    - all_input_ids (List[torch.Tensor]): List of tokenized chunks 
            (each as a tensor of IDs).
    - mapping (Dict[int, List[int]]): Maps each original text index 
            to the list of indices of its corresponding chunks in `all_input_ids`.
    """
    tokenizer.padding_side = "right"

    all_input_ids = []
    mapping = {}

    for idx, text in enumerate(input_texts):
        enc = tokenizer(
            text,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding="max_length"
        )
        # enc["input_ids"] is a list of tokenized chunks of that single string
        chunk_indices = []
        for chunk_input_ids in enc["input_ids"]:
            all_input_ids.append(torch.tensor(chunk_input_ids, dtype=torch.long))
            chunk_indices.append(len(all_input_ids) - 1)

        mapping[idx] = chunk_indices

    return all_input_ids, mapping
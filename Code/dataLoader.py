from torch.utils.data import DataLoader, Dataset
import torch
from unixcoder import UniXcoder
from typing import List, Dict


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
def create_chunk_dataloader(input_id_list: List[torch.Tensor], batch_size: int = 8) -> DataLoader:
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
    stride: int = 0):
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



# ====== preprocess_dataset_fast_unixcoder ====== #
def preprocess_dataset_fast_unixcoder(input_texts: List[str],
                                      ux: UniXcoder,
                                      max_length: int,
                                      stride: int = 0):
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
    # Set the padding side to the right
    ux.tokenizer.padding_side = "right"
    
    # Tokenize the input texts
    all_ids = ux.tokenize(input_texts,
                           mode="<decoder-only>",
                           max_length=max_length,
                           padding=True)
    
    # Create a mapping from original row index to chunk indices
    mapping: Dict[int, List[int]] = {}
    for idx, ids in enumerate(all_ids):
        mapping.setdefault(idx, []).append(idx)
    # Convert the input IDs to tensors
    all_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in all_ids]
    
    return all_input_ids, mapping


# ===================================================================================================
# ================================ OLD FUNCTIONS - NOT USED =========================================
# ===================================================================================================



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
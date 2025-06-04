from torch.utils.data import DataLoader, Dataset
import torch
from typing import List, Dict
from utility import compute_token_ranks_parallel


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
def preprocess_dataset_fast(input_texts: List[str], tokenizer: torch.nn.Module, max_length: int, stride: int = 0):
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


# ===================================================================================================
# ================================ OLD FUNCTIONS - NOT USED =========================================
# ===================================================================================================

# ====== CodeDataset ====== #
class CodeDataset(Dataset):
    """
    Custom dataset class for tokenizing code blocks. 
    
    Input:
    - code_blocks (List[str]): List of code blocks to tokenize.
    - tokenizer (torch.nn.Module): Tokenizer to use for tokenization.
    - max_length (int): Maximum length of the tokenized sequences.
    """
    def __init__(self, code_blocks: List[str], 
                 tokenizer: torch.nn.Module, max_length: int = 2048) -> None:
        self.code_blocks = code_blocks
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Returns the number of code blocks in the dataset.
        """
        return len(self.code_blocks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Tokenize the code block at the given index.
        Returns only the input_ids.
        """
        
        # Set the padding side to the right
        self.tokenizer.padding_side = "right"
        
        encoded = self.tokenizer(self.code_blocks[idx], 
                                 return_tensors="pt", 
                                 padding="max_length", 
                                 truncation=True, 
                                 max_length=self.max_length)
        return encoded["input_ids"].squeeze(0)  



# ====== create_dataloader ====== #
def create_dataloader(code_blocks: List[str], tokenizer: torch.nn.Module, 
                      batch_size: int = 8, max_length: int = 2048) -> DataLoader:
    """
    Create a DataLoader for the given code blocks. \n
    Input:
    - code_blocks (List[str]): List of code blocks to tokenize.
    - tokenizer (torch.nn.Module): Tokenizer to use for tokenization.
    - batch_size (int): Batch size for the DataLoader.
    - max_length (int): Maximum length of the tokenized sequences. \n
    Return:
    - dataloader (DataLoader): DataLoader for the tokenized code blocks.
    """
    dataset = CodeDataset(code_blocks, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader



# ====== compute_token_ranks ====== #
def compute_token_ranks_from_dataloader(dataloader: DataLoader, model: torch.nn.Module,
                                         tokenizer: torch.nn.Module, 
                                         device: str, pad_token_id: int) -> List[List[int]]: 
    """
    Compute token ranks for batches of input sequences from a DataLoader. \n
    Input:
    - dataloader (DataLoader): DataLoader containing batches of input sequences.
    - model (torch.nn.Module): Pre-trained language model.
    - tokenizer (torch.nn.Module): Tokenizer used to encode the input.
    - device (str): Device to run the model on (e.g., 'cuda' or 'cpu').
    - pad_token_id (int): ID of the padding token. \n
    Return:
    - all_rank_list (List[List[int]]): List of ranks for each sequence in the dataloader.
    """
    # Set the model to evaluation mode
    model.eval()  
    all_rank_list = []

    with torch.no_grad():  # Disable gradient calculations
        for batch in dataloader:  
            input_ids = batch.to(device) 
            # ranks = compute_token_ranks(input_ids, model, tokenizer, device, pad_token_id)
            ranks = compute_token_ranks_parallel(input_ids, model, tokenizer, device, pad_token_id)
            # Append results of current batch
            all_rank_list.extend(ranks)  

    return all_rank_list



# ====== split_text_into_chunks ====== #
def split_text_into_chunks(text: str, tokenizer: torch.nn.Module, max_length: int) -> list:
    """
    Tokenize a text into chunks of token length `max_length`. \n
    Input:
    - text (str): The input text string to tokenize and split.
    - tokenizer: Tokenizer used to encode and decode the text.
    - max_length (int): Maximum token length allowed per chunk. \n
    Return:
    - chunk_texts (List[str]): List of text chunks (each with <= max_length tokens).
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]
    chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    return chunk_texts



# ====== preprocess_dataset ====== #
def preprocess_dataset(input_texts: list, tokenizer: torch.nn.Module, max_length: int) -> tuple[list, dict]:
    """
    Preprocess a list of texts by splitting each one into token chunks of size <= max_length. \n
    Input:
    - input_texts (List[str]): List of raw text inputs (e.g., from a dataset).
    - tokenizer: Tokenizer used for text processing.
    - max_length (int): Maximum number of tokens per chunk. \n
    Return:
    - processed_texts (List[str]): Flattened list of all text chunks.
    - mapping (Dict[int, List[int]]): Mapping from original row index to indices of its chunks
      in `processed_texts`.
    """
    processed_texts = []
    mapping = {}

    for idx, text in enumerate(input_texts):
        chunks = split_text_into_chunks(text, tokenizer, max_length)
        start_index = len(processed_texts)
        processed_texts.extend(chunks)
        mapping[idx] = list(range(start_index, len(processed_texts)))

    return processed_texts, mapping
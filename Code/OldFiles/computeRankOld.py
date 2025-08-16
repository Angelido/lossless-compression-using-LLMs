import pandas as pd
import torch
from torch.utils.data import DataLoader
from unixcoder import UniXcoder
from typing import List


# ====== compute_token_ranks_fast_old ====== #
def compute_token_ranks_fast_old(
    dataloader: DataLoader,
    model: torch.nn.Module,
    pad_token_id: int,
    device: str
) -> List[List[int]]:
    """
    Vectorized computation of token ranks for a batch of sequences. 
    This function processes the entire batch in parallel, leveraging the model's ability to compute logits for all tokens at once. 

    Inputs:
    - dataloader: DataLoader containing the input sequences.
    - model: Pre-trained language model.
    - pad_token_id: ID of the padding token.
    - device: Device to run the model on (e.g., 'cuda' or 'cpu'). 
    
    Return:
    - all_ranks: List of ranks for each sequence in the batch.
    """
    model.eval()
    all_ranks = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.to(device) # [B, L]
            # Create attention mask 1s for non-padding tokens
            attention_mask = (input_ids != pad_token_id).long().to(device)

            # Vectorized forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, L, V]

            # We need to shift the logits to align with the target tokens
            vocab_size = logits.size(-1)
            logits_input = logits[:, :-1, :]               # [B, L-1, V]
            target_ids   = input_ids[:, 1:]                # [B, L-1]

            # Compute the logits for the target tokens
            target_logits = logits_input.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)                               # [B, L-1]

            # Compute ranks
            ranks = (logits_input > target_logits.unsqueeze(-1)).sum(dim=-1)  # [B, L-1]

            # Set ranks to -1 for padding tokens
            ranks = ranks.masked_fill(target_ids == pad_token_id, -1)

            # Convert ranks to a list
            for seq_ranks, mask in zip(ranks.tolist(), attention_mask[:,1:].tolist()):
                effective_len = sum(mask)
                all_ranks.append(seq_ranks[:effective_len])

    return all_ranks



# ====== compute_token_ranks_fast_unixcoder_old ====== #
def compute_token_ranks_fast_unixcoder_old(
    dataloader: DataLoader, 
    ux: UniXcoder, 
    pad_token_id: int, 
    device: str
) -> List[List[int]]:
    """
    Vectorized computation of token ranks for a batch of sequences using UniXcoder.
    This function processes the entire batch in parallel, leveraging the model's ability to compute logits for all tokens at once.
    
    Inputs:
    - dataloader: DataLoader containing the input sequences.
    - ux: Pre-trained UniXcoder model.
    - pad_token_id: ID of the padding token.
    - device: Device to run the model on (e.g., 'cuda' or 'cpu').
    
    Return:
    - all_ranks: List of ranks for each sequence in the batch.
    """
    # Set the model to evaluation mode
    ux.eval()
    all_ranks = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.to(device)
            # Create attention mask 1s for non-padding tokens
            mask = (input_ids != pad_token_id).to(device)
            # Vectorized forward pass
            token_embs, _ = ux(input_ids)
            # logits = V x H → [B, L, V]
            logits = ux.lm_head(token_embs)
            # Obtain the logits for the target tokens
            logits_input = logits[:, :-1, :]
            target_ids  = input_ids[:, 1:]
            target_logits = logits_input.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            ranks = (logits_input > target_logits.unsqueeze(-1)).sum(-1)
            ranks = ranks.masked_fill(target_ids == pad_token_id, -1)
            for seq_ranks, m in zip(ranks.tolist(), mask[:,1:].tolist()):
                eff = sum(m)
                all_ranks.append(seq_ranks[:eff])
    return all_ranks



# ====== compute_token_ranks_fast_precision ====== #
def compute_token_ranks_fast_precision(
    dataloader: DataLoader,
    model: torch.nn.Module,
    pad_token_id: int,
    device: str
) -> List[List[int]]:
    model.eval()
    all_ranks = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.to(device)
            attention_mask = (input_ids != pad_token_id).long().to(device)

            # --- forward ---
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # FORZA float32
            logits = outputs.logits.float()           # [B, L, V]

            # shift per i target
            logits_input = logits[:, :-1, :]          # [B, L-1, V]
            target_ids   = input_ids[:, 1:]           # [B, L-1]

            # --- estrai i logit del target ---
            target_logits = logits_input.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)                             # [B, L-1]

            # --- calcolo dei rank ---
            ranks = (logits_input > target_logits.unsqueeze(-1)).sum(dim=-1)  # [B, L-1]
            ranks = ranks.masked_fill(target_ids == pad_token_id, -1)

            # raccogli in lista
            for seq_ranks, mask in zip(ranks.tolist(), attention_mask[:,1:].tolist()):
                eff_len = sum(mask)
                all_ranks.append(seq_ranks[:eff_len])

    return all_ranks
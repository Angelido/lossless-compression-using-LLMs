import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM
from unixcoder import UniXcoder
from typing import List, Dict, Tuple
import time
from tqdm import tqdm


# =====================================================
# This file provides utility functions to:
# 1) compute token ranks for each file using the 
#    selected LLM;
# 2) perform the inverse operation (convert ranks 
#    back to tokens).
# =====================================================


# ====== compute_token_ranks_fast ====== #
def compute_token_ranks_fast(
    dataloader: DataLoader,
    model: torch.nn.Module,
    pad_token_id: int,
    device: str
) -> Tuple[List[List[int]], Dict[str, float]]:
    """
    Vectorized computation of token ranks for a batch of sequences. This function processes the entire 
    batch in parallel, leveraging the model's ability to compute logits for all tokens at once. 
    
    Version without -1s. For each chunk:
    1) computes logits and obtains token ranks;
    2) immediately filters out ranks where target_ids == pad_token_id using boolean indexing.

    Input:
    - dataloader (DataLoader): DataLoader containing batches of input_ids.
    - model (torch.nn.Module): HuggingFace-like model that returns `outputs.logits`.
    - pad_token_id (int): ID used for padding tokens.
    - device (str): 'cuda' or 'cpu'.

    Return:
    - all_ranks (List[List[int]]): a list of lists of integers, with no -1 values.
    - timings (Dict[str, float]): dict with cumulative time spent per section (seconds).
    """
    model.eval()
    all_ranks: List[List[int]] = []
    
    timers = {
        "data_to_device": 0.0,
        "forward_pass": 0.0,
        "compute_target_logits": 0.0,
        "compute_ranks": 0.0,
        "filter_pad_tokens": 0.0,
    }

    with torch.no_grad():
        for batch in dataloader:
            
            start = time.perf_counter()
            
            # (1) Move batch to device
            input_ids = batch.to(device)
            
            timers["data_to_device"] += time.perf_counter() - start
            
            # (2) Forward pass: compute logits
            start = time.perf_counter()
            attention_mask = (input_ids != pad_token_id).long().to(device)  # [B, L]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, L, V]
            
            timers["forward_pass"] += time.perf_counter() - start
            
            # (3) Align logits and targets (shifted by 1)
            start = time.perf_counter()
            logits_input = logits[:, :-1, :]   # predictions for positions [0 .. L-2]
            target_ids   = input_ids[:, 1:]    # targets are shifted by one  

            # Extract the logit corresponding to the true target token
            target_logits = logits_input.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)  # [B, L-1]                
            
            timers["compute_target_logits"] += time.perf_counter() - start
            
           # (4) Compute ranks of true tokens
            start = time.perf_counter()
            B, Lm1, V = logits_input.shape
            token_ids = torch.arange(V, device=logits_input.device).view(1, 1, -1).expand(B, Lm1, V)  
            # Sort by descending logit value; break ties by token ID
            sort_keys = torch.stack([-logits_input, token_ids], dim=-1) 
            sorted_indices = sort_keys.argsort(dim=-2)[..., 1]  # [B, L-1, V]
            
            # Find the position (rank) of each target token in the sorted list
            target_ids_exp = target_ids.unsqueeze(-1)  
            ranks = (sorted_indices == target_ids_exp).nonzero(as_tuple=False)[:, -1]
            ranks = ranks.view(target_ids.shape)  # [B, L-1]
                        
            timers["compute_ranks"] += time.perf_counter() - start
            
            # (5) Remove padding tokens and collect results
            start = time.perf_counter()
            
            mask_target = (target_ids != pad_token_id)  # [B, L-1]
            valid_counts = mask_target.sum(dim=1).tolist()  # number of valid tokens per sequence
            flat_valid = ranks[mask_target]     # concatenated valid ranks
            split_ranks = torch.split(flat_valid, valid_counts)  # split back into sequences

            for seq_tensor in split_ranks:
                all_ranks.append(seq_tensor.tolist())
            
            timers["filter_pad_tokens"] += time.perf_counter() - start

    return all_ranks, timers



# ====== compute_token_ranks_fast_unixcoder ====== #
def compute_token_ranks_fast_unixcoder(
    dataloader: DataLoader,
    ux: UniXcoder,
    pad_token_id: int,
    device: str
) -> List[List[int]]:
    """
    Vectorized computation of token ranks for a batch of sequences using UniXcoder.  
    Processes the entire batch in parallel, leveraging UniXcoder’s ability to compute logits for all tokens at once.  
    This version excludes `-1` values by filtering out padding tokens.

    Input:
    - dataloader (DataLoader): Yields batches of input token IDs (shape [B, L]).
    - ux (UniXCoder): UniXcoder model returning token embeddings and exposing `.lm_head` for logits.
    - pad_token_id (int): Token ID used for padding.
    - device (str): Target device ('cuda' or 'cpu').

    Return:
    - all_ranks (List[List[int]]): A list of rank sequences, one per input sequence,  
      containing only non-padding tokens.
    - total_times (Dict[str, float]): Cumulative execution times per processing stage (seconds).
    """
    ux.eval()
    all_ranks: List[List[int]] = []
    
    total_times = {
        "inference": 0.0,
        "logits_shift": 0.0,
        "gather_logits": 0.0,
        "compute_ranks": 0.0,
        "filter_pad": 0.0,
    }

    with torch.no_grad():
        for batch in dataloader:
            
            # === Move batch to device ===
            input_ids = batch.to(device)
            
            # === Inference (causal forward pass) ===
            t0 = time.perf_counter()
            L = input_ids.size(-1)
            # Extract the causal mask (upper-left L×L block from UniXcoder's bias buffer)
            causal_mask = ux.bias[:, :L, :L].to(device)
            token_embs = ux.model(input_ids, attention_mask=causal_mask)[0]
            logits = ux.lm_head(token_embs)
            t1 = time.perf_counter()
            total_times["inference"] += t1 - t0
            
            # === Shift logits and targets (causal alignment) ===
            t0 = time.perf_counter()
            
            # Remove last position from logits and align targets from position 1 onward
            logits_input = logits[:, :-1, :]     # [B, L-1, V]
            target_ids   = input_ids[:, 1:]      # [B, L-1]
            t1 = time.perf_counter()
            total_times["logits_shift"] += t1 - t0

            # === Gather target logits ===
            t0 = time.perf_counter()
            # For each position (i, j), extract the logit of the true target token
            # target_logits[i,j] = logits_input[i,j, target_ids[i,j]]
            target_logits = logits_input.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            t1 = time.perf_counter()
            total_times["gather_logits"] += t1 - t0# [B, L-1]

            # === Compute ranks ===
            t0 = time.perf_counter()
            # Rank of each token = number of vocabulary logits greater than its target logit
            ranks = (logits_input > target_logits.unsqueeze(-1)).sum(dim=-1)  # [B, L-1]
            t1 = time.perf_counter()
            total_times["compute_ranks"] += t1 - t0
            
            # === Filter out padding tokens ===
            t0 = time.perf_counter()            
            
            # Build mask to keep only non-padding target tokens
            mask_target = (target_ids != pad_token_id)    # [B, L-1]

            # Count valid tokens per sequence
            lengths = mask_target.sum(dim=1).tolist()   # [B]

            # Flatten all valid ranks into a single 1D tensor
            flat_valid = ranks[mask_target]       # [sum(lengths)]

            # Split back into per-sequence tensors
            seq_tensors = torch.split(flat_valid, lengths)  # tuple of [len(seq)] tensors

            # Convert to Python lists and collect results
            for seq in seq_tensors:
                all_ranks.append(seq.cpu().tolist())
            
            if device == "cuda":
                torch.cuda.synchronize()
            
    return all_ranks, total_times



# ===== decode_from_ranks ===== #
def decode_token_ids_from_ranks(
    rank_sequences: List[List[int]],
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    max_length: int,
    pad_token_id: int,
    device: str = "cuda",
    debug: bool = True,
    show_progress: bool = True,
    inner_progress: bool = False
) -> Dict[int, List[int]]:
    """
    Decode token IDs from sequences of ranks without converting back to text,
    processing one sequence at a time.

    Input:
    - rank_sequences (List[List[int]]): List of rank sequences (one per chunk).
    - model (torch.nn.Module): HuggingFace-compatible language model.
    - tokenizer (torch.nn.Module): Corresponding tokenizer.
    - max_length (int): Maximum chunk length (including BOS).
    - pad_token_id (int): ID used for padding.
    - device (str): Either "cuda" or "cpu".
    - debug (bool): If True, prints detailed information for each decoding step.
    - show_progress (bool): if True shows outer tqdm over chunks.
    - inner_progress (bool): if True shows inner tqdm over steps inside each chunk.

    Return:
    - Dict[int, List[int]]: A mapping from sequence index to the list of 
        decoded token IDs (excluding BOS).
    """
    model.eval()
    bos_token_id = getattr(tokenizer, "bos_token_id", tokenizer.eos_token_id)
    decoded_token_ids: Dict[int, List[int]] = {}

    with torch.no_grad():
        
        # Outer loop with optional tqdm
        outer_iter = rank_sequences
        if show_progress:
            outer_iter = tqdm(rank_sequences, desc="Chunks", unit="chunk")
                
        for idx, rank_list in enumerate(outer_iter):
            token_ids = []
            context = [bos_token_id]

            # Inner iterator (optional tqdm)
            if show_progress and inner_progress:
                inner_iter = enumerate(tqdm(rank_list, desc=f"Steps chunk {idx}", leave=False, unit="step"))
            else:
                inner_iter = enumerate(rank_list)

            for t, rank in inner_iter:
                input_tensor = torch.tensor([context], dtype=torch.long).to(device)
                attention_mask = (input_tensor != pad_token_id).long()
                outputs = model(input_ids=input_tensor, attention_mask=attention_mask)

                logits = outputs.logits[0, -1]  # last-step logits
                
                vocab_size = logits.shape[0]
                token_ids_range = torch.arange(vocab_size, device=logits.device)

                # Stable sort: first by descending logit, then by ascending token_id
                sort_keys = torch.stack([-logits, token_ids_range], dim=1)  # [V, 2]
                sorted_indices = torch.argsort(sort_keys, dim=0)[..., 1]  
                
                predicted_token_id = sorted_indices[rank].item()

                token_ids.append(predicted_token_id)
                context.append(predicted_token_id)

                if debug:
                    print(f"[Sample {idx} - Step {t}]")
                    print(f"  Requested rank: {rank}")
                    print(f"  Predicted token ID: {predicted_token_id}")
                    print(f"  Predicted token: '{tokenizer.decode([predicted_token_id])}'")
                    
                    # Display surrounding ranks for context
                    window = 3
                    start = max(0, rank - window)
                    end = min(len(sorted_indices), rank + window + 1)

                    surrounding_ranks = list(range(start, end))
                    surrounding_token_ids = sorted_indices[start:end].tolist()
                    surrounding_tokens = [tokenizer.decode([tid]) for tid in surrounding_token_ids]
                    surrounding_logits = logits[sorted_indices[start:end]].tolist()

                    print("  Tokens around requested rank (±3):")
                    for i, r in enumerate(surrounding_ranks):
                        tag = " <==" if r == rank else ""
                        print(f"    Rank {r:>4}: ID {surrounding_token_ids[i]:>6} → '{surrounding_tokens[i]}' | Logit: {surrounding_logits[i]:>10.4f}{tag}")
                    print()

                if len(context) == max_length:
                    # Reset context for the next chunk (starting with BOS)
                    context = [bos_token_id]  

            decoded_token_ids[idx] = token_ids

    return decoded_token_ids



# ===================================================================================================
# ================================ OLD FUNCTIONS - NOT USED =========================================
# ===================================================================================================



# ====== compute_token_ranks_fast_old ====== #
def compute_token_ranks_fast_old(
    dataloader: DataLoader,
    model: torch.nn.Module,
    pad_token_id: int,
    device: str
) -> List[List[int]]:
    """
    Vectorized computation of token ranks for a batch of sequences. This function processes the entire 
    batch in parallel, leveraging the model's ability to compute logits for all tokens at once. 
    
    Version without -1s. For each chunk:
    1) computes logits and obtains token ranks;
    2) immediately filters out ranks where target_ids == pad_token_id using boolean indexing.

    Input:
    - dataloader (DataLoader): DataLoader containing batches of input_ids.
    - model (torch.nn.Module): HuggingFace-like model that returns `outputs.logits`.
    - pad_token_id (int): ID used for padding tokens.
    - device (str): 'cuda' or 'cpu'.

    Return:
    - all_ranks (List[List[int]]): a list of lists of integers, with no -1 values.
    """
    model.eval()
    all_ranks: List[List[int]] = []

    with torch.no_grad():
        for batch in dataloader:
            # batch: tensor of shape [B, L]
            input_ids = batch.to(device)

            # Build attention mask (1 = real token, 0 = padding)
            attention_mask = (input_ids != pad_token_id).long().to(device)  # [B, L]

            # Forward pass: compute logits [B, L, V]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, L, V]

            # SHIFT: align logits and target_ids
            #   logits_input[i] = logits[i, :-1, :]  →   [B, L-1, V]
            logits_input = logits[:, :-1, :]   
            #   target_ids[i]   = input_ids[i, 1:]   →   [B, L-1]   
            target_ids   = input_ids[:, 1:]      

            # Extract the “true” logit at each position j:
            #   target_logits[i,j] = logits_input[i,j, target_ids[i,j]]
            target_logits = logits_input.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)                       # [B, L-1]

            # Compute ranks: for each token j, count how many vocab logits > target_logit
            #   ranks[i,j] = number of entries in the vocab where logit > target_logits[i,j]
            ranks = (logits_input > target_logits.unsqueeze(-1)).sum(dim=-1)  # [B, L-1]

            # Boolean mask to select only valid targets (non-pad tokens)
            #   mask_target[i,j] = True if target_ids[i,j] != pad_token_id
            mask_target = (target_ids != pad_token_id)  # [B, L-1], dtype=torch.bool

            # For each row in ranks, select only the values where mask_target is True,
            # effectively removing any ranks associated with pad tokens
            for i in range(ranks.size(0)):
                seq_ranks = ranks[i]        # Tensor of shape [L-1]
                seq_mask  = mask_target[i]  # Boolean tensor [L-1]

                # valid_ranks is a list of integers corresponding to valid token ranks
                valid_ranks = seq_ranks[seq_mask].tolist()
                all_ranks.append(valid_ranks)

    return all_ranks


# ====== compute_token_ranks_fast_unixcoder ====== #
def compute_token_ranks_fast_unixcoder_old(
    dataloader: DataLoader,
    ux: UniXcoder,
    pad_token_id: int,
    device: str
) -> List[List[int]]:
    """
    Vectorized computation of token ranks for a batch of sequences using UniXcoder.
    This function processes the entire batch in parallel, leveraging the model's ability to compute logits for all tokens at once.
    Version without -1s.

    Input:
    - dataloader (DataLoader): DataLoader yielding batches of input token IDs (shape [B, L]).
    - ux (UniXCoder): UniXcoder model returning token embeddings and supporting `.lm_head` for logits.
    - pad_token_id (int): Integer ID used to indicate padding tokens in the sequences.
    - device (str): Device identifier string, either 'cuda' or 'cpu'.

    Return:
    - all_ranks (List[List[int]]): A list of lists, where each sublist contains the rank values (integers)
      of the non-pad tokens in the corresponding input sequence.
    """
    ux.eval()
    all_ranks: List[List[int]] = []

    with torch.no_grad():
        for batch in dataloader:
            # batch: tensor [B, L]
            input_ids = batch.to(device)

            # Forward pass to compute logits over the entire sequence
            token_embs, _ = ux(input_ids)       # token_embs: [B, L, hidden_dim]
            logits = ux.lm_head(token_embs)      # logits: [B, L, vocab_size]

            # Shift for comparison: remove the last position from logits
            # and take the targets from position 1 onward
            logits_input = logits[:, :-1, :]     # [B, L-1, V]
            target_ids   = input_ids[:, 1:]      # [B, L-1]

            # Extract the "true" logits for each target token
            # target_logits[i,j] = logits_input[i,j, target_ids[i,j]]
            target_logits = logits_input.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)                       # [B, L-1]

            # Compute ranks: count how many logits are greater than the target logits
            # ranks[i,j] = number of vocab entries with logit > target_logits[i,j]
            ranks = (logits_input > target_logits.unsqueeze(-1)).sum(dim=-1)  # [B, L-1]

            # Build a boolean mask over valid (non-pad) target tokens
            # mask_target[i,j] = True if target_ids[i,j] != pad_token_id
            mask_target = (target_ids != pad_token_id)   # [B, L-1], dtype=torch.bool

            # Immediately filter: for each i-th sequence,
            # take only the ranks corresponding to True in mask_target[i]
            for i in range(ranks.size(0)):
                seq_ranks = ranks[i]         # Tensor of shape [L-1]
                seq_mask  = mask_target[i]   # Tensor bool [L-1]
                
                # valid_ranks contains only ranks of non-pad tokens
                valid_ranks = seq_ranks[seq_mask].tolist()
                all_ranks.append(valid_ranks)

    return all_ranks


# ===== compute_token_ranks_fast_seq2seq ====== #
def compute_token_ranks_fast_seq2seq(
    dataloader: DataLoader,
    model: AutoModelForSeq2SeqLM,
    pad_token_id: int,
    device: str
) -> List[List[int]]:
    """
    Compute token ranks for an encoder-decoder (Seq2Seq) model in a vectorized manner.

    Input:
    - dataloader (DataLoader): Batch loader of input sequences (token IDs).
    - model (AutoModelForSeq2SeqLM): Pretrained Seq2Seq model (e.g., CodeT5).
    - pad_token_id (int): Token ID used for padding.
    - device (str): Device to run inference on ('cuda' or 'cpu').

    Return:
    - List[List[int]]: List of rank sequences, one per input sequence.
    """
    # Set model to evaluation mode
    model.eval()
    all_ranks: List[List[int]] = []

    # No gradient calculation needed
    with torch.no_grad():
        for batch in dataloader:
            # Move input IDs to the specified device: shape [B, L]
            input_ids = batch.to(device)
            # Create attention mask: 1 for tokens != pad, 0 for pad
            attention_mask = (input_ids != pad_token_id).long().to(device)

            # ----------------- ENCODER -----------------
            # Encode all input tokens at once
            encoder_outputs = model.get_encoder()(  
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # ----------------- DECODER PREP -----------------
            # Prepare decoder inputs by shifting right: drop the last token
            decoder_input_ids = input_ids[:, :-1]             # shape [B, L-1]
            # Mask for decoder inputs (exclude pad tokens)
            decoder_attention_mask = (decoder_input_ids != pad_token_id).long()

            # ----------------- DECODER FORWARD -----------------
            # Run decoder to get logits over vocabulary for each position
            outputs = model(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            # Logits shape: [B, L-1, V]
            logits = outputs.logits

            # ----------------- RANK COMPUTATION -----------------
            # The target tokens are original inputs shifted left: drop the first token
            target_ids = input_ids[:, 1:]
            # Extract logits corresponding to the actual next-token targets
            target_logits = logits.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)  # shape [B, L-1]

            # For each position, count how many vocabulary logits exceed the target logit
            ranks = (logits > target_logits.unsqueeze(-1)).sum(dim=-1)  # shape [B, L-1]
            # Mark padding positions with -1 to ignore them
            ranks = ranks.masked_fill(target_ids == pad_token_id, -1)

            # ----------------- COLLECT RESULTS -----------------
            # Convert each sequence to Python list and truncate to actual length
            for seq_ranks, mask in zip(ranks.tolist(), decoder_attention_mask.tolist()):
                effective_length = sum(mask)
                all_ranks.append(seq_ranks[:effective_length])

    return all_ranks
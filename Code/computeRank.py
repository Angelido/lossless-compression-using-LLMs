import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax
from transformers import AutoModelForSeq2SeqLM
from unixcoder import UniXcoder
from typing import List, Tuple, Dict



# ====== compute_token_ranks_fast ====== #
def compute_token_ranks_fast(
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



# ====== compute_token_ranks_fast_unixcoder ====== #
def compute_token_ranks_fast_unixcoder(
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


# ===== compute_token_ranks_fast_seq2seq ====== #
def compute_token_ranks_fast_seq2seq(
    dataloader: DataLoader,
    model: AutoModelForSeq2SeqLM,
    pad_token_id: int,
    device: str
) -> List[List[int]]:
    """
    Compute token ranks for an encoder-decoder (Seq2Seq) model in a vectorized manner.

    Inputs:
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


def reconstruct_texts_from_rank_lists(
    sorted_rank_lists: List[List[int]],
    model,
    tokenizer,
    batch_size: int,
    max_length: int,
    device: str = "cuda"
) -> Dict[int, str]:
    """
    Vectorized reconstruction dei testi a partire dalle liste di rank:
    - Elabora in batch di dimensione batch_size
    - Per ogni passo temporale, genera in parallelo i token per tutto il batch
    - Reset del contesto a [BOS] quando raggiunge max_length
    """
    model.eval()
    reconstructed: Dict[int, str] = {}
    bos_id = tokenizer.bos_token_id
    total = len(sorted_rank_lists)

    # Processa batch per batch
    for batch_start in range(0, total, batch_size):
        batch = sorted_rank_lists[batch_start:batch_start + batch_size]
        bsz = len(batch)

        # Lunghezze delle sequenze di rank e massima lunghezza in questo batch
        lengths = torch.tensor([len(r) for r in batch], device=device)
        max_seq = int(lengths.max().item())

        # Crea tensor di rank padded: [bsz, max_seq]
        ranks_tensor = torch.zeros((bsz, max_seq), dtype=torch.long, device=device)
        for i, r in enumerate(batch):
            ranks_tensor[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)

        # Inizializza contesto e posizioni
        context = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)
        positions = torch.zeros(bsz, dtype=torch.long, device=device)

        # Accumulatore per token generati
        generated = torch.full((bsz, max_seq), bos_id, dtype=torch.long, device=device)

        # Itera passo a passo lungo la dimensione max_seq
        for t in range(max_seq):
            # Forward sul contesto corrente
            outputs = model(input_ids=context)
            last_logits = outputs.logits[:, -1, :]  # [bsz, V]
            probs = softmax(last_logits, dim=-1)    # [bsz, V]

            # Estrai i rank in questo passo per ciascuna sequenza
            current_ranks = ranks_tensor[:, t]     # [bsz]
            active = (t < lengths)                 # mask booleano [bsz]

            # Calcola topk una sola volta fino al massimo rank attivo
            R_max = int(current_ranks.max().item())
            topk_vals, topk_inds = torch.topk(probs, k=R_max + 1, dim=-1)  # [bsz, R_max+1]

            # Ottieni next_tokens vectorizzato: [bsz]
            next_tokens = topk_inds[torch.arange(bsz, device=device), current_ranks]
            # Per le posizioni non attive, rimani BOS
            next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, bos_id))

            # Appendi al contesto e al generato
            context = torch.cat([context, next_tokens.unsqueeze(1)], dim=1)
            generated[:, t] = next_tokens

            # Reset contesto se supera max_length
            if context.size(1) >= max_length:
                context = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)

        # Decode batch e salva
        for i in range(bsz):
            orig_idx = batch_start + i
            # Prendi solo fino a lengths[i]
            token_seq = generated[i, :lengths[i]].tolist()
            text = tokenizer.decode(token_seq, skip_special_tokens=True)
            reconstructed[orig_idx] = text

    return reconstructed


def reconstruct_texts_from_rank_lists_precision(
    sorted_rank_lists: List[List[int]],
    model,
    tokenizer,
    batch_size: int,
    max_length: int,
    device: str = "cuda"
) -> Dict[int, str]:
    model.eval()
    reconstructed: Dict[int, str] = {}
    bos_id = tokenizer.bos_token_id
    total = len(sorted_rank_lists)

    for batch_start in range(0, total, batch_size):
        batch = sorted_rank_lists[batch_start:batch_start + batch_size]
        bsz = len(batch)
        lengths = torch.tensor([len(r) for r in batch], device=device)
        max_seq = int(lengths.max().item())

        ranks_tensor = torch.zeros((bsz, max_seq), dtype=torch.long, device=device)
        for i, r in enumerate(batch):
            ranks_tensor[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)

        # inizio contesto
        context = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)
        generated = torch.full((bsz, max_seq), bos_id, dtype=torch.long, device=device)

        for t in range(max_seq):
            # --- forward e cast logits a float32 ---
            outputs = model(input_ids=context)
            last_logits = outputs.logits[:, -1, :].float()  # [bsz, V]
            probs = softmax(last_logits, dim=-1)            # [bsz, V]

            current_ranks = ranks_tensor[:, t]               # [bsz]
            active = (t < lengths)                           # [bsz]

            # --- ORDINA TUTTO il vocabolario (argsort) invece di topk ---
            sorted_inds = torch.argsort(probs, dim=-1, descending=True)  # [bsz, V]
            next_tokens = sorted_inds[torch.arange(bsz, device=device), current_ranks]

            # per posizioni inattive
            next_tokens = torch.where(active,
                                      next_tokens,
                                      torch.full_like(next_tokens, bos_id))

            context = torch.cat([context, next_tokens.unsqueeze(1)], dim=1)
            generated[:, t] = next_tokens

            if context.size(1) >= max_length:
                context = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)

        # decode
        for i in range(bsz):
            orig_idx = batch_start + i
            token_seq = generated[i, :lengths[i]].tolist()
            text = tokenizer.decode(token_seq, skip_special_tokens=True)
            reconstructed[orig_idx] = text

    return reconstructed
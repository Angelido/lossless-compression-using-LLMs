from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm
import math
import torch
import time



#===== compute_entropy ====== #
def compute_entropy(
    dataloader: DataLoader,
    model: torch.nn.Module,
    pad_token_id: int,
    device: str,
    show_progress: bool = True,
    clamp_min_log: float = -1e6,
    debug: bool = False
) -> Tuple[float, int, dict]:
    """
    Computes the average (true) Shannon entropy over the tokens predicted by the model:
        total_bits = sum_{t} sum_{k} -p(k|context_t) log2 p(k|context_t) 
    Excludes pad tokens.
    
    Input:
      - dataloader (DataLoader): DataLoader containing the input sequences.
      - model (torch.nn.Module): Pre-trained language model.
      - pad_token_id (int): ID of the padding token.
      - device (str): Device to run the model on (e.g., 'cuda' or 'cpu').
      - show_progress (bool): Whether to display a progress bar during computation.
      - clamp_min_log (float): Minimum log value for numerical stability (default: -1e6).
      - debug (bool): Whether to enable debug mode for additional logging/printing.
    
    Return:
      - total_bits (float): Total number of bits (in bits) summed over all predicted tokens.
      - total_tokens (int): Number of predicted tokens considered (excluding pad tokens).
      - timings (dict): Dictionary with cumulative timing information for profiling, with keys:
          * "data_to_device"  (float): Total time spent moving batches to the device.
          * "forward_pass"    (float): Total time spent in the model forward pass.
          * "logprob_compute" (float): Total time spent computing log-probabilities.
          * "accumulate"      (float): Total time spent accumulating results.
    """
    model.eval()
    total_bits = 0.0
    total_tokens = 0
    ln2 = math.log(2.0)

    timings = {"data_to_device": 0.0, "forward_pass": 0.0, "logprob_compute": 0.0, "accumulate": 0.0}
    iterator = tqdm(dataloader, desc="Computing entropy", unit="batch") if show_progress else dataloader

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            t0 = time.perf_counter()
            input_ids = batch.to(device)  # [B, L]
            timings["data_to_device"] += time.perf_counter() - t0

            # Forward pass
            t1 = time.perf_counter()
            attention_mask = (input_ids != pad_token_id).long().to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, L, V]
            timings["forward_pass"] += time.perf_counter() - t1

            # Shift to predict the next token
            logits_input = logits[:, :-1, :]  # [B, L-1, V]
            target_ids = input_ids[:, 1:]     # [B, L-1]

            if logits_input.dtype == torch.float16:
                logits_input = logits_input.float()

            # Numerically stable log-probabilities
            t2 = time.perf_counter()
            log_probs = torch.log_softmax(logits_input, dim=-1)  # [B, L-1, V]
            probs = log_probs.exp()
            timings["logprob_compute"] += time.perf_counter() - t2

            # Mask valid tokens
            mask_target = (target_ids != pad_token_id)  # [B, L-1]

            # Compute true entropy: -sum p log2 p
            entropy_bits = -(probs * log_probs / ln2).sum(dim=-1)  # [B, L-1]

            # Accumulate only over valid tokens
            tacc = time.perf_counter()
            valid_bits = entropy_bits[mask_target]
            if valid_bits.numel() > 0:
                total_bits += float(valid_bits.double().sum().cpu().item())
            total_tokens += int(mask_target.sum().item())
            timings["accumulate"] += time.perf_counter() - tacc

    return total_bits, total_tokens, timings



#===== compute_entropy_unixcoder ======#
def compute_entropy_unixcoder(
    dataloader: DataLoader,
    ux,
    pad_token_id: int,
    device: str,
    show_progress: bool = True,
    clamp_min_log: float = -1e6,
) -> Tuple[float, int, dict]:
    """
    Computes the average (true) Shannon entropy over the tokens predicted by UniXcoder:
        total_bits = sum_{t} sum_{k} -p(k|context_t) log2 p(k|context_t)
    Excludes padding tokens from the computation.

    Input:
      - dataloader (DataLoader): DataLoader containing the input sequences.
      - ux: UniXcoder model object (must provide `.bias`, `.model()`, and `.lm_head()`).
      - pad_token_id (int): ID of the padding token.
      - device (str): Device to run the model on (e.g., 'cuda' or 'cpu').
      - show_progress (bool): Whether to display a progress bar during computation.
      - clamp_min_log (float): Minimum log value for numerical stability (default: -1e6).

    Return:
      - total_bits (float): Total number of bits (in bits) summed over all predicted tokens.
      - total_tokens (int): Number of predicted tokens considered (excluding pad tokens).
      - timings (dict): Dictionary with cumulative timing information for profiling, with keys:
          * "data_to_device"  (float): Total time spent moving batches to the device.
          * "inference"       (float): Total time spent in model forward passes.
          * "logprob_compute" (float): Total time spent computing log-probabilities.
          * "accumulate"      (float): Total time spent accumulating results.
    """
    ux.eval()
    total_bits = 0.0
    total_tokens = 0
    ln2 = math.log(2.0)

    timings = {
        "data_to_device": 0.0,
        "inference": 0.0,
        "logprob_compute": 0.0,
        "accumulate": 0.0
    }

    iterator = tqdm(dataloader, desc="Computing entropy UniXcoder", unit="batch") if show_progress else dataloader

    with torch.no_grad():
        for batch in iterator:
            # Move inputs to the target device
            t0 = time.perf_counter()
            input_ids = batch.to(device)  # [B, L]
            timings["data_to_device"] += time.perf_counter() - t0

            # Forward pass with causal bias
            t1 = time.perf_counter()
            L = input_ids.size(-1)
            causal_mask = ux.bias[:, :L, :L].to(device)  # [1, L, L]
            token_embs = ux.model(input_ids, attention_mask=causal_mask)[0]
            logits = ux.lm_head(token_embs)  # [B, L, V]
            timings["inference"] += time.perf_counter() - t1

            # Shift logits & targets for next-token prediction
            logits_input = logits[:, :-1, :]  # [B, L-1, V]
            target_ids   = input_ids[:, 1:]   # [B, L-1]

            if logits_input.dtype == torch.float16:
                logits_input = logits_input.float()

            # Compute log-probabilities & probabilities
            t2 = time.perf_counter()
            log_probs = torch.log_softmax(logits_input, dim=-1)  # [B, L-1, V]
            probs = log_probs.exp()
            timings["logprob_compute"] += time.perf_counter() - t2

            # Mask out padding tokens
            mask_target = (target_ids != pad_token_id)  # [B, L-1]

            # Compute entropy: -sum p log2 p
            entropy_bits = -(probs * log_probs / ln2).sum(dim=-1)  # [B, L-1]

            # Accumulate results over valid tokens
            t3 = time.perf_counter()
            valid_bits = entropy_bits[mask_target]
            if valid_bits.numel() > 0:
                total_bits += float(valid_bits.double().sum().cpu().item())
            total_tokens += int(mask_target.sum().item())
            timings["accumulate"] += time.perf_counter() - t3

    return total_bits, total_tokens, timings



#===== compute_cross_entropy ====== #
def compute_cross_entropy(
    dataloader: DataLoader,
    model: torch.nn.Module,
    pad_token_id: int,
    device: str,
    show_progress: bool = True,
    clamp_min_log: float = -1e6,   
    debug: bool = False
) -> Tuple[float, int, dict]:
    """
    Computes the total number of bits required to encode the text according to the model:
        total_bits = sum_t -log2 p_model(w_t | context_t)
    and the total number of valid tokens N (excluding padding).

    Input:
      - dataloader (DataLoader): DataLoader containing the input sequences.
      - model (torch.nn.Module): Pre-trained language model.
      - pad_token_id (int): ID of the padding token.
      - device (str): Device to run the model on (e.g., 'cuda' or 'cpu').
      - show_progress (bool): Whether to display a progress bar during computation.
      - clamp_min_log (float or None): Minimum allowed log-probability value for stability.
                                        If None, non-finite values raise an error (default: -1e6).
      - debug (bool): Whether to enable debug output for diagnosing issues with target IDs or probabilities.

    Return:
      - total_bits (float): Sum of bits (in bits) over all valid predicted tokens.
      - total_tokens (int): Number of valid (non-padding) tokens.
      - timings (dict): Dictionary with cumulative timings for profiling, with keys:
          * "data_to_device"  (float): Time spent moving data to the device.
          * "forward_pass"    (float): Time spent in model forward passes.
          * "logprob_compute" (float): Time spent computing log-probabilities.
          * "accumulate"      (float): Time spent accumulating results.
          * "diagnostics"     (float): Time spent in NaN/Inf checks and clamping.

    """
    model.eval()
    total_bits = 0.0
    total_tokens = 0
    ln2 = math.log(2.0)

    timings = {"data_to_device": 0.0, "forward_pass": 0.0, "logprob_compute": 0.0, "accumulate": 0.0}
    iterator = dataloader
    if show_progress:
        try:
            iterator = tqdm(dataloader, desc="Computing cross-entropy", unit="batch")
        except Exception:
            iterator = dataloader

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            # Move inputs to the target device
            t0 = time.perf_counter()
            input_ids = batch.to(device)                 # [B, L]
            timings["data_to_device"] += time.perf_counter() - t0

            # Forward pass
            t1 = time.perf_counter()
            attention_mask = (input_ids != pad_token_id).long().to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits                        # [B, L, V]
            timings["forward_pass"] += time.perf_counter() - t1

            # Shift logits & targets for next-token prediction
            logits_input = logits[:, :-1, :]               # [B, L-1, V]
            target_ids   = input_ids[:, 1:]                # [B, L-1]

            # Convert to float32 if logits are in float16 for stability
            if logits_input.dtype == torch.float16:
                logits_input = logits_input.float()

            # Compute log-probabilities (numerically stable)
            t2 = time.perf_counter()
            log_probs = torch.log_softmax(logits_input, dim=-1)  # [B, L-1, V]
            timings["logprob_compute"] += time.perf_counter() - t2

            # Extract log-probabilities of the observed tokens
            target_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

            # Mask valid tokens (exclude padding)
            mask_target = (target_ids != pad_token_id)  # [B, L-1] bool

            # Debug check: tokens out of vocabulary?
            V = log_probs.size(-1)
            if debug:
                if int(target_ids.max()) >= V:
                    print(f"[DEBUG] target_id >= V (vocab size) in batch {batch_idx} -> max_target_id={int(target_ids.max())}, V={V}")

            # Handle non-finite log-probabilities
            tdiag = time.perf_counter()
            bad_mask = ~torch.isfinite(target_log_probs)  # True where NaN or +/-inf
            if bad_mask.any():
                if clamp_min_log is not None:
                    if debug:
                        idx = torch.nonzero(bad_mask, as_tuple=False)[0]
                        b_i, pos = int(idx[0].item()), int(idx[1].item())
                        tok = int(target_ids[b_i, pos].item())
                        print(f"[DEBUG] Non-finite target_log_prob at batch {batch_idx}, sample {b_i}, pos {pos}, token_id={tok}. Replacing with clamp_min_log={clamp_min_log}.")
                    target_log_probs = torch.where(bad_mask, torch.full_like(target_log_probs, clamp_min_log), target_log_probs)
                else:
                    raise RuntimeError("Non-finite target_log_probs encountered and clamp_min_log is None. Enable clamp or debug.")

            # Optional safety clamp
            if clamp_min_log is not None:
                target_log_probs = torch.clamp(target_log_probs, min=clamp_min_log)

            # Bits per observed token: -log2 p = -log_prob / ln2
            bits_per_token = - target_log_probs / ln2  # [B, L-1]

            # Accumulate over valid tokens
            tacc = time.perf_counter()
            valid_bits = bits_per_token[mask_target]    # 1D tensor
            if valid_bits.numel() > 0:
                total_bits += float(valid_bits.double().sum().cpu().item())
            total_tokens += int(mask_target.sum().item())
            timings["accumulate"] += time.perf_counter() - tacc
            timings["diagnostics"] = time.perf_counter() - tdiag

    return total_bits, total_tokens, timings



#===== compute_cross_entropy_unixcoder ====== #
def compute_cross_entropy_unixcoder(
    dataloader: DataLoader,
    ux,
    pad_token_id: int,
    device: str,
    clamp_min_log: float = -1e6,  # evita -inf
    show_progress: bool = True
) -> Tuple[float, int, dict]:
    """
    Computes the cross-entropy (in bits) using UniXcoder.

    Inputs:
      - dataloader (DataLoader): DataLoader containing the input sequences.
      - ux: UniXcoder model object (must provide `.bias`, `.model()`, and `.lm_head()`).
      - pad_token_id (int): ID of the padding token.
      - device (str): Device to run the model on (e.g., 'cuda' or 'cpu').
      - clamp_min_log (float or None): Minimum log-probability value to avoid -inf.
                                        If None, non-finite values will raise an error (default: -1e6).
      - show_progress (bool): Whether to display a progress bar during computation.

    Returns:
      - total_bits (float): Sum of `-log2(p)` over all valid tokens.
      - total_tokens (int): Number of non-padding tokens.
      - timings (dict): Dictionary with cumulative timings for profiling, with keys:
          * "data_to_device"  (float): Time spent moving data to the device.
          * "inference"       (float): Time spent in model forward passes.
          * "logprob_compute" (float): Time spent computing log-probabilities.
          * "accumulate"      (float): Time spent accumulating results.
    """
    ux.eval()
    total_bits = 0.0
    total_tokens = 0
    ln2 = math.log(2.0)

    timings = {
        "inference": 0.0,
        "logprob_compute": 0.0,
        "accumulate": 0.0
    }

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Computing cross-entropy UniXcoder", unit="batch")

    with torch.no_grad():
        for batch in iterator:
            # Move inputs to the target device
            t0 = time.perf_counter()
            input_ids = batch.to(device)
            timings["data_to_device"] = time.perf_counter() - t0

            # Inference
            t0 = time.perf_counter()
            L = input_ids.size(-1)
            causal_mask = ux.bias[:, :L, :L].to(device)
            token_embs = ux.model(input_ids, attention_mask=causal_mask)[0]
            logits = ux.lm_head(token_embs)  # [B, L, V]
            timings["inference"] += time.perf_counter() - t0

            # Shift logits & targets for next-token prediction
            logits_input = logits[:, :-1, :]   # [B, L-1, V]
            target_ids   = input_ids[:, 1:]    # [B, L-1]

            if logits_input.dtype == torch.float16:
                logits_input = logits_input.float()

            # Compute log-probabilities
            t0 = time.perf_counter()
            log_probs = torch.log_softmax(logits_input, dim=-1)  # [B, L-1, V]
            timings["logprob_compute"] += time.perf_counter() - t0

            # Gather log-probabilities of the observed tokens
            target_log_probs = log_probs.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)  # [B, L-1]

            # Handle non-finite values
            bad_mask = ~torch.isfinite(target_log_probs)
            if bad_mask.any():
                if clamp_min_log is not None:
                    target_log_probs = torch.where(
                        bad_mask,
                        torch.full_like(target_log_probs, clamp_min_log),
                        target_log_probs
                    )
                else:
                    raise RuntimeError("Non-finite log-probs found.")

            if clamp_min_log is not None:
                target_log_probs = torch.clamp(target_log_probs, min=clamp_min_log)

            # Bits per token 
            bits_per_token = - target_log_probs / ln2

            # Mask out padding tokens
            mask_target = (target_ids != pad_token_id)
            valid_bits = bits_per_token[mask_target]

            # Accumulate over valid tokens
            t0 = time.perf_counter()
            if valid_bits.numel() > 0:
                total_bits += float(valid_bits.double().sum().cpu().item())
            total_tokens += int(mask_target.sum().item())
            timings["accumulate"] += time.perf_counter() - t0

    return total_bits, total_tokens, timings
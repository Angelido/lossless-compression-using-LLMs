import torch
import time
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM
from unixcoder import UniXcoder
from typing import List, Dict, Tuple, Optional
from torch.nn import sdpa_kernel, SDPBackend
from tqdm import tqdm


# =====================================================
# This file provides utility functions to:
# (1) compute token ranks for each file using the 
#     selected LLM;
# (2) perform the inverse operation (convert ranks 
#     back to tokens).
# =====================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (1) Compute token ranks for each file using the selected LLM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ====== compute_token_ranks_topk_fast ====== #
@torch.no_grad()
def compute_token_ranks_topk_fast(
    dataloader: DataLoader,
    model: torch.nn.Module,
    pad_token_id: int,
    device: str,
    topk: int = 10,
    *,
    # tie handling (OFF if tie_eps_abs == 0)
    tie_as_exception: bool = True,
    tie_eps_abs: float = 0.0,   # e.g., 0.03 for 4-bit; 0 disables tie handling
) -> Tuple[List[List[int]], List[List[int]], Dict[str, float]]:
    """
    Compute target-token ranks using top-k extraction with a stable ordering.

    For each target token, we extract the top-k logits per position and
    determine the rank of the true target within those k candidates.We mark as 
    exceptions (rank=0) targets that are outside the top-k. We enforce a 
    deterministic, stable order among ties by breaking ties with the token id (ascending).

    Optional (but recommended) soft-tie handling: if the target is inside top-k and
    its nearest neighbor among the top-k is within `tie_eps_abs` the position
    is marked as an "exception" and the encoded rank is set to 0.

    Input:
    - dataloader (DataLoader):
        Batch loader of input sequences (token IDs).
    - model (torch.nn.Module):
        Pretrained language model (e.g., GPT-2, LLaMA) with a `.logits` output.
    - pad_token_id (int):
        Token ID used for padding.
    - device (str):
        Device to run inference on ('cuda' or 'cpu').
    - topk (int):
        Number of top logits to consider per position (must be > 0).
    - tie_as_exception (bool):
        If True, positions with soft ties are marked as exceptions (rank=0).
    - tie_eps_abs (float):
        Absolute logit difference threshold for soft-tie detection. If 0.0, tie handling is disabled.

    Return:
    - all_ranks (List[List[int]]):
        Per-sequence ranks of non-pad target tokens. Values are in {0} ∪ [1..K],
        where 0 indicates an exception due to soft-tie handling or target not in top-k
        under the soft-tie rule.
    - all_exceptions (List[List[int]]):
        Per-sequence lists of target token IDs where rank == 0 (exceptions).
    - timers (Dict[str, float]):
        Accumulated wall-clock timings (seconds) for each processing stage:
        "data_to_device", "forward_pass", "compute_target_logits",
        "topk_and_ranks", and "filter_and_split".
    """
    assert topk > 0, "topk must be > 0"
    model.eval()

    # Ensure deterministic numeric path (disable TF32 on CUDA)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    all_ranks: List[List[int]] = []
    all_exceptions: List[List[int]] = []

    # Timing buckets aligned with the original function
    timers = {
        "data_to_device": 0.0,
        "forward_pass": 0.0,
        "compute_target_logits": 0.0,
        "topk_and_ranks": 0.0,
        "filter_and_split": 0.0,
    }

    for batch in dataloader:
        # --- (1) Move batch to target device
        start = time.perf_counter()
        input_ids = batch.to(device)                         # [B, L]
        timers["data_to_device"] += time.perf_counter() - start

        # --- (2) Forward pass with attention mask (SDPA MATH)
        start = time.perf_counter()
        attention_mask = (input_ids != pad_token_id).long()  # [B, L]
        # Use pure MATH backend for SDPA to keep behavior consistent across runs
        with sdpa_kernel(SDPBackend.MATH):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits                              # [B, L, V]
        timers["forward_pass"] += time.perf_counter() - start

        # --- (3) Shift logits/targets for next-token prediction
        start = time.perf_counter()
        logits_input = logits[:, :-1, :]   # [B, L-1, V]
        target_ids   = input_ids[:, 1:]    # [B, L-1]
        timers["compute_target_logits"] += time.perf_counter() - start

        # --- (4) Stable top-k + rank/exception (+ optional soft-tie handling)
        start = time.perf_counter()
        topk_vals, topk_idx = torch.topk(logits_input, k=topk, dim=-1)  # [B, L-1, K]
        V = logits_input.size(-1)

        # Build stable sorting keys: descending by value, then ascending by token id.
        # Implemented via keys = -value + small epsilon(token_id) so that ties break on id.
        eps = (topk_idx.float() / max(V, 1)) * 1e-6                      # [B, L-1, K]
        keys = (-topk_vals).float() + eps                                # [B, L-1, K]
        order = torch.argsort(keys, dim=-1)                              # asc -> value desc, id asc
        stable_topk_idx  = torch.gather(topk_idx,  dim=-1, index=order)  # [B, L-1, K]
        stable_topk_vals = torch.gather(topk_vals, dim=-1, index=order)  # [B, L-1, K]

        K = stable_topk_idx.shape[-1]
        target_exp = target_ids.unsqueeze(-1).expand(-1, -1, K)          # [B, L-1, K]
        match = (stable_topk_idx == target_exp)                           # [B, L-1, K]
        in_topk = match.any(dim=-1)                                       # [B, L-1]
        pos = match.float().argmax(dim=-1).long()                         # [B, L-1], position in [0..K-1]

        # Encode ranks as 1..K for targets in top-k, else 0
        ranks_encoded = torch.where(in_topk, pos + 1, torch.zeros_like(pos))  # [B, L-1]

        # ---- Soft-tie → Exception (optional)
        if tie_as_exception and tie_eps_abs > 0.0:
            # Value at the matched position (for in-topk cases)
            pos_clamped = pos.clamp(min=0, max=K - 1)
            val_pos = torch.gather(stable_topk_vals, -1, pos_clamped.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

            # Neighbor values to measure the smallest gap around the matched position.
            BIG = torch.tensor(torch.finfo(stable_topk_vals.dtype).max,
                               device=stable_topk_vals.device)
            big_like = torch.full_like(val_pos, BIG)

            has_left  = pos > 0
            left_idx  = (pos - 1).clamp(min=0)
            left_g    = torch.gather(stable_topk_vals, -1, left_idx.unsqueeze(-1)).squeeze(-1)
            left_val  = torch.where(has_left, left_g, big_like)

            has_right = pos < (K - 1)
            right_idx = (pos + 1).clamp(max=K - 1)
            right_g   = torch.gather(stable_topk_vals, -1, right_idx.unsqueeze(-1)).squeeze(-1)
            right_val = torch.where(has_right, right_g, big_like)

            # Minimum neighbor gap inside top-k
            min_neighbor_diff = torch.minimum((val_pos - left_val).abs(),
                                              (val_pos - right_val).abs())

            tie_in_topk = in_topk & (min_neighbor_diff <= tie_eps_abs)

            # Border case: target outside top-k but close to the K-th logit
            kth_vals     = stable_topk_vals[..., K - 1]
            target_vals  = logits_input.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            tie_at_border = (~in_topk) & ((target_vals - kth_vals).abs() <= tie_eps_abs)

            tie_mask = tie_in_topk | tie_at_border
            ranks_encoded = torch.where(tie_mask, torch.zeros_like(ranks_encoded), ranks_encoded)

        timers["topk_and_ranks"] += time.perf_counter() - start

        # --- (5) Filter PAD positions and split per sequence (ranks and exceptions)
        start = time.perf_counter()
        mask_valid = (target_ids != pad_token_id)                         # [B, L-1]
        valid_counts = mask_valid.sum(dim=1).tolist()                     # [B]

        # Flatten then split back per sequence for ranks
        flat_ranks = ranks_encoded[mask_valid]                            # [sum(valid)]
        split_ranks = torch.split(flat_ranks, valid_counts)               # tuple of B tensors

        # Exceptions: (not in top-k) OR (soft-tie active and flagged)
        zeros_mask = mask_valid & (~in_topk | (tie_as_exception and tie_eps_abs > 0.0))
        # Recompute explicit tie mask for consistency if soft-tie is enabled
        if tie_as_exception and tie_eps_abs > 0.0:
            pos_clamped = pos.clamp(min=0, max=K - 1)
            val_pos = torch.gather(stable_topk_vals, -1, pos_clamped.unsqueeze(-1)).squeeze(-1)
            BIG = torch.tensor(torch.finfo(stable_topk_vals.dtype).max,
                               device=stable_topk_vals.device)
            big_like = torch.full_like(val_pos, BIG)
            has_left  = pos > 0
            left_idx  = (pos - 1).clamp(min=0)
            left_g    = torch.gather(stable_topk_vals, -1, left_idx.unsqueeze(-1)).squeeze(-1)
            left_val  = torch.where(has_left, left_g, big_like)
            has_right = pos < (K - 1)
            right_idx = (pos + 1).clamp(max=K - 1)
            right_g   = torch.gather(stable_topk_vals, -1, right_idx.unsqueeze(-1)).squeeze(-1)
            right_val = torch.where(has_right, right_g, big_like)
            min_neighbor_diff = torch.minimum((val_pos - left_val).abs(),
                                              (val_pos - right_val).abs())
            tie_in_topk = in_topk & (min_neighbor_diff <= tie_eps_abs)
            kth_vals     = stable_topk_vals[..., K - 1]
            target_vals  = logits_input.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            tie_at_border = (~in_topk) & ((target_vals - kth_vals).abs() <= tie_eps_abs)
            tie_mask = tie_in_topk | tie_at_border
            zeros_mask = mask_valid & (~in_topk | tie_mask)

        zeros_counts = zeros_mask.sum(dim=1).tolist()                     # [B]
        flat_exceptions = target_ids[zeros_mask]                          # [sum(zeros)]
        split_exceptions = torch.split(flat_exceptions, zeros_counts)     # tuple of B tensors

        # Accumulate per-sequence Python lists
        for seq_r, seq_e in zip(split_ranks, split_exceptions):
            all_ranks.append(seq_r.tolist())
            all_exceptions.append(seq_e.tolist())

        timers["filter_and_split"] += time.perf_counter() - start

    return all_ranks, all_exceptions, timers



# ====== compute_token_ranks_fullrank_fast ====== #
@torch.no_grad()
def compute_token_ranks_fullrank_fast(
    dataloader: DataLoader,
    model: torch.nn.Module,
    pad_token_id: int,
    device: str,
    topk: int = 10,  # ignored in this variant; kept for API compatibility
    *,
    # tie handling (OFF if tie_eps_abs == 0)
    tie_as_exception: bool = True,
    tie_eps_abs: float = 0.0,   # e.g., 0.03 for 4-bit; 0 disables tie handling
) -> Tuple[List[List[int]], List[List[int]], Dict[str, float]]:
    """
    Compute target-token ranks over the full vocabulary without using top-k or sorting.

    This function mirrors the semantics and timing layout of `compute_token_ranks_topk_fast`,
    but replaces the top-k step with a full-vocabulary comparison. For each target token, its
    rank is defined as:
        1 + #(vocab logits strictly greater than the target logit)
        + #(vocab logits equal to the target logit with token-id < target_id)   [stable tie-break]

    Optional (but recommended) soft-tie handling: if the minimum absolute difference between the target logit and
    any other vocabulary logit is <= `tie_eps_abs`, the position is marked as an "exception" and
    the encoded rank is set to 0.
    
    Input:
    - dataloader (DataLoader):
        Batch loader of input sequences (token IDs).
    - model (torch.nn.Module):
        Pretrained language model (e.g., GPT-2, LLaMA) with a `.logits` output.
    - pad_token_id (int):
        Token ID used for padding.
    - device (str):
        Device to run inference on ('cuda' or 'cpu').
    - topk (int):
        Ignored in this full-rank variant; kept for API compatibility.
    - tie_as_exception (bool):
        If True, positions with soft ties are marked as exceptions (rank=0).
    - tie_eps_abs (float):
        Absolute logit difference threshold for soft-tie detection. If 0.0, tie handling is disabled.

    Return:
    - all_ranks (List[List[int]]):
        Per-sequence ranks of non-pad target tokens. Values are in {0} ∪ [1..V],
            where 0 indicates an exception due to soft-tie handling.
    - all_exceptions (List[List[int]]):
        Per-sequence lists of target token IDs where rank == 0 (exceptions).
    - timers (Dict[str, float]):
        Accumulated wall-clock timings (seconds) for each processing stage:
        "data_to_device", "forward_pass", "compute_target_logits",
        "topk_and_ranks" (full-rank work here), and "filter_and_split".
    """
    assert topk > 0, "topk must be > 0 (kept for compatibility; not used here)"
    model.eval()

    # Ensure deterministic numeric path (disable TF32 on CUDA)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    all_ranks: List[List[int]] = []
    all_exceptions: List[List[int]] = []

    # Timing buckets aligned with the original function
    timers = {
        "data_to_device": 0.0,
        "forward_pass": 0.0,
        "compute_target_logits": 0.0,
        "topk_and_ranks": 0.0,   # name kept for compatibility; holds full-vocab rank work
        "filter_and_split": 0.0,
    }

    for batch in dataloader:
        # --- (1) Move batch to target device
        start = time.perf_counter()
        input_ids = batch.to(device)                         # [B, L]
        timers["data_to_device"] += time.perf_counter() - start

        # --- (2) Forward pass with attention mask (SDPA MATH)
        start = time.perf_counter()
        attention_mask = (input_ids != pad_token_id).long()  # [B, L]
        # Use pure MATH backend for SDPA to keep behavior consistent across runs
        with sdpa_kernel(SDPBackend.MATH):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits                              # [B, L, V]
        timers["forward_pass"] += time.perf_counter() - start

        # --- (3) Shift logits/targets for next-token prediction
        start = time.perf_counter()
        logits_input = logits[:, :-1, :]   # [B, L-1, V]
        target_ids   = input_ids[:, 1:]    # [B, L-1]
        timers["compute_target_logits"] += time.perf_counter() - start

        # --- (4) Full-vocabulary stable ranks and soft-tie exceptions (no top-k)
        start = time.perf_counter()
        B, Lm1, V = logits_input.shape
        ids = torch.arange(V, device=logits_input.device)  # [V]

        # Gather target logits per position: logits_input[b, t, target_ids[b, t]]
        target_vals = logits_input.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

        # Base rank: count of logits strictly greater than the target
        greater_cnt = (logits_input > target_vals.unsqueeze(-1)).sum(dim=-1)         # [B, L-1]

        # Stable tie-break by token id: among equal logits, tokens with id < target_id rank ahead
        same_val = (logits_input == target_vals.unsqueeze(-1))                        # [B, L-1, V]
        lower_id = (ids.view(1, 1, V) < target_ids.unsqueeze(-1))                     # [B, L-1, V]
        tie_stable_cnt = (same_val & lower_id).sum(dim=-1)                            # [B, L-1]

        # Encoded rank in 1..V (before applying soft-tie exceptions)
        ranks_encoded = greater_cnt + tie_stable_cnt + 1                               # [B, L-1]

        # In this full-rank variant (conceptually K=V), targets are "in top-k"
        # unless turned into exceptions by soft-tie logic
        in_topk = torch.ones_like(ranks_encoded, dtype=torch.bool)                    # [B, L-1]
        pos = (ranks_encoded - 1).clamp(min=0, max=max(V - 1, 0))                     # [B, L-1], 0..V-1

        # Soft tie handling: mark as exception (rank=0) if any other logit is within eps
        if tie_as_exception and tie_eps_abs > 0.0:
            # Compute absolute differences to all vocab logits, ignoring self
            diffs = (logits_input - target_vals.unsqueeze(-1)).abs()                  # [B, L-1, V]
            diffs.scatter_(-1, target_ids.unsqueeze(-1), float('inf'))                # exclude self
            min_neighbor_diff = diffs.min(dim=-1).values                              # [B, L-1]

            tie_mask = (min_neighbor_diff <= tie_eps_abs)                             # [B, L-1]
            ranks_encoded = torch.where(tie_mask, torch.zeros_like(ranks_encoded), ranks_encoded)
        else:
            tie_mask = torch.zeros_like(ranks_encoded, dtype=torch.bool)

        timers["topk_and_ranks"] += time.perf_counter() - start

        # --- (5) Filter PAD positions and split per sequence (ranks and exceptions)
        start = time.perf_counter()
        mask_valid = (target_ids != pad_token_id)                                     # [B, L-1]
        valid_counts = mask_valid.sum(dim=1).tolist()                                 # [B]

        # Flatten then split back per sequence for ranks
        flat_ranks = ranks_encoded[mask_valid]                                        # [sum(valid)]
        split_ranks = torch.split(flat_ranks, valid_counts)                           # tuple of B tensors

        # Exceptions occur only due to soft ties in this full-rank variant
        zeros_mask = mask_valid & tie_mask                                            # [B, L-1]

        zeros_counts = zeros_mask.sum(dim=1).tolist()                                 # [B]
        flat_exceptions = target_ids[zeros_mask]                                      # [sum(zeros)]
        split_exceptions = torch.split(flat_exceptions, zeros_counts)                 # tuple of B tensors

        # Accumulate per-sequence Python lists
        for seq_r, seq_e in zip(split_ranks, split_exceptions):
            all_ranks.append(seq_r.tolist())
            all_exceptions.append(seq_e.tolist())

        timers["filter_and_split"] += time.perf_counter() - start

    return all_ranks, all_exceptions, timers



# ====== compute_token_ranks_topk_fast_unixcoder ====== #
@torch.no_grad()
def compute_token_ranks_topk_fast_unixcoder(
    dataloader: DataLoader,
    ux: "UniXcoder",
    pad_token_id: int,
    device: str,
    topk: int = 10,
    *,
    # tie handling (OFF if tie_eps_abs == 0)
    tie_as_exception: bool = True,
    tie_eps_abs: float = 0.0,   # e.g., 0.03 for 4-bit; 0 disables tie handling
) -> Tuple[List[List[int]], List[List[int]], Dict[str, float]]:
    """
    Compute target-token ranks using a stable top-k procedure with UniXcoder.

    This function mirrors the semantics and timing layout of the generic top-k variant,
    but adapts the forward pass to UniXcoder (causal mask from `ux.bias`, embeddings
    from `ux.model`, logits via `ux.lm_head`). For each position, it extracts the top-k
    logits and determines the rank of the true target within those candidates.
    If the target is outside the top-k, it is marked as an exception (rank=0).
    A deterministic, stable ordering is enforced by breaking ties with ascending token id.

    Optional (but recommended) soft-tie handling: if the target is inside top-k 
    and its nearest neighbor (left/right in the stable ordering) is within `tie_eps_abs`, 
    mark the position as an exception.

    Input:
    - dataloader (DataLoader):
        Batch loader of input sequences (token IDs).
    - ux (UniXcoder):
        UniXcoder model providing `bias` (causal mask buffer), `model` (backbone),
        and `lm_head` (projection to logits).
    - pad_token_id (int):
        Token ID used for padding.
    - device (str):
        Device to run inference on ('cuda' or 'cpu').
    - topk (int):
        Number of top logits to consider per position (must be > 0).
    - tie_as_exception (bool):
        If True, positions with soft ties are marked as exceptions (rank=0).
    - tie_eps_abs (float):
        Absolute logit-difference threshold for soft-tie detection. If 0.0, tie
        handling is disabled.

    Return:
    - all_ranks (List[List[int]]):
        Per-sequence ranks of non-pad target tokens. Values are in {0} ∪ [1..K],
        where 0 indicates an exception (soft-tie or outside top-k under the rule).
    - all_exceptions (List[List[int]]):
        Per-sequence lists of target token IDs where rank == 0 (exceptions).
    - timers (Dict[str, float]):
        Accumulated wall-clock timings (seconds) for each processing stage:
        "data_to_device", "forward_pass", "compute_target_logits",
        "topk_and_ranks", and "filter_and_split".
    """
    assert topk > 0, "topk must be > 0"
    ux.eval()

    # Ensure deterministic numeric path (disable TF32 on CUDA)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    all_ranks: List[List[int]] = []
    all_exceptions: List[List[int]] = []

    # Timing buckets aligned with the reference implementation
    timers = {
        "data_to_device": 0.0,
        "forward_pass": 0.0,
        "compute_target_logits": 0.0,
        "topk_and_ranks": 0.0,
        "filter_and_split": 0.0,
    }

    for batch in dataloader:
        # --- (1) Move batch to target device
        start = time.perf_counter()
        input_ids = batch.to(device)                         # [B, L]
        timers["data_to_device"] += time.perf_counter() - start

        # --- (2) UniXcoder forward pass with causal mask
        start = time.perf_counter()
        L = input_ids.size(-1)
        # `ux.bias`: [1, max_len, max_len] → take the top-left L×L submatrix
        causal_mask = ux.bias[:, :L, :L].to(device)
        # Hidden states from the backbone, then project to logits via lm_head
        token_embs = ux.model(input_ids, attention_mask=causal_mask)[0]   # [B, L, H]
        logits = ux.lm_head(token_embs)                                   # [B, L, V]
        timers["forward_pass"] += time.perf_counter() - start

        # --- (3) Shift logits/targets for next-token prediction
        start = time.perf_counter()
        logits_input = logits[:, :-1, :]   # [B, L-1, V]
        target_ids   = input_ids[:, 1:]    # [B, L-1]
        timers["compute_target_logits"] += time.perf_counter() - start

        # --- (4) Stable top-k extraction + rank/exception (+ optional soft-tie)
        start = time.perf_counter()
        topk_vals, topk_idx = torch.topk(logits_input, k=topk, dim=-1)  # [B, L-1, K]
        V = logits_input.size(-1)

        # Stable ordering: sort by value desc, then token id asc (via tiny epsilon)
        eps = (topk_idx.float() / max(V, 1)) * 1e-6                      # [B, L-1, K]
        keys = (-topk_vals).float() + eps                                # [B, L-1, K]
        order = torch.argsort(keys, dim=-1)                              # asc → value desc, id asc
        stable_topk_idx  = torch.gather(topk_idx,  dim=-1, index=order)  # [B, L-1, K]
        stable_topk_vals = torch.gather(topk_vals, dim=-1, index=order)  # [B, L-1, K]

        K = stable_topk_idx.shape[-1]
        target_exp = target_ids.unsqueeze(-1).expand(-1, -1, K)          # [B, L-1, K]
        match = (stable_topk_idx == target_exp)                           # [B, L-1, K]
        in_topk = match.any(dim=-1)                                       # [B, L-1]
        pos = match.float().argmax(dim=-1).long()                         # [B, L-1], 0..K-1

        # Encode ranks as 1..K for in-top-k; 0 otherwise
        ranks_encoded = torch.where(in_topk, pos + 1, torch.zeros_like(pos))  # [B, L-1]

        # ---- Soft-tie → exception (optional)
        if tie_as_exception and tie_eps_abs > 0.0:
            # Value at the matched position (for in-top-k)
            pos_clamped = pos.clamp(min=0, max=K - 1)
            val_pos = torch.gather(stable_topk_vals, -1, pos_clamped.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

            # Neighbor values to compute smallest local gap
            BIG = torch.tensor(torch.finfo(stable_topk_vals.dtype).max, device=stable_topk_vals.device)
            big_like = torch.full_like(val_pos, BIG)

            has_left  = pos > 0
            left_idx  = (pos - 1).clamp(min=0)
            left_g    = torch.gather(stable_topk_vals, -1, left_idx.unsqueeze(-1)).squeeze(-1)
            left_val  = torch.where(has_left, left_g, big_like)

            has_right = pos < (K - 1)
            right_idx = (pos + 1).clamp(max=K - 1)
            right_g   = torch.gather(stable_topk_vals, -1, right_idx.unsqueeze(-1)).squeeze(-1)
            right_val = torch.where(has_right, right_g, big_like)

            min_neighbor_diff = torch.minimum((val_pos - left_val).abs(), (val_pos - right_val).abs())

            tie_in_topk = in_topk & (min_neighbor_diff <= tie_eps_abs)

            # Border case: target outside top-k but close to the K-th logit
            kth_vals     = stable_topk_vals[..., K - 1]
            target_vals  = logits_input.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            tie_at_border = (~in_topk) & ((target_vals - kth_vals).abs() <= tie_eps_abs)

            tie_mask = tie_in_topk | tie_at_border
            ranks_encoded = torch.where(tie_mask, torch.zeros_like(ranks_encoded), ranks_encoded)

        timers["topk_and_ranks"] += time.perf_counter() - start

        # --- (5) Filter PAD positions and split per sequence (ranks and exceptions)
        start = time.perf_counter()
        mask_valid = (target_ids != pad_token_id)                         # [B, L-1]
        valid_counts = mask_valid.sum(dim=1).tolist()                     # [B]

        # Flatten then split back per sequence for ranks
        flat_ranks = ranks_encoded[mask_valid]                            # [sum(valid)]
        split_ranks = torch.split(flat_ranks, valid_counts)               # tuple of B tensors

        # Exceptions: (not in top-k) OR (soft-tie active and flagged)
        zeros_mask = mask_valid & (~in_topk | (tie_as_exception and tie_eps_abs > 0.0))
        # Recompute explicit tie mask for consistency if soft-tie is enabled
        if tie_as_exception and tie_eps_abs > 0.0:
            pos_clamped = pos.clamp(min=0, max=K - 1)
            val_pos = torch.gather(stable_topk_vals, -1, pos_clamped.unsqueeze(-1)).squeeze(-1)
            BIG = torch.tensor(torch.finfo(stable_topk_vals.dtype).max, device=stable_topk_vals.device)
            big_like = torch.full_like(val_pos, BIG)
            has_left  = pos > 0
            left_idx  = (pos - 1).clamp(min=0)
            left_g    = torch.gather(stable_topk_vals, -1, left_idx.unsqueeze(-1)).squeeze(-1)
            left_val  = torch.where(has_left, left_g, big_like)
            has_right = pos < (K - 1)
            right_idx = (pos + 1).clamp(max=K - 1)
            right_g   = torch.gather(stable_topk_vals, -1, right_idx.unsqueeze(-1)).squeeze(-1)
            right_val = torch.where(has_right, right_g, big_like)
            min_neighbor_diff = torch.minimum((val_pos - left_val).abs(), (val_pos - right_val).abs())
            tie_in_topk = in_topk & (min_neighbor_diff <= tie_eps_abs)
            kth_vals     = stable_topk_vals[..., K - 1]
            target_vals  = logits_input.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            tie_at_border = (~in_topk) & ((target_vals - kth_vals).abs() <= tie_eps_abs)
            tie_mask = tie_in_topk | tie_at_border
            zeros_mask = mask_valid & (~in_topk | tie_mask)

        zeros_counts = zeros_mask.sum(dim=1).tolist()                     # [B]
        flat_exceptions = target_ids[zeros_mask]                          # [sum(zeros)]
        split_exceptions = torch.split(flat_exceptions, zeros_counts)     # tuple of B tensors

        # Accumulate per-sequence Python lists
        for seq_r, seq_e in zip(split_ranks, split_exceptions):
            all_ranks.append(seq_r.tolist())
            all_exceptions.append(seq_e.tolist())

        timers["filter_and_split"] += time.perf_counter() - start

    return all_ranks, all_exceptions, timers



# ====== compute_token_ranks_fullrank_fast_unixcoder ====== #
@torch.no_grad()
def compute_token_ranks_fullrank_fast_unixcoder(
    dataloader: DataLoader,
    ux: "UniXcoder",
    pad_token_id: int,
    device: str,
    topk: int = 10,  # ignored in this variant; kept for API compatibility
    *,
    # tie handling (OFF if tie_eps_abs == 0)
    tie_as_exception: bool = True,
    tie_eps_abs: float = 0.0,   # e.g., 0.03 for 4-bit; 0 disables tie handling
) -> Tuple[List[List[int]], List[List[int]], Dict[str, float]]:
    """
    Compute target-token ranks over the full vocabulary with UniXcoder (no top-k, no sorting).

    This function mirrors the semantics and timing layout of the full-rank variant for
    generic causal LMs, but adapts the forward pass to UniXcoder (causal mask from
    `ux.bias`, embeddings from `ux.model`, logits via `ux.lm_head`). For each target
    token, its rank is defined as:
        1 + #(vocab logits strictly greater than the target logit)
        + #(vocab logits equal to the target logit with token-id < target_id)   [stable tie-break]

    Optional (but recommended) soft-tie handling: if the minimum absolute difference
    between the target logit and any other vocabulary logit is <= `tie_eps_abs`,
    the position is marked as an "exception" and the encoded rank is set to 0.

    Input:
    - dataloader (DataLoader):
        Batch loader of input sequences (token IDs).
    - ux (UniXcoder):
        UniXcoder model providing `bias` (causal mask buffer), `model` (backbone),
        and `lm_head` (projection to logits).
    - pad_token_id (int):
        Token ID used for padding.
    - device (str):
        Device to run inference on ('cuda' or 'cpu').
    - topk (int):
        Ignored in this full-rank variant; kept for API compatibility.
    - tie_as_exception (bool):
        If True, positions with soft ties are marked as exceptions (rank=0).
    - tie_eps_abs (float):
        Absolute logit-difference threshold for soft-tie detection. If 0.0, tie
        handling is disabled.

    Return:
    - all_ranks (List[List[int]]):
        Per-sequence ranks of non-pad target tokens. Values are in {0} ∪ [1..V],
        where 0 indicates an exception due to soft-tie handling.
    - all_exceptions (List[List[int]]):
        Per-sequence lists of target token IDs where rank == 0 (exceptions).
    - timers (Dict[str, float]):
        Accumulated wall-clock timings (seconds) for each processing stage:
        "data_to_device", "forward_pass", "compute_target_logits",
        "topk_and_ranks" (full-vocab work here), and "filter_and_split".
    """
    assert topk > 0, "topk must be > 0 (kept for compatibility; not used here)"
    ux.eval()

    # Ensure deterministic numeric path (disable TF32 on CUDA)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    all_ranks: List[List[int]] = []
    all_exceptions: List[List[int]] = []

    # Timing buckets aligned with the reference implementation
    timers = {
        "data_to_device": 0.0,
        "forward_pass": 0.0,
        "compute_target_logits": 0.0,
        "topk_and_ranks": 0.0,   # name kept for compatibility; holds full-vocab rank work
        "filter_and_split": 0.0,
    }

    for batch in dataloader:
        # --- (1) Move batch to target device
        start = time.perf_counter()
        input_ids = batch.to(device)                         # [B, L]
        timers["data_to_device"] += time.perf_counter() - start

        # --- (2) UniXcoder forward pass with causal mask
        start = time.perf_counter()
        L = input_ids.size(-1)
        # `ux.bias`: [1, max_len, max_len] → take the top-left L×L submatrix
        causal_mask = ux.bias[:, :L, :L].to(device)
        # Hidden states from the backbone, then project to logits via lm_head
        token_embs = ux.model(input_ids, attention_mask=causal_mask)[0]   # [B, L, H]
        logits = ux.lm_head(token_embs)                                   # [B, L, V]
        timers["forward_pass"] += time.perf_counter() - start

        # --- (3) Shift logits/targets for next-token prediction
        start = time.perf_counter()
        logits_input = logits[:, :-1, :]   # [B, L-1, V]
        target_ids   = input_ids[:, 1:]    # [B, L-1]
        timers["compute_target_logits"] += time.perf_counter() - start

        # --- (4) Full-vocabulary stable ranks and soft-tie exceptions (no top-k)
        start = time.perf_counter()
        B, Lm1, V = logits_input.shape
        ids = torch.arange(V, device=logits_input.device)  # [V]

        # Gather target logits per position: logits_input[b, t, target_ids[b, t]]
        target_vals = logits_input.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

        # Base rank: count of logits strictly greater than the target
        greater_cnt = (logits_input > target_vals.unsqueeze(-1)).sum(dim=-1)         # [B, L-1]

        # Stable tie-break by token id: among equal logits, tokens with id < target_id rank ahead
        same_val = (logits_input == target_vals.unsqueeze(-1))                        # [B, L-1, V]
        lower_id = (ids.view(1, 1, V) < target_ids.unsqueeze(-1))                     # [B, L-1, V]
        tie_stable_cnt = (same_val & lower_id).sum(dim=-1)                            # [B, L-1]

        # Encoded rank in 1..V (before applying soft-tie exceptions)
        ranks_encoded = greater_cnt + tie_stable_cnt + 1                               # [B, L-1]

        # Conceptually K=V here; positions are "in top-k" unless turned into exceptions
        in_topk = torch.ones_like(ranks_encoded, dtype=torch.bool)                    # [B, L-1]
        pos = (ranks_encoded - 1).clamp(min=0, max=max(V - 1, 0))                     # [B, L-1], 0..V-1

        # Soft-tie handling: mark as exception (rank=0) if any other logit is within eps
        if tie_as_exception and tie_eps_abs > 0.0:
            # Compute absolute differences to all vocab logits, ignoring self
            diffs = (logits_input - target_vals.unsqueeze(-1)).abs()                  # [B, L-1, V]
            diffs.scatter_(-1, target_ids.unsqueeze(-1), float('inf'))                # exclude self
            min_neighbor_diff = diffs.min(dim=-1).values                              # [B, L-1]

            tie_mask = (min_neighbor_diff <= tie_eps_abs)                             # [B, L-1]
            ranks_encoded = torch.where(tie_mask, torch.zeros_like(ranks_encoded), ranks_encoded)
        else:
            tie_mask = torch.zeros_like(ranks_encoded, dtype=torch.bool)

        timers["topk_and_ranks"] += time.perf_counter() - start

        # --- (5) Filter PAD positions and split per sequence (ranks and exceptions)
        start = time.perf_counter()
        mask_valid = (target_ids != pad_token_id)                                     # [B, L-1]
        valid_counts = mask_valid.sum(dim=1).tolist()                                 # [B]

        # Flatten then split back per sequence for ranks
        flat_ranks = ranks_encoded[mask_valid]                                        # [sum(valid)]
        split_ranks = torch.split(flat_ranks, valid_counts)                           # tuple of B tensors

        # Exceptions occur only due to soft ties in this full-rank variant
        zeros_mask = mask_valid & tie_mask                                            # [B, L-1]

        zeros_counts = zeros_mask.sum(dim=1).tolist()                                 # [B]
        flat_exceptions = target_ids[zeros_mask]                                      # [sum(zeros)]
        split_exceptions = torch.split(flat_exceptions, zeros_counts)                 # tuple of B tensors

        # Accumulate per-sequence Python lists
        for seq_r, seq_e in zip(split_ranks, split_exceptions):
            all_ranks.append(seq_r.tolist())
            all_exceptions.append(seq_e.tolist())

        timers["filter_and_split"] += time.perf_counter() - start

    return all_ranks, all_exceptions, timers


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (2) Perform the inverse operation (convert ranks back to tokens).
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ====== decode_ranks_batched ====== #
def decode_ranks_batched(
    rank_sequences: List[List[int]],
    exception_sequences: List[List[int]],
    model: torch.nn.Module,
    tokenizer,
    pad_token_id: int,
    batch_size: int,
    topk: int,
    device: str = "cuda",
    debug: bool = False,
    show_progress: bool = True,
    inner_progress: bool = False,
    *,
    debug_step: Optional[int] = None,      # step index at which to print detailed diagnostics
    debug_topn_print: int = 10,            # number of top items to print in diagnostics
) -> Tuple[Dict[int, List[int]], Dict[int, List[List[int]]]]:
    """
    Decode token sequences from rank annotations in batched mode.

    Given per-sequence rank annotations (with 0 indicating "exception") and the matching
    exception token streams, this routine reconstructs the actual token IDs by iteratively
    running the model forward and selecting either:
      - the target within the stable top-k list according to the provided rank r ∈ [1..K], or
      - the next token ID from the exception list when r == 0.

    Stable top-k ordering is enforced by sorting logits descending and breaking ties with
    ascending token id using a tiny epsilon offset. The function processes sequences in
    mini-batches for efficiency and optionally records the decoding context (prefix tokens)
    at each step.

    Input:
    - rank_sequences (List[List[int]]):
        Per-sequence ranks for each time step. Values in {0} ∪ [1..K], where 0 denotes
        an exception (token taken from the corresponding exception list).
    - exception_sequences (List[List[int]]):
        Per-sequence token-ID streams consumed whenever the rank is 0. Must match the
        number of rank==0 positions for each sequence.
    - model (torch.nn.Module):
        Autoregressive LM that returns `.logits` for next-token prediction.
    - tokenizer:
        Tokenizer providing `bos_token_id` or `eos_token_id` fallback for BOS.
    - pad_token_id (int):
        Token used to pad unfinished positions in the working tensor.
    - batch_size (int):
        Number of sequences decoded in parallel.
    - topk (int):
        Size of the top-k list used to map ranks r ∈ [1..K] to token IDs (must be > 0).
    - device (str):
        Target device for tensors ('cuda' or 'cpu').
    - debug (bool):
        If True, print lightweight per-step logs (except the designated `debug_step`,
        which prints detailed diagnostics).
    - show_progress (bool):
        If True, show a progress bar over outer chunks.
    - inner_progress (bool):
        If True, show a progress bar over steps inside each chunk.
    - debug_step (Optional[int]):
        Step index at which to emit detailed diagnostics (tensor shapes, top-k vs full sort).
    - debug_topn_print (int):
        Number of top entries shown in diagnostics.

    Return:
    - decoded_token_ids (Dict[int, List[int]]):
        Mapping from global sequence index → reconstructed token-id list.
    - contexts_per_seq (Dict[int, List[List[int]]]):
        Mapping from global sequence index → list of context prefixes (token ids)
        used at each decoding step (for inspection/debugging).
    """
    from tqdm import tqdm

    assert topk > 0, "topk must be > 0"
    model.eval()

    # Determine BOS token; fall back to EOS when BOS is unavailable.
    bos_token_id = getattr(tokenizer, "bos_token_id", tokenizer.eos_token_id)

    # Sanity check: ranks and exceptions must have the same number of sequences.
    assert len(rank_sequences) == len(exception_sequences)
    N = len(rank_sequences)

    # Output containers: per-sequence decoded tokens and contexts.
    decoded_token_ids: Dict[int, List[int]] = {i: [] for i in range(N)}
    contexts_per_seq: Dict[int, List[List[int]]] = {i: [] for i in range(N)}

    # Iterate over sequence indices in mini-batches.
    outer_iter = range(0, N, batch_size)
    if show_progress:
        outer_iter = tqdm(outer_iter, desc="Chunks (mini-batch of lists)", unit="chunk")

    with torch.no_grad():
        for start_idx in outer_iter:
            end_idx = min(start_idx + batch_size, N)
            idxs = list(range(start_idx, end_idx))
            B = len(idxs)

            # Slice rank/exception data for this chunk; use iterators for exceptions.
            ranks_batch = [rank_sequences[i] for i in idxs]
            exc_iters   = [iter(exception_sequences[i]) for i in idxs]

            # Maximum number of decoding steps in this chunk (longest sequence by ranks).
            max_steps = max((len(r) for r in ranks_batch), default=0)
            if max_steps == 0:
                continue

            # Allocate working tensors: BOS at position 0, then decode up to max_steps tokens.
            T = 1 + max_steps
            input_ids = torch.full((B, T), pad_token_id, dtype=torch.long, device=device)
            input_ids[:, 0] = bos_token_id
            attention_mask = (input_ids != pad_token_id).long()  # 1 for BOS, 0 elsewhere initially

            # Optional inner progress over steps.
            step_iter = range(max_steps)
            if show_progress and inner_progress:
                step_iter = tqdm(step_iter,
                                 desc=f"Steps chunk [{start_idx}:{end_idx})",
                                 leave=False, unit="step")

            for t in step_iter:
                # Record current context (prefix up to and including position t) per sequence.
                for j in range(B):
                    if t < len(ranks_batch[j]):
                        ctx = input_ids[j, :t+1].tolist()
                        # Remove pad tokens if present in the trailing position.
                        if ctx and ctx[-1] == pad_token_id:
                            ctx = [x for x in ctx if x != pad_token_id]
                        contexts_per_seq[idxs[j]].append(ctx)

                # Forward pass: attention is effectively causal (only up to t+1 has mask==1).
                with sdpa_kernel(SDPBackend.MATH):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits     # [B, T, V]
                V = logits.size(-1)
                pos = t  # we predict the token at position t+1 from row 'pos' = t

                # Step-scoped debug switch.
                will_debug = (debug_step is not None and t == debug_step)
                if will_debug:
                    print(f"\n[DECODE DEBUG] chunk[{start_idx}:{end_idx}) step={t} (B,T,V)=({B},{T},{V})")
                    # Each row's attention sum reflects how many tokens are currently "visible".
                    print("  attn_sums_per_row:", [int(attention_mask[j, :].sum().item()) for j in range(B)])

                # Decode each sequence in the mini-batch at step t.
                for j in range(B):
                    ranks = ranks_batch[j]
                    if t >= len(ranks):
                        # This sequence has no more ranks to consume.
                        continue

                    r = ranks[t]
                    row_logits = logits[j, pos]  # [V] scores for next token

                    if r == 0:
                        # Exception slot: consume the next token id from the exception iterator.
                        try:
                            predicted_token_id = int(next(exc_iters[j]))
                            chosen_v = float('nan')
                        except StopIteration:
                            raise RuntimeError(
                                f"Global row {idxs[j]}: rank==0 but exceptions are exhausted at step {t}."
                            )
                    else:
                        # Rank must be within [1..topk]; map it to a token via stable top-k.
                        if r < 1 or r > topk:
                            raise RuntimeError(
                                f"Global row {idxs[j]}: rank={r} out of [1..{topk}] at step {t}."
                            )
                        vals_k, ids_k = torch.topk(row_logits, k=topk, dim=-1)  # [K], [K]

                        # Stable keys: sort by value desc, then by token id asc via tiny epsilon.
                        eps = (ids_k.float() / max(V, 1)) * 1e-6
                        keys = (-vals_k).float() + eps
                        order = torch.argsort(keys, dim=-1)

                        stable_topk_ids = ids_k[order]          # [K] stable token ids
                        predicted_token_id = int(stable_topk_ids[r-1].item())
                        chosen_v = float(vals_k[order][r-1].item())

                        if will_debug:
                            # Optional diagnostics: show stable top-k vs full-vocab sorted ids.
                            ids_all = torch.arange(V, device=row_logits.device)
                            eps_all = (ids_all.float() / max(V, 1)) * 1e-6
                            order_full = torch.argsort((-row_logits).float() + eps_all)
                            full_sorted = ids_all[order_full]

                            print(f"  [row j={j} global={idxs[j]}] ctx_ids[:t+1]={input_ids[j, :t+1].tolist()}")
                            print(f"    rank_req={r}  chosen_id={predicted_token_id}  chosen_logit={chosen_v:.6f}")
                            print(f"    topk(stable) ids[:{debug_topn_print}]={stable_topk_ids[:debug_topn_print].tolist()}")
                            print(f"    topk(stable) vals[:{debug_topn_print}]={vals_k[order][:debug_topn_print].tolist()}")
                            print(f"    fullsort ids[:{debug_topn_print}]={full_sorted[:debug_topn_print].tolist()}")

                    # Commit the predicted token and extend the attention window by one.
                    input_ids[j, t+1] = predicted_token_id
                    attention_mask[j, t+1] = 1
                    decoded_token_ids[idxs[j]].append(predicted_token_id)

                    # Lightweight per-step debug print (suppressed at the detailed debug step).
                    if debug and not will_debug:
                        tok_text = tokenizer.decode([predicted_token_id])
                        print(f"[global_row={idxs[j]} step={t}] rank={r} -> id={predicted_token_id} '{tok_text}'")

    return decoded_token_ids, contexts_per_seq




# ===================================================================================================
# ================================ OLD FUNCTIONS - NOT USED =========================================
# ===================================================================================================

# =====================================================
# These functions are retained for reference but not used
# in the current pipeline. They are used for debugging,
# when rich diagnostics are needed for the reconstruction
# process.
# =====================================================


# ====== compute_token_ranks_topk_debug ====== #
def compute_token_ranks_topk_debug(
    dataloader: DataLoader,
    model: torch.nn.Module,
    pad_token_id: int,
    device: str,
    topk: int = 10,
    *,
    # === tie handling (OFF if tie_eps_abs == 0) ===
    tie_as_exception: bool = True,
    tie_eps_abs: float = 0.0,           # e.g., 0.03 for 4-bit; 0 disables soft-tie handling
    # === optional debugging ===
    debug_step: Optional[int] = None,   # 0-based step index for detailed diagnostics
    debug_topn_print: int = 10,         # how many top entries to display in diagnostics
) -> Tuple[List[List[int]], List[List[int]], Dict[str, float], List[List[List[int]]]]:
    """
    Compute target-token ranks via a stable top-k procedure and emit rich debugging info.

    This function mirrors the semantics of the non-debug top-k variant while additionally
    collecting, per sequence and per step, the decoding contexts (prefix token ids) and
    optionally printing detailed diagnostics at a chosen time step. The stable ordering
    among top-k candidates is enforced by sorting scores descending and breaking ties with
    ascending token id using a tiny epsilon.

    Soft-tie handling (optional): a position is marked as an exception (encoded rank=0) if
      (i) the target is within top-k but its nearest neighbor (left/right in the stable
          order) is within `tie_eps_abs`, or
      (ii) the target is outside top-k but its logit is within `tie_eps_abs` of the k-th
           logit (border case).

    Input:
    - dataloader (DataLoader):
        Batches of tokenized input sequences (shape [B, L]).
    - model (torch.nn.Module):
        Autoregressive LM returning `.logits`.
    - pad_token_id (int):
        Token id used for padding in the input sequences.
    - device (str):
        Device identifier ('cuda' or 'cpu').
    - topk (int):
        Number of top logits to consider per position. Must be > 0.
    - tie_as_exception (bool):
        If True, soft ties trigger exception (rank=0).
    - tie_eps_abs (float):
        Absolute logit-distance threshold for soft-tie detection. 0 disables it.
    - debug_step (Optional[int]):
        If provided, prints detailed diagnostics at this step index (0-based) for each row
        that has that step.
    - debug_topn_print (int):
        Number of entries to print from top-k and full-sort diagnostics.

    Return:
    - all_ranks (List[List[int]]):
        Per-sequence ranks in {0} ∪ [1..K]; 0 marks exceptions or target outside top-k
        under the soft-tie rule.
    - all_exceptions (List[List[int]]):
        Per-sequence lists of target token ids where rank==0.
    - timers (Dict[str, float]):
        Accumulated timings in seconds for:
          "data_to_device", "forward_pass", "compute_target_logits",
          "topk_and_ranks", "filter_and_split".
    - all_contexts_per_step (List[List[List[int]]]):
        For each sequence, the list of context prefixes (token ids) used at each step.
    """
    assert topk > 0, "topk must be > 0"
    model.eval()

    all_ranks: List[List[int]] = []
    all_exceptions: List[List[int]] = []
    all_contexts_per_step: List[List[List[int]]] = []

    # Timing buckets aligned with the non-debug implementation
    timers = {
        "data_to_device": 0.0,
        "forward_pass": 0.0,
        "compute_target_logits": 0.0,
        "topk_and_ranks": 0.0,
        "filter_and_split": 0.0,
    }

    # Ensure deterministic numeric path (disable TF32 on CUDA)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with torch.no_grad():
        global_seq_counter = 0  # tracks absolute sequence index across batches

        for batch in dataloader:
            # --- (1) Move batch to device
            start = time.perf_counter()
            input_ids = batch.to(device)                         # [B, L]
            timers["data_to_device"] += time.perf_counter() - start

            # --- (2) Forward pass with attention mask (SDPA MATH)
            start = time.perf_counter()
            attention_mask = (input_ids != pad_token_id).long()  # [B, L]
            # Use pure MATH backend for SDPA to keep behavior consistent
            with sdpa_kernel(SDPBackend.MATH):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits                              # [B, L, V]
            timers["forward_pass"] += time.perf_counter() - start

            # --- (3) Shift logits/targets for next-token prediction
            start = time.perf_counter()
            logits_input = logits[:, :-1, :]   # [B, L-1, V]
            target_ids   = input_ids[:, 1:]    # [B, L-1]
            timers["compute_target_logits"] += time.perf_counter() - start

            # --- (3b) Trace contexts per step (prefixes up to each step)
            B, Lm1, V = logits_input.shape
            batch_contexts: List[List[List[int]]] = []
            for b in range(B):
                valid_len = int((input_ids[b] != pad_token_id).sum().item())
                seq_ctx: List[List[int]] = []
                num_steps = max(0, valid_len - 1)
                for t in range(num_steps):
                    prefix = input_ids[b, :t+1].tolist()
                    # Remove pad tokens if present at the end (defensive)
                    if prefix and prefix[-1] == pad_token_id:
                        prefix = [x for x in prefix if x != pad_token_id]
                    seq_ctx.append(prefix)
                batch_contexts.append(seq_ctx)

            # --- (3c) Optional per-step diagnostics at 'debug_step'
            if debug_step is not None:
                for b in range(B):
                    valid_len = int((input_ids[b] != pad_token_id).sum().item())
                    num_steps = max(0, valid_len - 1)
                    if debug_step < num_steps:
                        row_logits = logits_input[b, debug_step]   # [V]
                        tgt_id     = int(target_ids[b, debug_step].item())

                        # Stable top-k (value desc, id asc via epsilon)
                        vals_k, ids_k = torch.topk(row_logits, k=topk, dim=-1)
                        eps_k = (ids_k.float() / max(row_logits.numel(), 1)) * 1e-6
                        order = torch.argsort((-vals_k).float() + eps_k, dim=-1)
                        stable_topk_ids  = ids_k[order]
                        stable_topk_vals = vals_k[order]

                        # Full-vocabulary diagnostic sort (value desc, id asc via epsilon)
                        ids_all  = torch.arange(row_logits.numel(), device=row_logits.device)
                        eps_all  = (ids_all.float() / max(row_logits.numel(), 1)) * 1e-6
                        order_fs = torch.argsort((-row_logits).float() + eps_all)
                        full_sorted_ids  = ids_all[order_fs]
                        full_sorted_vals = row_logits[order_fs]

                        where = (stable_topk_ids == tgt_id).nonzero(as_tuple=False)
                        rank_if_in = int(where[0].item()+1) if where.numel() > 0 else 0

                        print(f"\n[COMPUTE DEBUG] global_seq={global_seq_counter + b} step={debug_step} "
                              f"(B,L,V)=({B},{input_ids.size(1)},{row_logits.numel()})")
                        print(f"  attn_sum={int(attention_mask[b].sum().item())}  valid_len={valid_len}  pos=t={debug_step}")
                        print(f"  context_ids[:t+1]={input_ids[b, :debug_step+1].tolist()}")
                        print(f"  target_id@t+1={tgt_id}")
                        print(f"  topk(stable) ids[:{debug_topn_print}]={stable_topk_ids[:debug_topn_print].tolist()}")
                        print(f"  topk(stable) vals[:{debug_topn_print}]={[float(x) for x in stable_topk_vals[:debug_topn_print]]}")
                        print(f"  fullsort ids[:{debug_topn_print}]={full_sorted_ids[:debug_topn_print].tolist()}")
                        print(f"  fullsort vals[:{debug_topn_print}]={[float(x) for x in full_sorted_vals[:debug_topn_print]]}")
                        print(f"  target_rank_in_topk={rank_if_in}")

            # --- (4) Stable top-k extraction + rank/exception (+ optional soft-tie)
            start = time.perf_counter()
            topk_vals, topk_idx = torch.topk(logits_input, k=topk, dim=-1)  # [B, L-1, K]
            eps = (topk_idx.float() / max(V, 1)) * 1e-6                      # [B, L-1, K]
            keys = (-topk_vals).float() + eps                                # [B, L-1, K]
            order = torch.argsort(keys, dim=-1)                              # asc → value desc, id asc
            stable_topk_idx  = torch.gather(topk_idx,  dim=-1, index=order)  # [B, L-1, K]
            stable_topk_vals = torch.gather(topk_vals, dim=-1, index=order)  # [B, L-1, K]

            K = stable_topk_idx.shape[-1]
            target_exp = target_ids.unsqueeze(-1).expand(-1, -1, K)          # [B, L-1, K]
            match = (stable_topk_idx == target_exp)                           # [B, L-1, K]
            in_topk = match.any(dim=-1)                                       # [B, L-1]
            pos = match.float().argmax(dim=-1).long()                         # [B, L-1], 0..K-1

            # Encode ranks as 1..K for in-top-k; 0 otherwise
            ranks_encoded = torch.where(in_topk, pos + 1, torch.zeros_like(pos))  # [B, L-1]

            # ---- Soft-tie → exception (optional, safe)
            tie_mask = torch.zeros_like(in_topk, dtype=torch.bool)  # [B, L-1]
            if tie_as_exception and tie_eps_abs > 0.0:
                # Value at matched position (for in-top-k)
                pos_clamped = pos.clamp(min=0, max=K-1)
                val_pos = torch.gather(stable_topk_vals, -1, pos_clamped.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

                # Large sentinel (same dtype) for positions without left/right neighbor
                BIG = torch.tensor(torch.finfo(stable_topk_vals.dtype).max, device=stable_topk_vals.device)
                big_like = torch.full_like(val_pos, BIG)

                # Left neighbor (only where pos > 0)
                has_left = pos > 0
                left_idx = (pos - 1).clamp(min=0)
                left_g   = torch.gather(stable_topk_vals, -1, left_idx.unsqueeze(-1)).squeeze(-1)
                left_val = torch.where(has_left, left_g, big_like)

                # Right neighbor (only where pos < K-1)
                has_right = pos < (K - 1)
                right_idx = (pos + 1).clamp(max=K-1)
                right_g   = torch.gather(stable_topk_vals, -1, right_idx.unsqueeze(-1)).squeeze(-1)
                right_val = torch.where(has_right, right_g, big_like)

                # Smallest neighbor gap in top-k
                min_neighbor_diff = torch.minimum((val_pos - left_val).abs(), (val_pos - right_val).abs())

                tie_in_topk = in_topk & (min_neighbor_diff <= tie_eps_abs)

                # Border case: target outside top-k but close to k-th logit
                kth_vals     = stable_topk_vals[..., K - 1]                         # [B, L-1]
                target_vals  = logits_input.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
                tie_at_border = (~in_topk) & ((target_vals - kth_vals).abs() <= tie_eps_abs)

                tie_mask = tie_in_topk | tie_at_border
                # Turn ranks into 0 where a soft tie is detected
                ranks_encoded = torch.where(tie_mask, torch.zeros_like(ranks_encoded), ranks_encoded)

            timers["topk_and_ranks"] += time.perf_counter() - start

            # --- (5) Filter PAD and split per sequence (ranks/exceptions/contexts)
            start = time.perf_counter()
            mask_valid = (target_ids != pad_token_id)                         # [B, L-1]
            valid_counts = mask_valid.sum(dim=1).tolist()                     # [B]

            # Ranks: flatten valid positions then split back per sequence
            flat_ranks = ranks_encoded[mask_valid]                            # [sum(valid)]
            split_ranks = torch.split(flat_ranks, valid_counts)               # tuple of B tensors

            # Exceptions: outside top-k OR soft-tie flagged
            zeros_mask = mask_valid & (~in_topk | tie_mask)                   # [B, L-1]
            zeros_counts = zeros_mask.sum(dim=1).tolist()                     # [B]
            flat_exceptions = target_ids[zeros_mask]                          # [sum(zeros)]
            split_exceptions = torch.split(flat_exceptions, zeros_counts)     # tuple of B tensors

            # Accumulate per-sequence outputs and contexts
            for b, (seq_r, seq_e) in enumerate(zip(split_ranks, split_exceptions)):
                all_ranks.append(seq_r.tolist())
                all_exceptions.append(seq_e.tolist())
                seq_ctx = batch_contexts[b][:valid_counts[b]]
                all_contexts_per_step.append(seq_ctx)

            timers["filter_and_split"] += time.perf_counter() - start
            global_seq_counter += B

    return all_ranks, all_exceptions, timers, all_contexts_per_step


# ====== compute_token_ranks_topk_stepwise ====== #
@torch.no_grad()
def compute_token_ranks_topk_stepwise(
    dataloader: DataLoader,
    model: torch.nn.Module,
    pad_token_id: int,
    device: str,
    topk: int = 10,
    *,
    # === tie handling (same as compute_token_ranks_topk4) ===
    tie_as_exception: bool = True,
    tie_eps_abs: float = 0.0,
    # === optional debug (t = 0-based, over the stepwise loop) ===
    debug_step: Optional[int] = None,
    debug_topn_print: int = 10,
) -> Tuple[List[List[int]], List[List[int]], Dict[str, float], List[List[List[int]]]]:
    """
    Compute top-k-based ranks step-by-step using teacher forcing (prefix-only evaluation).

    This stepwise variant reproduces the exact rank/exception semantics of the classic
    top-k compute function, but it performs a forward pass per time step with the
    *true* prefix (teacher forcing). At each step t:
      - The working input contains only the real tokens up to position t (others are PAD).
      - The model computes logits for that step; we extract row t (predicting token t+1).
      - After the loop, the collected per-step logits form a [B, L-1, V] tensor, and we
        apply the same stable top-k + soft-tie handling used by the batched version.

    Output matches the standard top-k implementation:
      - all_ranks:        ranks per sequence (values in {0} ∪ [1..K])
      - all_exceptions:   target token ids where rank == 0 (exceptions)
      - timers:           timing breakdown
      - all_contexts_per_step: list of per-step prefix contexts for each sequence

    Input:
    - dataloader (DataLoader):
        Batches of tokenized sequences (shape [B, L]).
    - model (torch.nn.Module):
        Autoregressive LM returning `.logits`.
    - pad_token_id (int):
        Padding token id used in input sequences.
    - device (str):
        Device identifier ('cuda' or 'cpu').
    - topk (int):
        Size of the top-k set (must be > 0).
    - tie_as_exception (bool):
        If True, soft ties are treated as exceptions (rank=0).
    - tie_eps_abs (float):
        Absolute logit-difference threshold for soft-tie detection. 0 disables it.
    - debug_step (Optional[int]):
        If provided, prints detailed diagnostics for this step index (0-based).
    - debug_topn_print (int):
        Number of entries to display in top-k/full-sort debug prints.

    Return:
    - all_ranks (List[List[int]]):
        Per-sequence ranks in {0} ∪ [1..K]. 0 denotes exception (soft-tie or outside top-k).
    - all_exceptions (List[List[int]]):
        Per-sequence lists of target token ids where rank==0.
    - timers (Dict[str, float]):
        Accumulated timings (seconds) for:
          "data_to_device", "forward_pass" (sum over steps),
          "compute_target_logits", "topk_and_ranks", "filter_and_split".
    - all_contexts_per_step (List[List[List[int]]]):
        For each sequence, the list of prefix contexts (token ids) used at each step.
    """
    assert topk > 0, "topk must be > 0"
    model.eval()

    # Ensure deterministic numeric path (disable TF32 on CUDA)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    all_ranks: List[List[int]] = []
    all_exceptions: List[List[int]] = []
    all_contexts_per_step: List[List[List[int]]] = []

    # Timing buckets aligned with the non-stepwise implementation
    timers = {
        "data_to_device": 0.0,
        "forward_pass": 0.0,            # sum over per-step forward passes
        "compute_target_logits": 0.0,   # negligible here (targets already aligned)
        "topk_and_ranks": 0.0,
        "filter_and_split": 0.0,
    }

    global_seq_counter = 0  # absolute sequence index across batches (for debug logs)

    for batch in dataloader:
        # --- (1) Move batch to device
        start = time.perf_counter()
        input_ids = batch.to(device)                 # [B, L]
        timers["data_to_device"] += time.perf_counter() - start

        B, L = input_ids.shape

        # Full validity mask (used to filter PAD targets later)
        attention_mask_full = (input_ids != pad_token_id).long()  # [B, L]

        # Will hold per-step logits for predicting positions 1..L-1  → shape [B, L-1, V]
        logits_input = None

        # Trace contexts per step: for each row, collect the real prefix used at step t
        batch_contexts: List[List[List[int]]] = [[] for _ in range(B)]

        # --- (2) Stepwise loop over t = 0..L-2 (predicting token at t+1)
        for t in range(L - 1):
            # Build stepwise working input: real tokens up to t (inclusive), PAD elsewhere
            work = torch.full_like(input_ids, pad_token_id)
            work[:, :t + 1] = input_ids[:, :t + 1]
            attn = (work != pad_token_id).long()

            # TRACE: save the prefix context used at this step for rows with a real target at t+1
            for b in range(B):
                if t < (attention_mask_full[b].sum().item() - 1):
                    ctx = work[b, :t + 1].tolist()
                    batch_contexts[b].append(ctx)

            # Per-step forward (use MATH SDPA backend for consistent behavior)
            start = time.perf_counter()
            with sdpa_kernel(SDPBackend.MATH):
                out = model(input_ids=work, attention_mask=attn)
            timers["forward_pass"] += time.perf_counter() - start

            # The row that predicts position t+1 is at index t
            row_logits_t = out.logits[:, t, :]      # [B, V]

            # Allocate the final [B, L-1, V] container at the first step
            if logits_input is None:
                V = row_logits_t.size(-1)
                logits_input = torch.empty(
                    B, L - 1, V,
                    device=row_logits_t.device,
                    dtype=row_logits_t.dtype
                )

            # Store the step logits in column t
            logits_input[:, t, :] = row_logits_t

            # --- Optional per-step debug diagnostics
            if debug_step is not None and t == debug_step:
                for b in range(B):
                    # Skip rows where the target t+1 is PAD
                    if int(input_ids[b, t + 1].item()) == pad_token_id:
                        continue

                    tgt_id = int(input_ids[b, t + 1].item())
                    vals_k, ids_k = torch.topk(row_logits_t[b], k=topk, dim=-1)  # descending
                    # Stable sort key: value desc, then token id asc via tiny epsilon
                    eps_k = (ids_k.float() / max(row_logits_t.size(-1), 1)) * 1e-6
                    order = torch.argsort((-vals_k).float() + eps_k, dim=-1)
                    stable_topk_ids  = ids_k[order]
                    stable_topk_vals = vals_k[order]

                    # Full-vocabulary diagnostic sort (value desc, id asc via epsilon)
                    ids_all  = torch.arange(row_logits_t.size(-1), device=row_logits_t.device)
                    eps_all  = (ids_all.float() / max(row_logits_t.size(-1), 1)) * 1e-6
                    order_fs = torch.argsort((-row_logits_t[b]).float() + eps_all)
                    full_sorted_ids  = ids_all[order_fs]
                    full_sorted_vals = row_logits_t[b][order_fs]

                    where = (stable_topk_ids == tgt_id).nonzero(as_tuple=False)
                    rank_if_in = int(where[0].item() + 1) if where.numel() > 0 else 0

                    print(f"\n[STEPWISE DEBUG] global_seq={global_seq_counter + b} step={t} "
                          f"(B,L,V)=({B},{L},{row_logits_t.size(-1)})")
                    print(f"  attn_sum={int(attn[b].sum().item())}  "
                          f"valid_len={int(attention_mask_full[b].sum().item())}  pos=t={t}")
                    print(f"  context_ids[:t+1]={work[b, :t+1].tolist()}")
                    print(f"  target_id@t+1={tgt_id}")
                    print(f"  topk(stable) ids[:{debug_topn_print}]={stable_topk_ids[:debug_topn_print].tolist()}")
                    print(f"  topk(stable) vals[:{debug_topn_print}]="
                          f"{[float(x) for x in stable_topk_vals[:debug_topn_print]]}")
                    print(f"  fullsort ids[:{debug_topn_print}]={full_sorted_ids[:debug_topn_print].tolist()}")
                    print(f"  fullsort vals[:{debug_topn_print}]="
                          f"{[float(x) for x in full_sorted_vals[:debug_topn_print]]}")
                    print(f"  target_rank_in_topk={rank_if_in}")

        # --- (3) Align targets as in the classic batched compute
        start = time.perf_counter()
        target_ids = input_ids[:, 1:]  # [B, L-1]
        timers["compute_target_logits"] += time.perf_counter() - start

        # --- (4) Stable top-k + ranks/exceptions (+ optional soft-tie) — IDENTICAL to topk4
        start = time.perf_counter()
        topk_vals, topk_idx = torch.topk(logits_input, k=topk, dim=-1)        # [B, L-1, K]
        V = logits_input.size(-1)
        # Stable ordering key: value desc, id asc via tiny epsilon
        eps = (topk_idx.float() / max(V, 1)) * 1e-6                           # [B, L-1, K]
        keys = (-topk_vals).float() + eps                                     # [B, L-1, K]
        order = torch.argsort(keys, dim=-1)                                   # asc → value desc, id asc
        stable_topk_idx  = torch.gather(topk_idx,  dim=-1, index=order)       # [B, L-1, K]
        stable_topk_vals = torch.gather(topk_vals, dim=-1, index=order)       # [B, L-1, K]

        K = stable_topk_idx.shape[-1]
        target_exp = target_ids.unsqueeze(-1).expand(-1, -1, K)               # [B, L-1, K]
        match = (stable_topk_idx == target_exp)                                # [B, L-1, K]
        in_topk = match.any(dim=-1)                                            # [B, L-1]
        pos = match.float().argmax(dim=-1).long()                               # [B, L-1], 0..K-1

        # Encode ranks as 1..K for in-top-k; 0 otherwise
        ranks_encoded = torch.where(in_topk, pos + 1, torch.zeros_like(pos))   # [B, L-1]

        # ---- Soft-tie → exception (optional)
        tie_mask = torch.zeros_like(in_topk, dtype=torch.bool)                 # [B, L-1]
        if tie_as_exception and tie_eps_abs > 0.0:
            pos_clamped = pos.clamp(min=0, max=K - 1)
            val_pos = torch.gather(stable_topk_vals, -1, pos_clamped.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

            # Large sentinel for missing neighbors
            BIG = torch.tensor(torch.finfo(stable_topk_vals.dtype).max, device=stable_topk_vals.device)
            big_like = torch.full_like(val_pos, BIG)

            # Left neighbor
            has_left  = pos > 0
            left_idx  = (pos - 1).clamp(min=0)
            left_g    = torch.gather(stable_topk_vals, -1, left_idx.unsqueeze(-1)).squeeze(-1)
            left_val  = torch.where(has_left, left_g, big_like)

            # Right neighbor
            has_right = pos < (K - 1)
            right_idx = (pos + 1).clamp(max=K - 1)
            right_g   = torch.gather(stable_topk_vals, -1, right_idx.unsqueeze(-1)).squeeze(-1)
            right_val = torch.where(has_right, right_g, big_like)

            # Smallest neighbor gap inside top-k
            min_neighbor_diff = torch.minimum((val_pos - left_val).abs(), (val_pos - right_val).abs())

            tie_in_topk = in_topk & (min_neighbor_diff <= tie_eps_abs)

            # Border case: target outside top-k but close to K-th logit
            kth_vals     = stable_topk_vals[..., K - 1]
            target_vals  = logits_input.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            tie_at_border = (~in_topk) & ((target_vals - kth_vals).abs() <= tie_eps_abs)

            tie_mask = tie_in_topk | tie_at_border
            # Force rank to 0 where a soft tie is detected
            ranks_encoded = torch.where(tie_mask, torch.zeros_like(ranks_encoded), ranks_encoded)

        timers["topk_and_ranks"] += time.perf_counter() - start

        # --- (5) Filter PAD and split per sequence (ranks/exceptions/contexts) — identical
        start = time.perf_counter()
        mask_valid = (target_ids != pad_token_id)                              # [B, L-1]
        valid_counts = mask_valid.sum(dim=1).tolist()                          # [B]

        # Ranks: flatten valid positions then split back per sequence
        flat_ranks = ranks_encoded[mask_valid]                                 # [sum(valid)]
        split_ranks = torch.split(flat_ranks, valid_counts)                    # tuple of B tensors

        # Exceptions: outside top-k OR soft-tie flagged
        zeros_mask = mask_valid & (~in_topk | tie_mask)                        # [B, L-1]
        zeros_counts = zeros_mask.sum(dim=1).tolist()                          # [B]
        flat_exceptions = target_ids[zeros_mask]                               # [sum(zeros)]
        split_exceptions = torch.split(flat_exceptions, zeros_counts)          # tuple of B tensors

        # Accumulate outputs and contexts per sequence
        for b, (seq_r, seq_e) in enumerate(zip(split_ranks, split_exceptions)):
            all_ranks.append(seq_r.tolist())
            all_exceptions.append(seq_e.tolist())
            seq_ctx = batch_contexts[b][:valid_counts[b]]
            all_contexts_per_step.append(seq_ctx)

        timers["filter_and_split"] += time.perf_counter() - start
        global_seq_counter += B

    return all_ranks, all_exceptions, timers, all_contexts_per_step


# =====================================================
# These functions provide utility functions used in a past
# phase of the project. They are retained here for reference
# but are not actively used in the current codebase.
# In particular, these functions is used in the TestModels
# directory.
# =====================================================


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
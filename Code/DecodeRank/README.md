# Decoding ranks

This repository folder contains the code and experiments used to verify that language models can **losslessly decode** token ranks produced during the forward pass.
In short: for each token we compute a *rank* (where rank=0 is the top logit, rank=1 the 2nd, …) and then try to reconstruct the original token sequence from those ranks. These scripts test whether that round-trip (`token → rank → token`) reproduces the original tokens exactly.

---

## Table of contents

* [What’s in this folder](#whats-in-this-folder)
* [High-level pipeline](#high-level-pipeline)
* [Key configuration variables](#key-configuration-variables)
* [Important files & functions](#important-files--functions)
* [How to run (example)](#how-to-run-example)
* [Dependencies & environment](#dependencies--environment)

---

## What’s in this folder

This folder contains example scripts used to run experiments for different models (one script per tested model). Each script contains:

* preprocessing and chunking utilities,
* dataloader creation,
* `compute_rank` and `decode` implementations,
* verification utilities to check reconstruction correctness,

The goal is experimental verification: pick a model, configure dataset and chunking parameters, run the pipeline and check that reconstructed tokens match original tokens.

---

## High-level pipeline

1. **Preprocess & chunk** the input dataset into model token-id chunks (maintaining BOS handling).
2. **Sort & batch** chunks to create a dataloader.
3. **Forward pass** on the model to compute *token → rank* lists (`compute_token_ranks_*`).
4. **Reconstruct per-row rank sequences** using the mapping from chunks → original rows.
5. **Decode ranks → token ids** using the decoding routine (`decode_token_ids_from_ranks*`).
6. **Verify** the reconstructed token ids against tokenization of the original text (`verify_reconstruction`).
7. Print result (`all reconstructed OK` vs `some mismatches`).

---

## Key configuration variables

Before running, set the most important variables in the example script you want to run:

```py
language = "Python"      # which dataset column / language file to load
batch_size = 16          # number of chunks per batch
max_length = 256         # chunk length (BOS included)
```

**Notes**

* `max_length` determines chunk size; keep it ≤ model's max position embeddings.
* `batch_size` trades off throughput vs memory.
* Prefer passing the `device` variable (e.g. `device = "cuda" if torch.cuda.is_available() else "cpu"`) to functions instead of hard-coding `"cuda"`.

---

## Important files & functions

### `dataLoader.py`

* `preprocess_dataset_fast(...)` 
  Tokenizes inputs and builds chunked `input_id_list` + `mapping`. Use the `_unixcoder` variant when testing UniXcoder.
* `create_chunk_dataloader(input_id_list, batch_size=...)`
  Wraps chunk lists into a generator/dataloader for batched forward passes.
* `get_token_info(tokenizer)`
  Returns special token ids/info (BOS, EOS, PAD, etc).

### `utility.py`

* `sort_chunks_by_length(input_id_list, mapping, pad_token_id, descending=True)`
  Sort chunk lists to reduce padding and accelerate batches.
* `check_bos_token_in_chunks(input_id_list, tokenizer)`
  Validates that chunks have a BOS token in the right positions (especially important when rows are split across chunks).
* `verify_reconstruction(input_texts, reconstructed_tokens, tokenizer)`
  Checks if reconstructed token ids match tokenization of original text.

### `computeRank.py`

* `compute_token_ranks_fast(dataloader, model, pad_token_id, device)`
  Forward-pass routine that computes a rank list for each token (rank = position in sorted logits).
* `decode_token_ids_from_ranks_new(rank_sequences, model, tokenizer, max_length, pad_token_id, device, debug, show_progress, inner_progress)`
  Decodes sequences of ranks back to token ids. Parameters:

  * `debug` (bool): print step-by-step debug info.
  * `show_progress` (bool): show outer `tqdm` over chunks.
  * `inner_progress` (bool): show inner `tqdm` inside each chunk.

There are also Unixcoder-specific variants:

* `preprocess_dataset_fast_unixcoder(...)`
* `compute_token_ranks_fast_unixcoder(...)`
* `decode_token_ids_from_ranks_unixcoder(...)`

These accept and expect the `UniXcoder` object and its API.

---

## How to run (example)

1. Edit the example script for the model you want to test (each script corresponds to one model).
2. Set `language`, `batch_size`, `max_length`.
3. Run on the machine/host that has the GPU you want to use.

Example invocation used on the remote machine (runs in background, writes output to `output.txt`):

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u DeepSeekDecode.py > output.txt &
```

* `CUDA_VISIBLE_DEVICES=1` → choose the GPU id to use.
* `nohup ... &` → run process in the background.
* `python -u` → unbuffered output so logs appear in `output.txt` immediately.

---

## Expected output

At the end of each run the verification print shows whether all reconstructions matched:

```py
if ok:
    print("🎉 All reconstructions are correct!")
else:
    print("⚠️ Some reconstructions do not match.")
```

If mismatches occur, the `verify_reconstruction` debug output prints the text, expected ids/tokens and reconstructed ids/tokens to help debug.

## Dependencies & environment

Recommended minimal environment:

* Python 3.8+ (the experiments used Python 3.10.x)
* PyTorch (matching your CUDA / CPU setup)
* `transformers`
* `tqdm`
* `pandas`, `numpy`
* `awq` (if testing AWQ models / `AutoAWQForCausalLM`)
* optionally: `accelerate` for device management

Install essentials with pip (example):

```bash
pip install torch transformers tqdm pandas numpy
# pip install awq   # only if you use AWQ loader
```

---

## Adding / testing a new model

1. Add a new script (copy an existing one) and update:

   * `model_name`
   * which loader to use: generic HF vs UniXcoder
   * `language`, `batch_size`, `max_length`
2. If you use UniXcoder, instantiate `ux = UniXcoder(model_name)` and use `_unixcoder` helpers.
3. Run with the background command shown above, and inspect `output.txt`.





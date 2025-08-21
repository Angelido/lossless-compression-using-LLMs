# Code Compression

This folder contains the code and experiments used to study compression results obtained with a rank-based compression method applied to LLMs.  
In short: for each phrase, we tokenize it, compute a *rank* for each token (rank=0 is the top logit, rank=1 the second, …), and then compress the resulting list of ranks (one list per phrase).  
To speed things up, multiple phrases are combined in a dataloader.  
The workflow implemented in this folder is:  
`text → token + dataloader → rank → compression`.

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

This folder contains example scripts used to run experiments for different models (one script per tested model). Each script provides:

* preprocessing and chunking utilities,
* dataloader creation,
* a `compute_rank` function,
* a compression function,
* a function to save results.

The main goal is **experimental evaluation**: choose a model, configure dataset and parameters (chunking, compression, etc.), run the pipeline, and collect results.

---

## High-level pipeline

1. **Preprocess & chunk** the dataset into token-id sequences (handling BOS correctly).
2. **Sort & batch** chunks to build a dataloader.
3. **Forward pass** through the model to compute *token → rank* lists (`compute_token_ranks_*`).
4. **Reconstruct per-row rank sequences** using the chunk → row mapping.
5. **Compress the rank lists** with `compress_and_save`, which also writes the compressed file.
6. **Save metadata and results** with `save_info_to_csv`.

---

## Key configuration variables

Before running, set the key variables inside the script you want to execute:

```py
language = "Python"      # which dataset column / language file to load
batch_size = 32          # number of chunks per batch
max_length = 256         # chunk length (including BOS)

binary = True            # whether to use NumPy (.npy) or pickle for serialization
use_zstd = True          # whether to use zstd (True) or bzip2 (False)
compression_level = 3    # compression level (zstd: 1–22, bzip2: 1–9)
```

**Notes**

* `max_length` must not exceed the model’s maximum context length.
* `batch_size` balances throughput vs memory.
* `binary`: if True, convert data into NumPy → `.npy`; if False, use `pickle.dumps`.
* We typically tested zstd with `compression_level` 3, 12, 22 and bzip2 with `compression_level` 3, 9.
* Always prefer passing the `device` variable (e.g. `device = "cuda" if torch.cuda.is_available() else "cpu"`) instead of hardcoding `"cuda"`.

---

## Important files & functions

### `dataLoader.py`

* `preprocess_dataset_fast(input_texts, tokenizer, max_length, stride)`
  Tokenizes inputs and builds chunked `input_id_list` + `mapping`.
  Use `_unixcoder` variant when testing UniXcoder.
* `create_chunk_dataloader(input_id_list, batch_size)`
  Wraps chunks into a generator/dataloader for batched forward passes.

### `utility.py`

* `sort_chunks_by_length(input_id_list, mapping, pad_token_id, descending=True)`
  Sorts chunks by length to reduce padding and speed up batches.
* `save_info_to_csv(folder_path, csv_filename, row_dict)`
  Saves values from a `row_dict` into a CSV file in `folder_path`.
* `compress_and_save(reconstructed_rank_list, results_dir, *, binary, use_zstd, compression_level, filename_prefix)`
  Serializes and compresses `reconstructed_rank_list`, saving it in `results_dir`.
  `filename_prefix` is used as a prefix for the output filename (e.g. `"DeepSeek_rank_list.bzip2"`).

### `computeRank.py`

* `compute_token_ranks_fast(dataloader, model, pad_token_id, device)`
  Forward-pass routine that computes rank lists for each token.
  Use `_unixcoder` variant when testing UniXcoder.

Unixcoder-specific helpers:

* `preprocess_dataset_fast_unixcoder(...)`
* `compute_token_ranks_fast_unixcoder(...)`

Both expect a `UniXcoder` object and use its API.

---

## How to run (example)

1. Open the example script for the model you want to test (each model has its own script).
2. Set `language`, `batch_size`, `max_length`.
3. Configure `binary`, `use_zstd`, `compression_level`.
4. Run on a machine with GPU access.

Example (runs in background, writes logs to `output.txt`):

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u DeepSeekCompression.py > output.txt &
```

* `CUDA_VISIBLE_DEVICES=1` → selects GPU id 1.
* `nohup ... &` → runs process in background.
* `python -u` → unbuffered output, so logs are flushed to `output.txt` immediately.

---

## Expected output

At the end of a run, the script prints and saves the collected metadata into the CSV:

```py
print("=== End-of-execution information ===")
for key, value in row_dict.items():
    print(f"{key:25s}: {value}")
print("=======================================\n")
```

This includes dataset and compression parameters, execution times (total and partial), and the sizes of original vs compressed files.

---

## Dependencies & environment

Recommended minimal setup:

* Python 3.8+ (tested with Python 3.10.x)
* PyTorch (version matching your CUDA / CPU setup)
* `transformers`
* `pandas`, `numpy`
* `time`
* `awq` (if testing AWQ models with `AutoAWQForCausalLM`)
* optionally: `accelerate` for device management

Install essentials via pip:

```bash
pip install torch transformers pandas numpy time
# pip install awq   # only if you need the AWQ loader
```

---

## Adding / testing a new model

1. Copy an existing script and update:

   * `model_name`
   * loader type (generic HF vs UniXcoder)
   * `language`, `batch_size`, `max_length`
   * compression settings (`binary`, `use_zstd`, `compression_level`)
2. For UniXcoder, instantiate `ux = UniXcoder(model_name)` and use `_unixcoder` helpers.
3. Run as shown in the example command and inspect `output.txt`.

```


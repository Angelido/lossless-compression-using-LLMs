import os
import io
import bz2
import time
import struct
import numpy as np
import zstandard as zstd 
import subprocess
import shutil


# =====================================================
# This file provides utilities to serialize and compress 
# lists of lists of integers. It supports Zstandard,
# bzip2 (via Python stdlib), and bzip3 (via external CLI).
# =====================================================


# ====== _select_min_dtype ====== #
def _select_min_dtype(min_val: int, max_val: int):
    """
    Select the smallest NumPy integer dtype that can represent the closed interval
    [min_val, max_val]. When all values are non-negative (min_val >= 0), prefer
    unsigned types; otherwise choose the minimal signed type.

    Input:
    - min_val (int):
        Minimum value in the dataset/range.
    - max_val (int):
        Maximum value in the dataset/range.

    Return:
    - dtype (np.dtype):
        One of {np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64},
        guaranteed to cover the given range.
    """
    if min_val >= 0:
        # Prefer unsigned widths when the entire range is non-negative.
        if max_val <= np.iinfo(np.uint16).max:
            return np.uint16
        elif max_val <= np.iinfo(np.uint32).max:
            return np.uint32
        else:
            return np.uint64
    else:
        # Fall back to signed widths when negative values are possible.
        if (min_val >= np.iinfo(np.int16).min) and (max_val <= np.iinfo(np.int16).max):
            return np.int16
        elif (min_val >= np.iinfo(np.int32).min) and (max_val <= np.iinfo(np.int32).max):
            return np.int32
        else:
            return np.int64


# ====== _append_list_of_lists_to_buffer ====== #
def _append_list_of_lists_to_buffer(buf: io.BytesIO, lol: list[list[int]]) -> None:
    """
    Serialize a list-of-lists of integers into a binary buffer as two .npy payloads:
      1) lengths    : np.int32 array of sub-list lengths (shape [N])
      2) flat_array : contiguous array of all elements using the minimal integer dtype

    The function avoids Python pickle. Each np.save() call writes a portable header
    (dtype/shape) followed by raw data, enabling robust, version-tolerant deserialization.

    Input:
    - buf (io.BytesIO):
        In-memory binary buffer to which arrays are appended.
    - lol (list[list[int]]):
        List of variable-length integer lists to serialize. Empty sub-lists are allowed.

    Return:
    - None (appends two .npy arrays into 'buf')
    """
    # First, store the length of each sub-list as int32.
    lengths = np.array([len(lst) for lst in lol], dtype=np.int32)

    # If there are no elements in total, store an empty small-typed array.
    if lengths.sum() == 0:
        flat_array = np.array([], dtype=np.int16)
    else:
        # Compute min/max over all non-empty sub-lists and flatten in one pass.
        flat_python = []
        min_val = None
        max_val = None
        for lst in lol:
            if lst:
                lmin = min(lst)
                lmax = max(lst)
                # Track global min/max only over non-empty lists.
                min_val = lmin if (min_val is None or lmin < min_val) else min_val
                max_val = lmax if (max_val is None or lmax > max_val) else max_val
                flat_python.extend(lst)

        # If all sub-lists were empty, emit an empty small-typed array.
        if min_val is None:
            flat_array = np.array([], dtype=np.int16)
        else:
            # Choose the minimal storage dtype that covers [min_val, max_val].
            dtype = _select_min_dtype(min_val, max_val)
            flat_array = np.array(flat_python, dtype=dtype)

    # Persist both arrays as .npy payloads (portable dtype/shape headers + raw data).
    np.save(buf, lengths, allow_pickle=False)
    np.save(buf, flat_array, allow_pickle=False)


# ====== compress_and_save ====== #
def compress_and_save(
    reconstructed_rank_list: list[list[int]],
    exception_list: list[list[int]],
    max_length: int,              # intentional misspelling preserved as requested
    batch_size: int,
    results_dir: str,
    *,
    use_zstd: bool,
    compression_level: int,
    filename_prefix: str
) -> tuple[str, int, float]:
    """
    Serialize two list-of-lists (ranks and exceptions) into a single binary blob and
    compress it using either Zstandard or bzip2, then write the result to disk.

    On-disk binary layout (in order):
    1) max_lenght      : int64 little-endian 
    2) batch_size      : int64 little-endian 
      3) reconstructed_rank_list :
           - lengths     : np.int32 (.npy)
           - flat_array  : minimal integer dtype (.npy)
      4) exception_list :
           - lengths     : np.int32 (.npy)
           - flat_array  : minimal integer dtype (.npy)

    Compression:
      - If use_zstd=True  : Zstandard with 'compression_level'   → extension '.zst'
      - If use_zstd=False : bzip2 with 'compression_level'       → extension '.bz2'

    Input:
    - reconstructed_rank_list (list[list[int]]):
        Encoded ranks per sequence (variable length per row).
    - exception_list (list[list[int]]):
        Token IDs for positions marked as exceptions (rank==0) per sequence.
    - max_length (int):
        Maximum effective sequence length (stored as metadata).
    - batch_size (int):
        Batch size used during upstream processing (stored as metadata).
    - results_dir (str):
        Output directory (created if missing).
    - use_zstd (bool):
        Toggle between Zstandard (True) and bzip2 (False).
    - compression_level (int):
        Codec-specific compression level.
    - filename_prefix (str):
        Prefix for the output filename.

    Return:
    - outfile_path (str):
        Full path to the compressed file written to disk.
    - compressed_size_bytes (int):
        Size in bytes of the compressed output.
    - compression_time (float):
        Elapsed time in seconds for serialization + compression.
    """
    # Ensure output directory exists.
    os.makedirs(results_dir, exist_ok=True)

    start = time.perf_counter()

    # ===== 1) Build the binary blob in memory =====
    buf = io.BytesIO()

    # Write metadata fields as 64-bit little-endian signed integers.
    buf.write(struct.pack("<q", int(max_length)))
    buf.write(struct.pack("<q", int(batch_size)))

    # Append both list-of-lists payloads (.npy lengths + .npy flat array).
    _append_list_of_lists_to_buffer(buf, reconstructed_rank_list)
    _append_list_of_lists_to_buffer(buf, exception_list)

    data_blob = buf.getvalue()

    # ===== 2) Compress =====
    if use_zstd:
        compressor = zstd.ZstdCompressor(level=int(compression_level))
        compressed_data = compressor.compress(data_blob)
        ext = "zst"
        compressor_name = f"zstd{compression_level}"
    else:
        compressed_data = bz2.compress(data_blob, compresslevel=int(compression_level))
        ext = "bz2"
        compressor_name = f"bzip2-{compression_level}"

    end = time.perf_counter()
    compression_time = end - start

    # ===== 3) Filename and write to disk =====
    # Preserve "binary" token in the filename for continuity with prior artifacts.
    filename = f"{filename_prefix}_binary_{compressor_name}.{ext}"
    outfile_path = os.path.join(results_dir, filename)

    # Atomic enough for most scenarios; caller can fsync if needed.
    with open(outfile_path, "wb") as f_out:
        f_out.write(compressed_data)

    compressed_size_bytes = len(compressed_data)
    return outfile_path, compressed_size_bytes, compression_time


# ====== compress_and_save_bz3 ====== #
def compress_and_save_bz3(
    reconstructed_rank_list: list[list[int]],
    exception_list: list[list[int]],
    max_length: int,
    batch_size: int,
    results_dir: str,
    *,
    filename_prefix: str,
    keep_default_block: bool = True,
    best: bool = True,
) -> tuple[str, int, float]:
    """
    Serialize inputs into a single binary blob (no pickle) and compress with the system
    `bzip3` CLI (stdin→stdout), then write the result to disk.

    On-disk layout (identical to the zstd/bzip2 variant):
      1) max_length            : int64 little-endian
      2) batch_size            : int64 little-endian
      3) reconstructed_rank_list : .npy lengths(int32) + .npy flat(minimal int dtype)
      4) exception_list          : .npy lengths(int32) + .npy flat(minimal int dtype)

    Compression (bzip3):
      - Invokes the external 'bzip3' executable, capturing stdout in-memory.
      - Threads: always forces the maximum (64) via `-j 64`.
      - Block size:
          * If keep_default_block=True  → use bzip3 defaults (e.g., 16 MiB).
          * If keep_default_block=False → pass '-b 64' (best) or '-b 1' (fast) per 'best'.

    Input:
    - reconstructed_rank_list (list[list[int]]):
        Encoded ranks per sequence.
    - exception_list (list[list[int]]):
        Token IDs for positions marked as exceptions (rank==0).
    - max_length (int):
        Maximum effective sequence length (stored as metadata).
    - batch_size (int):
        Batch size used during upstream processing (stored as metadata).
    - results_dir (str):
        Output directory (created if missing).
    - filename_prefix (str):
        Prefix used to build the output filename.
    - keep_default_block (bool):
        If True, do not override bzip3's default block size.
    - best (bool):
        If keep_default_block is False, choose '-b 64' (True) or '-b 1' (False).

    Return:
    - outfile_path (str):
        Full path to the compressed file written to disk.
    - compressed_size_bytes (int):
        Size in bytes of the compressed output.
    - compression_time (float):
        Elapsed time in seconds for serialization + compression (including bzip3).
    """
    # Ensure output directory exists.
    os.makedirs(results_dir, exist_ok=True)

    # Verify that 'bzip3' is installed and accessible.
    exe = shutil.which("bzip3")
    if not exe:
        raise RuntimeError("Cannot find 'bzip3' in PATH (e.g., on macOS: `brew install bzip3`).")

    # ===== 1) Build the binary blob =====
    start = time.perf_counter()

    buf = io.BytesIO()
    # Metadata: 64-bit little-endian signed integers.
    buf.write(struct.pack("<q", int(max_length)))
    buf.write(struct.pack("<q", int(batch_size)))
    # Payloads: two .npy arrays per list-of-lists.
    _append_list_of_lists_to_buffer(buf, reconstructed_rank_list)
    _append_list_of_lists_to_buffer(buf, exception_list)
    data_blob = buf.getvalue()

    # ===== 2) Compress with bzip3 (stdin -> stdout) =====
    argv = [exe, "-j", "64", "-c"]  # force 64 worker threads; write compressed bytes to stdout

    block_tag = "default"
    if not keep_default_block:
        argv += ["-b", "64" if best else "1"]
        block_tag = "b64" if best else "b1"

    # Run the external compressor; capture both stdout (payload) and stderr (diagnostics).
    proc = subprocess.run(
        argv,
        input=data_blob,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bzip3 exit {proc.returncode}: {proc.stderr.decode(errors='ignore')}")

    compressed_data = proc.stdout

    end = time.perf_counter()
    compression_time = end - start

    # ===== 3) Filename + write =====
    compressor_name = f"bz3-{block_tag}"
    filename = f"{filename_prefix}_binary_{compressor_name}.bz3"
    outfile_path = os.path.join(results_dir, filename)

    with open(outfile_path, "wb") as f_out:
        f_out.write(compressed_data)

    compressed_size_bytes = len(compressed_data)
    return outfile_path, compressed_size_bytes, compression_time

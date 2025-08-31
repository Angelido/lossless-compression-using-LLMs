import pandas as pd
import os
import io
import tarfile
import bz2
import subprocess
import shutil
import zstandard as zstd
import time
import sys

# Read also files from the parent folder (utility, dataLoader, computeRank)   
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility import save_info_to_csv


# ===================================================
# This file performs classical compression directly on the
# language datasets, without using an LLM to compute ranks.  
# The results of these calculations serve as a baseline
# for comparison with the rank-based method.
# ===================================================


#============ all_path_end_with_ext ============#
def all_path_end_with_ext(
    df: pd.DataFrame,
    extension: str, 
    column: str = 'path'
) -> bool:
    """
    Check if all entries in the specified DataFrame column end with '.extension'.

    Input:
    - df (pd.DataFrame): DataFrame containing at least the column `column`.
    - extension (str): the extension to check for (without dot).
    - column (str): name of the column to check (default: 'path').

    Return:
    - end_with_ext (bool): True if all strings in df[column] end with '.extension'; otherwise False.
    """
    # Ensure the column exists
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in the DataFrame.")
    
    # Use .str.endswith to create a boolean Series
    ends_with_ext = df[column].astype(str).str.endswith(f'.{extension}')

    return ends_with_ext.all()


#============ invalid_ext_paths ============#
def invalid_ext_paths(
    df: pd.DataFrame, 
    extension: str,
    column: str = 'path'
) -> pd.Series:
    """
    Return the entries of df[column] that do not end with '.extension'.
    
    Input:
    - df (pd.DataFrame): pd.DataFrame containing at least the column `column`.
    - extension (str): the extension to check for (without dot).
    - column (str): name of the column to check (default: 'path').
    
    Return:
    - (pd.Series): series containing all values from df[column] that do not end with '.extension'.
    """
    # Ensure the column exists
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    
    # Create a mask for entries that do not end with the given extension
    mask = ~df[column].astype(str).str.endswith(f'.{extension}')
    
    return df.loc[mask, column]


#============ fix_path ============#
def fix_path(
    path: str,
    extension: str
) -> str:
    """
    Take a file path (e.g., '/dir/ file', '/dir/file.ext', '/dir/file.txt~', etc.) and return a path where:
    - All spaces are removed.
    - If the final element (basename) has no '.', append '.extension'
    - Otherwise, replace the existing extension (everything after the last '.') with '.extension'
    
    Input:
    - path (str): original file path, possibly containing spaces or an unwanted extension.
    - extension (str): the extension to use at the end of the path.
    
    Return:
    - (str): cleaned file path ending with the '.extension' extension.
    """

    # Remove all spaces from the path
    clean_path = path.replace(" ", "")
    
    # Split directory and filename, and split base and extension
    dirpath, filename = os.path.split(clean_path)
    base, ext = os.path.splitext(filename)
    
    # Create the new filename with the desired extension
    new_filename = base + f".{extension}"
    
    return os.path.join(dirpath, new_filename)


#============ invert_full_path ============#
def invert_full_path(
    path: str,
    default_extension: str
) -> str:
    """
    Given a path like '/folder1/ folder2 /name.extension', return
    'extension.name/folder2/folder1/' after removing unnecessary spaces.

    Input:
    - path (str): original file path, possibly containing extra spaces or leading/trailing slashes.
    - extension (str): fallback extension to use if the filename has no extension.

    Return:
    - (str): inverted path composed of 'extension.basename/' followed by reversed directory segments, 
             each separated by '/' and ending with a slash.
    """
    # Remove leading/trailing slashes
    cleaned = path.strip("/")
    
    # Split path into segments and strip spaces around each
    parts = [p.strip() for p in cleaned.split("/") if p.strip() != ""]
    if not parts:
        return ""
    
    # Get the filename and strip spaces
    filename = parts[-1].strip()
    
    # Separate basename and extension
    base, ext = os.path.splitext(filename)
    ext_clean = ext.lstrip('.') if ext else default_extension
    base_clean = base.strip()
    
    # Create the 'extension.basename' part
    inverted_filename = f"{ext_clean}.{base_clean}"
    
    # Reverse the cleaned directory parts
    dirs = [dir_part.strip() for dir_part in parts[:-1]]
    dirs_reversed = list(reversed(dirs))
    
    # Reconstruct the final inverted path
    if not dirs_reversed:
        return f"{inverted_filename}/"
    else:
        return f"{inverted_filename}/" + "/".join(dirs_reversed) + "/"


#============ create_tar_from_texts ============#
def create_tar_from_texts(
    df: pd.DataFrame, 
    tar_output: str
) -> None:
    """
    Given a DataFrame with columns:
    - `text`: Code file content (string)
    - `inverted_path`: desired name (with directory structure) inside the archive (no trailing slash)
    Create a non-compressed .tar where each entry is built from df[`text`],
    using df[`inverted_path`] as the internal file path/name.

    Input:
    - df (pd.Dataframe): DataFrame with at least the columns `text` and `inverted_path`
    - tar_output (str): path (name included) of the file .tar to create
    """
    # 1) Apriamo (o creiamo) il file .tar in modalità scrittura non compressa
    with tarfile.open(tar_output, mode='w') as tar:
        # 2) Per ogni riga del DataFrame...
        for idx, row in df.iterrows():
            content_str = row['text']
            arcname_raw = row['inverted_path']

            # 2.1) Normalizziamo l'arcname: rimuoviamo slash finale se presente
            arcname = arcname_raw.rstrip('/')

            # 2.2) Convertiamo il testo in bytes
            data_bytes = content_str.encode('utf-8')
            bytes_io = io.BytesIO(data_bytes)

            # 2.3) Creiamo un oggetto TarInfo per questa voce
            tarinfo = tarfile.TarInfo(name=arcname)
            tarinfo.size = len(data_bytes)

            # 2.4) Aggiungiamo l’entry al tar direttamente dal buffer di memoria
            tar.addfile(tarinfo, fileobj=bytes_io)
    print(f"Archivio .tar creato con successo: {tar_output}")
    

#============ compress_bz3 ============#
def compress_bz3(
    input_tar_path: str,
    output_bz3_path: str,
    best: bool = True,
    keep_default_block: bool = True
) -> int:
    """
    Compress an existing .tar using the system 'bzip3' CLI.

    Parameters
    ----------
    input_tar_path : str
        Path to the .tar file to compress.
    output_bz3_path : str
        Output .bz3 path.
    best : bool
        Logical "best vs fast". When keep_default_block=True (default), it does
        NOT change the ratio because bzip3 CLI has no --best/--fast. It's kept
        only for API symmetry. If keep_default_block=False, maps to -b 64 (best)
        or -b 1 (fast).
    keep_default_block : bool
        If True (default), DO NOT set -b (use bzip3 default, 16 MiB).
        If False, emulate best/fast by setting -b accordingly.

    Returns
    -------
    int
        Size in bytes of the created .bz3 file.
    """
    exe = shutil.which("bzip3")
    if not exe:
        raise RuntimeError("Non trovo 'bzip3' nel PATH (su macOS: `brew install bzip3`).")

    # Always use all CPU threads
    threads = max(1, os.cpu_count() or 1)
    argv = [exe, "-j", str(threads)]

    # Optionally emulate best/fast by touching block size
    if not keep_default_block:
        argv += ["-b", "64" if best else "1"]

    # Stream to stdout and redirect to output file
    argv += ["-c", input_tar_path]

    with open(output_bz3_path, "wb") as f_out:
        proc = subprocess.run(argv, stdout=f_out, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"bzip3 CLI exit {proc.returncode}: {proc.stderr.decode(errors='ignore')}")

    return os.path.getsize(output_bz3_path)
    

#============ compress_bz2 ============#
def compress_bz2(
    input_tar_path: str, 
    output_bz2_path: str, 
    compresslevel: int = 3
) -> int:
    """
    Compress an existing tar file using bzip2 with the specified compression level.

    Input:
    - input_tar_path (str): path to the .tar file to compress.
    - output_bz2_path (str): path (including .bz2 filename) for the output compressed file.
    - compresslevel (int): bzip2 compression level (1-9). Default = 3.

    Return:
    - (int): size in bytes of the created compressed file.
    """

    # 1) Leggiamo tutto il contenuto del file tar
    with open(input_tar_path, 'rb') as f_in:
        tar_data = f_in.read()
    
    # 2) Comprimiamo utilizzando il modulo bz2
    compressed_data = bz2.compress(tar_data, compresslevel=compresslevel)
    
    # 3) Salviamo il risultato su disco
    with open(output_bz2_path, 'wb') as f_out:
        f_out.write(compressed_data)
    
    # 4) Misuriamo e restituiamo la dimensione del file compresso
    compressed_size = os.path.getsize(output_bz2_path)
    return compressed_size


#============ compress_bz2 ============#
def compress_zstd(
    input_tar_path: str, 
    output_zst_path: str, 
    compresslevel: int = 3
) -> int:
    """
    Compress an existing .tar file using Zstandard with the specified compression level.

    Input:
    - input_tar_path (str): path to the .tar file to compress.
    - output_zst_path (str): path (including .zst filename) for the output compressed file.
    - compresslevel (int): Zstandard compression level (1-22). Default = 3.

    Return:
    - (int): size in bytes of the created compressed file.
    """
    # 1) Leggiamo tutto il contenuto del file tar
    with open(input_tar_path, 'rb') as f_in:
        tar_data = f_in.read()

    # 2) Creiamo un compressore Zstandard
    compressor = zstd.ZstdCompressor(level=compresslevel)
    compressed_data = compressor.compress(tar_data)

    # 3) Scriviamo il file compresso
    with open(output_zst_path, 'wb') as f_out:
        f_out.write(compressed_data)

    # 4) Ritorniamo la dimensione del file zst
    compressed_size = os.path.getsize(output_zst_path)
    return compressed_size


#==============================#
#============ main ============#
#==============================#


if __name__ == "__main__":
    
    # Variables to change during experiments
    language = "Java"  # Change to the desired language
    extension = "java"  # Change to the desired file extension (e.g., "py", "java", "cpp", etc.)
    
    # Define output directories
    tar_dir = "Results/PlainText/Tar"
    compressed_dir = "Results/PlainText/CompressedFile"

    # Ensure directories exist
    os.makedirs(tar_dir, exist_ok=True)
    os.makedirs(compressed_dir, exist_ok=True)
    
    # File names
    output_tar = os.path.join(tar_dir, f"all_{language}_sources.tar")
    output_bz2_3 = os.path.join(compressed_dir, f"all_{language}_sources_bzip2-3.bz2")
    output_bz2_9 = os.path.join(compressed_dir, f"all_{language}_sources_bzip2-9.bz2")
    output_bz3_default = os.path.join(compressed_dir, f"all_{language}_sources_bzip3-default.bz3")
    output_zstd3 = os.path.join(compressed_dir, f"all_{language}_sources_zstd3.zst")
    output_zstd12 = os.path.join(compressed_dir, f"all_{language}_sources_zstd12.zst")
    output_zstd22 = os.path.join(compressed_dir, f"all_{language}_sources_zstd22.zst")
    
    # Load the dataset
    df = pd.read_csv(f"Dataset/{language}100MB.csv")
    total_bytes = df["length_bytes"].sum()
    
    #================ text preprocessing ================#
    
    start_text_preprocessing = time.perf_counter()
    
    # Apply path normalization
    df['fixed_path'] = df['path'].apply(lambda p: fix_path(p, extension))
    
    # Check if there is an error during normalization
    if not all_path_end_with_ext(df, extension, "fixed_path"):
        raise ValueError(f"Error: Some entries in 'fixed_path' do not end with '.{extension}'.")
    
    # Generate inverted paths and sort
    df["inverted_path"] = df["fixed_path"].apply(lambda p: invert_full_path(p, extension))
    df = df.sort_values('inverted_path').reset_index(drop=True)
    
    end_text_preprocessing = time.perf_counter()
    time_text_preprocessing = end_text_preprocessing - start_text_preprocessing  
    
    #================ .tar creation ================#
    
    start_create_tar = time.perf_counter()
    
    # Create tar archive
    create_tar_from_texts(df, output_tar)
    
    end_create_tar = time.perf_counter()
    time_create_tar = end_create_tar - start_create_tar   
    
    #================ bzip2 compression ================#
    
    start__bzip2_3 = time.perf_counter()
    
    # Compress with bzip2
    size_bytes_bz2_3 = compress_bz2(output_tar, output_bz2_3, compresslevel=3)
    
    end_bzip2_3 = time.perf_counter()
    time_bzip2_3 = end_bzip2_3 - start__bzip2_3 
    total_time_bzip2=time_bzip2_3+time_create_tar+time_text_preprocessing
    
    #================ bzip2-9 compression ================#
    
    start__bzip2_9 = time.perf_counter()
    
    # Compress with bzip2 level 9
    size_bytes_bz2_9 = compress_bz2(output_tar, output_bz2_9, compresslevel=9)
    
    end_bzip2_9 = time.perf_counter()
    time_bzip2_9 = end_bzip2_9 - start__bzip2_9 
    total_time_bzip2_9 = time_bzip2_9 + time_create_tar + time_text_preprocessing

    #================ bzip3 default compression ================#
    
    start__bzip3_default = time.perf_counter()
    
    # Compress with bzip3 deafault settings
    size_bytes_bz3_default = compress_bz3(output_tar, output_bz3_default, best=False, keep_default_block=True)
    
    end_bzip3_default = time.perf_counter()
    time_bzip3_default = end_bzip3_default - start__bzip3_default 
    total_time_bzip3_default = time_bzip3_default + time_create_tar + time_text_preprocessing
       
    #================ zstd3 compression ================#
    
    start__zstd3 = time.perf_counter()
    
    # Compress with Zstandard level 3
    size_bytes_zst3 = compress_zstd(output_tar, output_zstd3, compresslevel=3)
    
    end_zstd3 = time.perf_counter()
    time_zstd3 = end_zstd3 - start__zstd3
    total_time_zstd3=time_zstd3+time_text_preprocessing+time_create_tar   
    
    #================ zstd12 compression ================#
    
    start__zstd12 = time.perf_counter()
    
    # Compress with Zstandard level 12
    size_bytes_zst12 = compress_zstd(output_tar, output_zstd12, compresslevel=12)
    
    end_zstd12 = time.perf_counter()
    time_zstd12 = end_zstd12 - start__zstd12
    total_time_zstd12 = time_zstd12+time_text_preprocessing+time_create_tar
    
    #================ zstd22 compression ================#
    
    start__zstd22 = time.perf_counter()
    
    # Compress with Zstandard level 22
    size_bytes_zst22 = compress_zstd(output_tar, output_zstd22, compresslevel=22)
    
    end_zstd22 = time.perf_counter()
    time_zstd22 = end_zstd22 - start__zstd22
    total_time_zstd22 = time_zstd22 + time_text_preprocessing + time_create_tar
    
    #================ saving results ================#
    
    info_dir = "Results"
    csv_file = "Plain_Compression_Info_Final.csv"
    
    # Create a dictionary with the information of bzip2
    row_dict_bzip2_3 = {
        "language": language,
        "compression": "bzip2-3",
        "output_file": output_bz2_3,
        "text_preprocessing_time_s": round(time_text_preprocessing, 4),
        "create_tar_time_s": round(time_create_tar, 4),
        "compression_time_s": round(time_bzip2_3, 4),
        "total_time_s": round(total_time_bzip2, 4),
        "original_size_bytes": total_bytes,
        "compressed_size_bytes": size_bytes_bz2_3,
        "throughput_MBps": round(total_bytes / total_time_bzip2 / (1024 * 1024), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create a dictionary with the information of bzip2-9
    row_dict_bzip2_9 = {
        "language": language,
        "compression": "bzip2-9",
        "output_file": output_bz2_9,
        "text_preprocessing_time_s": round(time_text_preprocessing, 4),
        "create_tar_time_s": round(time_create_tar, 4),
        "compression_time_s": round(time_bzip2_9, 4),
        "total_time_s": round(total_time_bzip2_9, 4),
        "original_size_bytes": total_bytes,
        "compressed_size_bytes": size_bytes_bz2_9,
        "throughput_MBps": round(total_bytes / total_time_bzip2_9 / (1024 * 1024), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create a dictionary with the information of bzip3_default
    row_dict_bzip3_default = {
        "language": language,
        "compression": "bzip3-default",
        "output_file": output_bz3_default,
        "text_preprocessing_time_s": round(time_text_preprocessing, 4),
        "create_tar_time_s": round(time_create_tar, 4),
        "compression_time_s": round(time_bzip3_default, 4),
        "total_time_s": round(total_time_bzip3_default, 4),
        "original_size_bytes": total_bytes,
        "compressed_size_bytes": size_bytes_bz3_default,
        "throughput_MBps": round(total_bytes / total_time_bzip3_default / (1024 * 1024), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create a dictionary with the information of zstd3
    row_dict_zstd3 = {
        "language": language,
        "compression": "zstd3",
        "output_file": output_zstd3,
        "text_preprocessing_time_s": round(time_text_preprocessing, 4),
        "create_tar_time_s": round(time_create_tar, 4),
        "compression_time_s": round(time_zstd3, 4),
        "total_time_s": round(total_time_zstd3, 4),
        "original_size_bytes": total_bytes,
        "compressed_size_bytes": size_bytes_zst3,
        "throughput_MBps": round(total_bytes / total_time_zstd3 / (1024 * 1024), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create a dictionary with the information of zstd12
    row_dict_zstd12 = {
        "language": language,
        "compression": "zstd12",
        "output_file": output_zstd12,
        "text_preprocessing_time_s": round(time_text_preprocessing, 4),
        "create_tar_time_s": round(time_create_tar, 4),
        "compression_time_s": round(time_zstd12, 4),
        "total_time_s": round(total_time_zstd12, 4),
        "original_size_bytes": total_bytes,
        "compressed_size_bytes": size_bytes_zst12,
        "throughput_MBps": round(total_bytes / total_time_zstd12 / (1024 * 1024), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create a dictionary with the information of zstd22
    row_dict_zstd22 = {
        "language": language,
        "compression": "zstd22",
        "output_file": output_zstd22,
        "text_preprocessing_time_s": round(time_text_preprocessing, 4),
        "create_tar_time_s": round(time_create_tar, 4),
        "compression_time_s": round(time_zstd22, 4),
        "total_time_s": round(total_time_zstd22, 4),
        "original_size_bytes": total_bytes,
        "compressed_size_bytes": size_bytes_zst22,
        "throughput_MBps": round(total_bytes / total_time_zstd22 / (1024 * 1024), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save row_dict on the csv
    save_info_to_csv(info_dir, csv_file, row_dict_bzip2_3)
    save_info_to_csv(info_dir, csv_file, row_dict_bzip2_9)
    save_info_to_csv(info_dir, csv_file, row_dict_bzip3_default)
    save_info_to_csv(info_dir, csv_file, row_dict_zstd3)
    save_info_to_csv(info_dir, csv_file, row_dict_zstd12)
    save_info_to_csv(info_dir, csv_file, row_dict_zstd22)
    
    #================ print results on screen ================#
    
    print("=== End execution information of bzip2-3 compression ===")
    for key, value in row_dict_bzip2_3.items():
        print(f"{key:25s}: {value}")
    print("=======================================\n")
    
    print("=== End execution information of bzip2-9 compression ===")
    for key, value in row_dict_bzip2_9.items():
        print(f"{key:25s}: {value}")
    print("=======================================\n")
    
    print("=== End execution information of bzip3-default compression ===")
    for key, value in row_dict_bzip3_default.items():
        print(f"{key:25s}: {value}")
    print("=======================================\n")  
    
    print("=== End execution information of zstd3 compression ===")
    for key, value in row_dict_zstd3.items():
        print(f"{key:25s}: {value}")
    print("=======================================\n")
    
    print("=== End execution information of zstd12 compression ===")
    for key, value in row_dict_zstd12.items():
        print(f"{key:25s}: {value}")
    print("=======================================\n")
    
    print("=== End execution information of zstd22 compression ===")
    for key, value in row_dict_zstd22.items():
        print(f"{key:25s}: {value}")
    print("=======================================\n")
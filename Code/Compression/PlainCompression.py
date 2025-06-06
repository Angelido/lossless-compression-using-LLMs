import pandas as pd
import os
import io
import tarfile
import bz2
import zstandard as zstd
import time
import sys

# Read also files from the parent folder (utility, dataLoader, computeRank)   
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility import save_info_to_csv

#============ all_path_end_with_py ============#
def all_path_end_with_py(
    df: pd.DataFrame, 
    column: str = 'path'
) -> bool:
    """
    Check if all entries in the specified DataFrame column end with '.py'.

    Input:
    - df (pd.DataFrame): pd.DataFrame containing at least the column `column`.
    - column (str): name of the column to check (default: 'path').

    Return:
    - end_with_py (bool): True if all strings in df[column] end with '.py'; otherwise False.
    """
    # 1) Assicuriamoci che la colonna esista
    if column not in df.columns:
        raise KeyError(f"Colonna '{column}' non trovata nel DataFrame.")
    
    # 2) Usiamo .str.endswith per ottenere una Serie di booleani
    ends_with_py = df[column].astype(str).str.endswith('.py')
    
    # 3) Ritorniamo True solo se TUTTI i valori sono True
    return ends_with_py.all()


#============ invalid_py_paths ============#
def invalid_py_paths(
    df: pd.DataFrame, 
    column: str = 'path'
) -> pd.Series:
    """
    Return the entries of df[column] that do NOT end with '.py'.
    
    Input:
    - df (pd.DataFrame): pd.DataFrame containing at least the column `column`.
    - column (str): name of the column to check (default: 'path').
    
    Return:
    - (pd.Series): series containing all values from df[column] that do not end with '.py'.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    mask = ~df[column].astype(str).str.endswith('.py')
    return df.loc[mask, column]



#============ fix_path ============#
def fix_path(
    path: str
) -> str:
    """
    Take a file path (e.g., '/dir/ file', '/dir/file.ext', '/dir/file.txt~', etc.) and return a path where:
    - All spaces are removed.
    - If the final element (basename) has no '.', append '.py'
    - Otherwise, replace the existing extension (everything after the last '.') with '.py'
    
    Input:
    - path (str): original file path, possibly containing spaces or an unwanted extension.
    
    Return:
    - (str): cleaned file path ending with the '.py' extension.
    """

    # 1) Rimuoviamo tutti gli spazi dal path
    clean_path = path.replace(" ", "")
    
    # 2) Separiamo directory e nome file
    dirpath, filename = os.path.split(clean_path)
    
    # 3) Usiamo os.path.splitext per dividere nome base e estensione
    base, ext = os.path.splitext(filename)
    
    # 4) Costruiamo il nuovo filename:
    if ext == "":
        # Non c’era estensione: basta aggiungere ".py"
        new_filename = base + ".py"
    else:
        # C’era un'estensione: togliamo l’ext e mettiamo ".py"
        new_filename = base + ".py"
    
    # 5) Ricostruiamo il percorso completo (os.path.join ignorerà dirpath se è vuoto)
    return os.path.join(dirpath, new_filename)



#============ invert_full_path ============#
def invert_full_path(
    path: str
) -> str:
    """
    Given a path like '/folder1/ folder2 /name.py', return
    'py.name/folder2/folder1/' after removing unnecessary spaces.

    Input:
    - path (str): original file path, possibly containing extra spaces or leading/trailing slashes.

    Return:
    - (str): inverted path composed of 'extension.basename/' followed by reversed directory segments, each separated by '/' and ending with a slash.
    """
    # 1) Rimuoviamo slash iniziali e finali
    cleaned = path.strip("/")
    
    # 2) Splittiamo in segmenti e togliamo spazi attorno a ciascuno
    parts = [p.strip() for p in cleaned.split("/") if p.strip() != ""]
    
    if not parts:
        return ""
    
    # 3) Prendiamo il filename e lo ripuliamo di spazi
    filename = parts[-1].strip()
    
    # 4) Separiamo base ed estensione e li ripuliamo
    base, ext = os.path.splitext(filename)
    ext_clean = ext.lstrip('.') if ext else "py"
    base_clean = base.strip()
    
    # 5) Creiamo la parte invertita "py.base"
    inverted_filename = f"{ext_clean}.{base_clean}"
    
    # 6) Prendiamo e invertiamo le cartelle, togliendo spazi
    dirs = [dir_part.strip() for dir_part in parts[:-1]]
    dirs_reversed = list(reversed(dirs))
    
    # 7) Ricostruiamo il path invertito senza spazi o doppi slash
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
    - `text`: Python file content (string)
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
    language = "Python"
    suffix = ".py"
    
    # Define output directories
    tar_dir = "Results/PlainText/Tar"
    compressed_dir = "Results/PlainText/CompressedFile"

    # Ensure directories exist
    os.makedirs(tar_dir, exist_ok=True)
    os.makedirs(compressed_dir, exist_ok=True)
    
    # File names
    output_tar = os.path.join(tar_dir, f"all_{language}_sources.tar")
    output_bz2 = os.path.join(compressed_dir, f"all_{language}_sources_bzip2-3.bz2")
    output_zstd3 = os.path.join(compressed_dir, f"all_{language}_sources_zstd3.zst")
    output_zstd12 = os.path.join(compressed_dir, f"all_{language}_sources_zstd12.zst")
    
    # Load the dataset
    df = pd.read_csv(f"Dataset/{language}100MB.csv")
    total_bytes = df["length_bytes"].sum()
    
    #================ text preprocessing ================#
    
    start_text_preprocessing = time.perf_counter()
    
    # Apply path normalization
    df['fixed_path'] = df['path'].apply(fix_path)
    
    # Check if there is an error during normalization
    if not all_path_end_with_py(df, "fixed_path"):
        raise ValueError("Error: Some entries in 'fixed_path' do not end with '.py'.")
    
    # Generate inverted paths and sort
    df["inverted_path"] = df["fixed_path"].apply(invert_full_path)
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
    
    start__bzip2 = time.perf_counter()
    
    # Compress with bzip2
    size_bytes_bz2 = compress_bz2(output_tar, output_bz2, compresslevel=3)
    
    end_bzip2 = time.perf_counter()
    time_bzip2 = end_bzip2 - start__bzip2 
    total_time_bzip2=time_bzip2+time_create_tar+time_text_preprocessing   
    
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
    
    #================ saving results ================#
    
    info_dir = "Results"
    csv_file = "Plain_text_compression_info.csv"
    
    # Create a dictionary with the information of bzip2
    row_dict_bzip2 = {
        "language": language,
        "compression": "bzip2-3",
        "output_file": output_bz2,
        "text_preprocessing_time_s": round(time_text_preprocessing, 4),
        "create_tar_time_s": round(time_create_tar, 4),
        "compression_time_s": round(time_bzip2, 4),
        "total_time_s": round(total_time_bzip2, 4),
        "original_size_bytes": total_bytes,
        "compressed_size_bytes": size_bytes_bz2,
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
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save row_dict on the csv
    save_info_to_csv(info_dir, csv_file, row_dict_bzip2)
    save_info_to_csv(info_dir, csv_file, row_dict_zstd3)
    save_info_to_csv(info_dir, csv_file, row_dict_zstd12)
    
    
    #================ print results on screen ================#
    
    print("=== End execution information of bzip2-3 compression ===")
    for key, value in row_dict_bzip2.items():
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
import pandas as pd
import numpy as np
from collections import Counter
import time
import os
import subprocess



#====== prepare_text_for_newscan ======
def prepare_text_for_newscan(
    df: pd.DataFrame, 
    column: str, 
    output_path: str, 
    separator: str = "\x01\x02\x03ENDLINE\x03\x02\x01"
) -> None:
    """
    Merge the specified column of a DataFrame into a single text file.
    
    Input:
    - df: DataFrame containing the data.
    - column: Name of the column to merge.
    - output_path: Path where the output text file will be saved.
    - separator: String used to separate the texts in the output file.
    """
    assert column in df.columns, f"Column '{column}' not found in DataFrame."
    
    # Check if the separator is unique in the column
    if df[column].str.contains(separator).any():
        raise ValueError("The separator is not unique in the column. Please choose a different separator.")
    
    # Merge the column into a single string
    full_text = separator.join(df[column].astype(str).tolist())
    
    # Save the merged text to the specified output path
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"Text file ready: '{output_path}'")



#====== run_newscan ======
def run_newscan(
    input_filename: str, 
    w: int =10, 
    p: int =1000
)-> float:
    """
    Run the newscan.x program on the specified input file with given parameters.
    
    Input:
    - input_filename: Name of the input file to process.
    - w: Window size for newscan.
    - p: Integer on wich modulo is applied.
    
    Return:
    - elapsed: Time taken to run the newscan program.
    """
    # Paths to the input file and newscan.x
    input_path = os.path.join("Dataset", input_filename)
    newscan_path = os.path.join("Big-BWT-master", "newscan.x")
    
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if not os.path.isfile(newscan_path):
        raise FileNotFoundError(f"Compile newscan.x first: {newscan_path}")

    # Command to run newscan.x
    cmd = [newscan_path, "-c", "-w", str(w), "-p", str(p), input_path]

    # Misura il tempo
    start_time = time.perf_counter()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error during execution: {e}")
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    
    return elapsed



#====== analyze_parsing_files ======
def analyze_parsing_files(
    base_filename: str,
    language: str,
    w: int,
    p: int,
    elapsed_time: float,
    output_csv: str = "Results/exploringParsing.csv"
) -> None:
    """
    Analyze the parsing output files produced by newscan.x and append a single
    summary row to a CSV file.

    Input:
    - base_filename (str): base path (inside "Dataset") without extension.
                             e.g. if files are "Dataset/foo.parse" and "Dataset/foo.dicz.len",
                             pass base_filename="foo" or the full path "Dataset/foo".
    - language (str): language label to store in results (e.g. "Python").
    - w (int): window size parameter used during parsing (recorded for reference).
    - p (int): modulo p parameter used during parsing (recorded for reference).
    - elapsed_time (float): elapsed processing time in seconds.
    - output_csv (str): path to the CSV file where to store/append results.
                           Defaults to "Results/exploringParsing.csv".

    Return:
    - None: the function writes/appends one row to `output_csv` with computed metrics.
              It raises FileNotFoundError if any required input file is missing.
    """
    # Build paths for the expected files. base_filename may already include "Dataset/".
    base_path = os.path.join("Dataset", base_filename)
    parse_file = base_path + ".parse"
    dicz_len_file = base_path + ".dicz.len"
    original_text_file = base_path  # original input text

    # === Check if required files exist
    for f in [parse_file, dicz_len_file, original_text_file]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    # === Load the .parse file (sequence of token ranks)
    with open(parse_file, "rb") as f:
        parse = np.fromfile(f, dtype=np.uint32)

    # Basic counts and frequency statistics
    n_tokens = len(parse)
    token_counts = Counter(parse)
    n_unique_tokens = len(token_counts)
    pct_unique_tokens = n_unique_tokens / n_tokens
    avg_token_freq = n_tokens / n_unique_tokens if n_unique_tokens > 0 else 0

    # === Compute Shannon entropy (base 2) over the empirical token distribution
    probs = np.array(list(token_counts.values())) / n_tokens
    entropy = -np.sum(probs * np.log2(probs))

    # === Load the word lengths from .dicz.len
    with open(dicz_len_file, "rb") as f:
        word_lengths = np.fromfile(f, dtype=np.uint32)
    avg_word_len = np.mean(word_lengths)

    # === Compute compressed size (parse length + total word length)
    compressed_size = len(parse) + word_lengths.sum()

    # === Load the original text to get its size
    with open(original_text_file, "r", encoding="utf-8") as f:
        original_text_size = len(f.read())

    compression_ratio = compressed_size / original_text_size if original_text_size > 0 else None

    # === Collect all results in a row (dictionary)
    row = {
        "language": language,
        "windows_size": w,
        "modulo_p": p,
        "elapsed_time": elapsed_time,
        "n_tokens": n_tokens,
        "n_unique_tokens": n_unique_tokens,
        "pct_unique_tokens": pct_unique_tokens,
        "avg_token_freq": avg_token_freq,
        "entropy": entropy,
        "avg_word_len": avg_word_len,
        "compressed_size": compressed_size,
        "original_text_size": original_text_size,
        "compression_ratio": compression_ratio
    }

    df = pd.DataFrame([row])

    # === Save or append to the CSV file
    if not os.path.exists(output_csv):
        df.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, mode="a", index=False, header=False)



#============== main ================
if __name__ == "__main__":
    
    # Variables to change at every running
    CREATE_TEXT_FILE = False
    language = "Python"
    column = "text"
    
    # Parameters passed at NewScan:
    windows_size = 8
    modulo_p= 10
    
    df = pd.read_csv(f"Dataset/{language}100MB.csv")  
    
    
    # Optionally prepare a single concatenated text file used by NewScan.
    if CREATE_TEXT_FILE:
        prepare_text_for_newscan(
            df=df,
            column=column,
            output_path=f"Dataset/{language}_text.txt",
        )
    
    # Run NewScan on the prepared text file
    elapsed_time = run_newscan(
        input_filename=f"{language}_text.txt",
        w=windows_size,
        p=modulo_p
    )

    # Analyze the files produced by NewScan and append a summary row to a CSV file.
    analyze_parsing_files(
        base_filename=f"{language}_text.txt",
        language=language,
        w=windows_size,
        p=modulo_p,
        elapsed_time=elapsed_time
    )
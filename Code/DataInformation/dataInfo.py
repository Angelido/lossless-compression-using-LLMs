import pandas as pd 
import numpy as np 

language = {
    "python": "Python",
    "c": "C",
    "java": "Java",
    "javaScript": "JavaScript",
    "cpp": "Cpp",
    "cSharp": "cSharp"
}

for key, lang in language.items():
    filename=f"Dataset/{lang}100MB.csv"
    try:
        df = pd.read_csv(filename)
        
        num_rows = len(df)
        total_bytes = df["length_bytes"].sum()
        total_mb = total_bytes / (1024 * 1024)
        avg_bytes_per_file = total_bytes / num_rows if num_rows != 0 else 0
        
        print("==================================== \n")
        print("Language: ", lang, "\n")
        print(f"Numero di righe: {num_rows}")
        print(f"Somma totale in bytes: {total_bytes}")
        print(f"Somma totale in MB: {total_mb:.2f} MB")
        print(f"Media bytes per file: {avg_bytes_per_file:.2f}")
        
        
    except FileNotFoundError:
        print(f"File {filename} not found.")
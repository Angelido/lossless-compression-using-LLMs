import pandas as pd
import os

# Load datasets for each programming language
python = pd.read_csv("Dataset/Python.csv")
c = pd.read_csv("Dataset/C.csv")
cpp = pd.read_csv("Dataset/Cpp.csv")
java = pd.read_csv("Dataset/Java.csv")
java_script = pd.read_csv("Dataset/JavaScript.csv")
c_sharp = pd.read_csv("Dataset/C-Sharp.csv")

# Combine all datasets into a single DataFrame
code_dataset = pd.concat([python, c, cpp, java, java_script, c_sharp], ignore_index=True)

# Display the number of code samples available for each language
print("Number of code samples per programming language:")
print(code_dataset["language"].value_counts(), "\n")

# Compute the total number of bytes across all code samples
total_length_bytes = code_dataset['length_bytes'].sum()

# Compute the total number of bytes per language
length_bytes_per_language = code_dataset.groupby('language')['length_bytes'].sum()

print(f"Total size of all code samples: {total_length_bytes:,} bytes\n")

print("Total size per language (in bytes):")
print(length_bytes_per_language, "\n")

# Calculate the average number of bytes per code sample for each language
average_length_bytes_per_language = length_bytes_per_language / code_dataset["language"].value_counts()

print("Average size per code sample by language (in bytes):")
print(average_length_bytes_per_language, "\n")

# Convert total length in bytes to megabytes for each language
length_bytes_per_language_in_mb = length_bytes_per_language / (1024 * 1024)

print("Total size per language (in megabytes):")
print(length_bytes_per_language_in_mb.round(2), "\n")

os.makedirs("Dataset", exist_ok=True)
code_dataset.to_csv("Dataset/CodeDataset.csv")

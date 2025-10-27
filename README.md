# Lossless Compression using LLMs

**Status:** Work in Progress 🚧  
This repository is under active development. The current codebase is incomplete, and new modules are being progressively added.  

---

## Overview

This repository explores **lossless data compression using Large Language Models (LLMs)**.  
The primary objective is to investigate novel compression techniques that aim to improve the **compression ratio** while maintaining a **throughput comparable to classical compressors**.

The research focuses exclusively on **source code compression**, as the project is conducted within the context of **[Software Heritage](https://www.softwareheritage.org/)**.  
The source code datasets used in this study were collected from Software Heritage archives via the `boto` libraries, covering six widely-used programming languages:

- **C**
- **C#**
- **C++**
- **Python**
- **Java**
- **JavaScript**

---

## Datasets

Two phases of dataset preparation were performed:

### Phase 1 – Exploratory Dataset
A smaller exploratory dataset was first curated to allow broad experimentation with a large number of models.

| Language   | # Files | Total Size (B) | Total Size (MB) | Avg Size (B) |
|------------|---------|----------------|-----------------|--------------|
| C#         | 3,407   | 10,485,193     | 10.00           | 3,077.54     |
| Python     | 3,893   | 10,484,445     | 10.00           | 2,693.15     |
| Java       | 3,436   | 10,484,007     | 10.00           | 3,051.22     |
| C          | 3,941   | 10,479,511     | 9.99            | 2,659.10     |
| C++        | 4,059   | 10,464,180     | 9.98            | 2,578.02     |
| JavaScript | 4,010   | 10,453,595     | 9.97            | 2,606.88     |
| **Total**  | 23,746  | 62,850,931     | 59.94           | --           |

---

### Phase 2 – Per-Language Datasets
After selecting the most promising models, six larger datasets were built, one for each programming language.

| Language   | # Files | Total Size (B) | Total Size (MB) | Avg Size (B) |
|------------|---------|----------------|-----------------|--------------|
| Java       | 34,401  | 104,856,598    | 100.00          | 3,048.07     |
| C#         | 33,853  | 104,856,129    | 100.00          | 3,097.40     |
| Python     | 38,312  | 104,853,756    | 100.00          | 2,736.84     |
| JavaScript | 38,988  | 104,848,652    | 99.99           | 2,689.25     |
| C++        | 35,565  | 104,848,494    | 99.99           | 2,948.08     |
| C          | 38,716  | 104,844,579    | 99.99           | 2,708.04     |
| **Total**  | 219,835 | 629,108,208    | 599.97          | --           |

---

## Models

A total of **30 LLMs** were explored, including:

- **Code-specialized models** (e.g., *StarCoder2, CodeGemma, Granite Code, CodeT5*)  
- **Quantized models** (4-bit and 8-bit versions for efficiency)  
- **General-purpose models** (*Llama 3, Mistral, GPT-2*)  

All models were retrieved from [HuggingFace](https://huggingface.co/).  

A detailed table of models, including parameter counts, memory requirements, quantization status, and code specialization, is provided in the paper/manuscript accompanying this work.  

---

## Baseline Compressors

For benchmarking purposes, standard lossless compression algorithms were used as baselines:

- [**bzip2**](http://www.bzip.org/)  
- [**bzip3**](https://github.com/kspalaiologos/bzip3)  
- [**zstd**](https://facebook.github.io/zstd/)  

These compressors provide reference points in terms of both **compression ratio** and **throughput**.   

---

## Citation

If you use this repository or the associated datasets in your research, please cite the corresponding work (citation details will be provided soon).  




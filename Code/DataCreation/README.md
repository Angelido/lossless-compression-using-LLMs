# DataCreation

This directory contains the code used to **download and prepare the datasets** employed in the experimental phase of this work.  

Since the experiments were divided into two stages, the dataset creation process is organized accordingly:

- **Phase 1 (Preliminary Exploration):**  
  All available models were tested on a smaller, exploratory dataset.  
- **Phase 2 (Detailed Analysis):**  
  Only the most promising models were evaluated on larger, per-language datasets.

All source code files were retrieved from **[Software Heritage](https://www.softwareheritage.org/)**, specifically from the **StackEdu dataset**.

---

## File Overview

- **`firstDatasetDownloader.py`**  
  Script used to download the source code files for the *Phase 1* exploratory dataset directly from Software Heritage.  

- **`createFirstDataset.py`**  
  Script that merges the different downloaded datasets into a single multilingual dataset for *Phase 1* experiments.  

- **`datasetDownloader.py`**  
  Script used to download and build the larger, per-language datasets for *Phase 2* experiments.  

- **`datasetCppDownloader.py`**  
  Script specifically designed to download and construct the C++ dataset for *Phase 2*.  
  This specialized downloader was necessary to filter out files that were either too large or too small, which would have otherwise biased the dataset compared to those of the other programming languages.  

---

## Notes

The scripts in this folder are tailored for reproducibility of the datasets used in the experiments.  
Each dataset produced follows the specifications outlined in the paper associated with this project.  

# Code

This repository contains all the code developed during the preparation of the thesis.
Each subfolder groups code related to a specific topic, while the three shared modules (`utility.py`, `dataLoader.py`, and `computeRank.py`) provide common functionality used across the different experiments.

---

## Table of Contents

* [Folder structure](#folder-structure)
* [Core modules](#core-modules)
* [Subfolders usage](#subfolders-usage)
* [Notes](#notes)

---

## Folder structure

The repository is organized into topic-specific folders:

* **Compression** → Code for compression experiments following the pipeline
  `text → tokenization + dataloader → computeRank → classical compression`.

* **DataCreation** → Scripts for generating datasets used in the experiments.

* **DataInformation** → Tools for extracting insights from both datasets and experimental results.
  Includes general-purpose scripts for producing plots and visualizations.

* **DecodeRank** → Demonstrates how to decompress data without information loss.
  Pipeline:
  `text → tokenization + dataloader → computeRank → decodeRank → original tokens`.

* **Entropy** → Experiments measuring the entropy of each model on various datasets,
  in order to compare theoretical entropy bounds with practical compression results.

* **Parsing** → Explores parsing as a preprocessing step before compression.
  Pipeline:
  `text → parsing → tokenization + dataloader (dictionary) → computeRank (dataloader) → compression (all)`.

* **RankAnalysis** → Code for analyzing the rank predictions of different models,
  used to assess how well models predict tokens.

* **TestModels** → Scripts for the initial evaluation of candidate models.
  Pipeline:
  `text → tokenization + dataloader → computeRank`.
  The resulting ranks are later analyzed in **RankAnalysis**.

---

## Core modules

In addition to the folders above, three shared Python files provide fundamental functionality:

* **`computeRank.py`** → Functions for computing token ranks with different models,
  as well as decoding (reconstructing tokens from ranks).

* **`dataLoader.py`** → Functions for tokenizing text and building dataloaders for model input.
  Handles preprocessing of raw data before rank computation.

* **`utility.py`** → General-purpose utilities, such as:

  * saving results to CSV or other formats,
  * checking specific data properties,
  * sorting operations.

---

## Subfolders usage

To ensure access to the shared modules (`utility.py`, `dataLoader.py`, `computeRank.py`),
each script inside the subfolders should include the following snippet:

```python
import sys
import os

# Allow imports from the parent folder (utility, dataLoader, computeRank)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

---

## Notes

* This README is descriptive only; it does not contain runnable code.
* See the scripts inside each folder for concrete experiments and pipelines.



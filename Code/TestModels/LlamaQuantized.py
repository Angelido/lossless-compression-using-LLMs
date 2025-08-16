import pandas as pd
import time
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# Read also files from the parent folder (utility, dataLoader, computeRank)   
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from computeRank import compute_token_ranks_fast
from dataLoader import create_chunk_dataloader, preprocess_dataset_fast
from utility import ( 
    save_rank_list_to_file,
    count_nonpad_tokens_per_row, 
    sort_chunks_by_length, 
)

# Login to Hugging Face Hub
from huggingface_hub import login
login(token="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")

# Model name
# model_name = "unsloth/Llama-3.2-1B-bnb-4bit" # Quantized
model_name = "unsloth/Llama-3.2-3B-bnb-4bit" # Quantized

# Model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",            # pass the model to the GPU
    trust_remote_code=True
)

batch_size = 32
max_length = 512
PAD_TOKEN_ID = tokenizer.pad_token_id

# Set the device to cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
print("device=", device)

df = pd.read_csv("Dataset/CodeDataset.csv")
input_texts = df["text"].tolist()

# NEW VERSION
input_id_list, mapping = preprocess_dataset_fast(
    input_texts,
    tokenizer,
    max_length=max_length,
    stride=0           
)
input_id_list, mapping = sort_chunks_by_length(
    input_id_list,
    mapping,
    pad_token_id=PAD_TOKEN_ID,
    descending=True
)
dataloader = create_chunk_dataloader(
    input_id_list,
    batch_size=batch_size
)

print("After dataloader")

# # For choose the right max_length
# row_lengths = count_nonpad_tokens_per_row(input_id_list, mapping, PAD_TOKEN_ID)
# # Show the distribution of token lengths
# lengths = list(row_lengths.values())
# for p in [50, 75, 90, 95, 99]:
#     print(f"{p}° percentile token count: {np.percentile(lengths, p):.0f}")

# Timer start
start_time = time.perf_counter()

# Compute the rank list using the DataLoader
rank_list = compute_token_ranks_fast(
     dataloader,
     model,
     pad_token_id=PAD_TOKEN_ID,
     device=device
)

# Timer end
end_time = time.perf_counter()
execution_time = end_time - start_time

print("Finished computing rank list")

# Reconstruct the rank list using the mapping
reconstructed_rank_list = [
    [rank for chunk_idx in mapping[row_idx] for rank in rank_list[chunk_idx]]
    for row_idx in range(len(input_texts))
]

# Save the rank list to a file
save_rank_list_to_file(
    rank_list=reconstructed_rank_list,
    file_path="TextInformation/rank_list.txt",
    execution_time=execution_time,
    model_name=model_name  
)
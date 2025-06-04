import pandas as pd
import time
import torch
from unixcoder import UniXcoder
from dataLoader import create_chunk_dataloader, preprocess_dataset_fast_unixcoder
from utility import (
    save_rank_list_to_file,
    count_nonpad_tokens_per_row,
    sort_chunks_by_length,
    compute_token_ranks_fast_unixcoder  
)

# Model name
model_name = "microsoft/unixcoder-base"

# Tokenizer and model
ux = UniXcoder(model_name)
tokenizer = ux.tokenizer
model = ux   

batch_size = 64
max_length = 512
PAD_TOKEN_ID = tokenizer.pad_token_id  

# Set the device to cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("device=", device)

# Load the dataset
df = pd.read_csv("Dataset/CodeDataset.csv")
input_texts = df["text"].tolist()

# Preprocessing and chunking
input_id_list, mapping = preprocess_dataset_fast_unixcoder(
    input_texts,
    ux,
    max_length=max_length,
    stride=0           
)
# Sort the chunks by length
input_id_list, mapping = sort_chunks_by_length(
    input_id_list,
    mapping,
    pad_token_id=PAD_TOKEN_ID,
    descending=True
)
# Create a DataLoader for the input sequences
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
rank_list = compute_token_ranks_fast_unixcoder(
     dataloader,
     ux,
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
    file_path="TextInformation/UnixCoder2.txt",
    execution_time=execution_time,
    model_name=model_name  
)

# Regenerate texts based on the rank list
# generated_texts = regenerate_texts(rank_list, model, tokenizer, device)
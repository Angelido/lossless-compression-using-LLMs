import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataLoader import create_chunk_dataloader, preprocess_dataset_fast
from utility import ( 
    save_rank_list_to_file,
    regenerate_texts,
    count_nonpad_tokens_per_row, 
    sort_chunks_by_length, 
    compute_token_ranks_fast
)

# Model name
# model_name = "bigcode/starcoder2-1b"
model_name = "bigcode/starcoder2-3b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                         # Load the model in 4-bit   
    bnb_4bit_use_double_quant=True,            # Improve performance
    bnb_4bit_quant_type="nf4",                 # Can be "fp4" or "nf4"
    bnb_4bit_compute_dtype=torch.bfloat16      # Use bfloat16 for computation
)

# Model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

print(model.hf_device_map)
print(model.model.layers[0].self_attn.q_proj.weight.dtype)

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
    file_path="TextInformation/StarCoder2_4_rank_list.txt",
    execution_time=execution_time,
    model_name=model_name  
)

# Regenerate texts based on the rank list
# generated_texts = regenerate_texts(rank_list, model, tokenizer, device)

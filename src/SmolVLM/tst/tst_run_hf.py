from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",                                                
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager").to(DEVICE)

# get approriate data for SmolVLM
# https://huggingface.co/datasets/HuggingFaceTB/SmolVLM-Instruct-Data


# load data
ds = load_dataset('merve/vqav2-small', trust_remote_code=True)
split_ds = ds["validation"].train_test_split(test_size=0.5)
train_ds = split_ds["train"]
test_ds = split_ds["test"]

print(f"train_ds: {train_ds}")
print(f"test_ds: {test_ds}")

print(f"train_ds sample : {train_ds[0]}")
print(f"The image is: {train_ds[0]['image']}")
print(f"The question is: {train_ds[0]['question']}")
print(f"The answer is: {train_ds[0]['answer']}")




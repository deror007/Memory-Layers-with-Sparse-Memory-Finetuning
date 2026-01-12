from datasets import load_dataset
import os
save_dir = "./data_dir/tiny_stories"
os.makedirs(save_dir, exist_ok=True)

# Retrieve tiny stories dataset
ds = load_dataset("roneneldan/TinyStories", cache_dir=save_dir)

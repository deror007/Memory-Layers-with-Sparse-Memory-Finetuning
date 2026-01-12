from transformers import GPT2TokenizerFast

save_dir ="./tokenizer_dir"
s
tokenizer = GPT2TokenizerFast.from_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# Example encoding
tokens = tokenizer("Once upon a time, there was a Russell.", return_tensors="pt")
input_ids = tokens["input_ids"]
print(input_ids)

# get size of vocab in tokenizer
vocab_size = tokenizer.vocab_size
print(vocab_size)
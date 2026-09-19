import torch
import json
from transformers import AutoTokenizer

model_dir = "../gemma-4-12B-it-qat-q4_0-unquantized"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = """<|turn>system
<|think|>
You are a helpful, respectful and honest AI assistant operating in an AMD Ryzen computing runtime.
<turn|>
<|turn>user
How are you doing today?<turn|>
<|turn>model
<|channel>thought
The user is asking "How are you doing today?". This is a standard social greeting. I should respond politely and helpfully, acknowledging my nature as an AI.

Plan:
1. Acknowledge and respond to the greeting.
2. State that I'm doing well and ready to assist.
3. Ask how I can help the user.<channel|>I'm doing well, thank you for asking! As an AI, I don'"""

print(f"Tokenizing prompt...")
inputs = tokenizer(prompt, return_tensors="pt")
tokens = inputs.input_ids[0].tolist()
print(f"Tokens: {len(tokens)}")
print(f"Last 5 tokens: {[tokenizer.decode([t]) for t in tokens[-5:]]}")
print(f"Last 5 token IDs: {tokens[-5:]}")

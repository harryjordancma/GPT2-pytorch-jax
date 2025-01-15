# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3.11
#     language: python
#     name: py311
# ---

# %%
from transformers import GPT2LMHeadModel

# %% jupyter={"outputs_hidden": true}
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")  # 124M
sd_hf = model_hf.state_dict()

for k, v in sd_hf.items():
    print(k, v.shape)

# %%
import matplotlib.pyplot as plt

plt.imshow(sd_hf["transformer.wpe.weight"], cmap="gray")

# %%
plt.plot(sd_hf["transformer.wpe.weight"][:, 50])
plt.plot(sd_hf["transformer.wpe.weight"][:, 100])
plt.plot(sd_hf["transformer.wpe.weight"][:, 200])
plt.xlabel("embedding dimension")
plt.ylabel("weights")

# %%
plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:300, :300], cmap="gray")

# %%
from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="gpt2")

set_seed(42)
generator("Hello, I'm a large language model", max_length=30, num_return_sequences=5)

# %%
import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

response = requests.get(url)

if response.status_code == 200:

    with open("input.txt", "w", encoding="utf-8") as file:
        file.write(response.text)
    print("Shakespeare data saved!")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")

# %%
with open("input.txt", "r") as file:
    text = file.read()
data = text[:1000]
print(data[:100])

# %%
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(data)
print(tokens[:24])

# %% [markdown]
# We want to get the shakespeare data into the format of (B, T). i.e. batches of sequences of tokens.

# %%
import torch
buf = torch.tensor(tokens[:24 + 1]) # +1 for label tensor
x = buf[:-1].view(4, 6) # Create inputs to transformer
y = buf[1:].view(4, 6) # Create labels tensor
print(x)
print(y)

# %%
print(sd_hf["lm_head.weight"].shape)
print(sd_hf["transformer.wte.weight"].shape)

# %%
(sd_hf["lm_head.weight"] == sd_hf["transformer.wte.weight"]).all()

# %%
print(sd_hf["lm_head.weight"].data_ptr())
print(sd_hf["transformer.wte.weight"].data_ptr())

# %% jupyter={"source_hidden": true}
# standard deviation grows inside the residual stream
# scale dow c_proj
x = torch.zeros(768)
n = 100
for i in range(n):
    x += n**-0.5 * torch.randn(768)
print(x.std())

# %%

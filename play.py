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

# %%
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

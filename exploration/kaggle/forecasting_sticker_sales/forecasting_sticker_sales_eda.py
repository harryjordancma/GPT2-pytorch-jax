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

# %% [markdown]
# # Forecasting sticker sales - EDA

# %%
# !pip install seaborn

# %%
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# %%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
display(train.info())

# %%
train['date'] = pd.to_datetime(train['date'])

# %%
train.head()

# %%
# Check for missing values
display(train.isnull().sum())

# %% [markdown]
# Less than 4% of target values are null, lets drop them for now then we can use the later

# %%
train = train.dropna()

# %% [markdown]
# ## Visualise the columns

# %%
print(f"Unique countries: {train['country'].unique()}")
print(f"Unique stores: {train['store'].unique()}")
print(f"Unique products: {train['product'].unique()}")

# %%
# Distribution of 'num_sold'
plt.figure(figsize=(8, 6))
sns.histplot(train['num_sold'], bins=10, kde=True)
plt.title('Distribution of Number of Products Sold')
plt.xlabel('Number of Products Sold')
plt.ylabel('Frequency')
plt.show()

# %%
# Sales by Product
plt.figure(figsize=(8, 6))
sns.barplot(x='product', y='num_sold', data=train)
plt.title('Number of Products Sold by Product')
plt.xlabel('Product')
plt.ylabel('Number of Products Sold')
plt.xticks(rotation=45)
plt.show()

# %%
# Sales Over Time (if more dates are available)
plt.figure(figsize=(8, 6))
sns.lineplot(x='date', y='num_sold', data=train)
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.show()


# %%
# Sales by Country (if multiple countries are available)
plt.figure(figsize=(8, 6))
sns.barplot(x='country', y='num_sold', data=train)
plt.title('Number of Products Sold by Country')
plt.xlabel('Country')
plt.ylabel('Number of Products Sold')
plt.show()

# %%

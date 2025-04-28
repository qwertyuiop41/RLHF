from datasets import load_dataset

# 加载数据集
dataset = load_dataset("OpenRLHF/preference_dataset_mixture2_and_safe_pku", split="train")

# 保存为 JSON Lines 格式
# dataset.to_json("datasets/preference_dataset_mixture2_and_safe_pku.jsonl")

dataset.to_parquet('datasets/preference_dataset_mixture2_and_safe_pku.parquet')
""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm

import argparse
import json



# def make_prefix(dp, template_type):
#     if template_type == 'base':
#         prefix = ""

#     elif template_type == 'qwen-instruct':
#         prefix = f"<|im_start|>user\n{dp["chosen"][0]["content"]}\n<|im_end|>\n"
#     return prefix


def make_preference(dp, template_type):
    # print(dp)
    if template_type == 'base':
        chosen = ""
        rejected=""


    elif template_type == 'qwen-instruct':

        chosen = ""
        rejected=""

        chosen_user=dp["chosen"][0]["content"]
        chosen_assistant=dp["chosen"][1]["content"]
        chosen+=f"<|im_start|>user\n{chosen_user}\n<|im_end|>\n"
        chosen+=f"<|im_start|>assistant\n{chosen_assistant}\n<|im_end|>\n"

        rejected_user=dp["rejected"][0]["content"]
        rejected_assistant=dp["rejected"][1]["content"]
        rejected+=f"<|im_start|>user\n{rejected_user}\n<|im_end|>\n"
        rejected+=f"<|im_start|>assistant\n{rejected_assistant}\n<|im_end|>\n"

    return chosen,rejected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='dataset')
    parser.add_argument('--dataset_name', default='OpenRLHF/preference_dataset_mixture2_and_safe_pku')
    parser.add_argument('--template_type', type=str, default='qwen-instruct')


    args = parser.parse_args()
    
    dataset_name = args.dataset_name


    train_dataset = load_dataset(dataset_name,split="train")
    # test_dataset = load_dataset(dataset_name,split="test")

    def make_map_fn():
        def process_fn(example, idx):
            # if example["chosen_score"] is not None and example["rejected_score"] is not None:
                # question = make_prefix(example, template_type=args.template_type)
            chosen,rejected=make_preference(example, template_type=args.template_type)
            data = {
                "dataset": dataset_name,
                "ability": "human preference",
                "chosen":chosen,
                "rejected":rejected,
                "chosen_score":example["chosen_score"],
                "rejected_score":example["rejected_score"],
                'index': idx,
            }
            return data
            # else:
            #     return None  # 返回 None 以过滤掉该样本
        return process_fn
    # 使用 filter 函数来过滤掉 None 值
    train_dataset = train_dataset.filter(lambda example:  example["chosen_score"] is not None and example["rejected_score"] is not None)
    train_dataset = train_dataset.map(function=make_map_fn(), with_indices=True)
    

    # test_dataset = test_dataset.map(function=make_map_fn(), with_indices=True)

    local_dir = os.path.join(args.local_dir,dataset_name.split('/')[-1])

    # Create local directory if not exists
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    # test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    train_dataset.to_json(os.path.join(local_dir, 'train.jsonl'))
    # test_dataset.to_json(os.path.join(local_dir, 'test.jsonl'))

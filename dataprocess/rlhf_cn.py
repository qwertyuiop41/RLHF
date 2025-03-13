""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm

import argparse
import json



def make_prefix(dp, template_type):
    if template_type == 'base':
        prefix = ""

    elif template_type == 'qwen-instruct':
        prefix = ""
        for context in dp["context"]:
            role="user" if context["role"]=="human" else "assistant"
            text=context["text"]
            prefix+=f"<|im_start|>{role}\n{text}\n<|im_end|>\n"

    return prefix


def make_preference(dp, template_type):
    # print(dp)
    if template_type == 'base':
        chosen = ""
        rejected=""


    elif template_type == 'qwen-instruct':
        chosen = ""
        rejected=""
        chosen_role="user" if dp["chosen"]["role"]=="human" else "assistant"
        chosen_text=dp["chosen"]["text"]
        chosen+=f"<|im_start|>{chosen_role}\n{chosen_text}\n<|im_end|>\n"
        rejected_role="user" if dp["rejected"]["role"]=="human" else "assistant"
        rejected_text=dp["rejected"]["text"]
        rejected+=f"<|im_start|>{rejected_role}\n{rejected_text}\n<|im_end|>\n"

    return chosen,rejected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/wsy/NLP/RL/RLHF/dataset')
    parser.add_argument('--dataset_name', default='dikw/hh_rlhf_cn')
    parser.add_argument('--template_type', type=str, default='qwen-instruct')


    args = parser.parse_args()
    
    dataset_name = args.dataset_name


    train_dataset = load_dataset(dataset_name,split="train")
    test_dataset = load_dataset(dataset_name,split="test")

    def make_map_fn():
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            chosen,rejected=make_preference(example, template_type=args.template_type)
            data = {
                "dataset": dataset_name,
                "prompt": question,
                "ability": "human preference",
                "chosen":chosen,
                "rejected":rejected,
                'index': idx,
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn(), with_indices=True)

    test_dataset = test_dataset.map(function=make_map_fn(), with_indices=True)

    local_dir = os.path.join(args.local_dir,dataset_name.split('/')[-1])

    # Create local directory if not exists
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    train_dataset.to_json(os.path.join(local_dir, 'train.jsonl'))
    test_dataset.to_json(os.path.join(local_dir, 'test.jsonl'))

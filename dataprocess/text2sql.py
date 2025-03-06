""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm

import argparse
import json

def make_prefix(dp, template_type):
    question = dp['question']
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the final SQL query. The reasoning process and query are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> SQL query here </answer>. Now the user asks you to solve a text-to-SQL problem. After thinking, when you finally reach a conclusion, clearly state the SQL query within <answer> </answer> tags. For example, <answer> SELECT count(*) FROM head WHERE age > 56 </answer>.\n\nUser:{question}\nAssistant: <think>"""

    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the SQL query. The reasoning process and query are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> SQL query here </answer>. Now the user asks you to solve a text-to-SQL problem. After thinking, when you finally reach a conclusion, clearly state the SQL query within <answer> </answer> tags.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>think>"""

    return prefix




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/wsy/NLP/RL/RLHF/datatset')
    parser.add_argument('--dataset_name', default='xlangai/spider')
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--test_size', type=int, default=None)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')

    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    train_dataset = load_dataset(dataset_name, split='train')
    test_dataset=load_dataset(dataset_name, split='validation')

    if TRAIN_SIZE is not None:
        assert len(train_dataset) >= TRAIN_SIZE
        train_dataset = train_dataset.select(range(TRAIN_SIZE))
    if TEST_SIZE is not None:
        assert len(test_dataset) >= TEST_SIZE
        test_dataset = test_dataset.select(range(TEST_SIZE))


    

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            ground_truth = example['query']
            data = {
                "dataset": dataset_name,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "sql generation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                'split': split,
                'index': idx,
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    test_dataset = test_dataset.map(function=make_map_fn('validation'), with_indices=True)

    local_dir = os.path.join(args.local_dir,dataset_name.split('/')[-1])

    # Create local directory if not exists
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    train_dataset.to_json(os.path.join(local_dir, 'train.jsonl'))
    test_dataset.to_json(os.path.join(local_dir, 'test.jsonl'))

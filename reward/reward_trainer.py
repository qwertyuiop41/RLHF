import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm
from transformers import Trainer
import wandb


from openrlhf.models import LogExpLoss, PairWiseLoss

class RewardModelTrainer(Trainer):
    def __init__(
        self,
        model,
        optim: Optimizer,
        train_dataset,
        eval_dataset,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
    ):
        
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer

        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            print("LogSigmoid Loss")
        else:
            self.loss_fn = LogExpLoss()
            print("LogExp Loss")

        



    def fit(self):
        
        step = 0
        start_epoch = 0

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch")
        acc_sum = 0
        loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            for data in self.train_dataset:
                print(data)
                exit(0)



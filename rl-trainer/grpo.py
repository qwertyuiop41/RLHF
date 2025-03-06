from typing import Dict, List, Sequence
from policy_model import PolicyModel
from reference_model import RefModel
from reward_model import RewardModel


class GRPO():
    def __init__(self,
                 model,
                 config,
                 tokenizer,
                 reward_fn=None):
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.policyModel = PolicyModel(model)
        self.referenceModel = RefModel(model)
        self.rewardModel = RewardModel(model)
        

    def create_dataloader(self, model_path=None):
        pass
        
    def compute_advantage(data: Sequence[Dict[str, List[int]]], adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
        pass
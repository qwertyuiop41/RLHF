import copy
import numpy as np
import torch
from torch.distributions import MultivariateNormal

from torch import nn

from typing import Optional, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    TrainingArguments
)
from peft import PeftConfig, get_peft_model
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl import create_reference_model
from datasets import load_dataset



class PPO():
    def __init__(self):
        self.policy_model_path='/home/wsy/NLP/RL/Qwen2.5-0.5B-Instruct'
        self.reward_model_path='/home/wsy/NLP/RL/Qwen2.5-0.5B-Instruct'
        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode
        self.n_updates_per_iteration = 5
        self.clip = 0.2 # As recommended by the paper
        self.lr=0.005
        self.save_freq=10
        self.gamma = 0.95 
        self.train_dataset_path='/home/wsy/NLP/RL/RLHF/dataset/spider/train.parquet'
        self.test_dataset_path='/home/wsy/NLP/RL/RLHF/dataset/spider/test.parquet'
        self.batch_size=4

        self.train_dataset=load_dataset("parquet", data_files=self.train_dataset_path,split='train')
       
        self.test_dataset=load_dataset("parquet", data_files=self.test_dataset_path,split='train')


        self.policy_tokenizer=AutoTokenizer.from_pretrained(self.policy_model_path)
        self.policy_tokenizer.padding_side="left"

        self.policy_model=AutoModelForCausalLM.from_pretrained(self.policy_model_path)
        self.value_model=AutoModelForCausalLM.from_pretrained(self.policy_model_path)
        self.reward_model=AutoModelForCausalLM.from_pretrained(self.reward_model_path)
        self.ref_model=copy.deepcopy(self.policy_model)
        self.policy_optimizer=torch.optim.Adam(self.policy_model.parameters(),lr=self.lr)
        self.value_optimizer=torch.optim.Adam(self.value_model.parameters(),lr=self.lr)


        

        
    def learn(self):
        current_step=0
        for i in range(0, len(self.train_dataset), self.batch_size):
            batch = self.train_dataset[i : min(i + self.batch_size,len(self.train_dataset))]  # 获取一个 batch
            questions = [example[0]['content'] for example in batch["prompt"]]
            # print(questions)

            responses = []
            batch=[]
            for question in questions:
                input=self.policy_tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
                print(input)
                batch.append(input)
                with torch.no_grad():
                    output=self.policy_model.generate(**input, max_new_tokens=50)
                print(output)

                response=self.policy_tokenizer.batch_decode(output, skip_special_tokens=True)
                responses.append(response)
                print(response)
                

            # compute global_valid tokens
            batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()


            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with _timer('values', timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            exit(0)







        while current_step<total_steps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens=self.rollout()
            V , _ = self.evaluate(batch_obs,batch_acts)

            # TODO: 计算advance，根据算法的实际公式进行修改，这里是一个简化后的公式
            A_k=batch_rtgs-V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for i in range(self.n_updates_per_iteration):
                V, current_log_probs = self.evaluate(batch_obs,batch_acts)
                # Compute the ratio of new probs to old ones
                ratios = torch.exp(current_log_probs - batch_log_probs)
                # Compute surrogate losses.
                surr1 = ratios * A_k
                # 类似kl惩罚，防止变化过大
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                value_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                self.policy_optimizer.step()

                # Calculate gradients and perform backward propagation for critic network    
                self.value_optimizer.zero_grad()    
                value_loss.backward()    
                self.value_optimizer.step()

            # Calculate how many timesteps we collected this batch   
            current_step += np.sum(batch_lens)


            
            if current_step%self.save_freq==0:
                print(f"current step:{current_step}")


    
    def rollout(self):  
        """
        TODO 根据task进行修改
        """
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        t=0

        while t < self.timesteps_per_batch:
            ep_rews = []            # rewards collected per episode
            obs = self.env.reset()
            obs=list(obs)[0]
            # obs = torch.tensor(obs, dtype=torch.float32)
            done = False
            
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_obs.append(obs)

                # TODO: 修改这一部分，改的更契合数据集
                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _  = self.env.step(action)
                done = terminated | truncated
            
                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews) 


        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self,obs):
        """
        TODO: 根据task修改这一部分
        """
        # action=self.policy_model(obs)
        # log_prob=self.policy_model.get_log_prob(action)
        # return action,log_prob


        # Query the actor network for a mean action
        mean = self.policy_model(obs)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self,batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    

    def evaluate(self, batch_obs,batch_acts):
        """
        TODO 根据task进行修改
        """
        # Query critic network for a value V for each obs in batch_obs.
        V = self.value_model(batch_obs).squeeze()
        mean = self.policy_model(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V,log_probs
    


    # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        print(f"Inputs: {inputs}")
        inputs = self._prepare_inputs(inputs)
        print(f"Inputs: {inputs}")
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None
    
    # def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
    #     device = self.accelerator.device
    #     prompts = [x["prompt"] for x in inputs]
    #     prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
    #     prompt_inputs = self.processing_class(
    #         prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
    #     )
    #     prompt_inputs = super()._prepare_inputs(prompt_inputs)
    #     prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    #     if self.max_prompt_length is not None:
    #         prompt_ids = prompt_ids[:, -self.max_prompt_length :]
    #         prompt_mask = prompt_mask[:, -self.max_prompt_length :]

    #     # Generate completions using either vLLM or regular generation
    #     if self.args.use_vllm:
    #         # First, have main process load weights if needed
    #         if self.state.global_step != self._last_loaded_step:
    #             self._move_model_to_vllm()
    #             self._last_loaded_step = self.state.global_step

    #         # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
    #         all_prompts_text = gather_object(prompts_text)
    #         if self.accelerator.is_main_process:
    #             outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
    #             completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
    #         else:
    #             completion_ids = [None] * len(all_prompts_text)
    #         # Broadcast the completions from the main process to all processes, ensuring each process receives its
    #         # corresponding slice.
    #         completion_ids = broadcast_object_list(completion_ids, from_process=0)
    #         process_slice = slice(
    #             self.accelerator.process_index * len(prompts),
    #             (self.accelerator.process_index + 1) * len(prompts),
    #         )
    #         completion_ids = completion_ids[process_slice]

    #         # Pad the completions, and concatenate them with the prompts
    #         completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
    #         completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
    #         prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    #     else:
    #         # Regular generation path
    #         with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
    #             prompt_completion_ids = unwrapped_model.generate(
    #                 prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
    #             )

    #         # Compute prompt length and extract completion ids
    #         prompt_length = prompt_ids.size(1)
    #         prompt_ids = prompt_completion_ids[:, :prompt_length]
    #         completion_ids = prompt_completion_ids[:, prompt_length:]

    #     # Mask everything after the first EOS token
    #     is_eos = completion_ids == self.processing_class.eos_token_id
    #     eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    #     eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    #     sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    #     completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    #     # Concatenate prompt_mask with completion_mask for logit computation
    #     attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

    #     logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

    #     with torch.inference_mode():
    #         if self.ref_model is not None:
    #             ref_per_token_logps = self._get_per_token_logps(
    #                 self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
    #             )
    #         else:
    #             with self.accelerator.unwrap_model(self.model).disable_adapter():
    #                 ref_per_token_logps = self._get_per_token_logps(
    #                     self.model, prompt_completion_ids, attention_mask, logits_to_keep
    #                 )

    #     # Decode the generated completions
    #     completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    #     if is_conversational(inputs[0]):
    #         completions = []
    #         for prompt, completion in zip(prompts, completions_text):
    #             bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
    #             completions.append([{"role": "assistant", "content": bootstrap + completion}])
    #     else:
    #         completions = completions_text

    #     rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        
    #     for i, (reward_func, reward_processing_class) in enumerate(
    #         zip(self.reward_funcs, self.reward_processing_classes)
    #     ):
    #         if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
    #             if is_conversational(inputs[0]):
    #                 messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
    #                 texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
    #             else:
    #                 texts = [p + c for p, c in zip(prompts, completions)]
    #             reward_inputs = reward_processing_class(
    #                 texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
    #             )
    #             reward_inputs = super()._prepare_inputs(reward_inputs)
    #             with torch.inference_mode():
    #                 rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
    #         else:
    #             # Repeat all input columns (but "prompt" and "completion") to match the number of generations
    #             keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
    #             reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
    #             output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
    #             rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

    #     # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
    #     # completions may be distributed across processes
    #     rewards_per_func = gather(rewards_per_func)

    #     # Apply weights to each reward function's output and sum
    #     rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

    #     # Compute grouped-wise rewards
    #     mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
    #     std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

    #     # Normalize the rewards to compute the advantages
    #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    #     logging.info(f"mean_grouped_rewards: {mean_grouped_rewards.mean().item()}")
    #     logging.info(f"std_grouped_rewards: {std_grouped_rewards.mean().item()}")
    #     logging.info(f"Advantages: {advantages.mean().item()}")

    #     # Slice to keep only the local part of the data
    #     process_slice = slice(
    #         self.accelerator.process_index * len(prompts),
    #         (self.accelerator.process_index + 1) * len(prompts),
    #     )
    #     advantages = advantages[process_slice]
    #     logging.info(f"Advantages: {advantages}")

    #     # Log the metrics
    #     reward_per_func = rewards_per_func.mean(0)
    #     for i, reward_func in enumerate(self.reward_funcs):
    #         if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
    #             reward_func_name = reward_func.config._name_or_path.split("/")[-1]
    #         else:
    #             reward_func_name = reward_func.__name__
    #         self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

    #     self._metrics["reward"].append(rewards.mean().item())
    #     self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

    #     if (
    #         self.log_completions
    #         and self.state.global_step % self.args.logging_steps == 0
    #         and "wandb" in self.args.report_to
    #     ):
    #         import pandas as pd

    #         # For logging
    #         table = {
    #             "step": [str(self.state.global_step)] * len(rewards),
    #             "prompt": gather_object(prompts_text),
    #             "completion": gather_object(completions_text),
    #             "reward": rewards.tolist(),
    #         }
    #         df = pd.DataFrame(table)

    #         if wandb.run is not None and self.accelerator.is_main_process:
    #             wandb.log({"completions": wandb.Table(dataframe=df)})

    #     return {
    #         "prompt_ids": prompt_ids,
    #         "prompt_mask": prompt_mask,
    #         "completion_ids": completion_ids,
    #         "completion_mask": completion_mask,
    #         "ref_per_token_logps": ref_per_token_logps,
    #         "advantages": advantages,
    #     }

    


import gym
model = PPO()
model.learn()
import copy
import numpy as np
import torch
from torch.distributions import MultivariateNormal

from network import SampleNN
from torch import nn

class PPO():
    def __init__(self,env):
        # self.obs_dim=args.obs_dim
        # self.act_dim=args.act_dim

        # TODO 下面部分可以直接删除
        # Initialize hyperparameters for training with PPO
        # self._init_hyperparameters(hyperparameters)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)


        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode
        self.n_updates_per_iteration = 5
        self.clip = 0.2 # As recommended by the paper
        self.lr=0.005
        self.save_freq=10
        self.gamma = 0.95 


        self.policy_model=SampleNN(self.obs_dim,self.act_dim)
        self.value_model=SampleNN(self.obs_dim,1)
        self.reward_model=SampleNN(self.obs_dim,1)
        self.ref_model=copy.deepcopy(self.policy_model)
        self.policy_optimizer=torch.optim.Adam(self.policy_model.parameters(),lr=self.lr)
        self.value_optimizer=torch.optim.Adam(self.value_model.parameters(),lr=self.lr)


        

        
    def learn(self,total_steps):
        current_step=0
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
    


import gym
env = gym.make('Pendulum-v1')
model = PPO(env)
model.learn(10000)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
from Util import *
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim,1)
        )
        
        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)
    
    def forward(self, X):
        means = self.model(X)
        stds = torch.tensor([5.0])#torch.clamp(self.logstds.exp(), 1e-3, 50)
        
        return torch.distributions.Normal(means, stds)

class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim,1)
        )
    
    def forward(self, X):
        return self.model(X)
    
class A2CLearner():
    def __init__(self, actor, critic, gamma=0.9, entropy_beta=0,
                 actor_lr=1e-3, critic_lr=1e-3, max_grad_norm=0.5):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    
    #####################################################################
    ### Function discription:
    ###     learn the actor and critic
    ### Input argument discription:
    ###     state, reward, update flag, discount_rewards
    ### Returns
    ###     clipped actions
    #####################################################################
    def learn(self,state,reward,update,discount_rewards=False):
        if(update):
            if discount_rewards:
                td_target = reward
            else:
                td_target = reward + self.gamma*self.critic(state)
            value = self.critic(self.prev_state)
            advantage = td_target - value

            # actor
            norm_dists = self.actor(self.prev_state)
            logs_probs = norm_dists.log_prob(self.prev_actions)
            entropy = norm_dists.entropy().mean()
            
            actor_loss = (-logs_probs*advantage.detach()).mean() - entropy*self.entropy_beta
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            # critic

            critic_loss = F.mse_loss(td_target, value)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        dists = self.actor(state)
        actions = dists.sample().detach().data.numpy()
        actions_clipped = np.clip(actions,-30,30)
        self.prev_actions=to_ten(actions)
        self.prev_state=state

        return actions_clipped.item()
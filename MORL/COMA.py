import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import itertools
import argparse
from torch.distributions import Categorical
import math


class Actor(nn.Module):
    def __init__(self, params):
        super(Actor, self).__init__()
        self.params = params
        self.fc1 = nn.Linear(params.o_dim + 2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 30)

    def forward(self, input):
        x = self.convert_type(input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logist = self.out(x)
        return logist

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.params.device)
        return x


class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.params = params
        self.fc1 = nn.Linear(params.s_dim + params.o_dim + 2 + params.n_agents*3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 30)

    def forward(self, input):
        x = self.convert_type(input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.out(x)
        return out

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.params.device)
        return x


class COMA_Agent():
    def __init__(self, params):
        self.params = params
        self.Actor = [Actor(params).to(params.device) for _ in range(params.n_agents)]
        self.Critic = [Critic(params).to(params.device) for _ in range(params.n_agents)]
        parameters = [actor.parameters() for actor in self.Actor] + \
                     [critic.parameters() for critic in self.Critic]
        self.optim = torch.optim.Adam(itertools.chain(*parameters), lr=params.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=10, gamma=0.99, last_epoch=-1)

    def choose_action(self, obs, w):
        acts = []
        w = torch.Tensor(w)
        self.dist1 = []
        self.dist2 = []
        self.dist3 = []
        self.act1 = []
        self.act2 = []
        self.act3 = []
        for i in range(self.params.n_agents):
            o = torch.Tensor(obs[i])
            input = torch.cat([o, w])
            logist = self.Actor[i](input)
            prob1, prob2, prob3 = F.softmax(logist[0:10]), F.softmax(logist[10:20]), F.softmax(logist[20:30])
            self.dist1.append(Categorical(prob1))
            self.dist2.append(Categorical(prob2))
            self.dist3.append(Categorical(prob3))
            self.act1.append(self.dist1[i].sample())
            self.act2.append(self.dist2[i].sample())
            self.act3.append(self.dist3[i].sample())
            acts.append([int(self.act1[i].numpy()), int(self.act2[i].numpy()), int(self.act3[i].numpy())])
        return acts

    def coma(self, state, obs, w, actions):
        w = torch.Tensor(w)
        state = torch.Tensor(state)
        obs = [torch.Tensor(o) for o in obs]
        actions = [torch.Tensor(actions[:i]+actions[i+1:]+actions[i:i+1]).flatten() for i in range(self.params.n_agents)]
        inputs = [torch.cat([state, obs[i], w, actions[i]]) for i in range(self.params.n_agents)]
        Q = [self.Critic[i](inputs[i]).unsqueeze(-1) for i in range(len(inputs))]
        return Q

    def learn(self, state, obs, reward, w, actions, step):
        scalarized_reward = reward[0] * w[0] + reward[1] * w[1] + 5
        # scalarized_reward = reward[0] * w[0] + reward[1] * w[1] \
        #                     - 0.1 * math.sqrt(reward[0] ** 2 + reward[1] ** 2)
        coma = self.coma(state, obs, w, actions)
        Q = [torch.cat([coma[a][actions[a][0]], coma[a][actions[a][1]+10], coma[a][actions[a][2]]+20])
                                                                            for a in range(self.params.n_agents)]
        Q = torch.cat(Q)
        loss_C = torch.square(Q - scalarized_reward).sum()
        if step % 10 != 0:
            self.optim.zero_grad()
            loss_C.backward()
            self.optim.step()
            self.lr_scheduler.step()
            return
        coma = torch.cat([coma[i].reshape(1, -1) for i in range(self.params.n_agents)], dim=0).detach()
        prob1 = torch.cat([self.dist1[a].probs.unsqueeze(0) for a in range(self.params.n_agents)])
        prob2 = torch.cat([self.dist2[a].probs.unsqueeze(0) for a in range(self.params.n_agents)])
        prob3 = torch.cat([self.dist3[a].probs.unsqueeze(0) for a in range(self.params.n_agents)])
        prob = torch.cat([prob1, prob2, prob3], dim=-1)
        baseline = torch.mul(coma, prob).reshape(-1, 10).sum(dim=-1)
        neg_advantage = baseline - scalarized_reward
        actions = torch.Tensor(actions)
        log_prob1 = torch.cat([self.dist1[a].log_prob(actions[a][0]).unsqueeze(0) for a in range(self.params.n_agents)]).unsqueeze(-1)
        log_prob2 = torch.cat([self.dist2[a].log_prob(actions[a][1]).unsqueeze(0) for a in range(self.params.n_agents)]).unsqueeze(-1)
        log_prob3 = torch.cat([self.dist3[a].log_prob(actions[a][2]).unsqueeze(0) for a in range(self.params.n_agents)]).unsqueeze(-1)
        log_prob = torch.cat([log_prob1, log_prob2, log_prob3], dim=-1).flatten()
        loss_A = torch.mul(neg_advantage, log_prob).sum()
        loss = loss_A + loss_C
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.lr_scheduler.step()




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_dim', default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--lr', default=0.001)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    agent = COMA_Agent(args)
    agent.choose_action(torch.Tensor([1,2,3]), torch.Tensor([0.5, 0.5]))
    agent.learn([1,1], [0.1,0.9])

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from tensorboardX import writer
import tensorboardX
import itertools

writer = tensorboardX.SummaryWriter('./logs')

EPSILON = 0.8
BATCH_SIZE = 4096
BATCH_W = 32
MEMORY_CAPACITY = 20000

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.params = params
        self.fc1 = nn.Linear(params.s_dim, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.q1 = nn.Linear(64, params.a_dim * 2)
        self.q1.weight.data.normal_(0, 0.1)
        self.q2 = nn.Linear(64, params.a_dim * 2)
        self.q2.weight.data.normal_(0, 0.1)
        self.q3 = nn.Linear(64, params.a_dim * 2)
        self.q3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.convert_type(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q1 = self.q1(x).view(-1, self.params.a_dim, 2)
        q2 = self.q2(x).view(-1, self.params.a_dim, 2)
        q3 = self.q3(x).view(-1, self.params.a_dim, 2)
        return q1, q2, q3

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.params.device)
        return x

class MIX(nn.Module):
    def __init__(self):
        super(MIX, self).__init__()
        self.mix1 = nn.Sequential(
            nn.Linear(3 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.mix2 = nn.Sequential(
            nn.Linear(3 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, q1, q2, q3, s):
        x1 = torch.cat([q1[:, :, 0], q2[:, :, 0], q3[:, :, 0], s], dim=-1)
        x2 = torch.cat([q1[:, :, 1], q2[:, :, 1], q3[:, :, 1], s], dim=-1)
        r1 = self.mix1(x1)
        r2 = self.mix2(x2)
        r = torch.cat([r1, r2], dim=-1)
        return r


class MODQN():
    def __init__(self, params):
        self.params = params
        self.net = Net(params)
        self.mixer = MIX()
        self.optim = torch.optim.Adam(itertools.chain(self.net.parameters(), self.mixer.parameters()), lr=0.001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=100, gamma=0.9, last_epoch=-1)
        self.memory = [np.zeros((MEMORY_CAPACITY, params.s_dim * 2 + 5)), 0]
        self.epsilon = EPSILON
        self.counter = 0

    def choose_action(self, s, w):
        w = torch.Tensor([[w, 1-w]]).to(self.params.device)
        # x = np.concatenate([s, w], axis=-1)
        w = torch.cat([w.unsqueeze(1) for _ in range(self.params.a_dim)], dim=1).squeeze(0)
        q1, q2, q3 = self.net(s)
        q1 = q1.squeeze(0); q2 = q2.squeeze(0); q3 = q3.squeeze(0)
        q1_scalar = torch.matmul(w.unsqueeze(1), q1.unsqueeze(-1)).squeeze(-1)
        q2_scalar = torch.matmul(w.unsqueeze(1), q2.unsqueeze(-1)).squeeze(-1)
        q3_scalar = torch.matmul(w.unsqueeze(1), q3.unsqueeze(-1)).squeeze(-1)
        a1 = torch.argmax(q1_scalar, dim=0).data.cpu().numpy()[0]
        a2 = torch.argmax(q2_scalar, dim=0).data.cpu().numpy()[0]
        a3 = torch.argmax(q3_scalar, dim=0).data.cpu().numpy()[0]
        actions = [a1, a2, a3]
        for i in range(len(actions)):
            if random.random() > self.epsilon:
                actions[i] = random.randint(0, 9)

        return actions

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [*a, r[0], r[1]], s_))
        idx = self.memory[1]
        self.memory[0][idx, :] = transition
        self.memory[1] += 1
        if self.memory[1] == MEMORY_CAPACITY:
            self.memory[1] = self.memory[1] % MEMORY_CAPACITY

    def learn(self):
        if self.epsilon > 0.95:
            self.epsilon = 0.95
        else:
            self.epsilon = self.epsilon * 1.003
        if self.memory[1] < BATCH_SIZE:
            return
        idx = np.random.choice(self.memory[1], BATCH_SIZE, replace=False)
        b_memory = self.memory[0][idx, :]
        b_s = torch.FloatTensor(b_memory[:, :self.params.s_dim])
        b_a1 = torch.LongTensor(b_memory[:, self.params.s_dim:self.params.s_dim+1].astype(int)).unsqueeze(-1)
        b_a2 = torch.LongTensor(b_memory[:, self.params.s_dim+1:self.params.s_dim+2].astype(int)).unsqueeze(-1)
        b_a3 = torch.LongTensor(b_memory[:, self.params.s_dim+2:self.params.s_dim+3].astype(int)).unsqueeze(-1)
        b_r1 = torch.FloatTensor(b_memory[:, self.params.s_dim+3:self.params.s_dim+4])
        b_r2 = torch.FloatTensor(b_memory[:, self.params.s_dim+4:self.params.s_dim+5])
        b_s_ = torch.FloatTensor(b_memory[:, self.params.s_dim+5:self.params.s_dim+5+self.params.s_dim])
        b_a1 = torch.cat([b_a1, b_a1], dim=-1)
        b_a2 = torch.cat([b_a2, b_a2], dim=-1)
        b_a3 = torch.cat([b_a3, b_a3], dim=-1)
        b_r = torch.cat([b_r1, b_r2], dim=-1)
        # b_w = [random.random() for _ in range(BATCH_W)]
        # b_w = [[w, 1-w] for w in b_w]
        # b_w = torch.Tensor([b_w for i in range(BATCH_SIZE)]).to(self.params.device)
        # b_s = torch.cat([b_s.unsqueeze(1) for _ in range(BATCH_W)], dim=1)
        q1_eval, q2_eval, q3_eval = self.net(b_s)
        q1_eval = q1_eval.gather(dim=1, index=b_a1)
        q2_eval = q2_eval.gather(dim=1, index=b_a2)
        q3_eval = q3_eval.gather(dim=1, index=b_a3)
        # q_eval = self.mixer(q1_eval, q2_eval, q3_eval, b_s)
        q_eval = (q1_eval + q2_eval + q3_eval).squeeze(1)
        loss = (q_eval - b_r).pow(2)
        loss = loss.sum(dim=-1)
        loss = torch.mean(loss, dim=0)
        writer.add_scalar("loss", loss.data.cpu().numpy(), self.counter)
        self.counter += 1
        # loss = torch.nn.MSELoss(q_eval, b_r)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.lr_scheduler.step()




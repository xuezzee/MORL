import torch
import numpy as np

class ReplayBuffer():
    def __init__(self, args):
        self.args = args
        self.to_gpu = args.gpu
        self.obs_buffs = [np.zeros((args.buffer_size, args.o_dim)) for _ in range(args.n_agents)]
        self.act_buffs = [np.zeros((args.buffer_size, 3)) for _ in range(args.n_agents)]
        self.next_obs_buffs = [np.zeros((args.buffer_size, args.o_dim)) for _ in range(args.n_agents)]
        self.done_buffs = [np.zeros((args.buffer_size, 1)) for _ in range(args.n_agents)]
        self.state_buffs = np.zeros((args.buffer_size, args.s_dim))
        self.next_state_buffs = np.zeros((args.buffer_size, args.s_dim))
        self.rew_buffs = np.zeros((args.buffer_size, args.n_obj))
        self.pref_buffs = np.zeros((args.buffer_size, args.n_obj))

        self.filled_i = 0
        self.curr_i = 0

    def __len__(self):
        return self.filled_i

    def push(self, obs, act, rew, next_obs, dones, state, next_state, pref):
        obs = self.convert_type(obs)
        act = self.convert_type(act).transpose((1,0,2))
        rew = self.convert_type(rew)
        next_obs = self.convert_type(next_obs)
        dones = self.convert_type(dones)
        state = self.convert_type(state)
        next_state = self.convert_type(next_state)
        pref = self.convert_type(pref)

        nentries = obs.shape[0]
        if self.curr_i + nentries > self.args.buffer_size:
            rollover = self.args.buffer_size - self.curr_i
            for a in range(self.args.n_agents):
                self.obs_buffs[a] = np.roll(self.obs_buffs[a], rollover, axis=0)
                self.act_buffs[a] = np.roll(self.act_buffs[a], rollover, axis=0)
                self.rew_buffs[a] = np.roll(self.rew_buffs[a], rollover, axis=0)
                self.done_buffs[a] = np.roll(self.done_buffs[a], rollover, axis=0)
                self.state_buffs[a] = np.roll(self.state_buffs[a], rollover, axis=0)
                self.next_obs_buffs[a] = np.roll(self.next_obs_buffs[a], rollover, axis=0)
                self.next_state_buffs[a] = np.roll(self.next_state_buffs[a], rollover, axis=0)
                # self.pref_buffs[a] = np.roll(self.next_obs_buffs[a], rollover, axis=0)
            self.curr_i = 0
            self.filled_i = self.args.buffer_size

        self.done_buffs[0][self.curr_i:self.curr_i + nentries] = dones[0]
        self.state_buffs[self.curr_i:self.curr_i + nentries] = state
        self.next_state_buffs[self.curr_i:self.curr_i + nentries] = next_state
        self.rew_buffs[self.curr_i:self.curr_i + nentries] = rew
        # self.pref_buffs[self.curr_i:self.curr_i + nentries] = pref
        for a in range(self.args.n_agents):
            self.obs_buffs[a][self.curr_i:self.curr_i + nentries] = np.vstack(obs[:, a])
            self.act_buffs[a][self.curr_i:self.curr_i + nentries] = act[a]
            self.next_obs_buffs[a][self.curr_i:self.curr_i + nentries] = np.vstack(next_obs[:, a])

        self.curr_i += nentries
        if self.filled_i < self.args.buffer_size:
            self.filled_i += nentries
        if self.curr_i == self.args.buffer_size:
            self.curr_i = 0

    def convert_type(self, input):
        if not isinstance(input, np.ndarray):
            input = np.array(input)

        return input

    def sample(self, batch_size):
        inds = np.random.choice(np.arange(self.filled_i), size=batch_size, replace=False)
        if self.to_gpu:
            cast = lambda x: torch.autograd.Variable(torch.Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: torch.autograd.Variable(torch.Tensor(x), requires_grad = False)

        if self.args.norm_rews:
            ret_rews = cast((self.rew_buffs[inds] -
                              self.rew_buffs[:self.filled_i].mean()) /
                             (self.rew_buffs[:self.filled_i].std() + 0.1e-10))

        else:
            ret_rews = cast(self.rew_buffs[inds])

        return  {"obs":[cast(self.obs_buffs[i][inds]) for i in range(self.args.n_agents)],
                "act":[cast(self.act_buffs[i][inds]) for i in range(self.args.n_agents)],
                "next_obs":[cast(self.next_obs_buffs)[i][inds] for i in range(self.args.n_agents)],
                "state":cast(self.state_buffs)[inds],
                "next_state":cast(self.next_state_buffs)[inds],
                # "pref":cast(self.pref_buffs)[inds],
                "rew":ret_rews,}
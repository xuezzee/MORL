import torch
import numpy as np
import random

class Preference():
    def __init__(self, args):
        self.device = args.device
        self.n_obj = args.n_obj
        self.pref_dist = args.preference_distribution
        self.n_agents = args.n_agents
        self.n_threads = args.n_threads

    def sample(self, batch_size, require_tensor=True, train=False):
        w = []
        if train:
            if self.n_obj == 2 and self.pref_dist == "uniform":
                for i in range(batch_size):
                    # w1 = random.random()
                    w1 = 1
                    w2 = 1 - w1
                    w.append([w1, w2])
                if require_tensor:
                    preference = [torch.Tensor(w).to(self.device) for _ in range(self.n_agents)]
                else:
                    preference = [w for _ in range(self.n_agents)]
                return preference

        else:
            if self.n_obj == 2 and self.pref_dist == "uniform":
                for t in range(self.n_threads):
                    for i in range(batch_size):
                        # w1 = random.random()
                        w1 = 1
                        w2 = 1 - w1
                        if require_tensor:
                            w.append(torch.Tensor([[w1, w2] for _ in range(self.n_agents)]).to(self.device))
                        else:
                            w.append([[w1, w2] for _ in range(self.n_agents)])
                    preference = [w for _ in range(self.n_threads)]
            return preference
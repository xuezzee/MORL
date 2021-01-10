import torch
import numpy as np
import argparse
import sys
import os
from MORL.MODQN import MODQN
from MORL.A2C import Agent
from MORL.COMA import COMA_Agent
from utils.ReplayBuffer import ReplayBuffer
from envs.comm import CommEnv
from utils.env_wrapper import EnvWrapper, EnvWrapper_sigle
from utils.preference_pool import Preference
import tensorboardX
import copy
import time
import random
import matplotlib.pyplot as plt

MODE = "COMA"

def run():
    agent_args = get_args()
    env_args = get_env_args()
    agent_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", agent_args.device)
    writer = tensorboardX.SummaryWriter(env_args.log_dir)
    env = CommEnv(env_args)
    agent_args.o_dim = env.observation_space.shape[0]
    agent_args.s_dim = env.state_space.shape[0]
    agent_args.a_dim = env.action_space.n

    if MODE == "DQN":
        agents = MODQN(agent_args)
        total_step = 1
        update_step = 0
        record_step = 0
        x_plot = []
        y_plot = []
        for ep in range(agent_args.epoches // agent_args.n_threads):
            print('episode:{0}--------------------------------------------'.format(ep))
            obs = env.reset()
            # w = random.random()
            w = 0
            tot_rew = np.array([0, 0])
            tot_scal_rew = 0
            for step in range(1000):
                w = random.random()
                acts = [agents.choose_action(obs, w)]
                # acts = [[5, 5, 5]]
                # print("obs:",obs)
                # print("acts:",acts)
                obs_ , reward, done, info = env.step(acts)
                tot_rew = tot_rew + np.array(reward)
                tot_scal_rew += reward[0] * w + reward[1] * (1-w)
                # print(reward)
                agents.store_transition(obs[0], acts[0], reward, obs_[0])
                obs = obs_
                if step % 100 == 0:
                    agents.learn()
                # if total_step % 1000 == 0:
                #     agents.save_model(total_step)
                # total_step += agent_args.n_threads
                total_step += 1
                update_step += 1
                record_step += 1
            print("ep reward:{0}, scalarized_reward:{1}, preference:{2}".format(tot_rew, tot_scal_rew, [w, 1-w]))
            if ep >= 100:
                x_plot.append(tot_rew[0])
                y_plot.append(tot_rew[1])
            if ep >= 100 and ep % 20 == 0:
                plt.scatter(x_plot, y_plot)
                plt.show()
            print("epsilon:", agents.epsilon)

        #test phase
        print("================================================================")
        print("                          test phase")
        print("================================================================")
        agents.epsilon = 1
        for ep in range(1000):
            print('episode:{0}--------------------------------------------'.format(ep))
            w = random.random()
            obs = env.reset()
            tot_rew = np.array([0, 0])
            tot_scal_rew = 0
            for step in range(3000):
                acts = [agents.choose_action(obs, w)]
                obs_ , reward, done, info = env.step(acts)
                tot_rew = tot_rew + np.array(reward)
                tot_scal_rew += reward[0] * w + reward[1] * (1-w)
            print("ep_reward:{0}, scalarized_reward:{1}, preference:{2}".format(tot_rew, tot_scal_rew, [w, 1 - w]))
            if ep >= 100:
                x_plot.append(tot_rew[0])
                y_plot.append(tot_rew[1])
            if ep >= 100 and ep % 20 == 0:
                plt.scatter(x_plot, y_plot)
                plt.show()
            # writer.add_scalar("total reward:", tot_rew, ep)
    elif MODE == "PG":
        agent = [Agent(agent_args), Agent(agent_args)]
        for ep in range(agent_args.epoches):
            obs = env.reset()
            w = random.random()
            w = 0.5
            tot_rew = np.array([0, 0])
            tot_scal_rew = 0
            for step in range(3000):
                # w = random.random()
                acts = [agent[0].choose_action(obs[0], [w, 1-w]),]
                        # agent[1].choose_action(obs[1], [w, 1-w])]
                # print(acts)
                obs, reward, done, info = env.step(acts)
                average_rew = tot_scal_rew / (step + 0.0001)
                # reward[0] = reward[0] + 1; reward[1] = reward[1] + 1
                agent[0].learn(reward, [w, 1-w], average_rew)
                # agent[1].learn(reward, [w, 1-w], average_rew)
                tot_scal_rew += reward[0] * w + reward[1] * (1-w)
            print("reward:", tot_scal_rew)
            print("w:", w, 1-w)

        # test phase
        print("================================================================")
        print("                          test phase")
        print("================================================================")
        x_plot = []
        y_plot = []
        for ep in range(1000):
            print('episode:{0}--------------------------------------------'.format(ep))
            w = random.random()
            obs = env.reset()
            tot_rew = np.array([0, 0])
            tot_scal_rew = 0
            for step in range(3000):
                acts = [agent[0].choose_action(obs[0], [w, 1-w]),
                        agent[0].choose_action(obs[1], [w, 1-w])]
                obs_, reward, done, info = env.step(acts)
                tot_rew = tot_rew + np.array(reward)
                tot_scal_rew += reward[0] * w + reward[1] * (1 - w)
            print("ep_reward:{0}, scalarized_reward:{1}, preference:{2}".format(tot_rew, tot_scal_rew, [w, 1 - w]))
            x_plot.append(tot_rew[0])
            y_plot.append(tot_rew[1])
            if ep >= 100 and ep % 20 == 0:
                plt.scatter(x_plot, y_plot)
                plt.show()

    elif MODE == 'COMA':
        agent = COMA_Agent(agent_args)
        for ep in range(agent_args.epoches):
            obs = env.reset()
            w = random.random()
            w = 0
            tot_rew = np.array([0, 0])
            tot_scal_rew = 0
            for step in range(1000):
                w = random.random()
                acts = agent.choose_action(obs, [w, 1 - w])
                # acts = [[9, 1, 1] for j in range(agent_args.n_agents)]
                obs, reward, done, info = env.step(acts)
                state = env.get_state()
                if ep != 0:
                    agent.learn(state, obs, reward, [w, 1 - w], acts, step)
                tot_scal_rew += reward[0] * w + reward[1] * (1-w)
            print("reward:", tot_scal_rew)
            # test phase
        print("================================================================")
        print("                          test phase")
        print("================================================================")
        x_plot = []
        y_plot = []
        for ep in range(1000):
            print('episode:{0}--------------------------------------------'.format(ep))
            w = random.random()
            obs = env.reset()
            tot_rew = np.array([0, 0])
            tot_scal_rew = 0
            for step in range(3000):
                acts = agent.choose_action(obs, [w, 1 - w])
                obs_, reward, done, info = env.step(acts)
                tot_rew = tot_rew + np.array(reward)
                tot_scal_rew += reward[0] * w + reward[1] * (1 - w)
            print("ep_reward:{0}, scalarized_reward:{1}, preference:{2}".format(tot_rew, tot_scal_rew, [w, 1 - w]))
            x_plot.append(tot_rew[0])
            y_plot.append(tot_rew[1])
            if ep >= 40 and ep % 20 == 0:
                plt.scatter(x_plot, y_plot)
                plt.show()


def baseline():
    env_args = get_env_args()
    env_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CommEnv(env_args)
    plot_x = []
    plot_y = [[],[],[],[],[],[],[]]
    for ep in range(1000):
        obs = env.reset()
        # w = random.random()
        w = ['0', '0.2', '0.3', '0.5', '0.7', '0.8', '1']
        tot_rew = np.array([0, 0])
        tot_scal_rew = [0 for _ in range(7)]
        acts = [[ep // 100, ep % 100 // 10, ep % 10]]
        print("ep:{0}, act:{1}".format(ep, acts))
        for step in range(3000):
            obs, reward, done, info = env.step(acts)
            # average_rew = tot_scal_rew / (step + 0.0001)
            tot_scal_rew = [tot_scal_rew[0] + reward[0] * 0 + reward[1] * 1,
                             tot_scal_rew[1] + reward[0] * 0.2 + reward[1] * 0.8,
                             tot_scal_rew[2] + reward[0] * 0.3 + reward[1] * 0.7,
                             tot_scal_rew[3] + reward[0] * 0.5 + reward[1] * 0.5,
                             tot_scal_rew[4] + reward[0] * 0.7 + reward[1] * 0.3,
                             tot_scal_rew[5] + reward[0] * 0.8 + reward[1] * 0.2,
                             tot_scal_rew[6] + reward[0] * 1 + reward[1] * 0]
            tot_rew = tot_rew + reward
        print("reward:", tot_scal_rew, "reward:",tot_rew)
        # print("w:", w, 1 - w)
        plot_x.append(ep)
        ax = [plt.subplot(3,3,i+1,title='w='+w[i]) for i in range(7)]
        for i in range(7):
            plot_y[i].append(tot_scal_rew[i])
            ax[i].plot(plot_x, plot_y[i])
        if ep % 100 == 0:
            plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", default=4)
    parser.add_argument("--gpu", default=True)
    parser.add_argument("--buffer_size", default=50000)
    parser.add_argument("--h_dim", default=128)
    parser.add_argument("--n_obj", default=2)
    parser.add_argument("--hyper_h1", default=128)
    parser.add_argument("--n_threads", default=1)
    parser.add_argument("--batch_size", default=10000)
    parser.add_argument("--epoches", default=40)
    parser.add_argument('--preference_distribution', default="uniform")
    parser.add_argument('--epsilon', default=0.2)
    parser.add_argument('--norm_rews', default=True)
    parser.add_argument('--learning_rate', default=1e-7)
    parser.add_argument('--update_step', default=300)
    parser.add_argument('--batch_size_p', default=1)
    parser.add_argument('--update_times', default=2)
    parser.add_argument('--lr', default=0.01)

    return parser.parse_args()

def get_env_args():
    parser = argparse.ArgumentParser(description="computation offloading environment")
    parser.add_argument('--fe', default=10**14)
    parser.add_argument('--fc', default=10**15)
    parser.add_argument('--alpha', default=10**8)
    parser.add_argument('--beta', default=10**(-46))
    parser.add_argument('--T_max', default=8)
    parser.add_argument('--lam', default=10)
    parser.add_argument('--mean_normal', default=100000)
    parser.add_argument('--var_normal', default=10000)
    parser.add_argument('--num_user', default=4)
    parser.add_argument('--processing_period', default=0.1)
    parser.add_argument('--discrete', default=True)
    parser.add_argument("--n_threads", default=1)
    parser.add_argument('--log_dir', default='./log5')

    return parser.parse_args()


if __name__ == '__main__':
    run()
    # baseline()
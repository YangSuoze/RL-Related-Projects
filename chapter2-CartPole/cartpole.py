#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


# 初始化网络
class Net(nn.Module):
    # 4*128*2
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# 初始化不可变字典
Episode = namedtuple("Episode", field_names=["reward", "steps"])
EpisodeStep = namedtuple("EpisodeStep", field_names=["observation", "action"])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        # 将np转换为PyTorch的浮点张量
        obs_v = torch.FloatTensor([obs])
        # 将神经网络的输出转换为概率分布
        act_probs_v = sm(net(obs_v))
        # .data访问张量中的数据，忽略梯度信息
        # .numpy()将tensor转为numpy数组
        # [0]选择第一个数据，我们这里batch为1因此取出第一个数据即可
        act_probs = act_probs_v.data.numpy()[0]
        # 在0，1两个整数间按照act_probs进行采样得到当前的动作action
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _, _ = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:
            # 每个完整的片段需要保留两个属性：该轮游戏中的累积奖励，所有步骤列表，每个步骤包含当前的观察对应的动作
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    # 从生成的batch中过滤训练的数据
    rewards = list(map(lambda s: s.reward, batch))
    # 计算reward下限值和reward平均值，reward_bound是过滤训练集的标准，reward_mean是终止训练的条件
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    # 当前batch中只有reward >= reward_bound的obs和action会加入到训练数据中进行训练
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        print("满足奖励下限的集合数量", obs_v.shape[0])
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print(
            f"{iter_no}: loss={loss_v.item()}, reward_mean={reward_m}, rw_bound={reward_b}"
        )
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()

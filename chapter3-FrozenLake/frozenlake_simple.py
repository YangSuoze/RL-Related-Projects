"""
rl训练
Author: yangjie
Date: 2024-09-22
"""

import os
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from env import Env
from model import Net

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 80


Episode = namedtuple("Episode", field_names=["reward", "steps"])
EpisodeStep = namedtuple("EpisodeStep", field_names=["observation", "action"])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _, _ = env.step(action)
        # print(next_obs, reward, is_done)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = Env()
    obs_size = 16
    n_actions = 4
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-frozenlake-naive")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print(
            "%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f"
            % (iter_no, loss_v.item(), reward_m, reward_b)
        )
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m >= 13 and loss_v.item() < 0.10:
            current_file_path = os.path.abspath(__file__)
            current_dir_path = os.path.dirname(current_file_path)
            model_path = current_dir_path + "/models"
            if not os.path.exists(model_path):
                # 如果文件夹不存在，则创建它
                os.makedirs(model_path)
                print(f"文件夹 '{model_path}' 已创建。")
            else:
                print(f"文件夹 '{model_path}' 已存在。")
            torch.save(
                net.state_dict(),
                f"{model_path}/model_parameters_best.pth",
            )
            print("Solved!")
            break
    writer.close()

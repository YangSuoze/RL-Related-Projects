"""
Using rl model to play games.
Author: yangjie
Date: 2024-09-22
"""

import os
import torch
from model import Net
from env import Env


def play_game():
    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)
    model_path = current_dir_path + "/models/model_parameters_best.pth"
    model = Net(16, 128, 4)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    sm = torch.nn.Softmax(dim=1)
    env = Env()
    obs, _ = env.reset()
    print("真实游戏中模型实战结果：")
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(model(obs_v))
        act_probs = act_probs_v.data.numpy()[0].tolist()
        action = act_probs.index(max(act_probs))
        nxt_obs, r, is_done, _, _ = env.step(action)
        print(f"action:{action}  reward:{r}  is_done:{is_done}")
        print(f"nxt_obs:{nxt_obs}")

        obs = nxt_obs
        if is_done:
            break


if __name__ == "__main__":
    play_game()

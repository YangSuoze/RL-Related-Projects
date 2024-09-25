"""
自己实现一个迷宫环境Env
Author: yangjie
Date: 2024-09-22
"""

import random


def mhd_distance(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def get_distance(cur_index, next_index, target_index):
    map = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    ]
    if mhd_distance(map[cur_index], map[target_index]) < mhd_distance(
        map[next_index], map[target_index]
    ):
        return -1
    elif mhd_distance(map[cur_index], map[target_index]) > mhd_distance(
        map[next_index], map[target_index]
    ):
        return 1
    else:
        return -1


class Env:
    def __init__(self):
        # 0是普通位置，1是障碍物，2是奖励点
        self.observation = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.obstacles = [5, 7, 11, 12]
        self.target_index = 15
        self.done = False

    def replace_action(self, action):
        action_map = {0: [3, 2], 1: [2, 3], 2: [1, 0], 3: [0, 1]}
        if random.random() < 0.66:
            replace_one = random.sample(action_map[action], 1)[0]
            return replace_one
        return action

    def reset(self):
        self.done = False
        self.observation = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return self.observation, ""

    def step(self, action):
        action = self.replace_action(action)
        cur_index = self.observation.index(1)
        if action == 0:
            # top
            next_index = cur_index - 4
        elif action == 1:
            # bottom
            next_index = cur_index + 4
        elif action == 2:
            # left
            next_index = cur_index - 1
            if cur_index % 4 < next_index % 4:
                next_index = -1
        else:
            # right
            next_index = cur_index + 1
            if next_index % 4 < cur_index % 4:
                next_index = -1
        if next_index == self.target_index:
            reward = 8
            self.observation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            self.done = True
        elif next_index < 0 or next_index > 15:
            reward = 0
        elif next_index in self.obstacles:
            reward = -1
            self.done = True
            self.observation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.observation[next_index] = 1
        else:
            reward = get_distance(cur_index, next_index, self.target_index)
            self.observation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.observation[next_index] = 1
        return self.observation, reward, self.done, "", ""


if __name__ == "__main__":
    env = Env()
    obs, _ = env.reset()
    while True:
        action = random.sample([0, 1, 2, 3], 1)[0]
        nxt_obs, r, is_done, _, _ = env.step(action)
        print(action, nxt_obs, r, is_done)
        if is_done:
            break

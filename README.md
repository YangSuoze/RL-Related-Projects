## 创建强化学习环境
conda create --name env_name python=3.11
## 安装环境
```shell
pip install opencv-python
pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1
pip install tensorboard
pip install tensorboardX
pip install gym
pip install gym[atari]
pip install gym[accept-rom-license]
# 对应numpy修改bool8为bool
```
可直接安装requirements.txt对应环境即可
## 启动tensorboard查看日志
```shell
tensorboard --logdir=runs --port=6007 --host=0.0.0.0
```
## chapter1-GAN网络效果图
### loss变化
![dis_loss图](imgs/dis_loss.png)
![gen_loss图](imgs/gen_loss.png)
### 生成的虚假图像和真实图像对比
![生成器生成的图和真实的实验图](imgs/fake_real_res.png)
## chapter2-CartPole小车摆动实验
### loss变化
![loss变化](imgs/cartpole_loss.png)
### 平均奖励和奖励下限的变化
![平均奖励和奖励下限的变化](imgs/cartpole_res.png)
## chapter3-Frozenlake简单场景
### 环境
![环境](chapter3-FrozenLake/imgs/env.png)
### 训练过程
![训练过程](chapter3-FrozenLake/imgs/training_process.png)
### 使用训练的模型玩游戏
![使用训练的模型玩游戏](chapter3-FrozenLake/imgs/play_game.png)
可以看到精准使用最快的方式达到终点


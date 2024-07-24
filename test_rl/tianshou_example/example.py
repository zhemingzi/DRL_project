import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts

# 定义一些超参数：

task = 'CartPole-v1'
lr, epoch, batch_size = 1e-3, 10, 64
train_num, test_num = 10, 100
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, step_per_collect = 10000, 10

# 初始化记录器：

logger = ts.utils.TensorboardLogger(SummaryWriter('test_rl/tianshou_example/log/dqn'))
# For other loggers, see https://tianshou.readthedocs.io/en/master/01_tutorials/05_logger.html

# 创建环境：

# You can also try SubprocVectorEnv, which will use parallelization
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])


# 创建网络及其优化器：
from tianshou.utils.net.common import Net
# Note: You can easily define other networks.
# See https://tianshou.readthedocs.io/en/master/01_tutorials/00_dqn.html#build-the-network
env = gym.make(task, render_mode="human")
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 设置策略和收集器：

policy = ts.policy.DQNPolicy(
    model=net,
    optim=optim,
    discount_factor=gamma, 
    action_space=env.action_space,
    estimation_step=n_step,
    target_update_freq=target_freq
)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

# 训练：

result = ts.trainer.OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=epoch,
    step_per_epoch=step_per_epoch,
    step_per_collect=step_per_collect,
    episode_per_test=test_num,
    batch_size=batch_size,
    update_per_step=1 / step_per_collect,
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    logger=logger,
).run()
print(f"Finished training in {result.timing.total_time} seconds")

# 保存/加载经过训练的策略（与加载完全相同）：torch.nn.module

torch.save(policy.state_dict(), 'test_rl/tianshou_example/dqn.pth')
# policy.load_state_dict(torch.load('dqn.pth'))

# # 以 35 FPS 观看代理：
# policy.eval()
# policy.set_eps(eps_test)
# collector = ts.data.Collector(policy, env, exploration_noise=True)
# collector.collect(n_episode=1, render=1 / 35)
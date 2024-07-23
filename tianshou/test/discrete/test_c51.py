import argparse
import os
import pickle

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import (
    Collector,
    PrioritizedVectorReplayBuffer,
    ReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.env import DummyVectorEnv
from tianshou.policy import C51Policy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=8000)
    parser.add_argument("--step-per-collect", type=int, default=8)
    parser.add_argument("--update-per-step", type=float, default=0.125)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128])
    parser.add_argument("--training-num", type=int, default=8)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--prioritized-replay", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--save-interval", type=int, default=4)
    return parser.parse_known_args()[0]


def test_c51(args: argparse.Namespace = get_args()) -> None:
    env = gym.make(args.task)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v1": 195}
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )
    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax=True,
        num_atoms=args.num_atoms,
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy: C51Policy = C51Policy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=args.gamma,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)
    # buffer
    buf: ReplayBuffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, args.task, "c51")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=args.save_interval)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    def train_fn(epoch: int, env_step: int) -> None:
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / 40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.set_eps(args.eps_test)

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            },
            ckpt_path,
        )
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(train_collector.buffer, f)
        return ckpt_path

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint["model"])
            policy.optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                train_collector.buffer = pickle.load(f)
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")

    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        resume_from_log=args.resume,
        save_checkpoint_fn=save_checkpoint_fn,
    ).run()
    assert stop_fn(result.best_reward)


def test_c51_resume(args: argparse.Namespace = get_args()) -> None:
    args.resume = True
    test_c51(args)


def test_pc51(args: argparse.Namespace = get_args()) -> None:
    args.prioritized_replay = True
    args.gamma = 0.95
    args.seed = 1
    test_c51(args)

#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from examples.offline.utils import load_buffer_d4rl
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.policy import CQLPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OfflineTrainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="Hopper-v2",
        help="The name of the OpenAI Gym environment to train on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random seed to use.",
    )
    parser.add_argument(
        "--expert-data-task",
        type=str,
        default="hopper-expert-v2",
        help="The name of the OpenAI Gym environment to use for expert data collection.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000000,
        help="The size of the replay buffer.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        default=[256, 256],
        help="The list of hidden sizes for the neural networks.",
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=1e-4,
        help="The learning rate for the actor network.",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=3e-4,
        help="The learning rate for the critic network.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="The weight of the entropy term in the loss function.",
    )
    parser.add_argument(
        "--auto-alpha",
        default=True,
        action="store_true",
        help="Whether to use automatic entropy tuning.",
    )
    parser.add_argument(
        "--alpha-lr",
        type=float,
        default=1e-4,
        help="The learning rate for the entropy tuning.",
    )
    parser.add_argument(
        "--cql-alpha-lr",
        type=float,
        default=3e-4,
        help="The learning rate for the CQL entropy tuning.",
    )
    parser.add_argument(
        "--start-timesteps",
        type=int,
        default=10000,
        help="The number of timesteps before starting to train.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=200,
        help="The number of epochs to train for.",
    )
    parser.add_argument(
        "--step-per-epoch",
        type=int,
        default=5000,
        help="The number of steps per epoch.",
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=3,
        help="The number of steps to use for N-step TD learning.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="The batch size for training.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="The soft target update coefficient.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature for the Boltzmann policy.",
    )
    parser.add_argument(
        "--cql-weight",
        type=float,
        default=1.0,
        help="The weight of the CQL loss term.",
    )
    parser.add_argument(
        "--with-lagrange",
        type=bool,
        default=True,
        help="Whether to use the Lagrange multiplier for CQL.",
    )
    parser.add_argument(
        "--calibrated",
        type=bool,
        default=True,
        help="Whether to use calibration for CQL.",
    )
    parser.add_argument(
        "--lagrange-threshold",
        type=float,
        default=10.0,
        help="The Lagrange multiplier threshold for CQL.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor")
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1,
        help="The frequency of evaluation.",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=10,
        help="The number of episodes to evaluate for.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="log",
        help="The directory to save logs to.",
    )
    parser.add_argument(
        "--render",
        type=float,
        default=1 / 35,
        help="The frequency of rendering the environment.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to train on (cpu or cuda).",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="The path to the checkpoint to resume from.",
    )
    parser.add_argument(
        "--resume-id",
        type=str,
        default=None,
        help="The ID of the checkpoint to resume from.",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="offline_d4rl.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()


def test_cql() -> None:
    args = get_args()
    env = gym.make(args.task)
    assert isinstance(env.action_space, gym.spaces.Box)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.max_action = space_info.action_info.max_action
    args.min_action = space_info.action_info.min_action
    print("device:", args.device)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", args.min_action, args.max_action)

    args.state_dim = space_info.observation_info.obs_dim
    args.action_dim = space_info.action_info.action_dim
    print("Max_action", args.max_action)

    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)

    # model
    # actor network
    net_a = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    )
    actor = ActorProb(
        net_a,
        action_shape=args.action_shape,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # critic network
    net_c1 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic = Critic(net_c1, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -args.action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy: CQLPolicy = CQLPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        action_space=env.action_space,
        critic2=critic2,
        critic2_optim=critic2_optim,
        calibrated=args.calibrated,
        cql_alpha_lr=args.cql_alpha_lr,
        cql_weight=args.cql_weight,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        temperature=args.temperature,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        min_action=args.min_action,
        max_action=args.max_action,
        device=args.device,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    test_collector = Collector(policy, test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "cql"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger: WandbLogger | TensorboardLogger
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
        logger.load(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def watch() -> None:
        if args.resume_path is None:
            args.resume_path = os.path.join(log_path, "policy.pth")

        policy.load_state_dict(torch.load(args.resume_path, map_location=torch.device("cpu")))
        collector = Collector(policy, env)
        collector.collect(n_episode=1, render=1 / 35)

    if not args.watch:
        replay_buffer = load_buffer_d4rl(args.expert_data_task)
        # trainer
        result = OfflineTrainer(
            policy=policy,
            buffer=replay_buffer,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
        ).run()
        pprint.pprint(result)
    else:
        watch()

    # Let's watch its performance!
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(collector_stats)


if __name__ == "__main__":
    test_cql()

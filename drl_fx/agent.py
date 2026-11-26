"""The RL trader.

Architecture, following the course:
  * one small recurrent network (GRU = the "DRNN") per data type -- price/volume features and,
    if present, news/Fed sentiment -- instead of one network blending everything together;
  * their representations + current position feed the "command center": an actor-critic
    policy (PPO) that picks buy / sell / hold.
Training: stage 1 rewards PnL net of costs, stage 2 continues on the differential Sharpe ratio with
a lower learning rate.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv


class SplitGRUExtractor(BaseFeaturesExtractor):
    """Separate GRU encoders per input group, concatenated with the current position."""

    def __init__(self, observation_space: gym.spaces.Box, window: int, groups: list[list[int]],
                 hidden: int = 32, dropout: float = 0.1):
        n_cols = sum(len(g) for g in groups)
        flat = sorted(i for g in groups for i in g)
        if flat != list(range(n_cols)):
            raise ValueError(f"input groups must use each column 0..{n_cols - 1} exactly once, got {groups}")
        if observation_space.shape[0] != window * n_cols + 1:
            raise ValueError(f"observation size {observation_space.shape[0]} does not match "
                             f"window {window} x {n_cols} columns + position")
        super().__init__(observation_space, features_dim=hidden * len(groups) + 1)
        self.window, self.n_cols, self.groups = window, n_cols, groups
        self.encoders = nn.ModuleList(nn.GRU(len(g), hidden, batch_first=True) for g in groups)
        self.drop = nn.Dropout(dropout)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs[:, :-1].reshape(-1, self.window, self.n_cols)
        pos = obs[:, -1:]
        outs = []
        for enc, g in zip(self.encoders, self.groups):
            _, h = enc(x[:, :, g])
            outs.append(self.drop(h[-1]))
        return torch.cat([*outs, pos], dim=1)


def build_agent(env_fn, window: int, groups: list[list[int]], lr: float = 3e-4, hidden: int = 32,
                dropout: float = 0.1, seed: int = 0, n_envs: int = 4, ent_coef: float = 0.01) -> PPO:
    venv = DummyVecEnv([lambda i=i: env_fn(seed + i) for i in range(n_envs)])
    policy_kwargs = dict(
        features_extractor_class=SplitGRUExtractor,
        features_extractor_kwargs=dict(window=window, groups=groups, hidden=hidden, dropout=dropout),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )
    return PPO("MlpPolicy", venv, learning_rate=lr, n_steps=512, batch_size=256, n_epochs=5,
               gamma=0.99, gae_lambda=0.95, ent_coef=ent_coef, clip_range=0.2,
               policy_kwargs=policy_kwargs, seed=seed, verbose=0, device="cpu")


def train_two_stage(model: PPO, sharpe_env_fn, steps_pnl: int, steps_sharpe: int,
                    seed: int = 0, n_envs: int = 4, lr_stage2: float = 1e-4) -> PPO:
    model.learn(total_timesteps=steps_pnl, progress_bar=False)
    if steps_sharpe > 0:
        model.set_env(DummyVecEnv([lambda i=i: sharpe_env_fn(seed + 100 + i) for i in range(n_envs)]))
        _set_lr(model, lr_stage2)
        model.learn(total_timesteps=steps_sharpe, reset_num_timesteps=False, progress_bar=False)
    return model


def fine_tune(model: PPO, env_fn, steps: int, lr: float = 2e-5, seed: int = 0, n_envs: int = 4) -> PPO:
    """'Keep on training with smaller learning rates' on the newest data before going live."""
    model.set_env(DummyVecEnv([lambda i=i: env_fn(seed + 200 + i) for i in range(n_envs)]))
    _set_lr(model, lr)
    model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False)
    return model


def _set_lr(model: PPO, lr: float):
    model.learning_rate = lr
    model.lr_schedule = lambda _: lr
    for g in model.policy.optimizer.param_groups:
        g["lr"] = lr

"""Small Stable-Baselines3 weight readers used by evaluation scripts.

The original training used SB3, but a fresh machine may not have SB3
installed. These helpers load the saved zip checkpoints and run the tiny
MLPs directly with NumPy. They intentionally implement only the two
policies trained for this project:

* PPO MlpPolicy with two 128-unit tanh layers and continuous Box action.
* DQN MlpPolicy with two 128-unit relu layers and a 16-action table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import pickle
import sys
import types

import numpy as np


def _load_policy_state_dict(path: str | Path) -> dict[str, np.ndarray]:
    """Load SB3 `policy.pth` from a model zip as NumPy arrays."""
    import torch

    with ZipFile(path) as zf:
        with zf.open("policy.pth") as fh:
            state_dict = torch.load(fh, map_location="cpu")
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state_dict.items()}


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


class _PickleDummy:
    def __setstate__(self, state):
        self.__dict__.update(state)


class _VecNormalizeDummy(_PickleDummy):
    pass


class _RunningMeanStdDummy(_PickleDummy):
    pass


class _BoxDummy(_PickleDummy):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _install_pickle_shims() -> dict[str, types.ModuleType | None]:
    """Install tiny class shims so SB3 VecNormalize pickle can be read.

    The eval machine does not need Stable-Baselines3 at runtime; it only needs
    the saved observation-normalization statistics.
    """
    names = [
        "stable_baselines3",
        "stable_baselines3.common",
        "stable_baselines3.common.vec_env",
        "stable_baselines3.common.vec_env.vec_normalize",
        "stable_baselines3.common.running_mean_std",
        "gymnasium",
        "gymnasium.spaces",
        "gymnasium.spaces.box",
    ]
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["stable_baselines3.common.vec_env.vec_normalize"].VecNormalize = _VecNormalizeDummy
    sys.modules["stable_baselines3.common.running_mean_std"].RunningMeanStd = _RunningMeanStdDummy
    sys.modules["gymnasium.spaces.box"].Box = _BoxDummy
    return previous


def _restore_pickle_shims(previous: dict[str, types.ModuleType | None]) -> None:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@dataclass(frozen=True)
class ObsNormalizer:
    """SB3 VecNormalize observation transform."""

    mean: np.ndarray
    var: np.ndarray
    epsilon: float = 1e-8
    clip_obs: float = 10.0

    @classmethod
    def from_vecnormalize_pickle(cls, path: str | Path) -> "ObsNormalizer":
        previous = _install_pickle_shims()
        try:
            obj = pickle.loads(Path(path).read_bytes())
        finally:
            _restore_pickle_shims(previous)
        obs_rms = obj.obs_rms
        return cls(
            mean=np.asarray(obs_rms.mean, dtype=np.float32),
            var=np.asarray(obs_rms.var, dtype=np.float32),
            epsilon=float(getattr(obj, "epsilon", 1e-8)),
            clip_obs=float(getattr(obj, "clip_obs", 10.0)),
        )

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float32) - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(x, -self.clip_obs, self.clip_obs).astype(np.float32)


@dataclass(frozen=True)
class PPOPolicy:
    """Deterministic PPO policy head reconstructed from an SB3 checkpoint."""

    w0: np.ndarray
    b0: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    wa: np.ndarray
    ba: np.ndarray
    action_limit: float = 4.0
    obs_normalizer: ObsNormalizer | None = None

    @classmethod
    def from_zip(cls, path: str | Path, vecnorm_path: str | Path | None = None) -> "PPOPolicy":
        sd = _load_policy_state_dict(path)
        normalizer = None
        if vecnorm_path is not None and Path(vecnorm_path).exists():
            normalizer = ObsNormalizer.from_vecnormalize_pickle(vecnorm_path)
        return cls(
            w0=sd["mlp_extractor.policy_net.0.weight"],
            b0=sd["mlp_extractor.policy_net.0.bias"],
            w1=sd["mlp_extractor.policy_net.2.weight"],
            b1=sd["mlp_extractor.policy_net.2.bias"],
            wa=sd["action_net.weight"],
            ba=sd["action_net.bias"],
            obs_normalizer=normalizer,
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if self.obs_normalizer is not None:
            x = self.obs_normalizer.normalize(x)
        x = np.tanh(_linear(x, self.w0, self.b0))
        x = np.tanh(_linear(x, self.w1, self.b1))
        action = _linear(x, self.wa, self.ba)
        return np.clip(action, -self.action_limit, self.action_limit).astype(np.float32)


@dataclass(frozen=True)
class DQNPolicy:
    """Deterministic DQN policy head reconstructed from an SB3 checkpoint."""

    w0: np.ndarray
    b0: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    wq: np.ndarray
    bq: np.ndarray
    action_table: np.ndarray

    @classmethod
    def from_zip(cls, path: str | Path) -> "DQNPolicy":
        sd = _load_policy_state_dict(path)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        mags = np.array([2.0, 4.0], dtype=np.float32)
        action_table = np.stack(
            [np.stack([np.cos(angles) * m, np.sin(angles) * m], axis=-1) for m in mags]
        ).reshape(-1, 2).astype(np.float32)
        return cls(
            w0=sd["q_net.q_net.0.weight"],
            b0=sd["q_net.q_net.0.bias"],
            w1=sd["q_net.q_net.2.weight"],
            b1=sd["q_net.q_net.2.bias"],
            wq=sd["q_net.q_net.4.weight"],
            bq=sd["q_net.q_net.4.bias"],
            action_table=action_table,
        )

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        x = np.maximum(_linear(x, self.w0, self.b0), 0.0)
        x = np.maximum(_linear(x, self.w1, self.b1), 0.0)
        return _linear(x, self.wq, self.bq)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        idx = int(np.argmax(self.q_values(obs)))
        return self.action_table[idx].copy()


@dataclass(frozen=True)
class PolicyBundle:
    """Baseline policies used in the competitive rollout."""

    ppo_fish0: PPOPolicy
    dqn_fish1: DQNPolicy

    @classmethod
    def from_weights_dir(cls, weights_dir: str | Path) -> "PolicyBundle":
        weights = Path(weights_dir)
        return cls(
            ppo_fish0=PPOPolicy.from_zip(weights / "ppo_fish0.zip", weights / "ppo_fish0_vecnorm.pkl"),
            dqn_fish1=DQNPolicy.from_zip(weights / "dqn_fish1.zip"),
        )


__all__ = ["DQNPolicy", "ObsNormalizer", "PPOPolicy", "PolicyBundle"]

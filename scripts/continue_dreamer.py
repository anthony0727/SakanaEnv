"""Continue DreamerV3 training from a sheeprl checkpoint.

PyTorch 2.6 changed `torch.load`'s `weights_only` default to True, which
rejects sheeprl's pickled replay-buffer objects on checkpoint load.
Fresh training never exercises the load path, so the issue only surfaces
on checkpoint load. This wrapper monkey-patches `torch.load` to default
`weights_only=False` for the trusted in-house checkpoint, then delegates
to sheeprl's Hydra-decorated CLI entry point unchanged.

Usage mirrors `sheeprl` exactly — all CLI args are passed through to Hydra:

    python scripts/continue_dreamer.py --config-dir=configs \\
        exp=dreamer_v3_sakana \\
        algo.total_steps=200000 \\
        checkpoint.<ckpt-arg>=path/to/ckpt.ckpt
"""

import os

os.environ.setdefault("GYM_DISABLE_PLUGIN_AUTOLOAD", "1")

import torch
import sakana_env.gym_wrapper as _sakana_gym_wrapper

_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

_sakana_gym_wrapper.register_envs()

from sheeprl.cli import run  # noqa: E402

if __name__ == "__main__":
    run()

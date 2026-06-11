"""
Train DreamerV3 on SakanaEnv using SheepRL.

Usage:
    uv run sheeprl exp=dreamer_v3_sakana

Or with overrides:
    uv run sheeprl exp=dreamer_v3_sakana total_steps=500000 algo.dense_units=512
"""

# This file just ensures the env is registered before sheeprl runs.
# The actual training is done via the sheeprl CLI with Hydra configs.

import sakana_env.gym_wrapper  # registers SakanaEnv-v0

if __name__ == "__main__":
    import sys
    from sheeprl.cli import run
    sys.exit(run())

"""
Live reward curve plotter. Run repeatedly to update.
  uv run python scripts/plot_rewards.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def parse_log(logfile):
    steps, rewards = [], []
    current_step = 0
    try:
        for line in open(logfile):
            if "total_timesteps" in line:
                try: current_step = int(line.split("|")[2].strip())
                except: pass
            if "ep_rew_mean" in line:
                try:
                    rewards.append(float(line.split("|")[2].strip()))
                    steps.append(current_step)
                except: pass
    except: pass
    return steps, rewards

ppo_s, ppo_r = parse_log("/tmp/ppo_train4.log")
dqn_s, dqn_r = parse_log("/tmp/dqn_train4.log")

fig, ax = plt.subplots(figsize=(10, 5))
if ppo_r:
    ax.plot(ppo_s if ppo_s else range(len(ppo_r)), ppo_r, 'b-o', ms=3, lw=2,
            label=f'PPO ({len(ppo_r)} pts, latest={ppo_r[-1]:.2f})')
if dqn_r:
    ax.plot(dqn_s if dqn_s else range(len(dqn_r)), dqn_r, 'r-o', ms=3, lw=2,
            label=f'DQN ({len(dqn_r)} pts, latest={dqn_r[-1]:.2f})')
ax.axhline(y=0, color='gray', ls='--', alpha=0.5, label='break-even')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Episode Reward (mean)')
ax.set_title('SakanaEnv Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = Path(__file__).resolve().parents[1] / "assets" / "reward_curve.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"PPO: {len(ppo_r)} pts" + (f", latest={ppo_r[-1]:.2f}" if ppo_r else ""))
print(f"DQN: {len(dqn_r)} pts" + (f", latest={dqn_r[-1]:.2f}" if dqn_r else ""))
print(f"→ {out}")

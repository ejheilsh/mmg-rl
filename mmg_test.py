from stable_baselines3 import PPO
from mmg_env import MMGEnv
import matplotlib.pyplot as plt
import numpy as np
from general import *

# Environment configuration matching training setup
EVAL_KWARGS = {
    "u0": 0,
    "n_rays": N_RAYS,
    "max_steps": 20000,
}

env = MMGEnv(**EVAL_KWARGS)

try:
    model = PPO.load("mmg_best/best_model", env=env)
    print("using best model...")
except FileNotFoundError:
    model = PPO.load("mmg_ppo", env=env)

num_rollouts = 1
rollouts = []

for _ in range(num_rollouts):
    obs, _ = env.reset()

    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    rollouts.append({
        "traj": env.sim.df_history.copy(),
        "obstacles": env.sim.obstacles.copy(),
    })

fig, ax = plt.subplots(figsize=(8, 6))
fig_controls, (ax_rps, ax_delta) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

for data in rollouts:
    traj = data["traj"]
    ax.plot(traj["x"], traj["y"], linewidth=2, color="black")
    ax.scatter(traj["x"].iloc[0], traj["y"].iloc[0], color="green", s=20, zorder=5)
    ax.scatter(traj["x"].iloc[-1], traj["y"].iloc[-1], color="red", s=20, zorder=5)

    for obs in data["obstacles"]:
        circle = plt.Circle((obs["x"], obs["y"]), obs["r"], color="gray", alpha=0.3)
        ax.add_patch(circle)

    ax_rps.plot(traj["t"], traj["nP"], color="black", alpha=0.7)
    ax_delta.plot(traj["t"], np.degrees(traj["delta"]), color="black", alpha=0.7)

ax.set_xlim(env.sim.xmin, env.sim.xmax)
ax.set_ylim(env.sim.ymin, env.sim.ymax)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x-position")
ax.set_ylabel("y-position")
ax.set_title("MMG Ship Trajectory")

ax_rps.set_ylabel("Prop RPM (nP)")
ax_delta.set_ylabel("Rudder (deg)")
ax_delta.set_xlabel("Time (s)")
ax_rps.set_title("Propeller RPM History")
ax_delta.set_title("Rudder Angle History")

plt.show()

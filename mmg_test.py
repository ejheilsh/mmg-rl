from stable_baselines3 import PPO
from mmg_env import MMGEnv
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Load Environment + Model
# ---------------------------

env = MMGEnv()

# Load trained model
model = PPO.load("mmg_ppo", env=env)

# ---------------------------
# Rollouts with unique colors and markers
# ---------------------------

num_rollouts = 1
colors = plt.cm.viridis(np.linspace(0, 1, num_rollouts))
markers = ["o", "s", "^", "D", "P"]   # circle, square, triangle, diamond, plus

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(num_rollouts):

    obs, info = env.reset()
    color = colors[i]
    marker = markers[i % len(markers)]

    # roll one episode
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    traj = env.sim.df_history

    # ---- trajectory line ----
    ax.plot(traj["x"], traj["y"], color=color, linewidth=2, label=f"Rollout {i+1}")

    # ---- unique end marker ----
    ax.scatter(
        traj["x"].iloc[-1],
        traj["y"].iloc[-1],
        color=color,
        marker=marker,
        s=150,
        edgecolor="black",
        linewidth=1.2,
        zorder=10
    )

    # ---- obstacles matching the color ----
    for obs_i in env.sim.obstacles:
        circle = plt.Circle(
            (obs_i["x"], obs_i["y"]),
            obs_i["r"],
            color=color,
            alpha=0.25
        )
        ax.add_patch(circle)

ax.set_aspect("equal")
ax.legend()
plt.show()
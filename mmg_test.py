from stable_baselines3 import PPO
from mmg_env import MMGEnv
import matplotlib.pyplot as plt
import numpy as np
from general import *

# ---------------------------
# Helpers
# ---------------------------

def compute_ray_segments(env):
    """
    Returns a list of ((x0, y0), (x1, y1)) for each radar ray,
    clipped to the simulation bounds (simple clamp).
    """
    x0, y0, psi = env.sim.x[0], env.sim.x[1], env.sim.x[2]
    distances = env.get_radar_distances()
    segments = []

    xmin, xmax = env.sim.xmin, env.sim.xmax
    ymin, ymax = env.sim.ymin, env.sim.ymax

    for rel_ang, dist in zip(env.ray_angles, distances):
        ang = psi + rel_ang
        x1 = x0 + dist * np.cos(ang)
        y1 = y0 + dist * np.sin(ang)

        # simple clamp to domain (keeps +x wall visible while allowing exit)
        x1 = np.clip(x1, xmin, xmax)
        y1 = np.clip(y1, ymin, ymax)

        segments.append(((x0, y0), (x1, y1)))
    return segments


def compute_ray_segments_for_pose(env, pose):
    """
    Compute ray segments for an arbitrary pose (x, y, psi) using the same
    logic as env.get_radar_distances, without permanently mutating env state.
    """
    x0, y0, psi = pose

    # Temporarily set pose to reuse env.get_radar_distances logic (bounds + obstacles)
    original_state = env.sim.x.copy()
    env.sim.x[0] = x0
    env.sim.x[1] = y0
    env.sim.x[2] = psi
    distances = env.get_radar_distances()
    env.sim.x = original_state  # restore

    segments = []
    xmin, xmax = env.sim.xmin, env.sim.xmax
    ymin, ymax = env.sim.ymin, env.sim.ymax

    for rel_ang, dist in zip(env.ray_angles, distances):
        ang = psi + rel_ang
        dx = np.cos(ang)
        dy = np.sin(ang)
        x1 = x0 + dist * dx
        y1 = y0 + dist * dy
        x1 = np.clip(x1, xmin, xmax)
        y1 = np.clip(y1, ymin, ymax)
        segments.append(((x0, y0), (x1, y1)))

    return segments


def compute_ray_guides(env, pose):
    """
    Return full-length guide rays (max_range) for a pose to visualize the cone.
    """
    x0, y0, psi = pose
    guides = []
    xmin, xmax = env.sim.xmin, env.sim.xmax
    ymin, ymax = env.sim.ymin, env.sim.ymax

    for rel_ang in env.ray_angles:
        ang = psi + rel_ang
        x1 = x0 + env.max_range * np.cos(ang)
        y1 = y0 + env.max_range * np.sin(ang)
        x1 = np.clip(x1, xmin, xmax)
        y1 = np.clip(y1, ymin, ymax)
        guides.append(((x0, y0), (x1, y1)))
    return guides

# Match env settings to those used in training
EVAL_KWARGS = {
    "u0": 0,
    # must match training obs dimension
    "n_rays": N_RAYS,
    "max_steps": 20000,
}

# ---------------------------
# Load Environment + Model
# ---------------------------

env = MMGEnv(**EVAL_KWARGS)

# Load trained model (prefer best checkpoint if available)
try:
    model = PPO.load("mmg_best/best_model", env=env)
except FileNotFoundError:
    model = PPO.load("mmg_ppo", env=env)

# ---------------------------
# Rollouts with unique colors and markers
# ---------------------------

num_rollouts = 1
colors = plt.cm.viridis(np.linspace(0, 1, num_rollouts))
markers = ["o", "s", "^", "D", "P"]   # circle, square, triangle, diamond, plus

# collect rollout data for ensemble plots
rollouts = []

for i in range(num_rollouts):

    obs, info = env.reset()
    color = colors[i]
    marker = markers[i % len(markers)]

    # roll one episode
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    rollouts.append({
        "traj": env.sim.df_history.copy(),
        "color": color,
        "marker": marker,
        "obstacles": env.sim.obstacles.copy(),
    })

# Plot ensemble after collecting all rollouts
fig, ax_traj = plt.subplots(figsize=(8, 6))
fig2, (ax_rps, ax_delta) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
fig3, (ax_u, ax_yaw) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

for i, data in enumerate(rollouts, start=1):
    traj = data["traj"]
    color = data["color"]
    marker = data["marker"]

    # random pose for ray visualization
    rand_idx = np.random.randint(len(traj))
    pose = (traj["x"].iloc[rand_idx], traj["y"].iloc[rand_idx], traj["psi"].iloc[rand_idx])
    ray_segments = compute_ray_segments_for_pose(env, pose)
    for (x0, y0), (x1, y1) in ray_segments:
        ax_traj.plot([x0, x1], [y0, y1], color=color, alpha=0.5, linewidth=1)

    ax_traj.plot(traj["x"], traj["y"], color=color, linewidth=2, label=f"Rollout {i}")
    ax_traj.scatter(traj["x"].iloc[-1], traj["y"].iloc[-1], color=color,
                    marker=marker, s=150, edgecolor="black", linewidth=1.2, zorder=10)

    for obs_i in data["obstacles"]:
        circle = plt.Circle((obs_i["x"], obs_i["y"]), obs_i["r"],
                            color=color, alpha=0.15)
        ax_traj.add_patch(circle)

    ax_rps.plot(traj["t"], traj["nP"], color=color, label=f"Rollout {i}")
    ax_delta.plot(traj["t"], np.degrees(traj["delta"]), color=color, label=f"Rollout {i}")
    ax_u.plot(traj["t"], traj["u"], color=color, label=f"Rollout {i}")
    ax_yaw.plot(traj["t"], np.degrees(traj["psi"]), color=color, label=f"Rollout {i}")

ax_traj.set_aspect("equal")
ax_traj.legend()
ax_traj.set_title("Trajectory and Radar Rays")

ax_rps.set_ylabel("Prop RPM (nP)")
ax_delta.set_ylabel("Rudder (deg)")
ax_delta.set_xlabel("Time (s)")
ax_rps.legend()
ax_delta.legend()

ax_u.set_ylabel("Surge u (m/s)")
ax_yaw.set_ylabel("Heading psi (deg)")
ax_yaw.set_xlabel("Time (s)")
ax_u.legend()
ax_yaw.legend()

plt.show()

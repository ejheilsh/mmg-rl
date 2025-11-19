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
    clipped to the simulation bounds.
    """
    def clip_segment_to_bounds(p0, p1, bounds):
        """
        Liang-Barsky clipping for axis-aligned rectangle.
        bounds = (xmin, xmax, ymin, ymax)
        """
        (x0, y0), (x1, y1) = p0, p1
        xmin, xmax, ymin, ymax = bounds
        dx = x1 - x0
        dy = y1 - y0
        p = [-dx, dx, -dy, dy]
        q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
        u0, u1 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if pi == 0:
                if qi < 0:
                    return None  # parallel and outside
                continue
            t = -qi / pi
            if pi < 0:
                if t > u1:
                    return None
                if t > u0:
                    u0 = t
            else:
                if t < u0:
                    return None
                if t < u1:
                    u1 = t
        nx0, ny0 = x0 + u0 * dx, y0 + u0 * dy
        nx1, ny1 = x0 + u1 * dx, y0 + u1 * dy
        return (nx0, ny0), (nx1, ny1)

    x0, y0, psi = env.sim.x[0], env.sim.x[1], env.sim.x[2]
    distances = env.get_radar_distances()
    segments = []

    xmin, xmax = env.sim.xmin, env.sim.xmax
    ymin, ymax = env.sim.ymin, env.sim.ymax
    bounds = (xmin, xmax, ymin, ymax)

    for rel_ang, dist in zip(env.ray_angles, distances):
        ang = psi + rel_ang
        x1 = x0 + dist * np.cos(ang)
        y1 = y0 + dist * np.sin(ang)

        clipped = clip_segment_to_bounds((x0, y0), (x1, y1), bounds)
        if clipped is not None:
            segments.append(clipped)
    return segments

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

# Load trained model
model = PPO.load("mmg_ppo", env=env)

# ---------------------------
# Rollouts with unique colors and markers
# ---------------------------

num_rollouts = 3
colors = plt.cm.viridis(np.linspace(0, 1, num_rollouts))
markers = ["o", "s", "^", "D", "P"]   # circle, square, triangle, diamond, plus

fig, ax_traj = plt.subplots(figsize=(8, 6))
fig2, (ax_rps, ax_delta) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
fig3, (ax_u, ax_yaw) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

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

    traj = env.sim.df_history

    # plot radar rays at final state of rollout (clipped to bounds)
    ray_segments = compute_ray_segments(env)
    for (x0, y0), (x1, y1) in ray_segments:
        ax_traj.plot([x0, x1], [y0, y1], color=color, alpha=0.4, linewidth=1)

    # ---- trajectory line ----
    ax_traj.plot(traj["x"], traj["y"], color=color, linewidth=2, label=f"Rollout {i+1}")

    # ---- unique end marker ----
    ax_traj.scatter(
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
        ax_traj.add_patch(circle)

    # ---- time histories for prop RPM (nP) and rudder angle (delta) ----
    ax_rps.plot(traj["t"], traj["nP"], color=color, label=f"Rollout {i+1}")
    ax_delta.plot(traj["t"], np.degrees(traj["delta"]), color=color, label=f"Rollout {i+1}")
    ax_u.plot(traj["t"], traj["u"], color=color, label=f"Rollout {i+1}")
    ax_yaw.plot(traj["t"], np.degrees(traj["psi"]), color=color, label=f"Rollout {i+1}")

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

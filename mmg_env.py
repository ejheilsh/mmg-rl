import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mmg_class import mmg_class
from general import *


class MMGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        u0=0,
        u0_range=None,
        rand_seed=None,
        dt=0.1,
        n_rays=N_RAYS,
        max_steps=20000,
    ):
        super().__init__()

        self.dt = dt
        self.u0 = u0
        self.u0_range = u0_range
        self.max_range = 300
        self.nP_min = 0.0
        self.nP_max = 20.0
        self.delta_max = np.radians(35.0)
        self.rudder_rate_limit_rad = np.deg2rad(1.76)  # per second
        self.n_rays = int(n_rays)
        # Use the configured angles but allow overriding n_rays by interpolating between endpoints
        if len(RAY_ANGLES) == self.n_rays:
            self.ray_angles = np.array(RAY_ANGLES, dtype=float)
        else:
            self.ray_angles = np.linspace(RAY_ANGLES[0], RAY_ANGLES[-1], self.n_rays)
        self.max_steps = int(max_steps)
        self.step_count = 0

        # underlying MMG simulator
        self.sim = mmg_class(u0=u0, rand_seed=rand_seed)

        # -------- ACTION SPACE --------
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # -------- OBSERVATION SPACE --------
        low = np.array([
            self.sim.xmin, self.sim.ymin, -np.pi, -5, -5, -1,    # ship state
            *([0.0] * self.n_rays)                              # radar distances
        ], dtype=np.float32)

        high = np.array([
            self.sim.xmax, self.sim.ymax, np.pi, 5, 5, 1,
            *([self.max_range] * self.n_rays)
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # --------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # optionally randomize initial surge velocity
        u0_sample = self._sample_u0()
        self.sim.reset(u0=u0_sample)
        self.step_count = 0
        obs = self.get_obs()
        info = {}

        return obs.astype(np.float32), info

    # --------------------------------------------------------------

    def step(self, action):
        # action = [nP_norm, delta_norm]
        nP, delta = self.action_to_controls(action)
        # apply rudder rate limit (deg/s converted to per-step bound)
        prev_delta = self.sim.u[1]
        delta = np.clip(
            delta,
            prev_delta - self.rudder_rate_limit_rad * self.dt,
            prev_delta + self.rudder_rate_limit_rad * self.dt,
        )
        self.sim.u = np.array([nP, delta], dtype=float)

        x_old = self.sim.x[0]   # x before step
        self.step_count += 1

        self.sim.timestep(self.dt)

        x_new, y_new, psi, u, vm, r = self.sim.x
        terminated = self.sim.check_collision()
        truncated = self.step_count >= self.max_steps   # hard time limit

        # -------- reward --------
        reward = 0

        # forward progress (encourage +x movement with gradual ramp toward goal)
        goal_x = self.sim.xmax
        progress = x_new - x_old
        progress_weight = np.clip(0.3 + 0.7 * (x_new / goal_x), 0.3, 1.0)
        reward += progress_weight * progress

        # penalize heading error from straight-ahead
        reward -= 0.1 * abs(psi)
        # penalize lateral deviation from centerline
        reward -= 0.1 * abs(y_new)


        # collision / OOB punish (strong discouragement)
        if terminated:
            reward -= 500000

        # penalize proximity to obstacles via radar distances (closer → larger penalty)
        radar = self.get_radar_distances()
        clearance_penalty = np.sum((self.max_range - radar) / self.max_range)
        reward -= 0.1 * clearance_penalty

        # explicit out-of-bounds penalty (before finish line)
        out_of_bounds = (
            x_new < self.sim.xmin or
            y_new < self.sim.ymin or y_new > self.sim.ymax
        )
        if out_of_bounds:
            reward -= 5000
            terminated = True  # force terminate even if collision handler changes

        # success bonus
        if x_new >= goal_x:
            reward += 20000
            print("\n\nsuccess\n\n")
            terminated = True

        obs = self.get_obs().astype(np.float32)

        return obs, reward, terminated, truncated, {}

    # --------------------------------------------------------------

    def _advance_sim(self, action):
        """Advance MMG simulation by 1 step and compute reward."""

        x_old = self.sim.x[0]

        # apply action (expects already-normalized inputs)
        nP, delta = self.action_to_controls(action)
        prev_delta = self.sim.u[1]
        delta = np.clip(
            delta,
            prev_delta - self.rudder_rate_limit_rad * self.dt,
            prev_delta + self.rudder_rate_limit_rad * self.dt,
        )
        self.sim.u = np.array([nP, delta], dtype=float)

        # check collision BEFORE step? Usually we check after
        self.sim.timestep(self.dt)

        obs = self.get_obs().astype(np.float32)
        x_new = obs[0]

        collided = self.sim.check_collision()
        terminated = bool(collided)

        # reward = +Δx progress
        dx = x_new - x_old
        time_penalty = -0.001
        collision_penalty = -5000.0 if terminated else 0.0

        reward = dx + time_penalty + collision_penalty

        return obs, reward, terminated

    def action_to_controls(self, action):
        """
        Convert normalized agent action to physical controls.
        action = [a0, a1] where each element is in [-1, 1]
        """

        # Unpack
        a0, a1 = action

        # Map actions [-1,1] → physical ranges
        nP = self.nP_min + (a0 + 1) * 0.5 * (self.nP_max - self.nP_min)
        delta = a1 * self.delta_max

        return nP, delta

    def _sample_u0(self):
        """
        Sample an initial surge speed if a range is provided; otherwise return fixed u0.
        """
        if self.u0_range is None:
            return self.u0

        if isinstance(self.u0_range, (list, tuple)) and len(self.u0_range) == 2:
            return float(np.random.uniform(self.u0_range[0], self.u0_range[1]))

        # fallback: treat as fixed
        return self.u0

    def get_obs(self):
        x_pos, y_pos, psi, u, vm, r = self.sim.x

        radar = self.get_radar_distances()  # shape (n_rays,)

        return np.concatenate([
            np.array([x_pos, y_pos, psi, u, vm, r], dtype=float),
            radar
        ], dtype=np.float32)

    # --------------------------------------------------------------

    def render(self):
        # Optional: call your plot function or live-draw trajectory
        pass

    # --------------------------------------------------------------

    def close(self):
        pass


    def get_radar_distances(self):
        x, y, psi = self.sim.x[0], self.sim.x[1], self.sim.x[2]
        xmin, xmax, ymin, ymax = self.sim.xmin, self.sim.xmax, self.sim.ymin, self.sim.ymax

        distances = []
        for rel_ang in self.ray_angles:
            ang = psi + rel_ang
            dmin = self.max_range
            dx = np.cos(ang)
            dy = np.sin(ang)

            for obs in self.sim.obstacles:
                d = ray_circle_distance(
                    x, y, ang,
                    obs["x"], obs["y"], obs["r"],
                    max_range=self.max_range
                )
                dmin = min(dmin, d)

        # check intersection with domain bounds (exclude +x wall so agent can exit)
            ts = []
            if dx < 0:  # left boundary
                t = (xmin - x) / dx
                y_int = y + t * dy
                if t > 0 and ymin <= y_int <= ymax:
                    ts.append(t)
            if dy > 0:  # top boundary
                t = (ymax - y) / dy
                x_int = x + t * dx
                if t > 0 and xmin <= x_int <= xmax:
                    ts.append(t)
            if dy < 0:  # bottom boundary
                t = (ymin - y) / dy
                x_int = x + t * dx
                if t > 0 and xmin <= x_int <= xmax:
                    ts.append(t)

            if ts:
                d_bounds = min(ts)
                dmin = min(dmin, d_bounds, self.max_range)

            distances.append(dmin)

        return np.array(distances, dtype=float)

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
        rand_seed=None,
        dt=0.1,
        n_rays=N_RAYS,
        max_steps=5000,
    ):
        super().__init__()

        self.dt = dt
        self.u0 = u0
        self.max_range = 300
        self.nP_min = 0.0
        self.nP_max = 20.0
        self.delta_max = np.radians(35.0)
        self.nP_rate_limit = 5.0  # RPM change per second
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
        # self.action_space = spaces.Box(
        #     low=np.array([-1.0, -1.0], dtype=np.float32),
        #     high=np.array([1.0, 1.0], dtype=np.float32),
        # )
        self.action_space = spaces.Box(
            low=np.array([self.nP_min], dtype=np.float32),
            high=np.array([self.nP_max], dtype=np.float32),
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
        self.sim.reset()
        self.step_count = 0
        obs = self.get_obs()
        info = {}

        return obs.astype(np.float32), info

    # --------------------------------------------------------------

    def step(self, action):
        nP, delta = self.action_to_controls(action)
        self.sim.u = np.array([nP, delta], dtype=float)

        x_old = self.sim.x[0]
        self.step_count += 1

        self.sim.timestep(self.dt)

        x_new, y_new, psi, u, vm, r = self.sim.x
        terminated = self.sim.check_collision()
        truncated = self.step_count >= self.max_steps   # hard time limit

        # -------- reward --------
        reward = 0

        # forward progress (encourage +x movement with gradual ramp toward goal)
        goal_x = self.sim.xmax
        # progress = x_new - x_old
        # progress_weight = np.clip(0.3 + 0.7 * (x_new / goal_x), 0.3, 1.0)
        # reward += progress_weight * progress

        reward += (x_new - x_old) * 100

        # discourage sitting still
        reward -= 0.01

        # reward -= np.abs(psi) * self.dt

        # penalize heading error from straight-ahead
        # reward -= 0.1 * abs(psi)
        # penalize lateral deviation from centerline
        # reward -= 0.1 * abs(y_new)


        # collision / OOB punish (strong discouragement)
        # if terminated:
            # reward -= 5

        # penalize proximity to obstacles via radar distances (closer → larger penalty)
        # radar = self.get_radar_distances()
        # clearance_penalty = np.sum((self.max_range - radar) / self.max_range)
        # reward -= 0.1 * clearance_penalty

        # explicit out-of-bounds penalty (before finish line)
        out_of_bounds = (
            x_new < self.sim.xmin or
            y_new < self.sim.ymin or y_new > self.sim.ymax
        )
        if out_of_bounds:
            reward -= 5
            terminated = True  # force terminate even if collision handler changes

        # success bonus
        if x_new >= goal_x:
            reward += 200
            print("\n\nsuccess\n\n")
            terminated = True

        obs = self.get_obs().astype(np.float32)

        return obs, reward, terminated, truncated, {}

    def action_to_controls(self, action):
        """
        Convert agent action to physical controls.

        We currently expose only the propeller RPM as an action (already in
        physical units). The rudder is clamped at 0 for now, but the old
        normalized [-1, 1] interface is left in comments above for future use.
        """

        a = np.asarray(action, dtype=float)
        if a.ndim == 0:
            nP_cmd = float(a)
        else:
            nP_cmd = float(a[0])

        nP = float(np.clip(nP_cmd, self.nP_min, self.nP_max))
        prev_nP = self.sim.u[0]
        max_dnP = self.nP_rate_limit * self.dt
        nP = float(np.clip(nP, prev_nP - max_dnP, prev_nP + max_dnP))

        delta = 0.0
        return nP, delta

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

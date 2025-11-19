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
        n_obstacles=0,
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
        self.sim = mmg_class(u0=u0, rand_seed=rand_seed, n_obstacles=n_obstacles)

        # -------- ACTION SPACE --------
        # self.action_space = spaces.Box(
        #     low=np.array([-1.0, -1.0], dtype=np.float32),
        #     high=np.array([1.0, 1.0], dtype=np.float32),
        # )
        # -------- ACTION SPACE --------
        # Normalize action space to [-1, 1] for better learning
        # Action 0: Propeller RPS
        # Action 1: Rudder Angle
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
        y_old = self.sim.x[1]
        self.step_count += 1

        self.sim.timestep(self.dt)

        x_new, y_new, psi, u, vm, r = self.sim.x
        terminated = self.sim.check_collision()
        truncated = self.step_count >= self.max_steps   # hard time limit

        # -------- reward --------
        reward = 0

        # forward progress (encourage +x movement only)
        # scale factor to make reward meaningful
        reward += (x_new - x_old) * 100

        # Heading shaping: Encourage facing 0 degrees (straight ahead)
        # target_psi = 0
        # Normalize psi to [-pi, pi] for calculation
        # psi_norm = (psi + np.pi) % (2 * np.pi) - np.pi
        # heading_error = np.abs(psi_norm - target_psi)
        # Penalize heading error
        # reward -= heading_error * 0.1

        # discourage sitting still or moving backwards
        # small penalty per step to encourage speed
        reward -= 0.15

        # explicit out-of-bounds penalty (before finish line)
        out_of_bounds = (
            x_new < self.sim.xmin or
            y_new < self.sim.ymin or y_new > self.sim.ymax
        )
        if out_of_bounds:
            reward -= 10
            terminated = True  # force terminate

        # collision / OOB punish (strong discouragement)
        if terminated:
            reward -= 100

        # success bonus
        goal_x = self.sim.xmax
        if x_new >= goal_x:
            reward += 150
            terminated = True

        obs = self.get_obs().astype(np.float32)

        return obs, reward, terminated, truncated, {}

    def action_to_controls(self, action):
        """
        Convert agent action to physical controls.
        Action is in [-1, 1].
        """

        a = np.asarray(action, dtype=float)
        # Handle both 1D and 2D actions for backward compatibility/safety
        if a.size == 1:
            act_nP = float(a)
            act_delta = 0.0
        else:
            act_nP = float(a[0])
            act_delta = float(a[1])
            
        # Rescale [-1, 1] -> [nP_min, nP_max]
        nP_cmd = self.nP_min + (act_nP + 1.0) * 0.5 * (self.nP_max - self.nP_min)
        nP = float(np.clip(nP_cmd, self.nP_min, self.nP_max))

        # Rescale [-1, 1] -> [-delta_max, delta_max]
        delta_cmd = act_delta * self.delta_max
        delta = float(np.clip(delta_cmd, -self.delta_max, self.delta_max))
        
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

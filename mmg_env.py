import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mmg_class import mmg_class
from general import *


class MMGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, u0=0, rand_seed=None, dt=0.1):
        super().__init__()

        self.dt = dt
        self.max_range = 15
        self.nP_min = 0.0
        self.nP_max = 20.0
        self.delta_max = np.radians(35.0)

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
            *([0.0] * N_RAYS)                                   # radar distances
        ], dtype=np.float32)

        high = np.array([
            self.sim.xmax, self.sim.ymax, np.pi, 5, 5, 1,
            *([self.max_range] * N_RAYS)
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # --------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim.reset()
        obs = self.get_obs()
        info = {}

        return obs.astype(np.float32), info

    # --------------------------------------------------------------

    def step(self, action):
        # action = [nP_norm, delta_norm]
        nP, delta = self.action_to_controls(action)
        self.sim.u = np.array([nP, delta], dtype=float)

        x_old = self.sim.x[0]   # x before step

        self.sim.timestep(self.dt)

        x_new, y_new, psi, u, vm, r = self.sim.x
        terminated = self.sim.check_collision()
        truncated = False       # SB3 expects this too

        # -------- reward --------
        reward = 0

        # forward progress
        reward += 0.1 * (x_new - x_old)

        # penalize sway and rotation
        reward -= 0.05 * abs(vm)
        reward -= 0.01 * abs(r)

        # smooth control penalty (delta = action[1])
        reward -= 0.001 * delta**2

        # collision / OOB punish
        if terminated:
            reward -= 100

        # success bonus
        if x_new >= 48:
            reward += 200
            terminated = True

        obs = self.get_obs().astype(np.float32)

        return obs, reward, terminated, truncated, {}

    # --------------------------------------------------------------

    def _advance_sim(self, action):
        """Advance MMG simulation by 1 step and compute reward."""

        x_old = self.sim.x[0]

        # apply action (expects already-normalized inputs)
        nP, delta = self.action_to_controls(action)
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
        collision_penalty = -100.0 if terminated else 0.0

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

    def get_obs(self):
        x_pos, y_pos, psi, u, vm, r = self.sim.x

        radar = self.get_radar_distances()  # shape (9,)

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

        distances = []
        for rel_ang in RAY_ANGLES:
            ang = psi + rel_ang
            dmin = self.max_range

            for obs in self.sim.obstacles:
                d = ray_circle_distance(
                    x, y, ang,
                    obs["x"], obs["y"], obs["r"],
                    max_range=self.max_range
                )
                dmin = min(dmin, d)

            distances.append(dmin)

        return np.array(distances, dtype=float)


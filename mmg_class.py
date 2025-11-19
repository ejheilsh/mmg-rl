from general import *

@dataclass
class mmg_class:
    u0: float = 0
    rand_seed: int  = None
    
    def __post_init__(self):
        # define the state
        self.x = np.zeros(6)
        self.x[3] = self.u0
        # state: (x, y, psi, x_dot, y_dot, psi_dot (r))

        # control vector
        self.u = np.zeros(2)
        # controls: (nP, delta)

        # time
        self.t = 0

        # storage for state
        self.df_history = pd.DataFrame(columns=[
            "t", "x", "y", "psi", "u", "vm", "r",
            "nP", "delta"
        ])

        # simulation bounds
        self.xmax = 50
        self.xmin = 0
        self.ymin = -10
        self.ymax = 10

        self.init_obstacles()

    def init_obstacles(self):
        # use seed if provided
        if self.rand_seed is not None:
            np.random.seed(self.rand_seed)


        # how many obstacles?
        n_obs = 4

        self.obstacles = []

        for _ in range(n_obs):
            # random center within domain, min value is adjusted so that it doesn't overlap with the ship
            x = np.random.uniform(self.xmin + 0.1 * (self.xmax - self.xmin), self.xmax)
            y = np.random.uniform(self.ymin, self.ymax)

            # random radius
            r = np.random.uniform(1, 2)   # adjust sizes

            self.obstacles.append({"x": x, "y": y, "r": r})


    def plot_obstacles(self, ax):
        for obs in self.obstacles:
            circle = plt.Circle((obs["x"], obs["y"]), obs["r"],
                                color="gray", alpha=0.3)
            ax.add_patch(circle)


    def forces_hull_nondim(self, vm_pr, r_pr):
        X_H_pr = np.sum([
            -R_0_pr,
            X_vv_pr * vm_pr**2,
            X_vr_pr * vm_pr * r_pr,
            X_rr_pr * r_pr**2,
            X_vvvv_pr * vm_pr**4
        ])

        Y_H_pr = np.sum([
            Y_v_pr * vm_pr,
            Y_R_pr * r_pr,
            Y_vvv_pr * vm_pr**3,
            Y_vvr_pr * vm_pr**2 * r_pr,
            Y_vrr_pr * vm_pr * r_pr**2,
            Y_rrr_pr * r_pr**3
        ])

        N_H_pr = np.sum([
            N_v_pr * vm_pr,
            N_R_pr * r_pr,
            N_vvv_pr * vm_pr**3,
            N_vvr_pr * vm_pr**2 * r_pr,
            N_vrr_pr * vm_pr * r_pr**2,
            N_rrr_pr * r_pr**3
        ])

        return X_H_pr, Y_H_pr, N_H_pr


    def forces_hull_propeller_rudder(self, u, vm, r, n_P, delta):
        # calculate total velocity and drift angle
        U = np.sqrt(u**2 + vm**2)        
        eps = np.finfo(float).eps
        beta = np.arctan(-vm / (u + eps))
        # beta = np.arctan(-vm / u)

        # nondim sway vel and yaw rate
        vm_pr = vm / (U + eps)
        r_pr = r * Lpp / (U + eps)

        # get the nondim forces
        X_H_pr, Y_H_pr, N_H_pr = self.forces_hull_nondim(vm_pr, r_pr)

        # dimensionalize hull forces
        X_H = (1/2) * rho * Lpp * d * U**2 * X_H_pr
        Y_H = (1/2) * rho * Lpp * d * U**2 * Y_H_pr
        N_H = (1/2) * rho * Lpp * d * U**2 * N_H_pr

        # drift angle at the propeller
        beta_P = beta - x_P_pr * r_pr

        # wale region
        w_P_ratio = 1 + (1 - np.exp(-C1 * np.abs(beta_P))) * (C2(beta_P) - 1)
        w_P = w_P_ratio * (w_P0 - 1) + 1

        # thrust on propeller
        J_P = u * (1 - w_P) / (n_P * D_P + eps)
        K_T = k2 * J_P**2 + k1 * J_P + k0
        T = rho * n_P**2 * D_P**4 * K_T
        X_P = (1 - t_p) * T

        # drift angle at the rudder
        beta_R = beta - l_R_pr * r_pr

        # rudder velocity components
        v_R = U * gamma_R(beta_R) * beta_R
        u_R = epsilon * u * (1 - w_P) * np.sqrt(
            eta * (1 + kappa * (np.sqrt(1 + (8 * K_T)/(np.pi * J_P**2 + eps))-1))**2 + (1 - eta)
        )

        # rudder inflow angle and velocity
        alpha_R = delta - np.arctan(v_R / (u_R + eps))
        U_R = np.sqrt(u_R**2 + v_R**2)

        # rudder normal force
        F_N = (1/2) * rho * A_R * U_R**2 * f_alpha * np.sin(alpha_R)

        # rudder forces
        X_R = (t_R - 1) * F_N * np.sin(delta)
        Y_R = -(1 + a_H) * F_N * np.cos(delta)
        N_R = -(x_R + a_H * x_H) * F_N * np.cos(delta)

        return X_H, Y_H, N_H, X_P, X_R, Y_R, N_R


    def x_dot_func(self, x, t):
        # extract state and controls
        x_pos, y_pos, psi, u, vm, r = x
        n_P, delta = self.u

        # compute forces
        X_H, Y_H, N_H, X_P, X_R, Y_R, N_R = self.forces_hull_propeller_rudder(u, vm, r, n_P, delta)
        X = X_H + X_P + X_R
        Y = Y_H + Y_R
        N_m = N_H + N_R

        # want linear system Aa = b, need to solve for u_dot, vm_dot, and r_dot (eq 4)
        A = np.array([
            [m + m_x, 0, 0],
            [0, m + m_y, x_G * m],
            [0, x_G * m, I_zG + x_G**2 * m + J_z]
        ])

        b = np.array([
            X + (m + m_y) * vm * r + x_G * m * r**2,
            Y - (m + m_x) * u * r,
            N_m - x_G * m * u * r
        ])

        u_dot, vm_dot, r_dot = np.linalg.solve(A, b)

        x_pos_dot = u * np.cos(psi) - vm * np.sin(psi)
        y_pos_dot = u * np.sin(psi) + vm * np.cos(psi)
        psi_dot = r

        return np.array([x_pos_dot, y_pos_dot, psi_dot, u_dot, vm_dot, r_dot])


    def timestep(self, dt):
        # RK4 to integrate in time
        x = self.x
        t = self.t
        dto2 = dt / 2
        dto6 = dt / 6
        f0 = self.x_dot_func(x, t)
        f1 = self.x_dot_func(x + dto2 * f0, t + dto2)
        f2 = self.x_dot_func(x + dto2 * f1, t + dto2)
        f3 = self.x_dot_func(x + dt * f2, t + dt)
        self.x = x + dto6 * (f0 + f1 + f2 + f3)
        self.t += dt

        # unpack
        x_pos, y_pos, psi, u, vm, r = self.x
        nP, delta = self.u

        # append history
        self.df_history.loc[len(self.df_history)] = [
            self.t, x_pos, y_pos, psi, u, vm, r, nP, delta
        ]

    def check_collision(self):
        """
        Return True if the ship's midship position (x,y)
        lies inside any obstacle. Otherwise return False.
        """
        if not hasattr(self, "obstacles"):
            return False  # no obstacles defined

        x_pos, y_pos = self.x[0], self.x[1]

        # allow crossing +x bound; still terminate if below xmin or y out of bounds
        if x_pos < self.xmin or y_pos < self.ymin or y_pos > self.ymax:
            # print("out of bounds")
            return True

        for obs in self.obstacles:
            dx = x_pos - obs["x"]
            dy = y_pos - obs["y"]
            dist = np.hypot(dx, dy)

            if dist <= obs["r"]:
                # print(f"collision!")
                return True  # collision detected

        return False



    def plot_trajectory(self, show_obstacles=True, ax=None):
        """
        Plot the ship's trajectory from df_history.
        """

        import matplotlib.pyplot as plt

        # create axes if not passed in
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        df = self.df_history

        # Plot trajectory
        ax.plot(df["x"], df["y"], linewidth=2, label="trajectory")

        # Start point (green)
        ax.scatter(df["x"].iloc[0], df["y"].iloc[0],
                color="green", s=70, zorder=5, label="start")

        # End point (red)
        ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1],
                color="red", s=70, zorder=5, label="end")

        # Add obstacles if requested
        if show_obstacles and hasattr(self, "obstacles"):
            for obs in self.obstacles:
                circle = plt.Circle(
                    (obs["x"], obs["y"]), obs["r"],
                    color="gray", alpha=0.3
                )
                ax.add_patch(circle)

        # domain box (optional)
        if hasattr(self, "xmin"):
            ax.set_xlim(self.xmin, self.xmax)
            ax.set_ylim(self.ymin, self.ymax)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x-position")
        ax.set_ylabel("y-position")
        ax.set_title("MMG Ship Trajectory")
        ax.legend()

        plt.show()

    def reset(self, u0=None):
        """
        Reset MMG simulator state and return initial observation.
        """
        # reset state
        self.x = np.zeros(6)

        # restore initial surge velocity if provided
        if u0 is None:
            self.x[3] = self.u0
        else:
            self.x[3] = u0

        # reset time
        self.t = 0.0

        # clear history dataframe
        self.df_history = pd.DataFrame(columns=[
            "t", "x", "y", "psi", "u", "vm", "r",
            "nP", "delta"
        ])

        # re-generate obstacles (optional)
        # comment out next line if you want the same obstacles every episode
        self.init_obstacles()

        # default control inputs
        self.u = np.zeros(2)

        # log the initial state
        x_pos, y_pos, psi, u, vm, r = self.x
        nP, delta = self.u
        self.df_history.loc[0] = [self.t, x_pos, y_pos, psi, u, vm, r, nP, delta]

        # ------- return observation for RL --------
        return self.get_obs()

    def get_obs(self):
        x_pos, y_pos, psi, u, vm, r = self.x
        return np.array([x_pos, y_pos, psi, u, vm, r], dtype=float)



if __name__ == "__main__":
    mmg = mmg_class(u0=0)
    mmg.u = np.array([10, np.radians(1)])
    dt = 0.1
    t_final = 200

    for _ in range(int(t_final/dt)):
        if not mmg.check_collision():
            mmg.timestep(dt)
        else:
            break

    print(f"x_final = {mmg.x}")
    mmg.plot_trajectory()

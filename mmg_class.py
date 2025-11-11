from general import *

@dataclass
class mmg_class:
    
    def __post_init__(self):
        # define the state
        self.x = np.zeros(6)
        self.x_dot = np.zeros_like(self.x)
        # state: (x, y, psi, x_dot, y_dot, psi_dot (r))

        # control vector
        self.u = np.zeros(2)
        # controls: (nP, delta)


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


    def forces_hull(self, u, vm, r):
        # calculate total velocity
        U = np.sqrt(u**2 + vm**2)        

        # nondim sway vel and yaw rate
        vm_pr = vm / U
        r_pr = r * Lpp / U

        # get the nondim forces
        X_H_pr, Y_H_pr, N_H_pr = self.forces_hull_nondim(vm_pr, r_pr)

        # dimensionalize
        X_H = (1/2) * rho * Lpp * d * U**2 * X_H_pr
        Y_H = (1/2) * rho * Lpp * d * U**2 * Y_H_pr
        N_H = (1/2) * rho * Lpp * d * U**2 * N_H_pr

        return X_H, Y_H, N_H


    def forces_propeller(self, u, vm, r, nP):
        betaP = np.arctan(vm / u)
        U = np.sqrt(u**2 + vm**2)
        r_pr = r * Lpp / U

        # drift angle at the propeller
        betaP_pr = betaP - x_P_pr * r_pr



    def forces_rudder(self):
        pass


    def x_dot_func(self, x, t):
        pass
        




    def timestep(self, dt):
        # RK4 for integration in time
        x = self.x
        t = self.t
        dto2 = dt / 2
        dto6 = dt / 6
        f0 = self.x_dot_func(x, t)
        f1 = self.x_dot_func(x + dto2 * f0, t + dto2)
        f2 = self.x_dot_func(x + dto2 * f1, t + dto2)
        f3 = self.x_dot_func(x + dt * f2, t + dt)
        return u + dto6 * (f0 + f1 + f2 + f3)
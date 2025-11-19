from importing import *

dir_py = Path(__file__).resolve().parent

# density of water
rho = 1000

# KVLCC2 particulars (full scale)
# Lpp = 320           # length between perpendiculars
# B = 58              # beam
# d = 20.8            # draft
# displ = 312600      # volumetric displacement
# x_G = 11.2          # long. pos. of center of gravity
# C_b = 0.810         # block coefficient
# D_P = 9.86          # diameter of propeller
# H_R = 15.80         # rudder span
# A_R = 112.5         # rudder area

# KVLCC2 particulars (model scale)
Lpp = 7             # length between perpendiculars
B = 1.27            # beam
d = 0.46            # draft
displ = 3.27        # volumetric displacement
x_G = 0.25          # long. pos. of center of gravity
C_b = 0.810         # block coefficient
D_P = 0.216         # diameter of propeller
H_R = 0.345         # rudder span
A_R = 0.0539        # rudder area

# hydrodynamic force coefficients
R_0_pr = 0.022
X_vv_pr = -0.040
X_vr_pr = 0.002
X_rr_pr = 0.011
X_vvvv_pr = 0.771
Y_v_pr = -0.315
Y_R_pr = 0.083
Y_vvv_pr = -1.607
Y_vvr_pr = 0.379
Y_vrr_pr = -0.391
Y_rrr_pr = 0.008
N_v_pr = -0.137
N_R_pr = -0.049
N_vvv_pr = -0.030
N_vvr_pr = -0.294
N_vrr_pr = 0.055
N_rrr_pr = -0.013

m_x_pr = 0.022
m_y_pr = 0.223
J_z_pr = 0.011
t_p = 0.220
t_R = 0.387
a_H = 0.312
x_H_pr = -0.464
C1 = 2
C2 = lambda bP: 1.6 if bP >= 0 else 1.1
gamma_R = lambda bR: 0.640 if bR >= 0 else 0.395
l_R_pr = -0.710
epsilon = 1.09
kappa = 0.50
f_alpha = 2.747

# prop/rudder params
w_P0 = 0.4 # model scale (L7)
x_P_pr = -0.4 # guess... should check geometry!
k0 = 0.2931
k1 = -0.2753
k2 = -0.1385
eta = D_P / H_R # geometric ratio (eq 40)
x_R = -0.5 * Lpp
x_H = x_H_pr * Lpp

# mass and inertial properties
m = rho * displ
# m_pr = m / ((1/2) * rho * Lpp**2 * d)
m_x = m_x_pr * (1/2) * rho * Lpp**2 * d
m_y = m_y_pr * (1/2) * rho * Lpp**2 * d
J_z = J_z_pr * (1/2) * rho * Lpp**4 * d
K_zz = 0.25 * Lpp # yaw gyradius, estimate
I_zG = m * K_zz**2


N_RAYS = 0
# Wider front-facing cone: -30° to +30°, evenly spaced across 9 rays
# RAY_ANGLES = np.deg2rad(np.linspace(-30, 30, N_RAYS))
RAY_ANGLES = np.deg2rad([0])


def ray_circle_distance(x, y, ang, obs_x, obs_y, obs_r, max_range=50.0):
    # Ray direction
    dx = np.cos(ang)
    dy = np.sin(ang)

    # Shift to obstacle coordinates
    fx = x - obs_x
    fy = y - obs_y

    # Quadratic coefficients: t^2 + 2(f·d)t + (f·f - r^2) = 0
    B = 2 * (fx * dx + fy * dy)
    C = fx*fx + fy*fy - obs_r**2
    D = B*B - 4*C

    if D < 0:
        return max_range  # no intersection

    sqrt_D = np.sqrt(D)

    # Two solutions
    t1 = (-B - sqrt_D) / 2
    t2 = (-B + sqrt_D) / 2

    # Need the smallest positive t
    t_candidates = [t for t in (t1, t2) if t > 0]

    if not t_candidates:
        return max_range

    t_hit = min(t_candidates)
    return min(t_hit, max_range)

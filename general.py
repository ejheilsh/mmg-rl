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

w_P0 = 0.4 # model scale (L7)
x_P_pr = -0.4 # guess... should check geometry!

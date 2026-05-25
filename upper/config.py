import numpy as np

hbar = 1.0
Gamma = 1.0
m = 1.0

hbar_actual  = 1.0545718e-34
Gamma_actual = 2 * np.pi * 29.1e6
m_actual     = 174 * 1.66053906660e-27

mu_B_actual      = 9.274009994e-24
mu_eff_actual    = mu_B_actual
mu_eff_natural   = mu_eff_actual / (hbar_actual * Gamma_actual)

x_scale        = (hbar_actual / (m_actual * Gamma_actual))**0.5 * 1e9
lambda_actual  = 399
lambda_natural = lambda_actual / x_scale
k_399_natural  = 2 * np.pi / lambda_natural
v_scale        = x_scale * 1e-9 * Gamma_actual

z0 = -25e-2

# ZS beams — ±20° from z-axis in x-z plane
k_zs_top    = np.array([np.sin(-np.pi/9), 0, -np.cos(-np.pi/9)]) * k_399_natural
k_zs_bottom = np.array([np.sin( np.pi/9), 0, -np.cos( np.pi/9)]) * k_399_natural

# 2DMOT beams — ±45° in x-z plane, retroreflected
k_2DMOT_br = np.array([ np.sin(np.pi/4), 0, -np.cos(np.pi/4)]) * k_399_natural
k_2DMOT_tr = np.array([ np.sin(-np.pi/4), 0, -np.cos(-np.pi/4)]) * k_399_natural
k_2DMOT_bl = np.array([ np.sin(np.pi/4), 0,  np.cos(np.pi/4)]) * k_399_natural
k_2DMOT_tl = np.array([ np.sin(-np.pi/4), 0,  np.cos(-np.pi/4)]) * k_399_natural

I_sat        = 60
CF25_diameter = 2.3e-2
CF40_diameter = 3.9e-2

zs_power  = 94e-3    # W per ZS beam  (upper-model N=5 optimum ~86–102 mW; N=6 is ~100–113 mW)
w0_zs     = 2e-2 / 2

s0_zs_top    = 2 * (zs_power / (np.pi * w0_zs**2)) / (I_sat * 1e1)
s0_zs_bottom = s0_zs_top

r_cut_zs_top    = CF40_diameter / 2
r_cut_zs_bottom = CF40_diameter / 2
r0_zs_top    = np.array([0, 0, -3e-2])
r0_zs_bottom = np.array([0, 0, -3e-2])
w0_zs_top    = w0_zs
w0_zs_bottom = w0_zs

power_2DMOT = 35e-3   # W per beam  (upper-model N=5 2DMOT-only optimum; N=6 is ~42 mW)
w0_2DMOT_1  = 1.5e-2 / 2

s0_2DMOT_tr = 2 * (power_2DMOT / (np.pi * w0_2DMOT_1**2)) / (I_sat * 1e1)
s0_2DMOT_tl = s0_2DMOT_tr
s0_2DMOT_br = s0_2DMOT_tr
s0_2DMOT_bl = s0_2DMOT_tr

w0_2DMOT_tr = w0_2DMOT_tl = w0_2DMOT_br = w0_2DMOT_bl = w0_2DMOT_1
r_cut_2DMOT_tr = r_cut_2DMOT_tl = r_cut_2DMOT_br = r_cut_2DMOT_bl = CF40_diameter / 2
r0_2DMOT_tr = r0_2DMOT_tl = r0_2DMOT_br = r0_2DMOT_bl = np.array([0, 0, 0])

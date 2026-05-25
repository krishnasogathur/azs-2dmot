"""
Semiclassical force model — s_eff approximation (upper bound on scattering rate).

Total saturation uses detuning-weighted contributions per beam×polarization component:
    s_eff_tot = Σ_i s_i * I_q / (1 + (2Δ_q/Γ)²)
Each beam's rate is then:
    R_q = (Γ/2) * s_eff_q / (1 + s_eff_tot)    [no additional detuning in denominator]

This overestimates the true rate in our regime. The true scattering rate lies between
the lower/ result (shared saturation) and this upper/ result (s_eff model).
"""

import numpy as np
from config import *
from joblib import Parallel, delayed
import os
from magnets import B_field as _B_default


def scattering_rate(s0, s_tot, Gamma=Gamma):
    if s0 == 0.0:
        return 0.0
    return (Gamma / 2) * s0 / (1 + s_tot)


def scattering_rate_delta(s0, s_tot, delta, Gamma=Gamma):
    if s0 == 0.0:
        return 0.0
    return (Gamma / 2) * s0 / (1 + s_tot + (2 * delta / Gamma)**2)


def scattering_rate_multi(delta_list, s_list, Gamma=Gamma):
    # diagnostic helper illustrating the s_eff upper-bound formula; not called by the integrator
    # (state_propagate_3D computes rates inline with full polarization decomposition)
    s_eff_list = [s_i / (1 + (2 * delta_i / Gamma)**2) for s_i, delta_i in zip(s_list, delta_list)]
    s_eff_tot  = sum(s_eff_list)
    return [(Gamma / 2) * s_e / (1 + s_eff_tot) for s_e in s_eff_list]


def random_unit_vector():
    u   = np.random.uniform(-1.0, 1.0)
    phi = np.random.uniform(0.0, 2 * np.pi)
    sin_theta = np.sqrt(1.0 - u * u)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), u])


def sample_effusive_velocity_3D(T=320+273, m=m_actual, size=1, a_cap=0.5e-3, L_cap=20e-3):
    kB = 1.380649e-23
    a  = m / (2 * kB * T)
    u1 = np.random.rand(size)
    u2 = np.random.rand(size)
    v  = np.sqrt(-np.log(u1 * u2) / a)
    theta_cap = a_cap / L_cap * 4
    theta = np.random.uniform(0.0, theta_cap, size)
    phi   = np.random.uniform(0.0, 2 * np.pi, size)
    return np.column_stack((
        v * np.sin(theta) * np.cos(phi),
        v * np.sin(theta) * np.sin(phi),
        v * np.cos(theta),
    ))


def sample_initial_positions_3D(size=1, n_cap=133, a_cap=0.5e-3, z0=z0):
    A_total = n_cap * np.pi * (a_cap / 2)**2
    r_cap   = np.sqrt(A_total / np.pi) * 1.2
    r     = r_cap * np.sqrt(np.random.rand(size))
    theta = np.random.uniform(0, 2 * np.pi, size)
    return np.column_stack((
        np.zeros(size),
        np.zeros(size),
        np.full(size, z0),
    ))


def s_laser(x, k_beam, s0, w0, beam_cutoff, r_center):
    r_vec  = x - r_center
    k_hat  = k_beam / np.linalg.norm(k_beam)
    r_perp = np.linalg.norm(np.cross(r_vec, k_hat))
    if r_perp > beam_cutoff:
        return 0.0
    return s0 * np.exp(-2.0 * r_perp**2 / w0**2)


def s_zs_top(x):    return s_laser(x, k_zs_top,    s0_zs_top,    w0_zs_top,    r_cut_zs_top,    r0_zs_top)
def s_zs_bottom(x): return s_laser(x, k_zs_bottom, s0_zs_bottom, w0_zs_bottom, r_cut_zs_bottom, r0_zs_bottom)
def s_2DMOT_tr(x):  return s_laser(x, k_2DMOT_tr,  s0_2DMOT_tr,  w0_2DMOT_tr,  r_cut_2DMOT_tr,  r0_2DMOT_tr)
def s_2DMOT_tl(x):  return s_laser(x, k_2DMOT_tl,  s0_2DMOT_tl,  w0_2DMOT_tl,  r_cut_2DMOT_tl,  r0_2DMOT_tl)
def s_2DMOT_br(x):  return s_laser(x, k_2DMOT_br,  s0_2DMOT_br,  w0_2DMOT_br,  r_cut_2DMOT_br,  r0_2DMOT_br)
def s_2DMOT_bl(x):  return s_laser(x, k_2DMOT_bl,  s0_2DMOT_bl,  w0_2DMOT_bl,  r_cut_2DMOT_bl,  r0_2DMOT_bl)


def polarization_weights(k_beam, B):
    """See lower/physics.py for explanation of the polarization decomposition."""
    if np.abs(k_beam[0]) != np.abs(k_beam[2]):
        return {+1: 0.5, 0: 0.0, -1: 0.5}

    factor = 1
    if k_beam[0] * k_beam[2] > 0:
        factor *= -1

    khat = k_beam / np.linalg.norm(k_beam)
    Bhat = B / (np.linalg.norm(B) + 1e-11)
    cos_theta = np.dot(khat, Bhat)

    I_sp = (1 + factor * cos_theta)**2 / 4
    I_sm = (1 - factor * cos_theta)**2 / 4
    I_pi = 1 - I_sp - I_sm

    return {+1: I_sp, 0: I_pi, -1: I_sm}


def Delta_2DMOT(Delta0, v, k_beam, v_scale=v_scale):
    return Delta0 + np.dot(k_beam, v) / v_scale


def state_propagate_3D(
        dt, x_t, v_t,
        Delta0, s_funcs, k_list,
        mu_eff=mu_eff_natural,
        gravity=True,
        return_detunings=False,
        B_field_fn=_B_default,
):
    s_list = tuple(s_func(x_t) for s_func in s_funcs)
    k_list = tuple(k_list)

    B    = B_field_fn(x_t)
    Bmag = np.linalg.norm(B)

    # total saturation: detuning-weighted (s_eff model / upper bound)
    s_eff_tot = 0.0
    for Delta0_i, k_i, s_i in zip(Delta0, k_list, s_list):
        Delta_b = Delta_2DMOT(Delta0_i, v_t, k_i)
        weights = polarization_weights(k_i, B)
        for q, I_q in weights.items():
            Delta_q    = Delta_b + q * mu_eff * Bmag / hbar
            s_eff_tot += s_i * I_q / (1 + (2 * Delta_q / Gamma)**2)

    F       = np.zeros(3)
    R_total = 0.0
    detunings_p1 = []
    detunings_m1 = []

    for Delta0_i, k_i, s_i in zip(Delta0, k_list, s_list):
        Delta_b = Delta_2DMOT(Delta0_i, v_t, k_i)
        weights = polarization_weights(k_i, B)

        for q, I_q in weights.items():
            Delta_q = Delta_b + q * mu_eff * Bmag / hbar
            s_eff_q = s_i * I_q / (1 + (2 * Delta_q / Gamma)**2)

            if return_detunings:
                if q == +1 and np.abs(k_i[0]) != np.abs(k_i[2]):
                    detunings_p1.append(Delta_q)
                if q == -1:
                    detunings_m1.append(Delta_q)

            # detuning already absorbed into s_eff_q and s_eff_tot
            R_q      = scattering_rate(s_eff_q, s_eff_tot)
            F       += hbar * k_i * R_q
            R_total += R_q

    if R_total > 0.4:
        print(f"warning: high scattering rate {R_total:.2f} at x={x_t}, v={v_t}")

    # diffusion kick disabled (mean-force regime)
    v_new = v_t + (F / m) * dt * v_scale

    if gravity:
        v_new += np.array([-9.8, 0.0, 0.0]) * (dt / Gamma_actual)

    x_new = x_t + 0.5 * (v_t + v_new) * (dt / Gamma_actual)

    if return_detunings:
        return x_new, v_new, np.array(detunings_p1), np.array(detunings_m1)
    return x_new, v_new


def run_single_trajectory(
    x0, v0, dt, n_steps,
    Delta0, s_list, k_beam,
    return_detunings=False,
    B_field_fn=_B_default,
):
    x = x0.copy()
    v = v0.copy()

    x_traj = np.zeros((n_steps + 1, 3))
    v_traj = np.zeros((n_steps + 1, 3))
    t_traj = np.zeros(n_steps + 1)

    x_traj[0] = x
    v_traj[0] = v

    if return_detunings:
        det_p10 = np.zeros(n_steps + 1)
        det_p11 = np.zeros(n_steps + 1)
        det_m10 = np.zeros(n_steps + 1)
        det_m11 = np.zeros(n_steps + 1)

    for i in range(n_steps):
        if return_detunings:
            x, v, dp1, dm1 = state_propagate_3D(
                dt, x, v, Delta0, s_list, k_beam,
                return_detunings=True, B_field_fn=B_field_fn,
            )
            if len(dp1) >= 2:
                det_p10[i + 1] = dp1[0]
                det_p11[i + 1] = dp1[1]
            if len(dm1) >= 2:
                det_m10[i + 1] = dm1[0]
                det_m11[i + 1] = dm1[1]
        else:
            x, v = state_propagate_3D(
                dt, x, v, Delta0, s_list, k_beam,
                B_field_fn=B_field_fn,
            )

        x_traj[i + 1] = x
        v_traj[i + 1] = v
        t_traj[i + 1] = (i + 1) * dt / Gamma_actual

    if return_detunings:
        return t_traj, x_traj, v_traj, det_p10, det_p11, det_m10, det_m11
    return t_traj, x_traj, v_traj


def run_many_trajectories(
    Ntraj, dt, n_steps,
    Delta0, s_list, k_beam,
    output_dir, n_jobs=-1,
    B_field_fn=_B_default,
):
    os.makedirs(output_dir, exist_ok=True)
    x0s = sample_initial_positions_3D(size=Ntraj)
    v0s = sample_effusive_velocity_3D(size=Ntraj)

    def worker(i):
        t, x, v = run_single_trajectory(
            x0s[i], v0s[i], dt, n_steps,
            Delta0, s_list, k_beam,
            B_field_fn=B_field_fn,
        )
        np.savez_compressed(
            os.path.join(output_dir, f"traj_{i:05d}.npz"),
            t=t, x=x, v=v, v0=v0s[i],
        )
        return i

    Parallel(n_jobs=n_jobs, verbose=10)(delayed(worker)(i) for i in range(Ntraj))

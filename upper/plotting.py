"""
Trajectory visualization — single-atom diagnostic plots.

Plots:
  1. vz(z) and az(z) along the ZS+2DMOT path
  2. Detuning vs z for each beam/polarization component (2x2 grid)
     overlaid with beam saturation profile

Run as a script; adjust v0 and dt_s at the top as needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import *
from physics import *
from magnets import B_field_n5 as _B_active   # N=5 default; swap for B_field for N=6

Delta0_zs    = 4.6   # upper-model N=5 optimum; for N=6 use 5.1 Γ
Delta0_2DMOT = 2.45

Delta0_list = (Delta0_zs, Delta0_zs,
               Delta0_2DMOT, Delta0_2DMOT, Delta0_2DMOT, Delta0_2DMOT)

k_beam_list = (k_zs_top, k_zs_bottom,
               k_2DMOT_tr, k_2DMOT_tl, k_2DMOT_br, k_2DMOT_bl)

s_list = (s_zs_top, s_zs_bottom,
          s_2DMOT_tr, s_2DMOT_tl, s_2DMOT_br, s_2DMOT_bl)

if __name__ == "__main__":
    x0   = np.array([0, 0, z0])
    v0   = np.array([-1, 0.5, 148])
    dt_s = 20e-6
    t_tot = 25e-3

    dt      = dt_s * Gamma_actual
    n_steps = int(t_tot / dt_s)

    t, x_traj, v_traj, det_p10, det_p11, det_m10, det_m11 = run_single_trajectory(
        x0=x0, v0=v0, dt=dt, n_steps=n_steps,
        Delta0=Delta0_list, s_list=s_list, k_beam=k_beam_list,
        return_detunings=True, B_field_fn=_B_active,
    )

    # beam saturation profiles along z-axis
    z_line = np.linspace(-8e-2, 10e-2, 1000)
    pos_line = np.column_stack((np.zeros_like(z_line), np.zeros_like(z_line), z_line))

    s_top_prof    = np.array([s_zs_top(p)    / (1 + s_zs_top(p))    for p in pos_line])
    s_bottom_prof = np.array([s_zs_bottom(p) / (1 + s_zs_bottom(p)) for p in pos_line])

    z_cm = x_traj[:, 2] * 1e2
    mask = (z_line * 1e2 >= -6) & (z_line * 1e2 <= 1)

    # ---- detuning panels ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0, 0], det_p10, s_top_prof,    "Top ZS, m=+1"),
        (axes[0, 1], det_p11, s_bottom_prof, "Bottom ZS, m=+1"),
        (axes[1, 0], det_m10, s_top_prof,    "Top ZS, m=−1"),
        (axes[1, 1], det_m11, s_bottom_prof, "Bottom ZS, m=−1"),
    ]
    for ax, det, s_prof, title in panels:
        ax.plot(z_cm, det, color="C0", lw=2, label="Detuning")
        ax.set_xlim(-6, 1); ax.set_ylim(-4, 4)
        ax.set_xlabel("z (cm)"); ax.set_ylabel("Detuning (Γ)", color="C0")
        ax.set_title(title); ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(z_line[mask] * 1e2, s_prof[mask], color="C1", lw=2, ls="--", label="s(x)")
        ax2.set_ylabel("Saturation parameter", color="C1")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    fig.tight_layout()

    # ---- vz and az vs z ----
    a_z = np.gradient(v_traj[:, 2], dt_s)
    fig2, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(z_cm, v_traj[:, 2], color="tab:blue", lw=2.5, label=r"$v_z$")
    ax1.set_xlabel("z (cm)"); ax1.set_ylabel(r"$v_z$ (m/s)", color="tab:blue")
    ax1.set_xlim(-6, 2); ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(z_cm, a_z, color="tab:red", lw=2.5, ls="--", label=r"$a_z$")
    ax2.set_ylabel(r"$a_z$ (m/s²)", color="tab:red")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title(f"Trajectory: v0z = {v0[2]:.0f} m/s")
    fig2.tight_layout()

    plt.show()

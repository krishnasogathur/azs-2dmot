import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import time
from config import *
from physics import *
from magnets import B_field_n5 as _B_active   # N=5 default; swap for B_field for N=6

# -----------------------------------------------------------------------
# Active parameters — upper model N=5 optimum (see optimization-results.txt)
# For N=6: Delta0_zs ≈ 5.1 Γ, Delta0_2DMOT ≈ 2.7 Γ — swap B_field_n5 for B_field above
# -----------------------------------------------------------------------
Delta0_zs     = 4.6   # representative of 4.5–4.8 Γ range; see optimization-results.txt
Delta0_2DMOT  = 2.45  # upper-model N=5 2DMOT-only optimum

Delta0_list = (Delta0_zs, Delta0_zs,
               Delta0_2DMOT, Delta0_2DMOT, Delta0_2DMOT, Delta0_2DMOT)

k_beam_list = (k_zs_top, k_zs_bottom,
               k_2DMOT_tr, k_2DMOT_tl, k_2DMOT_br, k_2DMOT_bl)

s_list = (s_zs_top, s_zs_bottom,
          s_2DMOT_tr, s_2DMOT_tl, s_2DMOT_br, s_2DMOT_bl)

# -----------------------------------------------------------------------
# Capture probability for a given axial entry velocity
# -----------------------------------------------------------------------
OUTPUT_DIR = "results/capture"

def capture_probability(v0z, Ni, dt_s=20e-6, t_tot=25e-3, output_dir=OUTPUT_DIR):
    dt      = dt_s * Gamma_actual
    n_steps = int(t_tot / dt_s)

    vel_dir = os.path.join(output_dir, f"v0z_{v0z:.1f}")
    os.makedirs(vel_dir, exist_ok=True)

    x_trajs       = []
    v_trajs       = []
    captured_mask = []

    plt.figure(figsize=(6, 4), dpi=300)

    for i in range(Ni):
        print(f"  v0z={v0z:.1f} m/s  traj {i+1}/{Ni}", end="\r")

        x0 = sample_initial_positions_3D(size=1)
        _, x_traj, v_traj = run_single_trajectory(
            x0=x0, v0=np.array([-1.0, 0.5, v0z]),
            dt=dt, n_steps=n_steps,
            Delta0=Delta0_list, s_list=s_list, k_beam=k_beam_list,
            B_field_fn=_B_active,
        )

        x_trajs.append(x_traj)
        v_trajs.append(v_traj)

        plt.scatter(x_traj[:, 2] * 1e2, v_traj[:, 2], s=0.5, alpha=0.08, color="black")

        captured = abs(x_traj[-1, 2]) < 0.01 and np.linalg.norm(v_traj[-1]) < 5.0
        captured_mask.append(captured)

    print(" " * 60, end="\r")

    np.savez_compressed(
        os.path.join(vel_dir, "trajectories.npz"),
        x_trajs=np.stack(x_trajs),
        v_trajs=np.stack(v_trajs),
        captured_mask=np.array(captured_mask),
        v0z=v0z, dt=dt, n_steps=n_steps,
    )

    P   = sum(captured_mask) / Ni
    err = np.sqrt(P * (1 - P) / Ni)

    plt.xlabel("z (cm)")
    plt.ylabel(r"$v_z$ (m/s)")
    plt.title(f"Phase-space cloud — v0z = {v0z:.1f} m/s")
    plt.xlim(-30, 2)
    plt.tight_layout()
    plt.savefig(os.path.join(vel_dir, "vz_scatter.png"), dpi=500)
    plt.close()

    return P, err


if __name__ == "__main__":
    # velocity sweep — N=6 upper model ZS-assisted regime (~155–158 m/s)
    # note: each v0z runs Ni identical deterministic trajectories (position/velocity fixed);
    # capture is 0 or 1 per v0z, giving a binary step function around v_c
    v0z_sweep = np.linspace(125, 180, 9)
    Ni        = 200

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "capture_probability.csv")

    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["v0z (m/s)", "P_capture", "error"])
        for v0z in v0z_sweep:
            P, err = capture_probability(v0z, Ni)
            writer.writerow([v0z, P, err])
            print(f"v0z = {v0z:.1f} m/s  →  P = {P:.3f} ± {err:.3f}")

    print(f"\nDone in {time.time()-t0:.1f}s  |  results → {csv_path}")

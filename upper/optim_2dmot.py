"""
Bayesian optimisation of 2DMOT detuning and power (ZS off).

Objective: maximise capture velocity v_c for atoms entering along z with
small transverse velocity. One deterministic trajectory per (v0z, params) point.

Search space:
  Delta0_2DMOT  ∈ [2.0, 3.0] Γ
  power_2DMOT   ∈ [35, 55] mW  (N=6 upper-model optimum ~42–53 mW)

ZS is off (Delta0_zs = 0, power_zs_W = 0) in this stage.
Results in optim-results/{timestamp}/.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, json
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Real
from config import *
from physics import *

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR  = f"optim-results/{TIMESTAMP}"
os.makedirs(SAVE_DIR, exist_ok=True)

NV         = 15
N_INIT     = 100
N_ITER     = 50
V_MIN      = 45.0
V_MAX      = 70.0
DT_S       = 5e-6
T_TOT      = 10e-3
CAP_THRESH = 0.5
POS_THRESH = 0.005
VEL_THRESH = 5.0

SPACE = [
    Real(2.0,   3.0,  name="Delta0_2DMOT"),
    Real(35e-3, 55e-3, name="power_2DMOT"),
]

tracker = {
    "all_params": [], "all_vc": [],
    "best_vc": -np.inf, "best_params": None, "call_count": 0,
}


def make_s_list(power_2DMOT_W, power_zs_W=0.0):
    s0_zs = 2.0 * (power_zs_W    / (np.pi * w0_zs**2))      / (I_sat * 1e1) if power_zs_W > 0 else 0.0
    s0_2d = 2.0 * (power_2DMOT_W / (np.pi * w0_2DMOT_1**2)) / (I_sat * 1e1)
    return (
        lambda x: s_laser(x, k_zs_top,    s0_zs, w0_zs,      r_cut_zs_top,    r0_zs_top),
        lambda x: s_laser(x, k_zs_bottom, s0_zs, w0_zs,      r_cut_zs_bottom, r0_zs_bottom),
        lambda x: s_laser(x, k_2DMOT_tr,  s0_2d, w0_2DMOT_1, r_cut_2DMOT_tr,  r0_2DMOT_tr),
        lambda x: s_laser(x, k_2DMOT_tl,  s0_2d, w0_2DMOT_1, r_cut_2DMOT_tl,  r0_2DMOT_tl),
        lambda x: s_laser(x, k_2DMOT_br,  s0_2d, w0_2DMOT_1, r_cut_2DMOT_br,  r0_2DMOT_br),
        lambda x: s_laser(x, k_2DMOT_bl,  s0_2d, w0_2DMOT_1, r_cut_2DMOT_bl,  r0_2DMOT_bl),
    )


def get_capture_velocity(Delta0_zs, Delta0_2DMOT, power_2DMOT_W, power_zs_W=0.0):
    Delta0_list = (Delta0_zs, Delta0_zs,
                   Delta0_2DMOT, Delta0_2DMOT, Delta0_2DMOT, Delta0_2DMOT)
    k_beam_list = (k_zs_top, k_zs_bottom,
                   k_2DMOT_tr, k_2DMOT_tl, k_2DMOT_br, k_2DMOT_bl)
    s_funcs = make_s_list(power_2DMOT_W, power_zs_W)

    dt      = DT_S * Gamma_actual
    n_steps = int(T_TOT / DT_S)
    x0      = sample_initial_positions_3D(size=1)

    v0z_sweep    = np.linspace(V_MIN, V_MAX, NV)
    capture_frac = np.zeros(NV)

    for i, v0z in enumerate(v0z_sweep):
        _, x_traj, v_traj = run_single_trajectory(
            x0=x0, v0=np.array([-1.0, 0.5, v0z]),
            dt=dt, n_steps=n_steps,
            Delta0=Delta0_list, s_list=s_funcs, k_beam=k_beam_list,
        )
        capture_frac[i] = float(
            abs(x_traj[-1, 2]) < POS_THRESH and abs(v_traj[-1, 2]) < VEL_THRESH
        )
        print(f"    v0z={v0z:.1f}  →  {'CAPTURED' if capture_frac[i] else 'lost'}  ({i+1}/{NV})", flush=True)
        if capture_frac[i] == 0:
            break

    mask = capture_frac >= CAP_THRESH
    vc   = float(v0z_sweep[mask].max()) if mask.any() else float("nan")
    return {"vc": vc, "v0z_sweep": v0z_sweep, "capture_frac": capture_frac}


def objective(params):
    Delta0_2DMOT  = round(params[0], 2)
    power_2DMOT_W = round(params[1], 4)

    res    = get_capture_velocity(0.0, Delta0_2DMOT, power_2DMOT_W)
    vc_val = res["vc"] if not np.isnan(res["vc"]) else 0.0

    tracker["call_count"] += 1
    tracker["all_params"].append([0.0, Delta0_2DMOT, power_2DMOT_W, 0.0])
    tracker["all_vc"].append(vc_val)

    is_best = vc_val > tracker["best_vc"]
    if is_best:
        tracker["best_vc"]     = vc_val
        tracker["best_params"] = [0.0, Delta0_2DMOT, power_2DMOT_W, 0.0]

    phase    = "INIT" if tracker["call_count"] <= N_INIT else f"ITER {tracker['call_count']-N_INIT:2d}"
    best_str = " ← NEW BEST" if is_best else ""
    print(
        f"\n[{phase}  |  eval {tracker['call_count']:3d}/{N_INIT+N_ITER}]  "
        f"Δ2D={Delta0_2DMOT:.2f}  P_2D={power_2DMOT_W*1e3:.1f}mW  "
        f"→  vc={vc_val:.1f} m/s  (best: {tracker['best_vc']:.1f} m/s){best_str}",
        flush=True,
    )

    eval_dir = os.path.join(SAVE_DIR, "evals", f"eval_{tracker['call_count']:03d}")
    os.makedirs(eval_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(eval_dir, "data.npz"),
        v0z_sweep=res["v0z_sweep"], capture_frac=res["capture_frac"],
        Delta0_2DMOT=Delta0_2DMOT, power_2DMOT_mW=power_2DMOT_W*1e3, vc=vc_val,
    )
    return -vc_val


def _save_summary_plot():
    params      = np.array(tracker["all_params"])
    vcs         = np.array(tracker["all_vc"])
    iters       = np.arange(1, len(vcs) + 1)
    best_so_far = np.maximum.accumulate(vcs)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(iters, vcs, "o-", color="C0", ms=4, lw=1.2, alpha=0.6, label="vc")
    ax0.plot(iters, best_so_far, "s-", color="C3", ms=5, lw=1.8, label="best")
    ax0.axvline(N_INIT + 0.5, color="k", ls=":", lw=1)
    ax0.set_xlabel("Eval #"); ax0.set_ylabel("$v_c$ (m/s)"); ax0.set_title("Convergence")
    ax0.legend(fontsize=7)

    ax1 = fig.add_subplot(gs[1])
    sc1 = ax1.scatter(params[:, 1], params[:, 2]*1e3, c=vcs, cmap="viridis", s=60, edgecolors="k", lw=0.4)
    plt.colorbar(sc1, ax=ax1, label="$v_c$ (m/s)")
    ax1.set_xlabel(r"$\Delta_{2D}$ ($\Gamma$)"); ax1.set_ylabel("$P_{2D}$ (mW)")
    ax1.set_title("Parameter space")

    ax2 = fig.add_subplot(gs[2])
    sc2 = ax2.scatter(params[:, 2]*1e3, vcs, c=vcs, cmap="viridis", s=60, edgecolors="k", lw=0.4)
    plt.colorbar(sc2, ax=ax2, label="$v_c$ (m/s)")
    ax2.set_xlabel("$P_{2D}$ (mW)"); ax2.set_ylabel("$v_c$ (m/s)"); ax2.set_title("Power vs $v_c$")

    ax3 = fig.add_subplot(gs[3])
    best_idx  = int(np.argmax(vcs))
    eval_path = os.path.join(SAVE_DIR, "evals", f"eval_{best_idx+1:03d}", "data.npz")
    if os.path.exists(eval_path):
        d = np.load(eval_path)
        ax3.plot(d["v0z_sweep"], d["capture_frac"]*100, "o-", color="C2", ms=4, lw=1.5)
        ax3.axhline(CAP_THRESH*100, color="k", ls="--", lw=1)
        if not np.isnan(float(d["vc"])):
            ax3.axvline(float(d["vc"]), color="C3", ls="-.", lw=1.5,
                        label=f"vc = {float(d['vc']):.1f} m/s")
        ax3.legend(fontsize=8)
    ax3.set_xlabel(r"$v_{0z}$ (m/s)"); ax3.set_ylabel("Capture (%)"); ax3.set_ylim(-5, 105)
    ax3.set_title("Best capture curve")

    best_p = tracker["best_params"]
    plt.suptitle(
        f"BO 2DMOT — eval {len(vcs)}/{N_INIT+N_ITER}   |   "
        f"best vc = {tracker['best_vc']:.1f} m/s  "
        f"@ Δ2D={best_p[1]:.2f}, P_2D={best_p[2]*1e3:.1f}mW",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(SAVE_DIR, "summary.png"), dpi=180)
    plt.close()


def callback(res):
    if tracker["call_count"] >= 1:
        _save_summary_plot()


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  BO: 2DMOT detuning + power  (ZS off)")
    print(f"  Search:  Δ2D ∈ [2.0,3.0]Γ   P_2D ∈ [35,55]mW")
    print(f"  Budget:  {N_INIT} init + {N_ITER} BO = {N_INIT+N_ITER} evals")
    print(f"  Results → {SAVE_DIR}/")
    print(f"{'='*60}\n")

    result = gp_minimize(
        func=objective, dimensions=SPACE,
        n_calls=N_INIT+N_ITER, n_initial_points=N_INIT,
        noise=1e-2, callback=[callback], random_state=42, verbose=False,
    )

    best_2d  = result.x[0]
    best_p2d = result.x[1]
    best_vc  = -result.fun

    print(f"\n  Best Δ2D     = {best_2d:.4f} Γ")
    print(f"  Best P_2DMOT = {best_p2d*1e3:.2f} mW")
    print(f"  Best vc      = {best_vc:.2f} m/s")

    np.savez_compressed(
        os.path.join(SAVE_DIR, "optim_result.npz"),
        all_params=np.array(tracker["all_params"]),
        all_vc=np.array(tracker["all_vc"]),
        best_params=np.array(tracker["best_params"]),
        best_vc=tracker["best_vc"],
    )
    with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
        json.dump({
            "best_Delta0_2DMOT": best_2d, "best_power_2DMOT_mW": best_p2d*1e3,
            "best_vc": best_vc, "NV": NV, "N_INIT": N_INIT, "N_ITER": N_ITER,
            "V_MIN": V_MIN, "V_MAX": V_MAX, "timestamp": TIMESTAMP,
        }, f, indent=2)

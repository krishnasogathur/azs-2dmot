ZS + 2DMOT semiclassical simulation
=====================================

This folder (azs-2dmot/) is the refactored, presentable version of the ZS + 2DMOT
simulation code. The original working code lives in zs-2dmot/ — do not edit that
as a source of truth; this is the canonical version going forward.


PHYSICAL SETUP
--------------
Atom: Yb-174 (broad imaging line, 399 nm, Γ = 2π × 29.1 MHz)
Geometry: two Zeeman-slower beams at ±20° from z-axis, four 2DMOT beams at ±45°,
          all retroreflected. Atom source at z = -25 cm.
Magnetic field: permanent NdFeB cuboid magnets arranged to produce a 2D quadrupole
                in the x-z plane for the 2DMOT, with ZS compensation along z < 0.
Natural units: energy in ℏΓ, length in √(ℏ/mΓ), time in 1/Γ.


TWO SCATTERING MODELS — lower/ and upper/
------------------------------------------
The exact multi-beam, multi-detuning scattering rate is non-trivial because the
shared saturation in the denominator involves all beams simultaneously. We bracket
it with two models:

lower/  — shared saturation (exact when all beams are equally detuned)
          R_q = (Γ/2) · s_q / (1 + s_tot + (2Δ_q/Γ)²)
          where s_tot = Σ_i s_i (raw sum, no detuning weighting)
          → LOWER bound on the true scattering rate

upper/  — s_eff model
          s_eff_q = s_q / (1 + (2Δ_q/Γ)²),  s_eff_tot = Σ s_eff_q
          R_q = (Γ/2) · s_eff_q / (1 + s_eff_tot)    [no extra detuning in denom]
          → UPPER bound; verified to OVERESTIMATE in our regime

The true scattering rate lies between lower/ and upper/ results.
A separate comparison script (in zs-2dmot/testing-code/) confirmed this.

The two directories are fully self-contained — no cross-imports. The only
difference between them is physics.py (the scattering_rate_multi function)
and the beam powers in config.py (different operating points were found to
be optimal for each model).


MAGNETIC FIELD — static vs. adjustable (within each lower/ or upper/)
----------------------------------------------------------------------
Both magnets.py files define TWO variants of the magnet geometry:

  B_field(x)             — N=5 magnet stacks, fixed geometry. Used by runner.py
                           and optim_2dmot.py for standard simulations.

  make_B_field(delta_x)  — N=6 stacks with radial offset delta_x (metres).
                           Build once per Bayesian optimisation eval.
                           Used by optim_zs_magnets.py.
                           make_B_field(0.0) gives the N=6 static reference.

The magnet arrangement: four stacks (R1, R2, L1, L2) at ±xpos, ±ypos,
magnetized along ±x to produce the 2D quadrupole. xpos ≈ 5.45 cm from beam axis.


FILE STRUCTURE
--------------
Each of lower/ and upper/ contains:

  config.py             — all physical constants, beam wavevectors, saturation
                          parameters, beam geometry. Import with `from config import *`.

  physics.py            — scattering model (differs between lower/ and upper/),
                          trajectory integrator (state_propagate_3D, run_single_trajectory,
                          run_many_trajectories), sampling functions.

  magnets.py            — magpylib-based magnetic field. Defines B_field (static N=5)
                          and make_B_field(delta_x) (adjustable N=6).
                          Run as __main__ to print field diagnostics.

  runner.py             — capture probability sweep: runs Ni trajectories per v0z,
                          saves trajectories + scatter plots to results/capture/.
                          Edit Delta0_zs, Delta0_2DMOT at the top before running.

  plotting.py           — single-atom diagnostic: vz(z), az(z), plus detuning vs z
                          for each beam/polarization overlaid with saturation profile.
                          Edit v0 at the top.

  optim_2dmot.py        — Stage 1 Bayesian optimisation: ZS off, optimise 2DMOT
                          detuning and power to maximise capture velocity v_c.
                          Results → optim-results/{timestamp}/.

  optim_zs_magnets.py   — Stage 2 Bayesian optimisation: 2DMOT params fixed (from
                          Stage 1), optimise ZS detuning, ZS power, and optionally
                          magnet radial offset delta_x. Results → optim-results/{timestamp}/.

  results/              — output directory for runner.py (capture probability data,
                          trajectory .npz files, scatter plots).


OPTIMISATION WORKFLOW
---------------------
Run in order:

  1. optim_2dmot.py     — find best (Δ_2DMOT, P_2DMOT) with ZS off.
                          Capture velocity window is the 2DMOT-only range (~40-80 m/s).

  2. optim_zs_magnets.py — fix 2DMOT params from step 1, optimise (Δ_ZS, P_ZS)
                           and optionally delta_x (magnet offset).
                           Capture velocity window extends to the ZS range (~125-150 m/s).

  3. runner.py          — with final params, run the full capture probability sweep
                          and save trajectories for analysis.

Representative params (lower/ model, N=5 — see optimization-results.txt for full ranges):
  Δ_ZS ≈ 4.2 Γ,  P_ZS ≈ 90 mW,  Δ_2DMOT ≈ 2.27 Γ,  P_2DMOT ≈ 37.5 mW  →  vc ~ 131 m/s
  (N=6 lower-model ZS: not yet run — optim_zs_magnets.py is configured and ready)


RESULTS STATUS
--------------
lower/results/: ZS compensation verified (compensation.png + optim-curve plots; TODO — copy in).
                Capture probability sweep: TODO — run runner.py with final params.

upper/results/: TODO — results not yet produced with refactored code.

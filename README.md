# ZS + 2DMOT semiclassical simulation

Semiclassical trajectory simulation for a Zeeman slower (ZS) combined with a 2D magneto-optical trap (2DMOT), using permanent NdFeB magnets to produce the magnetic field. The atom is Yb-174 on the broad imaging transition (399 nm, Γ = 2π × 29.1 MHz).

The code brackets the true multi-beam scattering rate with two models: a **lower bound** (`lower/`) and an **upper bound** (`upper/`). True capture performance lies between them.

---

## Physical setup

| Parameter | Value |
|---|---|
| Atom | Yb-174 |
| Transition | 1S₀ → ¹P₁ (broad), 399 nm |
| Linewidth | Γ = 2π × 29.1 MHz |
| ZS beams | 2 beams at ±20° from z-axis, counter-propagating |
| 2DMOT beams | 4 beams at ±45° in x-z plane, retroreflected |
| Atom source | z = −25 cm |
| Magnetic field | Permanent NdFeB cuboid magnets (4 stacks), 2D quadrupole + ZS compensation |

Natural units used throughout: energy in ℏΓ, length in √(ℏ/mΓ), time in 1/Γ.

---

## Two scattering models

The exact multi-beam, multi-detuning scattering rate is non-trivial because all beams share a common saturation denominator. We bracket it:

**`lower/` - shared saturation (lower bound)**

```
s_tot = Σ_i s_i · I_q(i)          (raw sum, no detuning weighting)
R_q   = (Γ/2) · s_q / (1 + s_tot + (2Δ_q/Γ)²)
```

Exact when all beams are equally detuned. Gives a **lower bound** on the true rate.

**`upper/` - s_eff model (upper bound)**

```
s_eff_q  = s_i · I_q / (1 + (2Δ_q/Γ)²)
s_eff_tot = Σ s_eff_q
R_q       = (Γ/2) · s_eff_q / (1 + s_eff_tot)
```

Verified to **overestimate** the true rate in this regime. Gives an upper bound.

A separate comparison script (not in this repo) confirmed that the true rate lies between the two models.

---

## Magnet geometry

Four stacks of NdFeB cuboid magnets (each magnet 5 × 15 × 20 mm, B_r = 1.24 T) arranged in ±y, ±z to produce a 2D quadrupole in the x-z plane.

Each `magnets.py` defines two variants:

| Function | Description |
|---|---|
| `B_field(x)` | Fixed-geometry static field. Used by `runner.py` and `optim_2dmot.py`. |
| `make_B_field(delta_x)` | Adjustable radial offset. Used by `optim_zs_magnets.py` for magnet position search. |

**lower/** uses N = 5 magnets per stack (≈ 35 G/cm radial gradient).  
**upper/** uses N = 6 magnets per stack (≈ 42 G/cm radial gradient), larger flange radius.

---

## File structure

Each of `lower/` and `upper/` is fully self-contained:

```
config.py           - physical constants, beam vectors, saturation parameters
physics.py          - scattering model + trajectory integrator (differs between lower/ and upper/)
magnets.py          - magpylib magnetic field; B_field (static) and make_B_field (adjustable)
runner.py           - capture probability sweep over initial axial velocity
plotting.py         - single-atom diagnostic: vz(z), az(z), detuning panels
optim_2dmot.py      - Stage 1 BO: ZS off, optimise 2DMOT detuning and power
optim_zs_magnets.py - Stage 2 BO: 2DMOT fixed, optimise ZS detuning, power, magnet offset
results/            - output directory for runner.py
```

---

## Optimisation workflow

```
1. optim_2dmot.py       Find best (Δ_2DMOT, P_2DMOT) with ZS off.
                        Capture velocity window: 2DMOT-only regime (~40–65 m/s).

2. optim_zs_magnets.py  Fix 2DMOT params from step 1, optimise (Δ_ZS, P_ZS, δx).
                        Capture velocity window extends to ZS-assisted regime (~125–160 m/s).

3. runner.py            Full capture probability sweep with final params.
```

---

## Results summary

See `results-summary.txt` for the full parameter ranges and capture velocities from each model/N combination. Key outcomes:

| Model | N | Δ_ZS (Γ) | Δ_2D (Γ) | P_ZS (mW) | P_2D (mW) | v_c (m/s) |
|---|---|---|---|---|---|---|
| Lower | 5 | 4.0–4.3 | 2.2–2.3 | 85–94 | 37–38 | ~131 |
| Lower | 6 | - | 2.4–2.6 | - | 46–48 | ~54 (2DMOT only) |
| Upper | 5 | 4.5–4.8 | 2.3–2.5 | 86–102 | 35–40 | ~150 |
| Upper | 6 | 5.0–5.2 | 2.7 | 100–113 | 42 | ~155–158 |
| Upper | 7 | 4.7–5.2 | 2.85 | 104–124 | 46 | ~157–165 |

- N=7 was explored under the upper model only (overestimates; ruled out).
- Lower-model ZS optimisation at N=6 has not been run - `lower/optim_zs_magnets.py` is configured and ready.
- Decided experimental ZS detuning: **Δ_ZS ≈ 4.5 Γ** (between lower ~4.1 and upper ~4.7 optima).

---

## Dependencies

```
numpy
scipy
matplotlib
magpylib      (magnetic field computation)
scikit-optimize   (gp_minimize, for optim scripts only)
joblib        (parallel trajectory runs, for run_many_trajectories only)
```

Run from within `lower/` or `upper/`:

```bash
cd lower/
python runner.py          # capture probability sweep
python plotting.py        # single-atom diagnostic
python optim_2dmot.py     # Stage 1 Bayesian optimisation
python optim_zs_magnets.py  # Stage 2 Bayesian optimisation
```

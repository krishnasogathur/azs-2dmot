"""
Permanent magnet geometry for the lower/ configuration.

Two stacks of NdFeB cuboid magnets sit on each side (±z), above and below (±y),
magnetized along x. The arrangement produces a 2D quadrupole in the x-z plane
as needed for the 2DMOT, with a ZS compensation region along z < 0.

Two variants are defined:
  B_field         — N=5 stacks, fixed geometry. Default for runner and optim_2dmot.
  make_B_field    — N=6 stacks with adjustable radial offset delta_x.
                    Used by optim_zs_magnets to scan magnet position.

Both use the same base xpos = r_flange + delta_r (≈ 5.45 cm from axis).
make_B_field(0.0) reproduces the N=6 static field; pass delta_x to shift radially.
"""

import numpy as np
import magpylib as magpy


def _create_stack(N, magnetization, dimension):
    c = magpy.Collection()
    for i in range(N):
        c.add(magpy.magnet.Cuboid(
            magnetization=magnetization,
            dimension=dimension,
            position=[i * dimension[0], 0, 0]
        ))
    c.move([-(N // 2) * dimension[0], 0, 0])
    return c


B_r   = 1.24
mu0   = 4e-7 * np.pi
M     = B_r / mu0
dimension    = np.array([5e-3, 15e-3, 20e-3])   # each magnet: 5x15x20 mm
magnetization = np.array([M, 0, 0])

t        = 6.5e-2    # y-separation reference (m)
r_flange = 7.1e-2 / 2
delta_r  = 3.8e-2 / 2
delta_y  = 6e-3
l        = dimension[1]

_XPOS_BASE = r_flange + delta_r   # ≈ 5.45 cm from beam axis


def _build(N, delta_x=0.0):
    xpos = _XPOS_BASE + delta_x
    ypos = (t / 2) + (l / 2) + delta_y

    R1 = _create_stack(N,  magnetization, dimension)
    R2 = _create_stack(N,  magnetization, dimension)
    L1 = _create_stack(N, -magnetization, dimension)
    L2 = _create_stack(N, -magnetization, dimension)

    R1.move([0, -ypos,  xpos])
    R2.move([0,  ypos,  xpos])
    L1.move([0, -ypos, -xpos])
    L2.move([0,  ypos, -xpos])

    return magpy.Collection(R1, L1, R2, L2)


_static_system = _build(N=5)

def B_field(x):
    """Fixed N=5 geometry."""
    return _static_system.getB(x).flatten()


def make_B_field(delta_x=0.0):
    """N=6 geometry with radial offset delta_x (m). Build once per BO eval."""
    system = _build(N=6, delta_x=delta_x)
    def _B(x):
        return system.getB(x).flatten()
    return _B


if __name__ == "__main__":
    x = np.arange(-0.05, 0.05, 0.001)
    Bz = np.array([B_field([0, 0, xi])[0] for xi in x])
    slopes = np.gradient(Bz, x)

    Br = np.array([
        np.dot(B_field([xi, 0, xi]), np.array([1, 0, 1]) / np.sqrt(2))
        for xi in x
    ])
    slopes_r = np.gradient(Br, x)

    print(f"At z=0, radial (x=z) slope = {100*slopes_r[np.argmin(np.abs(x))]:.1f} G/cm")
    for z_cm in [-3, -4.5, -6]:
        xi = z_cm * 1e-2
        idx = np.argmin(np.abs(x - xi))
        print(f"At z={z_cm}cm: |B| = {np.linalg.norm(B_field([0,0,xi])):.4f} T, "
              f"slope = {100*slopes[idx]:.1f} G/cm")

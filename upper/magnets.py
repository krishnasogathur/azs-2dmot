"""
Permanent magnet geometry for the upper/ configuration.

Two magnet geometries are supported, corresponding to the two N values explored:

  N=5 (~35 G/cm) — same physical geometry as lower/magnets.py.
    B_field_n5 / make_B_field(..., N=5)
    r_flange = 3.55 cm, xpos ≈ 5.45 cm from beam axis.
    Used for the N=5 upper-model optimisation results in results/.

  N=6 (~42 G/cm) — larger flange geometry.
    B_field / make_B_field(..., N=6)   [default]
    r_flange = 4.00 cm, xpos ≈ 5.60 cm from beam axis.
    Used for the N=6 upper-model optimisation (see optimization-results.txt).

runner.py and plotting.py default to N=5. To switch to N=6, replace B_field_n5
with B_field and use the N=6 params from optimization-results.txt.
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
dimension    = np.array([5e-3, 15e-3, 20e-3])
magnetization = np.array([M, 0, 0])

t      = 6.5e-2
delta_y = 6e-3
l       = dimension[1]

# N=5 geometry — identical to lower/magnets.py
_N5_XPOS_BASE = 7.1e-2/2 + 3.8e-2/2          # 3.55 + 1.90 = 5.45 cm

# N=6 geometry — larger flange
_N6_XPOS_BASE = 8e-2/2 + (3.8e-2/2 - 0.3e-2) # 4.00 + 1.60 = 5.60 cm


def _build(N, xpos_base, delta_x=0.0):
    xpos = xpos_base + delta_x
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


_static_n5 = _build(N=5, xpos_base=_N5_XPOS_BASE)
_static_n6 = _build(N=6, xpos_base=_N6_XPOS_BASE)


def B_field_n5(x):
    """N=5 static field (~35 G/cm). Same geometry as lower/magnets.py."""
    return _static_n5.getB(x).flatten()


def B_field(x):
    """N=6 static field (~42 G/cm). Larger flange geometry."""
    return _static_n6.getB(x).flatten()


def make_B_field(delta_x=0.0, N=6):
    """Adjustable radial offset. N selects geometry (5 or 6). Build once per BO eval."""
    xpos_base = _N5_XPOS_BASE if N == 5 else _N6_XPOS_BASE
    system = _build(N=N, xpos_base=xpos_base, delta_x=delta_x)
    def _B(x):
        return system.getB(x).flatten()
    return _B


if __name__ == "__main__":
    for label, fn, n in [("N=5", B_field_n5, 5), ("N=6", B_field, 6)]:
        x = np.arange(-0.05, 0.05, 0.001)
        Bz = np.array([fn([0, 0, xi])[0] for xi in x])
        Br = np.array([np.dot(fn([xi, 0, xi]), np.array([1, 0, 1]) / np.sqrt(2)) for xi in x])
        slopes_r = np.gradient(Br, x)
        slopes   = np.gradient(Bz, x)
        print(f"\n{label}:")
        print(f"  radial gradient at z=0: {100*slopes_r[np.argmin(np.abs(x))]:.1f} G/cm")
        for z_cm in [-3, -4.5, -6]:
            xi  = z_cm * 1e-2
            idx = np.argmin(np.abs(x - xi))
            print(f"  z={z_cm}cm: |B| = {np.linalg.norm(fn([0,0,xi])):.4f} T, "
                  f"slope = {100*slopes[idx]:.1f} G/cm")

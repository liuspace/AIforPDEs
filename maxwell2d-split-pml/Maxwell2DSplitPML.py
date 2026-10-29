"""
2D Maxwell (TMz) PINN with a Berenger split-field PML, solved with DeepXDE.

A Gaussian Ez pulse starts at the center of the unit square and radiates
outward. The physical region [0.2, 0.8]^2 is surrounded by a 0.2-thick PML,
backed by a PEC wall (Ez = 0) on the outer boundary. See README.md for the
equations, the loss terms, and known caveats.

Network inputs:  (x, y, t), all normalized.
Network outputs: (Ezx, Hx, Hy, Ezy), with Ez = Ezx + Ezy.
"""
import os

os.environ["DDE_BACKEND"] = "pytorch"  # must be set before importing deepxde

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Normalized units: length in L0, time in T0 = L0 / c0, so the wave speed is 1.
L0 = 4.0             # characteristic length [m]
C0 = 299792458.0     # speed of light in vacuum [m/s]
T0 = L0 / C0         # characteristic time [s]

X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 0.8

# PML: zero conductivity inside [PML_X0, PML_X1] x [PML_Y0, PML_Y1], growing as
# SIGMA_MAX * (depth / PML_THICKNESS)**PML_ORDER into the layer.
PML_X0, PML_X1 = 0.2, 0.8
PML_Y0, PML_Y1 = 0.2, 0.8
PML_THICKNESS = 0.2
SIGMA_MAX = 10.0
PML_ORDER = 3

# Initial pulse: Ez(x, y, 0) = exp(-PULSE_ALPHA * ((x - x0)^2 + (y - y0)^2))
PULSE_X0, PULSE_Y0 = 0.5, 0.5
PULSE_ALPHA = 40.0

# Grid used for the t = 0 pulse constraint and for the output plots
NX, NY = 141, 141
NT_PLOT = 21

# Network and training
LAYERS = [3] + [80] * 4 + [4]
NUM_DOMAIN = 100000
NUM_BOUNDARY = 1000
NUM_INITIAL = 1000
ADAM_LR = 2e-3
ADAM_ITERATIONS = 10000
LBFGS_MAXITER = 100000
EZX_IC_WEIGHT = 10.0  # extra weight on the Gaussian initial condition

# Output component indices
EZX, HX, HY, EZY = 0, 1, 2, 3


# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
def pml_sigma(s, s0, s1):
    """Graded PML conductivity along one axis: 0 for s0 <= s <= s1."""
    depth = torch.clamp(s0 - s, min=0.0) + torch.clamp(s - s1, min=0.0)
    return SIGMA_MAX * (depth / PML_THICKNESS) ** PML_ORDER


def pde(x, y):
    """Residuals of the normalized split-field TMz Maxwell equations.

        dEzx/dt + sx*Ezx =  dHy/dx
        dEzy/dt + sy*Ezy = -dHx/dy
        dHx/dt  + sy*Hx  = -dEz/dy
        dHy/dt  + sx*Hy  =  dEz/dx
    """
    def d(component, wrt):
        return dde.grad.jacobian(y, x, i=component, j=wrt)

    ezx, hx, hy, ezy = y[:, EZX:EZX + 1], y[:, HX:HX + 1], y[:, HY:HY + 1], y[:, EZY:EZY + 1]
    sx = pml_sigma(x[:, 0:1], PML_X0, PML_X1)
    sy = pml_sigma(x[:, 1:2], PML_Y0, PML_Y1)

    ez_x = d(EZX, 0) + d(EZY, 0)
    ez_y = d(EZX, 1) + d(EZY, 1)

    f_ezx = d(EZX, 2) - d(HY, 0) + sx * ezx
    f_hx = d(HX, 2) + ez_y + sy * hx
    f_hy = d(HY, 2) - ez_x + sx * hy
    f_ezy = d(EZY, 2) + d(HX, 1) + sy * ezy
    return [f_ezx, f_hx, f_hy, f_ezy]


def gaussian_pulse(xyt):
    return np.exp(-PULSE_ALPHA * ((xyt[:, 0:1] - PULSE_X0) ** 2 + (xyt[:, 1:2] - PULSE_Y0) ** 2))


def zero(x):
    return np.zeros((len(x), 1))


def on_edge(dim, value):
    return lambda x, on_boundary: on_boundary and np.isclose(x[dim], value)


# --------------------------------------------------------------------------
# Problem setup
# --------------------------------------------------------------------------
def build_data():
    """Returns the DeepXDE TimePDE problem and the matching loss weights."""
    space = dde.geometry.Rectangle([X_MIN, Y_MIN], [X_MAX, Y_MAX])
    time = dde.geometry.TimeDomain(T_MIN, T_MAX)
    geomtime = dde.geometry.GeometryXTime(space, time)

    # PEC outer wall: Ez = Ezx + Ezy = 0, imposed as Ezx = 0 and Ezy = 0 on each edge.
    edges = [(0, X_MIN), (0, X_MAX), (1, Y_MIN), (1, Y_MAX)]
    pec_bcs = [
        dde.icbc.DirichletBC(geomtime, zero, on_edge(dim, value), component=c)
        for c in (EZX, EZY)
        for dim, value in edges
    ]

    # t = 0: H = 0 at randomly sampled initial points ...
    on_initial = lambda _, on_initial: on_initial
    ic_h = [dde.icbc.IC(geomtime, zero, on_initial, component=c) for c in (HX, HY)]

    # ... and the Gaussian pulse put entirely into Ezx (Ezy = 0) on a fixed grid.
    gx, gy = np.meshgrid(np.linspace(X_MIN, X_MAX, NX), np.linspace(Y_MIN, Y_MAX, NY))
    grid0 = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, T_MIN)])
    ezx0 = gaussian_pulse(grid0)
    ic_ezx = dde.icbc.PointSetBC(grid0, ezx0, component=EZX)
    ic_ezy = dde.icbc.PointSetBC(grid0, np.zeros_like(ezx0), component=EZY)

    conditions = pec_bcs + ic_h + [ic_ezx, ic_ezy]
    # Loss order: 4 PDE residuals, then the conditions in the order listed above.
    loss_weights = [1.0] * 4 + [1.0] * (len(pec_bcs) + len(ic_h)) + [EZX_IC_WEIGHT, 1.0]

    data = dde.data.TimePDE(
        geomtime,
        pde,
        conditions,
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        num_initial=NUM_INITIAL,
        train_distribution="Sobol",
    )
    return data, loss_weights


def train(data, loss_weights):
    net = dde.nn.FNN(LAYERS, "sin", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=ADAM_LR, loss="MSE", loss_weights=loss_weights)
    model.train(iterations=ADAM_ITERATIONS, display_every=1000)

    # Fine-tune with L-BFGS
    dde.optimizers.config.set_LBFGS_options(
        maxcor=50,
        ftol=np.finfo(float).eps,
        gtol=1e-10,
        maxiter=LBFGS_MAXITER,
        maxfun=LBFGS_MAXITER,
        maxls=50,
    )
    model.compile("L-BFGS", loss_weights=loss_weights)
    model.train()
    return model


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def save_snapshots(model):
    """Saves Ez = Ezx + Ezy on the plot grid at NT_PLOT times (slice_####.npz/.png)."""
    X, Y = np.meshgrid(np.linspace(X_MIN, X_MAX, NX), np.linspace(Y_MIN, Y_MAX, NY), indexing="ij")

    for k, tk in enumerate(np.linspace(T_MIN, T_MAX, NT_PLOT)):
        xyt = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, tk)])
        pred = model.predict(xyt)
        Ez = (pred[:, EZX] + pred[:, EZY]).reshape(NX, NY)

        np.savez(f"slice_{k:04d}.npz", x=X, y=Y, E=Ez, t=tk)

        plt.figure(figsize=(6, 5))
        plt.contourf(X, Y, Ez, 100, cmap="viridis")
        plt.colorbar(label="Ez")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Ez(x, y) at t = {tk:.3f}")
        plt.tight_layout()
        plt.savefig(f"slice_{k:04d}.png", dpi=120)
        plt.show()


def main():
    print(f"DeepXDE backend: {dde.backend.backend_name}")
    print(f"Domain: x in [{X_MIN}, {X_MAX}], y in [{Y_MIN}, {Y_MAX}], t in [{T_MIN}, {T_MAX}]")
    print(f"L0 = {L0} m, T0 = {T0:.4e} s")

    data, loss_weights = build_data()
    model = train(data, loss_weights)
    save_snapshots(model)


if __name__ == "__main__":
    main()

# 2D Maxwell's equations with a split-field PML, solved by a PINN

A physics-informed neural network (PINN) that solves the time-domain 2D Maxwell
equations (TMz polarization) in an open region. The region is truncated by a
Bérenger split-field perfectly matched layer (PML). It is built with
[DeepXDE](https://github.com/lululxvi/deepxde) on the PyTorch backend.

The network is not trained on simulation data. It learns the field only from the
PDE residual and the boundary and initial conditions.

## Problem setup

- A Gaussian Ez pulse starts at the center of the unit square and radiates outward.
- The physical region is $[0.2, 0.8]^2$. It is surrounded by a PML that is 0.2 thick.
- The outer boundary is a perfect electric conductor (PEC), where $E_z = 0$.
- The simulation covers $t \in [0, 0.8]$.

Everything is normalized. Lengths are in units of $L_0 = 4\,\text{m}$ and times in
units of $T_0 = L_0 / c_0$, so the wave speed is 1.

## Formulation

The network maps $(x, y, t) \mapsto (E_{zx}, H_x, H_y, E_{zy})$. The physical
field is $E_z = E_{zx} + E_{zy}$. The split-field equations are:

$$
\begin{aligned}
\partial_t E_{zx} + \sigma_x E_{zx} &= \partial_x H_y \\
\partial_t E_{zy} + \sigma_y E_{zy} &= -\partial_y H_x \\
\partial_t H_x + \sigma_y H_x &= -\partial_y E_z \\
\partial_t H_y + \sigma_x H_y &= \partial_x E_z
\end{aligned}
$$

The conductivity is zero in the physical region. Inside the PML it follows a cubic
profile, $\sigma(d) = \sigma_{\max}\,(d/0.2)^3$ with $\sigma_{\max} = 10$, where
$d$ is the depth into the layer. The electric and magnetic conductivities are equal
(the matched condition), so in theory the interface between the region and the
PML does not reflect.

### Loss terms

| Term | Collocation points | Weight |
|---|---|---|
| 4 PDE residuals | 100,000 Sobol points in space-time | 1 |
| PEC: $E_{zx} = E_{zy} = 0$ on each of the 4 edges (8 terms) | boundary samples | 1 |
| $H_x = H_y = 0$ at $t = 0$ | 1,000 initial samples | 1 |
| $E_{zx} = e^{-40 r^2}$ at $t = 0$ | 141 × 141 grid | 10 |
| $E_{zy} = 0$ at $t = 0$ | 141 × 141 grid | 1 |

Here $r$ is the distance from the domain center. All of the initial field is placed
in $E_{zx}$. Any split works, because only the sum is physical and the
conductivity is zero where the pulse starts.

### Network and training

- Fully connected network, 3 → 4 × 80 → 4, with `sin` activation.
- 10,000 Adam iterations (lr 2e-3), then L-BFGS until convergence.

## Running

```bash
pip install -r requirements.txt
python Maxwell2DSplitPML.py
```

The script writes $E_z$ snapshots at 21 times as `slice_XXXX.npz` (arrays `x`,
`y`, `E`, `t`) and `slice_XXXX.png`. All settings are constants at the top of the
script. Training uses 100k collocation points, so a GPU is recommended.

## Limitations and next steps

- **PML strength.** With $\sigma_{\max} = 10$, the theoretical round-trip
  reflection at normal incidence is $R = e^{-2\sigma_{\max} d/(n+1)} \approx 0.37$.
  Lowering it to about $10^{-3}$ needs $\sigma_{\max} \approx 70$. That makes the
  PDE stiffer and the PINN harder to train. Exploring this trade-off, for example
  with adaptive weights or by ramping $\sigma$ up gradually, is the next step.
- **Validation.** Compare the PINN solution against a finite-difference
  time-domain (FDTD) reference solved on the same grid.
- **Long-time training.** Plain PINNs often have trouble keeping wave propagation
  accurate over long times. Causal training or time-marching windows could
  extend the simulated time.

# Sliced Wasserstein Flow Lab

An interactive **Streamlit** experiment for empirical gradient descent of a finite-direction sliced-Wasserstein energy in two dimensions. The initial and target measures independently support restricted LaTeX Lebesgue densities (optionally restricted to a geometric region) and the target can also be a singular Hausdorff measure.

## Quick start (VS Code / Windows)

Open this folder in VS Code, then in its integrated terminal:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

If PowerShell blocks activation, skip it and run `.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `.venv\Scripts\python.exe -m streamlit run app.py`.

On macOS/Linux, use `python3 -m venv .venv`, `source .venv/bin/activate`, then the last three commands.

## Default experiment

- Initial: `\frac{1}{2\pi}e^{-\frac{x^2+y^2}{2}}` (standard Gaussian in 2D).
- Target: `\frac{1}{2\pi}e^{-\frac{(x-2)^2+(y-1)^2}{2}}` (translated Gaussian).
- 1,000 particles per measure, 384 evenly spaced projection directions in `[0,π)`.
- Euler step `dt=0.4`, 70 iterations, fixed target samples and fixed directions.

## LaTeX expressions (level 2)

The safe, deliberately limited parser accepts: `x`, `y`, `e`, `\pi`, decimal numbers, `+ - * /`, implicit multiplication, `^`, `\frac{a}{b}`, `\exp(...)`, `\max(a,b)`, parentheses and braces. Optional `\left` / `\right` are accepted. Enter the expression without `$` or `\rho(x,y)=`.

Examples:

| Distribution | Formula |
| --- | --- |
| Centered Gaussian | `\frac{1}{2\pi}e^{-\frac{x^2+y^2}{2}}` |
| Shifted Gaussian | `\frac{1}{2\pi}e^{-\frac{(x-2)^2+(y-1)^2}{2}}` |
| Gaussian mixture (unnormalized) | `0.5e^{-\frac{(x+2)^2+y^2}{2}}+0.5e^{-\frac{(x-2)^2+y^2}{2}}` |
| Compact support | `\max(0,1-x^2-y^2)` |

**Normalization is numerical on the selected rectangular domain.** The input does not have to integrate to one beforehand, but must be nonnegative and have positive mass there. A small box may truncate substantial probability mass. Numerical checks on a grid do not certify global positivity or integrability on all of `R²`.

No `eval`, `exec`, or arbitrary Python expressions are used. This is not a general LaTeX parser. Formula length is limited, but for public deployment additional server-side resource controls, request limits and performance tests are recommended.

## Densities restricted to geometric regions

Either Lebesgue-density formula may end with one characteristic-function factor:
`\mathbf{1}_A`, `\mathbb{1}_A`, or `\chi_A` (braced subscripts such as
`\mathbf{1}_{A}` are accepted too). Alternatively, enable **Restrict density to
a region A** for an ordinary formula. Choose a disk, rectangle, annulus,
ellipse, or clipped half-plane `x ≥ a`; initial and target regions are
configured independently. For each measure the app computes and displays its
own numerical normalizer on the computational box, intersected with its
selected region. Zero-mass, negative, or nonfinite values on the selected
support are rejected. Sampling uses grid-cell quadrature and rejection within
boundary cells, so masses near curved boundaries are resolution-dependent.

## Singular targets and visualization

The target may instead be normalized Hausdorff measure on a circle or segment
(`H¹`) or a finite set of distinct points (`H⁰`). Smooth 2D heatmaps and 3D
surfaces are visualization-only KDEs of the evolving particles; they do not
turn a singular target into a Lebesgue density. Particle plots retain their
animation controls, and selected region boundaries and exact singular support
are overlaid in 2D views.

## Mathematical model

Let `μᴺ = (1/N) Σ δₓᵢ`, and let `νᴺ` be the target empirical measure with the same number of equally weighted particles. For fixed directions `θ₁,…,θₘ`, the simulated energy is

```text
F_M(X) = (1 / 2M) Σ_k W₂²((P_θk)# μᴺ, (P_θk)# νᴺ).
```

For each direction, sort projected source and target samples. If `q_k(i)` is the target projection with the same rank as source particle `i`, the velocity is

```text
v_i = (1/M) Σ_k [q_k(i) − ⟨x_i, θ_k⟩] θ_k.
```

Then update `x_i ← x_i + dt v_i`. This is the Wasserstein-gradient scaling of `F_M` on equally weighted empirical measures. There is **no extra factor of 2** in dimension 2. Fixed equally spaced angles approximate normalized uniform spherical integration (antipodal directions give identical squared 1D transport costs).

The implementation evaluates the same discrete energy at each step and displays its numerical evolution. It does **not** establish convergence of the numerical scheme to the continuum PDE. A time step that is too large can undermine numerical stability.

## Repository structure

- `app.py` — Streamlit interface, Plotly animation, region and target selection, visualizations and energy plot.
- `density.py` — restricted LaTeX parser, grid normalization and sampling.
- `region.py` — validated geometric masks and boundary curves.
- `target_measure.py` — normalized singular Hausdorff targets and sampling.
- `smooth.py` — visualization-only smoothed empirical density (SciPy).
- `flow.py` — finite-direction sliced-Wasserstein particle evolution.
- `tests/` — basic parser, sampling and energy regression tests.

## Publishing a demo

Push this project to a public GitHub repository, then deploy via [Streamlit Community Cloud](https://streamlit.io/cloud) by selecting the repository and `app.py`. Check resource consumption before allowing many simultaneous users. **GitHub Pages alone cannot run a Streamlit Python backend.**

## Limitations and possible extensions

- Monte Carlo particle approximation; grid-based sampling; finite projection quadrature.
- Equal-weight source and target samples with the same particle count.
- Density normalization is on a bounded computational box and selected geometric region.
- No guarantee of continuum convergence or of physical-time accuracy.
- Future work: adaptive domains, arbitrary weighted particles, quantitative error analysis, comparison to analytically tractable cases, and exportable simulations.

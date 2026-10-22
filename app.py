"""Run with: streamlit run app.py"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from density import COMPACT, INITIAL, MIXTURE, TARGET, DensitySyntaxError, grid_density, sample_density, split_indicator
from flow import simulate
from region import Region
from smooth import smooth_density
from target_measure import SingularTarget, parse_points

st.set_page_config(page_title="Sliced Wasserstein Flow Lab", layout="wide")
st.title("Sliced Wasserstein Flow Lab")
st.caption("A numerical experiment: empirical W₂ gradient flow of a finite-direction sliced-Wasserstein energy in ℝ².")

PRESETS = {
    "Centered Gaussian": INITIAL,
    "Translated Gaussian": TARGET,
    "Gaussian mixture": MIXTURE,
    "Compactly supported bump": COMPACT,
}
REGION_SHAPES = ["Disk", "Rectangle", "Annulus", "Ellipse", "Half-plane x ≥ a"]


def density_input(title: str, default: str, key: str) -> str:
    st.subheader(title)
    selected = st.selectbox("Example", ["Custom LaTeX", *PRESETS], index=0, key=f"{key}_preset")
    if selected != "Custom LaTeX" and st.session_state.get(f"{key}_last_preset") != selected:
        st.session_state[f"{key}_latex"] = PRESETS[selected]
    st.session_state[f"{key}_last_preset"] = selected
    formula = st.text_area("LaTeX density (without $ signs)", value=default, key=f"{key}_latex", height=100)
    try:
        st.latex(r"\rho(x,y)=" + formula)
    except Exception:
        st.warning("LaTeX preview could not be rendered")
    return formula


def region_input(key: str) -> Region:
    st.caption("Region A is specified geometrically; arbitrary set-valued LaTeX is not evaluated.")
    shape = st.selectbox("Region A", REGION_SHAPES, key=f"{key}_shape")
    if shape in {"Disk", "Annulus", "Ellipse"}:
        left, right = st.columns(2)
        with left:
            center_x = st.number_input("Center x", value=0.0, key=f"{key}_center_x")
        with right:
            center_y = st.number_input("Center y", value=0.0, key=f"{key}_center_y")
        if shape == "Ellipse":
            left, right = st.columns(2)
            with left:
                radius_x = st.number_input("Horizontal semiaxis", min_value=0.01, value=2.0,
                                           key=f"{key}_radius_x")
            with right:
                radius_y = st.number_input("Vertical semiaxis", min_value=0.01, value=1.0,
                                           key=f"{key}_radius_y")
            return Region(shape, center=(center_x, center_y), radii=(radius_x, radius_y))
        radius = st.number_input("Outer radius", min_value=0.01, value=2.0, key=f"{key}_radius")
        if shape == "Annulus":
            inner_radius = st.number_input("Inner radius", min_value=0.0, value=1.0,
                                           key=f"{key}_inner_radius")
            return Region(shape, center=(center_x, center_y), radius=radius, inner_radius=inner_radius)
        return Region(shape, center=(center_x, center_y), radius=radius)
    if shape == "Rectangle":
        left, right = st.columns(2)
        with left:
            xmin = st.number_input("Rectangle xmin", value=-2.0, key=f"{key}_rectangle_xmin")
            ymin = st.number_input("Rectangle ymin", value=-1.0, key=f"{key}_rectangle_ymin")
        with right:
            xmax = st.number_input("Rectangle xmax", value=2.0, key=f"{key}_rectangle_xmax")
            ymax = st.number_input("Rectangle ymax", value=1.0, key=f"{key}_rectangle_ymax")
        return Region(shape, rectangle=(xmin, xmax, ymin, ymax))
    threshold = st.number_input("Threshold a", value=0.0, key=f"{key}_threshold")
    return Region(shape, threshold=threshold)


def masked_formula_input(formula: str, key: str) -> Region | None:
    try:
        _, has_indicator = split_indicator(formula)
    except DensitySyntaxError:
        has_indicator = False
    use_region = st.checkbox(
        "Restrict density to a region A (multiply by 1_A)",
        value=has_indicator,
        key=f"{key}_masked",
        disabled=has_indicator,
    )
    if has_indicator:
        st.caption(r"Detected the trailing indicator $\mathbf{1}_A$ in this formula.")
    if not use_region and not has_indicator:
        return None
    return region_input(key)


with st.sidebar:
    st.header("Numerical settings")
    n = st.slider("Particles per distribution", 100, 2000, 1000, step=100)
    m = st.slider("Projection directions", 32, 512, 384, step=32)
    steps = st.slider("Euler steps", 10, 150, 70, step=10)
    dt = st.slider("Time step", 0.05, 1.0, 0.4, step=0.05)
    seed = st.number_input("Random seed", 0, 1_000_000, 42)
    st.header("Sampling domain")
    xmin, xmax = st.slider("x interval", -15.0, 15.0, (-6.0, 8.0), step=0.5)
    ymin, ymax = st.slider("y interval", -15.0, 15.0, (-6.0, 7.0), step=0.5)
    resolution = st.slider("Sampling grid resolution", 80, 320, 180, step=20)
    st.header("Smooth display")
    display_resolution = st.slider("Display grid resolution", 40, 140, 80, step=10)
    bandwidth = st.slider("Smoothing bandwidth", 0.08, 0.8, 0.25, step=0.01)

bounds = (xmin, xmax, ymin, ymax)
left, right = st.columns(2)
with left:
    initial_formula = density_input("Initial density", INITIAL, "initial")
    initial_region = masked_formula_input(initial_formula, "initial")
with right:
    st.subheader("Target measure")
    target_kind = st.radio("Measure type", ["Lebesgue density", "Hausdorff restriction"], key="target_kind")
    target_formula = None
    target_region = None
    singular = None
    if target_kind == "Lebesgue density":
        target_formula = density_input("Target density", TARGET, "target")
        target_region = masked_formula_input(target_formula, "target")
    else:
        st.latex(r"\nu=\frac{\mathcal H^k\!\restriction_A}{\mathcal H^k(A)}")
        st.caption("Choose a supported geometric set A; arbitrary set descriptions are not parsed.")
        geometry = st.selectbox("Target support A", ["Circle", "Segment", "Finite points"], key="singular_geometry")
        if geometry == "Circle":
            center_x = st.number_input("Circle center x", value=2.0, key="singular_center_x")
            center_y = st.number_input("Circle center y", value=1.0, key="singular_center_y")
            radius = st.number_input("Circle radius", min_value=0.01, value=1.5, key="singular_radius")
            singular = SingularTarget("Circle", center=(center_x, center_y), radius=radius)
        else:
            default_points = "0,0\n2,1" if geometry == "Segment" else "-2,0\n2,0\n0,2"
            point_text = st.text_area("Coordinates (one x,y per line)", default_points, key="singular_points")
            try:
                singular = SingularTarget(geometry, points=parse_points(point_text))
                singular.validate()
            except ValueError as exc:
                st.error(str(exc))
        if singular is not None:
            st.write(f"Hausdorff dimension k = {singular.dimension}; Hᵏ(A) = {singular.mass():.5g}")

st.info(
    "Both Lebesgue formulas are independently normalized on the computational box, optionally intersected "
    "with a selected region. Singular targets use normalized Hausdorff measure. Smooth plots are "
    "visualization-only approximations; the empirical flow solver and energy are unchanged."
)


@st.cache_data(show_spinner=False, max_entries=24)
def normalized_formula_preview(formula: str, domain: tuple[float, float, float, float],
                               grid_resolution: int, region: Region | None) -> float:
    return grid_density(formula, domain, grid_resolution, region).mass_before_normalization


def show_normalization(formula: str, label: str, region: Region | None) -> None:
    try:
        normalizer = normalized_formula_preview(formula, bounds, resolution, region)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        st.error(f"{label}: {exc}")
        return
    domain = r"\Omega\cap A" if region is not None else r"\Omega"
    numerator = r"f(x,y)\mathbf{1}_A(x,y)" if region is not None else r"f(x,y)"
    st.latex(
        r"\rho(x,y)=\frac{" + numerator + r"}{Z},\qquad "
        + rf"Z=\int_{{{domain}}} f\,dx\,dy\approx {normalizer:.7g}"
    )
    st.caption("Normalization is numerical on the selected box; it does not imply mass 1 on all of ℝ².")


show_normalization(initial_formula, "Initial density", initial_region)
if target_kind == "Lebesgue density" and target_formula is not None:
    show_normalization(target_formula, "Target density", target_region)


@st.cache_data(show_spinner=False, max_entries=6)
def run(initial: str, initial_mask: Region | None, selected_target_kind: str,
        selected_target_formula: str | None, selected_target_mask: Region | None,
        selected_singular: SingularTarget | None, domain: tuple[float, float, float, float],
        grid_resolution: int, particle_count: int, direction_count: int,
        step_size: float, step_count: int, random_seed: int):
    initial_grid = grid_density(initial, domain, grid_resolution, initial_mask)
    source = sample_density(initial_grid, particle_count, random_seed)
    if selected_target_kind == "Lebesgue density":
        if selected_target_formula is None:
            raise ValueError("Target density formula is missing")
        target_grid = grid_density(selected_target_formula, domain, grid_resolution, selected_target_mask)
        target_samples = sample_density(target_grid, particle_count, random_seed + 1)
    else:
        if selected_singular is None:
            raise ValueError("Target Hausdorff geometry is invalid")
        target_grid = None
        target_samples = selected_singular.sample(particle_count, random_seed + 1)
    result = simulate(source, target_samples, direction_count, step_size, step_count)
    return target_grid, target_samples, result


settings = (
    initial_formula, initial_region, target_kind, target_formula, target_region, singular,
    bounds, resolution, n, m, dt, steps, int(seed),
)
if st.button("▶ Run simulation", type="primary"):
    try:
        with st.spinner("Sampling measures and evolving particles…"):
            st.session_state["run_result"] = run(*settings)
            st.session_state["run_settings"] = settings
    except (ValueError, OverflowError, FloatingPointError) as exc:
        st.error(f"Cannot simulate these measures: {exc}")

if "run_result" in st.session_state:
    target_grid, target_samples, result = st.session_state["run_result"]
    old_settings = st.session_state["run_settings"]
    old_initial_region = old_settings[1]
    old_target_region = old_settings[4]
    old_singular = old_settings[5]
    old_bounds = old_settings[6]
    if old_settings != settings:
        st.warning("The displayed simulation uses previous settings. Click Run simulation to update it.")

    mode = st.radio("Visualization", ["Particles", "Smooth 2D", "Surface 3D", "Combined"],
                    horizontal=True, key="visualization_mode")
    frame = st.slider("Simulation frame", 0, len(result.times) - 1, 0, key="simulation_frame")
    st.caption(f"Time t = {result.times[frame]:.2f} | frame {frame}/{len(result.times) - 1}")

    def region_traces(region: Region | None, name: str, color: str) -> list[go.Scatter]:
        if region is None:
            return []
        return [
            go.Scatter(
                x=curve[:, 0], y=curve[:, 1], mode="lines", name=name,
                line=dict(color=color, width=2, dash="dash"),
            )
            for curve in region.boundary(old_bounds)
        ]

    def support_trace() -> go.Scatter | None:
        if old_singular is None:
            return None
        support = old_singular.support()
        return go.Scatter(
            x=support[:, 0], y=support[:, 1],
            mode="markers" if old_singular.dimension == 0 else "lines",
            name="Exact singular support A",
            marker=dict(size=10, color="#e87924"),
            line=dict(color="#e87924", width=4),
        )

    def particles_figure():
        overlays = []
        exact_support = support_trace()
        if exact_support is not None:
            overlays.append(exact_support)
        overlays.extend(region_traces(old_initial_region, "Initial region A", "#2374b9"))
        overlays.extend(region_traces(old_target_region, "Target region A", "#e87924"))

        def scatter(step: int):
            return [
                go.Scattergl(
                    x=result.positions[step, :, 0], y=result.positions[step, :, 1], mode="markers",
                    name="Evolving", marker=dict(size=5, opacity=0.65, color="#2878b5"),
                ),
                go.Scattergl(
                    x=target_samples[:, 0], y=target_samples[:, 1], mode="markers",
                    name="Target samples", marker=dict(size=4, opacity=0.28, color="#d77927"),
                ),
                *overlays,
            ]

        plotted_points = np.vstack((result.positions.reshape(-1, 2), target_samples))
        x_padding = max(0.5, np.ptp(plotted_points[:, 0]) * 0.07)
        y_padding = max(0.5, np.ptp(plotted_points[:, 1]) * 0.07)
        x_range = [float(plotted_points[:, 0].min() - x_padding),
                   float(plotted_points[:, 0].max() + x_padding)]
        y_range = [float(plotted_points[:, 1].min() - y_padding),
                   float(plotted_points[:, 1].max() + y_padding)]
        frame_indices = sorted(set(np.linspace(0, len(result.times) - 1,
                                               min(41, len(result.times)), dtype=int)))
        active_frame = min(range(len(frame_indices)), key=lambda index: abs(frame_indices[index] - frame))
        figure = go.Figure(
            data=scatter(frame),
            frames=[go.Frame(data=scatter(index)[:2], name=str(index), traces=[0, 1])
                    for index in frame_indices],
        )
        figure.update_layout(
            xaxis=dict(range=x_range, title="x", scaleanchor="y", scaleratio=1),
            yaxis=dict(range=y_range, title="y"), height=570,
            margin=dict(l=25, r=20, t=20, b=25),
            legend=dict(orientation="h", y=1.08),
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.0, y=-0.13,
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 95, "redraw": True},
                                      "transition": {"duration": 0}, "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                active=active_frame, y=-0.09, pad={"t": 20}, currentvalue={"prefix": "Step "},
                steps=[dict(method="animate", label=str(index), args=[[str(index)],
                       {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0}}]) for index in frame_indices],
            )],
        )
        return figure

    @st.cache_data(show_spinner=False, max_entries=80)
    def get_smooth(points: np.ndarray, domain: tuple[float, float, float, float],
                   grid_resolution: int, smooth_bandwidth: float):
        return smooth_density(points, domain, grid_resolution, smooth_bandwidth)

    def smooth_figures():
        x_grid, y_grid, values = get_smooth(
            result.positions[frame], old_bounds, display_resolution, bandwidth
        )
        figure_2d = go.Figure(go.Heatmap(
            x=x_grid, y=y_grid, z=values, colorscale="Viridis", colorbar=dict(title="ρ KDE")
        ))
        exact_support = support_trace()
        if exact_support is not None:
            figure_2d.add_trace(exact_support)
        elif target_grid is not None:
            figure_2d.add_trace(go.Contour(
                x=target_grid.x, y=target_grid.y, z=target_grid.values,
                contours=dict(coloring="none", showlabels=False),
                line=dict(color="#ff9234", width=2), showscale=False, name="Target contours",
            ))
        for trace in region_traces(old_initial_region, "Initial region A", "#2374b9"):
            figure_2d.add_trace(trace)
        for trace in region_traces(old_target_region, "Target region A", "#e87924"):
            figure_2d.add_trace(trace)
        figure_2d.update_layout(
            xaxis=dict(range=old_bounds[:2], title="x", scaleanchor="y", scaleratio=1),
            yaxis=dict(range=old_bounds[2:], title="y"), height=480,
        )
        figure_3d = go.Figure(go.Surface(
            x=x_grid, y=y_grid, z=values, colorscale="Viridis", colorbar=dict(title="ρ KDE")
        ))
        figure_3d.update_layout(
            scene=dict(
                xaxis=dict(title="x", range=old_bounds[:2]),
                yaxis=dict(title="y", range=old_bounds[2:]),
                zaxis=dict(title="ρ KDE"),
            ),
            height=480,
        )
        return figure_2d, figure_3d

    if mode == "Particles":
        st.plotly_chart(particles_figure(), use_container_width=True)
    elif mode in {"Smooth 2D", "Surface 3D"}:
        figure_2d, figure_3d = smooth_figures()
        st.plotly_chart(figure_2d if mode == "Smooth 2D" else figure_3d, use_container_width=True)
    else:
        left, right = st.columns(2)
        figure_2d, figure_3d = smooth_figures()
        with left:
            st.plotly_chart(figure_2d, use_container_width=True)
        with right:
            st.plotly_chart(figure_3d, use_container_width=True)
        st.plotly_chart(particles_figure(), use_container_width=True)

    if old_singular is not None:
        st.caption("The exact singular support is shown in orange. Smooth views display a KDE of the "
                   "evolving particles, not a Lebesgue density of the singular target.")
    energy_figure = go.Figure(go.Scatter(
        x=result.times, y=result.energies, mode="lines+markers", name="Finite-direction energy"
    ))
    energy_figure.add_vline(x=float(result.times[frame]), line_dash="dash", line_color="gray")
    energy_figure.update_layout(
        xaxis_title="Time", yaxis_title="½ SW² (finite directions)", height=320,
        margin=dict(l=30, r=20, t=20, b=30),
    )
    st.plotly_chart(energy_figure, use_container_width=True)
    initial_metric, final_metric, increase_metric = st.columns(3)
    initial_metric.metric("Initial energy", f"{result.energies[0]:.5f}")
    final_metric.metric("Final energy", f"{result.energies[-1]:.5f}")
    increase_metric.metric(
        "Maximum upward energy step", f"{max(0.0, float(np.diff(result.energies).max())):.2e}"
    )
    st.caption("Energy uses the same fixed directions and target samples as the gradient. "
               "Sampling, finite projections, Euler time discretization and display smoothing "
               "introduce approximation errors.")
else:
    st.write("Choose the initial density and target measure, then click **Run simulation**.")

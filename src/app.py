"""Interactive web UI for landcover visualization.

Usage:
    uv run streamlit run src/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
from folium.raster_layers import ImageOverlay
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium

from src import config

# Page config
st.set_page_config(
    page_title="OlmoEarth UK Landcover Demo",
    page_icon="🌍",
    layout="wide",
)


def check_outputs_exist() -> bool:
    """Check if pipeline outputs exist."""
    required = [
        config.OUTPUT_DIR / "landcover_2021.tif",
        config.OUTPUT_DIR / "landcover_2023.tif",
        config.OUTPUT_DIR / "change_map.tif",
        config.OUTPUT_DIR / "transitions.json",
    ]
    return all(p.exists() for p in required)


def check_experiments_exist(session_dir: Path | None = None) -> bool:
    """Check if any experiment iterations exist."""
    experiments_dir = session_dir or config.EXPERIMENTS_DIR
    if not experiments_dir.exists():
        return False
    return any(
        d.is_dir() and d.name.startswith("iteration_")
        for d in experiments_dir.iterdir()
    )


def _list_iterations_from_dir(session_dir: Path) -> list[dict]:
    """List iterations from a specific session directory (read-only, for UI).

    This is a thin helper for reading iteration data from non-latest sessions
    without modifying the list_iterations() function's global state.
    """
    if not session_dir.exists():
        return []
    iterations = []
    for entry in sorted(session_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("iteration_"):
            continue
        suffix = entry.name[len("iteration_"):]
        try:
            num = int(suffix)
        except ValueError:
            continue
        metadata_path = entry / "metadata.json"
        if not metadata_path.exists():
            continue
        with open(metadata_path) as f:
            metadata = json.load(f)
        item = {
            "iteration": num,
            "timestamp": metadata.get("timestamp", ""),
            "status": metadata.get("status", "unknown"),
        }
        metrics_path = entry / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            item["overall_accuracy"] = metrics.get("overall_accuracy")
        else:
            item["overall_accuracy"] = None
        iterations.append(item)
    iterations.sort(key=lambda x: x["iteration"])
    return iterations


def _render_compare_sessions(sessions: list[dict]) -> None:
    """Render the Compare Sessions table."""
    import pandas as pd

    if len(sessions) < 2:
        return

    with st.expander("Compare Sessions", expanded=False):
        rows = []
        for s in sessions:
            # Count accepted changes (exclude baseline iteration 1)
            accepted_count = 0
            session_path = s.get("path")
            if session_path and session_path.exists():
                for entry in session_path.iterdir():
                    if not entry.is_dir() or not entry.name.startswith("iteration_"):
                        continue
                    # Extract iteration number
                    suffix = entry.name[len("iteration_"):]
                    try:
                        iter_num = int(suffix)
                    except ValueError:
                        continue
                    if iter_num <= 1:
                        continue  # Skip baseline
                    meta_path = entry / "metadata.json"
                    if meta_path.exists():
                        with open(meta_path) as f:
                            meta = json.load(f)
                        if meta.get("status") == "accepted":
                            accepted_count += 1

            rows.append({
                "Session": s["start_time"][:19] if s["start_time"] else s["session_id"],
                "Iterations": s["n_iterations"],
                "Best Score": f"{s['final_score']:.2f}" if s["final_score"] else "N/A",
                "Accepted Changes": accepted_count,
                "Stop Reason": s.get("stop_reason") or "N/A",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)


def load_iteration_details(iteration_num: int, session_dir: Path | None = None) -> dict:
    """Load all available data for a single experiment iteration.

    Args:
        iteration_num: The iteration number to load.
        session_dir: Session directory to load from. Defaults to config.EXPERIMENTS_DIR.

    Returns dict with keys: metadata, metrics, config, hypothesis,
    evaluations, images, config_diff, metrics_diff.
    Missing optional files are returned as None.

    IMPORTANT: hypothesis.json is stored in the iteration that was
    *diagnosed* (analyzed), which is the PREVIOUS iteration (N-1).
    So for iteration N, we read hypothesis.json from iteration N-1.
    """
    from src.experiment import load_experiment, compare_experiments

    experiments_dir = session_dir or config.EXPERIMENTS_DIR

    # Load core data (config, metrics, metadata) - always present
    experiment = load_experiment(iteration_num, session_dir=experiments_dir)

    iteration_dir = experiments_dir / f"iteration_{iteration_num:03d}"

    # Load hypothesis from the PREVIOUS iteration's directory.
    # hypothesis.json in iteration N-1 contains the hypothesis that was
    # applied to create iteration N. For iteration 1 (baseline), there
    # is no previous iteration and no hypothesis.
    hypothesis = None
    if iteration_num > 1:
        prev_dir = experiments_dir / f"iteration_{iteration_num - 1:03d}"
        hypothesis_path = prev_dir / "hypothesis.json"
        if hypothesis_path.exists():
            with open(hypothesis_path) as f:
                hypothesis = json.load(f)

    # Load optional evaluation JSONs
    evaluations = {}
    for year in ["2021", "2023"]:
        eval_path = iteration_dir / f"evaluation_{year}.json"
        if eval_path.exists():
            with open(eval_path) as f:
                evaluations[year] = json.load(f)
        else:
            evaluations[year] = None

    # Collect comparison image paths
    images = {}
    for year in ["2021", "2023"]:
        for view in ["full", "nw", "ne", "sw", "se"]:
            img_path = iteration_dir / f"comparison_{year}_{view}.png"
            images[f"{year}_{view}"] = img_path if img_path.exists() else None

    # Config and metrics diff vs previous iteration
    config_diff = None
    metrics_diff = None
    if iteration_num > 1:
        try:
            comparison = compare_experiments(
                iteration_num - 1, iteration_num, session_dir=experiments_dir
            )
            config_diff = comparison.get("config_diff")
            metrics_diff = comparison.get("metrics_diff")
        except FileNotFoundError:
            pass  # Previous iteration data missing - skip diff

    return {
        "metadata": experiment["metadata"],
        "metrics": experiment["metrics"],
        "config": experiment["config"],
        "hypothesis": hypothesis,
        "evaluations": evaluations,
        "images": images,
        "config_diff": config_diff,
        "metrics_diff": metrics_diff,
    }


def _render_hypothesis_callout(hypothesis: dict | None, iteration_num: int) -> None:
    """Render hypothesis as a prominent callout box at top of iteration card.

    parameter_changes are intentionally omitted here because the config diff
    section (rendered immediately below) shows the same parameters with
    before/after context, which is more informative.
    """
    if hypothesis:
        component = hypothesis.get("component", "N/A")
        text = hypothesis.get("hypothesis", "")
        expected_impact = hypothesis.get("expected_impact")
        confidence = hypothesis.get("confidence")

        # Build callout content
        lines = [f"**Component:** `{component}`"]
        if isinstance(confidence, (int, float)):
            lines.append(f"**Confidence:** {confidence:.0%}")
        lines.append("")  # blank line before hypothesis text
        lines.append(text)
        if expected_impact:
            lines.append("")
            lines.append(f"**Expected impact:** {expected_impact}")

        st.info("\n".join(lines))
    elif iteration_num == 1:
        st.info("**Baseline** \u2014 default configuration")
    else:
        # Iteration > 1 but no hypothesis (legacy/missing data)
        st.warning("No hypothesis recorded for this iteration.")


def _render_config_diff_inline(config_diff: dict | None, iteration_num: int) -> None:
    """Render config changes as compact inline before/after values."""
    if config_diff:
        st.markdown("**Config Changes:**")
        for param, vals in config_diff.items():
            old_val = vals.get("a", "?")
            new_val = vals.get("b", "?")
            st.markdown(f"- `{param}`: {old_val} \u2192 {new_val}")
    elif iteration_num > 1:
        st.caption("No configuration changes from previous iteration.")
    elif iteration_num == 1:
        st.caption("Baseline \u2014 no prior config to compare.")


def _render_gemini_summary(evaluations: dict) -> None:
    """Render Gemini overall assessment and key recommendations prominently."""
    has_any = False
    for year in ["2021", "2023"]:
        eval_data = evaluations.get(year)
        if not eval_data:
            continue
        evaluation = eval_data.get("evaluation", {})
        overall_score = evaluation.get("overall_score")
        spatial_quality = evaluation.get("spatial_quality")
        recommendations = evaluation.get("recommendations", [])

        if overall_score is None and not recommendations:
            continue

        has_any = True
        st.markdown(f"**Gemini Assessment \u2014 {year}:**")
        parts = []
        if overall_score is not None:
            parts.append(f"Score: **{overall_score}/10**")
        if spatial_quality:
            parts.append(f"Spatial quality: {spatial_quality}")
        if parts:
            st.markdown(" \u00b7 ".join(parts))

        if recommendations:
            st.markdown("Key recommendations:")
            for rec in recommendations:
                st.markdown(f"- {rec}")

    if not has_any:
        st.caption("No Gemini evaluation available.")


def render_iteration_card(data: dict, is_best: bool) -> None:
    """Render one iteration as a collapsible expander with full details."""
    import pandas as pd

    metadata = data["metadata"]
    metrics = data["metrics"]
    iteration_num = metadata["iteration"]
    status = metadata["status"]
    timestamp = metadata.get("timestamp", "")

    # Build expander label with status badge
    status_icon = {"accepted": "\u2705", "reverted": "\u274c", "pending": "\u23f3"}.get(status, "\u2753")
    best_tag = " \u2b50 BEST" if is_best else ""
    label = f"Iteration {iteration_num:03d} \u2014 {status_icon} {status}{best_tag}"

    with st.expander(label, expanded=is_best):
        # ---------------------------------------------------------------
        # Section 1: Header metrics row (UNCHANGED)
        # ---------------------------------------------------------------
        cols = st.columns(4)
        with cols[0]:
            st.metric("Status", status.capitalize())
        with cols[1]:
            acc = metrics.get("overall_accuracy")
            st.metric("Accuracy", f"{acc:.4f}" if acc is not None else "N/A")
        with cols[2]:
            eval_data = data["evaluations"].get("2021")
            if eval_data:
                gemini_score = eval_data.get("evaluation", {}).get("overall_score")
                st.metric(
                    "Gemini Score",
                    f"{gemini_score}/10" if gemini_score is not None else "N/A",
                )
            else:
                st.metric("Gemini Score", "N/A")
        with cols[3]:
            st.metric("Timestamp", timestamp[:10] if timestamp else "N/A")

        # ---------------------------------------------------------------
        # Section 2: Score delta vs previous (UNCHANGED)
        # ---------------------------------------------------------------
        if data["metrics_diff"]:
            acc_diff = data["metrics_diff"]["overall_accuracy"]
            delta = acc_diff["delta"]
            delta_color = "green" if delta > 0 else ("red" if delta < 0 else "gray")
            st.markdown(
                f"**Accuracy change:** {acc_diff['a']:.4f} \u2192 {acc_diff['b']:.4f} "
                f"(<span style='color:{delta_color}'>{delta:+.4f}</span>)",
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------
        # Section 3: Hypothesis callout (MOVED UP, NEW FORMAT)
        # ---------------------------------------------------------------
        _render_hypothesis_callout(data["hypothesis"], iteration_num)

        # ---------------------------------------------------------------
        # Section 4: Config diff inline (MOVED UP, NEW FORMAT)
        # ---------------------------------------------------------------
        _render_config_diff_inline(data.get("config_diff"), iteration_num)

        # ---------------------------------------------------------------
        # Section 5: Gemini feedback summary (MOVED UP, NEW FORMAT)
        # ---------------------------------------------------------------
        _render_gemini_summary(data["evaluations"])

        # ---------------------------------------------------------------
        # Section 6: Per-class F1 scores (MOVED DOWN, in sub-expander)
        # ---------------------------------------------------------------
        per_class = metrics.get("per_class", {})
        if per_class:
            with st.expander("Per-Class F1 Scores", expanded=False):
                f1_data = {
                    name: scores.get("f1", 0)
                    for name, scores in per_class.items()
                }
                df_f1 = pd.DataFrame({
                    "Class": list(f1_data.keys()),
                    "F1 Score": list(f1_data.values()),
                })
                st.bar_chart(df_f1, x="Class", y="F1 Score")

        # ---------------------------------------------------------------
        # Section 7: Gemini detailed evaluation (MOVED DOWN, in sub-expander)
        # Recommendations shown in summary above (Section 5)
        # ---------------------------------------------------------------
        for year in ["2021", "2023"]:
            eval_data = data["evaluations"].get(year)
            if not eval_data:
                continue
            evaluation = eval_data.get("evaluation", {})
            eval_per_class = evaluation.get("per_class", [])
            error_regions = evaluation.get("error_regions", [])

            if eval_per_class or error_regions:
                with st.expander(f"Gemini Details \u2014 {year}", expanded=False):
                    if eval_per_class:
                        st.markdown("**Per-class scores:**")
                        for item in eval_per_class:
                            st.markdown(
                                f"- **{item.get('class_name', '?')}** "
                                f"({item.get('score', '?')}/10): "
                                f"{item.get('notes', '')}"
                            )
                    if error_regions:
                        st.markdown("**Error Regions:**")
                        for region in error_regions:
                            severity = region.get("severity", "unknown")
                            st.markdown(
                                f"- [{severity.upper()}] "
                                f"{region.get('location', '?')}: "
                                f"expected {region.get('expected', '?')}, "
                                f"predicted {region.get('predicted', '?')}"
                            )

        # ---------------------------------------------------------------
        # Section 8: Comparison Images (MOVED DOWN, in sub-expander)
        # ---------------------------------------------------------------
        has_images = any(v is not None for v in data["images"].values())
        if has_images:
            with st.expander("Comparison Images", expanded=False):
                for year in ["2021", "2023"]:
                    full_img = data["images"].get(f"{year}_full")
                    if full_img:
                        st.markdown(f"**{year} \u2014 Full View:**")
                        st.image(str(full_img), use_container_width=True)

                    quadrant_keys = [
                        f"{year}_nw", f"{year}_ne",
                        f"{year}_sw", f"{year}_se",
                    ]
                    quadrant_imgs = [data["images"].get(k) for k in quadrant_keys]
                    if any(q is not None for q in quadrant_imgs):
                        st.markdown(f"**{year} \u2014 Quadrants:**")
                        q_col1, q_col2 = st.columns(2)
                        with q_col1:
                            if quadrant_imgs[0]:
                                st.image(
                                    str(quadrant_imgs[0]),
                                    caption="NW",
                                    use_container_width=True,
                                )
                            if quadrant_imgs[2]:
                                st.image(
                                    str(quadrant_imgs[2]),
                                    caption="SW",
                                    use_container_width=True,
                                )
                        with q_col2:
                            if quadrant_imgs[1]:
                                st.image(
                                    str(quadrant_imgs[1]),
                                    caption="NE",
                                    use_container_width=True,
                                )
                            if quadrant_imgs[3]:
                                st.image(
                                    str(quadrant_imgs[3]),
                                    caption="SE",
                                    use_container_width=True,
                                )


def _detect_loop_status(session_dir: Path | None = None) -> str:
    """Detect whether the autocorrect loop is running, complete, or idle.

    Returns one of: "running", "complete", "idle".
    - "running": experiments exist and at least one iteration has "pending" status
    - "complete": experiments exist and SUMMARY.md is present (loop finished)
    - "idle": no experiments found
    """
    experiments_dir = session_dir or config.EXPERIMENTS_DIR

    if not check_experiments_exist(session_dir=experiments_dir):
        return "idle"

    # Check for pending iterations (indicates loop is still running)
    from src.experiment import list_iterations

    # For non-latest sessions, list iterations from the specific dir
    if session_dir and session_dir != config.EXPERIMENTS_DIR:
        iterations = _list_iterations_from_dir(experiments_dir)
    else:
        iterations = list_iterations()

    if any(it["status"] == "pending" for it in iterations):
        return "running"

    # SUMMARY.md presence indicates the loop ran to completion
    summary_path = experiments_dir / "SUMMARY.md"
    if summary_path.exists():
        return "complete"

    # Experiments exist but no pending and no summary -- likely still in progress
    # (summary is written at end of loop)
    return "running"


@st.fragment(run_every=10)
def render_experiments_section() -> None:
    """Render the full experiments timeline section.

    Uses @st.fragment(run_every=10) to auto-refresh every 10 seconds,
    allowing live monitoring of the autocorrect loop without affecting
    the Map View tab.
    """
    from src.experiment import list_iterations, get_session_dir

    # Read session from session_state (set by selector in main())
    selected_session_id = st.session_state.get("session_selector")
    if selected_session_id:
        try:
            experiments_dir = get_session_dir(selected_session_id)
        except FileNotFoundError:
            experiments_dir = config.EXPERIMENTS_DIR
    else:
        experiments_dir = config.EXPERIMENTS_DIR

    st.header("Experiment History")
    st.markdown("Timeline of all autocorrect iterations with hypotheses, scores, and Gemini feedback.")

    # Show loop status indicator
    loop_status = _detect_loop_status(session_dir=experiments_dir)
    if loop_status == "running":
        st.status("Loop running... (auto-refreshing every 10s)", state="running")
    elif loop_status == "complete":
        st.success("Loop complete")

    # Guard: no experiments
    if not check_experiments_exist(session_dir=experiments_dir):
        st.info(
            "No experiments found. Run the autocorrect loop first:\n\n"
            "```\nuv run python -m src.autocorrect\n```"
        )
        return

    # Show SUMMARY.md if it exists
    summary_path = experiments_dir / "SUMMARY.md"
    if summary_path.exists():
        with st.expander("Experiment Summary (SUMMARY.md)", expanded=False):
            st.markdown(summary_path.read_text())

    # Load iteration list
    # Use list_iterations() for latest, _list_iterations_from_dir() for others
    is_latest = (experiments_dir == config.EXPERIMENTS_DIR or
                 (experiments_dir.resolve() == config.EXPERIMENTS_DIR.resolve()
                  if config.EXPERIMENTS_DIR.exists() else False))
    if is_latest:
        iterations = list_iterations()
    else:
        iterations = _list_iterations_from_dir(experiments_dir)

    if not iterations:
        st.warning("Experiment directories exist but no valid iterations found.")
        return

    # Determine best iteration: last accepted iteration
    accepted = [it for it in iterations if it["status"] == "accepted"]
    best_iteration = accepted[-1]["iteration"] if accepted else None

    # Summary metrics row
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Iterations", len(iterations))
    with cols[1]:
        accepted_count = len(accepted)
        st.metric("Accepted", accepted_count)
    with cols[2]:
        reverted_count = len([it for it in iterations if it["status"] == "reverted"])
        st.metric("Reverted", reverted_count)
    with cols[3]:
        if best_iteration is not None:
            st.metric("Best Iteration", f"#{best_iteration:03d}")
        else:
            st.metric("Best Iteration", "N/A")

    st.divider()

    # Render each iteration as an expandable card
    for it_summary in iterations:
        iteration_num = it_summary["iteration"]
        is_best = (iteration_num == best_iteration)

        try:
            details = load_iteration_details(iteration_num, session_dir=experiments_dir)
            render_iteration_card(details, is_best=is_best)
        except FileNotFoundError as e:
            st.warning(f"Could not load iteration {iteration_num:03d}: {e}")
        except Exception as e:
            st.error(f"Error loading iteration {iteration_num:03d}: {e}")


def load_landcover(year: str) -> tuple[np.ndarray, dict]:
    """Load classified landcover GeoTIFF and metadata."""
    path = config.OUTPUT_DIR / f"landcover_{year}.tif"
    with rasterio.open(path) as src:
        data = src.read(1)
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return data, {"bounds": bounds}


def load_change_map() -> tuple[np.ndarray, dict]:
    """Load change detection map."""
    path = config.OUTPUT_DIR / "change_map.tif"
    with rasterio.open(path) as src:
        data = src.read(1)
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return data, {"bounds": bounds}


def load_transitions() -> dict:
    """Load transition statistics."""
    path = config.OUTPUT_DIR / "transitions.json"
    with open(path) as f:
        return json.load(f)


def landcover_to_rgba(data: np.ndarray, alpha: int = 180) -> np.ndarray:
    """Convert landcover class array to RGBA image."""
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    color_map = {
        0: (255, 0, 0),      # Built-up - Red
        1: (255, 255, 0),    # Cropland - Yellow
        2: (144, 238, 144),  # Grassland - Light green
        3: (0, 100, 0),      # Tree cover - Dark green
        4: (0, 0, 255),      # Water - Blue
        5: (128, 128, 128),  # Other - Gray
    }

    for cls_idx, rgb in color_map.items():
        mask = data == cls_idx
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = alpha

    return rgba


def change_to_rgba(data: np.ndarray) -> np.ndarray:
    """Convert change map to RGBA (changed pixels highlighted)."""
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    changed = data == 1
    rgba[changed, 0] = 255   # Red
    rgba[changed, 1] = 0
    rgba[changed, 2] = 255   # Magenta
    rgba[changed, 3] = 200

    return rgba


def create_map(
    bounds: tuple,
    lc_2021_rgba: np.ndarray,
    lc_2023_rgba: np.ndarray,
    change_rgba: np.ndarray,
    active_layers: list[str],
) -> folium.Map:
    """Create folium map with landcover overlays."""
    # Center on AOI
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    # Add satellite basemap option
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)

    # Bounds for image overlay: [[south, west], [north, east]]
    img_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    if "Landcover 2021" in active_layers:
        ImageOverlay(
            image=lc_2021_rgba,
            bounds=img_bounds,
            name="Landcover 2021",
            opacity=0.7,
        ).add_to(m)

    if "Landcover 2023" in active_layers:
        ImageOverlay(
            image=lc_2023_rgba,
            bounds=img_bounds,
            name="Landcover 2023",
            opacity=0.7,
        ).add_to(m)

    if "Change Map" in active_layers:
        ImageOverlay(
            image=change_rgba,
            bounds=img_bounds,
            name="Change 2021→2023",
            opacity=0.8,
        ).add_to(m)

    folium.LayerControl().add_to(m)

    return m


def main():
    st.title("OlmoEarth UK Landcover Demo")
    st.markdown("Exeter, Devon — 5km x 5km AOI — OlmoEarth Tiny embeddings")

    tab_map, tab_experiments = st.tabs(["Map View", "Experiments"])

    with tab_map:
        if not check_outputs_exist():
            st.error(
                "Pipeline outputs not found. Run the pipeline first:\n\n"
                "```\nuv run python -m src.pipeline\n```"
            )
        else:
            # Load data
            lc_2021, meta_2021 = load_landcover("2021")
            lc_2023, meta_2023 = load_landcover("2023")
            change_map, meta_change = load_change_map()
            transitions = load_transitions()

            bounds = meta_2021["bounds"]

            # Convert to RGBA
            lc_2021_rgba = landcover_to_rgba(lc_2021)
            lc_2023_rgba = landcover_to_rgba(lc_2023)
            change_rgba = change_to_rgba(change_map)

            # Sidebar (inside tab_map so controls only appear on Map View)
            with st.sidebar:
                st.header("Layers")
                layers = []
                if st.checkbox("Landcover 2021", value=True):
                    layers.append("Landcover 2021")
                if st.checkbox("Landcover 2023", value=False):
                    layers.append("Landcover 2023")
                if st.checkbox("Change Map", value=False):
                    layers.append("Change Map")

                st.divider()

                # Legend
                st.header("Legend")
                for idx, name in config.LANDCOVER_CLASSES.items():
                    color = config.LANDCOVER_COLORS[idx]
                    st.markdown(
                        f'<span style="background-color:{color};padding:2px 12px;'
                        f'margin-right:8px;border:1px solid #333;">&nbsp;</span> {name}',
                        unsafe_allow_html=True,
                    )

                st.divider()

                # Summary stats
                st.header("Change Summary")
                summary = transitions["summary"]
                st.metric("Total Area", f"{summary['total_area_ha']:.0f} ha")
                st.metric("Changed Area", f"{summary['changed_area_ha']:.1f} ha")
                st.metric("% Changed", f"{summary['pct_changed']:.1f}%")

            # Main content - two columns
            col1, col2 = st.columns([3, 1])

            with col1:
                m = create_map(bounds, lc_2021_rgba, lc_2023_rgba, change_rgba, layers)
                st_folium(m, width=800, height=600)

            with col2:
                # Per-class area comparison
                st.subheader("Class Areas (ha)")

                pixel_area = summary["pixel_area_ha"]
                class_data = []
                for idx, name in config.LANDCOVER_CLASSES.items():
                    area_2021 = float((lc_2021 == idx).sum() * pixel_area)
                    area_2023 = float((lc_2023 == idx).sum() * pixel_area)
                    change_ha = area_2023 - area_2021
                    class_data.append({
                        "Class": name,
                        "2021 (ha)": f"{area_2021:.1f}",
                        "2023 (ha)": f"{area_2023:.1f}",
                        "Change": f"{change_ha:+.1f}",
                    })

                st.dataframe(class_data, hide_index=True, use_container_width=True)

                # Top transitions
                st.subheader("Top Transitions")
                for t in transitions["transitions"][:8]:
                    st.markdown(
                        f"**{t['from_name']}** → **{t['to_name']}**: "
                        f"{t['area_ha']:.1f} ha"
                    )

            # Transition matrix
            st.subheader("Transition Matrix (pixels)")
            matrix = np.array(transitions["matrix"])
            class_names = [config.LANDCOVER_CLASSES[i] for i in range(len(config.LANDCOVER_CLASSES))]

            import pandas as pd
            df = pd.DataFrame(
                matrix,
                index=[f"From: {n}" for n in class_names],
                columns=[f"To: {n}" for n in class_names],
            )
            st.dataframe(df, use_container_width=True)

    with tab_experiments:
        from src.experiment import list_sessions, get_session_dir

        sessions = list_sessions()
        if sessions:
            session_options = {
                s["session_id"]: (
                    f"{s['start_time'][:19]} "
                    f"({s['n_iterations']} iters, "
                    f"score: {s['final_score']:.2f})" if s["final_score"] else
                    f"{s['start_time'][:19]} ({s['n_iterations']} iters)"
                )
                for s in sessions
            }
            st.selectbox(
                "Session",
                options=list(session_options.keys()),
                format_func=lambda x: session_options[x],
                index=0,
                key="session_selector",
            )

            # Compare Sessions table
            if len(sessions) > 1:
                _render_compare_sessions(sessions)
        else:
            # No sessions yet -- clear session_state
            if "session_selector" in st.session_state:
                del st.session_state["session_selector"]

        render_experiments_section()


if __name__ == "__main__":
    main()

"""Self-correction orchestrator -- iterative improvement loop.

Runs classify -> evaluate -> diagnose -> modify config -> re-classify
in a loop until quality converges or budget is exhausted.

Usage:
    uv run python -m src.autocorrect
    uv run python -m src.autocorrect --max-iterations 5 --target-score 9.0 --patience 2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src import config

# === Constants ===
DEFAULT_MAX_ITERATIONS: int = 10
DEFAULT_TARGET_SCORE: float = 8.5
DEFAULT_PATIENCE: int = 3
CLASS_REGRESSION_THRESHOLD: float = 1.0  # max per-class score drop (0-10 scale)


# ---------------------------------------------------------------------------
# Section 1: Scoring Protocol
# ---------------------------------------------------------------------------

@dataclass
class IterationScore:
    """Uniform score representation for accept/revert decisions."""
    iteration: int
    overall_score: float           # Normalized to 0-10 scale
    class_scores: dict[str, float] = field(default_factory=dict)  # class_name -> score (0-10)
    source: str = "metrics"        # "vlm" or "metrics" -- provenance tracking


def _score_from_vlm(iteration: int, eval_result: dict) -> IterationScore:
    """Extract IterationScore from a single year's VLM evaluation.

    Uses evaluation.overall_score (already 1-10) and evaluation.per_class[].score.
    """
    evaluation = eval_result["evaluation"]
    class_scores = {
        pc["class_name"]: float(pc["score"])
        for pc in evaluation.get("per_class", [])
    }
    return IterationScore(
        iteration=iteration,
        overall_score=float(evaluation["overall_score"]),
        class_scores=class_scores,
        source="vlm",
    )


def _score_from_metrics(iteration: int, metrics: dict) -> IterationScore:
    """Extract IterationScore from classification metrics, scaled to 0-10.

    Maps overall_accuracy (0-1) to 0-10 scale: score = accuracy * 10.
    Maps per_class F1 (0-1) to 0-10 scale: score = f1 * 10.
    """
    class_scores = {
        name: round(data["f1"] * 10, 2)
        for name, data in metrics.get("per_class", {}).items()
    }
    return IterationScore(
        iteration=iteration,
        overall_score=round(metrics.get("overall_accuracy", 0.0) * 10, 2),
        class_scores=class_scores,
        source="metrics",
    )


def _extract_score(iteration: int, metrics: dict, eval_results: dict | None) -> IterationScore:
    """Extract blended score: combines VLM visual assessment with quantitative metrics.

    Blending prevents the problem where Gemini's coarse integer scale (1-10)
    can't distinguish between e.g. 83% and 86% accuracy (both score "8").
    The blend uses VLM as the primary signal but adds a fractional component
    from metrics so accuracy improvements always register.

    Formula: score = vlm_score * 0.7 + metrics_score * 0.3
    """
    metrics_score = _score_from_metrics(iteration, metrics)

    if eval_results is not None:
        year_result = eval_results.get("2021")
        if year_result and isinstance(year_result, dict) and "evaluation" in year_result:
            vlm_score = _score_from_vlm(iteration, year_result)
            # Blend: VLM provides qualitative assessment, metrics provide precision
            blended_overall = round(vlm_score.overall_score * 0.7 + metrics_score.overall_score * 0.3, 2)
            blended_classes = {}
            for name in set(list(vlm_score.class_scores.keys()) + list(metrics_score.class_scores.keys())):
                vlm_cs = vlm_score.class_scores.get(name, metrics_score.class_scores.get(name, 0.0))
                met_cs = metrics_score.class_scores.get(name, vlm_cs)
                blended_classes[name] = round(vlm_cs * 0.7 + met_cs * 0.3, 2)
            return IterationScore(
                iteration=iteration,
                overall_score=blended_overall,
                class_scores=blended_classes,
                source="blended",
            )
    return metrics_score


# ---------------------------------------------------------------------------
# Section 2: Acceptance Policy
# ---------------------------------------------------------------------------

def _check_pareto_acceptance(
    best: IterationScore,
    candidate: IterationScore,
    max_class_regression: float = CLASS_REGRESSION_THRESHOLD,
) -> tuple[bool, str]:
    """Check if candidate should be accepted over current best.

    Pareto-relaxed constraint:
    1. Overall score must strictly improve (candidate > best)
    2. No individual class score may drop by more than max_class_regression

    Returns (accepted: bool, reason: str).
    """
    # Rule 1: overall must improve (or match with blended scoring precision)
    if candidate.overall_score < best.overall_score:
        delta = candidate.overall_score - best.overall_score
        return False, (
            f"Overall score did not improve: "
            f"{best.overall_score:.2f} -> {candidate.overall_score:.2f} "
            f"(delta={delta:+.2f})"
        )

    # Rule 2: no class may regress beyond threshold
    if best.class_scores:  # Guard: skip if no per-class data (RT4-9)
        for class_name, best_class_score in best.class_scores.items():
            candidate_class_score = candidate.class_scores.get(class_name, 0.0)
            drop = best_class_score - candidate_class_score
            if drop > max_class_regression:
                return False, (
                    f"Class '{class_name}' regressed beyond threshold: "
                    f"{best_class_score:.2f} -> {candidate_class_score:.2f} "
                    f"(drop={drop:.2f} > {max_class_regression})"
                )

    delta = candidate.overall_score - best.overall_score
    return True, (
        f"Overall score improved: "
        f"{best.overall_score:.2f} -> {candidate.overall_score:.2f} "
        f"(delta={delta:+.2f})"
    )


# ---------------------------------------------------------------------------
# Section 3: Config Mutation
# ---------------------------------------------------------------------------

def _apply_hypothesis(current_config: dict, hypothesis_data: dict) -> dict:
    """Apply hypothesis parameter_changes to config. Returns new validated config.

    Uses _validate_hypothesis from diagnose.py which:
    - Converts dot-notation keys to nested dict
    - Deep-merges with current config
    - Validates the result via validate_config()
    - Verifies at least one change was actually applied

    Args:
        current_config: Current experiment config dict.
        hypothesis_data: Raw hypothesis dict with 'parameter_changes' key.

    Returns:
        New config dict (deep copy, does not mutate input).

    Raises:
        ValueError: If merged config fails validation or no changes were applied.
    """
    from src.diagnose import Hypothesis, _validate_hypothesis

    # Convert raw dict to Hypothesis Pydantic model for _validate_hypothesis
    hypothesis = Hypothesis(**{
        k: v for k, v in hypothesis_data.items()
        if k in Hypothesis.model_fields
    })

    new_config = _validate_hypothesis(hypothesis, current_config)

    # No-op detection (RT2-14, RT4-7): if config unchanged, reject
    if new_config == current_config:
        raise ValueError(
            "No effective changes: hypothesis parameter_changes produce "
            "an identical config to the current one."
        )

    return new_config


def _get_latest_iteration() -> int:
    """Return the iteration number of the most recently created experiment.

    Scans list_iterations() sorted by timestamp. More robust than
    get_next_iteration_number() - 1 which has race conditions.

    Raises RuntimeError if no iterations exist.
    """
    from src.experiment import list_iterations

    iterations = list_iterations()
    if not iterations:
        raise RuntimeError("No experiment iterations found in ledger.")
    # list_iterations() returns sorted by iteration number
    return iterations[-1]["iteration"]


# ---------------------------------------------------------------------------
# Section 4: Diagnosis Integration
# ---------------------------------------------------------------------------

def _run_diagnosis_safe(iteration: int) -> dict | None:
    """Run diagnosis with full graceful degradation.

    Returns hypothesis dict or None if diagnosis unavailable/failed.
    """
    try:
        from src.diagnose import run_diagnosis
    except ImportError:
        print("[autocorrect] WARNING: src.diagnose not available. "
              "Install Issue #7 dependency first.")
        return None

    try:
        hypothesis = run_diagnosis(iteration=iteration)
        # run_diagnosis() always returns a Hypothesis object or raises (RT3-3)
        return hypothesis.model_dump()
    except Exception as e:
        print(f"[autocorrect] WARNING: Diagnosis failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Section 5: GeoTIFF Management
# ---------------------------------------------------------------------------

def _backup_outputs(iteration_dir: Path) -> None:
    """Copy current landcover GeoTIFFs to the iteration directory.

    Called after each successful classification to preserve outputs.
    Raises FileNotFoundError if no output files exist.
    """
    import shutil

    copied = []
    for year in config.TIME_RANGES:
        src_path = config.OUTPUT_DIR / f"landcover_{year}.tif"
        if src_path.exists():
            dst_path = iteration_dir / f"landcover_{year}.tif"
            shutil.copy2(src_path, dst_path)
            copied.append(dst_path)
            print(f"[autocorrect] Backed up {src_path.name} to {iteration_dir.name}/")
        else:
            print(f"[autocorrect] WARNING: {src_path.name} not found, skipping backup")

    if not copied:
        raise FileNotFoundError(
            "No landcover outputs found to back up. "
            "Classification may have failed silently."
        )


def _restore_best_outputs(best_iter: int) -> None:
    """Copy best iteration's GeoTIFFs to output/ as final result.

    Uses atomic copy (write to .tmp, rename) to prevent corruption.
    """
    import shutil

    iteration_dir = config.EXPERIMENTS_DIR / f"iteration_{best_iter:03d}"
    for year in config.TIME_RANGES:
        src = iteration_dir / f"landcover_{year}.tif"
        if not src.exists():
            print(f"[autocorrect] WARNING: Best iteration missing {src.name}")
            continue
        dst = config.OUTPUT_DIR / f"landcover_{year}.tif"
        tmp = dst.with_suffix(".tif.tmp")
        shutil.copy2(src, tmp)
        tmp.rename(dst)
        print(f"[autocorrect] Restored best output: {dst.name} "
              f"(from iteration {best_iter:03d})")


# ---------------------------------------------------------------------------
# Section 6: Signal Handler
# ---------------------------------------------------------------------------

def _register_signal_handler(state: dict) -> None:
    """Register SIGINT/SIGTERM handler to restore best config and save partial summary.

    Args:
        state: Mutable dict with keys 'best_cfg', 'best_iter', 'history',
               'best_score', 'original_config', and run parameters.
               Updated by the main loop so the handler always sees current state.
    """
    import signal

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _shutdown_handler(signum, frame):
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n[autocorrect] Received {signal_name}, cleaning up...")

        try:
            from src.experiment import save_config
            cfg_to_restore = state.get("best_cfg") or state.get("original_config")
            if cfg_to_restore:
                save_config(cfg_to_restore)
                print("[autocorrect] Restored best experiment_config.json")
        except Exception as e:
            print(f"[autocorrect] WARNING: Failed to restore config: {e}")

        # Attempt to restore best outputs and save partial summary (RT4-17)
        try:
            best_iter = state.get("best_iter")
            if best_iter is not None:
                _restore_best_outputs(best_iter)
        except Exception as e:
            print(f"[autocorrect] WARNING: Failed to restore outputs: {e}")

        try:
            if state.get("history"):
                partial_summary = _build_summary(state, stop_reason="interrupted")
                _save_summary_md(partial_summary)
        except Exception:
            pass  # Best effort

        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(1)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)


# ---------------------------------------------------------------------------
# Section 7: Summary Generation
# ---------------------------------------------------------------------------

def _build_summary(state: dict, stop_reason: str) -> dict:
    """Build a summary dict from the loop state.

    Uses actual runtime parameters (RT1-1) not defaults.
    """
    history = state.get("history", [])
    best_score_obj = state.get("best_score")

    return {
        "iterations_run": len(history),
        "iterations_classified": sum(
            1 for h in history if h.get("iteration") is not None
        ),
        "accepted": sum(1 for h in history if h["status"] == "accepted"),
        "reverted": sum(1 for h in history if h["status"] == "reverted"),
        "skipped": sum(1 for h in history if h["status"] == "skipped"),
        "initial_score": history[0]["score"] if history else None,
        "final_score": best_score_obj.overall_score if best_score_obj else None,
        "best_iteration": state.get("best_iter"),
        "best_score": best_score_obj.overall_score if best_score_obj else None,
        "stop_reason": stop_reason,
        "history": history,
        "max_iterations": state.get("max_iterations", DEFAULT_MAX_ITERATIONS),
        "target_score": state.get("target_score", DEFAULT_TARGET_SCORE),
        "patience": state.get("patience", DEFAULT_PATIENCE),
    }


def _print_summary(summary: dict) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'=' * 60}")
    print("[autocorrect] Self-Correction Complete")
    print(f"{'=' * 60}")
    print(f"[autocorrect] Iterations: {summary['iterations_run']} "
          f"({summary['accepted']} accepted, {summary['reverted']} reverted)")
    if summary["initial_score"] is not None:
        final = summary["final_score"]
        if final is not None:
            delta = final - summary["initial_score"]
            print(f"[autocorrect] Score: {summary['initial_score']:.2f} "
                  f"-> {final:.2f} ({delta:+.2f})")
    best_iter = summary.get("best_iteration")
    if best_iter is not None:
        print(f"[autocorrect] Best iteration: {best_iter:03d}")
    print(f"[autocorrect] Stop reason: {summary['stop_reason']}")
    print()

    accepted_changes = [h for h in summary["history"]
                        if h["status"] == "accepted" and h.get("config_changes")]
    if accepted_changes:
        print("[autocorrect] Accepted changes:")
        for h in accepted_changes:
            if h["iteration"] is not None:
                print(f"[autocorrect]   Iteration {h['iteration']:03d}: {h['hypothesis']}")

    reverted_changes = [h for h in summary["history"]
                        if h["status"] == "reverted" and h.get("iteration") is not None]
    if reverted_changes:
        print("[autocorrect] Reverted attempts:")
        for h in reverted_changes:
            iter_str = f"{h['iteration']:03d}" if h["iteration"] is not None else "N/A"
            print(f"[autocorrect]   Iteration {iter_str}: {h['reason']}")

    _print_final_table(summary["history"])

    print(f"{'=' * 60}")


def _save_summary_md(summary: dict) -> None:
    """Save experiments/SUMMARY.md with the full narrative.

    Uses actual runtime parameters (RT1-1), not DEFAULT_* constants.
    """
    import json
    import datetime

    max_iters = summary.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    target = summary.get("target_score", DEFAULT_TARGET_SCORE)
    pat = summary.get("patience", DEFAULT_PATIENCE)

    lines = [
        "# Self-Correction Loop Summary",
        "",
        f"**Run date:** {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Parameters:** max_iterations={max_iters}, "
        f"target_score={target}, patience={pat}",
        f"**Stop reason:** {summary['stop_reason']}",
    ]

    best_iter = summary.get("best_iteration")
    if best_iter is not None:
        lines.append(f"**Best iteration:** {best_iter:03d}")

    if summary["initial_score"] is not None and summary["final_score"] is not None:
        delta = summary["final_score"] - summary["initial_score"]
        lines.append(f"**Score trajectory:** {summary['initial_score']:.2f} "
                      f"-> {summary['final_score']:.2f} ({delta:+.2f})")

    if best_iter is not None:
        lines.extend([
            "",
            f"**Note:** If the process was killed unexpectedly, `experiment_config.json` "
            f"may contain a hypothesis config. Restore from "
            f"`experiments/iteration_{best_iter:03d}/config.json`.",
        ])

    lines.extend([
        "",
        "## Iteration History",
        "",
        "| # | Status | Score | Change | Hypothesis |",
        "|---|--------|-------|--------|------------|",
    ])

    prev_score = None
    for h in summary["history"]:
        iter_str = f"{h['iteration']:03d}" if h["iteration"] is not None else "N/A"
        score_str = f"{h['score']:.2f}" if h["score"] is not None else "N/A"
        if prev_score is not None and h["score"] is not None:
            delta = h["score"] - prev_score
            change_str = f"{delta:+.2f}"
        elif h["iteration"] is not None and h["score"] is not None:
            change_str = "baseline"
        else:
            change_str = "N/A"
        hyp_str = h.get("hypothesis") or "--"
        lines.append(f"| {iter_str} | {h['status']} | {score_str} | "
                      f"{change_str} | {hyp_str} |")
        if h["score"] is not None:
            prev_score = h["score"]

    # Accepted changes section
    lines.extend(["", "## Accepted Changes", ""])
    accepted = [h for h in summary["history"]
                if h["status"] == "accepted" and h.get("config_changes")]
    if accepted:
        for h in accepted:
            if h["iteration"] is not None:
                lines.append(f"### Iteration {h['iteration']:03d}: {h['hypothesis']}")
                lines.append(f"- **Change:** `{json.dumps(h['config_changes'])}`")
                lines.append(f"- **Score:** {h['score']:.2f}")
                lines.append("")
    else:
        lines.append("No improvements were accepted beyond the baseline.")
        lines.append("")

    # Reverted attempts section
    lines.extend(["## Reverted Attempts", ""])
    reverted = [h for h in summary["history"]
                if h["status"] == "reverted" and h.get("iteration") is not None]
    if reverted:
        for h in reverted:
            if h["iteration"] is not None:
                lines.append(f"### Iteration {h['iteration']:03d}")
                lines.append(f"- **Hypothesis:** {h.get('hypothesis', 'N/A')}")
                lines.append(f"- **Reason:** {h['reason']}")
                lines.append("")
    else:
        lines.append("No iterations were reverted.")
        lines.append("")

    content = "\n".join(lines) + "\n"

    # Atomic write
    output_path = config.EXPERIMENTS_DIR / "SUMMARY.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".md.tmp")
    with open(tmp_path, "w") as f:
        f.write(content)
    tmp_path.rename(output_path)
    print(f"[autocorrect] Summary saved to {output_path}")


# ---------------------------------------------------------------------------
# Section 7a: Progress Display
# ---------------------------------------------------------------------------

def _format_progress_bar(current: int, total: int, width: int = 10) -> str:
    """Format a progress bar string: [2/5] ████░░░░░░ 40%

    Args:
        current: Current iteration (0 = baseline, 1+ = improvement).
        total: Total improvement iterations.
        width: Character width of the bar (default 10).

    Returns:
        Formatted progress bar string.
    """
    if total <= 0:
        fraction = 0.0
    else:
        fraction = min(current / total, 1.0)
    filled = round(fraction * width)
    empty = width - filled
    pct = round(fraction * 100)
    return f"[{current}/{total}] {'█' * filled}{'░' * empty} {pct}%"


def _print_phase_status(phase: str, message: str) -> None:
    """Print an inline status line for a long-running operation.

    Example: [classify] Training...

    Args:
        phase: Module name (classify, evaluate, diagnose).
        message: Status message (e.g., "Training...").
    """
    print(f"[{phase}] {message}")


def _print_iteration_box(
    iteration_index: int,
    max_iterations: int,
    hypothesis: str | None,
    prev_score: float,
    new_score: float,
    accepted: bool,
    class_scores: dict[str, float],
) -> None:
    """Print a box-drawing summary after each iteration's accept/revert decision.

    Example:
        ┌─ Iteration 2/5 ─────────────────────────────┐
        │ Hypothesis: Add NDVI features (Tier 1)       │
        │ Score: 6.0 → 7.2 (+1.2) ✓ ACCEPTED          │
        │ Best classes: Other (9.4), Cropland (8.4)    │
        │ Worst class: Water (4.8)                     │
        └──────────────────────────────────────────────┘

    Args:
        iteration_index: 1-based iteration number (improvement iterations only).
        max_iterations: Total iterations including baseline (N in run_autocorrect).
        hypothesis: Human-readable hypothesis text, or None.
        prev_score: Best score before this iteration (0-10).
        new_score: Score of this iteration (0-10).
        accepted: Whether the iteration was accepted.
        class_scores: Per-class scores dict {class_name: score}. May be empty.
    """
    box_w = 48  # total outer width including borders
    inner_w = box_w - 2  # content area between │ chars

    # Title line: ┌─ Iteration X/Y ─...─┐
    total_improvement = max_iterations - 1
    title_text = f" Iteration {iteration_index}/{total_improvement} "
    top = ("┌─" + title_text).ljust(box_w - 1, "─") + "┐"
    print(top)

    def _line(text: str) -> str:
        """Format a content line padded to inner_w, bordered with │."""
        return "│ " + text.ljust(inner_w - 2) + " │"

    # Hypothesis line
    hyp_text = hypothesis if hypothesis else "(Baseline)"
    max_hyp_len = inner_w - 2 - len("Hypothesis: ")  # = 32
    if len(hyp_text) > max_hyp_len:
        hyp_text = hyp_text[: max_hyp_len - 3] + "..."
    print(_line(f"Hypothesis: {hyp_text}"))

    # Score line
    delta = new_score - prev_score
    decision = "✓ ACCEPTED" if accepted else "✗ REVERTED"
    print(_line(f"Score: {prev_score:.1f} → {new_score:.1f} ({delta:+.1f}) {decision}"))

    # Best/worst class lines (only if class_scores is non-empty)
    if class_scores:
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        # Best classes (top 2)
        top_n = sorted_classes[:2]
        best_str = ", ".join(f"{name} ({score:.1f})" for name, score in top_n)
        print(_line(f"Best classes: {best_str}"))
        # Worst class (bottom 1)
        worst_name, worst_score = sorted_classes[-1]
        print(_line(f"Worst class: {worst_name} ({worst_score:.1f})"))

    # Bottom border
    print("└" + "─" * (box_w - 2) + "┘")


def _print_final_table(history: list[dict]) -> None:
    """Print a formatted table of all iterations at the end of the loop.

    Example:
        ┌─ Iteration Summary ──────────────────────────────────────────────┐
        │  #  │ Status   │ Score │ Delta  │ Hypothesis                     │
        │─────┼──────────┼───────┼────────┼────────────────────────────────│
        │ 001 │ accepted │  7.52 │   --   │ (Baseline)                     │
        │ 002 │ accepted │  7.80 │ +0.28  │ Add NDVI features              │
        │ 003 │ reverted │  7.60 │ -0.20  │ Increase trees                 │
        └──────────────────────────────────────────────────────────────────┘

    Args:
        history: List of history entry dicts from the autocorrect loop.
    """
    # Column widths (content only, not including separators)
    c_num = 3       # iteration number
    c_status = 8    # status string
    c_score = 5     # score value
    c_delta = 6     # delta value
    c_hyp = 30      # hypothesis text

    # Total width: │ + space + col + space + │ for each column
    # 5 columns, each with " col " padding = col + 2 spaces each
    # 4 inner │ separators + 2 outer │ = 6 border chars
    # total = (c_num+2) + (c_status+2) + (c_score+2) + (c_delta+2) + (c_hyp+2) + 6
    table_w = c_num + c_status + c_score + c_delta + c_hyp + 16

    # Title
    title_text = " Iteration Summary "
    top = ("┌─" + title_text).ljust(table_w - 1, "─") + "┐"
    print(top)

    def _row(num: str, status: str, score: str, delta: str, hyp: str) -> str:
        return (f"│ {num:>{c_num}} │ {status:<{c_status}} │ "
                f"{score:>{c_score}} │ {delta:>{c_delta}} │ {hyp:<{c_hyp}} │")

    # Header row
    print(_row("#", "Status", "Score", "Delta", "Hypothesis"))

    # Separator row
    sep = (f"│{'─' * (c_num + 2)}┼{'─' * (c_status + 2)}┼"
           f"{'─' * (c_score + 2)}┼{'─' * (c_delta + 2)}┼"
           f"{'─' * (c_hyp + 2)}│")
    print(sep)

    if not history:
        print(_row("", "", "", "", "(no iterations)"))
    else:
        prev_score: float | None = None
        for h in history:
            # Iteration number
            num_str = f"{h['iteration']:03d}" if h["iteration"] is not None else "N/A"

            # Status
            status_str = h["status"]

            # Score
            score_str = f"{h['score']:.2f}" if h["score"] is not None else "N/A"

            # Delta (relative to previous non-None score)
            if prev_score is not None and h["score"] is not None:
                d = h["score"] - prev_score
                delta_str = f"{d:+.2f}"
            elif h["score"] is not None and prev_score is None:
                delta_str = "--"
            else:
                delta_str = "N/A"

            # Hypothesis (truncated)
            hyp_raw = h.get("hypothesis") or "(Baseline)"
            if len(hyp_raw) > c_hyp:
                hyp_str = hyp_raw[: c_hyp - 3] + "..."
            else:
                hyp_str = hyp_raw

            print(_row(num_str, status_str, score_str, delta_str, hyp_str))

            # Track previous score for delta
            if h["score"] is not None:
                prev_score = h["score"]

    # Bottom border
    print("└" + "─" * (table_w - 2) + "┘")


# ---------------------------------------------------------------------------
# Section 8: Main Loop
# ---------------------------------------------------------------------------

def run_autocorrect(
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    target_score: float = DEFAULT_TARGET_SCORE,
    patience: int = DEFAULT_PATIENCE,
) -> dict:
    """Run the self-correction loop.

    Iteration 0 is always the baseline (run classification with current config).
    max_iterations=N means 1 baseline + (N-1) improvement iterations = N total.

    Returns a summary dict with iteration history, final scores, and stop reason.
    """
    import copy

    # --- Argument validation ---
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if not (0.0 <= target_score <= 10.0):
        raise ValueError(f"target_score must be between 0 and 10, got {target_score}")
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")

    # --- Lazy imports ---
    from src.experiment import (
        load_config, save_config, load_experiment,
        update_experiment_status, get_next_iteration_number,
    )
    from src.classify import run_classification
    from src.evaluate import run_evaluation

    # --- Initialization ---
    original_config = copy.deepcopy(load_config())

    # Mutable state container for signal handler (RT2-6, RT4-17)
    state = {
        "original_config": original_config,
        "best_cfg": copy.deepcopy(original_config),
        "best_iter": None,
        "best_score": None,
        "history": [],
        "max_iterations": max_iterations,
        "target_score": target_score,
        "patience": patience,
    }
    _register_signal_handler(state)

    history = state["history"]
    best_cfg = state["best_cfg"]
    best_iter = None
    best_score = None
    stale_count = 0
    stop_reason = "max_iterations"

    print("[autocorrect] Starting self-correction loop")
    print(f"[autocorrect]   max_iterations: {max_iterations}")
    print(f"[autocorrect]   target_score:   {target_score}")
    print(f"[autocorrect]   patience:       {patience}")

    # === ITERATION 0: BASELINE ===
    print(f"\n{'=' * 60}")
    print("[autocorrect] Iteration 0: Baseline classification")
    print(f"{'=' * 60}")

    save_config(best_cfg)  # Ensure config on disk matches loaded config

    _print_phase_status("classify", "Training baseline classifier...")
    try:
        run_classification()
    except Exception as e:
        save_config(original_config)  # Restore before raising
        raise RuntimeError(
            f"Baseline classification failed. Cannot start self-correction loop. "
            f"Error: {e}"
        ) from e

    # Find the iteration just saved
    baseline_iter = _get_latest_iteration()
    baseline_dir = config.EXPERIMENTS_DIR / f"iteration_{baseline_iter:03d}"

    # Back up GeoTIFFs
    _backup_outputs(baseline_dir)

    # Evaluate (graceful degradation)
    eval_results = None
    _print_phase_status("evaluate", "Sending to Gemini...")
    try:
        eval_results = run_evaluation(output_dir=baseline_dir)
    except Exception as e:
        print(f"[autocorrect] WARNING: Baseline evaluation failed: {e}")
        print("[autocorrect] Continuing with metrics-only scoring.")

    # Score
    baseline_exp = load_experiment(baseline_iter)
    baseline_score = _extract_score(baseline_iter, baseline_exp["metrics"], eval_results)
    update_experiment_status(baseline_iter, "accepted")

    best_iter = baseline_iter
    best_score = baseline_score
    best_cfg = copy.deepcopy(baseline_exp["config"])

    # Update state for signal handler
    state["best_iter"] = best_iter
    state["best_score"] = best_score
    state["best_cfg"] = best_cfg

    history.append({
        "iteration": baseline_iter,
        "score": baseline_score.overall_score,
        "status": "accepted",
        "hypothesis": None,
        "config_changes": None,
        "reason": "Baseline iteration",
    })

    print(f"[autocorrect] Baseline score: {baseline_score.overall_score:.2f} "
          f"(source: {baseline_score.source})")

    # === IMPROVEMENT LOOP ===
    for iteration_index in range(1, max_iterations):
        print(f"\n{'=' * 60}")
        print(f"[autocorrect] Iteration {iteration_index}/{max_iterations - 1} "
              f"(stale: {stale_count}/{patience})")
        print(_format_progress_bar(iteration_index, max_iterations - 1))
        print(f"{'=' * 60}")

        # --- Stopping conditions (checked BEFORE each iteration) ---
        if best_score.overall_score >= target_score:
            stop_reason = "target_reached"
            print(f"[autocorrect] Target score {target_score} reached "
                  f"({best_score.overall_score:.2f})!")
            break

        if stale_count >= patience:
            stop_reason = "patience_exhausted"
            print(f"[autocorrect] Patience exhausted "
                  f"({stale_count} consecutive non-accepted iterations)")
            break

        # --- Step 1: Diagnose the best iteration ---
        _print_phase_status("diagnose", "Calling Claude...")
        hypothesis_data = _run_diagnosis_safe(best_iter)
        if hypothesis_data is None:
            stale_count += 1
            history.append({
                "iteration": None,
                "score": None,
                "status": "skipped",
                "hypothesis": None,
                "config_changes": None,
                "reason": "Diagnosis returned no hypothesis",
            })
            print(f"[autocorrect] No hypothesis. Stale count: {stale_count}/{patience}")
            continue

        # --- Step 2: Apply hypothesis to config ---
        try:
            new_cfg = _apply_hypothesis(best_cfg, hypothesis_data)
        except ValueError as e:
            stale_count += 1
            history.append({
                "iteration": None,
                "score": None,
                "status": "skipped",
                "hypothesis": hypothesis_data.get("hypothesis", ""),
                "config_changes": hypothesis_data.get("parameter_changes", {}),
                "reason": f"Invalid hypothesis config: {e}",
            })
            print(f"[autocorrect] Hypothesis rejected (invalid config): {e}")
            continue

        save_config(new_cfg)

        # --- Step 3: Classify ---
        _print_phase_status("classify", "Training classifier...")
        try:
            run_classification()
        except Exception as e:
            print(f"[autocorrect] Classification failed: {e}")
            save_config(best_cfg)  # Restore best config (RT4-5)
            stale_count += 1
            history.append({
                "iteration": None,
                "score": None,
                "status": "reverted",
                "hypothesis": hypothesis_data.get("hypothesis", ""),
                "config_changes": hypothesis_data.get("parameter_changes", {}),
                "reason": f"Classification failed: {e}",
            })
            continue

        # --- Step 4: Find the new iteration ---
        current_iter = _get_latest_iteration()
        current_dir = config.EXPERIMENTS_DIR / f"iteration_{current_iter:03d}"

        # --- Step 5: Back up GeoTIFFs ---
        try:
            _backup_outputs(current_dir)
        except FileNotFoundError as e:
            print(f"[autocorrect] WARNING: Output backup failed: {e}")

        # --- Step 6: Evaluate ---
        eval_results = None
        _print_phase_status("evaluate", "Sending to Gemini...")
        try:
            eval_results = run_evaluation(output_dir=current_dir)
        except Exception as e:
            print(f"[autocorrect] WARNING: Evaluation failed: {e}")
            print("[autocorrect] Continuing with metrics-only scoring.")

        # --- Step 7: Score ---
        current_exp = load_experiment(current_iter)
        current_score = _extract_score(
            current_iter, current_exp["metrics"], eval_results
        )
        print(f"[autocorrect] Score: {current_score.overall_score:.2f} "
              f"(best: {best_score.overall_score:.2f})")

        # --- Step 8: Accept/Revert ---
        prev_best_overall = best_score.overall_score  # capture for summary box
        accepted, reason = _check_pareto_acceptance(best_score, current_score)

        if accepted:
            update_experiment_status(current_iter, "accepted")
            best_iter = current_iter
            best_score = current_score
            best_cfg = copy.deepcopy(new_cfg)
            stale_count = 0
            print(f"[autocorrect] ACCEPTED: {reason}")
        else:
            update_experiment_status(current_iter, "reverted")
            save_config(best_cfg)  # Restore best config
            stale_count += 1
            print(f"[autocorrect] REVERTED: {reason}")

        # Update state for signal handler
        state["best_iter"] = best_iter
        state["best_score"] = best_score
        state["best_cfg"] = best_cfg

        history.append({
            "iteration": current_iter,
            "score": current_score.overall_score,
            "status": "accepted" if accepted else "reverted",
            "hypothesis": hypothesis_data.get("hypothesis", ""),
            "config_changes": hypothesis_data.get("parameter_changes", {}),
            "reason": reason,
        })

        # Print iteration summary box
        _print_iteration_box(
            iteration_index=iteration_index,
            max_iterations=max_iterations,
            hypothesis=hypothesis_data.get("hypothesis", ""),
            prev_score=prev_best_overall,
            new_score=current_score.overall_score,
            accepted=accepted,
            class_scores=current_score.class_scores,
        )

    # === FINALIZATION ===
    print(f"\n{'=' * 60}")
    print("[autocorrect] Finalization")
    print(f"{'=' * 60}")

    # Restore best iteration's outputs to output/
    if best_iter is not None:
        _restore_best_outputs(best_iter)

    # Ensure best config is the active config
    save_config(best_cfg)

    # Run change detection on best outputs
    try:
        from src.change import run_change_detection
        print("[autocorrect] Running change detection on best outputs...")
        run_change_detection()
    except Exception as e:
        print(f"[autocorrect] WARNING: Change detection failed: {e}")
        print("[autocorrect] Run 'python -m src.change' manually if needed.")

    # Build and display summary
    summary = _build_summary(state, stop_reason=stop_reason)
    _print_summary(summary)
    _save_summary_md(summary)

    return summary


# ---------------------------------------------------------------------------
# Section 9: CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Self-correction orchestrator -- iterative improvement loop."
    )
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=(
            f"Total iterations including baseline "
            f"(1 baseline + N-1 improvements, default: {DEFAULT_MAX_ITERATIONS})"
        ),
    )
    parser.add_argument(
        "--target-score", type=float, default=DEFAULT_TARGET_SCORE,
        help=f"Target score (0-10) to stop at (default: {DEFAULT_TARGET_SCORE})",
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help=(
            f"Consecutive non-accepted iterations before stopping "
            f"(default: {DEFAULT_PATIENCE})"
        ),
    )
    args = parser.parse_args()

    run_autocorrect(
        max_iterations=args.max_iterations,
        target_score=args.target_score,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()

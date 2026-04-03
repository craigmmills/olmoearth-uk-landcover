"""Diagnose classification failures and propose hypothesis for next experiment.

Analyzes VLM evaluation results and experiment history to generate a single,
actionable hypothesis for improving landcover classification accuracy.

Usage:
    uv run python -m src.diagnose
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src import config


class Hypothesis(BaseModel):
    """A single proposed change to improve classification accuracy.

    Attributes:
        hypothesis: WHY this change should help (natural language).
        component: WHICH config section (training|features|post_processing).
        parameter_changes: WHAT changes, as dot-notation keys -> new values.
        expected_impact: WHAT should get better.
        risk: WHAT might get worse.
        tier: HOW aggressive (1=safe, 2=moderate, 3=aggressive).
        confidence: Probability this change will improve overall accuracy (0.0-1.0).
        reasoning: Chain-of-thought analysis (for debugging/auditing).
    """
    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    component: str
    parameter_changes: dict[str, Any]
    expected_impact: str
    risk: str
    tier: int
    confidence: float
    reasoning: str

    @field_validator("component")
    @classmethod
    def component_must_be_valid(cls, v):
        allowed = {"training", "features", "post_processing"}
        if v not in allowed:
            raise ValueError(f"component must be one of {allowed}, got '{v}'")
        return v

    @field_validator("tier")
    @classmethod
    def tier_must_be_valid(cls, v):
        if v not in (1, 2, 3):
            raise ValueError(f"tier must be 1, 2, or 3, got {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {v}")
        return v

    @field_validator("parameter_changes")
    @classmethod
    def changes_must_use_dot_notation(cls, v):
        if not v:
            raise ValueError("parameter_changes must not be empty")
        for key in v:
            if "." not in key:
                raise ValueError(
                    f"parameter_changes keys must use dot notation "
                    f"(e.g., 'training.n_estimators'), got '{key}'"
                )
        return v

    @model_validator(mode="after")
    def changes_must_match_component(self) -> "Hypothesis":
        """All parameter_changes must belong to the declared component."""
        for key in self.parameter_changes:
            section = key.split(".")[0]
            if section != self.component:
                raise ValueError(
                    f"Parameter '{key}' belongs to '{section}' but hypothesis "
                    f"component is '{self.component}'. One change per component per iteration."
                )
        return self


# Tier parameter membership
# NOTE: Keep in sync with src/experiment.py validate_config() constraints.
# "classifier swap" (requirements Tier 2) not representable -- only RandomForest in config.
# "embedding changes" (requirements Tier 3) not representable -- no embedding model config param.
TIER_PARAMS: dict[int, set[str]] = {
    1: {
        "training.max_samples_per_class",
        "training.exclude_boundary_pixels",
        "training.boundary_buffer_px",
        "training.class_weight",
        "training.max_depth",           # Moved from Tier 2 (RT3-C3, RT4-C1)
        "post_processing.mode_filter_size",
        "post_processing.min_mapping_unit_px",
    },
    2: {
        "training.n_estimators",
        "features.add_ndvi",
        "features.add_ndwi",
    },
    3: {
        "features.add_spatial_context",
    },
}


def _load_evaluation_results() -> list[dict] | None:
    """Load VLM evaluation results from disk.

    Returns None if no evaluation files exist or directory is missing.
    Handles corrupt/unreadable files gracefully (logs warning, skips them).
    """
    eval_dir = config.EVALUATION_DIR
    if not eval_dir.exists():
        print("[diagnose] No evaluation directory found, using metrics only")
        return None

    results = []
    for path in sorted(eval_dir.glob("evaluation_*.json")):
        try:
            with open(path) as f:
                results.append(json.load(f))
        except json.JSONDecodeError as e:
            print(f"[diagnose] WARNING: Corrupt evaluation file {path}: {e}")
        except OSError as e:
            print(f"[diagnose] WARNING: Cannot read evaluation file {path}: {e}")

    if not results:
        print("[diagnose] No evaluation files found, using metrics only")
        return None

    print(f"[diagnose] Loaded {len(results)} evaluation(s)")
    return results


def _summarize_experiment_history() -> str:
    """Build a compact text summary of experiment iterations.

    Format:
        Iteration 1: accuracy=0.75, status=accepted
        Iteration 2: accuracy=0.78 (+0.03), changed training.n_estimators 100->200, status=accepted

    Returns "No experiment history yet." if no iterations exist.
    Caps to last MAX_HISTORY_ENTRIES iterations.
    """
    from src.experiment import list_iterations, compare_experiments

    iterations = list_iterations()
    if not iterations:
        return "No experiment history yet. This is the first iteration."

    iterations = iterations[-config.MAX_HISTORY_ENTRIES:]

    lines = []
    for i, entry in enumerate(iterations):
        iteration_num = entry["iteration"]
        accuracy = entry.get("overall_accuracy")
        status = entry.get("status", "unknown")

        acc_str = f"accuracy={accuracy:.4f}" if accuracy is not None else "accuracy=N/A"

        change_str = ""
        if i > 0:
            prev_num = iterations[i - 1]["iteration"]
            try:
                comparison = compare_experiments(prev_num, iteration_num)
                config_diff = comparison.get("config_diff", {})
                if config_diff:
                    parts = []
                    for key, diff in config_diff.items():
                        parts.append(f"{key} {diff['a']}->{diff['b']}")
                    change_str = ", changed " + ", ".join(parts)

                metrics_diff = comparison.get("metrics_diff", {})
                acc_delta_info = metrics_diff.get("overall_accuracy", {})
                acc_delta = acc_delta_info.get("delta") if isinstance(acc_delta_info, dict) else None
                if acc_delta is not None and accuracy is not None:
                    sign = "+" if acc_delta >= 0 else ""
                    acc_str = f"accuracy={accuracy:.4f} ({sign}{acc_delta:.4f})"
            except (FileNotFoundError, KeyError, ValueError):
                print(f"[diagnose] WARNING: Could not compare iterations {prev_num} and {iteration_num}")

        line = f"  Iteration {iteration_num}: {acc_str}{change_str}, status={status}"
        lines.append(line)

    return "Experiment History:\n" + "\n".join(lines)


def _determine_tier(iterations: list[dict]) -> int:
    """Determine which tier of changes to propose based on experiment history.

    Tier 1: Training data, class weights, post-processing, tree depth (safe, incremental)
    Tier 2: Feature engineering, classifier hyperparams (moderate risk)
    Tier 3: Structural changes (high risk)

    Escalation: If 3+ Tier 1 changes tried without net accuracy improvement,
    escalate to Tier 2. If 2+ Tier 2 changes tried without improvement,
    escalate to Tier 3. Tier 2 threshold is lower because it has fewer
    parameters (3 vs 7).
    """
    if not iterations:
        return 1

    from src.experiment import load_experiment
    # NOTE: _flatten_dict is a private API from experiment.py.
    # This cross-module dependency is accepted; the function is stable.
    from src.experiment import _flatten_dict

    tier1_tried = 0
    tier1_improved = 0
    tier2_tried = 0
    tier2_improved = 0

    for idx, entry in enumerate(iterations):
        if entry.get("status") not in ("accepted", "reverted"):
            continue
        if idx == 0:
            continue  # No previous iteration to compare against

        try:
            exp = load_experiment(entry["iteration"])
            prev_entry = iterations[idx - 1]
            prev_exp = load_experiment(prev_entry["iteration"])

            curr_flat = _flatten_dict(exp["config"])
            prev_flat = _flatten_dict(prev_exp["config"])
            changed_keys = {k for k in curr_flat if curr_flat.get(k) != prev_flat.get(k)}

            is_tier1 = bool(changed_keys & TIER_PARAMS[1])
            is_tier2 = bool(changed_keys & TIER_PARAMS[2])

            curr_acc = entry.get("overall_accuracy") or 0
            prev_acc = prev_entry.get("overall_accuracy") or 0
            improved = curr_acc > prev_acc

            if is_tier1:
                tier1_tried += 1
                if improved:
                    tier1_improved += 1
            elif is_tier2:
                tier2_tried += 1
                if improved:
                    tier2_improved += 1
        except Exception:
            continue

    if tier1_tried >= 3 and tier1_improved == 0:
        if tier2_tried >= 2 and tier2_improved == 0:
            return 3
        return 2

    return 1


def _build_diagnosis_prompt(
    metrics: dict,
    evaluation: list[dict] | None,
    history_summary: str,
    current_config: dict,
    tier: int,
) -> tuple[str, str]:
    """Build system and user prompts for Claude diagnosis.

    Returns (system_prompt, user_prompt).
    """
    # System prompt: role, constraints, output format
    # NOTE: Keep parameter constraints in sync with src/experiment.py validate_config()
    system_prompt = (
        "You are a remote sensing landcover classification diagnostician. "
        "You analyze evaluation results from a satellite landcover classification "
        "pipeline and propose exactly ONE hypothesis for what parameter change to "
        "make next.\n\n"
        "RULES:\n"
        "1. Propose exactly ONE change per iteration\n"
        "2. All parameter changes must be in a SINGLE component "
        "(training, features, or post_processing)\n"
        "3. Respect parameter constraints:\n"
        "   - training.n_estimators: int, 1-1000\n"
        "   - training.max_depth: int 1-100 or null\n"
        "   - training.max_samples_per_class: int, 1-100000\n"
        "   - training.class_weight: 'balanced', 'balanced_subsample', 'none', or null\n"
        "     (use the string 'none' for no class weighting, not JSON null)\n"
        "   - training.exclude_boundary_pixels: bool\n"
        "   - training.boundary_buffer_px: int >= 0 (>= 1 if exclude_boundary_pixels is true)\n"
        "   - training.random_state: int >= 0\n"
        "   - features.use_embeddings: bool (at least one feature source must be true)\n"
        "   - features.add_ndvi: bool\n"
        "   - features.add_ndwi: bool\n"
        "   - post_processing.mode_filter_size: 0 (disabled) or odd int >= 3, < 512\n"
        "   - post_processing.min_mapping_unit_px: int >= 0\n"
        "4. Tier escalation: Stay within the specified tier unless all its options are exhausted\n"
        "5. Do NOT propose changes that repeat a reverted experiment\n"
        "6. If you identify a class taxonomy issue (e.g., classes should be merged), "
        "express this through the 'reasoning' field of a 'training' component hypothesis "
        "that addresses the symptom via available parameters\n\n"
        "OUTPUT FORMAT (strict JSON, no markdown wrapping):\n"
        "{\n"
        '  "hypothesis": "Natural language WHY this change should help",\n'
        '  "component": "training|features|post_processing",\n'
        '  "parameter_changes": {"dotted.param.name": new_value},\n'
        '  "expected_impact": "What should improve and by roughly how much",\n'
        '  "risk": "What might get worse",\n'
        f'  "tier": {tier},\n'
        '  "confidence": 0.0-1.0,\n'
        '  "reasoning": "Step-by-step analysis chain"\n'
        "}"
    )

    # User prompt: all runtime data
    sections = []

    # Section 1: Current metrics
    per_class = metrics.get("per_class", {})
    per_class_lines = []
    for cls_name, scores in per_class.items():
        per_class_lines.append(
            f"  {cls_name}: P={scores.get('precision', 0):.2f} "
            f"R={scores.get('recall', 0):.2f} "
            f"F1={scores.get('f1', 0):.2f} "
            f"support={scores.get('support', 0)}"
        )
    sections.append(
        f"## Current Classification Metrics\n"
        f"Overall accuracy: {metrics.get('overall_accuracy', 'N/A')}\n"
        f"Training accuracy: {metrics.get('training_accuracy', 'N/A')}\n"
        f"Per-class:\n" + "\n".join(per_class_lines)
    )

    # Section 2: Confusion matrix (labeled table format)
    cm = metrics.get("confusion_matrix")
    class_names = metrics.get("class_names", [])
    if cm and class_names:
        header = "Predicted ->  " + "  ".join(f"{n:>10}" for n in class_names)
        rows = []
        for i, row in enumerate(cm):
            label = class_names[i] if i < len(class_names) else f"Class {i}"
            rows.append(f"{label:>12}  " + "  ".join(f"{v:>10}" for v in row))
        sections.append(
            f"## Confusion Matrix (rows=true, cols=predicted)\n{header}\n" + "\n".join(rows)
        )

    # Section 3: VLM evaluation (optional, loosely coupled)
    if evaluation:
        sections.append(f"## VLM Evaluation Results\n{json.dumps(evaluation, indent=2)}")

    # Section 4: Experiment history
    sections.append(f"## {history_summary}")

    # Section 5: Current config
    sections.append(f"## Current Configuration\n{json.dumps(current_config, indent=2)}")

    # Section 6: Tier constraint
    tier_desc = {
        1: "Tier 1 (safe): training.max_samples_per_class, training.exclude_boundary_pixels, "
           "training.boundary_buffer_px, training.class_weight, training.max_depth, "
           "post_processing.mode_filter_size, post_processing.min_mapping_unit_px",
        2: "Tier 2 (moderate): training.n_estimators, "
           "features.add_ndvi, features.add_ndwi",
        3: "Tier 3 (aggressive): features.add_spatial_context (warning: not yet implemented), "
           "combinations of parameters from lower tiers",
    }
    sections.append(
        f"## Current Tier: {tier}\n"
        f"You MUST propose changes from: {tier_desc.get(tier, tier_desc[1])}"
    )

    user_prompt = "\n\n".join(sections)
    return system_prompt, user_prompt


def _parse_hypothesis_response(response_text: str) -> dict:
    """Extract JSON from Claude's response text.

    Tries three strategies in order:
    1. Extract from last ```json ... ``` fenced block
    2. Parse the entire response as JSON
    3. Find first '{' and last '}' and parse that substring

    Raises ValueError if all strategies fail.
    """
    text = response_text.strip()

    # Strategy 1: Last fenced JSON block
    try:
        if "```json" in text:
            start = text.rfind("```json") + len("```json")
            end = text.find("```", start)
            if end != -1:
                return json.loads(text[start:end].strip())
        elif "```" in text:
            marker = text.rfind("```")
            # Find the opening ``` before this closing one
            block_start = text.rfind("```", 0, marker)
            if block_start != -1:
                content_start = text.find("\n", block_start)
                if content_start != -1:
                    return json.loads(text[content_start + 1:marker].strip())
    except (json.JSONDecodeError, ValueError):
        pass  # Fall through to next strategy

    # Strategy 2: Raw JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Substring extraction
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Failed to parse hypothesis from Claude response. "
        f"Raw response:\n{text[:500]}"
    )


def _call_claude(system_prompt: str, user_prompt: str) -> Hypothesis:
    """Call Claude API and return a validated Hypothesis.

    Includes one retry for transient errors (rate limits, timeouts) and
    parsing failures (Claude output is non-deterministic).

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY not set or API fails after retries.
        ValueError: If Claude's response cannot be parsed into a valid Hypothesis
                    after retries.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. "
            "Set it in your environment or use the rule-based fallback."
        )

    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic package not installed. "
            "Install it: uv pip install 'olmoearth-uk-landcover[all]'"
        )

    from pydantic import ValidationError

    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Anthropic client: {e}"
        ) from e

    raw_text = ""
    last_error = None
    for attempt in range(2):  # One initial attempt + one retry
        try:
            print(f"[diagnose] Calling Claude ({config.DIAGNOSE_MODEL})...")
            response = client.messages.create(
                model=config.DIAGNOSE_MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            if not response.content:
                raise ValueError("Empty response from Claude API (no content blocks)")
            raw_text = getattr(response.content[0], "text", None) or ""
            if not raw_text.strip():
                raise ValueError("Empty text in Claude API response")

            # Parse JSON from response
            data = _parse_hypothesis_response(raw_text)

            # Validate through Pydantic
            return Hypothesis(**data)

        except anthropic.AuthenticationError:
            raise RuntimeError(
                "Anthropic API authentication failed. "
                "Check your ANTHROPIC_API_KEY."
            )
        except (anthropic.RateLimitError, anthropic.APITimeoutError,
                anthropic.APIConnectionError) as e:
            last_error = e
            if attempt == 0:
                import time
                wait = 3
                print(f"[diagnose] API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            continue
        except anthropic.APIError as e:
            raise RuntimeError(
                f"Anthropic API call failed: {e}. "
                f"Check ANTHROPIC_API_KEY and network connectivity."
            ) from e
        except ValidationError as e:
            # Structural issue -- Pydantic validation failure is unlikely to
            # resolve on retry (same prompt produces similar structure)
            raise ValueError(
                f"Claude's response failed Pydantic validation: {e}. "
                f"Raw response:\n{raw_text[:500]}"
            ) from e
        except (json.JSONDecodeError, ValueError) as e:
            # Parsing failure -- Claude output is non-deterministic, retry once
            last_error = e
            if attempt == 0:
                print(f"[diagnose] Parse error: {e}. Retrying...")
                continue
            raise ValueError(
                f"Failed to parse Claude's response after 2 attempts: {e}. "
                f"Raw response:\n{raw_text[:500]}"
            ) from e

    raise RuntimeError(
        f"Claude API failed after 2 attempts. Last error: {last_error}"
    )


def _rule_based_diagnosis(
    metrics: dict,
    current_config: dict,
    tier: int,
) -> Hypothesis:
    """Generate a hypothesis using simple rules based on per-class metrics.

    This is the fallback when ANTHROPIC_API_KEY is not set.
    Rules are ordered by priority. Each rule checks preconditions and
    returns a Hypothesis if applicable.

    Known limitation: The rule-based engine is stateless -- it does not check
    experiment history to avoid re-proposing reverted changes. The Claude-based
    path handles history-aware deduplication via the prompt.
    """
    per_class = metrics.get("per_class", {})
    training = current_config.get("training", {})
    features = current_config.get("features", {})
    post_proc = current_config.get("post_processing", {})

    training_acc = metrics.get("training_accuracy", 0)
    overall_acc = metrics.get("overall_accuracy", 0)

    # Handle empty per_class metrics (RT4-E1)
    if not per_class:
        return Hypothesis(
            hypothesis="No per-class metrics available. Increasing training samples "
                       "as a baseline intervention.",
            component="training",
            parameter_changes={"training.max_samples_per_class": 10000},
            expected_impact="More training data should produce meaningful class metrics",
            risk="Slower training",
            tier=1,
            confidence=0.5,
            reasoning="Rule-based: per_class metrics dict is empty, "
                      "proposing sample increase as default.",
        )

    # Find worst-performing class
    worst_class = None
    worst_f1 = 1.0
    worst_recall = 1.0
    for name, scores in per_class.items():
        if scores.get("support", 0) == 0:
            continue  # Skip zero-support classes
        f1 = scores.get("f1", 1.0)
        if f1 < worst_f1:
            worst_f1 = f1
            worst_class = name
            worst_recall = scores.get("recall", 0)

    if worst_class is None:
        worst_class = "Unknown"

    # Rule 1 (Tier 1): Overfitting detection
    if tier <= 1 and training_acc and overall_acc and (training_acc - overall_acc) > 0.15:
        new_depth = max((training.get("max_depth") or 20) - 5, 5)
        return Hypothesis(
            hypothesis=f"Overfitting detected: training accuracy ({training_acc:.2f}) >> "
                       f"test accuracy ({overall_acc:.2f}). Reducing tree depth.",
            component="training",
            parameter_changes={"training.max_depth": new_depth},
            expected_impact="Reduce overfitting gap, improve generalization",
            risk="May underfit if depth too shallow",
            tier=1,
            confidence=0.8,
            reasoning=f"Rule-based: training-test gap is {training_acc - overall_acc:.2f} "
                      f"(threshold: 0.15). Reducing max_depth from "
                      f"{training.get('max_depth')} to {new_depth}.",
        )

    # Rule 2 (Tier 1): Class weight adjustment for low recall
    if tier <= 1 and worst_recall < 0.5 and training.get("class_weight") != "balanced":
        return Hypothesis(
            hypothesis=f"{worst_class} has recall={worst_recall:.2f}. "
                       f"Setting class_weight to 'balanced' should help underrepresented classes.",
            component="training",
            parameter_changes={"training.class_weight": "balanced"},
            expected_impact=f"{worst_class} recall should improve",
            risk="May slightly reduce precision for majority classes",
            tier=1,
            confidence=0.75,
            reasoning=f"Rule-based: {worst_class} has lowest F1={worst_f1:.2f}, "
                      f"recall={worst_recall:.2f}. Class weighting addresses imbalance.",
        )

    # Rule 3 (Tier 1): Boundary pixel exclusion
    if tier <= 1 and not training.get("exclude_boundary_pixels") and worst_f1 < 0.6:
        return Hypothesis(
            hypothesis=f"{worst_class} has F1={worst_f1:.2f}. Boundary pixels between "
                       f"classes add noise to training data.",
            component="training",
            parameter_changes={
                "training.exclude_boundary_pixels": True,
                "training.boundary_buffer_px": 2,
            },
            expected_impact=f"Cleaner training data should improve {worst_class} F1",
            risk="Reduces training sample count, may hurt classes with few samples",
            tier=1,
            confidence=0.7,
            reasoning=f"Rule-based: {worst_class} has F1={worst_f1:.2f}. "
                      f"Boundary exclusion removes mixed-pixel noise.",
        )

    # Rule 4 (Tier 1): Enable post-processing
    if tier <= 1 and post_proc.get("mode_filter_size", 0) == 0:
        return Hypothesis(
            hypothesis="No spatial smoothing applied. Mode filter removes "
                       "salt-and-pepper noise.",
            component="post_processing",
            parameter_changes={"post_processing.mode_filter_size": 5},
            expected_impact="Spatially coherent classes (Cropland, Grassland) should improve",
            risk="May smooth out small legitimate features (narrow water bodies)",
            tier=1,
            confidence=0.7,
            reasoning="Rule-based: mode_filter_size=0. Adding 5x5 mode filter.",
        )

    # Rule 5 (Tier 1): Increase samples
    if tier <= 1 and training.get("max_samples_per_class", 5000) < 10000:
        return Hypothesis(
            hypothesis="More training samples should improve generalization.",
            component="training",
            parameter_changes={"training.max_samples_per_class": 10000},
            expected_impact="Better generalization across all classes",
            risk="Slower training, minimal accuracy risk",
            tier=1,
            confidence=0.6,
            reasoning=f"Rule-based: max_samples_per_class="
                      f"{training.get('max_samples_per_class', 5000)}. Doubling to 10000.",
        )

    # Rule 6 (Tier 2): Add NDVI
    if tier <= 2 and not features.get("add_ndvi"):
        return Hypothesis(
            hypothesis=f"{worst_class} confusion may be due to missing vegetation index. "
                       f"NDVI helps discriminate vegetation types.",
            component="features",
            parameter_changes={"features.add_ndvi": True},
            expected_impact="Better vegetation discrimination, especially Cropland vs Grassland",
            risk="Adds feature dimensionality, marginal if embeddings already capture this",
            tier=2,
            confidence=0.7,
            reasoning=f"Rule-based: Tier 2. NDVI not enabled, worst class={worst_class}.",
        )

    # Rule 7 (Tier 2): Increase n_estimators
    if tier <= 2 and training.get("n_estimators", 100) < 300:
        return Hypothesis(
            hypothesis=f"RandomForest with {training.get('n_estimators', 100)} trees "
                       f"may underfit. More trees = more stable predictions.",
            component="training",
            parameter_changes={"training.n_estimators": 300},
            expected_impact="More stable predictions across all classes",
            risk="Slower training (3x), minimal accuracy risk",
            tier=2,
            confidence=0.6,
            reasoning=f"Rule-based: Tier 2. n_estimators={training.get('n_estimators', 100)}.",
        )

    # Rule 8 (Tier 2): Add NDWI
    if tier <= 2 and not features.get("add_ndwi"):
        return Hypothesis(
            hypothesis="Adding NDWI for water body discrimination.",
            component="features",
            parameter_changes={"features.add_ndwi": True},
            expected_impact="Better water/non-water discrimination",
            risk="Marginal: adds one feature dimension",
            tier=2,
            confidence=0.5,
            reasoning="Rule-based: NDWI not enabled.",
        )

    # Exhausted: increase max_depth (with cap check)
    current_depth = training.get("max_depth") or 20
    if current_depth >= 100:
        # Fully exhausted -- convergence signal
        return Hypothesis(
            hypothesis="All standard config interventions have been tried. "
                       "Manual review recommended: consider class merging or "
                       "different embedding strategy.",
            component="training",
            parameter_changes={"training.random_state": training.get("random_state", 42) + 1},
            expected_impact="Minimal -- this is a signal to review the approach",
            risk="Changing random_state alters bootstrap samples and feature subsets, "
                 "producing different accuracy. This is a convergence signal, not a no-op.",
            tier=3,
            confidence=0.2,
            reasoning="Rule-based: all rules exhausted, max_depth already at cap. "
                      "Suggesting random_state change as a convergence signal.",
        )

    new_depth = min(current_depth + 10, 100)
    return Hypothesis(
        hypothesis="All standard interventions tried. Increasing tree depth "
                   "for more complex decision boundaries.",
        component="training",
        parameter_changes={"training.max_depth": new_depth},
        expected_impact="More complex boundaries may capture subtle class differences",
        risk="Possible overfitting (watch training_accuracy vs test accuracy gap)",
        tier=3,
        confidence=0.4,
        reasoning=f"Rule-based: all standard rules exhausted. "
                  f"Increasing max_depth from {current_depth} to {new_depth}.",
    )


def _apply_hypothesis_to_config(current_config: dict, hypothesis: Hypothesis) -> dict:
    """Merge a hypothesis's parameter_changes into a config dict.

    Used for validation only. Does NOT write to disk. Returns a merged dict
    for legality checking via validate_config().

    Converts dot-notation keys to nested dict and merges with current config
    using _deep_merge from src.experiment.
    """
    import copy
    # NOTE: _deep_merge is a private API from experiment.py.
    # It silently drops unknown keys with a warning.
    from src.experiment import _deep_merge

    override = {}
    for dotted_key, value in hypothesis.parameter_changes.items():
        parts = dotted_key.split(".")
        d = override
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    return _deep_merge(copy.deepcopy(current_config), override)


def _validate_hypothesis(hypothesis: Hypothesis, current_config: dict) -> dict:
    """Validate that proposed parameter changes produce a legal config.

    Returns the merged config if valid.
    Raises ValueError if:
    - The resulting config violates constraints (from validate_config)
    - No proposed changes were actually applied (all keys unknown/dropped)
    """
    from src.experiment import validate_config

    merged = _apply_hypothesis_to_config(current_config, hypothesis)

    # Verify at least one change was actually applied (RT3-M1)
    # _deep_merge silently drops unknown keys, so we check the result
    from src.experiment import _flatten_dict
    merged_flat = _flatten_dict(merged)
    changes_applied = False
    for key, value in hypothesis.parameter_changes.items():
        if key in merged_flat and merged_flat[key] == value:
            changes_applied = True
            break
    if not changes_applied:
        raise ValueError(
            f"None of the proposed parameter changes were applied. "
            f"Keys {list(hypothesis.parameter_changes.keys())} may not exist in the config schema."
        )

    validate_config(merged)  # Raises ValueError on invalid config
    return merged


def _save_hypothesis(hypothesis: Hypothesis, iteration: int, source: str) -> Path:
    """Save hypothesis JSON to the experiment's ledger directory.

    The hypothesis is saved to the iteration that was analyzed (not the next
    iteration). The orchestrator (Issue #8) reads hypothesis.json from the
    latest completed iteration to configure the next run.

    Creates the directory if it doesn't exist.
    Adds timestamp and source metadata at persistence layer.
    Uses atomic write pattern (write to .tmp, then rename).
    """
    from datetime import datetime, timezone

    iter_dir = config.EXPERIMENTS_DIR / f"iteration_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    output = hypothesis.model_dump()
    output["timestamp"] = datetime.now(timezone.utc).isoformat()
    output["source"] = source  # "claude" or "rule_based"

    output_path = iter_dir / "hypothesis.json"
    tmp_path = output_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
        f.write("\n")
    tmp_path.rename(output_path)

    print(f"[diagnose] Saved hypothesis to {output_path}")
    return output_path


def run_diagnosis(iteration: int | None = None) -> Hypothesis:
    """Analyze the latest (or specified) experiment and propose a hypothesis.

    Steps:
    1. Load the experiment's metrics
    2. Load VLM evaluation results (if available)
    3. Summarize experiment history
    4. Determine which tier of changes to propose
    5. Call Claude (or fall back to rule-based diagnosis)
    6. Validate the hypothesis against config constraints
    7. Save the hypothesis to the experiment ledger

    NOTE: This function does NOT modify experiment_config.json.
    The orchestrator (Issue #8) is responsible for applying the
    hypothesis's parameter_changes to the live config before
    running the next classification.

    Args:
        iteration: Specific iteration to diagnose. If None, uses the latest.

    Returns:
        A validated Hypothesis object.
    """
    from src.experiment import (
        list_iterations, load_experiment, load_config,
    )

    # Input validation (RT2-6.4)
    if iteration is not None and iteration < 1:
        raise ValueError(f"iteration must be >= 1, got {iteration}")

    # Step 1: Find the iteration to diagnose
    iterations = list_iterations()
    if iteration is not None:
        exp = load_experiment(iteration)
    elif iterations:
        latest = iterations[-1]
        exp = load_experiment(latest["iteration"])
        iteration = latest["iteration"]
    else:
        raise RuntimeError(
            "No experiments in ledger. Run classification first: "
            "uv run python -m src.classify"
        )

    metrics = exp.get("metrics")
    if not metrics:
        raise RuntimeError(
            f"No metrics found for iteration {iteration}. "
            f"The classification may not have completed successfully."
        )

    # Use stored config, fall back to live config (RT2-6.3)
    current_config = exp.get("config") if exp.get("config") is not None else load_config()
    print(f"[diagnose] Analyzing iteration {iteration}, "
          f"accuracy={metrics.get('overall_accuracy', 'N/A')}")

    # Warn about existing hypothesis (with status context)
    hypothesis_path = config.EXPERIMENTS_DIR / f"iteration_{iteration:03d}" / "hypothesis.json"
    if hypothesis_path.exists():
        status = exp.get("metadata", {}).get("status", "unknown")
        print(f"[diagnose] WARNING: hypothesis already exists for iteration {iteration} "
              f"(status={status}), it will be overwritten")

    # Step 2: Load evaluation results (optional)
    evaluation = _load_evaluation_results()

    # Step 3: Summarize history
    history_summary = _summarize_experiment_history()

    # Step 4: Determine tier
    tier = _determine_tier(iterations)
    print(f"[diagnose] Proposing Tier {tier} changes")

    # Step 5: Generate hypothesis
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    source = "unknown"

    if has_api_key:
        system_prompt, user_prompt = _build_diagnosis_prompt(
            metrics, evaluation, history_summary, current_config, tier,
        )
        try:
            hypothesis = _call_claude(system_prompt, user_prompt)
            source = "claude"
        except (RuntimeError, ValueError) as e:
            print(f"[diagnose] Claude API failed: {e}")
            print("[diagnose] Falling back to rule-based diagnosis")
            hypothesis = _rule_based_diagnosis(metrics, current_config, tier)
            source = "rule_based"
    else:
        print("[diagnose] No ANTHROPIC_API_KEY set, using rule-based diagnosis")
        hypothesis = _rule_based_diagnosis(metrics, current_config, tier)
        source = "rule_based"

    # Step 6: Validate against config constraints
    try:
        _validate_hypothesis(hypothesis, current_config)
    except ValueError as e:
        print(f"[diagnose] WARNING: Hypothesis would produce invalid config: {e}")
        if source == "claude":
            print("[diagnose] Falling back to rule-based diagnosis")
            hypothesis = _rule_based_diagnosis(metrics, current_config, tier)
            source = "rule_based"
            try:
                _validate_hypothesis(hypothesis, current_config)
            except Exception as ve:
                raise RuntimeError(
                    f"Rule-based diagnosis produced invalid config: {ve}. "
                    f"This is a bug in _rule_based_diagnosis()."
                ) from ve
        else:
            raise  # Rule-based producing invalid config is a bug

    # Step 7: Save hypothesis to ledger
    _save_hypothesis(hypothesis, iteration, source)

    print(f"[diagnose] Hypothesis: {hypothesis.hypothesis}")
    print(f"[diagnose] Component: {hypothesis.component}")
    print(f"[diagnose] Changes: {hypothesis.parameter_changes}")
    return hypothesis


def main() -> None:
    """CLI entry point: diagnose the latest experiment."""
    run_diagnosis()


if __name__ == "__main__":
    main()

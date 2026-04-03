"""Experiment configuration and ledger management.

Load/save/compare experiment configs and track iterations with metrics.

Usage:
    uv run python -m src.experiment
"""
from __future__ import annotations

from pathlib import Path

from src import config

DEFAULT_CONFIG: dict = {
    "training": {
        "max_samples_per_class": 5000,
        "exclude_boundary_pixels": False,
        "boundary_buffer_px": 0,
        "classifier": "RandomForest",
        "n_estimators": 100,
        "max_depth": 20,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "features": {
        "use_embeddings": True,
        "add_ndvi": False,
        "add_ndwi": False,
        "add_spatial_context": False,
    },
    "post_processing": {
        "mode_filter_size": 0,
        "min_mapping_unit_px": 0,
    },
}


def _deep_merge(base: dict, override: dict, prefix: str = "") -> dict:
    """Recursively merge override into a deep copy of base.

    Unknown keys produce a warning and are ignored.
    """
    import copy

    result = copy.deepcopy(base)

    for key in override:
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in result:
            print(f"[experiment] WARNING: unknown key '{full_key}' in config, ignoring")
            continue
        if isinstance(result[key], dict) and isinstance(override[key], dict):
            result[key] = _deep_merge(result[key], override[key], prefix=full_key)
        else:
            result[key] = copy.deepcopy(override[key])

    return result


def load_config(config_path: Path | None = None) -> dict:
    """Load experiment config from JSON, falling back to defaults.

    If the config file does not exist, prints a message and returns defaults.
    Validates the merged config before returning.
    """
    import json
    import copy

    if config_path is None:
        config_path = config.PROJECT_ROOT / "experiment_config.json"

    if not config_path.exists():
        print(f"[experiment] No config file found at {config_path}, using defaults")
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        validate_config(cfg)
        return cfg

    try:
        with open(config_path) as f:
            user_config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {config_path} at line {e.lineno}, column {e.colno}: {e.msg}. "
            f"Fix the JSON or delete the file to use defaults."
        ) from e

    merged = _deep_merge(DEFAULT_CONFIG, user_config)
    validate_config(merged)
    return merged


def save_config(cfg: dict, config_path: Path | None = None) -> None:
    """Save experiment config to JSON with atomic write.

    Validates config before saving.
    """
    import json

    validate_config(cfg)

    if config_path is None:
        config_path = config.PROJECT_ROOT / "experiment_config.json"

    tmp_path = config_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    tmp_path.rename(config_path)


def validate_config(cfg: dict) -> None:
    """Validate experiment config values.

    Raises ValueError with descriptive messages for invalid values.
    Maps class_weight "none" to Python None.
    """
    training = cfg.get("training", {})
    features = cfg.get("features", {})
    post_processing = cfg.get("post_processing", {})

    # --- training section ---
    if training.get("classifier") != "RandomForest":
        raise ValueError(
            f"training.classifier must be 'RandomForest', got {training.get('classifier')!r}"
        )

    n_est = training.get("n_estimators")
    if not isinstance(n_est, int) or n_est < 1 or n_est > 1000:
        raise ValueError(
            f"training.n_estimators must be int between 1 and 1000, got {n_est!r}"
        )

    max_depth = training.get("max_depth")
    if max_depth is not None:
        if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 100:
            raise ValueError(
                f"training.max_depth must be int between 1 and 100, or null, got {max_depth!r}"
            )

    max_samples = training.get("max_samples_per_class")
    if not isinstance(max_samples, int) or max_samples < 1 or max_samples > 100000:
        raise ValueError(
            f"training.max_samples_per_class must be int between 1 and 100000, "
            f"got {max_samples!r}"
        )

    buffer_px = training.get("boundary_buffer_px")
    if not isinstance(buffer_px, int) or buffer_px < 0:
        raise ValueError(
            f"training.boundary_buffer_px must be int >= 0, got {buffer_px!r}"
        )

    exclude_boundary = training.get("exclude_boundary_pixels")
    if not isinstance(exclude_boundary, bool):
        raise ValueError(
            f"training.exclude_boundary_pixels must be bool, got {exclude_boundary!r}"
        )

    if exclude_boundary and buffer_px < 1:
        raise ValueError(
            "training.boundary_buffer_px must be >= 1 when exclude_boundary_pixels is True"
        )

    class_weight = training.get("class_weight")
    if class_weight == "none":
        cfg["training"]["class_weight"] = None
    elif class_weight not in ("balanced", "balanced_subsample", None):
        raise ValueError(
            f"training.class_weight must be 'balanced', 'balanced_subsample', 'none', or null, "
            f"got {class_weight!r}"
        )

    random_state = training.get("random_state")
    if not isinstance(random_state, int) or random_state < 0:
        raise ValueError(
            f"training.random_state must be non-negative int, got {random_state!r}"
        )

    # --- features section ---
    use_emb = features.get("use_embeddings")
    add_ndvi = features.get("add_ndvi")
    add_ndwi = features.get("add_ndwi")
    add_spatial = features.get("add_spatial_context")

    for name, val in [("use_embeddings", use_emb), ("add_ndvi", add_ndvi),
                       ("add_ndwi", add_ndwi), ("add_spatial_context", add_spatial)]:
        if not isinstance(val, bool):
            raise ValueError(f"features.{name} must be bool, got {val!r}")

    if not any([use_emb, add_ndvi, add_ndwi]):
        raise ValueError(
            "At least one of use_embeddings, add_ndvi, add_ndwi must be enabled"
        )

    if add_spatial:
        print("[experiment] WARNING: add_spatial_context is not yet implemented, ignoring")

    # --- post_processing section ---
    mode_size = post_processing.get("mode_filter_size")
    if not isinstance(mode_size, int) or mode_size < 0:
        raise ValueError(
            f"post_processing.mode_filter_size must be non-negative int, got {mode_size!r}"
        )
    if mode_size == 1:
        raise ValueError(
            "post_processing.mode_filter_size must be 0 (disabled) or odd >= 3, got 1"
        )
    if mode_size > 0 and (mode_size % 2 == 0):
        raise ValueError(
            f"post_processing.mode_filter_size must be odd, got {mode_size}"
        )
    if mode_size >= 512:
        raise ValueError(
            f"post_processing.mode_filter_size must be < 512, got {mode_size}"
        )
    if mode_size > 7:
        print(
            f"[experiment] WARNING: mode_filter_size={mode_size} may be slow "
            f"for images > 1024x1024"
        )

    mmu = post_processing.get("min_mapping_unit_px")
    if not isinstance(mmu, int) or mmu < 0:
        raise ValueError(
            f"post_processing.min_mapping_unit_px must be non-negative int, got {mmu!r}"
        )
    if mmu == 1:
        print(
            "[experiment] WARNING: min_mapping_unit_px=1 is effectively a no-op, "
            "consider 0 or >= 2"
        )


def validate_prerequisites(cfg: dict, years: list[str]) -> None:
    """Check that required band files exist for configured spectral indices.

    Does NOT check embedding files (load_embeddings handles that).
    """
    features = cfg["features"]

    required_bands: set[str] = set()
    if features["add_ndvi"]:
        required_bands.update(["B04", "B08"])
    if features["add_ndwi"]:
        required_bands.update(["B03", "B08"])

    for year in years:
        for band_name in sorted(required_bands):
            path = config.SENTINEL2_DIR / year / f"{band_name}.tif"
            if not path.exists():
                raise FileNotFoundError(
                    f"Band {band_name} not found: {path}. "
                    f"Run `python -m src.acquire` first."
                )


def get_next_iteration_number() -> int:
    """Get the next iteration number by scanning existing experiment directories.

    Returns 1 if no iterations exist.
    """
    experiments_dir = config.EXPERIMENTS_DIR
    if not experiments_dir.exists():
        return 1

    max_num = 0
    for entry in experiments_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("iteration_"):
            suffix = name[len("iteration_"):]
            try:
                num = int(suffix)
                if num > max_num:
                    max_num = num
            except ValueError:
                continue

    return max_num + 1


def save_experiment(cfg: dict, metrics: dict, status: str = "pending") -> tuple[Path, int]:
    """Save experiment config, metrics, and metadata to the ledger.

    Creates experiments/iteration_NNN/ with three JSON files.
    Uses atomic writes (tmp + rename) for each file.

    Returns (iteration_dir_path, iteration_number).
    """
    import json
    import copy
    import datetime

    iteration_num = get_next_iteration_number()
    iteration_dir = config.EXPERIMENTS_DIR / f"iteration_{iteration_num:03d}"

    try:
        iteration_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RuntimeError(
            f"Directory {iteration_dir} already exists. Possible concurrent run."
        )

    metadata = {
        "iteration": iteration_num,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": status,
    }

    files_to_write = {
        "config.json": copy.deepcopy(cfg),
        "metrics.json": metrics,
        "metadata.json": metadata,
    }

    # Write all files as .tmp first
    for filename, data in files_to_write.items():
        tmp_file = iteration_dir / f"{filename}.tmp"
        with open(tmp_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.write("\n")

    # Rename all .tmp files to final names
    for filename in files_to_write:
        tmp_file = iteration_dir / f"{filename}.tmp"
        final_file = iteration_dir / filename
        tmp_file.rename(final_file)

    return iteration_dir, iteration_num


def load_experiment(iteration: int) -> dict:
    """Load an experiment's config, metrics, and metadata.

    Returns {"config": ..., "metrics": ..., "metadata": ...}.
    Raises FileNotFoundError if the iteration directory or any file is missing.
    """
    import json

    iteration_dir = config.EXPERIMENTS_DIR / f"iteration_{iteration:03d}"
    if not iteration_dir.exists():
        raise FileNotFoundError(
            f"Experiment iteration_{iteration:03d} not found at {iteration_dir}"
        )

    result = {}
    for filename in ["config.json", "metrics.json", "metadata.json"]:
        filepath = iteration_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Missing {filename} in {iteration_dir}"
            )
        with open(filepath) as f:
            result[filename.replace(".json", "")] = json.load(f)

    return result


def list_iterations() -> list[dict]:
    """List all experiment iterations with metadata.

    Returns list of dicts sorted by iteration number, each containing:
    iteration, timestamp, status, overall_accuracy.
    """
    import json

    experiments_dir = config.EXPERIMENTS_DIR
    if not experiments_dir.exists():
        return []

    iterations = []
    for entry in sorted(experiments_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("iteration_"):
            continue
        suffix = name[len("iteration_"):]
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

        # Try to load overall_accuracy from metrics
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


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict to dot-notation keys."""
    flat = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def compare_experiments(iter_a: int, iter_b: int) -> dict:
    """Compare two experiments, showing config diffs and metric deltas.

    Returns dict with config_diff and metrics_diff.
    """
    exp_a = load_experiment(iter_a)
    exp_b = load_experiment(iter_b)

    # Flatten configs for comparison
    flat_a = _flatten_dict(exp_a["config"])
    flat_b = _flatten_dict(exp_b["config"])

    all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))
    config_diff = {}
    for key in all_keys:
        val_a = flat_a.get(key)
        val_b = flat_b.get(key)
        if val_a != val_b:
            config_diff[key] = {"a": val_a, "b": val_b}

    # Metrics comparison
    metrics_a = exp_a["metrics"]
    metrics_b = exp_b["metrics"]

    acc_a = metrics_a.get("overall_accuracy", 0)
    acc_b = metrics_b.get("overall_accuracy", 0)

    metrics_diff = {
        "overall_accuracy": {
            "a": acc_a,
            "b": acc_b,
            "delta": round(acc_b - acc_a, 4),
        },
    }

    # Per-class F1 comparison
    per_class_a = metrics_a.get("per_class", {})
    per_class_b = metrics_b.get("per_class", {})
    all_classes = sorted(set(per_class_a.keys()) | set(per_class_b.keys()))

    per_class_diff = {}
    for class_name in all_classes:
        ca = per_class_a.get(class_name, {})
        cb = per_class_b.get(class_name, {})
        f1_a = ca.get("f1", 0)
        f1_b = cb.get("f1", 0)
        per_class_diff[class_name] = {
            "f1": {
                "a": f1_a,
                "b": f1_b,
                "delta": round(f1_b - f1_a, 4),
            }
        }

    metrics_diff["per_class"] = per_class_diff

    return {"config_diff": config_diff, "metrics_diff": metrics_diff}


def update_experiment_status(iteration: int, status: str) -> None:
    """Update the status of an experiment iteration.

    Valid statuses: pending, accepted, reverted.
    """
    import json

    valid_statuses = ("pending", "accepted", "reverted")
    if status not in valid_statuses:
        raise ValueError(
            f"Status must be one of: {', '.join(valid_statuses)}, got {status!r}"
        )

    iteration_dir = config.EXPERIMENTS_DIR / f"iteration_{iteration:03d}"
    metadata_path = iteration_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Experiment iteration_{iteration:03d} not found at {iteration_dir}"
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    metadata["status"] = status

    tmp_path = metadata_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
        f.write("\n")
    tmp_path.rename(metadata_path)


def run_experiment() -> None:
    """Run the classification pipeline (convenience entry point)."""
    # Lazy import to avoid circular dependency
    from src.classify import run_classification
    run_classification()


def main() -> None:
    """CLI entry point: show current config and recent experiments."""
    import json

    cfg = load_config()
    print("[experiment] Current config:")
    print(json.dumps(cfg, indent=2))

    iterations = list_iterations()
    if iterations:
        print(f"\n[experiment] Last {min(3, len(iterations))} experiments:")
        for item in iterations[-3:]:
            acc = item.get("overall_accuracy")
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            print(f"  iteration_{item['iteration']:03d}: "
                  f"accuracy={acc_str}, status={item['status']}")
    else:
        print("\n[experiment] No experiments recorded yet.")


if __name__ == "__main__":
    main()

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
        # Preprocessing (applied in order: L2 norm -> scale -> PCA)
        "pca_components": 0,        # 0 = disabled; valid range 2-500 when enabled
        "scale_features": False,
        "l2_normalize": False,
        # RandomForest / GradientBoosting shared
        "n_estimators": 100,
        "max_depth": 20,
        # RandomForest / SVM / LogisticRegression
        "class_weight": "balanced",
        # GradientBoosting
        "learning_rate": 0.1,
        # SVM
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        # LogisticRegression / MLP shared
        "max_iter": 1000,
        # KNN
        "n_neighbors": 5,
        "weights": "uniform",
        # MLP
        "hidden_layer_sizes": [100],
        "alpha": 0.0001,
        # Shared
        "random_state": 42,
    },
    "features": {
        "use_embeddings": True,
        "add_ndvi": False,
        "add_ndwi": False,
        "add_spatial_context": False,
        "spatial_context_size": 3,
        "spatial_context_stats": ["mean"],
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


VALID_CLASSIFIERS: set[str] = {
    "RandomForest", "GradientBoosting", "SVM",
    "LogisticRegression", "KNN", "MLP",
}


def _validate_classifier_params(classifier: str, training: dict) -> None:
    """Validate hyperparameters specific to the active classifier.

    Only validates params relevant to the given classifier type.
    Params for other classifiers are silently preserved.
    """
    if classifier in ("RandomForest", "GradientBoosting"):
        n_est = training.get("n_estimators")
        if not isinstance(n_est, int) or n_est < 1 or n_est > 1000:
            raise ValueError(
                f"training.n_estimators must be int between 1 and 1000, got {n_est!r}"
            )
        max_depth = training.get("max_depth")
        if max_depth is not None:
            if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 100:
                raise ValueError(
                    f"training.max_depth must be int between 1 and 100, or null, "
                    f"got {max_depth!r}"
                )

    if classifier in ("RandomForest", "SVM", "LogisticRegression"):
        class_weight = training.get("class_weight")
        if class_weight == "none":
            training["class_weight"] = None
        elif class_weight not in ("balanced", "balanced_subsample", None):
            raise ValueError(
                f"training.class_weight must be 'balanced', 'balanced_subsample', "
                f"'none', or null, got {class_weight!r}"
            )

    if classifier == "GradientBoosting":
        lr = training.get("learning_rate")
        if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1.0:
            raise ValueError(
                f"training.learning_rate must be float in (0, 1.0], got {lr!r}"
            )

    if classifier in ("SVM", "LogisticRegression"):
        c_val = training.get("C")
        if not isinstance(c_val, (int, float)) or c_val <= 0:
            raise ValueError(
                f"training.C must be float > 0, got {c_val!r}"
            )

    if classifier == "SVM":
        kernel = training.get("kernel")
        if kernel not in ("rbf", "linear", "poly", "sigmoid"):
            raise ValueError(
                f"training.kernel must be 'rbf', 'linear', 'poly', or 'sigmoid', "
                f"got {kernel!r}"
            )
        gamma = training.get("gamma")
        if gamma not in ("scale", "auto"):
            if not isinstance(gamma, (int, float)) or gamma <= 0:
                raise ValueError(
                    f"training.gamma must be 'scale', 'auto', or float > 0, "
                    f"got {gamma!r}"
                )

    if classifier in ("LogisticRegression", "MLP"):
        max_iter = training.get("max_iter")
        if not isinstance(max_iter, int) or max_iter < 100 or max_iter > 10000:
            raise ValueError(
                f"training.max_iter must be int between 100 and 10000, got {max_iter!r}"
            )

    if classifier == "KNN":
        n_neighbors = training.get("n_neighbors")
        if not isinstance(n_neighbors, int) or n_neighbors < 1 or n_neighbors > 100:
            raise ValueError(
                f"training.n_neighbors must be int between 1 and 100, "
                f"got {n_neighbors!r}"
            )
        weights = training.get("weights")
        if weights not in ("uniform", "distance"):
            raise ValueError(
                f"training.weights must be 'uniform' or 'distance', got {weights!r}"
            )

    if classifier == "MLP":
        hidden = training.get("hidden_layer_sizes")
        if not isinstance(hidden, list) or len(hidden) == 0:
            raise ValueError(
                f"training.hidden_layer_sizes must be a non-empty list of positive "
                f"ints, got {hidden!r}"
            )
        if not all(isinstance(x, int) and x > 0 for x in hidden):
            raise ValueError(
                f"training.hidden_layer_sizes must contain only positive ints, "
                f"got {hidden!r}"
            )
        alpha = training.get("alpha")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise ValueError(
                f"training.alpha must be float >= 0, got {alpha!r}"
            )


def _validate_preprocessing_params(training: dict) -> None:
    """Validate preprocessing configuration flags."""
    pca = training.get("pca_components")
    if not isinstance(pca, int) or pca < 0:
        raise ValueError(
            f"training.pca_components must be non-negative int (0=disabled), "
            f"got {pca!r}"
        )
    if pca > 0 and (pca < 2 or pca > 500):
        raise ValueError(
            f"training.pca_components must be 0 (disabled) or 2-500, got {pca}"
        )

    scale = training.get("scale_features")
    if not isinstance(scale, bool):
        raise ValueError(f"training.scale_features must be bool, got {scale!r}")

    l2 = training.get("l2_normalize")
    if not isinstance(l2, bool):
        raise ValueError(f"training.l2_normalize must be bool, got {l2!r}")


def validate_config(cfg: dict) -> None:
    """Validate experiment config values.

    Raises ValueError with descriptive messages for invalid values.
    Maps class_weight "none" to Python None for classifiers that support it.
    """
    training = cfg.get("training", {})
    features = cfg.get("features", {})
    post_processing = cfg.get("post_processing", {})

    # --- training section ---
    classifier = training.get("classifier")
    if classifier not in VALID_CLASSIFIERS:
        raise ValueError(
            f"training.classifier must be one of {sorted(VALID_CLASSIFIERS)}, "
            f"got {classifier!r}"
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

    random_state = training.get("random_state")
    if not isinstance(random_state, int) or random_state < 0:
        raise ValueError(
            f"training.random_state must be non-negative int, got {random_state!r}"
        )

    # Per-classifier hyperparameter validation
    _validate_classifier_params(classifier, training)

    # Preprocessing validation
    _validate_preprocessing_params(training)

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

    # --- spatial context validation ---
    ctx_size = features.get("spatial_context_size")
    if not isinstance(ctx_size, int) or ctx_size < 3:
        raise ValueError(
            f"features.spatial_context_size must be int >= 3, got {ctx_size!r}"
        )
    if ctx_size % 2 == 0:
        raise ValueError(
            f"features.spatial_context_size must be odd, got {ctx_size}"
        )
    if ctx_size > 99:
        raise ValueError(
            f"features.spatial_context_size must be <= 99, got {ctx_size}"
        )
    if ctx_size > 15:
        print(
            f"[experiment] WARNING: spatial_context_size={ctx_size} is large, "
            f"may be slow for 512x512 images"
        )

    VALID_STATS = {"mean", "std", "max", "min"}
    ctx_stats = features.get("spatial_context_stats")
    if not isinstance(ctx_stats, list) or len(ctx_stats) == 0:
        raise ValueError(
            f"features.spatial_context_stats must be non-empty list, got {ctx_stats!r}"
        )
    invalid_stats = set(ctx_stats) - VALID_STATS
    if invalid_stats:
        raise ValueError(
            f"features.spatial_context_stats contains invalid values: {invalid_stats}. "
            f"Valid options: {sorted(VALID_STATS)}"
        )
    if len(ctx_stats) != len(set(ctx_stats)):
        raise ValueError(
            f"features.spatial_context_stats contains duplicates: {ctx_stats}"
        )

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

    experiments_dir = config.EXPERIMENTS_DIR
    if not experiments_dir.exists() and not experiments_dir.is_symlink():
        raise RuntimeError(
            f"Experiments directory {experiments_dir} does not exist. "
            f"Call create_session() before saving experiments."
        )

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


def load_experiment(iteration: int, session_dir: Path | None = None) -> dict:
    """Load an experiment's config, metrics, and metadata.

    Args:
        iteration: Iteration number to load.
        session_dir: Session directory to load from. Defaults to config.EXPERIMENTS_DIR.

    Returns {"config": ..., "metrics": ..., "metadata": ...}.
    Raises FileNotFoundError if the iteration directory or any file is missing.
    """
    import json

    experiments_dir = session_dir or config.EXPERIMENTS_DIR
    iteration_dir = experiments_dir / f"iteration_{iteration:03d}"
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


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------


def _update_latest_symlink(base_dir: Path, session_dir: Path) -> None:
    """Atomically update the latest/ symlink to point to session_dir.

    Uses a temporary symlink + rename for atomicity on POSIX systems.
    The symlink target is relative (just the directory name) for portability.
    """
    import os

    symlink_path = base_dir / "latest"

    # Atomic symlink update: create temp symlink, then rename over target
    tmp_link = base_dir / f".latest_tmp_{os.getpid()}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(session_dir.name)  # Relative symlink
    tmp_link.rename(symlink_path)


def _migrate_legacy_experiments(base_dir: Path) -> None:
    """Migrate flat iteration_NNN/ dirs from experiments/ root to experiments/session_legacy/.

    This is a one-time migration. If experiments/session_legacy/ already exists,
    or no flat iteration dirs are found, this is a no-op.
    """
    import shutil
    import json
    import datetime

    legacy_dir = base_dir / "session_legacy"
    if legacy_dir.exists():
        return  # Already migrated

    # Check for flat iteration dirs in experiments/ root
    flat_iters = sorted(
        entry for entry in base_dir.iterdir()
        if entry.is_dir() and entry.name.startswith("iteration_")
    )
    if not flat_iters:
        return  # Nothing to migrate

    print(f"[experiment] Migrating {len(flat_iters)} legacy iterations...")
    legacy_dir.mkdir()

    for iter_dir in flat_iters:
        dst = legacy_dir / iter_dir.name
        shutil.move(str(iter_dir), str(dst))
        print(f"[experiment]   Moved {iter_dir.name} -> session_legacy/{iter_dir.name}")

    # Move SUMMARY.md if present
    summary_path = base_dir / "SUMMARY.md"
    if summary_path.exists():
        shutil.move(str(summary_path), str(legacy_dir / "SUMMARY.md"))

    # Create session_meta.json for the legacy session
    earliest_time = None
    latest_time = None
    n_iters = len(flat_iters)
    best_iter = None
    best_acc = -1.0

    for iter_dir_entry in sorted(legacy_dir.iterdir()):
        if not iter_dir_entry.is_dir() or not iter_dir_entry.name.startswith("iteration_"):
            continue
        meta_path = iter_dir_entry / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            ts = meta.get("timestamp", "")
            if ts:
                if earliest_time is None or ts < earliest_time:
                    earliest_time = ts
                if latest_time is None or ts > latest_time:
                    latest_time = ts
        metrics_path = iter_dir_entry / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            acc = metrics.get("overall_accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                suffix = iter_dir_entry.name[len("iteration_"):]
                try:
                    best_iter = int(suffix)
                except ValueError:
                    pass

    legacy_meta = {
        "session_id": "session_legacy",
        "start_time": earliest_time or "",
        "end_time": latest_time,
        "stop_reason": "migrated_from_flat",
        "best_iteration": best_iter,
        "final_score": best_acc if best_acc > 0 else None,
        "initial_config": None,
        "parameter_count": None,
        "n_iterations": n_iters,
    }

    meta_path = legacy_dir / "session_meta.json"
    tmp_path = meta_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(legacy_meta, f, indent=2, default=str)
        f.write("\n")
    tmp_path.rename(meta_path)

    print(f"[experiment] Legacy migration complete: {n_iters} iterations")


def create_session(initial_config: dict) -> Path:
    """Create a new session directory and update the latest/ symlink.

    Creates experiments/session_YYYYMMDD_HHMMSS/ with session_meta.json.
    Updates experiments/latest/ symlink to point to the new session.
    Triggers legacy migration on first call if needed.

    Returns the session directory Path.
    """
    import datetime
    import json
    import copy

    base_dir = config.EXPERIMENTS_BASE_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    # Migrate legacy flat iterations if present (one-time, best-effort)
    try:
        _migrate_legacy_experiments(base_dir)
    except Exception as e:
        print(f"[experiment] WARNING: Legacy migration failed: {e}")
        print("[experiment] Continuing with session creation.")

    now = datetime.datetime.now(datetime.timezone.utc)
    session_id = f"session_{now.strftime('%Y%m%d_%H%M%S')}"
    session_dir = base_dir / session_id
    session_dir.mkdir(exist_ok=False)

    meta = {
        "session_id": session_id,
        "start_time": now.isoformat(),
        "end_time": None,
        "stop_reason": None,
        "best_iteration": None,
        "final_score": None,
        "initial_config": copy.deepcopy(initial_config),
        "parameter_count": len(_flatten_dict(initial_config)),
        "n_iterations": 0,
    }

    meta_path = session_dir / "session_meta.json"
    tmp_path = meta_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
        f.write("\n")
    tmp_path.rename(meta_path)

    _update_latest_symlink(base_dir, session_dir)

    print(f"[experiment] Created session: {session_id}")
    return session_dir


def list_sessions() -> list[dict]:
    """List all experiment sessions sorted by date (most recent first).

    Returns list of dicts with: session_id, start_time, end_time,
    stop_reason, best_iteration, final_score, n_iterations, path.

    Also triggers legacy migration if flat iteration dirs are found.
    """
    import json

    base_dir = config.EXPERIMENTS_BASE_DIR
    if not base_dir.exists():
        return []

    # Trigger migration if legacy dirs exist (so UI works before autocorrect)
    try:
        _migrate_legacy_experiments(base_dir)
    except Exception:
        pass  # Best effort

    sessions = []
    for entry in sorted(base_dir.iterdir(), reverse=True):
        if not entry.is_dir() or not entry.name.startswith("session_"):
            continue
        if entry.is_symlink():
            continue  # Skip latest/ symlink

        meta_path = entry / "session_meta.json"
        if not meta_path.exists():
            sessions.append({
                "session_id": entry.name,
                "start_time": "",
                "end_time": None,
                "stop_reason": None,
                "best_iteration": None,
                "final_score": None,
                "n_iterations": 0,
                "path": entry,
            })
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        sessions.append({
            "session_id": meta.get("session_id", entry.name),
            "start_time": meta.get("start_time", ""),
            "end_time": meta.get("end_time"),
            "stop_reason": meta.get("stop_reason"),
            "best_iteration": meta.get("best_iteration"),
            "final_score": meta.get("final_score"),
            "n_iterations": meta.get("n_iterations", 0),
            "path": entry,
        })

    sessions.sort(key=lambda s: s["start_time"], reverse=True)
    return sessions


def update_session_meta(session_dir: Path, updates: dict) -> None:
    """Update session_meta.json with new values.

    Args:
        session_dir: Path to the session directory.
        updates: Dict of fields to update (merged into existing meta).
    """
    import json

    meta_path = session_dir / "session_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"session_meta.json not found in {session_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    meta.update(updates)

    tmp_path = meta_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
        f.write("\n")
    tmp_path.rename(meta_path)


def get_session_dir(session_id: str | None = None) -> Path:
    """Get the directory for a session by ID.

    If session_id is None, returns the latest session path (via symlink).
    Raises FileNotFoundError if the session does not exist.
    """
    if session_id is None:
        return config.EXPERIMENTS_DIR  # latest/ symlink

    session_dir = config.EXPERIMENTS_BASE_DIR / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session {session_id!r} not found at {session_dir}")
    return session_dir


def compare_experiments(iter_a: int, iter_b: int, session_dir: Path | None = None) -> dict:
    """Compare two experiments, showing config diffs and metric deltas.

    Args:
        iter_a: First iteration number.
        iter_b: Second iteration number.
        session_dir: Session directory to load from. Defaults to config.EXPERIMENTS_DIR.

    Returns dict with config_diff and metrics_diff.
    """
    exp_a = load_experiment(iter_a, session_dir=session_dir)
    exp_b = load_experiment(iter_b, session_dir=session_dir)

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

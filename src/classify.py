"""Train classifier on OlmoEarth embeddings and produce landcover maps.

Usage:
    uv run python -m src.classify
"""
from __future__ import annotations

from src import config


def load_worldcover_labels():
    """Load WorldCover 2021 and remap to 6 simplified classes."""
    import numpy as np
    import rasterio

    wc_path = config.WORLDCOVER_DIR / "worldcover_2021.tif"
    if not wc_path.exists():
        raise FileNotFoundError(f"WorldCover not found: {wc_path}")

    with rasterio.open(wc_path) as src:
        wc = src.read(1)

    # Remap WorldCover codes to simplified classes
    labels = np.full(wc.shape, 5, dtype=np.uint8)  # Default to "Other"
    for wc_code, class_idx in config.WORLDCOVER_REMAP.items():
        labels[wc == wc_code] = class_idx

    # Print class distribution
    print("[classify] WorldCover label distribution:")
    for idx, name in config.LANDCOVER_CLASSES.items():
        count = (labels == idx).sum()
        pct = count / labels.size * 100
        print(f"  {idx} ({name}): {count:,} pixels ({pct:.1f}%)")

    return labels


def load_embeddings(year: str):
    """Load pre-computed embeddings for a year."""
    import numpy as np

    path = config.EMBEDDINGS_DIR / f"embeddings_{year}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {path}. Run `python -m src.embeddings` first."
        )
    embeddings = np.load(path)
    print(f"[classify] [{year}] Loaded embeddings: {embeddings.shape}")
    return embeddings


def _load_raw_band(year: str, band_name: str):
    """Load a single Sentinel-2 band as float32 from disk."""
    import rasterio
    import numpy as np

    path = config.SENTINEL2_DIR / year / f"{band_name}.tif"
    if not path.exists():
        raise FileNotFoundError(
            f"Band {band_name} not found: {path}. "
            f"Run `python -m src.acquire` first."
        )
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _compute_spectral_index(year: str, index_name: str, expected_h: int, expected_w: int):
    """Compute NDVI or NDWI from raw Sentinel-2 band files."""
    import numpy as np

    band_map = {
        "ndvi": ("B08", "B04"),  # (NIR - Red) / (NIR + Red)
        "ndwi": ("B03", "B08"),  # (Green - NIR) / (Green + NIR)
    }

    if index_name not in band_map:
        raise ValueError(f"Unknown spectral index: {index_name}")

    band_a_name, band_b_name = band_map[index_name]
    band_a = _load_raw_band(year, band_a_name)
    band_b = _load_raw_band(year, band_b_name)

    # Shape validation
    for name, arr in [(band_a_name, band_a), (band_b_name, band_b)]:
        if arr.shape != (expected_h, expected_w):
            raise RuntimeError(
                f"Band {name} shape {arr.shape} doesn't match "
                f"expected ({expected_h}, {expected_w})"
            )

    # Safe division
    denominator = band_a + band_b
    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.where(denominator != 0, (band_a - band_b) / denominator, 0.0)

    # NaN/Inf cleanup and clamping
    index = np.nan_to_num(index, nan=0.0, posinf=1.0, neginf=-1.0)
    index = np.clip(index, -1.0, 1.0)

    print(f"[classify] [{year}] Computed {index_name.upper()}: "
          f"min={index.min():.3f}, max={index.max():.3f}, mean={index.mean():.3f}")
    return index


def _augment_features(embeddings, year: str, cfg: dict):
    """Concatenate spectral indices with embeddings based on config.

    Respects use_embeddings flag: when False, only spectral indices are used.
    validate_config() guarantees at least one feature source is enabled.
    """
    import numpy as np

    features_cfg = cfg["features"]
    h, w, d = embeddings.shape
    arrays = []

    if features_cfg["use_embeddings"]:
        arrays.append(embeddings)

    if features_cfg["add_ndvi"]:
        ndvi = _compute_spectral_index(year, "ndvi", h, w)
        arrays.append(ndvi[:, :, np.newaxis])
        print(f"[classify] [{year}] Added NDVI feature")

    if features_cfg["add_ndwi"]:
        ndwi = _compute_spectral_index(year, "ndwi", h, w)
        arrays.append(ndwi[:, :, np.newaxis])
        print(f"[classify] [{year}] Added NDWI feature")

    if not arrays:
        # Should not happen if validate_config passed, but guard anyway
        raise ValueError("No features enabled. Check config.")

    if len(arrays) == 1:
        result = arrays[0]
    else:
        result = np.concatenate(arrays, axis=-1)

    if result.shape[-1] != d:
        print(f"[classify] [{year}] Feature shape: ({h}, {w}, {d}) -> {result.shape}")

    # Spatial context: append neighborhood statistics of the full feature vector
    if features_cfg.get("add_spatial_context"):
        result = _add_spatial_context(
            result,
            size=features_cfg.get("spatial_context_size", 3),
            stats=features_cfg.get("spatial_context_stats", ["mean"]),
            year=year,
        )

    return result


def _add_spatial_context(
    features, size: int, stats: list[str], year: str
):
    """Compute neighborhood statistics over feature map and concatenate.

    For each channel in the input feature map, compute sliding-window
    statistics (mean, std, max, min) and append them as new channels.

    Args:
        features: (H, W, D) float32 array -- the full augmented feature map.
        size: sliding window size (odd int, e.g. 3 = 3x3 neighborhood).
        stats: list of statistics to compute. Options: mean, std, max, min.
        year: year label for logging only.

    Returns:
        (H, W, D + D * len(stats)) array with context channels appended.
    """
    import numpy as np

    try:
        from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
    except ImportError:
        raise RuntimeError(
            "scipy is required for spatial context features. "
            "Install it: uv add scipy"
        )

    h, w, d = features.shape
    context_arrays = []

    for stat_name in stats:
        stat_result = np.empty((h, w, d), dtype=features.dtype)

        for ch in range(d):
            channel = features[:, :, ch]
            if stat_name == "mean":
                stat_result[:, :, ch] = uniform_filter(
                    channel, size=size, mode="reflect"
                )
            elif stat_name == "max":
                stat_result[:, :, ch] = maximum_filter(
                    channel, size=size, mode="reflect"
                )
            elif stat_name == "min":
                stat_result[:, :, ch] = minimum_filter(
                    channel, size=size, mode="reflect"
                )
            elif stat_name == "std":
                # std = sqrt(E[x^2] - E[x]^2), clamped to avoid sqrt of negatives
                mean_x = uniform_filter(channel, size=size, mode="reflect")
                mean_x2 = uniform_filter(
                    channel ** 2, size=size, mode="reflect"
                )
                variance = np.maximum(mean_x2 - mean_x ** 2, 0.0)
                stat_result[:, :, ch] = np.sqrt(variance)
            else:
                raise ValueError(f"Unknown spatial context stat: {stat_name}")

        context_arrays.append(stat_result)

    context = np.concatenate(context_arrays, axis=-1)
    result = np.concatenate([features, context], axis=-1)

    print(
        f"[classify] [{year}] Added spatial context: "
        f"window={size}x{size}, stats={stats}, "
        f"features ({h}, {w}, {d}) -> {result.shape}"
    )
    return result


def _compute_boundary_mask(labels, buffer_px):
    """Return boolean mask: True = interior pixel (keep), False = boundary (exclude)."""
    import numpy as np

    try:
        from scipy.ndimage import maximum_filter, minimum_filter
    except ImportError:
        raise RuntimeError(
            "scipy is required for boundary pixel exclusion. "
            "Install it: uv add scipy"
        )

    size = 2 * buffer_px + 1
    local_max = maximum_filter(labels.astype(np.int32), size=size)
    local_min = minimum_filter(labels.astype(np.int32), size=size)
    is_interior = local_max == local_min

    excluded = (~is_interior).sum()
    print(f"[classify] Excluded {excluded:,} boundary pixels "
          f"(buffer={buffer_px}px), {is_interior.sum():,} interior remain")

    return is_interior


def _apply_post_processing(landcover, cfg: dict, year: str):
    """Apply mode filter and minimum mapping unit if configured.

    Order: mode filter first, then MMU.
    """
    pp = cfg["post_processing"]
    mode_size = pp.get("mode_filter_size", 0)
    mmu = pp.get("min_mapping_unit_px", 0)

    if mode_size > 0:
        landcover = _apply_mode_filter(landcover, mode_size, year)

    if mmu > 0:
        landcover = _apply_min_mapping_unit(landcover, mmu, year)

    return landcover


def _apply_mode_filter(landcover, filter_size: int, year: str):
    """Apply spatial mode filter to smooth classification noise."""
    import numpy as np

    try:
        from scipy.ndimage import generic_filter
    except ImportError:
        raise RuntimeError(
            "scipy is required for mode filter post-processing. "
            "Install it: uv add scipy"
        )

    def _mode(values):
        return np.bincount(values.astype(int)).argmax()

    print(f"[classify] [{year}] Applying mode filter (size={filter_size})...")
    result = generic_filter(
        landcover.astype(np.float64), _mode, size=filter_size, mode="nearest"
    ).astype(np.uint8)

    changed = (result != landcover).sum()
    print(f"[classify] [{year}] Mode filter changed {changed:,}/{landcover.size:,} pixels "
          f"({changed / landcover.size * 100:.1f}%)")

    return result


def _apply_min_mapping_unit(landcover, min_px: int, year: str):
    """Remove small connected components below minimum mapping unit threshold."""
    import numpy as np

    try:
        from scipy.ndimage import label as scipy_label, binary_dilation
    except ImportError:
        raise RuntimeError(
            "scipy is required for minimum mapping unit post-processing. "
            "Install it: uv add scipy"
        )

    result = landcover.copy()
    total_removed = 0

    for class_idx in config.LANDCOVER_CLASSES:
        class_mask = result == class_idx
        labeled, n_components = scipy_label(class_mask)

        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            if comp_mask.sum() < min_px:
                dilated = binary_dilation(comp_mask)
                neighbor_mask = dilated & ~comp_mask
                if neighbor_mask.any():
                    neighbor_vals = result[neighbor_mask]
                    replacement = np.bincount(neighbor_vals).argmax()
                    result[comp_mask] = replacement
                    total_removed += 1

    print(f"[classify] [{year}] MMU filter (min={min_px}px) removed "
          f"{total_removed} small patches")

    return result


def _compute_metrics(predictions, labels) -> dict:
    """Compute per-class precision/recall/F1 and overall classification metrics."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    y_true = labels.reshape(-1)
    y_pred = predictions.reshape(-1)
    class_labels = sorted(config.LANDCOVER_CLASSES.keys())

    overall_acc = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, zero_division=0
    )

    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="weighted", zero_division=0
    )

    per_class = {}
    for i, class_idx in enumerate(class_labels):
        name = config.LANDCOVER_CLASSES[class_idx]
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    metrics = {
        "overall_accuracy": round(float(overall_acc), 4),
        "evaluation_year": "2021",
        "per_class": per_class,
        "weighted_avg": {
            "precision": round(float(w_precision), 4),
            "recall": round(float(w_recall), 4),
            "f1": round(float(w_f1), 4),
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_axes": {
            "rows": "true_class",
            "columns": "predicted_class",
        },
        "class_names": [config.LANDCOVER_CLASSES[i] for i in class_labels],
    }

    print(f"[classify] Evaluation metrics (vs WorldCover):")
    print(f"  Overall accuracy: {overall_acc:.3f}")
    for name, m in per_class.items():
        print(f"  {name}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    return metrics


def _build_classifier(training: dict):
    """Build an unfitted sklearn classifier from config.

    Returns the estimator instance ready for .fit().
    The class_weight "none" -> None mapping is already done by validate_config().
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    classifier = training["classifier"]
    class_weight = training.get("class_weight")
    random_state = training.get("random_state", 42)

    if classifier == "RandomForest":
        return RandomForestClassifier(
            n_estimators=training["n_estimators"],
            max_depth=training["max_depth"],
            n_jobs=-1,
            random_state=random_state,
            class_weight=class_weight,
        )
    elif classifier == "GradientBoosting":
        if training.get("max_depth", 20) > 10:
            print(
                f"[classify] WARNING: GradientBoosting with max_depth="
                f"{training['max_depth']} may be slow and overfit. "
                f"Typical range is 3-5."
            )
        return GradientBoostingClassifier(
            n_estimators=training["n_estimators"],
            max_depth=training["max_depth"],
            learning_rate=training["learning_rate"],
            random_state=random_state,
        )
    elif classifier == "SVM":
        return SVC(
            C=training["C"],
            kernel=training["kernel"],
            gamma=training["gamma"],
            random_state=random_state,
            class_weight=class_weight,
        )
    elif classifier == "LogisticRegression":
        return LogisticRegression(
            C=training["C"],
            max_iter=training["max_iter"],
            random_state=random_state,
            class_weight=class_weight,
        )
    elif classifier == "KNN":
        return KNeighborsClassifier(
            n_neighbors=training["n_neighbors"],
            weights=training["weights"],
            n_jobs=-1,
        )
    elif classifier == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=tuple(training["hidden_layer_sizes"]),
            max_iter=training["max_iter"],
            alpha=training["alpha"],
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier!r}")


def _build_preprocessing_steps(training: dict):
    """Build preprocessing pipeline steps from config.

    Order: L2 normalize -> StandardScaler -> PCA (per spec).
    Returns list of (name, transformer) tuples. Empty list if no preprocessing.
    """
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.decomposition import PCA
    import numpy as np

    steps = []

    if training.get("l2_normalize", False):
        def _l2_normalize(X):
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            return X / (norms + 1e-10)

        steps.append(("l2_normalize", FunctionTransformer(
            func=_l2_normalize,
            validate=True,
        )))

    if training.get("scale_features", False):
        steps.append(("scaler", StandardScaler()))

    pca_components = training.get("pca_components", 0)
    if pca_components > 0:
        steps.append(("pca", PCA(
            n_components=pca_components,
            random_state=training.get("random_state", 42),
        )))

    return steps


def train_classifier(embeddings, labels, cfg=None):
    """Train a classifier on embeddings + WorldCover labels.

    Returns (clf, training_accuracy, n_training_samples).
    """
    import numpy as np
    from sklearn.pipeline import Pipeline

    if cfg is None:
        from src.experiment import load_config
        cfg = load_config()

    training = cfg["training"]
    h, w, d = embeddings.shape
    X = embeddings.reshape(-1, d)
    y = labels.reshape(-1)

    # Optional: exclude boundary pixels between classes
    valid_mask = np.ones(len(y), dtype=bool)
    if training["exclude_boundary_pixels"] and training["boundary_buffer_px"] > 0:
        interior_mask = _compute_boundary_mask(labels, training["boundary_buffer_px"])
        valid_mask = interior_mask.reshape(-1)

        if not valid_mask.any():
            raise ValueError(
                f"Boundary exclusion with buffer_px={training['boundary_buffer_px']} "
                f"excluded ALL pixels. Reduce boundary_buffer_px."
            )

    # Stratified sampling with config parameters
    max_per_class = training["max_samples_per_class"]
    sample_indices = []

    # RNG created per-class to match legacy behavior (intentional for baseline reproduction)
    for class_idx in config.LANDCOVER_CLASSES:
        rng = np.random.RandomState(training["random_state"])
        class_mask = (y == class_idx) & valid_mask
        class_indices = np.where(class_mask)[0]
        if len(class_indices) == 0:
            continue
        if len(class_indices) < 10 and training["exclude_boundary_pixels"]:
            print(f"[classify] WARNING: class {class_idx} "
                  f"({config.LANDCOVER_CLASSES[class_idx]}) has only "
                  f"{len(class_indices)} samples after boundary exclusion")
        n_samples = min(max_per_class, len(class_indices))
        chosen = rng.choice(class_indices, n_samples, replace=False)
        sample_indices.extend(chosen)

    sample_indices = np.array(sample_indices)
    X_train = X[sample_indices]
    y_train = y[sample_indices]

    # Build preprocessing + classifier pipeline
    preprocess_steps = _build_preprocessing_steps(training)
    estimator = _build_classifier(training)

    # Guard: PCA components must not exceed feature dimensions
    pca_components = training.get("pca_components", 0)
    if pca_components > 0 and pca_components >= d:
        print(f"[classify] WARNING: pca_components={pca_components} >= feature dim "
              f"{d}, disabling PCA")
        preprocess_steps = [s for s in preprocess_steps if s[0] != "pca"]

    if preprocess_steps:
        pipeline_steps = preprocess_steps + [("classifier", estimator)]
        clf = Pipeline(pipeline_steps)
    else:
        clf = estimator  # No preprocessing -- bare estimator (backward compat)

    classifier_name = training["classifier"]

    # SVM warning for large datasets
    if classifier_name == "SVM" and len(X_train) > 50000:
        print(f"[classify] WARNING: SVM with {len(X_train):,} samples may be "
              f"very slow (O(n^2)). Consider reducing max_samples_per_class.")

    print(f"[classify] Training {classifier_name} on {len(X_train):,} samples "
          f"({len(config.LANDCOVER_CLASSES)} classes)")

    try:
        clf.fit(X_train, y_train)
    except ValueError as e:
        raise ValueError(
            f"Classifier training failed: {e}. "
            f"Try adjusting preprocessing config (pca_components, scale_features) "
            f"or classifier hyperparameters."
        ) from e

    # MLP convergence check
    if classifier_name == "MLP":
        inner = clf.named_steps["classifier"] if hasattr(clf, "named_steps") else clf
        if hasattr(inner, "n_iter_") and inner.n_iter_ == inner.max_iter:
            print(f"[classify] WARNING: MLP did not converge within "
                  f"max_iter={inner.max_iter} iterations. "
                  f"Consider increasing training.max_iter.")

    train_acc = clf.score(X_train, y_train)
    print(f"[classify] Training accuracy: {train_acc:.3f}")

    return clf, float(train_acc), len(X_train)


def predict_landcover(clf, features, year, cfg=None):
    """Predict landcover class for every pixel.

    Accepts augmented features (any number of channels).
    Applies post-processing if configured.
    """
    import numpy as np

    h, w, d = features.shape
    X = features.reshape(-1, d)

    print(f"[classify] [{year}] Predicting landcover for {h}x{w} pixels...")
    predictions = clf.predict(X)
    landcover = predictions.reshape(h, w).astype(np.uint8)

    # Post-processing
    if cfg is not None:
        landcover = _apply_post_processing(landcover, cfg, year)

    # Print class distribution
    print(f"[classify] [{year}] Predicted distribution:")
    for idx, name in config.LANDCOVER_CLASSES.items():
        count = (landcover == idx).sum()
        pct = count / landcover.size * 100
        print(f"  {idx} ({name}): {count:,} pixels ({pct:.1f}%)")

    return landcover


def save_landcover_geotiff(landcover, year):
    """Save classified landcover as GeoTIFF with correct georeferencing."""
    import rasterio

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.OUTPUT_DIR / f"landcover_{year}.tif"

    # Get transform from a reference band
    ref_path = config.SENTINEL2_DIR / year / "B02.tif"
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs = src.crs

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": landcover.shape[1],
        "height": landcover.shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(landcover, 1)

    print(f"[classify] [{year}] Saved landcover to {output_path}")
    return output_path


def run_classification():
    """Run the full config-driven classification pipeline."""
    from src.experiment import load_config, validate_prerequisites, save_experiment

    # 1. Load config (defaults if no file exists)
    cfg = load_config()

    # 2. Validate prerequisites (fail-fast before doing work)
    years = list(config.TIME_RANGES.keys())
    validate_prerequisites(cfg, years=years)

    # 3. Load ground truth labels
    labels = load_worldcover_labels()

    # 4. Load embeddings and augment with spectral indices if configured
    emb_2021 = load_embeddings("2021")
    emb_2023 = load_embeddings("2023")
    features_2021 = _augment_features(emb_2021, "2021", cfg)
    features_2023 = _augment_features(emb_2023, "2023", cfg)

    # 5. Train classifier with config hyperparameters
    clf, train_acc, n_train_samples = train_classifier(features_2021, labels, cfg)

    # 6. Predict and post-process for both years
    lc_2021 = predict_landcover(clf, features_2021, "2021", cfg)
    lc_2023 = predict_landcover(clf, features_2023, "2023", cfg)

    # 7. Save GeoTIFFs to standard output paths (app.py compatibility)
    save_landcover_geotiff(lc_2021, "2021")
    save_landcover_geotiff(lc_2023, "2023")

    # 8. Compute evaluation metrics (2021 predictions vs WorldCover labels)
    metrics = _compute_metrics(lc_2021, labels)

    # 9. Add training metrics
    metrics["training_accuracy"] = train_acc
    metrics["n_training_samples"] = n_train_samples

    # 10. Save experiment to ledger
    iteration_dir, iteration_num = save_experiment(cfg, metrics)
    print(f"[classify] Experiment {iteration_num:03d} saved to {iteration_dir}")

    return lc_2021, lc_2023


def main() -> None:
    """CLI entry point."""
    run_classification()


if __name__ == "__main__":
    main()

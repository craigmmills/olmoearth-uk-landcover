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


def train_classifier(embeddings, labels):
    """Train a RandomForest classifier on 2021 embeddings + WorldCover labels.

    Uses stratified sampling to handle class imbalance.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    h, w, d = embeddings.shape
    X = embeddings.reshape(-1, d)
    y = labels.reshape(-1)

    # Stratified sampling — take up to 5000 samples per class
    max_per_class = 5000
    sample_indices = []
    for class_idx in config.LANDCOVER_CLASSES:
        class_mask = y == class_idx
        class_indices = np.where(class_mask)[0]
        if len(class_indices) == 0:
            continue
        n_samples = min(max_per_class, len(class_indices))
        rng = np.random.RandomState(42)
        chosen = rng.choice(class_indices, n_samples, replace=False)
        sample_indices.extend(chosen)

    sample_indices = np.array(sample_indices)
    X_train = X[sample_indices]
    y_train = y[sample_indices]

    print(f"[classify] Training RandomForest on {len(X_train):,} samples "
          f"({len(config.LANDCOVER_CLASSES)} classes)...")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Quick accuracy check on training data
    train_acc = clf.score(X_train, y_train)
    print(f"[classify] Training accuracy: {train_acc:.3f}")

    return clf


def predict_landcover(clf, embeddings, year):
    """Predict landcover class for every pixel."""
    import numpy as np

    h, w, d = embeddings.shape
    X = embeddings.reshape(-1, d)

    print(f"[classify] [{year}] Predicting landcover for {h}x{w} pixels...")
    predictions = clf.predict(X)
    landcover = predictions.reshape(h, w).astype(np.uint8)

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
    """Run the full classification pipeline."""
    labels = load_worldcover_labels()
    emb_2021 = load_embeddings("2021")
    emb_2023 = load_embeddings("2023")

    clf = train_classifier(emb_2021, labels)

    lc_2021 = predict_landcover(clf, emb_2021, "2021")
    lc_2023 = predict_landcover(clf, emb_2023, "2023")

    save_landcover_geotiff(lc_2021, "2021")
    save_landcover_geotiff(lc_2023, "2023")

    return lc_2021, lc_2023


def main() -> None:
    """CLI entry point."""
    run_classification()


if __name__ == "__main__":
    main()

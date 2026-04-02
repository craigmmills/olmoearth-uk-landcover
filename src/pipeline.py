"""Orchestrate the full OlmoEarth landcover pipeline end-to-end.

Usage:
    uv run python -m src.pipeline
"""
from __future__ import annotations


def run_pipeline() -> None:
    """Run complete pipeline: acquire -> preprocess -> embed -> classify -> change."""
    print("=" * 60)
    print("OlmoEarth UK Landcover Pipeline")
    print("=" * 60)

    # Step 1: Data acquisition
    print("\n" + "=" * 60)
    print("STEP 1: Data Acquisition")
    print("=" * 60)
    from src.acquire import run_acquisition
    run_acquisition()

    # Step 2: Preprocessing + Embedding extraction
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing + Embedding Extraction")
    print("=" * 60)
    from src.embeddings import run_embedding_extraction
    run_embedding_extraction()

    # Step 3: Classification
    print("\n" + "=" * 60)
    print("STEP 3: Landcover Classification")
    print("=" * 60)
    from src.classify import run_classification
    run_classification()

    # Step 4: Change Detection
    print("\n" + "=" * 60)
    print("STEP 4: Change Detection")
    print("=" * 60)
    from src.change import run_change_detection
    run_change_detection()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Results in output/:")
    print("  landcover_2021.tif  — classified landcover map for 2021")
    print("  landcover_2023.tif  — classified landcover map for 2023")
    print("  change_map.tif      — pixel-wise change detection")
    print("  transitions.json    — transition matrix and statistics")
    print("\nVisualize: uv run streamlit run src/app.py")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()

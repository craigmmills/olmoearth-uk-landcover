"""Extract per-pixel feature embeddings using OlmoEarth Tiny encoder.

Usage:
    uv run python -m src.embeddings
"""
from __future__ import annotations

from src import config


def _select_device():
    """Select the best available device: MPS > CPU."""
    import torch

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("[embeddings] Using device: mps")
        return torch.device("mps")
    print("[embeddings] Using device: cpu")
    return torch.device("cpu")


def load_olmoearth_model(device):
    """Load OlmoEarth Tiny model from HuggingFace (~57MB on first run)."""
    try:
        from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

        model = load_model_from_id(ModelID.OLMOEARTH_V1_TINY)
        model.eval()
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[embeddings] OlmoEarth Tiny loaded ({n_params:.1f}M params) on {device}")
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load OlmoEarth model. Check internet connection or "
            f"HuggingFace cache at ~/.cache/huggingface/. Original error: {e}"
        ) from e


def _make_timestamp(year: str) -> list[int]:
    """Create OlmoEarth timestamp for mid-July (summer composite midpoint).

    Returns [day, month_0indexed, year] — months are 0-indexed (July=6).
    """
    return [15, 6, int(year)]


def _prepare_patch_input(patch, timestamp, device):
    """Prepare a single 64x64 patch as a MaskedOlmoEarthSample."""
    import numpy as np
    import torch
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    # (64, 64, 12) -> (1, 64, 64, 1, 12)  [B, H, W, T, C]
    patch_5d = patch[np.newaxis, :, :, np.newaxis, :]
    s2_tensor = torch.from_numpy(patch_5d).float().to(device)

    # Timestamp: (1, 1, 3) [B, T, D]
    ts_tensor = torch.tensor([[timestamp]], dtype=torch.long, device=device)

    # Mask: (1, 64, 64, 1, 3) [B, H, W, T, num_band_sets]
    # num_band_sets=3 for SENTINEL2_L2A (not num_bands=12)
    s2_mask = torch.zeros(1, 64, 64, 1, 3, dtype=torch.long, device=device)

    masked_sample = MaskedOlmoEarthSample(
        timestamps=ts_tensor,
        sentinel2_l2a=s2_tensor,
        sentinel2_l2a_mask=s2_mask,
    )
    return masked_sample


def extract_patch_embedding(model, patch, timestamp, device):
    """Extract embedding for a single 64x64 patch.

    Returns (64, 64, 192) float32 array.
    """
    import numpy as np
    import torch
    from olmoearth_pretrain.nn.flexi_vit import PoolingType

    masked_sample = _prepare_patch_input(patch, timestamp, device)

    with torch.no_grad():
        output = model.encoder(
            masked_sample,
            patch_size=config.ENCODER_PATCH_SIZE,
            fast_pass=True,
        )
        tokens_and_masks = output["tokens_and_masks"]

    # Pool: (B, P_H, P_W, T, BandSets, D) -> (B, P_H, P_W, D)
    pooled = tokens_and_masks.pool_unmasked_tokens(
        pooling_type=PoolingType.MEAN,
        spatial_pooling=True,
        concat_features=False,
    )
    # (1, 16, 16, 192) -> (16, 16, 192)
    pooled_np = pooled.squeeze(0).cpu().numpy()

    # Upsample from 16x16 to 64x64 (nearest-neighbor)
    upsampled = np.repeat(
        np.repeat(pooled_np, config.ENCODER_PATCH_SIZE, axis=0),
        config.ENCODER_PATCH_SIZE,
        axis=1,
    )
    return upsampled


def extract_embeddings(composite, year, model, device):
    """Extract full AOI embeddings by tiling into 64x64 patches.

    Returns (512, 512, 192) float32 array.
    """
    import numpy as np
    import torch

    n_patches = config.AOI_SIZE_PX // config.PATCH_SIZE_PX  # 8
    output = np.zeros(
        (config.AOI_SIZE_PX, config.AOI_SIZE_PX, config.EMBEDDING_DIM),
        dtype=np.float32,
    )
    timestamp = _make_timestamp(year)
    patch_count = 0
    total_patches = n_patches * n_patches  # 64

    print(f"[embeddings] === Extracting embeddings for {year} ===")

    for row in range(n_patches):
        for col in range(n_patches):
            r0 = row * config.PATCH_SIZE_PX
            c0 = col * config.PATCH_SIZE_PX
            patch = composite[r0:r0 + config.PATCH_SIZE_PX,
                              c0:c0 + config.PATCH_SIZE_PX, :]

            # MPS fallback on first patch
            if patch_count == 0 and device.type == "mps":
                try:
                    embedding = extract_patch_embedding(
                        model, patch, timestamp, device
                    )
                except RuntimeError as e:
                    print(f"[embeddings] WARNING: MPS failed ({e}). "
                          f"Switching to CPU.")
                    device = torch.device("cpu")
                    model.to(device)
                    embedding = extract_patch_embedding(
                        model, patch, timestamp, device
                    )
            else:
                try:
                    embedding = extract_patch_embedding(
                        model, patch, timestamp, device
                    )
                except Exception as e:
                    print(f"[embeddings] WARNING: Patch ({row},{col}) failed: "
                          f"{e}. Filling with zeros.")
                    embedding = np.zeros(
                        (config.PATCH_SIZE_PX, config.PATCH_SIZE_PX,
                         config.EMBEDDING_DIM),
                        dtype=np.float32,
                    )

            output[r0:r0 + config.PATCH_SIZE_PX,
                   c0:c0 + config.PATCH_SIZE_PX, :] = embedding
            patch_count += 1

        print(f"[embeddings] [{year}] Row {row + 1}/{n_patches} complete "
              f"({patch_count}/{total_patches} patches)")

    if device.type == "mps":
        torch.mps.empty_cache()

    return output


def _save_embeddings(embeddings, year):
    """Save embeddings array to disk with atomic write."""
    import numpy as np

    expected = (config.AOI_SIZE_PX, config.AOI_SIZE_PX, config.EMBEDDING_DIM)
    assert embeddings.shape == expected, f"Expected {expected}, got {embeddings.shape}"
    assert embeddings.dtype == np.float32

    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.EMBEDDINGS_DIR / f"embeddings_{year}.npy"
    # np.save auto-appends .npy, so use a .tmp dir approach
    tmp_path = config.EMBEDDINGS_DIR / f"embeddings_{year}_tmp.npy"
    np.save(tmp_path, embeddings)
    tmp_path.rename(output_path)

    print(f"[embeddings] [{year}] Saved embeddings to {output_path} "
          f"({embeddings.nbytes / 1e6:.0f} MB)")
    return output_path


def run_embedding_extraction() -> None:
    """Run full embedding extraction pipeline."""
    from src.preprocess import run_preprocessing

    composites = run_preprocessing()
    device = _select_device()
    model = load_olmoearth_model(device)

    errors = {}
    for year, composite in composites.items():
        try:
            embeddings = extract_embeddings(composite, year, model, device)
            _save_embeddings(embeddings, year)
        except Exception as e:
            errors[year] = str(e)
            print(f"[embeddings] ERROR for {year}: {e}")

    if len(errors) == len(composites):
        raise RuntimeError(f"Embedding extraction failed for all years: {errors}")

    print(f"\n[embeddings] Extraction complete for "
          f"{len(composites) - len(errors)} years")


def main() -> None:
    """CLI entry point."""
    run_embedding_extraction()


if __name__ == "__main__":
    main()

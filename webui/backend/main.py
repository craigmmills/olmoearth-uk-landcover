"""FastAPI backend for the landcover classification web UI.

Serves:
- REST API for sessions, iterations, metrics, hypotheses
- SSE events for live experiment updates
- Map tiles for classification GeoTIFFs via rio-tiler

Usage:
    uv run uvicorn webui.backend.main:app --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Landcover Classification API",
    description="REST API and tile server for the self-correcting landcover classification system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all for unhandled exceptions. Logs with [backend] prefix and returns 500."""
    print(f"[backend] ERROR: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Pydantic Response Models
# ---------------------------------------------------------------------------


class SessionSummary(BaseModel):
    """Response model for GET /api/sessions."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    start_time: str
    end_time: str | None = None
    stop_reason: str | None = None
    best_iteration: int | None = None
    final_score: float | None = None
    n_iterations: int


class SessionDetail(SessionSummary):
    """Response model for GET /api/sessions/{id}."""

    initial_config: dict | None = None
    parameter_count: int | None = None


class IterationSummary(BaseModel):
    """Response model for items in GET /api/sessions/{id}/iterations."""

    model_config = ConfigDict(extra="forbid")

    iteration: int
    timestamp: str
    status: str
    overall_accuracy: float | None = None


class IterationDetail(BaseModel):
    """Response model for GET /api/sessions/{id}/iterations/{num}."""

    model_config = ConfigDict(extra="forbid")

    metadata: dict
    metrics: dict
    config: dict
    hypothesis: dict | None = None
    evaluations: dict[str, dict | None] = {}
    images: dict[str, str | None] = {}
    config_diff: dict | None = None
    metrics_diff: dict | None = None


class SSEEvent(BaseModel):
    """Shape of SSE new_iteration events."""

    event: str = "new_iteration"
    session_id: str
    iteration: int
    timestamp: str


# ---------------------------------------------------------------------------
# Input Validation Helpers
# ---------------------------------------------------------------------------


def _validate_session_id(session_id: str) -> None:
    """Validate session_id format to prevent path traversal."""
    if not re.match(r"^session_\d{8}_\d{6}$", session_id) and session_id != "session_legacy":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid session_id format: {session_id!r}. "
            f"Expected 'session_YYYYMMDD_HHMMSS'.",
        )


def _validate_iteration_num(iteration_num: int) -> None:
    """Validate iteration number is positive and within reasonable bounds."""
    if iteration_num < 1 or iteration_num > 999:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid iteration number: {iteration_num}. Must be 1-999.",
        )


def _validate_image_filename(filename: str) -> None:
    """Validate image filename to prevent path traversal."""
    pattern = r"^comparison_(2021|2023)_(full|nw|ne|sw|se)\.png$"
    if not re.match(pattern, filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image filename: {filename!r}. "
            f"Expected 'comparison_{{year}}_{{view}}.png'.",
        )


# ---------------------------------------------------------------------------
# Landcover Colormap
# ---------------------------------------------------------------------------


def _build_landcover_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Convert LANDCOVER_COLORS hex strings to RGBA tuples for rio-tiler."""
    from src import config

    cmap: dict[int, tuple[int, int, int, int]] = {}
    for class_idx, hex_color in config.LANDCOVER_COLORS.items():
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        cmap[class_idx] = (r, g, b, 255)
    return cmap


LANDCOVER_CMAP = _build_landcover_colormap()


# ---------------------------------------------------------------------------
# Iteration Data Helpers (reimplemented from src/app.py to avoid Streamlit imports)
# ---------------------------------------------------------------------------


def _list_iterations_from_dir(session_dir: Path) -> list[dict]:
    """List iterations for a session directory. Returns list sorted by iteration number.

    Reimplemented locally to avoid importing from src.app (which triggers Streamlit side effects).
    """
    iterations = []
    if not session_dir.exists():
        return iterations

    for entry in sorted(session_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("iteration_"):
            continue
        try:
            num = int(entry.name[len("iteration_"):])
        except ValueError:
            continue

        meta_path = entry / "metadata.json"
        metrics_path = entry / "metrics.json"

        timestamp = ""
        status = "unknown"
        overall_accuracy = None

        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                timestamp = meta.get("timestamp", "")
                status = meta.get("status", "unknown")
            except (json.JSONDecodeError, OSError):
                pass

        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                overall_accuracy = metrics.get("overall_accuracy")
            except (json.JSONDecodeError, OSError):
                pass

        iterations.append({
            "iteration": num,
            "timestamp": timestamp,
            "status": status,
            "overall_accuracy": overall_accuracy,
        })

    return iterations


def _load_iteration_details(iteration_num: int, session_dir: Path) -> dict:
    """Load full iteration details including hypothesis, evaluations, images, and diffs.

    Reimplemented locally to avoid importing from src.app (which triggers Streamlit side effects).
    Key business rule: hypothesis.json is stored in iteration N-1, describing the change
    that produced iteration N.
    """
    from src.experiment import load_experiment, compare_experiments

    # Load core data (config, metrics, metadata)
    data = load_experiment(iteration_num, session_dir=session_dir)

    iteration_dir = session_dir / f"iteration_{iteration_num:03d}"

    # Load hypothesis from PREVIOUS iteration (business rule: hypothesis in N-1 describes N)
    hypothesis = None
    if iteration_num > 1:
        prev_dir = session_dir / f"iteration_{iteration_num - 1:03d}"
        hyp_path = prev_dir / "hypothesis.json"
        if hyp_path.exists():
            try:
                with open(hyp_path) as f:
                    hypothesis = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    # Load evaluations
    evaluations = {}
    for year in ("2021", "2023"):
        eval_path = iteration_dir / f"evaluation_{year}.json"
        if eval_path.exists():
            try:
                with open(eval_path) as f:
                    evaluations[year] = json.load(f)
            except (json.JSONDecodeError, OSError):
                evaluations[year] = None
        else:
            evaluations[year] = None

    # Build images dict (Path objects for existing files, None for missing)
    images = {}
    for year in ("2021", "2023"):
        for view in ("full", "nw", "ne", "sw", "se"):
            key = f"{year}_{view}"
            img_path = iteration_dir / f"comparison_{year}_{view}.png"
            images[key] = img_path if img_path.exists() else None

    # Config and metrics diffs vs previous iteration
    config_diff = None
    metrics_diff = None
    if iteration_num > 1:
        try:
            diff = compare_experiments(iteration_num - 1, iteration_num, session_dir=session_dir)
            config_diff = diff.get("config_diff")
            metrics_diff = diff.get("metrics_diff")
        except (FileNotFoundError, Exception):
            pass

    return {
        "metadata": data.get("metadata", {}),
        "metrics": data.get("metrics", {}),
        "config": data.get("config", {}),
        "hypothesis": hypothesis,
        "evaluations": evaluations,
        "images": images,
        "config_diff": config_diff,
        "metrics_diff": metrics_diff,
    }


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/sessions", response_model=list[SessionSummary])
def list_all_sessions() -> list[dict]:
    """List all experiment sessions."""
    from src.experiment import list_sessions

    sessions = list_sessions()
    # Strip 'path' field (Path objects aren't JSON-serializable and shouldn't be exposed)
    return [
        {k: v for k, v in s.items() if k != "path"}
        for s in sessions
    ]


@app.get("/api/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str) -> dict:
    """Get session detail including initial_config."""
    from src.experiment import get_session_dir

    _validate_session_id(session_id)
    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    meta_path = session_dir / "session_meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"No metadata for session {session_id!r}")

    with open(meta_path) as f:
        meta = json.load(f)

    return {
        "session_id": meta.get("session_id", session_id),
        "start_time": meta.get("start_time", ""),
        "end_time": meta.get("end_time"),
        "stop_reason": meta.get("stop_reason"),
        "best_iteration": meta.get("best_iteration"),
        "final_score": meta.get("final_score"),
        "n_iterations": meta.get("n_iterations", 0),
        "initial_config": meta.get("initial_config"),
        "parameter_count": meta.get("parameter_count"),
    }


@app.get("/api/sessions/{session_id}/iterations", response_model=list[IterationSummary])
def list_session_iterations(session_id: str) -> list[dict]:
    """List iterations for a given session."""
    from src.experiment import get_session_dir

    _validate_session_id(session_id)
    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    return _list_iterations_from_dir(session_dir)


@app.get("/api/sessions/{session_id}/iterations/{iteration_num}", response_model=IterationDetail)
def get_iteration_detail(session_id: str, iteration_num: int) -> dict:
    """Full iteration detail including metrics, hypothesis, evaluations, image URLs, and diffs."""
    from src.experiment import get_session_dir

    _validate_session_id(session_id)
    _validate_iteration_num(iteration_num)

    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    try:
        details = _load_iteration_details(iteration_num, session_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Convert Path objects in images dict to URL strings
    images_urls = {}
    for key, path in details.get("images", {}).items():
        if path and isinstance(path, Path) and path.exists():
            filename = path.name
            images_urls[key] = (
                f"/api/sessions/{session_id}/iterations/{iteration_num}/images/{filename}"
            )
        else:
            images_urls[key] = None

    details["images"] = images_urls
    return details


# ---------------------------------------------------------------------------
# Image Serving
# ---------------------------------------------------------------------------


@app.get("/api/sessions/{session_id}/iterations/{iteration_num}/images/{filename}")
def get_iteration_image(session_id: str, iteration_num: int, filename: str) -> FileResponse:
    """Serve comparison PNG images as static files."""
    from src.experiment import get_session_dir

    _validate_session_id(session_id)
    _validate_iteration_num(iteration_num)
    _validate_image_filename(filename)

    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    iteration_dir = session_dir / f"iteration_{iteration_num:03d}"
    image_path = iteration_dir / filename

    if not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image {filename!r} not found")

    return FileResponse(image_path, media_type="image/png")


# ---------------------------------------------------------------------------
# Tile Proxy Endpoint (frontend-facing, no filesystem paths exposed)
# ---------------------------------------------------------------------------


@app.get("/api/sessions/{session_id}/iterations/{iteration_num}/tiles/{year}/{z}/{x}/{y}.png")
def get_landcover_tile(
    session_id: str,
    iteration_num: int,
    year: str,
    z: int,
    x: int,
    y: int,
) -> Response:
    """Proxy tile requests through a clean URL, resolving filesystem paths server-side.

    Converts source GeoTIFF to COG on first request, then renders the tile
    with the landcover colormap using rio-tiler.
    """
    from rio_tiler.io import Reader
    from rio_tiler.errors import TileOutsideBounds
    from src.experiment import get_session_dir
    from webui.backend.cog import ensure_cog

    _validate_session_id(session_id)
    _validate_iteration_num(iteration_num)
    if year not in ("2021", "2023"):
        raise HTTPException(status_code=400, detail=f"Invalid year: {year!r}")

    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    tif_path = session_dir / f"iteration_{iteration_num:03d}" / f"landcover_{year}.tif"
    if not tif_path.exists():
        raise HTTPException(status_code=404, detail=f"GeoTIFF not found for {year}")

    # Ensure COG exists (lazy conversion on first access)
    cog_path = ensure_cog(tif_path)

    try:
        with Reader(str(cog_path)) as reader:
            tile = reader.tile(x, y, z, indexes=(1,))
    except TileOutsideBounds:
        raise HTTPException(status_code=404, detail="Tile outside raster bounds")

    # Apply landcover colormap and render as PNG
    image = tile.render(colormap=LANDCOVER_CMAP)

    return Response(content=image, media_type="image/png")


# ---------------------------------------------------------------------------
# SSE Endpoint
# ---------------------------------------------------------------------------


@app.get("/api/events")
async def sse_events() -> StreamingResponse:
    """Stream real-time events when new iteration directories appear."""
    return StreamingResponse(
        _iteration_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _iteration_event_generator():
    """Yield SSE events when new iteration directories are detected.

    Polls every 2 seconds. Sends keepalive comment every 15 seconds
    (every 7-8 poll cycles) to prevent proxy/client timeouts.
    """
    known_iterations: dict[str, set[int]] = {}
    _scan_all_sessions(known_iterations)

    poll_count = 0
    while True:
        await asyncio.sleep(2)
        poll_count += 1
        new_events = _scan_for_new_iterations(known_iterations)

        for event in new_events:
            data = json.dumps(event)
            yield f"event: new_iteration\ndata: {data}\n\n"

        # Send keepalive comment every ~15 seconds (every 7 poll cycles)
        if poll_count % 7 == 0:
            yield ": keepalive\n\n"


def _scan_all_sessions(known: dict[str, set[int]]) -> None:
    """Populate known iterations from all existing sessions."""
    from src import config

    base = config.EXPERIMENTS_BASE_DIR
    if not base.exists():
        return

    for entry in base.iterdir():
        if not entry.is_dir() or not entry.name.startswith("session_"):
            continue
        if entry.is_symlink():
            continue
        session_id = entry.name
        known[session_id] = _get_iteration_nums(entry)


def _get_iteration_nums(session_dir: Path) -> set[int]:
    """Return set of iteration numbers in a session directory."""
    nums: set[int] = set()
    for entry in session_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("iteration_"):
            try:
                nums.add(int(entry.name[len("iteration_"):]))
            except ValueError:
                continue
    return nums


def _scan_for_new_iterations(known: dict[str, set[int]]) -> list[dict]:
    """Scan for new iterations and return events for any found."""
    from src import config

    events: list[dict] = []
    base = config.EXPERIMENTS_BASE_DIR
    if not base.exists():
        return events

    for entry in base.iterdir():
        if not entry.is_dir() or not entry.name.startswith("session_"):
            continue
        if entry.is_symlink():
            continue

        session_id = entry.name
        current_nums = _get_iteration_nums(entry)
        prev_nums = known.get(session_id, set())
        new_nums = current_nums - prev_nums

        for num in sorted(new_nums):
            meta_path = entry / f"iteration_{num:03d}" / "metadata.json"
            timestamp = ""
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    timestamp = meta.get("timestamp", "")
                except (json.JSONDecodeError, OSError):
                    pass

            events.append({
                "session_id": session_id,
                "iteration": num,
                "timestamp": timestamp,
            })

        known[session_id] = current_nums

    return events


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}

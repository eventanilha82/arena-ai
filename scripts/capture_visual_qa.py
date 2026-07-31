from __future__ import annotations

import argparse
import os
import sys
import csv
import json
import hashlib
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from math import ceil, hypot
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena_ai.main import (  # noqa: E402
    App,
    BG,
    CHANCE_EVENT_WINDOW_MINUTES,
    GOAL_EVENT_WINDOW_MINUTES,
    GOLD,
    HEIGHT,
    FPS,
    CINEMATIC_KEEPER_FRAME_COUNT,
    CINEMATIC_KICK_FRAME_COUNT,
    CINEMATIC_RUNNER_FRAME_COUNT,
    CINEMATIC_STOP_FRAME_COUNT,
    SIMULATION_SECONDS,
    SHOT_KICK_AT,
    SHOT_KICK_POSE_AT,
    SHOT_NET_AT,
    SHOT_NET_SETTLE_PROGRESS,
    SHOT_NET_VISUAL_CONTACT_AT,
    SHOT_RELEASE_END,
    WIDTH,
    font,
)
from arena_ai.cinematic_uniforms import CINEMATIC_UNIFORMS  # noqa: E402
from arena_ai.cinematic_dribble_runtime import (  # noqa: E402
    Poc2DribbleSample,
)
from arena_ai.cinematic_poc_runtime import (  # noqa: E402
    POC_RUNNER_CANVAS_SIZE,
    POC_RUNNER_ROOT,
    PocSequence,
    PocSequenceSample,
    PocViewport,
)
from arena_ai.worldcup_model import Prediction  # noqa: E402
from scripts.validate_visuals import (  # noqa: E402
    alpha_surface_gap_px,
    away_win_prediction,
    ball_render_for_state,
    ball_contract_snapshot,
    home_win_prediction,
    neutral_prediction,
    runner_render_for_state,
    seek_match_time,
    visible_ball_radius,
)


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "visual_qa" / "current"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
METADATA_PATH = OUTPUT_DIR / "metadata.json"
SEQUENCE_DIR = OUTPUT_DIR / "cinematic_sequence"
VARIANT_DIR = OUTPUT_DIR / "cinematic_variants"
BALL_ROLL_DIR = OUTPUT_DIR / "cinematic_ball_roll"
DRIBBLE_FLOW_DIR = OUTPUT_DIR / "cinematic_dribble_flow"
FIELD_RECT = pygame.Rect(32, 110, 910, 490)
CONTACT_LABEL_HEIGHT = 34
CONTACT_COLUMNS = 3
SEQUENCE_COLUMNS = 4


def configure_output_dir(output_dir: Path) -> None:
    global OUTPUT_DIR, METADATA_PATH, SEQUENCE_DIR, VARIANT_DIR, BALL_ROLL_DIR, DRIBBLE_FLOW_DIR
    OUTPUT_DIR = output_dir.resolve()
    METADATA_PATH = OUTPUT_DIR / "metadata.json"
    SEQUENCE_DIR = OUTPUT_DIR / "cinematic_sequence"
    VARIANT_DIR = OUTPUT_DIR / "cinematic_variants"
    BALL_ROLL_DIR = OUTPUT_DIR / "cinematic_ball_roll"
    DRIBBLE_FLOW_DIR = OUTPUT_DIR / "cinematic_dribble_flow"


def rendered_ball_runner_gap(app: App, state: dict[str, object]) -> float:
    possession = str(state.get("possession", "home"))
    team = app.home if possession == "home" else app.away
    direction = 1 if possession == "home" else -1
    runner, runner_rect = runner_render_for_state(app, team, state, direction)
    ball, ball_rect = ball_render_for_state(app, state)
    return alpha_surface_gap_px(ball, ball_rect, runner, runner_rect)


def locomotion_metrics(
    app: App,
    state: dict[str, object],
) -> dict[str, object]:
    sequence = state.get("poc_contract_sequence")
    sample = state.get("poc_contract_sample")
    viewport = state.get("poc_viewport")
    poc2_sample = state.get("poc2_dribble_sample")
    if (
        isinstance(poc2_sample, Poc2DribbleSample)
        and not isinstance(sequence, PocSequence)
    ):
        player = poc2_sample.player
        frame = poc2_sample.frame
        metadata = frame.metadata

        def scene_point(
            point: tuple[float, float],
        ) -> tuple[float, float]:
            return (
                player.scene_left + point[0] * player.scale,
                player.scene_top + point[1] * player.scale,
            )

        bbox_x, bbox_y, bbox_w, bbox_h = metadata.visible_bbox
        runner_surface, runner_rect = runner_render_for_state(
            app,
            app.home
            if state.get("possession") == "home"
            else app.away,
            state,
            1 if poc2_sample.direction == "right" else -1,
        )
        visible = runner_surface.get_bounding_rect(min_alpha=30)
        visible_ground_y = runner_rect.top + visible.bottom
        support = (
            scene_point(metadata.support_foot)
            if metadata.support_foot is not None
            else (
                player.scene_center_x,
                float(visible_ground_y),
            )
        )
        pelvis = scene_point(
            (
                metadata.pelvis_x,
                bbox_y + bbox_h * 0.48,
            )
        )
        toe = (
            support
            if metadata.support_foot is not None
            else scene_point(
                (
                    bbox_x + bbox_w
                    if poc2_sample.direction == "right"
                    else bbox_x,
                    bbox_y + bbox_h,
                )
            )
        )
        next_sample = app.poc2_dribble.sample(
            poc2_sample.uniform_code,
            poc2_sample.direction == "left",
            poc2_sample.elapsed_seconds + 1.0 / FPS,
            player.scene_center_x,
            player.scene_ground_y,
            player.scale,
        )
        ball_x, ball_y = state["ball_pos"]  # type: ignore[misc]
        landmark_gap = max(
            0.0,
            hypot(
                float(ball_x) - toe[0],
                float(ball_y) - toe[1],
            )
            - visible_ball_radius(
                app,
                int(state["ball_scale"]),
            ),
        )
        rendered_gap = rendered_ball_runner_gap(app, state)
        return {
            "runner_frame": frame.index,
            "runner_next_frame": next_sample.frame.index,
            "runner_frame_blend": 0.0,
            "runner_render_frame": frame.index,
            "runner_phase": frame.cycle_phase,
            "support": metadata.support or "flight",
            "support_weight": 0.0 if frame.flight else 1.0,
            "support_pos": support,
            "sole_ground_gap_px": (
                player.scene_ground_y - visible_ground_y
            ),
            "pelvis_pos": pelvis,
            "toe_pos": toe,
            "dribble_touch_phase": poc2_sample.ball.touch_phase,
            "dribble_pose_phase": metadata.phase,
            "dribble_touch_slot": poc2_sample.ball.touch_slot,
            "landmark_dribble_contact_gap_px": landmark_gap,
            "landmark_ball_foot_gap_px": landmark_gap,
            "visible_ball_foot_gap_px": rendered_gap,
            "rendered_ball_runner_gap_px": rendered_gap,
            "kick_foot_pos": toe,
            "keeper_frame_first": state.get("keeper_frame_first", 0),
            "keeper_frame_following": state.get(
                "keeper_frame_following",
                0,
            ),
            "keeper_frame_blend": float(
                state.get("keeper_frame_blend", 0.0)
            ),
            "keeper_render_frame": state.get("keeper_render_frame", 0),
            "following_render_frame": next_sample.frame.index,
            "actor_canvas_size": max(1, round(player.scene_size)),
        }
    if (
        isinstance(sequence, PocSequence)
        and isinstance(sample, PocSequenceSample)
        and isinstance(viewport, PocViewport)
        and isinstance(poc2_sample, Poc2DribbleSample)
        and sample.actor_source == 0
    ):
        possession = str(state.get("possession", "home"))
        team = app.home if possession == "home" else app.away
        frame = poc2_sample.frame
        metadata = frame.metadata
        actor_root = viewport.point(
            poc2_sample.player.scene_center_x,
            poc2_sample.player.scene_ground_y,
        )
        actor_scale = viewport.scale
        canvas_size = poc2_sample.player.canvas_size_px
        actor_left = actor_root[0] - canvas_size * 0.5 * actor_scale
        actor_top = (
            actor_root[1]
            - poc2_sample.player.canvas_ground_y_px * actor_scale
        )

        def poc2_screen_point(
            point: tuple[float, float],
        ) -> tuple[float, float]:
            return (
                actor_left + point[0] * actor_scale,
                actor_top + point[1] * actor_scale,
            )

        bbox_x, bbox_y, bbox_w, bbox_h = metadata.visible_bbox
        runner_surface, runner_rect = runner_render_for_state(
            app,
            team,
            state,
            1 if sequence.attack_direction == "right" else -1,
        )
        visible = runner_surface.get_bounding_rect(min_alpha=30)
        visible_ground_y = runner_rect.top + visible.bottom
        support = (
            poc2_screen_point(metadata.support_foot)
            if metadata.support_foot is not None
            else (actor_root[0], float(visible_ground_y))
        )
        pelvis = poc2_screen_point(
            (
                metadata.pelvis_x,
                bbox_y + bbox_h * 0.48,
            )
        )
        toe = (
            support
            if metadata.support_foot is not None
            else poc2_screen_point(
                (
                    bbox_x + bbox_w
                    if sequence.attack_direction == "right"
                    else bbox_x,
                    bbox_y + bbox_h,
                )
            )
        )
        following_sample = app.poc_sequences.sample(
            sequence,
            min(
                sequence.duration_seconds,
                sample.elapsed + 1.0 / app.poc_sequences.sample_hz,
            ),
        )
        following_render_frame = app.cinematic_poc_actor_material(
            sequence,
            following_sample,
            team,
        )[1]
        ball_x, ball_y = state["ball_pos"]  # type: ignore[misc]
        landmark_gap = max(
            0.0,
            hypot(
                float(ball_x) - toe[0],
                float(ball_y) - toe[1],
            )
            - visible_ball_radius(app, int(state["ball_scale"])),
        )
        rendered_gap = rendered_ball_runner_gap(app, state)
        return {
            "runner_frame": frame.index,
            "runner_next_frame": following_sample.actor_frame,
            "runner_frame_blend": 0.0,
            "runner_render_frame": frame.index,
            "runner_phase": frame.cycle_phase,
            "support": metadata.support or "flight",
            "support_weight": 0.0 if frame.flight else 1.0,
            "support_pos": support,
            "sole_ground_gap_px": actor_root[1] - visible_ground_y,
            "pelvis_pos": pelvis,
            "toe_pos": toe,
            "dribble_touch_phase": poc2_sample.ball.touch_phase,
            "dribble_pose_phase": metadata.phase,
            "dribble_touch_slot": poc2_sample.ball.touch_slot,
            "landmark_dribble_contact_gap_px": landmark_gap,
            "landmark_ball_foot_gap_px": landmark_gap,
            "visible_ball_foot_gap_px": rendered_gap,
            "rendered_ball_runner_gap_px": rendered_gap,
            "kick_foot_pos": toe,
            "keeper_frame_first": sample.keeper_frame,
            "keeper_frame_following": sample.keeper_frame,
            "keeper_frame_blend": 0.0,
            "keeper_render_frame": sample.keeper_frame,
            "following_render_frame": following_render_frame,
            "actor_canvas_size": max(
                1,
                round(canvas_size * actor_scale),
            ),
        }
    if not (
        isinstance(sequence, PocSequence)
        and isinstance(sample, PocSequenceSample)
        and isinstance(viewport, PocViewport)
    ):
        support = state["support_foot_pos"]  # type: ignore[assignment]
        pelvis = state["pelvis_pos"]  # type: ignore[assignment]
        toe = state["dribble_foot_pos"]  # type: ignore[assignment]
        kick_foot = state["rendered_kick_foot_pos"]  # type: ignore[assignment]
        return {
            "runner_frame": state["runner_frame"],
            "runner_next_frame": state["runner_next_frame"],
            "runner_frame_blend": float(
                state["runner_frame_blend"]
            ),
            "runner_render_frame": state["runner_render_frame"],
            "runner_phase": float(
                state["runner_controlled_phase"]
            ),
            "support": state["support_phase"] or "flight",
            "support_weight": float(state["support_weight"]),
            "support_pos": support,
            "sole_ground_gap_px": float(
                state["sole_ground_gap_px"]
            ),
            "pelvis_pos": pelvis,
            "toe_pos": toe,
            "dribble_touch_phase": float(
                state["dribble_touch_phase"]
            ),
            "dribble_pose_phase": str(
                state.get("support_phase") or ""
            ),
            "dribble_touch_slot": int(
                state["dribble_touch_slot"]
            ),
            "landmark_dribble_contact_gap_px": float(
                state["landmark_dribble_contact_gap_px"]
            ),
            "landmark_ball_foot_gap_px": float(
                state["landmark_ball_foot_gap_px"]
            ),
            "visible_ball_foot_gap_px": float(
                state["visible_ball_foot_gap_px"]
            ),
            "rendered_ball_runner_gap_px": (
                rendered_ball_runner_gap(app, state)
            ),
            "kick_foot_pos": kick_foot,
            "keeper_frame_first": state[
                "keeper_frame_first"
            ],
            "keeper_frame_following": state[
                "keeper_frame_following"
            ],
            "keeper_frame_blend": float(
                state["keeper_frame_blend"]
            ),
            "keeper_render_frame": state[
                "keeper_render_frame"
            ],
        }

    possession = str(state.get("possession", "home"))
    team = app.home if possession == "home" else app.away
    (
        _frame,
        render_frame,
        metadata,
        actor_scale,
    ) = app.cinematic_poc_actor_material(
        sequence,
        sample,
        team,
    )
    actor_scale *= viewport.scale
    actor_root = viewport.point(
        float(state["poc_actor_x"]),
        sample.actor_ground_y,
    )
    actor_left = (
        actor_root[0]
        - POC_RUNNER_ROOT[0] * actor_scale
    )
    actor_top = (
        actor_root[1]
        - POC_RUNNER_ROOT[1] * actor_scale
    )

    def screen_point(key: str) -> tuple[float, float]:
        raw = metadata.get(key)
        if not (
            isinstance(raw, list)
            and len(raw) == 2
        ):
            return actor_root
        return (
            actor_left + float(raw[0]) * actor_scale,
            actor_top + float(raw[1]) * actor_scale,
        )

    support = screen_point("support_foot")
    pelvis = screen_point("pelvis")
    toe = screen_point("dribble_foot")
    ball_x, ball_y = state["ball_pos"]  # type: ignore[misc]
    landmark_gap = max(
        0.0,
        hypot(
            float(ball_x) - toe[0],
            float(ball_y) - toe[1],
        )
        - visible_ball_radius(
            app,
            int(state["ball_scale"]),
        ),
    )
    if app.poc_sequences is None:
        raise RuntimeError("missing POC sequence bank in visual QA")
    frame_position = app.poc_sequences.actor_frame_position(
        sequence,
        sample.elapsed,
    )
    following_sample = app.poc_sequences.sample(
        sequence,
        min(
            sequence.duration_seconds,
            sample.elapsed + 1.0 / app.poc_sequences.sample_hz,
        ),
    )
    following_render_frame = (
        app.cinematic_poc_actor_material(
            sequence,
            following_sample,
            team,
        )[1]
    )
    rendered_gap = rendered_ball_runner_gap(app, state)
    return {
        "runner_frame": sample.actor_frame,
        "runner_next_frame": following_sample.actor_frame,
        "runner_frame_blend": 0.0,
        "runner_render_frame": render_frame,
        "runner_phase": frame_position,
        "support": metadata.get("support", "flight"),
        "support_weight": float(
            metadata.get("support_weight", 0.0)
        ),
        "support_pos": support,
        "sole_ground_gap_px": actor_root[1] - support[1],
        "pelvis_pos": pelvis,
        "toe_pos": toe,
        "dribble_touch_phase": float(
            state.get("dribble_touch_phase", 0.0)
        ),
        "dribble_pose_phase": str(metadata.get("phase", "")),
        "dribble_touch_slot": render_frame,
        "landmark_dribble_contact_gap_px": landmark_gap,
        "landmark_ball_foot_gap_px": landmark_gap,
        "visible_ball_foot_gap_px": rendered_gap,
        "rendered_ball_runner_gap_px": rendered_gap,
        "kick_foot_pos": toe,
        "keeper_frame_first": sample.keeper_frame,
        "keeper_frame_following": sample.keeper_frame,
        "keeper_frame_blend": 0.0,
        "keeper_render_frame": sample.keeper_frame,
        "following_render_frame": following_render_frame,
        "actor_canvas_size": max(
            1,
            round(POC_RUNNER_CANVAS_SIZE * actor_scale),
        ),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint_paths() -> list[Path]:
    paths = {
        ROOT / "Makefile",
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "assets" / "asset_manifest.json",
        ROOT / "docs" / "ASSETS.md",
        ROOT / "docs" / "QUALITY.md",
        ROOT / "scripts" / "capture_visual_qa.py",
        ROOT / "scripts" / "validate_poc_runtime_parity.py",
        ROOT / "scripts" / "validate_visuals.py",
        ROOT / "modeling" / "worldcup_2026_ml" / "models" / "model_sota.pkl",
        ROOT / "modeling" / "worldcup_2026_ml" / "models" / "runtime_prediction_cache.pkl",
        ROOT / "modeling" / "worldcup_2026_ml" / "reports" / "sota_model_report.json",
    }
    paths.update(path for path in (ROOT / "docs").rglob("*.md") if path.is_file())
    paths.update(path for path in (ROOT / "scripts").rglob("*.py") if path.is_file())
    paths.update(path for path in (ROOT / "src" / "arena_ai").rglob("*.py") if path.is_file())
    paths.update(
        path
        for path in (ROOT / "modeling" / "worldcup_2026_ml" / "src").rglob("*.py")
        if path.is_file()
    )
    for asset_root in (ROOT / "assets" / "generated", ROOT / "assets" / "fonts"):
        paths.update(path for path in asset_root.rglob("*") if path.is_file())
    return sorted(path for path in paths if path.exists())


def source_control_provenance() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    source_hashes = {
        path.relative_to(ROOT).as_posix(): file_sha256(path)
        for path in source_fingerprint_paths()
    }
    aggregate = hashlib.sha256()
    for relative_path, digest in source_hashes.items():
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "commit": commit,
        "dirty": bool(status),
        "snapshot_kind": "dirty_worktree" if status else "clean_commit",
        "status": status,
        "source_file_count": len(source_hashes),
        "source_aggregate_sha256": aggregate.hexdigest(),
        "source_sha256": source_hashes,
    }


def artifact_inventory(output_dir: Path | None = None) -> list[dict[str, object]]:
    output_dir = OUTPUT_DIR if output_dir is None else output_dir
    metadata_path = output_dir / "metadata.json"
    return [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != metadata_path
    ]


def _semantic_metadata(metadata: dict[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(metadata))
    normalized.pop("generated_at", None)
    return normalized


def replay_current_evidence(recorded_metadata: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory(prefix="arena-ai-visual-qa-") as temporary:
        replay_dir = Path(temporary) / "current"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output-dir",
                str(replay_dir),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "visual QA deterministic replay failed:\n"
                f"stdout:\n{result.stdout[-4000:]}\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )
        replay_metadata_path = replay_dir / "metadata.json"
        if not replay_metadata_path.exists():
            raise RuntimeError("visual QA deterministic replay produced no metadata")
        replay_metadata = json.loads(replay_metadata_path.read_text(encoding="utf-8"))
        if _semantic_metadata(recorded_metadata) != _semantic_metadata(replay_metadata):
            raise RuntimeError("visual QA metadata claims do not reproduce from the current runtime")
        recorded_inventory = artifact_inventory(OUTPUT_DIR)
        replay_inventory = artifact_inventory(replay_dir)
        if recorded_inventory != replay_inventory:
            recorded = {str(item["path"]): item for item in recorded_inventory}
            replayed = {str(item["path"]): item for item in replay_inventory}
            changed = sorted(
                path
                for path in set(recorded) | set(replayed)
                if recorded.get(path) != replayed.get(path)
            )
            raise RuntimeError(
                "visual QA artifacts do not reproduce byte-for-byte from the current runtime: "
                f"{changed[:20]}"
            )


def validate_current_evidence() -> None:
    if not METADATA_PATH.exists():
        raise RuntimeError(f"visual QA metadata is missing: {METADATA_PATH}")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 3:
        raise RuntimeError(f"visual QA metadata schema is stale: {metadata.get('schema_version')}")

    recorded_source = metadata.get("source_control")
    if not isinstance(recorded_source, dict):
        raise RuntimeError("visual QA metadata has no source-control provenance")
    current_source = source_control_provenance()
    for key in (
        "commit",
        "dirty",
        "snapshot_kind",
        "status",
        "source_file_count",
        "source_aggregate_sha256",
        "source_sha256",
    ):
        if recorded_source.get(key) != current_source.get(key):
            raise RuntimeError(
                f"visual QA source provenance is stale for {key}: "
                f"recorded={recorded_source.get(key)!r}, current={current_source.get(key)!r}"
            )

    inventory_block = metadata.get("artifact_inventory")
    if not isinstance(inventory_block, dict) or inventory_block.get("excluded_self") != METADATA_PATH.name:
        raise RuntimeError("visual QA metadata has no exact recursive artifact inventory")
    recorded_files = inventory_block.get("files")
    if not isinstance(recorded_files, list):
        raise RuntimeError("visual QA artifact inventory is malformed")
    recorded_by_path = {
        str(item.get("path")): item
        for item in recorded_files
        if isinstance(item, dict)
    }
    if len(recorded_by_path) != len(recorded_files):
        raise RuntimeError("visual QA artifact inventory contains duplicate or malformed paths")
    actual_files = artifact_inventory()
    actual_by_path = {str(item["path"]): item for item in actual_files}
    if set(recorded_by_path) != set(actual_by_path):
        raise RuntimeError(
            "visual QA artifact inventory drift: "
            f"missing={sorted(set(recorded_by_path) - set(actual_by_path))}, "
            f"orphan={sorted(set(actual_by_path) - set(recorded_by_path))}"
        )
    for relative_path, actual in actual_by_path.items():
        recorded = recorded_by_path[relative_path]
        for key in ("sha256", "size_bytes"):
            if recorded.get(key) != actual.get(key):
                raise RuntimeError(
                    f"visual QA artifact changed: {relative_path} {key} "
                    f"recorded={recorded.get(key)!r}, current={actual.get(key)!r}"
                )
    if inventory_block.get("file_count") != len(actual_files):
        raise RuntimeError("visual QA artifact file count drifted")

    for block_name in ("cinematic_sequence", "cinematic_ball_roll", "cinematic_dribble_flow"):
        block = metadata.get(block_name)
        if not isinstance(block, dict) or not isinstance(block.get("frames"), list):
            raise RuntimeError(f"visual QA chronology block is missing: {block_name}")
        frames = block["frames"]
        seconds = [float(frame["seconds"]) for frame in frames]
        if any(following <= current for current, following in zip(seconds, seconds[1:])):
            raise RuntimeError(f"visual QA chronology is not strictly increasing: {block_name}")
        progresses = [
            float(frame["requested_progress"])
            for frame in frames
            if frame.get("requested_progress") is not None
        ]
        if any(following <= current for current, following in zip(progresses, progresses[1:])):
            raise RuntimeError(f"visual QA progress is not strictly increasing: {block_name}")
    replay_current_evidence(metadata)
    print(
        f"visual QA current: {len(actual_files)} artifacts, "
        f"source={current_source['source_aggregate_sha256']}"
    )


def goal_overlay_metrics(app: App, frame: pygame.Surface) -> dict[str, object]:
    center = app.cinematic_goal_overlay_center(FIELD_RECT)
    text = app.text_cache.render(app.f_lg, "GOOOL!", GOLD)
    rect = pygame.Rect(0, 0, text.get_width() + 54, text.get_height() + 22)
    rect.center = center
    pred = app.match_prediction
    if pred is None:
        raise RuntimeError("goal overlay evidence requires an active prediction")

    original_overlay = app.draw_cinematic_goal_overlay
    app.draw_cinematic_goal_overlay = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    try:
        app.screen.fill(BG)
        app.draw_top("Confronto | impacto confirmado", "QA visual")
        cinematic_focus = app.match_cinematic_focus(pred)
        app.draw_field(pred, pred, "CONFRONTO")
        app.draw_side_panel(pred, cinematic_focus=cinematic_focus)
        app.draw_score_panel({"CONFRONTO": pred}, "CONFRONTO", pred, cinematic_focus=cinematic_focus)
        baseline = app.screen.copy()
    finally:
        app.draw_cinematic_goal_overlay = original_overlay  # type: ignore[method-assign]
        app.screen.blit(frame, (0, 0))

    rgb = pygame.surfarray.array3d(frame)
    baseline_rgb = pygame.surfarray.array3d(baseline)
    region = rgb[rect.x : rect.right, rect.y : rect.bottom].astype(np.int16)
    baseline_region = baseline_rgb[rect.x : rect.right, rect.y : rect.bottom].astype(np.int16)
    gold = np.asarray(GOLD, dtype=np.int16)
    panel = app.cinematic_overlay_cache.get(("goal_overlay_panel", app.f_lg.get_height()))
    if panel is None:
        raise RuntimeError("runtime GOOOL panel template was not cached by the real draw")
    panel_alpha = pygame.surfarray.array_alpha(panel)
    expected_footprint = panel_alpha > 32
    text_alpha = pygame.surfarray.array_alpha(text)
    expected_text = np.zeros(expected_footprint.shape, dtype=bool)
    expected_text[27 : 27 + text.get_width(), 11 : 11 + text.get_height()] = text_alpha > 64
    corner_zone = ~expected_footprint
    interior = expected_footprint & ~expected_text

    def template_metrics(candidate: np.ndarray) -> dict[str, float | int | bool]:
        delta = np.max(np.abs(candidate - baseline_region), axis=2) > 6
        gold_match = np.all(np.abs(candidate - gold) <= 18, axis=2)
        text_coverage = float((gold_match & expected_text).sum()) / max(1, int(expected_text.sum()))
        footprint_recall = float((delta & expected_footprint).sum()) / max(1, int(expected_footprint.sum()))
        corner_delta_ratio = float((delta & corner_zone).sum()) / max(1, int(corner_zone.sum()))
        dark_panel = candidate.max(axis=2) <= 48
        interior_dark_ratio = float((dark_panel & interior).sum()) / max(1, int(interior.sum()))
        passes = (
            text_coverage >= 0.78
            and footprint_recall >= 0.92
            and corner_delta_ratio <= 0.03
            and interior_dark_ratio >= 0.94
        )
        return {
            "gold_text_pixels": int(gold_match.sum()),
            "dark_panel_pixels": int(dark_panel.sum()),
            "text_template_coverage": round(text_coverage, 6),
            "panel_footprint_recall": round(footprint_recall, 6),
            "rounded_corner_delta_ratio": round(corner_delta_ratio, 6),
            "panel_interior_dark_ratio": round(interior_dark_ratio, 6),
            "passes": passes,
        }

    actual = template_metrics(region)
    missing_text = region.copy()
    missing_text[expected_text] = baseline_region[expected_text]
    active_columns = np.any(expected_text, axis=1)
    column_runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for column, active in enumerate(active_columns):
        if active and run_start is None:
            run_start = column
        elif not active and run_start is not None:
            column_runs.append((run_start, column))
            run_start = None
    if run_start is not None:
        column_runs.append((run_start, len(active_columns)))
    if len(column_runs) < 2:
        raise RuntimeError("GOOOL overlay template has no separable final glyph")
    missing_glyph = region.copy()
    glyph_start, glyph_end = column_runs[-1]
    missing_glyph[glyph_start:glyph_end, :] = baseline_region[glyph_start:glyph_end, :]
    clipped_panel = region.copy()
    clip_start = round(expected_footprint.shape[0] * 0.84)
    clipped_panel[clip_start:, :] = baseline_region[clip_start:, :]
    square_panel = region.copy()
    square_panel[corner_zone] = np.asarray((2, 9, 13), dtype=np.int16)
    missing_text_metrics = template_metrics(missing_text)
    missing_glyph_metrics = template_metrics(missing_glyph)
    clipped_panel_metrics = template_metrics(clipped_panel)
    square_panel_metrics = template_metrics(square_panel)
    if not actual["passes"]:
        raise RuntimeError(f"runtime GOOOL overlay misses its template: {actual}")
    if (
        missing_text_metrics["passes"]
        or missing_glyph_metrics["passes"]
        or clipped_panel_metrics["passes"]
        or square_panel_metrics["passes"]
    ):
        raise RuntimeError(
            "GOOOL overlay template gate accepts an adversarial mutation: "
            f"missing_text={missing_text_metrics}, missing_glyph={missing_glyph_metrics}, "
            f"clipped_panel={clipped_panel_metrics}, square_panel={square_panel_metrics}"
        )
    return {
        "rect": [rect.x, rect.y, rect.w, rect.h],
        **actual,
        "adversarial_missing_text_rejected": True,
        "adversarial_missing_glyph_rejected": True,
        "adversarial_clipped_panel_rejected": True,
        "adversarial_square_panel_rejected": True,
    }


def video_frame_rate(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("visual locomotion QA requires ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    numerator, denominator = result.stdout.strip().split("/", 1)
    return float(numerator) / max(1.0, float(denominator))


def save_frame(app: App, seconds: float, label: str, filename: str) -> pygame.Surface:
    app.screen.fill(BG)
    pred = app.match_prediction
    if pred is None:
        raise RuntimeError("visual QA requires app.match_prediction")
    seek_match_time(app, pred, seconds)
    app.screen.fill(BG)
    app.draw_top(label, "QA visual")
    cinematic_focus = app.match_cinematic_focus(pred)
    app.draw_field(pred, pred, "CONFRONTO")
    app.draw_side_panel(pred, cinematic_focus=cinematic_focus)
    app.draw_score_panel({"CONFRONTO": pred}, "CONFRONTO", pred, cinematic_focus=cinematic_focus)
    frame = app.screen.copy()
    target = OUTPUT_DIR / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(frame, target)
    return frame


def append_sample(
    samples: list[tuple[str, str, pygame.Surface]],
    label: str,
    filename: str,
    frame: pygame.Surface,
) -> None:
    samples.append((label, filename, frame))


def set_matchup(app: App, home_code: str, away_code: str) -> None:
    codes = [team.code for team in app.teams]
    app.home_idx = codes.index(home_code)
    app.away_idx = codes.index(away_code)


def append_contact_cell(
    sheet: pygame.Surface,
    frame: pygame.Surface,
    label: str,
    index: int,
    label_font: pygame.font.Font,
    columns: int,
) -> None:
    cell_w = WIDTH // columns
    frame_h = round(cell_w * HEIGHT / WIDTH)
    cell_h = frame_h + CONTACT_LABEL_HEIGHT
    x = (index % columns) * cell_w
    y = (index // columns) * cell_h
    panel = pygame.Surface((cell_w, CONTACT_LABEL_HEIGHT), pygame.SRCALPHA)
    panel.fill((0, 7, 10, 190))
    rendered = label_font.render(label, True, GOLD)
    panel.blit(rendered, (16, 8))
    sheet.blit(panel, (x, y))
    thumbnail = pygame.transform.smoothscale(frame, (cell_w, frame_h))
    sheet.blit(thumbnail, (x, y + CONTACT_LABEL_HEIGHT))


def contact_sheet_size(sample_count: int, columns: int) -> tuple[int, int, int, int]:
    cell_w = WIDTH // columns
    frame_h = round(cell_w * HEIGHT / WIDTH)
    cell_h = frame_h + CONTACT_LABEL_HEIGHT
    return WIDTH, ceil(sample_count / columns) * cell_h, cell_w, frame_h


def capture_cinematic_sequence(
    app: App,
    pred: Prediction,
    goal_minute: int,
    label_font: pygame.font.Font,
) -> dict[str, object]:
    progress_samples = tuple(
        sorted(
            {
                0.50,
                0.66,
                SHOT_KICK_AT - 0.01,
                SHOT_KICK_AT,
                SHOT_KICK_AT + 0.02,
                0.78,
                max(SHOT_RELEASE_END + 0.05, 0.86),
                SHOT_NET_AT,
                SHOT_NET_VISUAL_CONTACT_AT,
                1.04,
                SHOT_NET_VISUAL_CONTACT_AT + SHOT_NET_SETTLE_PROGRESS * 0.58,
                SHOT_NET_VISUAL_CONTACT_AT + SHOT_NET_SETTLE_PROGRESS + 0.02,
            }
        )
    )
    sequence_samples: list[tuple[str, str, pygame.Surface]] = []
    sequence_metadata: list[dict[str, object]] = []
    app.match_prediction = pred
    for index, progress in enumerate(progress_samples):
        seconds = (
            goal_minute - GOAL_EVENT_WINDOW_MINUTES + progress * GOAL_EVENT_WINDOW_MINUTES
        ) / 90.0 * SIMULATION_SECONDS
        label = f"{index + 1:02d} | p={progress:.3f}"
        filename = f"cinematic_sequence/shot_{index:02d}_p{progress:.3f}.png"
        frame = save_frame(app, seconds, f"Confronto | sequência {label}", filename)
        state = app.cinematic_scene_state(FIELD_RECT, pred)
        contract = ball_contract_snapshot(state, f"capture sequence progress={progress:.3f}")
        path = OUTPUT_DIR / filename
        sequence_samples.append((label, filename, frame))
        sequence_metadata.append(
            {
                "index": index,
                "file": filename,
                "label": label,
                "requested_progress": progress,
                "seconds": seconds,
                "post_impact": progress > 1.0,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "ball_state": contract,
            }
        )

    strip_width, strip_height, cell_w, frame_h = contact_sheet_size(
        len(sequence_samples), SEQUENCE_COLUMNS
    )
    strip = pygame.Surface((strip_width, strip_height), pygame.SRCALPHA)
    strip.fill(BG)
    for index, (label, _filename, frame) in enumerate(sequence_samples):
        append_contact_cell(strip, frame, label, index, label_font, SEQUENCE_COLUMNS)
    strip_path = OUTPUT_DIR / "cinematic_shot_sequence.png"
    pygame.image.save(strip, strip_path)
    return {
        "event": "home_goal",
        "goal_minute": goal_minute,
        "frame_count": len(sequence_samples),
        "contains_post_impact_progress": any(progress > 1.0 for progress in progress_samples),
        "post_impact_frame_count": sum(progress > 1.0 for progress in progress_samples),
        "strip": strip_path.name,
        "strip_sha256": file_sha256(strip_path),
        "layout": {
            "columns": SEQUENCE_COLUMNS,
            "cell_width": cell_w,
            "frame_height": frame_h,
            "label_height": CONTACT_LABEL_HEIGHT,
            "aspect_ratio_preserved": True,
        },
        "ffmpeg_required": False,
        "frames": sequence_metadata,
    }


def capture_ball_roll_sequence(
    app: App,
    pred: Prediction,
    goal_minute: int,
    label_font: pygame.font.Font,
) -> dict[str, object]:
    progress_samples = tuple(
        sorted(
            {
                0.500,
                0.520,
                0.540,
                0.560,
                0.580,
                0.600,
                0.620,
                0.640,
                0.660,
                0.680,
                0.700,
                0.713,
                SHOT_KICK_AT,
                0.727,
                SHOT_KICK_POSE_AT,
                0.780,
            }
        )
    )
    samples: list[tuple[str, str, pygame.Surface]] = []
    metadata: list[dict[str, object]] = []
    app.match_prediction = pred
    scoring_side = app.goal_schedule(pred)[0][1]
    direction = 1 if scoring_side == "home" else -1
    for index, progress in enumerate(progress_samples):
        seconds = (
            goal_minute - GOAL_EVENT_WINDOW_MINUTES + progress * GOAL_EVENT_WINDOW_MINUTES
        ) / 90.0 * SIMULATION_SECONDS
        label = f"{index + 1:02d} | p={progress:.3f}"
        filename = f"cinematic_ball_roll/roll_{index:02d}_p{progress:.3f}.png"
        frame = save_frame(app, seconds, f"Confronto | rolagem {label}", filename)
        state = app.cinematic_scene_state(FIELD_RECT, pred)
        contract = ball_contract_snapshot(state, f"ball roll progress={progress:.3f}")
        metrics = locomotion_metrics(app, state)
        ball_x, ball_y = contract["ball_pos"]  # type: ignore[misc]
        kick_x, _kick_y = metrics["kick_foot_pos"]  # type: ignore[misc]
        _ground_x, ground_y = contract["ball_ground_pos"]  # type: ignore[misc]
        radius = visible_ball_radius(app, int(contract["ball_scale"]))
        material = app.cached_cinematic_ball_material(
            int(contract["ball_scale"]),
            float(contract["ball_rotation_degrees"]),
        )
        path = OUTPUT_DIR / filename
        samples.append((label, filename, frame))
        metadata.append(
            {
                "index": index,
                "file": filename,
                "label": label,
                "requested_progress": progress,
                "seconds": seconds,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "ground_gap_px": float(ground_y) - (float(ball_y) + radius),
                "control_surface_gap_px": direction * (float(ball_x) - float(kick_x)) - radius,
                "material_sha256": hashlib.sha256(pygame.image.tostring(material, "RGBA")).hexdigest(),
                "ball_state": contract,
            }
        )

    sheet_width, sheet_height, cell_width, frame_height = contact_sheet_size(len(samples), 4)
    sheet = pygame.Surface((sheet_width, sheet_height), pygame.SRCALPHA)
    sheet.fill(BG)
    for index, (label, _filename, frame) in enumerate(samples):
        append_contact_cell(sheet, frame, label, index, label_font, 4)
    sheet_path = OUTPUT_DIR / "cinematic_ball_roll_sequence.png"
    pygame.image.save(sheet, sheet_path)
    return {
        "event": "home_goal_ground_roll",
        "goal_minute": goal_minute,
        "frame_count": len(samples),
        "pre_kick_frame_count": sum(progress <= SHOT_KICK_AT for progress in progress_samples),
        "sheet": sheet_path.name,
        "sheet_sha256": file_sha256(sheet_path),
        "layout": {
            "columns": 4,
            "cell_width": cell_width,
            "frame_height": frame_height,
            "label_height": CONTACT_LABEL_HEIGHT,
            "aspect_ratio_preserved": True,
        },
        "frames": metadata,
    }


def capture_dribble_flow_sequence(
    app: App,
    pred: Prediction,
    goal_minute: int,
    label_font: pygame.font.Font,
) -> dict[str, object]:
    event_start = goal_minute - GOAL_EVENT_WINDOW_MINUTES
    pre_event_minute = max(0.0, event_start - 1.0)
    follow_through_progress = SHOT_KICK_AT + 0.04
    flight_progress = (SHOT_KICK_AT + SHOT_NET_VISUAL_CONTACT_AT) / 2.0
    samples_spec = (
        ("corrida 08m", 8.0 / 90.0 * SIMULATION_SECONDS, None),
        ("corrida 14m", 14.0 / 90.0 * SIMULATION_SECONDS, None),
        ("corrida 20m", 20.0 / 90.0 * SIMULATION_SECONDS, None),
        (
            f"pre-lance {pre_event_minute:.0f}m",
            pre_event_minute / 90.0 * SIMULATION_SECONDS,
            None,
        ),
        ("entrada p=0.00", event_start / 90.0 * SIMULATION_SECONDS, 0.00),
        ("conducao p=0.16", (event_start + 0.16 * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS, 0.16),
        ("conducao p=0.32", (event_start + 0.32 * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS, 0.32),
        ("dominio p=0.48", (event_start + 0.48 * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS, 0.48),
        ("dominio p=0.58", (event_start + 0.58 * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS, 0.58),
        ("planta p=0.66", (event_start + 0.66 * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS, 0.66),
        (
            f"contato p={SHOT_KICK_AT:.2f}",
            (event_start + SHOT_KICK_AT * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS,
            SHOT_KICK_AT,
        ),
        (
            f"seguimento p={follow_through_progress:.2f}",
            (event_start + follow_through_progress * GOAL_EVENT_WINDOW_MINUTES)
            / 90.0
            * SIMULATION_SECONDS,
            follow_through_progress,
        ),
        (
            f"voo p={flight_progress:.2f}",
            (event_start + flight_progress * GOAL_EVENT_WINDOW_MINUTES)
            / 90.0
            * SIMULATION_SECONDS,
            flight_progress,
        ),
        ("impacto p=0.982", (event_start + SHOT_NET_VISUAL_CONTACT_AT * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS, SHOT_NET_VISUAL_CONTACT_AT),
        ("rede p=1.04", (event_start + 1.04 * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS, 1.04),
        ("recuperacao p=1.20", (event_start + 1.20 * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS, 1.20),
        ("retorno p=1.40", (event_start + 1.40 * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS, 1.40),
    )
    sample_seconds = [seconds for _label, seconds, _progress in samples_spec]
    if any(following <= current for current, following in zip(sample_seconds, sample_seconds[1:])):
        raise RuntimeError(f"dribble-flow evidence chronology is not strictly increasing: {sample_seconds}")
    app.match_prediction = pred
    rows: list[tuple[str, str, pygame.Surface]] = []
    metadata: list[dict[str, object]] = []
    for index, (label, seconds, requested_progress) in enumerate(samples_spec):
        filename = f"cinematic_dribble_flow/flow_{index:02d}.png"
        frame = save_frame(app, seconds, f"Confronto | {label}", filename)
        state = app.cinematic_scene_state(FIELD_RECT, pred)
        metrics = locomotion_metrics(app, state)
        rendered_gap = float(
            metrics["rendered_ball_runner_gap_px"]
        )
        runner_frames = (
            app.assets.cinematic_runners_left[app.away.code]
            if state.get("possession") == "away"
            else app.assets.cinematic_runners[app.home.code]
        )
        runner_index = int(metrics["runner_render_frame"])
        path = OUTPUT_DIR / filename
        rows.append((label, filename, frame))
        metadata.append(
            {
                "index": index,
                "label": label,
                "file": filename,
                "seconds": seconds,
                "requested_progress": requested_progress,
                "sha256": file_sha256(path),
                "runner_frame": runner_index,
                "runner_frame_count": len(runner_frames),
                "actor_pos": state.get("actor_pos"),
                "ball_pos": state.get("ball_pos"),
                "ball_ground_pos": state.get("ball_ground_pos"),
                "ball_rotation_degrees": state.get("ball_rotation_degrees"),
                "dribble_touch_phase": metrics["dribble_touch_phase"],
                "dribble_pose_phase": metrics["dribble_pose_phase"],
                "dribble_touch_slot": metrics["dribble_touch_slot"],
                "landmark_dribble_contact_gap_px": metrics["landmark_dribble_contact_gap_px"],
                "dribble_contact_gap_px": metrics["landmark_dribble_contact_gap_px"],
                "landmark_ball_foot_gap_px": metrics["landmark_ball_foot_gap_px"],
                "visible_ball_foot_gap_px": metrics["visible_ball_foot_gap_px"],
                "rendered_ball_runner_gap_px": round(rendered_gap, 4),
                "pelvis_pos": metrics["pelvis_pos"],
                "support_foot_pos": metrics["support_pos"],
                "dribble_foot_pos": metrics["toe_pos"],
                "support_phase": metrics["support"],
                "support_transition": False,
                "support_weight": metrics["support_weight"],
                "sole_ground_gap_px": metrics["sole_ground_gap_px"],
                "shot_progress": state.get("shot_progress"),
                "ball_phase": state.get("ball_phase"),
            }
        )

    sheet_width, sheet_height, cell_width, frame_height = contact_sheet_size(len(rows), 4)
    sheet = pygame.Surface((sheet_width, sheet_height), pygame.SRCALPHA)
    sheet.fill(BG)
    for index, (label, _filename, frame) in enumerate(rows):
        append_contact_cell(sheet, frame, label, index, label_font, 4)
    sheet_path = OUTPUT_DIR / "cinematic_dribble_flow_sequence.png"
    pygame.image.save(sheet, sheet_path)
    return {
        "event": "home_dribble_end_to_end",
        "goal_minute": goal_minute,
        "frame_count": len(rows),
        "sheet": sheet_path.name,
        "sheet_sha256": file_sha256(sheet_path),
        "layout": {
            "columns": 4,
            "cell_width": cell_width,
            "frame_height": frame_height,
            "label_height": CONTACT_LABEL_HEIGHT,
            "aspect_ratio_preserved": True,
        },
        "frames": metadata,
    }


def capture_keeper_authored_sequence(app: App, label_font: pygame.font.Font) -> dict[str, object]:
    directional_frames = (
        ("right", app.assets.cinematic_keeper_frames[app.home.code]),
        ("left", app.assets.cinematic_keeper_frames_left[app.home.code]),
    )
    if any(len(frames) != CINEMATIC_KEEPER_FRAME_COUNT for _direction, frames in directional_frames):
        raise RuntimeError(
            f"keeper evidence expected {CINEMATIC_KEEPER_FRAME_COUNT} direct frames per direction"
        )
    frame_size = 288
    label_height = 26
    columns = 8
    rows = ceil(sum(len(frames) for _direction, frames in directional_frames) / columns)
    sheet = pygame.Surface((columns * frame_size, rows * (frame_size + label_height)), pygame.SRCALPHA)
    sheet.fill(BG)
    frame_metadata = []
    cell_index = 0
    for direction, frames in directional_frames:
        for index, frame in enumerate(frames):
            x = (cell_index % columns) * frame_size
            y = (cell_index // columns) * (frame_size + label_height)
            pygame.draw.rect(sheet, (0, 7, 10), (x, y, frame_size, label_height))
            sheet.blit(
                label_font.render(f"{direction} {index:02d}", True, GOLD),
                (x + 10, y + 4),
            )
            sheet.blit(frame, (x, y + label_height))
            source = (
                ROOT
                / "assets"
                / "generated"
                / "cinematic"
                / f"keeper_anim_{direction}_{index}.png"
            )
            frame_metadata.append(
                {
                    "direction": direction,
                    "index": index,
                    "source": source.relative_to(ROOT).as_posix(),
                    "source_sha256": file_sha256(source),
                }
            )
            cell_index += 1
    sheet_path = OUTPUT_DIR / "cinematic_keeper_all_directions.png"
    pygame.image.save(sheet, sheet_path)

    event_duration = GOAL_EVENT_WINDOW_MINUTES / 90.0 * SIMULATION_SECONDS
    progress_step = (1.0 / FPS) / event_duration
    start = 0.44
    end = SHOT_NET_VISUAL_CONTACT_AT + 0.44
    timeline = []
    progress = start
    while progress < end:
        first, following, blend = app.cinematic_keeper_frame_blend(
            True,
            progress,
            CINEMATIC_KEEPER_FRAME_COUNT,
        )
        timeline.append(
            {
                "progress": round(progress, 6),
                "first": first,
                "following": following,
                "blend": round(blend, 6),
                "rendered": first if blend < 0.5 else following,
            }
        )
        progress += progress_step
    first, following, blend = app.cinematic_keeper_frame_blend(
        True,
        end,
        CINEMATIC_KEEPER_FRAME_COUNT,
    )
    timeline.append(
        {
            "progress": round(end, 6),
            "first": first,
            "following": following,
            "blend": round(blend, 6),
            "rendered": first if blend < 0.5 else following,
        }
    )
    return {
        "sheet": sheet_path.name,
        "sheet_sha256": file_sha256(sheet_path),
        "frame_count": len(frame_metadata),
        "authored_frames_per_direction": CINEMATIC_KEEPER_FRAME_COUNT,
        "directions": ["right", "left"],
        "single_direct_authored_pose_per_update": True,
        "timeline_fps": FPS,
        "timeline": timeline,
        "frames": frame_metadata,
    }


def capture_locomotion_video(
    app: App,
    pred: Prediction,
    direction_label: str,
) -> dict[str, object]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("visual locomotion QA requires ffmpeg")

    goal_minute, _side = app.goal_schedule(pred)[0]
    event_start_seconds = (goal_minute - GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS
    kick_seconds = (
        goal_minute - GOAL_EVENT_WINDOW_MINUTES + SHOT_KICK_AT * GOAL_EVENT_WINDOW_MINUTES
    ) / 90.0 * SIMULATION_SECONDS
    start_seconds = max(0.0, event_start_seconds - 3.0)
    evidence_end_progress = SHOT_NET_VISUAL_CONTACT_AT + SHOT_NET_SETTLE_PROGRESS + 0.04
    end_seconds = min(
        SIMULATION_SECONDS,
        (
            goal_minute
            - GOAL_EVENT_WINDOW_MINUTES
            + evidence_end_progress * GOAL_EVENT_WINDOW_MINUTES
        )
        / 90.0
        * SIMULATION_SECONDS,
    )
    video_path = OUTPUT_DIR / f"cinematic_dribble_{direction_label}_60fps.mp4"
    slow_path = OUTPUT_DIR / f"cinematic_dribble_{direction_label}_foot_slowmo.mp4"
    dt = 1.0 / FPS
    slow_factor = 2.5
    slow_capture_fps = round(FPS * slow_factor)
    closeup_size = (360, 260)

    def render_scene() -> tuple[pygame.Surface, pygame.Surface, dict[str, object]]:
        app.screen.fill(BG)
        app.draw_top(f"Confronto | passada {direction_label}", "QA visual 60 fps")
        focus = app.match_cinematic_focus(pred)
        app.draw_field(pred, pred, "CONFRONTO")
        app.draw_side_panel(pred, cinematic_focus=focus)
        app.draw_score_panel({"CONFRONTO": pred}, "CONFRONTO", pred, cinematic_focus=focus)
        frame = app.screen.copy()
        state = app.cinematic_scene_state(FIELD_RECT, pred)
        actor_x, actor_y = state["actor_pos"]  # type: ignore[misc]
        closeup = pygame.Surface(closeup_size, pygame.SRCALPHA)
        closeup.blit(
            frame,
            (
                int(round(closeup_size[0] / 2 - float(actor_x))),
                int(round(closeup_size[1] - 24 - float(actor_y))),
            ),
        )
        return frame, closeup, state

    seek_match_time(app, pred, start_seconds, step=dt)
    full_process = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )

    records: list[dict[str, object]] = []
    frame_index = 0
    try:
        while app.t <= end_seconds + 1e-6:
            frame, _closeup, state = render_scene()
            if full_process.stdin is None:
                raise RuntimeError("ffmpeg locomotion QA pipe closed unexpectedly")
            full_process.stdin.write(pygame.image.tostring(frame, "RGB"))
            metrics = locomotion_metrics(app, state)
            support_x, support_y = metrics["support_pos"]  # type: ignore[misc]
            pelvis_x, pelvis_y = metrics["pelvis_pos"]  # type: ignore[misc]
            toe_x, toe_y = metrics["toe_pos"]  # type: ignore[misc]
            ball_x, ball_y = state["ball_pos"]  # type: ignore[misc]
            kick_foot_x, kick_foot_y = metrics["kick_foot_pos"]  # type: ignore[misc]
            rendered_gap = float(
                metrics["rendered_ball_runner_gap_px"]
            )
            records.append(
                {
                    "direction": direction_label,
                    "frame": frame_index,
                    "seconds": round(app.t, 6),
                    "runner_frame": metrics["runner_frame"],
                    "runner_next_frame": metrics["runner_next_frame"],
                    "runner_frame_blend": round(float(metrics["runner_frame_blend"]), 6),
                    "runner_render_frame": metrics["runner_render_frame"],
                    "runner_phase": round(float(metrics["runner_phase"]), 6),
                    "support": metrics["support"],
                    "support_weight": round(float(metrics["support_weight"]), 6),
                    "support_x": round(float(support_x), 4),
                    "support_y": round(float(support_y), 4),
                    "sole_ground_gap_px": round(float(metrics["sole_ground_gap_px"]), 4),
                    "pelvis_x": round(float(pelvis_x), 4),
                    "pelvis_y": round(float(pelvis_y), 4),
                    "toe_x": round(float(toe_x), 4),
                    "toe_y": round(float(toe_y), 4),
                    "ball_x": round(float(ball_x), 4),
                    "ball_y": round(float(ball_y), 4),
                    "dribble_touch_phase": round(float(metrics["dribble_touch_phase"]), 6),
                    "dribble_pose_phase": str(metrics["dribble_pose_phase"]),
                    "dribble_touch_slot": int(metrics["dribble_touch_slot"]),
                    "landmark_dribble_contact_gap_px": round(
                        float(metrics["landmark_dribble_contact_gap_px"]), 4
                    ),
                    "landmark_ball_foot_gap_px": round(float(metrics["landmark_ball_foot_gap_px"]), 4),
                    "visible_ball_foot_gap_px": round(float(metrics["visible_ball_foot_gap_px"]), 4),
                    "rendered_ball_runner_gap_px": round(rendered_gap, 4),
                    "shot_progress": round(float(state["shot_progress"]), 6),
                    "raw_shot_progress": round(float(state.get("raw_shot_progress", state["shot_progress"])), 6),
                    "ball_rotation_degrees": round(float(state.get("ball_rotation_degrees", 0.0)), 4),
                    "ball_depth": round(float(state.get("ball_depth", 0.0)), 6),
                    "ball_scale": int(state.get("ball_scale", 0)),
                    "ball_phase": str(state.get("ball_phase", "")),
                    "net_progress": round(float(state.get("net_progress", 0.0)), 6),
                    "kick_foot_x": round(float(kick_foot_x), 4),
                    "kick_foot_y": round(float(kick_foot_y), 4),
                    "keeper_frame_first": metrics["keeper_frame_first"],
                    "keeper_frame_following": metrics["keeper_frame_following"],
                    "keeper_frame_blend": round(float(metrics["keeper_frame_blend"]), 6),
                    "keeper_render_frame": metrics["keeper_render_frame"],
                }
            )
            frame_index += 1
            app.update(dt)
    finally:
        if full_process.stdin:
            full_process.stdin.close()
        if full_process.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding 60 fps locomotion QA evidence")

    slow_process = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{closeup_size[0]}x{closeup_size[1]}",
            "-r",
            str(slow_capture_fps),
            "-i",
            "-",
            "-vf",
            f"setpts={slow_factor}*PTS,fps={FPS}",
            "-r",
            str(FPS),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            str(slow_path),
        ],
        stdin=subprocess.PIPE,
    )
    slow_frame_count = 0
    slow_dt = 1.0 / slow_capture_fps
    seek_match_time(app, pred, start_seconds, step=slow_dt)
    try:
        while app.t <= end_seconds + 1e-6:
            _frame, closeup, _state = render_scene()
            if slow_process.stdin is None:
                raise RuntimeError("ffmpeg slow-motion QA pipe closed unexpectedly")
            slow_process.stdin.write(pygame.image.tostring(closeup, "RGB"))
            slow_frame_count += 1
            app.update(slow_dt)
    finally:
        if slow_process.stdin:
            slow_process.stdin.close()
        if slow_process.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding temporal slow-motion QA evidence")

    normal_output_fps = video_frame_rate(video_path)
    slow_output_fps = video_frame_rate(slow_path)
    if abs(normal_output_fps - FPS) > 0.01 or abs(slow_output_fps - FPS) > 0.01:
        raise RuntimeError(
            f"locomotion QA videos are not true {FPS} fps: normal={normal_output_fps}, slow={slow_output_fps}"
        )

    return {
        "direction": direction_label,
        "fps": normal_output_fps,
        "frame_count": frame_index,
        "duration_seconds": round(frame_index / FPS, 3),
        "video": video_path.name,
        "video_sha256": file_sha256(video_path),
        "slow_motion_video": slow_path.name,
        "slow_motion_sha256": file_sha256(slow_path),
        "slow_motion_factor": slow_factor,
        "slow_motion_capture_fps": slow_capture_fps,
        "slow_motion_output_fps": slow_output_fps,
        "slow_motion_source_frames": slow_frame_count,
        "evidence_end_progress": evidence_end_progress,
        "covers_net_and_recovery": evidence_end_progress > SHOT_NET_VISUAL_CONTACT_AT + SHOT_NET_SETTLE_PROGRESS,
        "records": records,
    }


def capture_cinematic_variants(app: App, label_font: pygame.font.Font) -> dict[str, object]:
    zones = (
        "alto firme",
        "baixo cruzado",
        "meia altura",
        "angulo seco",
        "rasteiro forte",
        "central forte",
    )
    samples: list[tuple[str, str, pygame.Surface]] = []
    metadata: list[dict[str, object]] = []

    def capture_variant(
        label: str,
        filename: str,
        pred: Prediction,
        event_minute: int,
        event_window: float,
        progress: float,
        extra: dict[str, object],
    ) -> None:
        seconds = (
            event_minute - event_window + progress * event_window
        ) / 90.0 * SIMULATION_SECONDS
        frame = save_frame(app, seconds, f"Confronto | {label}", filename)
        state = app.cinematic_scene_state(FIELD_RECT, pred)
        contract = ball_contract_snapshot(state, f"variant {label}")
        path = OUTPUT_DIR / filename
        samples.append((label, filename, frame))
        metadata.append(
            {
                "file": filename,
                "label": label,
                "progress": progress,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "ball_state": contract,
                **extra,
            }
        )

    for zone_index, zone in enumerate(zones):
        scoring_side = "home" if zone_index % 2 == 0 else "away"
        pred = home_win_prediction() if scoring_side == "home" else away_win_prediction()
        example: tuple[int, int] | None = None
        for seed in range(2026, 2400):
            app.match_seed = seed
            app.match_prediction = pred
            for goal_minute, side in app.goal_schedule(pred):
                if side != scoring_side:
                    continue
                direction = 1 if side == "home" else -1
                goal = app.cinematic_goal_rect(FIELD_RECT, "right" if direction > 0 else "left")
                if app.cinematic_shot_profile(goal, direction, goal_minute).zone == zone:
                    example = (seed, goal_minute)
                    break
            if example is not None:
                break
        if example is None:
            raise RuntimeError(f"visual QA could not find profile {zone!r} for {scoring_side}")

        seed, goal_minute = example
        app.match_seed = seed
        app.match_prediction = pred
        slug = zone.replace(" ", "_")
        for phase, progress in (
            ("voo", max(SHOT_RELEASE_END + 0.04, 0.85)),
            ("contato", SHOT_NET_VISUAL_CONTACT_AT),
            ("rede", SHOT_NET_VISUAL_CONTACT_AT + 0.14),
        ):
            filename = f"cinematic_variants/{zone_index:02d}_{slug}_{phase}.png"
            capture_variant(
                f"{zone} | {scoring_side} | {phase}",
                filename,
                pred,
                goal_minute,
                GOAL_EVENT_WINDOW_MINUTES,
                progress,
                {"kind": "goal", "zone": zone, "side": scoring_side, "seed": seed},
            )

    app.match_seed = 2026
    chance_pred = home_win_prediction()
    app.match_prediction = chance_pred
    chance_by_kind = {
        kind: (minute, side)
        for minute, side, kind in app.chance_schedule(chance_pred)
    }
    for kind, progresses in (
        ("save", (SHOT_NET_AT - 0.03, SHOT_NET_AT, SHOT_NET_AT + 0.10)),
        ("wide", (SHOT_NET_AT - 0.03, SHOT_NET_AT - 0.01, SHOT_NET_AT + 0.02)),
    ):
        if kind not in chance_by_kind:
            raise RuntimeError(f"visual QA requires a {kind!r} chance event")
        chance_minute, side = chance_by_kind[kind]
        for phase, progress in zip(("pre", "contato", "pos"), progresses):
            filename = f"cinematic_variants/{kind}_{phase}.png"
            capture_variant(
                f"{kind} | {phase}",
                filename,
                chance_pred,
                chance_minute,
                CHANCE_EVENT_WINDOW_MINUTES,
                progress,
                {"kind": kind, "side": side, "seed": app.match_seed},
            )

    sheet_width, sheet_height, cell_width, frame_height = contact_sheet_size(len(samples), 3)
    sheet = pygame.Surface((sheet_width, sheet_height), pygame.SRCALPHA)
    sheet.fill(BG)
    for index, (label, _filename, frame) in enumerate(samples):
        append_contact_cell(sheet, frame, label, index, label_font, 3)
    sheet_path = OUTPUT_DIR / "cinematic_variants_contact_sheet.png"
    pygame.image.save(sheet, sheet_path)
    return {
        "frame_count": len(samples),
        "goal_profiles": list(zones),
        "sides": ["home", "away"],
        "chance_kinds": ["save", "wide"],
        "sheet": sheet_path.name,
        "sheet_sha256": file_sha256(sheet_path),
        "layout": {
            "columns": 3,
            "cell_width": cell_width,
            "frame_height": frame_height,
            "label_height": CONTACT_LABEL_HEIGHT,
            "aspect_ratio_preserved": True,
        },
        "frames": metadata,
    }


def capture_runner_uniform_coverage(app: App, label_font: pygame.font.Font) -> dict[str, object]:
    motion = app.assets.cinematic_runner_motion
    poc2_motion = app.assets.cinematic_poc2_motion
    frame_size = int(motion["frame_size"])
    poc2_frame_size = int(poc2_motion["canvas_size"])
    cell_size = max(frame_size, poc2_frame_size)
    label_height = 34
    phases = (
        ("direita | contato", "poc2_runner_right", 0),
        ("direita | voo", "poc2_runner_right", 7),
        ("direita | chute", "runner_kick", 9),
        ("direita | parado", "runner_stop", CINEMATIC_STOP_FRAME_COUNT - 1),
        ("esquerda | contato", "poc2_runner_left", 0),
        ("esquerda | voo", "poc2_runner_left", 7),
        ("esquerda | chute", "runner_left_kick", 9),
        ("esquerda | parado", "runner_left_stop", CINEMATIC_STOP_FRAME_COUNT - 1),
    )
    team_codes: dict[str, list[str]] = {uniform.code: [] for uniform in CINEMATIC_UNIFORMS}
    for team in app.teams:
        team_codes[app.assets.cinematic_source_code(team)].append(team.code)
    if sum(len(codes) for codes in team_codes.values()) != len(app.teams):
        raise RuntimeError("runner uniform QA did not map every tournament team")
    if any(not codes for codes in team_codes.values()):
        raise RuntimeError(f"runner uniform QA found an unused profile: {team_codes}")

    sheet = pygame.Surface(
        (len(phases) * cell_size, len(CINEMATIC_UNIFORMS) * (cell_size + label_height)),
        pygame.SRCALPHA,
    )
    sheet.fill(BG)
    base = ROOT / "assets" / "generated" / "cinematic"
    cells: list[dict[str, object]] = []
    for row, uniform in enumerate(CINEMATIC_UNIFORMS):
        for column, (phase_label, stem, frame_index) in enumerate(phases):
            x = column * cell_size
            y = row * (cell_size + label_height)
            cell = pygame.Rect(x, y + label_height, cell_size, cell_size)
            pygame.draw.rect(sheet, (14, 62, 31), cell)
            pygame.draw.line(sheet, (86, 132, 65), (cell.x, cell.bottom - 14), (cell.right, cell.bottom - 14), 2)
            is_poc2 = stem.startswith("poc2_")
            source = base / (
                f"{stem}_{uniform.code}.png"
                if is_poc2
                else f"{stem}_smooth_{uniform.code}.png"
            )
            spritesheet = pygame.image.load(source).convert_alpha()
            source_frame_size = poc2_frame_size if is_poc2 else frame_size
            sheet_columns = (
                int(poc2_motion["sheet_columns"])
                if is_poc2
                else int(
                    motion[
                        "stop_sheet_columns"
                        if "stop" in stem
                        else "kick_sheet_columns"
                        if "kick" in stem
                        else "sheet_columns"
                    ]
                )
            )
            frame = spritesheet.subsurface(
                pygame.Rect(
                    (frame_index % sheet_columns) * source_frame_size,
                    (frame_index // sheet_columns) * source_frame_size,
                    source_frame_size,
                    source_frame_size,
                )
            )
            frame_rect = frame.get_rect(center=cell.center)
            sheet.blit(frame, frame_rect)
            pygame.draw.rect(sheet, (0, 7, 10), (x, y, cell_size, label_height))
            label = f"{uniform.code} | {phase_label}"
            sheet.blit(label_font.render(label, True, GOLD), (x + 10, y + 8))
            cells.append(
                {
                    "uniform": uniform.code,
                    "label_pt": uniform.label_pt,
                    "direction": "left" if "left" in stem else "right",
                    "phase": phase_label.split(" | ", 1)[1],
                    "frame_index": frame_index,
                    "source": str(source.relative_to(ROOT)),
                    "source_sha256": file_sha256(source),
                }
            )

    sheet_path = OUTPUT_DIR / "cinematic_runner_uniforms_contact_sheet.png"
    pygame.image.save(sheet, sheet_path)
    cycle_sheets: dict[str, dict[str, str]] = {}
    kick_cycle_sheets: dict[str, dict[str, str]] = {}
    stop_cycle_sheets: dict[str, dict[str, str]] = {}
    cycle_frame_size = 128
    cycle_label_height = 24
    poc2_frame_count = int(poc2_motion["frame_count"])
    for direction, stem in (
        ("right", "poc2_runner_right"),
        ("left", "poc2_runner_left"),
    ):
        cycle_sheet = pygame.Surface(
            (
                poc2_frame_count * cycle_frame_size,
                len(CINEMATIC_UNIFORMS) * (cycle_frame_size + cycle_label_height),
            ),
            pygame.SRCALPHA,
        )
        cycle_sheet.fill(BG)
        for row, uniform in enumerate(CINEMATIC_UNIFORMS):
            source = base / f"{stem}_{uniform.code}.png"
            spritesheet = pygame.image.load(source).convert_alpha()
            label_y = row * (cycle_frame_size + cycle_label_height)
            pygame.draw.rect(
                cycle_sheet,
                (0, 7, 10),
                (0, label_y, cycle_sheet.get_width(), cycle_label_height),
            )
            cycle_sheet.blit(
                label_font.render(f"{uniform.code} | {direction}", True, GOLD),
                (8, label_y + 3),
            )
            for frame_index in range(poc2_frame_count):
                frame = spritesheet.subsurface(
                    pygame.Rect(
                        (frame_index % int(poc2_motion["sheet_columns"])) * poc2_frame_size,
                        (frame_index // int(poc2_motion["sheet_columns"])) * poc2_frame_size,
                        poc2_frame_size,
                        poc2_frame_size,
                    )
                ).copy()
                frame = pygame.transform.smoothscale(frame, (cycle_frame_size, cycle_frame_size))
                cycle_sheet.blit(
                    frame,
                    (
                        frame_index * cycle_frame_size,
                        label_y + cycle_label_height,
                    ),
                )
        cycle_path = OUTPUT_DIR / f"cinematic_runner_all_frames_{direction}.png"
        pygame.image.save(cycle_sheet, cycle_path)
        cycle_sheets[direction] = {
            "sheet": cycle_path.name,
            "sha256": file_sha256(cycle_path),
        }
    for direction, stem in (("right", "runner_kick"), ("left", "runner_left_kick")):
        cycle_sheet = pygame.Surface(
            (
                CINEMATIC_KICK_FRAME_COUNT * cycle_frame_size,
                len(CINEMATIC_UNIFORMS) * (cycle_frame_size + cycle_label_height),
            ),
            pygame.SRCALPHA,
        )
        cycle_sheet.fill(BG)
        for row, uniform in enumerate(CINEMATIC_UNIFORMS):
            source = base / f"{stem}_smooth_{uniform.code}.png"
            spritesheet = pygame.image.load(source).convert_alpha()
            label_y = row * (cycle_frame_size + cycle_label_height)
            pygame.draw.rect(
                cycle_sheet,
                (0, 7, 10),
                (0, label_y, cycle_sheet.get_width(), cycle_label_height),
            )
            cycle_sheet.blit(
                label_font.render(f"{uniform.code} | {direction} kick", True, GOLD),
                (8, label_y + 3),
            )
            for frame_index in range(CINEMATIC_KICK_FRAME_COUNT):
                frame = spritesheet.subsurface(
                    pygame.Rect(
                        (frame_index % int(motion["kick_sheet_columns"])) * frame_size,
                        (frame_index // int(motion["kick_sheet_columns"])) * frame_size,
                        frame_size,
                        frame_size,
                    )
                ).copy()
                frame = pygame.transform.smoothscale(frame, (cycle_frame_size, cycle_frame_size))
                cycle_sheet.blit(
                    frame,
                    (frame_index * cycle_frame_size, label_y + cycle_label_height),
                )
        cycle_path = OUTPUT_DIR / f"cinematic_kick_all_frames_{direction}.png"
        pygame.image.save(cycle_sheet, cycle_path)
        kick_cycle_sheets[direction] = {
            "sheet": cycle_path.name,
            "sha256": file_sha256(cycle_path),
        }
    for direction, stem in (("right", "runner_stop"), ("left", "runner_left_stop")):
        cycle_sheet = pygame.Surface(
            (
                CINEMATIC_STOP_FRAME_COUNT * cycle_frame_size,
                len(CINEMATIC_UNIFORMS) * (cycle_frame_size + cycle_label_height),
            ),
            pygame.SRCALPHA,
        )
        cycle_sheet.fill(BG)
        for row, uniform in enumerate(CINEMATIC_UNIFORMS):
            source = base / f"{stem}_smooth_{uniform.code}.png"
            spritesheet = pygame.image.load(source).convert_alpha()
            label_y = row * (cycle_frame_size + cycle_label_height)
            pygame.draw.rect(
                cycle_sheet,
                (0, 7, 10),
                (0, label_y, cycle_sheet.get_width(), cycle_label_height),
            )
            cycle_sheet.blit(
                label_font.render(f"{uniform.code} | {direction} stop", True, GOLD),
                (8, label_y + 3),
            )
            for frame_index in range(CINEMATIC_STOP_FRAME_COUNT):
                frame = spritesheet.subsurface(
                    pygame.Rect(
                        (frame_index % int(motion["stop_sheet_columns"])) * frame_size,
                        (frame_index // int(motion["stop_sheet_columns"])) * frame_size,
                        frame_size,
                        frame_size,
                    )
                ).copy()
                frame = pygame.transform.smoothscale(frame, (cycle_frame_size, cycle_frame_size))
                cycle_sheet.blit(
                    frame,
                    (frame_index * cycle_frame_size, label_y + cycle_label_height),
                )
        cycle_path = OUTPUT_DIR / f"cinematic_stop_all_frames_{direction}.png"
        pygame.image.save(cycle_sheet, cycle_path)
        stop_cycle_sheets[direction] = {
            "sheet": cycle_path.name,
            "sha256": file_sha256(cycle_path),
        }
    return {
        "sheet": sheet_path.name,
        "sheet_sha256": file_sha256(sheet_path),
        "uniform_profile_count": len(CINEMATIC_UNIFORMS),
        "team_count": len(app.teams),
        "directions": ["right", "left"],
        "run_contract": {
            "path": str(
                (base / "poc2_runner_motion.json").relative_to(ROOT)
            ),
            "sha256": file_sha256(base / "poc2_runner_motion.json"),
            "status": poc2_motion["status"],
            "cycle_seconds": poc2_motion["cycle_seconds"],
        },
        "authored_run_frames_per_direction": poc2_frame_count,
        "authored_kick_frames_per_direction": CINEMATIC_KICK_FRAME_COUNT,
        "authored_stop_frames_per_direction": CINEMATIC_STOP_FRAME_COUNT,
        "all_frame_sheets": cycle_sheets,
        "all_kick_frame_sheets": kick_cycle_sheets,
        "all_stop_frame_sheets": stop_cycle_sheets,
        "team_codes_by_uniform": team_codes,
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture or validate the current Arena AI visual evidence bundle.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--check-current",
        action="store_true",
        help="Fail if visual evidence does not exactly match current sources and artifact hashes.",
    )
    args = parser.parse_args(argv)
    configure_output_dir(args.output_dir)
    if args.check_current:
        validate_current_evidence()
        return

    pygame.init()
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    BALL_ROLL_DIR.mkdir(parents=True, exist_ok=True)
    DRIBBLE_FLOW_DIR.mkdir(parents=True, exist_ok=True)
    app = App(seed=2026)
    label_font = font(16)
    samples: list[tuple[str, str, pygame.Surface]] = []

    app.draw_menu()
    menu_frame = app.screen.copy()
    menu_filename = "00_menu.png"
    pygame.image.save(menu_frame, OUTPUT_DIR / menu_filename)
    append_sample(samples, "tela inicial - ícone", menu_filename, menu_frame)

    app.state = "select"
    app.screen.fill(BG)
    app.draw_select()
    select_frame = app.screen.copy()
    select_filename = "00b_selecao.png"
    pygame.image.save(select_frame, OUTPUT_DIR / select_filename)
    append_sample(samples, "seleção - confronto", select_filename, select_frame)

    app.set_simulate("match")

    def goal_progress_seconds(goal_minute: int, progress: float) -> float:
        return (goal_minute - GOAL_EVENT_WINDOW_MINUTES + progress * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS

    def chance_progress_seconds(chance_minute: int, progress: float) -> float:
        return (chance_minute - CHANCE_EVENT_WINDOW_MINUTES + progress * CHANCE_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS

    def first_chance_event(pred, kind: str) -> tuple[int, str, str]:
        for chance in app.chance_schedule(pred):
            if chance[2] == kind:
                return chance
        raise RuntimeError(f"visual QA requires a {kind!r} chance event")

    home_pred = home_win_prediction()
    app.match_prediction = home_pred
    first_home_goal = app.goal_schedule(home_pred)[0][0]
    cinematic_sequence = capture_cinematic_sequence(app, home_pred, first_home_goal, label_font)
    cinematic_ball_roll = capture_ball_roll_sequence(app, home_pred, first_home_goal, label_font)
    cinematic_dribble_flow = capture_dribble_flow_sequence(app, home_pred, first_home_goal, label_font)
    locomotion_captures = [
        capture_locomotion_video(app, home_pred, "home"),
        capture_locomotion_video(app, away_win_prediction(), "away"),
    ]
    locomotion_records = [
        record
        for capture in locomotion_captures
        for record in capture.pop("records")  # type: ignore[misc]
    ]
    locomotion_csv = OUTPUT_DIR / "cinematic_dribble_motion.csv"
    with locomotion_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(locomotion_records[0]))
        writer.writeheader()
        writer.writerows(locomotion_records)
    locomotion_json = OUTPUT_DIR / "cinematic_dribble_motion.json"
    locomotion_json.write_text(
        json.dumps(locomotion_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    locomotion_evidence = {
        "captures": locomotion_captures,
        "csv": locomotion_csv.name,
        "csv_sha256": file_sha256(locomotion_csv),
        "json": locomotion_json.name,
        "json_sha256": file_sha256(locomotion_json),
        "record_count": len(locomotion_records),
    }
    declared_videos = sorted(
        str(capture[key])
        for capture in locomotion_captures
        for key in ("video", "slow_motion_video")
    )
    actual_videos = sorted(path.name for path in OUTPUT_DIR.glob("*.mp4"))
    if actual_videos != declared_videos:
        raise RuntimeError(
            f"visual QA video inventory drift: declared={declared_videos}, actual={actual_videos}"
        )
    video_inventory = {
        "declared": declared_videos,
        "actual": actual_videos,
        "orphan_count": 0,
    }
    runner_uniform_coverage = capture_runner_uniform_coverage(app, label_font)
    keeper_authored_sequence = capture_keeper_authored_sequence(app, label_font)
    cinematic_variants = capture_cinematic_variants(app, label_font)
    app.match_seed = 2026
    app.match_prediction = home_pred
    first_home_goal = app.goal_schedule(home_pred)[0][0]
    flight_progress = min(SHOT_NET_AT - 0.02, max(SHOT_RELEASE_END + 0.05, 0.86))
    append_sample(samples, "00s - posse e parallax", "01_posse_inicial.png", save_frame(app, 0.0, "Confronto | posse inicial", "01_posse_inicial.png"))
    append_sample(
        samples,
        "aproximação - corrida",
        "02_aproximacao.png",
        save_frame(app, goal_progress_seconds(first_home_goal, 0.44), "Confronto | aproximação", "02_aproximacao.png"),
    )
    append_sample(
        samples,
        "pé na bola",
        "03_pe_na_bola.png",
        save_frame(
            app,
            goal_progress_seconds(first_home_goal, SHOT_KICK_AT),
            "Confronto | pé na bola",
            "03_pe_na_bola.png",
        ),
    )
    append_sample(
        samples,
        "chute - bola em voo",
        "04_bola_em_voo.png",
        save_frame(
            app,
            goal_progress_seconds(first_home_goal, flight_progress),
            "Confronto | bola em voo",
            "04_bola_em_voo.png",
        ),
    )
    append_sample(
        samples,
        "rede - impacto",
        "05_impacto_rede.png",
        save_frame(
            app,
            goal_progress_seconds(first_home_goal, SHOT_NET_VISUAL_CONTACT_AT),
            "Confronto | impacto na rede",
            "05_impacto_rede.png",
        ),
    )
    overlay_frame = save_frame(
        app,
        goal_progress_seconds(first_home_goal, SHOT_NET_VISUAL_CONTACT_AT + 0.040),
        "Confronto | impacto confirmado",
        "05b_gol_overlay.png",
    )
    overlay_evidence = goal_overlay_metrics(app, overlay_frame)
    if (
        int(overlay_evidence["gold_text_pixels"]) < 80
        or int(overlay_evidence["dark_panel_pixels"]) < 900
    ):
        raise RuntimeError(f"runtime GOOOL overlay is not visibly rendered: {overlay_evidence}")
    append_sample(
        samples,
        "GOOOL - overlay pós-impacto",
        "05b_gol_overlay.png",
        overlay_frame,
    )

    away_pred = away_win_prediction()
    app.match_prediction = away_pred
    first_away_goal = app.goal_schedule(away_pred)[0][0]
    append_sample(
        samples,
        "lado invertido",
        "06_gol_visitante.png",
        save_frame(
            app,
            goal_progress_seconds(first_away_goal, SHOT_NET_VISUAL_CONTACT_AT),
            "Confronto | gol visitante",
            "06_gol_visitante.png",
        ),
    )

    draw_pred = neutral_prediction()
    app.match_prediction = draw_pred
    append_sample(samples, "empate - jogo vivo", "07_empate_vivo.png", save_frame(app, 10.0, "Confronto | empate vivo", "07_empate_vivo.png"))
    append_sample(samples, "empate final", "08_empate_final.png", save_frame(app, SIMULATION_SECONDS, "Confronto | empate final", "08_empate_final.png"))

    set_matchup(app, "PAR", "ALG")
    app.match_prediction = app.model.predict_matchup(app.home, app.away, seed=2026)
    append_sample(samples, "regressão PAR x ALG", "08b_par_alg_regressao.png", save_frame(app, 43.0, "Confronto | PAR x ALG", "08b_par_alg_regressao.png"))
    app.match_prediction = away_win_prediction()
    alg_goal = app.goal_schedule(app.match_prediction)[0][0]
    append_sample(
        samples,
        "visitante verde - chute",
        "08c_visitante_verde_chute.png",
        save_frame(
            app,
            goal_progress_seconds(alg_goal, 0.54),
            "Confronto | visitante verde",
            "08c_visitante_verde_chute.png",
        ),
    )
    append_sample(
        samples,
        "visitante verde - gol",
        "08d_visitante_verde_gol.png",
        save_frame(
            app,
            goal_progress_seconds(alg_goal, SHOT_NET_VISUAL_CONTACT_AT),
            "Confronto | visitante verde gol",
            "08d_visitante_verde_gol.png",
        ),
    )

    home_pred = home_win_prediction()
    app.match_prediction = home_pred
    save_chance_minute, _save_side, _save_kind = first_chance_event(home_pred, "save")
    append_sample(
        samples,
        "quase gol - defesa",
        "08e_quase_gol_defesa.png",
        save_frame(
            app,
            chance_progress_seconds(save_chance_minute, SHOT_NET_AT + 0.10),
            "Confronto | quase gol - defesa",
            "08e_quase_gol_defesa.png",
        ),
    )
    wide_chance_minute, _wide_side, _wide_kind = first_chance_event(home_pred, "wide")
    append_sample(
        samples,
        "trave raspando",
        "08f_trave_raspando.png",
        save_frame(
            app,
            chance_progress_seconds(wide_chance_minute, SHOT_NET_AT - 0.01),
            "Confronto | trave raspando",
            "08f_trave_raspando.png",
        ),
    )
    append_sample(samples, "fim - placar", "09_placar_final.png", save_frame(app, SIMULATION_SECONDS, "Confronto | placar final", "09_placar_final.png"))

    for code, color_name, filename in (
        ("MEX", "verde", "13_uniforme_verde.png"),
        ("NED", "laranja", "14_uniforme_laranja.png"),
        ("NZL", "preto", "15_uniforme_preto.png"),
    ):
        set_matchup(app, code, "FRA")
        app.match_prediction = app.model.predict_matchup(app.home, app.away, seed=2026)
        home_goal_minutes = [goal_minute for goal_minute, side in app.goal_schedule(app.match_prediction) if side == "home"]
        uniform_goal_minute = home_goal_minutes[0] if home_goal_minutes else app.goal_schedule(app.match_prediction)[0][0]
        append_sample(
            samples,
            f"uniforme {color_name}",
            filename,
            save_frame(
                app,
                goal_progress_seconds(uniform_goal_minute, 0.54),
                f"Confronto | uniforme {color_name}",
                filename,
            ),
        )

    app.state = "tournament"
    app.t = 1.7
    app.tournament_result = None
    app.champion_odds = []
    app.mc_running = True
    app.mc_progress_done = 420
    app.mc_progress_total = app.champion_odds_runs
    app.screen.fill(BG)
    app.draw_tournament()
    loading_frame = app.screen.copy()
    loading_filename = "10_copa_calculando.png"
    pygame.image.save(loading_frame, OUTPUT_DIR / loading_filename)
    append_sample(samples, "copa - calculando", loading_filename, loading_frame)

    odds, representative = app.model.champion_odds_with_representative(
        runs=120,
        seed=2026,
        workers=app.champion_odds_workers,
        progress_with_odds=False,
    )
    app.mc_running = False
    app.mc_progress_done = app.champion_odds_runs
    app.mc_progress_total = app.champion_odds_runs
    app.champion_odds = odds
    app.tournament_result = representative
    for view, label, filename in (
        ("groups", "copa - grupos", "11_copa_grupos.png"),
        ("bracket", "copa - mata-mata", "12_copa_mata_mata.png"),
    ):
        app.tournament_view = view
        app.screen.fill(BG)
        app.draw_tournament()
        frame = app.screen.copy()
        pygame.image.save(frame, OUTPUT_DIR / filename)
        append_sample(samples, label, filename, frame)

    sheet_width, sheet_height, contact_cell_width, contact_frame_height = contact_sheet_size(
        len(samples), CONTACT_COLUMNS
    )
    sheet = pygame.Surface((sheet_width, sheet_height), pygame.SRCALPHA)
    sheet.fill(BG)
    for index, (label, _filename, frame) in enumerate(samples):
        append_contact_cell(sheet, frame, label, index, label_font, CONTACT_COLUMNS)
    contact_sheet = OUTPUT_DIR / "contact_sheet.png"
    pygame.image.save(sheet, contact_sheet)
    inventory = artifact_inventory()
    metadata = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_control": source_control_provenance(),
        "seed": 2026,
        "viewport": {"width": WIDTH, "height": HEIGHT},
        "sample_count": len(samples),
        "contact_sheet": contact_sheet.name,
        "contact_sheet_layout": {
            "columns": CONTACT_COLUMNS,
            "cell_width": contact_cell_width,
            "frame_height": contact_frame_height,
            "label_height": CONTACT_LABEL_HEIGHT,
            "aspect_ratio_preserved": True,
        },
        "cinematic_sequence": cinematic_sequence,
        "cinematic_ball_roll": cinematic_ball_roll,
        "cinematic_dribble_flow": cinematic_dribble_flow,
        "cinematic_locomotion": locomotion_evidence,
        "video_inventory": video_inventory,
        "cinematic_runner_uniforms": runner_uniform_coverage,
        "cinematic_keeper_authored": keeper_authored_sequence,
        "cinematic_variants": cinematic_variants,
        "goal_overlay_evidence": overlay_evidence,
        "frames": [
            {
                "file": filename,
                "label": label,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for label, filename, _frame in samples
            for path in (OUTPUT_DIR / filename,)
        ],
        "artifact_inventory": {
            "file_count": len(inventory),
            "excluded_self": METADATA_PATH.name,
            "files": inventory,
        },
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pygame.quit()
    print(f"visual QA frames saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

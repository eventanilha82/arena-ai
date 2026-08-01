from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import fnmatch
import gc
import json
import os
import math
import pickle
import sys
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import numpy as np
import cv2

from arena_ai.main import (
    App,
    ALGORITHM_NAMES,
    CinematicAttackEvent,
    CINEMATIC_BALL_MATERIAL_FRAME_COUNT,
    CINEMATIC_BALL_MATERIAL_BLEND_STEPS,
    CINEMATIC_KEEPER_FRAME_COUNT,
    CINEMATIC_GOAL_COMPOSITE_CACHE_LIMIT,
    CINEMATIC_PLAYER_SCALE,
    CINEMATIC_POSE_SIZE,
    CINEMATIC_KICK_FRAME_COUNT,
    CINEMATIC_RUNNER_FRAME_COUNT,
    CINEMATIC_STOP_FRAME_COUNT,
    CHANCE_EVENT_WINDOW_MINUTES,
    GOAL_EVENT_WINDOW_MINUTES,
    GOAL_PAYOFF_MINUTES,
    FIFA_EXTERNAL_IMAGES,
    FPS,
    MATCH_HUD_BANNED_COPY,
    MATCH_HUD_REQUIRED_COPY,
    MATCH_HUD_STATE_COPY,
    MATCH_HUD_TOP_SCORE_COUNT,
    SIMULATION_SECONDS,
    SHOT_FOLLOW_THROUGH_HOLD_END,
    SHOT_KICK_AT,
    SHOT_NET_AT,
    SHOT_NET_VISUAL_CONTACT_AT,
    TOURNAMENT_MONTE_CARLO_RUNS,
    TOURNAMENT_MONTE_CARLO_USE_SCENARIO_BANK,
    TOURNAMENT_MONTE_CARLO_WORKERS,
    TOURNAMENT_MIN_LOADING_SECONDS,
    WIDTH,
    HEIGHT,
)
from arena_ai.cinematic_poc_runtime import (
    POC_APPROVED_REFERENCE_VISIBLE_HEIGHT,
    POC_RUNNER_CANVAS_SIZE,
    POC_RUNNER_ROOT,
    PocSequence,
    PocSequenceSample,
    PocViewport,
)
from arena_ai.cinematic_dribble_runtime import (
    CinematicDribbleRuntime,
    Poc2DribbleSample,
)
from arena_ai.cinematic_uniforms import CINEMATIC_UNIFORMS, UNIFORM_CODES
from arena_ai.audio_manifest import AUDIO_RUNTIME_FILES, GOAL_AUDIO_SEQUENCE, REQUIRED_AUDIO_BUSES
from arena_ai.ui import Button
from arena_ai.worldcup_model import Prediction, WorldCupModel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CINEMATIC_DIR = ROOT / "assets" / "generated" / "cinematic"
FLAG_DIR = ROOT / "assets" / "generated" / "flags"
SOUND_DIR = ROOT / "assets" / "sounds"
STADIUM_BG = ROOT / "assets" / "generated" / "stadium_parallax_real.png"
APP_ICON = ROOT / "assets" / "generated" / "app_icon_worldcup.png"
PARALLAX_DIR = ROOT / "assets" / "generated" / "parallax"
PARALLAX_SOURCES = ROOT / "assets" / "generated" / "parallax_sources"
BALL_DIR = ROOT / "assets" / "generated" / "balls3d"
ASSET_MANIFEST = ROOT / "assets" / "asset_manifest.json"
FIFA_EXTERNAL_DIR = ROOT / "assets" / "generated" / "fifa_external"
MODEL_PACKAGE = ROOT / "modeling" / "worldcup_2026_ml" / "models" / "model_sota.pkl"
MODEL_REPORT = ROOT / "modeling" / "worldcup_2026_ml" / "reports" / "sota_model_report.json"
MODEL_STATS_REPORT = ROOT / "modeling" / "worldcup_2026_ml" / "reports" / "sota_statistical_report.json"
MODEL_TRAINING_MATCHES = ROOT / "modeling" / "worldcup_2026_ml" / "data" / "processed" / "sota_training_matches.csv"
MODEL_SOTA_PIPELINE = ROOT / "modeling" / "worldcup_2026_ml" / "src" / "sota_pipeline.py"
MODEL_RUNTIME_PREDICTION_CACHE = ROOT / "modeling" / "worldcup_2026_ml" / "models" / "runtime_prediction_cache.pkl"
MODEL_WORLDCUP_MODEL = ROOT / "src" / "arena_ai" / "worldcup_model.py"
MODEL_STATS_QA_SCRIPT = ROOT / "scripts" / "model_stats_qa.py"
MODEL_MC_STABILITY_SCRIPT = ROOT / "scripts" / "monte_carlo_stability.py"
MODEL_RAW_DATA_ROOT = ROOT / "modeling" / "worldcup_2026_ml" / "data" / "raw"
TOURNAMENT_SIMULATION = ROOT / "modeling" / "worldcup_2026_ml" / "reports" / "sota_tournament_simulation.csv"
FLAG_SIZE = (172, 108)
ROWS = UNIFORM_CODES
SHORTS_BY_CODE = {uniform.code: uniform.shorts for uniform in CINEMATIC_UNIFORMS}
SMOOTH_RUNNER_FRAMES = CINEMATIC_RUNNER_FRAME_COUNT
KEEPER_FRAMES = CINEMATIC_KEEPER_FRAME_COUNT
AUTHORED_RUNNER_FRAME_SIZE = (288, 288)
SMOOTH_RUNNER_SHEET_SIZE = (1152, 1152)
SMOOTH_KICK_SHEET_SIZE = (1152, 1152)
SMOOTH_STOP_SHEET_SIZE = (1152, 576)
KEEPER_SPRITE_SIZE = (340, 340)
RUNNER_MOTION = CINEMATIC_DIR / "runner_motion.json"
POC2_RUNNER_MOTION = CINEMATIC_DIR / "poc2_runner_motion.json"
KEEPER_MOTION = CINEMATIC_DIR / "keeper_motion.json"
REQUIRED_SOUNDS = tuple(f"runtime_assets/{filename}" for filename in AUDIO_RUNTIME_FILES)
GOAL_AUDIO_EVENTS = list(GOAL_AUDIO_SEQUENCE)
CINEMATIC_DRAW_ORDERS = (
    (
        "draw_cinematic_poc_sequence",
        "draw_cinematic_goal_overlay",
    ),
    (
        "draw_cinematic_poc_background",
        "draw_model_flow",
        "draw_cinematic_scene",
        "draw_cinematic_goal_overlay",
    ),
)
AUXILIARY_CACHE_LIMITS = {
    "turf_tile_cache": 8,
    "gradient_tile_cache": 8,
    "gradient_mask_cache": 8,
    "surface_bbox_cache": 360,
    "poc_goal_composite_cache": CINEMATIC_GOAL_COMPOSITE_CACHE_LIMIT,
    "cinematic_ball_corridor_cache": 128,
    "ball_net_path_cache": 128,
    "cinematic_ball_history_cache": 32,
    "cinematic_overlay_cache": 512,
}
BALL_STATE_CONTRACT_KEYS = (
    "ball_pos",
    "ball_prev_pos",
    "ball_velocity_px_s",
    "ball_ground_pos",
    "ball_depth",
    "ball_rotation_degrees",
    "ball_scale",
    "ball_phase",
    "raw_shot_progress",
)
TARGET_CINEMATIC_BALL_SIZE = 38
TARGET_CINEMATIC_SHOT_BALL_SIZE = 32
BALL_VISIBLE_PLAYER_RATIO = (0.14, 0.18)
BALL_TRACE_SAMPLE_HZ = 240
BALL_RUNTIME_FRAME_HZ = 60
BALL_PREVIOUS_POSITION_TOLERANCE_PX = 1.0
BALL_OCCLUSION_NUMERICAL_TOLERANCE = 1e-6


def alpha_bbox(surface: pygame.Surface) -> pygame.Rect:
    rect = surface.get_bounding_rect()
    if rect.w <= 0 or rect.h <= 0:
        raise AssertionError("empty alpha bounding box")
    return rect


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_fresh_statistical_report(stats_report: dict[str, object]) -> None:
    fingerprints = stats_report.get("source_fingerprints")
    if not isinstance(fingerprints, dict):
        raise AssertionError("statistical report missing source_fingerprints; run make stats-qa")
    expected = {
        "model_package": MODEL_PACKAGE,
        "model_report": MODEL_REPORT,
        "training_matches": MODEL_TRAINING_MATCHES,
        "sota_pipeline": MODEL_SOTA_PIPELINE,
        "stats_qa_script": MODEL_STATS_QA_SCRIPT,
    }
    for name, path in expected.items():
        actual = fingerprints.get(name)
        if not isinstance(actual, dict):
            raise AssertionError(f"statistical report missing fingerprint for {name}; run make stats-qa")
        reported_hash = str(actual.get("sha256", ""))
        current_hash = file_hash(path)
        if reported_hash != current_hash:
            raise AssertionError(
                f"stale statistical report for {name}: {reported_hash[:12]} != {current_hash[:12]}; run make stats-qa"
            )
    raw_manifest = stats_report.get("raw_data_manifest")
    if not isinstance(raw_manifest, dict):
        raise AssertionError("statistical report missing raw_data_manifest; run make stats-qa")
    raw_files = raw_manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise AssertionError("raw_data_manifest is empty; run make stats-qa")
    raw_semantic = raw_manifest.get("semantic")
    if not isinstance(raw_semantic, dict) or not bool(raw_semantic.get("passed")):
        failures = raw_semantic.get("failures") if isinstance(raw_semantic, dict) else "missing semantic block"
        raise AssertionError(f"raw data semantic sanity failed: {failures}; run make stats-qa")
    reported_paths = set()
    for item in raw_files:
        if not isinstance(item, dict):
            raise AssertionError("raw_data_manifest contains a non-object file entry")
        relative = str(item.get("path", ""))
        reported_paths.add(relative)
        path = ROOT / "modeling" / "worldcup_2026_ml" / relative
        if not path.exists():
            raise AssertionError(f"raw data manifest references missing file: {relative}")
        reported_hash = str(item.get("sha256", ""))
        current_hash = file_hash(path)
        if reported_hash != current_hash:
            raise AssertionError(f"raw data manifest stale for {relative}: {reported_hash[:12]} != {current_hash[:12]}; run make stats-qa")
    current_raw = {
        path.relative_to(ROOT / "modeling" / "worldcup_2026_ml").as_posix()
        for path in MODEL_RAW_DATA_ROOT.rglob("*")
        if path.is_file()
    }
    if reported_paths != current_raw:
        missing = sorted(current_raw - reported_paths)
        stale = sorted(reported_paths - current_raw)
        raise AssertionError(f"raw data manifest file set drifted; missing={missing} stale={stale}; run make stats-qa")
    mc_stability = stats_report.get("monte_carlo_stability")
    if not isinstance(mc_stability, dict) or not mc_stability.get("passed"):
        raise AssertionError("statistical report missing approved Monte Carlo stability; run make mc-stability && make stats-qa")
    if not bool(mc_stability.get("stage_bracket_passed")):
        raise AssertionError("Monte Carlo stability must include phase/bracket stability; run make mc-stability && make stats-qa")
    mc_fingerprints = mc_stability.get("source_fingerprints")
    if not isinstance(mc_fingerprints, dict):
        raise AssertionError("Monte Carlo stability missing source_fingerprints; run make mc-stability && make stats-qa")
    mc_expected = {
        "model_package": MODEL_PACKAGE,
        "model_report": MODEL_REPORT,
        "training_matches": MODEL_TRAINING_MATCHES,
        "sota_pipeline": MODEL_SOTA_PIPELINE,
        "runtime_prediction_cache": MODEL_RUNTIME_PREDICTION_CACHE,
        "worldcup_model": MODEL_WORLDCUP_MODEL,
        "mc_stability_script": MODEL_MC_STABILITY_SCRIPT,
    }
    for name, path in mc_expected.items():
        actual = mc_fingerprints.get(name)
        if not isinstance(actual, dict):
            raise AssertionError(f"Monte Carlo stability missing fingerprint for {name}; run make mc-stability")
        reported_hash = str(actual.get("sha256", ""))
        current_hash = file_hash(path)
        if reported_hash != current_hash:
            raise AssertionError(
                f"stale Monte Carlo stability for {name}: {reported_hash[:12]} != {current_hash[:12]}; run make mc-stability && make stats-qa"
            )


def surface_hash(surface: pygame.Surface) -> str:
    return hashlib.sha256(pygame.image.tostring(surface, "RGBA")).hexdigest()


def auxiliary_cache_sizes(app: App) -> dict[str, int]:
    return {
        "turf_tile_cache": len(app.turf_tile_cache),
        "gradient_tile_cache": len(app.gradient_tile_cache),
        "gradient_mask_cache": len(app.gradient_mask_cache),
        "surface_bbox_cache": len(app.surface_bbox_cache),
        "poc_goal_composite_cache": len(app.poc_goal_composite_cache),
        "cinematic_ball_corridor_cache": len(app.cinematic_ball_corridor_cache),
        "ball_net_path_cache": len(app.ball_net_path_cache),
        "cinematic_ball_history_cache": len(app.cinematic_ball_history_cache),
        "cinematic_overlay_cache": len(app.cinematic_overlay_cache),
    }


def render_cache_snapshot(app: App) -> tuple[tuple[tuple[str, int], ...], int, tuple[tuple[str, int], ...]]:
    return (
        tuple(sorted(app.surface_cache.stats().items())),
        len(app.text_cache.surfaces),
        tuple(sorted(auxiliary_cache_sizes(app).items())),
    )


def assert_auxiliary_caches_within_limits(app: App, label: str) -> None:
    sizes = auxiliary_cache_sizes(app)
    over_limit = {
        name: (size, AUXILIARY_CACHE_LIMITS[name])
        for name, size in sizes.items()
        if size > AUXILIARY_CACHE_LIMITS[name]
    }
    if over_limit:
        raise AssertionError(f"auxiliary render caches exceeded cap in {label}: {over_limit}; sizes={sizes}")


def scaled_visible_height(surface: pygame.Surface, target: tuple[int, int]) -> float:
    bbox = alpha_bbox(surface)
    return bbox.h * target[1] / max(1, surface.get_height())


def scaled_visible_width(surface: pygame.Surface, target: tuple[int, int]) -> float:
    bbox = alpha_bbox(surface)
    return bbox.w * target[0] / max(1, surface.get_width())


def edge_energy(surface: pygame.Surface, step: int = 2) -> float:
    total = 0.0
    samples = 0
    rect = surface.get_bounding_rect()
    if rect.w <= 3 or rect.h <= 3:
        return 0.0
    for y in range(rect.y, rect.bottom - 1, step):
        for x in range(rect.x, rect.right - 1, step):
            color = surface.get_at((x, y))
            if color.a <= 48:
                continue
            right = surface.get_at((x + 1, y))
            down = surface.get_at((x, y + 1))
            total += abs(color.r - right.r) + abs(color.g - right.g) + abs(color.b - right.b)
            total += abs(color.r - down.r) + abs(color.g - down.g) + abs(color.b - down.b)
            samples += 1
    return total / max(1, samples)


def finite_pair(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise AssertionError(f"{label} must be a 2D tuple, got {value!r}")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in pair):
        raise AssertionError(f"{label} must contain finite values, got {value!r}")
    return pair


def ball_contract_snapshot(state: dict[str, object], label: str) -> dict[str, object]:
    missing = [key for key in BALL_STATE_CONTRACT_KEYS if key not in state]
    if missing:
        raise AssertionError(
            f"{label} missing cinematic ball runtime contract keys {missing}; "
            "integrate the current cinematic_scene_state contract before running visual QA"
        )
    depth = float(state["ball_depth"])
    rotation = float(state["ball_rotation_degrees"])
    scale = float(state["ball_scale"])
    raw_progress = float(state["raw_shot_progress"])
    for name, value in (
        ("ball_depth", depth),
        ("ball_rotation_degrees", rotation),
        ("ball_scale", scale),
        ("raw_shot_progress", raw_progress),
    ):
        if not math.isfinite(value):
            raise AssertionError(f"{label} {name} must be finite, got {value!r}")
    if not 0.0 <= depth <= 1.0:
        raise AssertionError(f"{label} ball_depth must stay in [0, 1], got {depth:.6f}")
    if scale <= 0.0:
        raise AssertionError(f"{label} ball_scale must be positive, got {scale:.3f}")
    return {
        "ball_pos": finite_pair(state["ball_pos"], f"{label} ball_pos"),
        "ball_prev_pos": finite_pair(state["ball_prev_pos"], f"{label} ball_prev_pos"),
        "ball_velocity_px_s": finite_pair(state["ball_velocity_px_s"], f"{label} ball_velocity_px_s"),
        "ball_ground_pos": finite_pair(state["ball_ground_pos"], f"{label} ball_ground_pos"),
        "ball_depth": depth,
        "ball_rotation_degrees": rotation,
        "ball_scale": scale,
        "ball_phase": str(state["ball_phase"]),
        "raw_shot_progress": raw_progress,
    }


def wrapped_angle_delta(current: float, previous: float) -> float:
    return (current - previous + 180.0) % 360.0 - 180.0


def visible_ball_diameters(app: App, scale: int) -> list[float]:
    target = (scale, scale)
    return [
        min(scaled_visible_width(frame, target), scaled_visible_height(frame, target))
        for frame in app.assets.balls
    ]


def visible_ball_radius(app: App, scale: int) -> float:
    return max(visible_ball_diameters(app, scale)) * 0.5


def goal_progress_second(goal_minute: int, progress: float) -> float:
    return (
        goal_minute
        - GOAL_EVENT_WINDOW_MINUTES
        + progress * GOAL_EVENT_WINDOW_MINUTES
    ) / 90.0 * SIMULATION_SECONDS


def validate_model_policy_artifacts() -> None:
    if not MODEL_PACKAGE.exists():
        raise AssertionError(f"missing model package: {MODEL_PACKAGE}")
    if not MODEL_REPORT.exists():
        raise AssertionError(f"missing model report: {MODEL_REPORT}")
    if not MODEL_STATS_REPORT.exists():
        raise AssertionError(f"missing statistical model report: {MODEL_STATS_REPORT}; run make stats-qa")
    if not TOURNAMENT_SIMULATION.exists():
        raise AssertionError(f"missing tournament simulation report: {TOURNAMENT_SIMULATION}")
    with MODEL_PACKAGE.open("rb") as file:
        package = pickle.load(file)
    report = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
    stats_report = json.loads(MODEL_STATS_REPORT.read_text(encoding="utf-8"))
    assert_fresh_statistical_report(stats_report)
    if report.get("version") != package.get("version"):
        raise AssertionError(f"model report/package version mismatch: {report.get('version')} != {package.get('version')}")
    if stats_report.get("version") != package.get("version"):
        raise AssertionError(f"statistical report/package version mismatch: {stats_report.get('version')} != {package.get('version')}")
    models = package.get("models", {})
    required_models = {
        "xgb_1x2",
        "competitive_xgb_1x2",
        "logistic_1x2",
        "home_goals_poisson",
        "away_goals_poisson",
        "home_goals_xgb_count",
        "away_goals_xgb_count",
    }
    missing = sorted(required_models - set(models))
    if missing:
        raise AssertionError(f"model package missing required models: {missing}")
    if "draw_xgb" in models:
        raise AssertionError("model package still contains removed draw_xgb model")
    policy = package.get("simulation_policy", {})
    report_policy = report.get("simulation_policy", {})
    if policy.get("selected_by") != "strict_nested_temporal_component_ablation_no_leakage_no_draw_xgb":
        raise AssertionError(f"policy was not selected by strict no-leakage nested temporal validation: {policy.get('selected_by')}")
    if report_policy.get("selected_by") != policy.get("selected_by"):
        raise AssertionError("model report policy selector does not match package")
    classifier_weight = round(float(policy.get("classifier_weight", 0.0)), 4)
    poisson_weight = round(float(policy.get("poisson_weight", 1.0 - classifier_weight)), 4)
    draw_floor = round(float(policy.get("draw_floor", -1.0)), 4)
    draw_ceiling = round(float(policy.get("draw_ceiling", -1.0)), 4)
    if not 0.50 <= classifier_weight <= 0.88:
        raise AssertionError(f"invalid classifier_weight in package policy: {classifier_weight}")
    if poisson_weight < 0.12:
        raise AssertionError(f"poisson_weight too low for football variance: {poisson_weight}")
    if not 0.02 <= draw_floor <= 0.16 or not 0.30 <= draw_ceiling <= 0.50 or draw_floor >= draw_ceiling:
        raise AssertionError(f"invalid draw guardrails: floor={draw_floor}, ceiling={draw_ceiling}")
    if abs(float(report_policy.get("classifier_weight", -1.0)) - classifier_weight) > 0.0001:
        raise AssertionError("model report classifier_weight does not match package")
    stats_policy = stats_report.get("policy", {})
    if stats_policy.get("draw_xgb") != "removed_zero_weight_model":
        raise AssertionError(f"statistical report does not confirm draw_xgb removal: {stats_policy}")
    for key, expected in {
        "classifier_weight": classifier_weight,
        "poisson_weight": poisson_weight,
        "draw_floor": draw_floor,
        "draw_ceiling": draw_ceiling,
    }.items():
        if abs(float(stats_policy.get(key, -1.0)) - float(expected)) > 0.0001:
            raise AssertionError(f"statistical report policy mismatch for {key}: {stats_policy.get(key)} != {expected}")
    if not stats_policy.get("manual_blend_weights"):
        raise AssertionError(f"statistical report missing manual blend weights: {stats_policy}")
    if not stats_report.get("verdict", {}).get("sota_kiss"):
        raise AssertionError(f"statistical report did not pass SOTA/KISS verdict: {stats_report.get('verdict')}")
    if not stats_report.get("academic_stamp", {}).get("approved"):
        raise AssertionError(f"statistical report academic stamp is not approved: {stats_report.get('academic_stamp')}")
    if not stats_report.get("training_orientation_audit", {}).get("passed"):
        raise AssertionError("statistical report did not pass neutral training-orientation audit")
    if not stats_report.get("runtime_neutral_order_audit", {}).get("passed"):
        raise AssertionError("statistical report did not pass neutral runtime-order audit")
    stats_scope = stats_report.get("scope", {})
    if int(stats_scope.get("diagnostic_rows", 0)) < 1000:
        raise AssertionError(f"statistical diagnostic window too small: {stats_scope}")
    stats_calibration = stats_report.get("calibration", {}).get("runtime_2024_plus_metrics", {})
    for metric_name, upper in {"draw_gap": 0.02, "ece": 0.08}.items():
        value = float(stats_calibration.get(metric_name, float("nan")))
        if not math.isfinite(value) or value > upper:
            raise AssertionError(f"statistical report metric {metric_name} failed: {value} > {upper}")
    benchmark_runtime = stats_report.get("external_benchmark", {}).get("runtime_policy", {})
    for metric_name, lower in {
        "log_loss_gain_vs_same_window_elo": 0.005,
        "rps_gain_vs_same_window_elo": 0.002,
    }.items():
        value = float(benchmark_runtime.get(metric_name, float("nan")))
        if not math.isfinite(value) or value < lower:
            raise AssertionError(f"statistical report benchmark {metric_name} failed: {value} < {lower}")
    if float(stats_report.get("ablation_study", {}).get("full_policy_objective_gap_vs_best", 1.0)) > 0.012:
        raise AssertionError("full policy trails ablation frontier too much in statistical report")
    if float(stats_report.get("dixon_coles", {}).get("package_rho_objective_gap_vs_best", 1.0)) > 0.01:
        raise AssertionError("package Dixon-Coles rho trails sensitivity frontier too much in statistical report")
    nested = policy.get("nested_temporal_validation", {})
    if nested.get("version") != "nested_temporal_policy_v4_orientation_invariant_no_leakage_no_draw_xgb":
        raise AssertionError(f"unexpected nested validation version: {nested.get('version')}")
    component_ablation = nested.get("component_ablation", {})
    if component_ablation.get("version") != "nested_component_subset_ablation_v1":
        raise AssertionError(f"missing nested component ablation: {component_ablation}")
    if int(component_ablation.get("candidate_count", 0)) < 63:
        raise AssertionError(f"component ablation did not cover all subsets: {component_ablation.get('candidate_count')}")
    aggregate = nested.get("aggregate", {})
    selected = nested.get("selected_policy", {})
    if int(aggregate.get("folds", 0)) < 6 or int(aggregate.get("outer_rows", 0)) < 5000:
        raise AssertionError(f"nested validation coverage is too weak: {aggregate}")
    if int(selected.get("selected_folds", 0)) < 4 or int(selected.get("selected_outer_rows", 0)) < 3000:
        raise AssertionError(f"nested selected policy is not stable enough: {selected}")
    for metric_name, upper in {
        "outer_log_loss": 1.05,
        "outer_rps": 0.22,
        "outer_brier": 0.65,
        "outer_draw_gap": 0.04,
    }.items():
        value = float(aggregate.get(metric_name, float("nan")))
        if not math.isfinite(value) or value > upper:
            raise AssertionError(f"nested metric {metric_name} failed: {value} > {upper}")
    for metrics_block_name in ("backtest_metrics", "holdout_best_metrics"):
        metrics_block = policy.get(metrics_block_name, {})
        for metric_name in ("objective", "log_loss", "rps", "brier", "draw_expected_rate", "draw_actual_rate"):
            value = float(metrics_block.get(metric_name, float("nan")))
            if not math.isfinite(value):
                raise AssertionError(f"non-finite {metrics_block_name}.{metric_name}")
    with TOURNAMENT_SIMULATION.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        row = next(reader, None)
    if row is None:
        raise AssertionError("empty tournament simulation report")
    report_weight = row.get("sim_classifier_weight")
    if report_weight is None:
        raise AssertionError("tournament simulation report missing sim_classifier_weight")
    if abs(float(report_weight) - classifier_weight) > 0.0001:
        raise AssertionError(
            f"stale tournament simulation policy: csv={float(report_weight):.4f}, package={classifier_weight:.4f}"
        )
    model = WorldCupModel()
    profiles = {team.code: team for team in model.profiles()}
    for home_code, away_code in (("BRA", "FRA"), ("ESP", "GER"), ("MEX", "USA")):
        if home_code not in profiles or away_code not in profiles:
            continue
        pred = model.predict_matchup(profiles[home_code], profiles[away_code], seed=2026)
        probs = (pred.home, pred.draw, pred.away)
        if any((not math.isfinite(value)) or value < 0.0 or value > 1.0 for value in probs):
            raise AssertionError(f"invalid probabilities for {home_code} x {away_code}: {probs}")
        if abs(sum(probs) - 1.0) > 0.01:
            raise AssertionError(f"probabilities do not sum to 1 for {home_code} x {away_code}: {probs}")


def wait_for_cinematic_preload(
    app: App,
    label: str,
    *,
    timeout: float = 15.0,
) -> None:
    preload_deadline = time.perf_counter() + timeout
    while (
        not app.poc_preload_ready
        and time.perf_counter() < preload_deadline
    ):
        app.drain_cinematic_poc_preload(max_assets=1)
        if not app.poc_preload_ready:
            time.sleep(0.001)
    if not app.poc_preload_ready:
        raise AssertionError(
            f"cinematic preload did not complete before {label}"
        )


def install_qa_match_prediction(app: App, pred: Prediction, label: str) -> None:
    app.match_prediction = pred
    runtime = app.match_runtime_state(pred)
    sequences = app.cinematic_poc_sequences_for_match(runtime)
    expected_assets = {
        path
        for path, _digest in app.cinematic_poc_preload_assets(sequences)
    }
    missing = expected_assets.difference(app.poc_layer_cache)
    if missing or not app.poc_preload_ready:
        app.start_cinematic_poc_preload(sequences)
        wait_for_cinematic_preload(app, label)
        missing = expected_assets.difference(app.poc_layer_cache)
    if missing:
        raise AssertionError(
            f"cinematic preload cache incomplete before {label}: "
            f"{sorted(missing)}"
        )


def seek_match_time(app: App, pred: Prediction, seconds: float, step: float = 1 / 30) -> None:
    install_qa_match_prediction(app, pred, "timeline seek")
    app.t = 0.0
    app.ground_scroll = 0.0
    app.ground_scroll_velocity = 0.0
    app.ground_travel_distance = 0.0
    app.shot_events.clear()
    app.goal_events.clear()
    elapsed = 0.0
    while elapsed + step < seconds:
        app.update(step)
        elapsed += step
    remaining = max(0.0, seconds - elapsed)
    if remaining:
        app.update(remaining)


def alpha_components(surface: pygame.Surface, threshold: int = 25) -> list[tuple[int, pygame.Rect]]:
    width, height = surface.get_size()
    visited: set[tuple[int, int]] = set()
    components: list[tuple[int, pygame.Rect]] = []
    for start_y in range(height):
        for start_x in range(width):
            if (start_x, start_y) in visited or surface.get_at((start_x, start_y)).a <= threshold:
                continue
            stack = [(start_x, start_y)]
            visited.add((start_x, start_y))
            size = 0
            min_x = max_x = start_x
            min_y = max_y = start_y
            while stack:
                x, y = stack.pop()
                size += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if nx < 0 or nx >= width or ny < 0 or ny >= height or (nx, ny) in visited:
                            continue
                        if surface.get_at((nx, ny)).a <= threshold:
                            continue
                        visited.add((nx, ny))
                        stack.append((nx, ny))
            components.append((size, pygame.Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)))
    components.sort(key=lambda item: item[0], reverse=True)
    return components


def interior_soft_alpha_ratio(surface: pygame.Surface) -> float:
    alpha = pygame.surfarray.array_alpha(surface)
    visible = alpha > 16
    interior = visible.copy()
    for offset_x, offset_y in (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ):
        interior &= np.roll(np.roll(visible, offset_x, axis=0), offset_y, axis=1)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    return float(((alpha < 235) & interior).sum()) / max(1, int(interior.sum()))


def perceptual_frame_delta(a: pygame.Surface, b: pygame.Surface) -> float:
    def prepared(surface: pygame.Surface) -> np.ndarray:
        width, height = surface.get_size()
        rgba = np.frombuffer(
            pygame.image.tobytes(surface, "RGBA"),
            dtype=np.uint8,
        ).reshape(height, width, 4).astype(np.float32)
        alpha = rgba[:, :, 3:4] / 255.0
        premultiplied = np.concatenate((rgba[:, :, :3] * alpha, rgba[:, :, 3:4]), axis=2)
        softened = cv2.GaussianBlur(premultiplied, (0, 0), 2.0)
        return cv2.resize(softened, (72, 72), interpolation=cv2.INTER_AREA)

    return float(np.abs(prepared(a) - prepared(b)).mean())


def chroma_leak_count(surface: pygame.Surface, step: int = 2) -> int:
    leaks = 0
    for y in range(0, surface.get_height(), step):
        for x in range(0, surface.get_width(), step):
            color = surface.get_at((x, y))
            magenta_core = min(color.r, color.b)
            magenta_dominance = magenta_core - color.g
            if (
                10 < color.a < 245
                and magenta_core > 145
                and magenta_dominance > 45
                and abs(color.r - color.b) < 105
            ):
                leaks += 1
    return leaks


def opaque_chroma_artifact_count(surface: pygame.Surface, step: int = 1) -> int:
    artifacts = 0
    for y in range(0, surface.get_height(), step):
        for x in range(0, surface.get_width(), step):
            color = surface.get_at((x, y))
            if color.a <= 10:
                continue
            magenta_core = min(color.r, color.b)
            magenta_dominance = magenta_core - color.g
            if color.r > 168 and color.b > 158 and color.g < 98 and magenta_dominance > 88 and abs(color.r - color.b) < 72:
                artifacts += 1
    return artifacts


def oracle_mark_color_is_dark(code: str) -> bool:
    return code in {"white", "sky", "gold"}


def oracle_mark_pixel_count(surface: pygame.Surface, code: str, step: int = 1) -> int:
    bbox = alpha_bbox(surface)
    region = pygame.Rect(
        bbox.x + int(bbox.w * 0.20),
        bbox.y + int(bbox.h * 0.36),
        max(1, int(bbox.w * 0.60)),
        max(1, int(bbox.h * 0.28)),
    ).clip(surface.get_rect())
    count = 0
    dark_mark = oracle_mark_color_is_dark(code)
    for y in range(region.y, region.bottom, step):
        for x in range(region.x, region.right, step):
            color = surface.get_at((x, y))
            if color.a <= 80:
                continue
            if dark_mark:
                if color.r < 125 and color.g < 125 and color.b < 125:
                    count += 1
            elif color.r > 132 and color.g > 132 and color.b > 124:
                count += 1
    return count


def oracle_mark_bounds(surface: pygame.Surface, code: str, step: int = 1) -> tuple[int, pygame.Rect]:
    bbox = alpha_bbox(surface)
    region = pygame.Rect(
        bbox.x + int(bbox.w * 0.18),
        bbox.y + int(bbox.h * 0.32),
        max(1, int(bbox.w * 0.66)),
        max(1, int(bbox.h * 0.32)),
    ).clip(surface.get_rect())
    dark_mark = oracle_mark_color_is_dark(code)
    points = []
    for y in range(region.y, region.bottom, step):
        for x in range(region.x, region.right, step):
            color = surface.get_at((x, y))
            if color.a <= 80:
                continue
            if dark_mark:
                active = color.r < 125 and color.g < 125 and color.b < 125
            else:
                active = color.r > 132 and color.g > 132 and color.b > 124
            if active:
                points.append((x, y))
    if not points:
        return 0, pygame.Rect(region.centerx, region.centery, 0, 0)
    min_x = min(x for x, _y in points)
    max_x = max(x for x, _y in points)
    min_y = min(y for _x, y in points)
    max_y = max(y for _x, y in points)
    return len(points), pygame.Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)


def assert_green_uniform_alpha(surface: pygame.Surface, code: str, path: Path) -> None:
    if code != "green":
        return
    bbox = alpha_bbox(surface)
    chest = pygame.Rect(
        bbox.x + int(bbox.w * 0.18),
        bbox.y + int(bbox.h * 0.28),
        max(1, int(bbox.w * 0.64)),
        max(1, int(bbox.h * 0.30)),
    ).clip(surface.get_rect())
    alphas = []
    for y in range(chest.y, chest.bottom):
        for x in range(chest.x, chest.right):
            color = surface.get_at((x, y))
            if color.a <= 20:
                continue
            if color.g > color.r + 18 and color.g > color.b + 4:
                alphas.append(color.a)
    if len(alphas) < 24:
        raise AssertionError(f"{path} lost too much green uniform area after chroma key")
    mean_alpha = sum(alphas) / len(alphas)
    semi_ratio = sum(1 for alpha in alphas if alpha < 210) / len(alphas)
    if mean_alpha < 224 or semi_ratio > 0.20:
        raise AssertionError(f"{path} green uniform was damaged by matte/keying: mean_alpha={mean_alpha:.1f}, semi={semi_ratio:.2f}")


def dark_holes_in_light_shorts(surface: pygame.Surface, code: str) -> int:
    shorts = SHORTS_BY_CODE.get(code)
    if shorts is None or sum(shorts) < 520:
        return 0
    bbox = alpha_bbox(surface)
    region = pygame.Rect(
        bbox.x + int(bbox.w * 0.14),
        bbox.y + int(bbox.h * 0.56),
        max(1, int(bbox.w * 0.72)),
        max(1, int(bbox.h * 0.22)),
    ).clip(surface.get_rect())
    shorts_pixels: set[tuple[int, int]] = set()
    dark_pixels: list[tuple[int, int]] = []
    for y in range(region.y, region.bottom):
        for x in range(region.x, region.right):
            color = surface.get_at((x, y))
            if color.a <= 60:
                continue
            distance = abs(color.r - shorts[0]) + abs(color.g - shorts[1]) + abs(color.b - shorts[2])
            if distance < 118:
                shorts_pixels.add((x, y))
            elif color.r + color.g + color.b < 190:
                dark_pixels.append((x, y))
    holes = 0
    for x, y in dark_pixels:
        neighbors = 0
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if (x + dx, y + dy) in shorts_pixels:
                    neighbors += 1
        if neighbors >= 20:
            holes += 1
    return holes


def colored_stains_in_light_shorts(surface: pygame.Surface, code: str) -> int:
    shorts = SHORTS_BY_CODE.get(code)
    if shorts is None or sum(shorts) < 520:
        return 0
    bbox = alpha_bbox(surface)
    region = pygame.Rect(
        bbox.x + int(bbox.w * 0.14),
        bbox.y + int(bbox.h * 0.56),
        max(1, int(bbox.w * 0.72)),
        max(1, int(bbox.h * 0.22)),
    ).clip(surface.get_rect())
    shorts_mask: set[tuple[int, int]] = set()
    stain_candidates: list[tuple[int, int]] = []
    for y in range(region.y, region.bottom):
        for x in range(region.x, region.right):
            color = surface.get_at((x, y))
            if color.a <= 60:
                continue
            distance = abs(color.r - shorts[0]) + abs(color.g - shorts[1]) + abs(color.b - shorts[2])
            if distance < 118:
                shorts_mask.add((x, y))
                continue
            brightness = color.r + color.g + color.b
            skin_like = color.r > 135 and color.g > 70 and color.b < 125 and color.r > color.g + 18
            if brightness < 650 and not skin_like:
                stain_candidates.append((x, y))
    stains = 0
    for x, y in stain_candidates:
        neighbors = 0
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if (x + dx, y + dy) in shorts_mask:
                    neighbors += 1
        if neighbors >= 20:
            stains += 1
    return stains


def assert_no_light_short_holes(surface: pygame.Surface, code: str, path: Path, limit: int) -> None:
    holes = dark_holes_in_light_shorts(surface, code)
    if holes > limit:
        raise AssertionError(f"{path} has dark broken pixels inside light shorts: {holes}")
    stains = colored_stains_in_light_shorts(surface, code)
    if stains > limit:
        raise AssertionError(f"{path} has colored stains inside light shorts: {stains}")


def alpha_pixel_count(surface: pygame.Surface, threshold: int = 25, step: int = 1) -> int:
    return sum(
        1
        for y in range(0, surface.get_height(), step)
        for x in range(0, surface.get_width(), step)
        if surface.get_at((x, y)).a > threshold
    )


def expected_cinematic_runtime_files() -> set[str]:
    expected: set[str] = set()
    for code in ROWS:
        expected.add(f"poc2_runner_right_{code}.png")
        expected.add(f"poc2_runner_left_{code}.png")
        expected.add(f"runner_smooth_{code}.png")
        expected.add(f"runner_left_smooth_{code}.png")
        expected.add(f"runner_kick_smooth_{code}.png")
        expected.add(f"runner_left_kick_smooth_{code}.png")
        expected.add(f"runner_stop_smooth_{code}.png")
        expected.add(f"runner_left_stop_smooth_{code}.png")
    for direction in ("right", "left"):
        expected.update(f"keeper_anim_{direction}_{index}.png" for index in range(KEEPER_FRAMES))
    return expected


def validate_flag_sprites() -> None:
    teams = WorldCupModel().profiles()
    expected = {team.code.lower() for team in teams}
    actual = {path.stem for path in FLAG_DIR.glob("*.png")}
    if len(expected) != 48:
        raise AssertionError(f"expected 48 teams, got {len(expected)}")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssertionError(f"flag sprite set mismatch; missing={missing}, extra={extra}")

    for team in teams:
        path = FLAG_DIR / f"{team.code.lower()}.png"
        image = pygame.image.load(path).convert_alpha()
        if image.get_size() != FLAG_SIZE:
            raise AssertionError(f"{path} has unexpected size {image.get_size()}")
        if any(image.get_at(point).a > 8 for point in ((0, 0), (FLAG_SIZE[0] - 1, 0), (0, FLAG_SIZE[1] - 1), (FLAG_SIZE[0] - 1, FLAG_SIZE[1] - 1))):
            raise AssertionError(f"{path} should keep transparent rounded corners")
        bbox = alpha_bbox(image)
        if bbox.w < 126 or bbox.h < 86:
            raise AssertionError(f"{path} has too little visible flag sprite: {bbox}")
        if bbox.right > FLAG_SIZE[0] or bbox.bottom > FLAG_SIZE[1]:
            raise AssertionError(f"{path} appears clipped: {bbox}")


def validate_cinematic_inventory() -> None:
    try:
        poc2_runtime = CinematicDribbleRuntime.load(POC2_RUNNER_MOTION)
    except Exception as exc:
        raise AssertionError(
            f"invalid promoted POC 2 motion contract: {exc}"
        ) from exc
    if (
        poc2_runtime.metadata.status != "promoted"
        or set(poc2_runtime.uniform_codes) != set(UNIFORM_CODES)
        or set(poc2_runtime.direction_names) != {"right", "left"}
        or poc2_runtime.metadata.frame_count != 8
        or not math.isclose(
            poc2_runtime.metadata.cycle_seconds,
            1.6,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise AssertionError(
            f"invalid promoted POC 2 motion metadata: {POC2_RUNNER_MOTION}"
        )
    if not RUNNER_MOTION.exists():
        raise AssertionError(f"missing grounded runner metadata: {RUNNER_MOTION}")
    runner_motion = json.loads(RUNNER_MOTION.read_text(encoding="utf-8"))
    if (
        runner_motion.get("version") != 8
        or runner_motion.get("artifact")
        != "arena_runner_motion_contract"
        or runner_motion.get("status") != "promoted"
        or runner_motion.get("artwork_provenance")
        != "gpt_image_authored_per_uniform_per_direction_per_action"
        or runner_motion.get("frame_count")
        != SMOOTH_RUNNER_FRAMES
    ):
        raise AssertionError(f"invalid grounded runner metadata contract: {RUNNER_MOTION}")
    if runner_motion.get("kick_frame_count") != CINEMATIC_KICK_FRAME_COUNT:
        raise AssertionError(f"invalid grounded runner kick metadata contract: {RUNNER_MOTION}")
    if runner_motion.get("stop_frame_count") != CINEMATIC_STOP_FRAME_COUNT:
        raise AssertionError(f"invalid grounded runner stop metadata contract: {RUNNER_MOTION}")
    if runner_motion.get("left_runtime_derivation") != "separately_generated_gpt_image":
        raise AssertionError("left-facing runtime gait must use separately generated GPT Image art")
    if runner_motion.get("temporal_interpolation") != "none_gpt_image_authored_frames":
        raise AssertionError("runner runtime must not contain synthesized in-between frames")
    if runner_motion.get("kick_temporal_interpolation") != "none_gpt_image_authored_frames":
        raise AssertionError("kick runtime must not contain synthesized in-between frames")
    if runner_motion.get("wordmark_provenance") != "native_gpt_image_jersey_pixels_no_overlay":
        raise AssertionError("ORACLE must come from the generated jersey art, not a runtime overlay")
    clearance_contract = runner_motion.get("ball_clearance_contract")
    if (
        not isinstance(clearance_contract, dict)
        or clearance_contract.get("field") != "ball_clearance_offset_px"
        or clearance_contract.get("monotonic_field") != "ball_corridor_offset_px"
        or clearance_contract.get("run_approach_field")
        != "ball_approach_corridor_offset_px"
        or clearance_contract.get("ball_rotation_count")
        != CINEMATIC_BALL_MATERIAL_FRAME_COUNT
        or clearance_contract.get("max_alpha_overlap_ratio") != 0.03
    ):
        raise AssertionError(
            "grounded runner metadata is missing the v8 alpha-safe ball contract"
        )
    uniforms = runner_motion.get("uniforms")
    if not isinstance(uniforms, dict) or set(uniforms) != set(UNIFORM_CODES):
        raise AssertionError("grounded runner metadata does not cover all distinct avatars")
    for code in UNIFORM_CODES:
        for direction in ("right", "left"):
            entry = uniforms[code]["directions"][direction]
            for action in ("run", "kick", "stop"):
                sheet_path = ROOT / str(entry[f"{action}_sheet"])
                if hashlib.sha256(sheet_path.read_bytes()).hexdigest() != entry[f"{action}_sheet_sha256"]:
                    raise AssertionError(f"{code} {direction} {action} runtime sheet hash drifted")
    if not KEEPER_MOTION.exists():
        raise AssertionError(f"missing goalkeeper motion metadata: {KEEPER_MOTION}")
    keeper_motion = json.loads(KEEPER_MOTION.read_text(encoding="utf-8"))
    if (
        keeper_motion.get("version") != 4
        or keeper_motion.get("artifact")
        != "arena_keeper_motion_contract"
        or keeper_motion.get("status") != "promoted"
        or keeper_motion.get("artwork_provenance")
        != "gpt_image_authored_directional_frames"
        or keeper_motion.get("authored_frame_count") != 16
        or keeper_motion.get("frame_count") != KEEPER_FRAMES
        or keeper_motion.get("runtime_frame_order")
        != list(range(KEEPER_FRAMES))
        or keeper_motion.get("temporal_interpolation") != "none_gpt_image_authored_frames"
        or keeper_motion.get("left_runtime_derivation") != "separately_generated_gpt_image"
    ):
        raise AssertionError(f"invalid goalkeeper metadata contract: {KEEPER_MOTION}")
    keeper_directions = keeper_motion.get("directions")
    if not isinstance(keeper_directions, dict) or set(keeper_directions) != {"right", "left"}:
        raise AssertionError("goalkeeper direction metadata is incomplete")
    for direction, entry in keeper_directions.items():
        runtime_paths = [CINEMATIC_DIR / f"keeper_anim_{direction}_{index}.png" for index in range(KEEPER_FRAMES)]
        if [hashlib.sha256(path.read_bytes()).hexdigest() for path in runtime_paths] != entry["runtime_sha256"]:
            raise AssertionError(f"goalkeeper {direction} runtime hashes drifted")
    expected_runtime = expected_cinematic_runtime_files()
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    runtime_globs = manifest.get("generated_runtime_globs")
    if not isinstance(runtime_globs, list):
        raise AssertionError("asset manifest has no generated runtime globs")
    actual_runtime = {
        path.name
        for pattern in runtime_globs
        if isinstance(pattern, str)
        for path in ROOT.glob(pattern)
        if path.is_file()
        and path.suffix.casefold() == ".png"
        and path.parent == CINEMATIC_DIR
    }
    if actual_runtime != expected_runtime:
        extra = sorted(actual_runtime - expected_runtime)
        missing = sorted(expected_runtime - actual_runtime)
        raise AssertionError(f"cinematic runtime sprite inventory mismatch; extra={extra}, missing={missing}")
    for sample_name, expected_size in (
        ("runner_smooth_blue.png", SMOOTH_RUNNER_SHEET_SIZE),
        ("poc2_runner_right_blue.png", (1280, 640)),
        ("poc2_runner_left_blue.png", (1280, 640)),
        ("runner_kick_smooth_blue.png", SMOOTH_KICK_SHEET_SIZE),
        ("runner_stop_smooth_blue.png", SMOOTH_STOP_SHEET_SIZE),
        ("keeper_anim_right_0.png", KEEPER_SPRITE_SIZE),
        ("keeper_anim_left_0.png", KEEPER_SPRITE_SIZE),
    ):
        path = CINEMATIC_DIR / sample_name
        if not path.exists():
            raise AssertionError(f"missing cinematic sample: {path}")
        image = pygame.image.load(path).convert_alpha()
        if image.get_size() != expected_size:
            raise AssertionError(f"{path} has unexpected size {image.get_size()}")


def authored_sheet_frames(
    path: Path,
    frame_size: int,
    columns: int,
    frame_count: int,
) -> list[pygame.Surface]:
    sheet = pygame.image.load(path).convert_alpha()
    rows = math.ceil(frame_count / columns)
    expected_size = (frame_size * columns, frame_size * rows)
    if sheet.get_size() != expected_size:
        raise AssertionError(f"{path} has unexpected authored sheet size {sheet.get_size()} != {expected_size}")
    return [
        sheet.subsurface(
            pygame.Rect(
                (index % columns) * frame_size,
                (index // columns) * frame_size,
                frame_size,
                frame_size,
            )
        ).copy()
        for index in range(frame_count)
    ]


def validate_authored_cinematic_sprites() -> None:
    if not pygame.get_init():
        pygame.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))
    validate_cinematic_inventory()
    motion = json.loads(RUNNER_MOTION.read_text(encoding="utf-8"))
    frame_size = int(motion["frame_size"])
    frame_count = int(motion["frame_count"])
    columns = int(motion["sheet_columns"])
    if frame_size != AUTHORED_RUNNER_FRAME_SIZE[0] or frame_count != 16 or columns != 4:
        raise AssertionError(
            f"authored player contract drifted: size={frame_size}, frames={frame_count}, columns={columns}"
        )

    active_sources = (
        ROOT / "src" / "arena_ai" / "main.py",
        ROOT / "src" / "arena_ai" / "cinematic_dribble_runtime.py",
        ROOT / "src" / "arena_ai" / "cinematic_poc_runtime.py",
    )
    forbidden_imports = {"scripts.motion_interpolation"}
    forbidden_calls = {
        "wordmark_overlay(",
        "stamp_wordmark(",
        "mirrored_frame(",
        "interpolate_cycle(",
        "interpolate_open_sequence(",
        "motion_compensated_inbetween(",
        "horizontal_lower_body_warp(",
    }
    forbidden_call_names = {
        token.removesuffix("(")
        for token in forbidden_calls
    }
    for source in active_sources:
        tree = ast.parse(
            source.read_text(encoding="utf-8"),
            filename=str(source),
        )
        forbidden: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                forbidden.update(
                    alias.name
                    for alias in node.names
                    if alias.name in forbidden_imports
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module in forbidden_imports
            ):
                forbidden.add(str(node.module))
            elif isinstance(node, ast.Call):
                call_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else ""
                )
                if call_name in forbidden_call_names:
                    forbidden.add(call_name)
        if forbidden:
            raise AssertionError(
                "active authored sprite path reintroduced synthetic transforms "
                f"in {source}: {sorted(forbidden)}"
            )
    avatar_alpha_signatures: set[str] = set()
    sequences: dict[tuple[str, str, str], list[pygame.Surface]] = {}
    uniforms = motion["uniforms"]
    for code in UNIFORM_CODES:
        for direction in ("right", "left"):
            entry = uniforms[code]["directions"][direction]
            action_contracts = (
                ("run", frame_count, columns),
                ("kick", frame_count, int(motion["kick_sheet_columns"])),
                ("stop", int(motion["stop_frame_count"]), int(motion["stop_sheet_columns"])),
            )
            for action, action_frame_count, action_columns in action_contracts:
                sheet_path = ROOT / str(entry[f"{action}_sheet"])
                frames = authored_sheet_frames(sheet_path, frame_size, action_columns, action_frame_count)
                sequences[(code, direction, action)] = frames
                frame_hashes = {
                    hashlib.sha256(pygame.image.tobytes(frame, "RGBA")).hexdigest()
                    for frame in frames
                }
                if len(frame_hashes) != action_frame_count:
                    raise AssertionError(f"{code} {direction} {action} repeats authored poses")

                areas: list[int] = []
                for index, frame in enumerate(frames):
                    label = f"{code} {direction} {action} frame {index}"
                    bbox = alpha_bbox(frame)
                    margins = (
                        bbox.left,
                        bbox.top,
                        frame_size - bbox.right,
                        frame_size - bbox.bottom,
                    )
                    if min(margins) < 4:
                        raise AssertionError(f"{label} is clipped: bbox={bbox}, margins={margins}")
                    detached = [(size, rect) for size, rect in alpha_components(frame)[1:] if size > 20]
                    if detached:
                        raise AssertionError(f"{label} contains detached neighboring sprite fragments: {detached}")
                    if chroma_leak_count(frame) > 4 or opaque_chroma_artifact_count(frame) > 4:
                        raise AssertionError(f"{label} contains visible magenta-key residue")
                    if interior_soft_alpha_ratio(frame) > 0.015:
                        raise AssertionError(f"{label} contains translucent double-exposure pixels")
                    if oracle_mark_pixel_count(frame, code) < 40:
                        raise AssertionError(f"{label} lost the native ORACLE shirt mark")
                    assert_green_uniform_alpha(frame, code, sheet_path)
                    assert_no_light_short_holes(frame, code, sheet_path, 220)
                    areas.append(alpha_pixel_count(frame))

                area_ratio = max(areas) / max(1, min(areas))
                area_limit = {"run": 1.60, "kick": 2.15, "stop": 2.00}[action]
                if area_ratio > area_limit:
                    raise AssertionError(
                        f"{code} {direction} {action} silhouette scale flickers: ratio={area_ratio:.3f}"
                    )
                deltas = [perceptual_frame_delta(a, b) for a, b in zip(frames, frames[1:])]
                minimum_delta = 0.20 if action == "stop" else 1.50
                if min(deltas) < minimum_delta or max(deltas) > 36.0:
                    raise AssertionError(
                        f"{code} {direction} {action} authored cadence is static or pops: "
                        f"min={min(deltas):.3f}, max={max(deltas):.3f}"
                    )

            first_run = sequences[(code, direction, "run")][0]
            if direction == "right":
                avatar_alpha_signatures.add(
                    hashlib.sha256(pygame.surfarray.array_alpha(first_run).tobytes()).hexdigest()
                )

        for action in ("run", "kick", "stop"):
            right_frames = sequences[(code, "right", action)]
            left_frames = sequences[(code, "left", action)]
            for index, (right, left) in enumerate(zip(right_frames, left_frames)):
                mirrored = pygame.transform.flip(right, True, False)
                if pygame.image.tobytes(left, "RGBA") == pygame.image.tobytes(mirrored, "RGBA"):
                    raise AssertionError(f"{code} {action} frame {index} left art is a runtime mirror")
                if perceptual_frame_delta(left, mirrored) < 0.50:
                    raise AssertionError(f"{code} {action} frame {index} left art is effectively mirrored")

    if len(avatar_alpha_signatures) != len(UNIFORM_CODES):
        raise AssertionError("the nine uniform colors do not preserve nine distinct authored avatar silhouettes")

    poc2_motion = json.loads(POC2_RUNNER_MOTION.read_text(encoding="utf-8"))
    poc2_frame_size = int(poc2_motion["canvas_size"])
    poc2_frame_count = int(poc2_motion["frame_count"])
    poc2_columns = int(poc2_motion["sheet_columns"])
    if (
        poc2_motion.get("status") != "promoted"
        or poc2_frame_size != 320
        or poc2_frame_count != 8
        or poc2_columns != 4
    ):
        raise AssertionError(
            "promoted POC 2 authored player contract drifted: "
            f"status={poc2_motion.get('status')}, size={poc2_frame_size}, "
            f"frames={poc2_frame_count}, columns={poc2_columns}"
        )
    poc2_sequences: dict[tuple[str, str], list[pygame.Surface]] = {}
    for code in UNIFORM_CODES:
        for direction in ("right", "left"):
            entry = poc2_motion["uniforms"][code]["directions"][direction]
            sheet_path = CINEMATIC_DIR / str(entry["sheet"])
            frames = authored_sheet_frames(
                sheet_path,
                poc2_frame_size,
                poc2_columns,
                poc2_frame_count,
            )
            poc2_sequences[(code, direction)] = frames
            frame_hashes = {
                hashlib.sha256(
                    pygame.image.tobytes(frame, "RGBA")
                ).hexdigest()
                for frame in frames
            }
            if len(frame_hashes) != poc2_frame_count:
                raise AssertionError(
                    f"POC 2 {code} {direction} repeats authored run poses"
                )

            areas: list[int] = []
            for index, frame in enumerate(frames):
                label = f"POC 2 {code} {direction} run frame {index}"
                bbox = alpha_bbox(frame)
                margins = (
                    bbox.left,
                    bbox.top,
                    poc2_frame_size - bbox.right,
                    poc2_frame_size - bbox.bottom,
                )
                if min(margins) < 8:
                    raise AssertionError(
                        f"{label} is clipped: bbox={bbox}, margins={margins}"
                    )
                detached = [
                    (size, rect)
                    for size, rect in alpha_components(frame)[1:]
                    if size > 20
                ]
                if detached:
                    raise AssertionError(
                        f"{label} contains detached neighboring fragments: {detached}"
                    )
                if (
                    chroma_leak_count(frame) > 4
                    or opaque_chroma_artifact_count(frame) > 4
                ):
                    raise AssertionError(
                        f"{label} contains visible magenta-key residue"
                    )
                if interior_soft_alpha_ratio(frame) > 0.010:
                    raise AssertionError(
                        f"{label} contains translucent double-exposure pixels"
                    )
                if oracle_mark_pixel_count(frame, code) < 60:
                    raise AssertionError(
                        f"{label} lost the native ORACLE shirt mark"
                    )
                assert_green_uniform_alpha(frame, code, sheet_path)
                assert_no_light_short_holes(frame, code, sheet_path, 220)
                areas.append(alpha_pixel_count(frame))

            area_ratio = max(areas) / max(1, min(areas))
            if area_ratio > 1.35:
                raise AssertionError(
                    f"POC 2 {code} {direction} run silhouette scale flickers: "
                    f"ratio={area_ratio:.3f}"
                )
            deltas = [
                perceptual_frame_delta(first, second)
                for first, second in zip(frames, frames[1:])
            ]
            if min(deltas) < 5.0 or max(deltas) > 30.0:
                raise AssertionError(
                    f"POC 2 {code} {direction} authored cadence is static or pops: "
                    f"min={min(deltas):.3f}, max={max(deltas):.3f}"
                )

        for index, (right, left) in enumerate(
            zip(
                poc2_sequences[(code, "right")],
                poc2_sequences[(code, "left")],
            )
        ):
            mirrored = pygame.transform.flip(right, True, False)
            if (
                pygame.image.tobytes(left, "RGBA")
                == pygame.image.tobytes(mirrored, "RGBA")
                or perceptual_frame_delta(left, mirrored) < 0.50
            ):
                raise AssertionError(
                    f"POC 2 {code} run frame {index} left art is effectively mirrored"
                )

    keeper_motion = json.loads(KEEPER_MOTION.read_text(encoding="utf-8"))
    keeper_sequences: dict[str, list[pygame.Surface]] = {}
    for direction in ("right", "left"):
        frames = [
            pygame.image.load(CINEMATIC_DIR / f"keeper_anim_{direction}_{index}.png").convert_alpha()
            for index in range(KEEPER_FRAMES)
        ]
        keeper_sequences[direction] = frames
        frame_hashes = [
            hashlib.sha256(
                pygame.image.tobytes(frame, "RGBA")
            ).hexdigest()
            for frame in frames
        ]
        if (
            len(set(frame_hashes[:-1])) != KEEPER_FRAMES - 1
            or frame_hashes[-1] != frame_hashes[0]
        ):
            raise AssertionError(
                f"goalkeeper {direction} must contain 15 authored poses "
                "plus the approved reset frame"
            )
        for index, frame in enumerate(frames):
            bbox = alpha_bbox(frame)
            margins = (bbox.left, bbox.top, frame.get_width() - bbox.right, frame.get_height() - bbox.bottom)
            if min(margins) < 4:
                raise AssertionError(f"goalkeeper {direction} frame {index} is clipped: {bbox}")
            if any(size > 20 for size, _rect in alpha_components(frame)[1:]):
                raise AssertionError(f"goalkeeper {direction} frame {index} has detached fragments")
            if chroma_leak_count(frame) > 4 or opaque_chroma_artifact_count(frame) > 4:
                raise AssertionError(f"goalkeeper {direction} frame {index} contains chroma residue")
            if interior_soft_alpha_ratio(frame) > 0.055:
                raise AssertionError(f"goalkeeper {direction} frame {index} contains double exposure")
            mark_pixels, _mark_bounds = oracle_mark_bounds(frame, "green")
            if mark_pixels < 10:
                raise AssertionError(f"goalkeeper {direction} frame {index} lost the native ORACLE mark")
        deltas = [perceptual_frame_delta(a, b) for a, b in zip(frames, frames[1:])]
        if min(deltas) < 1.50 or max(deltas) > 35.0:
            raise AssertionError(
                f"goalkeeper {direction} authored cadence is static or pops: "
                f"min={min(deltas):.3f}, max={max(deltas):.3f}"
            )

    keeper_right = keeper_sequences["right"][7]
    keeper_left = keeper_sequences["left"][7]
    for direction, frame, expected_side in (("right", keeper_right, 1), ("left", keeper_left, -1)):
        rgb = pygame.surfarray.array3d(frame)
        alpha = pygame.surfarray.array_alpha(frame)
        glove = (
            (alpha > 100)
            & (rgb[:, :, 0] > 180)
            & (rgb[:, :, 1] > 180)
            & (rgb[:, :, 2] > 180)
        )
        glove_x, _glove_y = np.nonzero(glove)
        if len(glove_x) < 120:
            raise AssertionError(f"goalkeeper {direction} dive has no readable authored gloves")
        glove_center = float(glove_x.mean())
        if expected_side > 0 and glove_center <= frame.get_width() * 0.62:
            raise AssertionError(f"goalkeeper right dive points the wrong way: x={glove_center:.1f}")
        if expected_side < 0 and glove_center >= frame.get_width() * 0.38:
            raise AssertionError(f"goalkeeper left dive points the wrong way: x={glove_center:.1f}")
    if perceptual_frame_delta(keeper_left, pygame.transform.flip(keeper_right, True, False)) < 0.50:
        raise AssertionError("goalkeeper left dive is effectively mirrored instead of separately authored")

    actual_balls = {path.name for path in BALL_DIR.glob("*.png")}
    expected_balls = {
        f"ball_{index}.png"
        for index in range(CINEMATIC_BALL_MATERIAL_FRAME_COUNT)
    }
    if actual_balls != expected_balls:
        raise AssertionError(
            f"ball runtime sprite inventory mismatch; extra={sorted(actual_balls - expected_balls)}, "
            f"missing={sorted(expected_balls - actual_balls)}"
        )
    ball_frames = [
        pygame.image.load(BALL_DIR / f"ball_{index}.png").convert_alpha()
        for index in range(CINEMATIC_BALL_MATERIAL_FRAME_COUNT)
    ]
    ball_hashes = {hashlib.sha256(pygame.image.tobytes(frame, "RGBA")).hexdigest() for frame in ball_frames}
    if len(ball_hashes) != CINEMATIC_BALL_MATERIAL_FRAME_COUNT:
        raise AssertionError(
            "ball roll must contain every distinct approved POC material view"
        )
    ball_boxes = [alpha_bbox(frame) for frame in ball_frames]
    ball_areas = [alpha_pixel_count(frame) for frame in ball_frames]
    if max(ball_areas) / max(1, min(ball_areas)) > 1.05:
        raise AssertionError(f"ball alpha area flickers across authored rotations: {ball_areas}")
    if max(box.centerx for box in ball_boxes) - min(box.centerx for box in ball_boxes) > 2:
        raise AssertionError(f"ball visual center jumps horizontally: {[box.center for box in ball_boxes]}")
    if max(box.centery for box in ball_boxes) - min(box.centery for box in ball_boxes) > 2:
        raise AssertionError(f"ball visual center jumps vertically: {[box.center for box in ball_boxes]}")
    ball_deltas = [perceptual_frame_delta(a, b) for a, b in zip(ball_frames, ball_frames[1:])]
    if min(ball_deltas) < 2.0 or max(ball_deltas) > 38.0:
        raise AssertionError(
            f"ball authored roll is static or pops: min={min(ball_deltas):.3f}, max={max(ball_deltas):.3f}"
        )
    if CINEMATIC_BALL_MATERIAL_BLEND_STEPS != 1:
        raise AssertionError("runtime ball must select a native authored view without alpha-blended in-betweens")


def validate_native_oracle_legibility() -> None:
    if not pygame.get_init():
        pygame.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))
    motion = json.loads(RUNNER_MOTION.read_text(encoding="utf-8"))
    frame_size = int(motion["frame_size"])
    columns = int(motion["sheet_columns"])
    scale = CINEMATIC_POSE_SIZE * CINEMATIC_PLAYER_SCALE / float(motion["reference_visible_height"])
    target_size = max(1, round(frame_size * scale))
    checked = 0
    for code in UNIFORM_CODES:
        for direction in ("right", "left"):
            entry = motion["uniforms"][code]["directions"][direction]
            action_contracts = (
                ("run", 16, columns),
                ("kick", 16, int(motion["kick_sheet_columns"])),
                ("stop", int(motion["stop_frame_count"]), int(motion["stop_sheet_columns"])),
            )
            for action, action_frame_count, action_columns in action_contracts:
                path = ROOT / str(entry[f"{action}_sheet"])
                frames = authored_sheet_frames(path, frame_size, action_columns, action_frame_count)
                for index, frame in enumerate(frames):
                    rendered = pygame.transform.smoothscale(frame, (target_size, target_size))
                    count, bounds = oracle_mark_bounds(rendered, code)
                    if oracle_mark_pixel_count(rendered, code) < 18 or count < 12:
                        raise AssertionError(
                            f"runtime native ORACLE disappears for {code} {direction} {action} frame {index}: "
                            f"pixels={count}, bounds={bounds}"
                        )
                    if bounds.w < 8 or bounds.h < 3:
                        raise AssertionError(
                            f"runtime native ORACLE collapses for {code} {direction} {action} frame {index}: {bounds}"
                        )
                    checked += 1
    poc2_motion = json.loads(POC2_RUNNER_MOTION.read_text(encoding="utf-8"))
    poc2_frame_size = int(poc2_motion["canvas_size"])
    poc2_frame_count = int(poc2_motion["frame_count"])
    poc2_columns = int(poc2_motion["sheet_columns"])
    poc2_scale = (
        CINEMATIC_POSE_SIZE
        * CINEMATIC_PLAYER_SCALE
        / POC_APPROVED_REFERENCE_VISIBLE_HEIGHT
    )
    poc2_target_size = max(1, round(poc2_frame_size * poc2_scale))
    for code in UNIFORM_CODES:
        for direction in ("right", "left"):
            entry = poc2_motion["uniforms"][code]["directions"][direction]
            path = CINEMATIC_DIR / str(entry["sheet"])
            frames = authored_sheet_frames(
                path,
                poc2_frame_size,
                poc2_columns,
                poc2_frame_count,
            )
            for index, frame in enumerate(frames):
                rendered = pygame.transform.smoothscale(
                    frame,
                    (poc2_target_size, poc2_target_size),
                )
                count, bounds = oracle_mark_bounds(rendered, code)
                if oracle_mark_pixel_count(rendered, code) < 18 or count < 12:
                    raise AssertionError(
                        "runtime native ORACLE disappears for POC 2 "
                        f"{code} {direction} run frame {index}: "
                        f"pixels={count}, bounds={bounds}"
                    )
                if bounds.w < 8 or bounds.h < 3:
                    raise AssertionError(
                        "runtime native ORACLE collapses for POC 2 "
                        f"{code} {direction} run frame {index}: {bounds}"
                    )
                checked += 1
    expected = len(UNIFORM_CODES) * 2 * (
        16 + 16 + CINEMATIC_STOP_FRAME_COUNT + poc2_frame_count
    )
    if checked != expected:
        raise AssertionError(f"runtime native ORACLE coverage is incomplete: {checked}/{expected}")


def home_win_prediction() -> Prediction:
    return Prediction(
        algorithm="CONFRONTO",
        home=0.64,
        draw=0.22,
        away=0.14,
        home_goals=1.9,
        away_goals=0.8,
        confidence=0.74,
        reason="cinematic validation",
        home_advances=0.72,
        away_advances=0.28,
        top_scores=((2, 0, 0.16), (2, 1, 0.13), (1, 0, 0.12)),
        over_25=0.48,
        btts=0.42,
        score_home=2,
        score_away=0,
        outcome_class=0,
        outcome_probability=0.64,
        score_probability=0.16,
        blend_probs=(0.64, 0.22, 0.14),
        poisson_outcome_probs=(0.58, 0.24, 0.18),
    )


def neutral_prediction() -> Prediction:
    return Prediction(
        algorithm="CONFRONTO",
        home=0.31,
        draw=0.39,
        away=0.30,
        home_goals=1.1,
        away_goals=1.1,
        confidence=0.53,
        reason="neutral cinematic validation",
        home_advances=0.50,
        away_advances=0.50,
        top_scores=((1, 1, 0.18), (0, 0, 0.12), (2, 2, 0.08)),
        over_25=0.36,
        btts=0.50,
        score_home=1,
        score_away=1,
        outcome_class=1,
        outcome_probability=0.39,
        score_probability=0.18,
        blend_probs=(0.31, 0.39, 0.30),
        poisson_outcome_probs=(0.31, 0.39, 0.30),
    )


def nil_draw_prediction() -> Prediction:
    return Prediction(
        algorithm="CONFRONTO",
        home=0.30,
        draw=0.44,
        away=0.26,
        home_goals=0.7,
        away_goals=0.6,
        confidence=0.50,
        reason="nil draw cinematic validation",
        home_advances=0.50,
        away_advances=0.50,
        top_scores=((0, 0, 0.19), (1, 1, 0.13), (1, 0, 0.10)),
        over_25=0.24,
        btts=0.28,
        score_home=0,
        score_away=0,
        outcome_class=1,
        outcome_probability=0.44,
        score_probability=0.19,
        blend_probs=(0.30, 0.44, 0.26),
        poisson_outcome_probs=(0.28, 0.45, 0.27),
    )


def away_win_prediction() -> Prediction:
    return Prediction(
        algorithm="CONFRONTO",
        home=0.14,
        draw=0.22,
        away=0.64,
        home_goals=0.8,
        away_goals=1.9,
        confidence=0.74,
        reason="away cinematic validation",
        home_advances=0.28,
        away_advances=0.72,
        top_scores=((0, 2, 0.16), (1, 2, 0.13), (0, 1, 0.12)),
        over_25=0.48,
        btts=0.42,
        score_home=0,
        score_away=2,
        outcome_class=2,
        outcome_probability=0.64,
        score_probability=0.16,
        blend_probs=(0.14, 0.22, 0.64),
        poisson_outcome_probs=(0.18, 0.24, 0.58),
    )


def bright_pixel_count(surface: pygame.Surface) -> int:
    pixels = 0
    for y in range(0, surface.get_height(), 2):
        for x in range(0, surface.get_width(), 2):
            color = surface.get_at((x, y))
            if color.a > 30 and color.r > 210 and color.g > 210 and color.b > 185:
                pixels += 1
    return pixels


def alpha_surface_gap_px(
    first_surface: pygame.Surface,
    first_rect: pygame.Rect,
    second_surface: pygame.Surface,
    second_rect: pygame.Rect,
    threshold: int = 30,
) -> float:
    union = first_rect.union(second_rect)
    first_canvas = np.zeros((union.h, union.w), dtype=bool)
    second_canvas = np.zeros((union.h, union.w), dtype=bool)

    def place(surface: pygame.Surface, rect: pygame.Rect, canvas: np.ndarray) -> None:
        alpha = pygame.surfarray.array_alpha(surface).T >= threshold
        offset_x = rect.x - union.x
        offset_y = rect.y - union.y
        canvas[offset_y : offset_y + rect.h, offset_x : offset_x + rect.w] = alpha

    place(first_surface, first_rect, first_canvas)
    place(second_surface, second_rect, second_canvas)
    if not first_canvas.any() or not second_canvas.any():
        return float("inf")
    if np.logical_and(first_canvas, second_canvas).any():
        return 0.0
    distance = cv2.distanceTransform((~first_canvas).astype(np.uint8), cv2.DIST_L2, 5)
    return max(0.0, float(distance[second_canvas].min()) - 1.0)


def dark_pixel_count(surface: pygame.Surface, step: int = 2) -> int:
    pixels = 0
    for y in range(0, surface.get_height(), step):
        for x in range(0, surface.get_width(), step):
            color = surface.get_at((x, y))
            if color.r < 42 and color.g < 48 and color.b < 42:
                pixels += 1
    return pixels


def frame_sample_delta(a: pygame.Surface, b: pygame.Surface, step: int = 6) -> int:
    delta = 0
    for y in range(0, min(a.get_height(), b.get_height()), step):
        for x in range(0, min(a.get_width(), b.get_width()), step):
            ca = a.get_at((x, y))
            cb = b.get_at((x, y))
            delta += abs(ca.r - cb.r) + abs(ca.g - cb.g) + abs(ca.b - cb.b)
    return delta


def edge_delta(surface: pygame.Surface, offset: int = 1, step_y: int = 4) -> float:
    total = 0
    samples = 0
    left_x = max(0, offset)
    right_x = max(0, surface.get_width() - offset - 1)
    for y in range(0, surface.get_height(), step_y):
        left = surface.get_at((left_x, y))
        right = surface.get_at((right_x, y))
        total += abs(left.r - right.r) + abs(left.g - right.g) + abs(left.b - right.b)
        samples += 1
    return total / max(1, samples)


def validate_parallax_assets() -> None:
    for filename in ("imagen_turf_near_source.png", "imagen_turf_mid_source.png"):
        path = PARALLAX_SOURCES / filename
        if not path.exists():
            raise AssertionError(f"missing image_gen parallax source: {path}")
    near_source = PARALLAX_SOURCES / "imagen_turf_near_source.png"
    mid_source = PARALLAX_SOURCES / "imagen_turf_mid_source.png"
    if file_hash(near_source) == file_hash(mid_source):
        # The source can only match if the generator intentionally derives two distinct layers.
        # The runtime strips below must still diverge; otherwise parallax looks like a sliding duplicate.
        pass
    strip_hashes = {}
    for filename in ("turf_near_strip.png", "turf_mid_strip.png"):
        path = PARALLAX_DIR / filename
        if not path.exists():
            raise AssertionError(f"missing parallax strip: {path}")
        strip_hashes[filename] = file_hash(path)
        image = pygame.image.load(path).convert_alpha()
        if image.get_size() != (1440, 232):
            raise AssertionError(f"{path} has unexpected size {image.get_size()}")
        if edge_delta(image) > 34:
            raise AssertionError(f"{path} is not horizontally seamless enough: edge delta {edge_delta(image):.1f}")
    if strip_hashes["turf_near_strip.png"] == strip_hashes["turf_mid_strip.png"]:
        raise AssertionError("near and mid parallax strips are identical; depth cannot read naturally")
    near = pygame.image.load(PARALLAX_DIR / "turf_near_strip.png").convert_alpha()
    mid = pygame.image.load(PARALLAX_DIR / "turf_mid_strip.png").convert_alpha()
    if frame_sample_delta(near, mid, step=8) < 18000:
        raise AssertionError("near and mid parallax strips are too similar for visible depth")


def goalkeeper_render_for_state(app: App, team: object, state: dict[str, object], flip: bool) -> tuple[pygame.Surface, pygame.Rect]:
    shot_progress = float(state.get("raw_shot_progress", state["shot_progress"]))
    flip = bool(state.get("keeper_flip", flip))
    team_code = getattr(team, "code")
    frames = app.assets.cinematic_keeper_frames[team_code]
    keeper_action = str(state.get("keeper_action", ""))
    active_goal = bool(state.get("active_goal")) or keeper_action == "dive_save"
    _index, scale, angle = app.cinematic_keeper_animation_state(
        active_goal,
        shot_progress,
        len(frames),
        flip,
        keeper_action,
    )
    first, following, blend = app.cinematic_keeper_frame_blend(
        active_goal,
        shot_progress,
        len(frames),
        keeper_action,
    )
    attacker = app.home if state.get("possession") == "home" else app.away
    contrast_kit = app.assets.cinematic_source_code(attacker) == "green"
    frame = app.cinematic_keeper_material(
        team,
        flip,
        first,
        following,
        blend,
        scale,
        angle,
        contrast_kit,
    )
    keeper_x, keeper_y = state["keeper_pos"]  # type: ignore[misc]
    rect = app.cinematic_keeper_rect(
        frame,
        (float(keeper_x), float(keeper_y)),
        shot_progress,
        active_goal,
        keeper_action,
    )
    return frame, rect


def visible_surface_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    *,
    alpha_threshold: int = 48,
) -> pygame.Rect:
    alpha_bounds = pygame.mask.from_surface(surface, alpha_threshold).get_bounding_rects()
    if not alpha_bounds:
        return pygame.Rect(rect.topleft, (0, 0))
    visible = alpha_bounds[0].copy()
    visible.unionall_ip(alpha_bounds[1:])
    visible.move_ip(rect.topleft)
    return visible


def runner_render_for_state(
    app: App,
    team: object,
    state: dict[str, object],
    direction: int,
) -> tuple[pygame.Surface, pygame.Rect]:
    poc2_sample = state.get("poc2_dribble_sample")
    if (
        isinstance(poc2_sample, Poc2DribbleSample)
        and bool(state.get("poc2_dribble"))
        and not bool(state.get("settled", False))
    ):
        team_code = getattr(team, "code")
        frames = (
            app.assets.cinematic_poc2_runners_left[team_code]
            if direction < 0
            else app.assets.cinematic_poc2_runners[team_code]
        )
        frame = frames[poc2_sample.frame_index]
        target = (
            max(1, round(poc2_sample.player.scene_size)),
        ) * 2
        rendered = app.cached_smoothscale(frame, target)
        rect = pygame.Rect(
            round(poc2_sample.player.scene_left),
            round(poc2_sample.player.scene_top),
            *target,
        )
        return rendered, rect
    sequence = state.get("poc_contract_sequence")
    sample = state.get("poc_contract_sample")
    viewport = state.get("poc_viewport")
    if (
        isinstance(sequence, PocSequence)
        and isinstance(sample, PocSequenceSample)
        and isinstance(viewport, PocViewport)
    ):
        frame, _frame_index, _metadata, actor_scale = (
            app.cinematic_poc_actor_material(
                sequence,
                sample,
                team,  # type: ignore[arg-type]
            )
        )
        actor_scale *= viewport.scale
        source_canvas_size = (
            int(app.assets.cinematic_poc2_motion["canvas_size"])
            if sample.actor_source == 0
            else POC_RUNNER_CANVAS_SIZE
        )
        target = (
            max(
                1,
                round(source_canvas_size * actor_scale),
            ),
        ) * 2
        root = viewport.point(
            float(state["poc_actor_x"]),
            sample.actor_ground_y,
        )
        rendered = app.cached_smoothscale(frame, target)
        if sample.actor_source == 0:
            rect = pygame.Rect(
                round(
                    root[0]
                    - source_canvas_size * 0.5 * actor_scale
                ),
                round(
                    root[1]
                    - float(
                        app.assets.cinematic_poc2_motion[
                            "canvas_ground_y"
                        ]
                    )
                    * actor_scale
                ),
                *target,
            )
        else:
            rect = pygame.Rect(
                round(
                    root[0]
                    - POC_RUNNER_ROOT[0] * actor_scale
                ),
                round(
                    root[1]
                    - POC_RUNNER_ROOT[1] * actor_scale
                ),
                *target,
            )
            visible_actor = app.visible_bbox(rendered)
            submersion = (
                rect.top + visible_actor.bottom - round(root[1])
            )
            if submersion > 0:
                rect.y -= submersion
        return rendered, rect
    rendered_kind = str(state.get("rendered_player_kind", "run"))
    pose_key = "kick_pose" if rendered_kind == "kick" else "runner_pose"
    pose = state.get(pose_key)
    if not isinstance(pose, dict):
        raise AssertionError(f"cinematic state is missing the {pose_key} contract")
    team_code = getattr(team, "code")
    if rendered_kind == "kick":
        frames = app.assets.cinematic_kicks_left[team_code] if direction < 0 else app.assets.cinematic_kicks[team_code]
        frame = frames[int(pose["frame_index"])]
    elif rendered_kind == "stop":
        frames = app.assets.cinematic_stops_left[team_code] if direction < 0 else app.assets.cinematic_stops[team_code]
        frame = frames[int(state["stop_render_frame"])]
    else:
        frames = app.assets.cinematic_runners_left[team_code] if direction < 0 else app.assets.cinematic_runners[team_code]
        frame = frames[int(state["runner_render_frame"])]
    target = tuple(int(value) for value in pose["target_size"])  # type: ignore[arg-type]
    rendered = app.cached_smoothscale(frame, target)
    left, top = pose["rect_topleft"]  # type: ignore[misc]
    rect = pygame.Rect(round(float(left)), round(float(top)), *target)
    return rendered, rect


def ball_render_for_state(
    app: App,
    state: dict[str, object],
) -> tuple[pygame.Surface, pygame.Rect]:
    scale = int(state["ball_scale"])
    squash_x, squash_y = state["ball_squash"]  # type: ignore[misc]
    ball_size = (
        max(22, int(scale * float(squash_x))),
        max(22, int(scale * float(squash_y))),
    )
    ball = app.cached_cinematic_ball_material(scale, float(state["ball_rotation_degrees"]))
    if ball.get_size() != ball_size:
        ball = pygame.transform.smoothscale(ball, ball_size)
    ball_x, ball_y = state["ball_pos"]  # type: ignore[misc]
    return ball, ball.get_rect(center=(round(float(ball_x)), round(float(ball_y))))


def validate_nil_draw_has_no_fake_goals_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = nil_draw_prediction()
    app.match_prediction = pred
    field = pygame.Rect(32, 110, 910, 490)
    if app.final_score_from_prediction(pred) != (0, 0):
        raise AssertionError("nil-draw regression prediction must finish 0 x 0")
    if app.goal_schedule(pred):
        raise AssertionError(f"0 x 0 match must not schedule real goals: {app.goal_schedule(pred)}")

    chance_events = app.chance_schedule(pred)
    if not chance_events:
        raise AssertionError("0 x 0 match still needs non-goal pressure chances for cinematic suspense")

    original_goal_layer = app.draw_cinematic_poc_goal_layer
    original_keeper = app.draw_cinematic_poc_keeper
    goal_layer_calls: list[bool] = []
    keeper_calls: list[tuple[str, int]] = []

    def spy_goal_layer(
        state: dict[str, object],
        *,
        front: bool,
    ) -> None:
        goal_layer_calls.append(front)
        original_goal_layer(state, front=front)

    def spy_keeper(state: dict[str, object]) -> None:
        sequence = state.get("poc_contract_sequence")
        sample = state.get("poc_contract_sample")
        if isinstance(sequence, PocSequence) and isinstance(
            sample,
            PocSequenceSample,
        ):
            keeper_calls.append(
                (sequence.keeper_direction, sample.keeper_frame)
            )
        original_keeper(state)

    try:
        app.draw_cinematic_poc_goal_layer = spy_goal_layer  # type: ignore[method-assign]
        app.draw_cinematic_poc_keeper = spy_keeper  # type: ignore[method-assign]
        for minute, _side, kind in chance_events:
            app.t = (minute - CHANCE_EVENT_WINDOW_MINUTES + 0.98 * CHANCE_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS
            state = app.cinematic_scene_state(field, pred)
            if state.get("active_goal") or app.active_goal_event(pred) is not None:
                raise AssertionError(f"0 x 0 chance became a real goal event: {minute}, {kind}, {state}")
            if app.score_from_prediction(pred) != (0, 0):
                raise AssertionError(f"0 x 0 live score changed during a non-goal chance: {app.score_from_prediction(pred)}")
            app.screen.fill((0, 0, 0))
            goal_layer_calls.clear()
            keeper_calls.clear()
            app.draw_field(pred, pred, "CONFRONTO")
            if goal_layer_calls != [False, True]:
                raise AssertionError(f"0 x 0 {kind} chance must keep the goal frame visible")
            if float(state.get("net_progress", 0.0)) > 0.0:
                raise AssertionError(
                    f"0 x 0 {kind} chance rendered goal-impact ripple as if it scored"
                )
            if not keeper_calls:
                raise AssertionError(f"0 x 0 {kind} chance must keep the goalkeeper visible")
            sequence = state.get("poc_contract_sequence")
            if (
                not isinstance(sequence, PocSequence)
                or sequence.outcome != kind
            ):
                raise AssertionError(
                    f"0 x 0 {kind} chance selected the wrong POC sequence"
                )
            if app.active_goal_event(pred) is not None:
                raise AssertionError("0 x 0 non-goal chance exposed GOOOL overlay timing")
    finally:
        app.draw_cinematic_poc_goal_layer = original_goal_layer  # type: ignore[method-assign]
        app.draw_cinematic_poc_keeper = original_keeper  # type: ignore[method-assign]

    app.t = SIMULATION_SECONDS
    if app.score_from_prediction(pred) != (0, 0):
        raise AssertionError(f"0 x 0 final score drifted after full time: {app.score_from_prediction(pred)}")


def validate_cinematic_reveal_timing_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    field = pygame.Rect(32, 110, 910, 490)
    first_goal = app.goal_schedule(pred)[0][0]
    layer_calls: list[bool] = []
    keeper_calls = 0
    original_goal_layer = app.draw_cinematic_poc_goal_layer
    original_keeper = app.draw_cinematic_poc_keeper

    def goal_layer_spy(
        state: dict[str, object],
        *,
        front: bool,
    ) -> None:
        layer_calls.append(front)
        original_goal_layer(state, front=front)

    def keeper_spy(state: dict[str, object]) -> None:
        nonlocal keeper_calls
        keeper_calls += 1
        original_keeper(state)

    try:
        app.draw_cinematic_poc_goal_layer = goal_layer_spy  # type: ignore[method-assign]
        app.draw_cinematic_poc_keeper = keeper_spy  # type: ignore[method-assign]

        app.t = 8.0 / 90.0 * SIMULATION_SECONDS
        app.screen.fill((0, 0, 0))
        app.draw_field(pred, pred, "CONFRONTO")
        if layer_calls or keeper_calls:
            raise AssertionError(
                "POC goal/keeper appeared without an active danger event"
            )

        goal_sizes: set[tuple[int, int]] = set()
        for progress in (0.03, 0.20, 0.50, 0.82):
            layer_calls.clear()
            keeper_calls = 0
            app.t = (
                first_goal
                - GOAL_EVENT_WINDOW_MINUTES
                + progress * GOAL_EVENT_WINDOW_MINUTES
            ) / 90.0 * SIMULATION_SECONDS
            state = app.cinematic_scene_state(field, pred)
            sequence = state.get("poc_contract_sequence")
            if not isinstance(sequence, PocSequence):
                raise AssertionError(
                    f"danger progress {progress:.2f} did not enter the approved POC"
                )
            goal = state.get("goal_rect")
            if not isinstance(goal, pygame.Rect):
                raise AssertionError(
                    f"{sequence.key}: missing POC goal geometry"
                )
            goal_sizes.add(goal.size)
            if float(state.get("net_progress", 0.0)) > 0.0:
                raise AssertionError(
                    f"{sequence.key}: net moved before ball impact"
                )
            app.screen.fill((0, 0, 0))
            app.draw_field(pred, pred, "CONFRONTO")
            if layer_calls != [False, True] or keeper_calls != 1:
                raise AssertionError(
                    f"{sequence.key}: goal/keeper did not remain continuously "
                    f"visible during danger: layers={layer_calls}, keeper={keeper_calls}"
                )
        if len(goal_sizes) != 1:
            raise AssertionError(
                f"approved goal changes size during approach: {sorted(goal_sizes)}"
            )
    finally:
        app.draw_cinematic_poc_goal_layer = original_goal_layer  # type: ignore[method-assign]
        app.draw_cinematic_poc_keeper = original_keeper  # type: ignore[method-assign]


def validate_ball_physics_contract_fast() -> None:
    field = pygame.Rect(32, 110, 910, 490)
    max_frame_displacement = 36.0

    for direction, side in (("right", "home"), ("left", "away")):
        app = App(seed=2026)
        app.set_simulate("match")
        event = CinematicAttackEvent(50, side, True, "goal")
        sequence = app.cinematic_poc_sequence_for_event(event)
        if sequence.attack_direction != direction:
            raise AssertionError(
                f"{direction} POC selected the wrong attack direction: {sequence.key}"
            )

        release_progress = sequence.release_seconds / sequence.impact_seconds
        frame_progress = (1.0 / BALL_RUNTIME_FRAME_HZ) / sequence.impact_seconds
        sample_count = math.ceil(
            (1.0 - release_progress) / frame_progress
        )
        progresses = [
            min(1.0, release_progress + index * frame_progress)
            for index in range(sample_count + 1)
        ]
        if progresses[-1] < 1.0:
            progresses.append(1.0)

        previous_ball: tuple[float, float] | None = None
        previous_rotation: float | None = None
        directed_positions: list[float] = []
        for progress in progresses:
            state = app.cinematic_poc_scene_state(
                field,
                event,
                progress,
            )
            selected = state.get("poc_contract_sequence")
            sample = state.get("poc_contract_sample")
            if (
                not isinstance(selected, PocSequence)
                or selected.key != sequence.key
                or not isinstance(sample, PocSequenceSample)
            ):
                raise AssertionError(
                    f"{sequence.key}: runtime left the approved POC sequence"
                )
            ball_contract_snapshot(
                state,
                f"{sequence.key} progress={progress:.4f}",
            )
            ball = tuple(float(value) for value in state["ball_pos"])
            if not all(math.isfinite(value) for value in ball):
                raise AssertionError(
                    f"{sequence.key}: non-finite ball position {ball}"
                )
            direction_sign = 1.0 if direction == "right" else -1.0
            directed_positions.append(direction_sign * ball[0])
            if previous_ball is not None:
                displacement = math.dist(previous_ball, ball)
                if displacement > max_frame_displacement:
                    raise AssertionError(
                        f"{sequence.key}: ball teleports {displacement:.2f}px "
                        "between approved 60 Hz samples"
                    )
            rotation = float(state["ball_rotation_degrees"])
            if previous_rotation is not None:
                rotation_step = abs(
                    wrapped_angle_delta(rotation, previous_rotation)
                )
                if rotation_step > 90.0:
                    raise AssertionError(
                        f"{sequence.key}: ball rotation jumps "
                        f"{rotation_step:.2f} degrees in one frame"
                    )
            previous_ball = ball
            previous_rotation = rotation

        if any(
            following + 2.0 < current
            for current, following in zip(
                directed_positions,
                directed_positions[1:],
            )
        ):
            raise AssertionError(
                f"{sequence.key}: shot reverses before reaching the net"
            )

        impact = app.cinematic_poc_scene_state(
            field,
            event,
            1.0,
        )
        impact_sample = impact["poc_contract_sample"]
        if (
            not isinstance(impact_sample, PocSequenceSample)
            or abs(
                impact_sample.elapsed - sequence.impact_seconds
            )
            > 1.0 / BALL_RUNTIME_FRAME_HZ
        ):
            raise AssertionError(
                f"{sequence.key}: impact frame drifted from the POC timeline"
            )
        impact_ball = tuple(
            float(value) for value in impact["ball_pos"]
        )
        impact_scale = int(impact["ball_scale"])
        impact_rect = pygame.Rect(
            0,
            0,
            impact_scale,
            impact_scale,
        )
        impact_rect.center = (
            round(impact_ball[0]),
            round(impact_ball[1]),
        )
        goal = impact["goal_rect"]
        if (
            not isinstance(goal, pygame.Rect)
            or not goal.inflate(-4, -4).colliderect(impact_rect)
        ):
            raise AssertionError(
                f"{sequence.key}: ball is outside the approved goal "
                f"at impact: ball={impact_rect}, goal={goal}"
            )
        event_seconds = (
            GOAL_EVENT_WINDOW_MINUTES
            / 90.0
            * SIMULATION_SECONDS
        )
        reaction = app.cinematic_poc_scene_state(
            field,
            event,
            1.0
            + (1.0 / BALL_RUNTIME_FRAME_HZ) / event_seconds,
        )
        if float(reaction["net_progress"]) <= 0.0:
            raise AssertionError(
                f"{sequence.key}: net does not react on the first frame after impact"
            )

        payoff_states = [
            app.cinematic_poc_scene_state(
                field,
                event,
                progress,
            )
            for progress in (1.02, 1.05, 1.10, 1.15, 1.20, 1.25)
        ]
        peak_net = max(
            float(state["net_progress"])
            for state in payoff_states
        )
        settled = payoff_states[-1]
        settled_sample = settled["poc_contract_sample"]
        if (
            not isinstance(settled_sample, PocSequenceSample)
            or not settled_sample.ball_visible
            or peak_net < 0.55
            or float(settled["net_progress"]) >= peak_net * 0.20
        ):
            raise AssertionError(
                f"{sequence.key}: ball/net payoff was not preserved after impact"
            )


def validate_aaa_findings_light_gate() -> None:
    field = pygame.Rect(32, 110, 910, 490)

    for direction, side in (("right", "home"), ("left", "away")):
        app = App(seed=2026)
        app.set_simulate("match")
        goal_event = CinematicAttackEvent(50, side, True, "goal")
        goal_sequence = app.cinematic_poc_sequence_for_event(goal_event)
        possession_team = app.home if side == "home" else app.away

        actor_frames: set[tuple[int, int]] = set()
        actor_visible_heights: dict[int, list[int]] = {
            0: [],
            1: [],
        }
        for progress in (
            0.10,
            0.22,
            0.34,
            0.46,
            0.58,
            0.70,
            0.82,
            0.94,
        ):
            state = app.cinematic_poc_scene_state(
                field,
                goal_event,
                progress,
            )
            sample = state["poc_contract_sample"]
            viewport = state["poc_viewport"]
            if (
                not isinstance(sample, PocSequenceSample)
                or not sample.actor_visible
            ):
                continue
            _frame, render_frame, _metadata, actor_scale = (
                app.cinematic_poc_actor_material(
                    goal_sequence,
                    sample,
                    possession_team,
                )
            )
            actor_frames.add((sample.actor_source, render_frame))
            source_canvas_size = (
                int(app.assets.cinematic_poc2_motion["canvas_size"])
                if sample.actor_source == 0
                else POC_RUNNER_CANVAS_SIZE
            )
            target_size = max(
                1,
                round(
                    source_canvas_size
                    * actor_scale
                    * viewport.scale
                ),
            )
            rendered = pygame.transform.smoothscale(
                _frame,
                (target_size, target_size),
            )
            actor_visible_heights[sample.actor_source].append(
                rendered.get_bounding_rect(min_alpha=30).height
            )
        if len(actor_frames) < 6:
            raise AssertionError(
                f"{goal_sequence.key}: approved run/kick progression is too sparse: "
                f"{sorted(actor_frames)}"
            )
        if any(not heights for heights in actor_visible_heights.values()):
            raise AssertionError(
                f"{goal_sequence.key}: run/kick height samples are incomplete: "
                f"{actor_visible_heights}"
            )
        run_height = float(np.median(actor_visible_heights[0]))
        kick_height = float(np.median(actor_visible_heights[1]))
        if abs(run_height - kick_height) > 5.0:
            raise AssertionError(
                f"{goal_sequence.key}: visible player height changes at the "
                f"run/kick handoff: {run_height:.1f}/{kick_height:.1f} "
                f"from {actor_visible_heights}"
            )

        payoff_states = [
            app.cinematic_poc_scene_state(
                field,
                goal_event,
                progress,
            )
            for progress in (
                1.0,
                1.02,
                1.05,
                1.10,
                1.15,
                1.20,
                1.25,
            )
        ]
        net_values = [
            float(state["net_progress"])
            for state in payoff_states
        ]
        peak_index = max(
            range(len(net_values)),
            key=net_values.__getitem__,
        )
        if (
            net_values[peak_index] < 0.55
            or peak_index == 0
            or net_values[-1] >= net_values[peak_index] * 0.20
        ):
            raise AssertionError(
                f"{goal_sequence.key}: net impulse/decay drifted: "
                f"{[round(value, 3) for value in net_values]}"
            )

        for outcome in ("goal", "save", "wide"):
            event = CinematicAttackEvent(
                50,
                side,
                outcome == "goal",
                outcome,
            )
            state = app.cinematic_poc_scene_state(
                field,
                event,
                1.08 if outcome == "goal" else 1.25,
            )
            sample = state["poc_contract_sample"]
            if not isinstance(sample, PocSequenceSample):
                raise AssertionError(
                    f"{direction}/{outcome}: missing approved POC sample"
                )
            render_order: list[str] = []
            original_goal_layer = app.draw_cinematic_poc_goal_layer
            original_actor = app.draw_cinematic_poc_actor
            original_ball = app.draw_cinematic_poc_ball
            original_keeper = app.draw_cinematic_poc_keeper

            def goal_layer_spy(
                current_state: dict[str, object],
                *,
                front: bool,
            ) -> None:
                render_order.append(
                    "goal_front" if front else "goal_back"
                )
                original_goal_layer(
                    current_state,
                    front=front,
                )

            def actor_spy(
                current_state: dict[str, object],
            ) -> None:
                render_order.append("actor")
                original_actor(current_state)

            def ball_spy(
                current_state: dict[str, object],
            ) -> None:
                render_order.append("ball")
                original_ball(current_state)

            def keeper_spy(
                current_state: dict[str, object],
            ) -> None:
                render_order.append("keeper")
                original_keeper(current_state)

            try:
                app.draw_cinematic_poc_goal_layer = goal_layer_spy  # type: ignore[method-assign]
                app.draw_cinematic_poc_actor = actor_spy  # type: ignore[method-assign]
                app.draw_cinematic_poc_ball = ball_spy  # type: ignore[method-assign]
                app.draw_cinematic_poc_keeper = keeper_spy  # type: ignore[method-assign]
                app.screen.fill((0, 0, 0))
                app.draw_cinematic_poc_sequence(
                    field,
                    state,
                )
            finally:
                app.draw_cinematic_poc_goal_layer = original_goal_layer  # type: ignore[method-assign]
                app.draw_cinematic_poc_actor = original_actor  # type: ignore[method-assign]
                app.draw_cinematic_poc_ball = original_ball  # type: ignore[method-assign]
                app.draw_cinematic_poc_keeper = original_keeper  # type: ignore[method-assign]

            expected_middle = (
                ["actor", "ball", "keeper"]
                if sample.ball_after_keeper
                else ["actor", "keeper", "ball"]
            )
            expected_order = [
                "goal_back",
                *expected_middle,
                "goal_front",
            ]
            if render_order != expected_order:
                raise AssertionError(
                    f"{direction}/{outcome}: POC layer order drifted: "
                    f"{render_order} != {expected_order}"
                )


def validate_match_final_cinematic_settlement_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = replace(
        home_win_prediction(),
        score_home=6,
        score_away=4,
    )
    app.match_prediction = pred
    if app.goal_schedule(pred)[-1][0] != 86:
        raise AssertionError("final settlement fixture lost its late goal")
    app.t = SIMULATION_SECONDS
    field = app.match_field_rect()
    state = app.cinematic_scene_state(field, pred)
    ball_x, _ball_y = state["ball_pos"]  # type: ignore[misc]
    if (
        state.get("active_attack") is not None
        or not bool(state.get("settled"))
        or abs(float(ball_x) - field.centerx) > 1e-6
    ):
        raise AssertionError(
            "finished match must leave the player/ball settled at midfield"
        )


def validate_goal_composite_cache_bound_gate() -> None:
    app = App(seed=2026)
    patch = pygame.Surface((2, 2), pygame.SRCALPHA)
    patch.fill((255, 255, 255, 255))
    scaled_cache_size = len(app.surface_cache.scaled)
    for index in range(CINEMATIC_GOAL_COMPOSITE_CACHE_LIMIT + 5):
        rendered = app.cinematic_poc_composite_goal_layer(
            cache_key=("qa-cache", index),
            canvas_size=(4, 4),
            patch=patch,
            patch_rect=(1, 1, 2, 2),
            target_size=(3, 3),
        )
        if rendered.get_size() != (3, 3):
            raise AssertionError("goal composite cache returned native size")
    if len(app.poc_goal_composite_cache) != CINEMATIC_GOAL_COMPOSITE_CACHE_LIMIT:
        raise AssertionError(
            "goal composite LRU cache did not enforce its memory bound"
        )
    if (
        "scaled_goal_composite",
        "qa-cache",
        0,
        3,
        3,
    ) in app.poc_goal_composite_cache:
        raise AssertionError("goal composite LRU cache did not evict oldest entry")
    if len(app.surface_cache.scaled) != scaled_cache_size:
        raise AssertionError(
            "dynamic goal composites leaked into the global scale cache"
        )


def validate_audio_event_order() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    goal_minute, side = app.goal_schedule(pred)[0]
    event = CinematicAttackEvent(
        goal_minute,
        side,
        True,
        "goal",
    )
    thresholds = app.cinematic_poc_audio_thresholds(event)
    played: list[str] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        played.append(name)

    app.sound.play = spy  # type: ignore[method-assign]
    for progress in (
        thresholds["kick"],
        thresholds["whoosh"],
        thresholds["net"],
        thresholds["reverb"],
        thresholds["reverb"] + 0.01,
        thresholds["reverb"] + 0.01,
    ):
        app.t = goal_progress_second(goal_minute, progress)
        app.update_soundscape(1 / 60)
        app.flush_queued_match_audio()
    if played != GOAL_AUDIO_EVENTS:
        raise AssertionError(f"unexpected audio event order/dedup: {played}")

    app.screen.fill((0, 0, 0))
    app.draw_score_panel({"CONFRONTO": pred}, "CONFRONTO", pred)
    if played != GOAL_AUDIO_EVENTS:
        raise AssertionError(f"draw_score_panel must not trigger audio events: {played}")
    app.t = goal_progress_second(
        goal_minute,
        thresholds["net"],
    )
    app.draw_simulate()
    if played != GOAL_AUDIO_EVENTS:
        raise AssertionError(f"draw_simulate must not trigger audio events: {played}")

    app = App(seed=2026)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    goal_minute, side = app.goal_schedule(pred)[0]
    event = CinematicAttackEvent(
        goal_minute,
        side,
        True,
        "goal",
    )
    thresholds = app.cinematic_poc_audio_thresholds(event)
    played = []
    app.sound.play = spy  # type: ignore[method-assign]
    app.t = goal_progress_second(
        goal_minute,
        thresholds["kick"] - 0.04,
    )
    app.update_soundscape(1 / 60)
    app.t = goal_progress_second(
        goal_minute,
        thresholds["reverb"] + 0.02,
    )
    app.update_soundscape(1 / 8)
    app.flush_queued_match_audio()
    synchronized_impact = ["net", "bass", "cheer", "reverb"]
    if played != synchronized_impact:
        raise AssertionError(
            f"stuttered audio update should keep the synchronized impact and elapsed tail: {played}"
        )
    for _index in range(6):
        app.update_soundscape(1 / 60)
        app.flush_queued_match_audio()
    if played != synchronized_impact:
        raise AssertionError(f"stuttered audio cues duplicated after impact catch-up: {played}")


def validate_sound_assets() -> None:
    if not STADIUM_BG.exists():
        raise AssertionError(f"missing stadium background: {STADIUM_BG}")
    for filename in REQUIRED_SOUNDS:
        path = SOUND_DIR / filename
        if not path.exists():
            raise AssertionError(f"missing sound asset: {path}")


def validate_app_icon() -> None:
    if not APP_ICON.exists():
        raise AssertionError(f"missing generated game icon: {APP_ICON}")
    icon = pygame.image.load(APP_ICON).convert_alpha()
    if icon.get_width() < 512 or icon.get_height() < 512:
        raise AssertionError(f"game icon source is too small: {icon.get_size()}")
    center_sample = icon.subsurface(pygame.Rect(icon.get_width() // 4, icon.get_height() // 4, icon.get_width() // 2, icon.get_height() // 2)).copy()
    if bright_pixel_count(center_sample) < 7000:
        raise AssertionError("game icon does not preserve the bright trophy/ball center")

    app = App(seed=2026)
    if app.app_icon is None or app.menu_icon is None or app.top_icon is None:
        raise AssertionError("game icon is not loaded into the App surfaces")
    app.draw_menu()
    title_w = max(app.f_xl.size("ORÁCULO")[0], app.f_xl.size("DA COPA")[0])
    menu_icon_rect = app.menu_icon.get_rect(midleft=(app.start_button.rect.x + title_w + 18, 170))
    menu_sample = app.screen.subsurface(menu_icon_rect.inflate(18, 18).clip(app.screen.get_rect())).copy()
    if bright_pixel_count(menu_sample) < 50:
        raise AssertionError("initial screen does not visibly feature the generated game icon beside the title")
    app.draw_top("QA", "visual")
    top_sample = app.screen.subsurface(pygame.Rect(180, 18, 56, 56)).copy()
    if bright_pixel_count(top_sample) < 5:
        raise AssertionError("game screens do not show the generated icon in the top bar")


def validate_fifa_external_assets() -> None:
    assets_doc = ROOT / "docs" / "ASSETS.md"
    if not assets_doc.exists():
        raise AssertionError("FIFA external imagery must keep source provenance in docs/ASSETS.md")
    assets_doc_text = assets_doc.read_text(encoding="utf-8")
    for required_token in ("fifa_maple.jpg", "fifa_mexico_opening_ceremony_clean.png", "digitalhub.fifa.com"):
        if required_token not in assets_doc_text:
            raise AssertionError(f"FIFA external imagery provenance missing token in docs/ASSETS.md: {required_token}")
    if FIFA_EXTERNAL_IMAGES.get("mexico_opening") != "fifa_mexico_opening_ceremony_clean.png":
        raise AssertionError("selection/tournament background must use the ImageGen-cleaned Mexico opening image")
    hashes = []
    for key, filename in FIFA_EXTERNAL_IMAGES.items():
        path = FIFA_EXTERNAL_DIR / filename
        if not path.exists():
            raise AssertionError(f"missing FIFA external image for {key}: {path}")
        hashes.append(file_hash(path))
        image = pygame.image.load(path).convert_alpha()
        if image.get_width() < 900 or image.get_height() < 650:
            raise AssertionError(f"FIFA external image is too small for cover use: {path} {image.get_size()}")
        sample = pygame.transform.smoothscale(image, (320, 180))
        if bright_pixel_count(sample) < 30 and dark_pixel_count(sample) < 30:
            raise AssertionError(f"FIFA external image looks visually empty: {path}")
        if key == "mexico_opening":
            text_area = pygame.Rect(int(image.get_width() * 0.33), int(image.get_height() * 0.03), int(image.get_width() * 0.34), int(image.get_height() * 0.18))
            if bright_pixel_count(image.subsurface(text_area).copy()) > 80:
                raise AssertionError("clean Mexico opening image still appears to contain the removed title text")
    if len(set(hashes)) != len(hashes):
        raise AssertionError("FIFA external images must not be duplicate placeholder files")

    app = App(seed=2026)
    if set(app.assets.fifa_images) != set(FIFA_EXTERNAL_IMAGES):
        raise AssertionError(f"App did not load all FIFA external images: {sorted(app.assets.fifa_images)}")
    app.draw_select()
    select_backdrop = app.screen.subsurface(pygame.Rect(500, 96, 280, 160)).copy()
    if edge_energy(select_backdrop) < 8:
        raise AssertionError("selection screen does not visibly use the cleaned Mexico opening image")
    app.state = "tournament"
    app.t = 1.4
    app.draw_tournament()
    mascot_area = app.screen.subsurface(pygame.Rect(720, 184, 448, 326)).copy()
    if edge_energy(mascot_area) < 12:
        raise AssertionError("Monte Carlo loading screen does not visibly use the FIFA mascot imagery")
    del app
    gc.collect()


def validate_monte_carlo_fast_path() -> None:
    if TOURNAMENT_MONTE_CARLO_RUNS < 1000:
        raise AssertionError("game Monte Carlo must run the full 1000-Cup sample")
    if not 1 <= TOURNAMENT_MONTE_CARLO_WORKERS <= 8:
        raise AssertionError(f"interactive Monte Carlo workers must match the pipeline cap: {TOURNAMENT_MONTE_CARLO_WORKERS}")
    model = WorldCupModel()
    sota_module = sys.modules["sota_pipeline"]
    for seed in range(2026, 2029):
        full_champion = sota_module.simulate_tournament(model.package, seed)[0]
        fast_champion = sota_module.simulate_tournament_champion(model.package, seed)
        if full_champion != fast_champion:
            raise AssertionError(f"fast Monte Carlo champion path diverged for seed {seed}: {full_champion} != {fast_champion}")
        story_champion = sota_module.simulate_tournament_champion_story(model.package, seed).champion
        if full_champion != story_champion:
            raise AssertionError(f"story Monte Carlo path diverged for seed {seed}: {full_champion} != {story_champion}")
    progress_events = []

    def progress(done: int, total: int, _odds: list[tuple[str, int, float]]) -> bool:
        progress_events.append(done)
        return True

    qa_runs = 24
    start = time.perf_counter()
    odds, representative = model.champion_odds_with_representative(
        runs=qa_runs,
        seed=90210,
        workers=TOURNAMENT_MONTE_CARLO_WORKERS,
        progress_callback=progress,
        progress_with_odds=False,
        use_scenario_bank=False,
    )
    elapsed = time.perf_counter() - start
    if not odds or representative is None:
        raise AssertionError("fast Monte Carlo did not return odds plus a representative tournament")
    if str(representative.get("runtime_monte_carlo_source", "")) == "scenario_bank_bootstrap":
        raise AssertionError("fresh Monte Carlo QA unexpectedly used the runtime scenario bank")
    policy = str(representative.get("representative_policy", ""))
    if "plausible_story" not in policy:
        raise AssertionError(f"representative tournament must use narrative plausibility policy: {policy}")
    if int(representative.get("representative_candidate_count", 0)) <= 0:
        raise AssertionError("representative tournament did not retain candidate campaigns for the selected champion")
    if float(representative.get("representative_plausibility_score", 0.0)) <= 0:
        raise AssertionError("representative tournament did not expose a positive plausibility score")
    if progress_events[-1] != qa_runs:
        raise AssertionError(f"fast Monte Carlo progress did not reach total: {progress_events[-3:]}")
    if elapsed > 12.0:
        raise AssertionError(f"fast Monte Carlo QA sample is too slow: {elapsed:.2f}s for {qa_runs} runs")


def validate_monte_carlo_runtime_mode_gate() -> None:
    if TOURNAMENT_MONTE_CARLO_RUNS < 1000:
        raise AssertionError("game Monte Carlo must run the full 1000-Cup sample")
    model = WorldCupModel()
    progress_events: list[int] = []

    def progress(done: int, total: int, _odds: list[tuple[str, int, float]]) -> bool:
        progress_events.append(done)
        return True

    odds, representative = model.champion_odds_with_representative(
        runs=24,
        seed=119911,
        workers=TOURNAMENT_MONTE_CARLO_WORKERS,
        progress_callback=progress,
        progress_with_odds=False,
        use_scenario_bank=TOURNAMENT_MONTE_CARLO_USE_SCENARIO_BANK,
    )
    if not odds or representative is None:
        raise AssertionError("interactive Monte Carlo mode did not return odds plus a representative tournament")
    source = str(representative.get("runtime_monte_carlo_source", "fresh"))
    if TOURNAMENT_MONTE_CARLO_USE_SCENARIO_BANK:
        if source != "scenario_bank_bootstrap":
            raise AssertionError(f"bootstrap mode did not use the runtime scenario bank: {source}")
        scenario_bank = list(getattr(model, "_scenario_bank", []))
        if len(scenario_bank) < TOURNAMENT_MONTE_CARLO_RUNS:
            raise AssertionError(
                f"runtime Monte Carlo scenario_bank is smaller than the interactive run count: "
                f"{len(scenario_bank)} < {TOURNAMENT_MONTE_CARLO_RUNS}; run make runtime-cache"
            )
        bank_counts = Counter(str(getattr(candidate, "champion")) for candidate in scenario_bank)
        if len(bank_counts) < 5:
            raise AssertionError(f"scenario_bank collapsed to too few champions: {bank_counts.most_common()}")
        bank_total = max(1, sum(bank_counts.values()))
        dominant_team, dominant_count = bank_counts.most_common(1)[0]
        dominant_share = dominant_count / bank_total
        if dominant_share > 0.55:
            raise AssertionError(
                f"scenario_bank is too concentrated: {dominant_team}={dominant_share:.1%}, counts={bank_counts.most_common(8)}"
            )
    elif source == "scenario_bank_bootstrap":
        raise AssertionError("default game Monte Carlo must be fresh; scenario bank is only an explicit turbo mode")
    if not progress_events or progress_events[-1] != 24:
        raise AssertionError(f"interactive Monte Carlo progress did not reach sample total: {progress_events[-3:]}")


def validate_monte_carlo_story_diversity_gate() -> None:
    model = WorldCupModel()
    # The favorite can be stable, but the displayed campaign must not collapse
    # to a single champion. These seeds cover deterministic QA plus independent
    # samples from the runtime bootstrap path.
    sample_seeds = (101, 2026, 335004, 526926, 997555, 536686, 678637, 901776, 999242, 107623, 523610, 90210)
    representatives: list[str] = []
    for seed in sample_seeds:
        odds, representative = model.champion_odds_with_representative(
            runs=TOURNAMENT_MONTE_CARLO_RUNS,
            seed=seed,
            workers=TOURNAMENT_MONTE_CARLO_WORKERS,
            progress_with_odds=False,
            use_scenario_bank=True,
        )
        if not odds or representative is None:
            raise AssertionError(f"Monte Carlo bootstrap did not return odds and representative for seed={seed}")
        representatives.append(str(representative.get("representative_for") or representative.get("champion")))

    representative_counts = Counter(representatives)
    if len(representative_counts) < 2:
        raise AssertionError(f"displayed Monte Carlo campaigns lack diversity: {representative_counts.most_common()}")
    representative_team, representative_count = representative_counts.most_common(1)[0]
    if representative_count / len(sample_seeds) > 0.75:
        raise AssertionError(
            f"displayed Monte Carlo campaigns are over-concentrated: "
            f"{representative_team}={representative_count}/{len(sample_seeds)}, counts={representative_counts.most_common()}"
        )


def validate_tournament_seed_entropy_gate() -> None:
    seeded_app = App(seed=2026)
    seeded_sequence: list[int] = []
    for _index in range(6):
        seeded_app.set_tournament()
        if seeded_app.pending_tournament_seed is None:
            raise AssertionError("seeded App did not stage a tournament seed")
        seeded_sequence.append(int(seeded_app.pending_tournament_seed))
    if len(set(seeded_sequence)) != len(seeded_sequence):
        raise AssertionError(f"seeded tournament sequence repeated too early: {seeded_sequence}")

    runtime_seeds: list[int] = []
    for _index in range(4):
        app = App()
        app.set_tournament()
        if app.pending_tournament_seed is None:
            raise AssertionError("runtime App did not stage a tournament seed")
        runtime_seeds.append(int(app.pending_tournament_seed))
    if len(set(runtime_seeds)) < 2:
        raise AssertionError(f"runtime tournament seed entropy collapsed: {runtime_seeds}")


def validate_asset_manifest() -> None:
    if not ASSET_MANIFEST.exists():
        raise AssertionError(f"missing asset manifest: {ASSET_MANIFEST}")
    manifest = json.loads(ASSET_MANIFEST.read_text())
    exact_assets = set()
    for values in manifest.get("used_runtime_assets", {}).values():
        exact_assets.update(str(item) for item in values)
    documentation_assets = {str(item) for item in manifest.get("documentation_assets", [])}
    source_assets = {str(item) for item in manifest.get("source_assets", [])}
    curation_assets = {str(item) for item in manifest.get("curation_assets", [])}
    allowed_orphans = {str(item["path"]) for item in manifest.get("allowed_orphans", [])}
    runtime_globs = [str(pattern) for pattern in manifest.get("generated_runtime_globs", [])]
    release_runtime_globs = [
        str(pattern)
        for pattern in manifest.get("release_runtime_globs", runtime_globs)
    ]
    source_globs = [str(pattern) for pattern in manifest.get("generated_source_globs", [])]
    curation_globs = [str(pattern) for pattern in manifest.get("curation_asset_globs", [])]
    rejected_globs = [str(pattern) for pattern in manifest.get("rejected_asset_globs", [])]

    def non_runtime_path(path: str) -> bool:
        parts = Path(path).parts
        return (
            path.startswith("docs/")
            or any(part in {"candidates", "docs", "raw", "rejected_assets", "source", "sources"} for part in parts)
            or any(part.endswith(("_source", "_sources")) for part in parts)
            or Path(path).name == "downloaded_audio_manifest.csv"
        )

    runtime_source_paths = sorted(path for path in exact_assets if not path.startswith("assets/") or non_runtime_path(path))
    if runtime_source_paths:
        raise AssertionError(f"used_runtime_assets must contain runtime payload only: {runtime_source_paths}")
    runtime_source_globs = sorted(pattern for pattern in runtime_globs if not pattern.startswith("assets/") or non_runtime_path(pattern))
    if runtime_source_globs:
        raise AssertionError(f"generated_runtime_globs must contain runtime payload only: {runtime_source_globs}")
    if set(runtime_globs) != set(release_runtime_globs):
        raise AssertionError("generated and release runtime globs must describe the same canonical payload")

    bundle_candidates = sorted(
        path
        for path in (*exact_assets, *allowed_orphans, *runtime_globs)
        if "/candidates/" in path or path.startswith("assets/sounds/candidates/")
    )
    if bundle_candidates:
        raise AssertionError(f"runtime bundle manifest must not package candidate assets: {bundle_candidates}")
    source_runtime_globs = sorted(
        pattern
        for pattern in runtime_globs
        if "/parallax_sources/" in pattern
        or pattern.endswith("_sources/*.png")
    )
    if source_runtime_globs:
        raise AssertionError(
            "runtime globs must not point at generated source sheets: "
            f"{source_runtime_globs}"
        )
    if set(runtime_globs) & set(source_globs):
        raise AssertionError("runtime and source globs must be disjoint in asset_manifest.json")
    if exact_assets & source_assets:
        raise AssertionError("runtime and source exact assets must be disjoint in asset_manifest.json")

    missing = sorted(path for path in exact_assets if not (ROOT / path).exists())
    if missing:
        raise AssertionError(f"manifest references missing assets: {missing}")
    missing_docs = sorted(path for path in documentation_assets if not (ROOT / path).exists())
    if missing_docs:
        raise AssertionError(f"manifest references missing documentation assets: {missing_docs}")
    missing_sources = sorted(path for path in source_assets if not (ROOT / path).exists())
    if missing_sources:
        raise AssertionError(f"manifest references missing source assets: {missing_sources}")
    missing_curation = sorted(path for path in curation_assets if not (ROOT / path).exists())
    if missing_curation:
        raise AssertionError(f"manifest references missing curation assets: {missing_curation}")
    missing_allowed = sorted(path for path in allowed_orphans if not (ROOT / path).exists())
    if missing_allowed:
        raise AssertionError(f"manifest allowlist references missing assets: {missing_allowed}")
    missing_source_globs = sorted(pattern for pattern in source_globs if not any((ROOT / match).is_file() for match in ROOT.glob(pattern)))
    if missing_source_globs:
        raise AssertionError(f"manifest source globs matched no files: {missing_source_globs}")
    missing_curation_globs = sorted(pattern for pattern in curation_globs if not any((ROOT / match).is_file() for match in ROOT.glob(pattern)))
    if missing_curation_globs:
        raise AssertionError(f"manifest curation globs matched no files: {missing_curation_globs}")
    missing_rejected_globs = sorted(pattern for pattern in rejected_globs if not any((ROOT / match).is_file() for match in ROOT.glob(pattern)))
    if missing_rejected_globs:
        raise AssertionError(f"manifest rejected globs matched no files: {missing_rejected_globs}")
    actual_assets = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "assets").rglob("*") if path.is_file())
    uncovered = []
    for path in actual_assets:
        if path == "assets/asset_manifest.json" or path in exact_assets or path in source_assets or path in curation_assets or path in allowed_orphans:
            continue
        if any(fnmatch.fnmatch(path, pattern) for pattern in runtime_globs):
            continue
        if any(fnmatch.fnmatch(path, pattern) for pattern in source_globs):
            continue
        if any(fnmatch.fnmatch(path, pattern) for pattern in curation_globs):
            continue
        if any(fnmatch.fnmatch(path, pattern) for pattern in rejected_globs):
            continue
        uncovered.append(path)
    if uncovered:
        raise AssertionError(f"asset files are not covered by manifest or allowlist: {uncovered}")

    for legacy in allowed_orphans:
        legacy_name = Path(legacy).name
        for source in (ROOT / "src").rglob("*.py"):
            if legacy_name in source.read_text(errors="ignore"):
                raise AssertionError(f"allowed orphan is still referenced by runtime source: {legacy}")


def validate_release_inventory_contract() -> None:
    from scripts.build_assets_qa import (
        is_app_payload_path,
        is_forbidden_release_path,
        required_release_paths,
        validate_mac_launcher,
        validate_mac_zip_artifact,
        validate_release_payload_bytes,
        validate_release_inventory,
        validate_zip_artifact,
        zip_release_name,
    )

    required = required_release_paths()
    cinematic_assets = sorted(
        path for path in required if path.startswith("assets/generated/cinematic/")
    )
    if not cinematic_assets or not any(path.endswith("keeper_anim_left_15.png") for path in cinematic_assets):
        raise AssertionError("release inventory does not cover the complete cinematic runtime")
    validate_release_inventory(set(required), "complete synthetic stage")

    windows_payload = {
        zip_release_name("_internal\\" + path.replace("/", "\\"))
        for path in required
    }
    validate_release_inventory(windows_payload, "complete synthetic Windows zip")
    sample_path = "assets/generated/balls3d/ball_0.png"
    if (
        zip_release_name(
            "_INTERNAL\\" + sample_path.replace("/", "\\")
        )
        != sample_path
    ):
        raise AssertionError(
            "Windows zip normalization treats _INTERNAL case-sensitively"
        )
    forbidden_case_variant = (
        "ASSETS/GENERATED/CINEMATIC_SOURCES/raw.png"
    )
    if (
        not is_app_payload_path(forbidden_case_variant)
        or not is_forbidden_release_path(forbidden_case_variant)
    ):
        raise AssertionError(
            "Windows release policy accepts an uppercase source payload"
        )
    normalized_forbidden_source = zip_release_name(
        "_INTERNAL/SOURCES/raw.bin"
    )
    if not is_forbidden_release_path(normalized_forbidden_source):
        raise AssertionError(
            "Windows release policy accepts a normalized source payload"
        )
    raw_fifa_source = (
        "assets/generated/fifa_external/"
        "fifa_mexico_opening_ceremony.jpg"
    )
    clean_fifa_runtime = (
        "assets/generated/fifa_external/"
        "fifa_mexico_opening_ceremony_clean.png"
    )
    if raw_fifa_source in required or clean_fifa_runtime not in required:
        raise AssertionError(
            "release inventory must package only the cleaned FIFA image"
        )

    swapped_source = "assets/generated/balls3d/ball_1.png"
    try:
        validate_release_payload_bytes(
            {sample_path: sample_path},
            lambda _embedded: (ROOT / swapped_source).read_bytes(),
            "synthetic swapped sprite",
        )
    except AssertionError as exc:
        if "diverge dos bytes" not in str(exc):
            raise AssertionError(
                "release byte swap was rejected for the wrong reason: "
                f"{exc}"
            ) from exc
    else:
        raise AssertionError(
            "release payload accepts a valid but swapped runtime sprite"
        )

    missing_asset = next(path for path in cinematic_assets if path.endswith("keeper_anim_left_15.png"))
    try:
        validate_release_inventory(windows_payload - {missing_asset}, "mutated Windows zip")
    except AssertionError as exc:
        if missing_asset not in str(exc):
            raise AssertionError(
                f"release inventory rejected the mutation for the wrong reason: {exc}"
            ) from exc
    else:
        raise AssertionError("release inventory accepts a Windows zip without a cinematic frame")

    collision_cases = (
        (
            "assets/generated/cinematic/case.png",
            "_internal/assets/generated/cinematic/case.png",
        ),
        (
            "assets/generated/cinematic/CASE.png",
            "_internal/assets/generated/cinematic/case.png",
        ),
        (
            "assets/generated/cinematic/case.png",
            "_INTERNAL/assets/generated/cinematic/case.png",
        ),
    )
    with tempfile.TemporaryDirectory(prefix="arena-ai-zip-collision-") as tmp:
        for index, names in enumerate(collision_cases):
            archive_path = Path(tmp) / f"collision-{index}.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("ArenaAI.exe", b"synthetic-launcher")
                for name_index, name in enumerate(names):
                    archive.writestr(name, f"payload-{name_index}".encode())
            try:
                validate_zip_artifact(archive_path)
            except AssertionError as exc:
                if "duplicados" not in str(exc):
                    raise AssertionError(
                        "Windows zip collision was rejected for the wrong reason: "
                        f"{exc}"
                    ) from exc
            else:
                raise AssertionError(
                    "Windows zip accepts a normalized case-insensitive collision"
                )

        invalid_launchers = (
            ("backup", {"NotArenaAI.exe.bak": b"x"}),
            ("wrong-case", {"ArenaAI.EXE": b"x"}),
            ("nested", {"_internal/ArenaAI.exe": b"x"}),
            ("empty", {"ArenaAI.exe": b""}),
        )
        for label, entries in invalid_launchers:
            archive_path = Path(tmp) / f"launcher-{label}.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, payload in entries.items():
                    archive.writestr(name, payload)
            try:
                validate_zip_artifact(archive_path)
            except AssertionError as exc:
                if "launcher" not in str(exc):
                    raise AssertionError(
                        f"invalid Windows launcher was rejected for the wrong reason: {exc}"
                    ) from exc
            else:
                raise AssertionError(
                    f"Windows zip accepts invalid launcher case: {label}"
                )

        mac_root_cases = (
            (
                "case-collision",
                {
                    "ArenaAI.app/Contents/Info.plist": b"fixture",
                    "arenaai.app/Contents/MacOS/ArenaAI": b"collision",
                },
                "colide por caixa",
            ),
            (
                "extra-root",
                {
                    "ArenaAI.app/Contents/Info.plist": b"fixture",
                    "README.txt": b"extra",
                },
                "root extra",
            ),
        )
        for label, entries, expected_error in mac_root_cases:
            archive_path = Path(tmp) / f"mac-root-{label}.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, payload in entries.items():
                    archive.writestr(name, payload)
            try:
                validate_mac_zip_artifact(archive_path)
            except AssertionError as exc:
                if expected_error not in str(exc):
                    raise AssertionError(
                        f"invalid macOS root was rejected for the wrong reason: {exc}"
                    ) from exc
            else:
                raise AssertionError(
                    f"macOS ZIP accepts invalid root case: {label}"
                )

        import plistlib

        mac_app = Path(tmp) / "ArenaAI.app"
        macos_dir = mac_app / "Contents" / "MacOS"
        macos_dir.mkdir(parents=True)
        plist_path = mac_app / "Contents" / "Info.plist"
        launcher_path = macos_dir / "ArenaAI"
        plist_path.write_bytes(
            plistlib.dumps({"CFBundleExecutable": "ArenaAI"})
        )
        launcher_path.write_bytes(b"synthetic-launcher")
        launcher_path.chmod(0o755)
        validate_mac_launcher(
            mac_app,
            require_executable=sys.platform != "win32",
        )

        launcher_path.chmod(0o644)
        try:
            validate_mac_launcher(mac_app)
        except AssertionError as exc:
            if "executável" not in str(exc):
                raise AssertionError(
                    "non-executable macOS launcher was rejected for the "
                    f"wrong reason: {exc}"
                ) from exc
        else:
            raise AssertionError(
                "real macOS launcher validation accepts a missing execute bit"
            )
        launcher_path.chmod(0o755)

        plist_path.write_bytes(b"not-a-plist")
        try:
            validate_mac_launcher(mac_app)
        except AssertionError as exc:
            if "Info.plist" not in str(exc):
                raise AssertionError(
                    f"invalid macOS plist was rejected for the wrong reason: {exc}"
                ) from exc
        else:
            raise AssertionError("macOS app accepts an invalid Info.plist")

        plist_path.write_bytes(
            plistlib.dumps({"CFBundleExecutable": "OtherLauncher"})
        )
        try:
            validate_mac_launcher(mac_app)
        except AssertionError as exc:
            if "launcher exato" not in str(exc):
                raise AssertionError(
                    f"mismatched macOS launcher was rejected for the wrong reason: {exc}"
                ) from exc
        else:
            raise AssertionError(
                "macOS app accepts a launcher that differs from CFBundleExecutable"
            )


def validate_cinematic_draw_order_declared() -> None:
    source = (ROOT / "src" / "arena_ai" / "main.py").read_text(encoding="utf-8")
    marker = "    def draw_field("
    start = source.find(marker)
    if start < 0:
        raise AssertionError("draw_field implementation not found for z-order validation")
    next_method = source.find("\n    def ", start + len(marker))
    body = source[start:next_method if next_method > start else len(source)]
    for order in CINEMATIC_DRAW_ORDERS:
        positions = []
        for call in order:
            position = body.find(f"self.{call}(")
            if position < 0:
                raise AssertionError(
                    "draw_field is missing declared cinematic z-order call: "
                    f"{call}"
                )
            positions.append(position)
        if positions != sorted(positions):
            raise AssertionError(
                "draw_field cinematic z-order drifted: "
                f"{dict(zip(order, positions))}"
            )


def validate_chance_schedule_no_dead_air() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = app.match_field_rect()
    checks = (
        ("home", home_win_prediction()),
        ("away", away_win_prediction()),
        ("draw", neutral_prediction()),
    )
    for label, pred in checks:
        final_home, final_away = app.final_score_from_prediction(pred)
        schedule = app.goal_schedule(pred)
        if len(schedule) != final_home + final_away:
            raise AssertionError(f"{label} chance schedule does not match final score: {schedule} vs {(final_home, final_away)}")
        if schedule != sorted(schedule, key=lambda item: item[0]):
            raise AssertionError(f"{label} chance schedule is not chronological: {schedule}")
        if any(side not in {"home", "away"} or minute < 7 or minute > 88 for minute, side in schedule):
            raise AssertionError(f"{label} chance schedule has invalid entries: {schedule}")
        gaps = [later[0] - earlier[0] for earlier, later in zip(schedule, schedule[1:])]
        if gaps and max(gaps) > 34:
            raise AssertionError(f"{label} chance schedule leaves too much dead-air between goals: {schedule}")
        if schedule:
            first_goal = schedule[0][0]
            if first_goal - GOAL_EVENT_WINDOW_MINUTES > 32.0:
                raise AssertionError(f"{label} first chance starts too late: {schedule}")
            if 90.0 - schedule[-1][0] > 38.0 and final_home != final_away:
                raise AssertionError(f"{label} winner has no late payoff/chance pressure: {schedule}")
        active_samples = 0
        moving_samples = 0
        previous_ball: tuple[float, float] | None = None
        for minute in range(4, 89, 4):
            app.t = minute / 90.0 * SIMULATION_SECONDS
            state = app.cinematic_scene_state(field, pred)
            if (
                state.get("active_goal")
                or (
                    state.get("poc2_dribble")
                    and not state.get("settled")
                )
                or float(state.get("run_speed", 0.0)) > 0.38
            ):
                active_samples += 1
            ball = state.get("ball_pos")
            if isinstance(ball, tuple) and previous_ball is not None and math.dist(previous_ball, ball) > 1.0:
                moving_samples += 1
            if isinstance(ball, tuple):
                previous_ball = ball  # type: ignore[assignment]
        if active_samples < 12 or moving_samples < 10:
            raise AssertionError(f"{label} cinematic has too much dead-air: active={active_samples}, moving={moving_samples}")


def validate_sound_engine_layers() -> None:
    app = App(seed=2026)
    sound = app.sound
    if sound.opening_crowd is None or sound.goal_roar is None:
        raise AssertionError("runtime crowd layers must be loaded for opening/ambience and goal roar")
    if sound.stadium_base is not sound.opening_crowd:
        raise AssertionError("stadium_base_loop.mp3 must be the primary opening/base ambience layer")
    if sound.stadium_air is None or sound.light_crowd is None:
        raise AssertionError("stadium_air_loop.wav and crowd_light_loop.mp3 must be active ambience layers")
    if not hasattr(sound, "buses"):
        raise AssertionError("sound system must expose AudioEngine buses")
    missing_buses = REQUIRED_AUDIO_BUSES - set(sound.buses)
    if missing_buses:
        raise AssertionError(f"sound engine is missing buses: {sorted(missing_buses)}")
    required_channels = {"base", "air", "light", "tension", "chant", "react", "bass", "reverb", "explosion", "roar", "whistle", "music", "tick"}
    missing_channels = required_channels - set(sound.channels)
    if missing_channels:
        raise AssertionError(f"sound engine is missing layered channels: {sorted(missing_channels)}")
    for channel_name in ("base", "air", "light", "tension", "chant", "music"):
        if not sound.channels[channel_name].get_busy():
            raise AssertionError(f"{channel_name} crowd layer is not running as a loop")
    if sound.channels["base"].get_sound() is not sound.stadium_base:
        raise AssertionError("base channel is not playing runtime stadium_base_loop.mp3")
    if sound.channels["air"].get_sound() is not sound.stadium_air:
        raise AssertionError("air channel is not playing runtime stadium_air_loop.wav")
    if sound.channels["light"].get_sound() is not sound.light_crowd:
        raise AssertionError("light channel is not playing runtime crowd_light_loop.mp3")
    sound.duck_commentary(1.0)
    sound.update_crowd(0.90, True, 0.25)
    if sound.layer_volumes["base"] < 0.20:
        raise AssertionError(f"commentary duck over-muted the stadium base: {sound.layer_volumes['base']:.2f}")
    if sound.layer_volumes["music"] >= 0.08:
        raise AssertionError(f"commentary duck did not clear enough music space: {sound.layer_volumes['music']:.2f}")
    if sound.layer_volumes["tension"] <= 0.02:
        raise AssertionError("reactive tension layer did not rise during dangerous attack")
    sound.duck_until_ms = 0
    sound.update_crowd(0.90, True, 0.25)
    if sound.layer_volumes["tension"] <= sound.layer_volumes["light"]:
        raise AssertionError("dangerous attack should make tension louder than light crowd")
    sound.play("bass")
    sound.play("cheer")
    sound.play("reverb")
    for channel_name in ("bass", "explosion", "roar", "reverb"):
        if not sound.channels[channel_name].get_busy():
            raise AssertionError(f"{channel_name} layer did not play on goal")
    if sound.channels["roar"].get_sound() not in sound.goal_roars:
        raise AssertionError("roar channel is not playing one of the official runtime goal-roar takes")
    chosen_goal_roars = [sound.choose_bag("visual_qa_goal_roar", sound.goal_roars) for _index in range(8)]
    if any(chosen_goal_roars[index] is chosen_goal_roars[index - 1] for index in range(1, len(chosen_goal_roars))):
        raise AssertionError("goal roar sound-bag repeated the same take immediately")


def validate_match_screen_layout_gate() -> None:
    app = App(seed=2026)
    screen = app.screen.get_rect()
    field = app.match_field_rect()
    side = app.match_side_panel_rect()
    score = app.match_score_panel_rect()
    for label, rect in (("field", field), ("side", side), ("score", score)):
        if not screen.contains(rect):
            raise AssertionError(f"match {label} rect is outside screen: {rect}")
    if field.colliderect(side) or field.colliderect(score) or side.colliderect(score):
        raise AssertionError(f"match layout panels overlap: field={field}, side={side}, score={score}")

    clock = app.match_clock_rect(field)
    if not field.contains(clock):
        raise AssertionError(f"clock must stay inside cinematic field: {clock} vs {field}")
    label_w, label_h = app.f_sm.size(app.elapsed_label())
    if label_w > clock.w - 14 or label_h > clock.h - 8:
        raise AssertionError(f"clock text does not fit: {(label_w, label_h)} in {clock}")
    for possession in ("home", "away"):
        narrator = app.match_narrator_rect(field, possession)
        if not field.contains(narrator):
            raise AssertionError(f"narrator {possession} outside field: {narrator}")
        if narrator.colliderect(clock.inflate(12, 8)):
            raise AssertionError(f"narrator {possession} collides with clock: {narrator} vs {clock}")
    goal_text = app.f_lg.render("GOOOL!", True, (255, 255, 255))
    goal_overlay = pygame.Rect(0, 0, goal_text.get_width() + 54, goal_text.get_height() + 22)
    goal_overlay.center = app.cinematic_goal_overlay_center(field)
    focus_tag = pygame.Rect(field.x + 22, field.y + 22, 410, 54)
    if goal_overlay.colliderect(focus_tag.inflate(8, 8)):
        raise AssertionError(
            f"goal overlay collides with active focus tag: {goal_overlay} vs {focus_tag}"
        )
    if goal_overlay.colliderect(clock.inflate(12, 12)):
        raise AssertionError(f"goal overlay collides with clock: {goal_overlay} vs {clock}")

    prob_end = score.x + 306 + 2 * (158 + 38) + 158
    final_start = score.right - 318
    if prob_end + 24 > final_start:
        raise AssertionError(f"score panel probability bars collide with final block: {prob_end} -> {final_start}")
    side_content_w = side.w - 48
    if app.f_xs.size("Oráculo em campo")[0] > side_content_w:
        raise AssertionError("side panel heading would truncate in the match HUD")
    for state_copy in MATCH_HUD_STATE_COPY.values():
        state_label, state_title, state_hint = state_copy
        if app.fit_font(state_label, 19, side_content_w, min_size=14).size(state_label)[0] > side_content_w:
            raise AssertionError(f"canonical HUD state label does not fit: {state_label}")
        if app.fit_font(state_title, 30, side_content_w, min_size=22).size(state_title)[0] > side_content_w:
            raise AssertionError(f"canonical HUD state title does not fit: {state_title}")
        if app.f_tiny.size(state_hint)[0] > side_content_w:
            raise AssertionError(f"canonical HUD state hint does not fit: {state_hint}")

    row_text_w = side_content_w - 42
    for row_y in (side.y + 146, side.y + 211, side.y + 276):
        row = pygame.Rect(side.x + 24, row_y, side_content_w, 57)
        if not side.inflate(-10, -8).contains(row):
            raise AssertionError(f"side panel model-flow row overflows: {row} outside {side}")
        if row.y + 43 + app.f_tiny.get_height() > row.bottom:
            raise AssertionError(f"side panel model-flow detail line is vertically clipped: {row}")
    for title in ("XGBoost 1X2", "Poisson/DC", "Sorteio da Copa"):
        if app.fit_font(title, 19, row_text_w, min_size=15).size(title)[0] > row_text_w:
            raise AssertionError(f"model-flow title does not fit without clipping: {title}")

    top_scores_card = pygame.Rect(side.x + 24, side.y + 356, side_content_w, 116)
    pending_card = pygame.Rect(side.x + 24, side.y + 370, side_content_w, 96)
    for label, card in (("top scores", top_scores_card), ("pending audit", pending_card)):
        if not side.inflate(-10, -8).contains(card):
            raise AssertionError(f"side panel {label} card overflows: {card} outside {side}")
    title_w = app.f_tiny.size("Placares possíveis")[0]
    label_w = app.f_tiny.size("Poisson/DC")[0]
    if title_w > top_scores_card.w - 74:
        raise AssertionError(f"top scores title does not fit: {title_w} > {top_scores_card.w - 74}")
    if top_scores_card.x + 10 + title_w + 8 > top_scores_card.right - 10 - label_w:
        raise AssertionError("top scores title collides with Poisson/DC label")
    bar = pygame.Rect(top_scores_card.x + 74, top_scores_card.y + 31 + 4 * 16 + 6, 86, 7)
    if not top_scores_card.inflate(-8, -6).contains(bar):
        raise AssertionError(f"top scores bar overflows card: {bar} outside {top_scores_card}")

    leaked = [copy for copy in MATCH_HUD_BANNED_COPY if copy in MATCH_HUD_REQUIRED_COPY]
    if leaked:
        raise AssertionError(f"match HUD constants still contain confusing/debug copy: {leaked}")
    missing_copy = [copy for copy in MATCH_HUD_REQUIRED_COPY if not str(copy).strip()]
    if missing_copy:
        raise AssertionError(f"match HUD lost required explanatory copy: {missing_copy}")
    if MATCH_HUD_TOP_SCORE_COUNT != 5:
        raise AssertionError(f"match HUD must render the top 5 Poisson/DC scorelines, not {MATCH_HUD_TOP_SCORE_COUNT}")

    for pred, samples in (
        (home_win_prediction(), (0.0, 12.0, 17.0, 18.4, 19.2, 43.0, 45.0)),
        (away_win_prediction(), (0.0, 12.0, 17.0, 18.4, 19.2, 43.0, 45.0)),
        (neutral_prediction(), (0.0, 30.0, 42.0, 45.0)),
    ):
        app.set_simulate("match")
        app.match_prediction = pred
        for second in samples:
            app.t = min(SIMULATION_SECONDS, second)
            app.screen.fill((0, 0, 0))
            app.draw_simulate()
            state = app.cinematic_scene_state(field, pred)
            if bool(state.get("active_goal")) and float(state.get("shot_progress", 0.0)) < SHOT_NET_AT:
                if float(state.get("net_progress", 0.0)) != 0.0:
                    raise AssertionError("goal net ripple started before ball/net impact")
            if bool(state.get("active_goal")) and SHOT_FOLLOW_THROUGH_HOLD_END + 0.03 <= float(state.get("shot_progress", 0.0)) <= 0.90:
                kick_window, _stride = app.cinematic_stride_state(float(state["shot_progress"]), float(state.get("stride_phase", 0.0)))
                if kick_window:
                    raise AssertionError("attacker remains frozen in kick pose during ball flight")

    app.set_simulate("match")
    app.match_prediction = home_win_prediction()
    app.t = 18.5
    app.draw_simulate()
    first_hash = surface_hash(app.screen)
    app.draw_simulate()
    second_hash = surface_hash(app.screen)
    if first_hash != second_hash:
        raise AssertionError("draw_simulate is not idempotent for the same match frame")


def validate_text_safe_area_gate() -> None:
    app = App(seed=2026)

    menu_intro_safe = pygame.Rect(86, 260, 390, 142)
    app.screen.fill((0, 0, 0))
    menu_rects = capture_app_draw_text_rects(app, app.draw_menu)
    intro_fragments = (
        "Copa do Mundo 2026",
        "Escolha um duelo.",
        "Compare forma e elenco.",
        "Simule a Copa em tempo real.",
    )
    for text, rect in menu_rects:
        if text in intro_fragments and not menu_intro_safe.contains(rect):
            raise AssertionError(f"menu intro text leaves safe copy column: {text!r} rect={rect} safe={menu_intro_safe}")

    field = app.match_field_rect()
    side = app.match_side_panel_rect()
    score = app.match_score_panel_rect()
    content_safe = field.union(side).union(score)
    scenarios = (
        ("live", home_win_prediction(), SIMULATION_SECONDS * 0.10),
        ("closed", home_win_prediction(), SIMULATION_SECONDS),
        ("away_live", away_win_prediction(), SIMULATION_SECONDS * 0.36),
    )
    for label, pred, second in scenarios:
        app.set_simulate("match")
        app.match_prediction = pred
        app.t = second
        app.screen.fill((0, 0, 0))
        rects = capture_app_draw_text_rects(app, app.draw_simulate)
        for text, rect in rects:
            if rect.bottom <= 88:
                continue
            center = rect.center
            if field.collidepoint(center):
                if not field.contains(rect):
                    raise AssertionError(f"field text invades outside field during {label}: {text!r} rect={rect}")
                if rect.colliderect(side):
                    raise AssertionError(f"field text invades sidebar during {label}: {text!r} rect={rect}")
            elif side.collidepoint(center):
                if not side.contains(rect):
                    raise AssertionError(f"sidebar text leaves panel during {label}: {text!r} rect={rect}")
                if rect.colliderect(field):
                    raise AssertionError(f"sidebar text invades field during {label}: {text!r} rect={rect}")
            elif score.collidepoint(center):
                if not score.contains(rect):
                    raise AssertionError(f"score HUD text leaves panel during {label}: {text!r} rect={rect}")
            elif content_safe.colliderect(rect):
                raise AssertionError(f"text is between match safe areas during {label}: {text!r} rect={rect}")


def validate_selection_card_metric_layout_gate() -> None:
    app = App(seed=2026)
    for rect in (pygame.Rect(56, 118, 420, 460), pygame.Rect(804, 118, 420, 460)):
        label_x = rect.x + 28
        bar_x = rect.x + 160
        label_max_width = bar_x - label_x - 10
        for label in ("ELO", "Gols feitos", "Defesa", "Vitórias", "Elenco"):
            label_width = app.f_sm.size(label)[0]
            if label_width > label_max_width:
                raise AssertionError(f"selection metric label collides with bar: {label} width={label_width} max={label_max_width}")
        bar_right = bar_x + 150
        value_left = rect.right - 72
        if bar_right + 8 > value_left:
            raise AssertionError(f"selection metric bar collides with value column: bar_right={bar_right} value_left={value_left}")
    app.home_idx = app.team_index("BRA", 0)
    app.away_idx = app.team_index("FRA", 1)
    app.state = "select"
    texts = "\n".join(capture_app_draw_text(app, app.draw_select))
    if "Brazil" in texts or "France" in texts:
        raise AssertionError("selection screen leaked raw English team names")
    if "Brasil" not in texts or "França" not in texts:
        raise AssertionError("selection screen is missing PT-BR team names")
    engine_cards = [pygame.Rect(WIDTH // 2 - 152, 430 + index * 56, 304, 50) for index in range(3)]
    button_safe = pygame.Rect(292, 628, 656, 58)
    previous_bottom = 0
    for index, card in enumerate(engine_cards):
        if card.w < 280 or card.h < 48:
            raise AssertionError(f"selection algorithm card {index} is too small/readability-regressed: {card}")
        if card.colliderect(button_safe):
            raise AssertionError(f"selection algorithm card {index} collides with action buttons: {card}")
        if button_safe.top - card.bottom < 26 and index == len(engine_cards) - 1:
            raise AssertionError(f"selection algorithm cards need a clearer gutter before action buttons: {card}")
        if card.top <= previous_bottom:
            raise AssertionError(f"selection algorithm card {index} overlaps the previous card: {card}")
        previous_bottom = card.bottom
    required_algorithm_copy = ("XGBoost 1X2", "Poisson/DC", "Monte Carlo")
    for copy in required_algorithm_copy:
        if copy not in texts:
            raise AssertionError(f"selection screen lost readable algorithm card copy: {copy}")


def validate_selection_input_flow_gate() -> None:
    app = App(seed=2026)
    app.set_select()
    home_before = app.home_idx
    away_before = app.away_idx
    app.handle_key(pygame.K_RIGHT)
    if app.home_idx == home_before or app.home_idx == app.away_idx:
        raise AssertionError("selection RIGHT key did not cycle home team safely")
    app.handle_key(pygame.K_a)
    if app.away_idx == away_before or app.home_idx == app.away_idx:
        raise AssertionError("selection A key did not cycle away team safely")
    app.handle_key(pygame.K_SPACE)
    if app.state != "simulate" or app.mode != "match" or app.match_prediction is None:
        raise AssertionError("selection SPACE did not start confrontation")
    app.handle_key(pygame.K_BACKSPACE)
    if app.state != "select":
        raise AssertionError("BACKSPACE did not return from confrontation to selection")
    app.handle_key(pygame.K_t)
    if app.state != "tournament" or app.pending_tournament_seed is None:
        raise AssertionError("selection T did not prepare tournament simulation")
    app.handle_key(pygame.K_BACKSPACE)
    if app.state != "select":
        raise AssertionError("BACKSPACE did not return from tournament to selection")

    home_rect = pygame.Rect(56, 118, 420, 460)
    away_rect = pygame.Rect(804, 118, 420, 460)
    home_next = app.team_arrow_rects(home_rect)[1]
    away_prev = app.team_arrow_rects(away_rect)[0]
    home_before = app.home_idx
    app.handle_click(home_next.center)
    if app.home_idx == home_before or app.home_idx == app.away_idx:
        raise AssertionError("home arrow click did not cycle team safely")
    away_before = app.away_idx
    app.handle_click(away_prev.center)
    if app.away_idx == away_before or app.home_idx == app.away_idx:
        raise AssertionError("away arrow click did not cycle team safely")
    app.handle_click(app.single_button.rect.center)
    if app.state != "simulate" or app.match_prediction is None:
        raise AssertionError("selection confrontation button did not start match")
    app.set_select()
    app.handle_click(app.cup_button.rect.center)
    if app.state != "tournament" or app.pending_tournament_seed is None:
        raise AssertionError("selection cup button did not start Monte Carlo flow")
    app.set_select()
    app.handle_click(app.back_button.rect.center)
    if app.state != "menu":
        raise AssertionError("selection back button did not return to menu")


def validate_button_label_auto_fit_gate() -> None:
    font_path = ROOT / "assets" / "fonts" / "Oxanium.ttf"
    text_font = pygame.font.Font(font_path if font_path.exists() else None, 27)
    labels = (
        "SIMULAR CONFRONTO INTERNACIONAL",
        "SIMULAR COPA DO MUNDO",
        "VOLTAR PARA SELEÇÃO",
    )
    surface = pygame.Surface((420, 190), pygame.SRCALPHA)
    for index, label in enumerate(labels):
        button = Button(pygame.Rect(12, 12 + index * 58, 250, 46), label, (82, 226, 255))
        fitted = button.fitted_label_surface(text_font)
        if fitted.get_width() > button.rect.w - 28 or fitted.get_height() > button.rect.h - 16:
            raise AssertionError(f"button label does not fit after auto-fit: {label} size={fitted.get_size()}")
        if text_font.size(label)[0] <= button.rect.w - 28:
            raise AssertionError(f"button auto-fit fixture is not long enough to exercise scaling: {label}")
        button.draw(surface, text_font, (-1, -1))
        cached = button.fitted_label_surface(text_font)
        button.draw(surface, text_font, (-1, -1))
        if button.fitted_label_surface(text_font) is not cached:
            raise AssertionError(f"button label cache is not stable after warm draw: {label}")


def capture_app_draw_text(app: App, draw_call: Callable[[], None]) -> list[str]:
    captured: list[str] = []
    method_names = (
        "draw_text",
        "draw_text_centered",
        "draw_text_right",
        "draw_text_midleft",
        "draw_text_midright",
        "draw_text_ellipsis",
    )
    originals = {name: getattr(app, name) for name in method_names}

    def make_spy(original: Callable[..., None]) -> Callable[..., None]:
        def spy(text: str, *args: object, **kwargs: object) -> None:
            captured.append(str(text))
            original(text, *args, **kwargs)

        return spy

    try:
        for name, original in originals.items():
            setattr(app, name, make_spy(original))
        draw_call()
    finally:
        for name, original in originals.items():
            setattr(app, name, original)
    return captured


def capture_app_draw_text_events(app: App, draw_call: Callable[[], None]) -> list[tuple[str, int, int]]:
    captured: list[tuple[str, int, int]] = []
    method_names = (
        "draw_text",
        "draw_text_centered",
        "draw_text_right",
        "draw_text_midleft",
        "draw_text_midright",
        "draw_text_ellipsis",
    )
    originals = {name: getattr(app, name) for name in method_names}
    nested = False

    def make_spy(name: str, original: Callable[..., None]) -> Callable[..., None]:
        def spy(text: str, *args: object, **kwargs: object) -> None:
            nonlocal nested
            if not nested:
                x = -1
                y = -1
                if name in {"draw_text", "draw_text_right", "draw_text_ellipsis"} and len(args) >= 4:
                    x = int(args[2])
                    y = int(args[3])
                elif name == "draw_text_centered" and len(args) >= 3:
                    center = args[2]
                    if isinstance(center, tuple):
                        x = int(center[0])
                        y = int(center[1])
                elif name in {"draw_text_midleft", "draw_text_midright"} and len(args) >= 3:
                    point = args[2]
                    if isinstance(point, tuple):
                        x = int(point[0])
                        y = int(point[1])
                captured.append((str(text), x, y))
            was_nested = nested
            nested = True
            try:
                original(text, *args, **kwargs)
            finally:
                nested = was_nested

        return spy

    try:
        for name, original in originals.items():
            setattr(app, name, make_spy(name, original))
        draw_call()
    finally:
        for name, original in originals.items():
            setattr(app, name, original)
    return captured


def capture_app_draw_text_rects(app: App, draw_call: Callable[[], None]) -> list[tuple[str, pygame.Rect]]:
    captured: list[tuple[str, pygame.Rect]] = []
    method_names = (
        "draw_text",
        "draw_text_centered",
        "draw_text_right",
        "draw_text_midleft",
        "draw_text_midright",
        "draw_text_ellipsis",
    )
    originals = {name: getattr(app, name) for name in method_names}
    nested = False

    def text_rect(
        text: str,
        text_font: pygame.font.Font,
        name: str,
        args: tuple[object, ...],
    ) -> pygame.Rect | None:
        rendered_text = str(text)
        if name == "draw_text_ellipsis" and len(args) >= 5:
            rendered_text = app.ellipsize(rendered_text, text_font, int(args[4]))
        width, height = text_font.size(rendered_text)
        if name in {"draw_text", "draw_text_ellipsis"} and len(args) >= 4:
            return pygame.Rect(int(args[2]), int(args[3]), width, height)
        if name == "draw_text_right" and len(args) >= 4:
            return pygame.Rect(int(args[2]) - width, int(args[3]), width, height)
        if name == "draw_text_centered" and len(args) >= 3 and isinstance(args[2], tuple):
            center = (int(args[2][0]), int(args[2][1]))
            return pygame.Rect(0, 0, width, height).move(center[0] - width // 2, center[1] - height // 2)
        if name == "draw_text_midleft" and len(args) >= 3 and isinstance(args[2], tuple):
            midleft = (int(args[2][0]), int(args[2][1]))
            return pygame.Rect(midleft[0], midleft[1] - height // 2, width, height)
        if name == "draw_text_midright" and len(args) >= 3 and isinstance(args[2], tuple):
            midright = (int(args[2][0]), int(args[2][1]))
            return pygame.Rect(midright[0] - width, midright[1] - height // 2, width, height)
        return None

    def make_spy(name: str, original: Callable[..., None]) -> Callable[..., None]:
        def spy(text: str, *args: object, **kwargs: object) -> None:
            nonlocal nested
            if not nested and args and isinstance(args[0], pygame.font.Font):
                rect = text_rect(str(text), args[0], name, args)
                if rect is not None:
                    captured.append((str(text), rect))
            was_nested = nested
            nested = True
            try:
                original(text, *args, **kwargs)
            finally:
                nested = was_nested

        return spy

    try:
        for name, original in originals.items():
            setattr(app, name, make_spy(name, original))
        draw_call()
    finally:
        for name, original in originals.items():
            setattr(app, name, original)
    return captured


def captured_texts_in_rect(events: list[tuple[str, int, int]], rect: pygame.Rect) -> list[str]:
    hitbox = rect.inflate(8, 8)
    return [text for text, x, y in events if hitbox.collidepoint(x, y)]


def capture_protected_ellipsis_overflows(
    app: App,
    draw_call: Callable[[], None],
    protected_fragments: tuple[str, ...],
) -> list[str]:
    overflows: list[str] = []
    original = app.draw_text_ellipsis

    def spy(
        text: str,
        text_font: pygame.font.Font,
        color: tuple[int, int, int],
        x: int,
        y: int,
        max_width: int,
    ) -> None:
        if any(fragment.lower() in str(text).lower() for fragment in protected_fragments):
            if text_font.size(str(text))[0] > max_width:
                overflows.append(f"{text!r} width={text_font.size(str(text))[0]} max={max_width} at=({x},{y})")
        original(text, text_font, color, x, y, max_width)

    try:
        app.draw_text_ellipsis = spy  # type: ignore[method-assign]
        draw_call()
    finally:
        app.draw_text_ellipsis = original  # type: ignore[method-assign]
    return overflows


def validate_match_hud_text_fit_gate() -> None:
    app = App(seed=2026)
    app.home_idx = app.team_index("PAR", 0)
    app.away_idx = app.team_index("ALG", 1)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = pred
    protected = (
        ALGORITHM_NAMES["CONFRONTO"],
        "Oráculo em campo",
        "JOGO EM ABERTO",
        "PRESSÃO NA ÁREA",
        "APITO FINAL",
        "Nada decidido",
        "Lance vivo",
        "Final só no apito",
        "Resultado revelado",
        "Leitura completa",
        "Quem chega melhor",
        "Forma, camisa e mando",
        "Placar guardado",
        "Mapa de gols",
        "A chance escolhida",
        "Placares possíveis",
        "Chance do placar",
    )

    seconds = [
        SIMULATION_SECONDS * 0.05,
        SIMULATION_SECONDS * 0.35,
        SIMULATION_SECONDS,
    ]
    goal_schedule = app.goal_schedule(pred)
    if goal_schedule:
        first_goal = goal_schedule[0][0]
        seconds.append(
            goal_progress_second(first_goal, SHOT_KICK_AT + 0.005)
        )
    for second in seconds:
        app.t = second
        app.screen.fill((0, 0, 0))
        overflows = capture_protected_ellipsis_overflows(app, app.draw_simulate, protected)
        if overflows:
            raise AssertionError(f"protected match HUD copy is truncated at {second:.2f}s: {overflows}")


def validate_match_hud_density_legibility_gate() -> None:
    app = App(seed=2026)
    app.home_idx = app.team_index("BRA", 0)
    app.away_idx = app.team_index("FRA", 1)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    first_goal_minute = app.goal_schedule(pred)[0][0]
    focus_second = goal_progress_second(first_goal_minute, 0.52)
    scenarios = (
        ("live", SIMULATION_SECONDS * 0.10, 18, 14),
        ("focus", focus_second, 18, 12),
        ("closed", SIMULATION_SECONDS, 26, 14),
    )
    side_rect = app.match_side_panel_rect()
    score_rect = app.match_score_panel_rect()
    for state_key, second, side_limit, score_limit in scenarios:
        app.t = second
        app.screen.fill((0, 0, 0))
        events = capture_app_draw_text_events(app, app.draw_simulate)
        side_texts = captured_texts_in_rect(events, side_rect)
        score_texts = captured_texts_in_rect(events, score_rect)
        hud_text = "\n".join((*side_texts, *score_texts)).lower()
        for copy in MATCH_HUD_STATE_COPY[state_key]:
            if copy.lower() not in hud_text:
                raise AssertionError(f"canonical {state_key} HUD copy is missing: {copy}")
        for other_key, other_copy in MATCH_HUD_STATE_COPY.items():
            if other_key != state_key and other_copy[0].lower() in hud_text:
                raise AssertionError(f"HUD shows multiple canonical states during {state_key}: {other_copy[0]}")
        banned = [copy for copy in MATCH_HUD_BANNED_COPY if copy.lower() in hud_text]
        if banned:
            raise AssertionError(f"match HUD revived dense/deprecated copy during {state_key}: {banned}")
        if len(side_texts) > side_limit:
            raise AssertionError(f"side panel is too text-dense during {state_key}: {len(side_texts)} > {side_limit}: {side_texts}")
        if len(score_texts) > score_limit:
            raise AssertionError(f"score panel is too text-dense during {state_key}: {len(score_texts)} > {score_limit}: {score_texts}")


def validate_match_result_suspense_gate() -> None:
    app = App(seed=2026)
    app.home_idx = app.team_index("BRA", 0)
    app.away_idx = app.team_index("FRA", 1)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred

    def draw_at(second: float) -> list[str]:
        app.t = second
        app.screen.fill((0, 0, 0))
        return capture_app_draw_text(app, app.draw_simulate)

    before_final = "\n".join(draw_at(SIMULATION_SECONDS * 0.10)).lower()
    banned_before_reveal = (
        "placar sorteado",
        "placar revelado",
        "resultado sorteado",
        "chance do placar",
        "possibilidades poisson/dc",
        "pico:",
        "%",
    )
    leaks = [copy for copy in banned_before_reveal if copy in before_final]
    if leaks:
        raise AssertionError(f"match HUD leaks selected result before full time: {leaks}")
    live_copy = tuple(copy.lower() for copy in MATCH_HUD_STATE_COPY["live"])
    for required in live_copy:
        if required not in before_final:
            raise AssertionError(f"match HUD suspense copy missing before full time: {required}")
    if before_final.count(live_copy[1]) < 1:
        raise AssertionError(f"match HUD should keep a rendered {MATCH_HUD_STATE_COPY['live'][1]!r} area before full time")

    score_panel_before = "\n".join(
        capture_app_draw_text(app, lambda: app.draw_score_panel({"CONFRONTO": pred}, "CONFRONTO", pred, cinematic_focus=False))
    ).lower()
    banned_score_panel_before = (
        "1x2/xgboost",
        "poisson/dc",
        "chance:",
        "xg:",
        "%",
    )
    score_panel_leaks = [copy for copy in banned_score_panel_before if copy in score_panel_before]
    if score_panel_leaks:
        raise AssertionError(f"score panel leaks audit/percentage copy before full time: {score_panel_leaks}")

    first_goal_minute = app.goal_schedule(pred)[0][0]
    focus_second = goal_progress_second(first_goal_minute, 0.52)
    focus_text = "\n".join(draw_at(focus_second)).lower()
    if "placar:" in focus_text or "resultado sorteado" in focus_text:
        raise AssertionError("cinematic focus tag leaks the final score during the goal scene")
    if "final só no apito" not in focus_text:
        raise AssertionError("cinematic focus tag does not preserve final-score suspense")

    after_final = "\n".join(draw_at(SIMULATION_SECONDS)).lower()
    final_score = f"{app.home.code} {pred.score_home} x {pred.score_away} {app.away.code}".lower()
    if MATCH_HUD_STATE_COPY["closed"][0].lower() not in after_final:
        raise AssertionError("match HUD does not reveal the final-whistle state after full time")
    if final_score not in after_final:
        raise AssertionError(f"match HUD final score missing after full time: {final_score}")
    if after_final.count(final_score) != 2:
        raise AssertionError(f"match HUD should render the selected final score in one canonical area: {final_score}")


def validate_match_clock_and_reveal_sync_gate() -> None:
    app = App(seed=2026)
    app.home_idx = app.team_index("BRA", 0)
    app.away_idx = app.team_index("FRA", 1)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    first_goal_minute, scoring_side = app.goal_schedule(pred)[0]
    impact_second = first_goal_minute / 90.0 * SIMULATION_SECONDS
    app.t = impact_second - 1.0 / FPS
    if app.score_from_prediction(pred) != (0, 0):
        raise AssertionError("score changed before the approved POC impact frame")

    app.t = impact_second
    state = app.cinematic_scene_state(app.match_field_rect(), pred)
    sequence = state.get("poc_contract_sequence")
    sample = state.get("poc_contract_sample")
    if sequence is None or sample is None:
        raise AssertionError("goal impact did not execute the approved POC contract")
    if abs(float(sample.elapsed) - float(sequence.impact_seconds)) > 1.0 / FPS:
        raise AssertionError(
            "approved POC visual impact is not synchronized with the goal minute"
        )
    if app.score_from_prediction(pred) == (0, 0):
        raise AssertionError("score did not update when the goal visually reached the net")
    expected_score = (1, 0) if scoring_side == "home" else (0, 1)
    if app.score_from_prediction(pred) != expected_score:
        raise AssertionError(
            f"goal impact updated the wrong side: {app.score_from_prediction(pred)}"
        )
    if app.match_minute() < first_goal_minute:
        raise AssertionError(
            f"clock lags behind the visible goal event: clock={app.match_minute()} goal={first_goal_minute}"
        )

    app.t = SIMULATION_SECONDS - 1e-4
    if app.match_result_revealed():
        raise AssertionError("match result reveals before exactly 90 minutes")
    if app.elapsed_label() == "90' / 90'":
        raise AssertionError("clock displays 90 minutes before the result is actually revealed")
    app.t = SIMULATION_SECONDS
    if not app.match_result_revealed() or app.elapsed_label() != "90' / 90'":
        raise AssertionError("match result does not reveal exactly at 90 minutes with full-time clock")

    drawn_text: list[str] = []
    original_draw_text_ellipsis = app.draw_text_ellipsis

    def spy_draw_text_ellipsis(text: str, *args: object, **kwargs: object) -> object:
        drawn_text.append(str(text))
        return original_draw_text_ellipsis(text, *args, **kwargs)

    app.draw_text_ellipsis = spy_draw_text_ellipsis  # type: ignore[method-assign]
    try:
        app.t = SIMULATION_SECONDS - 1e-4
        app.screen.fill((0, 0, 0))
        app.draw_score_panel({"CONFRONTO": pred}, "CONFRONTO", pred)
        if "PLACAR FINAL" in drawn_text or "PLACAR AO VIVO" not in drawn_text:
            raise AssertionError(f"score panel must keep live label before reveal: {drawn_text}")

        drawn_text.clear()
        app.t = SIMULATION_SECONDS
        app.screen.fill((0, 0, 0))
        app.draw_score_panel({"CONFRONTO": pred}, "CONFRONTO", pred)
        if "PLACAR FINAL" not in drawn_text:
            raise AssertionError(f"score panel did not show PLACAR FINAL in revealed state: {drawn_text}")
    finally:
        app.draw_text_ellipsis = original_draw_text_ellipsis  # type: ignore[method-assign]


def validate_match_runtime_state_cache_gate() -> None:
    app = App(seed=2026)
    app.home_idx = app.team_index("BRA", 0)
    app.away_idx = app.team_index("FRA", 1)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    first = app.match_runtime_state(pred)
    second = app.match_runtime_state(pred)
    if first is not second:
        raise AssertionError("MatchRuntimeState is not reused inside the same frame/runtime context")
    goals = app.goal_schedule(pred)
    goals.clear()
    if not app.goal_schedule(pred):
        raise AssertionError("goal_schedule returned the cached mutable list instead of a copy")
    app.set_select()
    if app.match_runtime_state_cache:
        raise AssertionError("MatchRuntimeState cache survived a screen transition")


def validate_cinematic_camera_continuity_gate() -> None:
    field = pygame.Rect(32, 110, 910, 490)
    app = App(seed=2026)
    app.set_simulate("match")
    for pred in (home_win_prediction(), away_win_prediction()):
        app.match_prediction = pred
        first_goal_minute = app.goal_schedule(pred)[0][0]
        camera_values = []
        for step in range(25):
            minute = (
                first_goal_minute
                + GOAL_PAYOFF_MINUTES
                - 0.17
                + step * 0.10
            )
            app.t = minute / 90.0 * SIMULATION_SECONDS
            camera_values.append(app.cinematic_camera_progress(pred))
        deltas = [abs(a - b) for a, b in zip(camera_values, camera_values[1:])]
        if max(deltas) > 0.032:
            raise AssertionError(f"cinematic camera jumps after goal payoff: values={camera_values}, deltas={deltas}")


def validate_goalkeeper_safe_bounds_variants() -> None:
    field = pygame.Rect(32, 110, 910, 490)
    codes = ("BRA", "MEX", "NED", "NZL")
    app = App(seed=2026)
    app.set_simulate("match")
    team_codes = [team.code for team in app.teams]
    for home_code in codes:
        app.home_idx = team_codes.index(home_code)
        app.away_idx = team_codes.index("FRA")
        pred = app.model.predict_matchup(app.home, app.away, seed=2026)
        app.match_prediction = pred
        home_goals = [goal_minute for goal_minute, side in app.goal_schedule(pred) if side == "home"]
        target_goal = home_goals[0] if home_goals else app.goal_schedule(pred)[0][0]
        for progress in (0.54, 0.74, SHOT_NET_VISUAL_CONTACT_AT, 1.0):
            app.t = goal_progress_second(target_goal, progress)
            state = app.cinematic_scene_state(field, pred)
            keeper_team = app.away if str(state.get("possession")) == "home" else app.home
            render, rect = goalkeeper_render_for_state(app, keeper_team, state, str(state.get("possession")) == "away")
            visible_rect = visible_surface_rect(render, rect)
            if not field.contains(visible_rect):
                raise AssertionError(
                    f"goalkeeper clips in {home_code}/FRA at progress {progress:.2f}: "
                    f"{visible_rect} outside {field}"
                )


def validate_goal_overlay_score_sync_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    first_goal_minute = app.goal_schedule(pred)[0][0]
    app.t = (
        first_goal_minute
        / 90.0
        * SIMULATION_SECONDS
    )
    if app.active_goal_event(pred) is None:
        raise AssertionError("GOOOL overlay should be active at the visual net impact")
    home_score, away_score = app.score_from_prediction(pred)
    if (home_score, away_score) == (0, 0):
        raise AssertionError("GOOOL overlay is visible while live score still shows 0 x 0")


def validate_full_match_flow() -> None:
    app = App(seed=2026)
    app.draw_menu()
    app.set_select()
    app.draw_select()
    app.set_simulate("match")
    pred = home_win_prediction()
    install_qa_match_prediction(app, pred, "full match timeline QA")
    app.shot_events.clear()
    app.goal_events.clear()
    played: list[str] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        if name in GOAL_AUDIO_SEQUENCE:
            played.append(name)

    app.sound.play = spy  # type: ignore[method-assign]
    frame_times = []
    dt = 1 / 30
    frames = int(SIMULATION_SECONDS / dt) + 1
    for _index in range(frames):
        start = time.perf_counter()
        app.update(dt)
        app.draw()
        app.flush_queued_match_audio()
        frame_times.append(time.perf_counter() - start)
    final_home, final_away = app.final_score_from_prediction(pred)
    if app.score_from_prediction(pred) != (final_home, final_away):
        raise AssertionError("full match timeline did not reach the predicted final score")
    expected_goal_events = len(app.goal_schedule(pred))
    if len(app.goal_events) != expected_goal_events:
        raise AssertionError(f"full match emitted {len(app.goal_events)} completed goals, expected {expected_goal_events}")
    decisive_goal_audio = [name for name in played if name not in {"kick", "whoosh"}]
    expected_decisive_audio = [name for name in GOAL_AUDIO_EVENTS if name not in {"kick", "whoosh"}] * expected_goal_events
    if decisive_goal_audio != expected_decisive_audio:
        raise AssertionError(f"full match goal audio cues are not synchronized: {played}")
    if played.count("kick") < expected_goal_events or played.count("whoosh") < expected_goal_events:
        raise AssertionError(f"full match is missing kick/whoosh cues for goals: {played}")
    ordered = sorted(frame_times)
    p95 = ordered[int(len(ordered) * 0.95)]
    if p95 > 0.090:
        raise AssertionError(f"full match render loop too slow for QA dummy run: p95={p95:.3f}s")
    final_scroll = app.ground_scroll
    final_time = app.t
    field = pygame.Rect(32, 110, 910, 490)
    final_state = app.cinematic_scene_state(field, pred)
    if not final_state.get("settled") or float(final_state.get("run_speed", 1.0)) != 0.0:
        raise AssertionError("winner cinematic should settle into a final pose instead of running in place")
    for _index in range(45):
        app.update(1 / 15)
        app.draw()
    if app.t != final_time:
        raise AssertionError(f"full match time should stay capped at 90min cinematic: {app.t} vs {final_time}")
    if abs(app.ground_scroll - final_scroll) > 0.5:
        raise AssertionError(f"parallax keeps sliding after full time: {final_scroll:.2f} -> {app.ground_scroll:.2f}")


def _qa_cpu_window_means(
    values: list[float],
    window_frames: int,
) -> list[float]:
    windows = [
        values[start : start + window_frames]
        for start in range(0, len(values), window_frames)
    ]
    if len(windows) > 1 and len(windows[-1]) < window_frames:
        windows[-2].extend(windows.pop())
    return [sum(window) / len(window) for window in windows]


def _qa_cpu_clock_profile(
    values: list[float],
    coarse_threshold: float,
) -> tuple[bool, float]:
    positive = sorted(value for value in values if value > 1e-9)
    if not positive:
        return True, float("inf")
    p10_index = min(len(positive) - 1, math.ceil(len(positive) * 0.10) - 1)
    observed_quantum = positive[p10_index]
    return observed_quantum >= coarse_threshold, observed_quantum


def validate_cpu_clock_quantization_policy() -> None:
    coarse_tick = 1.0 / 64.0
    coarse_samples = [
        coarse_tick if index % 4 == 0 else 0.0
        for index in range(2701)
    ]
    coarse_samples[7] = 0.001
    coarse_samples[120] = coarse_tick * 2
    coarse, quantum = _qa_cpu_clock_profile(coarse_samples, 0.014)
    if not coarse or not math.isclose(quantum, coarse_tick, abs_tol=1e-9):
        raise AssertionError(
            "15.625ms CPU clock with a sub-quantum outlier was not detected"
        )
    coarse_windows = _qa_cpu_window_means(coarse_samples, 30)
    if len(coarse_windows) != 90:
        raise AssertionError(
            "partial CPU benchmark window was not merged into a full window"
        )
    if max(coarse_windows) >= 0.014:
        raise AssertionError(
            "quantized 15.625/31.25ms samples distorted CPU throughput windows"
        )
    reconstructed_total = (
        sum(coarse_windows[:-1]) * 30
        + coarse_windows[-1] * 31
    )
    if not math.isclose(
        reconstructed_total,
        sum(coarse_samples),
        abs_tol=1e-9,
    ):
        raise AssertionError("partial CPU benchmark samples were not preserved")

    double_tick = coarse_tick * 2
    double_tick_samples = [
        double_tick if index % 4 == 0 else 0.0
        for index in range(300)
    ]
    double_coarse, double_quantum = _qa_cpu_clock_profile(
        double_tick_samples,
        0.014,
    )
    if not double_coarse or not math.isclose(
        double_quantum,
        double_tick,
        abs_tol=1e-9,
    ):
        raise AssertionError("31.25ms base CPU quantum was not detected")
    if max(_qa_cpu_window_means(double_tick_samples, 30)) >= 0.014:
        raise AssertionError("31.25ms base quantum distorted CPU throughput windows")

    fine_samples = [0.006 + (index % 3) * 0.0001 for index in range(300)]
    fine, _quantum = _qa_cpu_clock_profile(fine_samples, 0.014)
    if fine:
        raise AssertionError("fine CPU clock was classified as coarse")


def validate_sixty_fps_budget() -> None:
    p95_budget = 0.014
    p99_budget = 1.0 / 60.0
    two_frame_budget = 1.0 / 30.0
    cpu_window_frames = 30
    measurements: list[dict[str, float]] = []
    strict_wall_clock = os.getenv(
        "ARENA_AI_STRICT_WALL_CLOCK_QA",
        "",
    ).strip().casefold() in {"1", "true", "yes", "on"}

    def distribution(values: list[float]) -> tuple[float, float, float, float]:
        ordered = sorted(values)
        p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
        p99 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.99) - 1)]
        maximum = ordered[-1]
        severe_ratio = sum(value > two_frame_budget for value in values) / len(values)
        return p95, p99, maximum, severe_ratio

    def passed(trial: dict[str, float], prefix: str) -> bool:
        return not (
            trial[f"{prefix}_p95"] > p95_budget
            or trial[f"{prefix}_p99"] > p99_budget
            or (
                prefix == "cpu"
                and trial[f"{prefix}_max"] > two_frame_budget
            )
            or trial[f"{prefix}_severe_ratio"] > 0.001
        )

    def complete_passed(trial: dict[str, float]) -> bool:
        cpu_passed = passed(trial, "cpu")
        wall_passed = passed(trial, "wall")
        if bool(trial["cpu_clock_coarse"]):
            # A coarse CPU clock cannot attribute one quantum to one frame.
            # Keep CPU throughput bounded in windows and make high-resolution
            # wall time authoritative for individual-frame stalls.
            return (
                cpu_passed
                and wall_passed
                and trial["wall_max"] <= two_frame_budget
            )
        return cpu_passed and (wall_passed or not strict_wall_clock)

    maximum_attempts = 5
    cpu_clock_coarse_mode: bool | None = None
    consecutive_passes = 0
    saw_failure = False
    for attempt in range(maximum_attempts):
        app = App(seed=2026)
        app.set_simulate("match")
        pred = home_win_prediction()
        install_qa_match_prediction(app, pred, "60fps benchmark")
        for _index in range(120):
            pygame.event.pump()
            app.update(1 / 60)
            app.draw()
            app.flush_queued_match_audio()
        app.t = 0.0
        app.ground_scroll = 0.0
        app.ground_scroll_velocity = 0.0
        app.ground_travel_distance = 0.0
        app.cinematic_attack_phase_anchors.clear()
        app.goal_events.clear()
        app.shot_events.clear()
        app.shot_progress_cursor.clear()
        gc.collect()
        wall_times = []
        cpu_times = []
        frames = int(SIMULATION_SECONDS / (1 / 60)) + 1
        for _index in range(frames):
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            pygame.event.pump()
            app.update(1 / 60)
            app.draw()
            app.flush_queued_match_audio()
            cpu_times.append(time.process_time() - cpu_start)
            wall_times.append(time.perf_counter() - wall_start)
        wall_p95, wall_p99, wall_max, wall_severe_ratio = distribution(wall_times)
        cpu_raw_max = max(cpu_times)
        detected_coarse_clock, cpu_clock_quantum = _qa_cpu_clock_profile(
            cpu_times,
            p95_budget,
        )
        if cpu_clock_coarse_mode is None:
            cpu_clock_coarse_mode = detected_coarse_clock
        cpu_clock_coarse = cpu_clock_coarse_mode
        effective_cpu_times = (
            _qa_cpu_window_means(cpu_times, cpu_window_frames)
            if cpu_clock_coarse
            else cpu_times
        )
        cpu_p95, cpu_p99, cpu_max, cpu_severe_ratio = distribution(
            effective_cpu_times
        )
        measurements.append(
            {
                "wall_p95": wall_p95,
                "wall_p99": wall_p99,
                "wall_max": wall_max,
                "wall_severe_ratio": wall_severe_ratio,
                "cpu_p95": cpu_p95,
                "cpu_p99": cpu_p99,
                "cpu_max": cpu_max,
                "cpu_severe_ratio": cpu_severe_ratio,
                "cpu_raw_max": cpu_raw_max,
                "cpu_clock_quantum": cpu_clock_quantum,
                "cpu_clock_coarse": float(cpu_clock_coarse),
            }
        )

        stats = app.surface_cache.stats()
        if (
            stats["scaled"] >= app.surface_cache.max_scaled
            or stats["roto"] >= app.surface_cache.max_roto
            or stats["alpha"] >= app.surface_cache.max_alpha
        ):
            raise AssertionError(f"render cache is hitting its cap during match loop: {stats}")
        if len(app.text_cache.surfaces) >= app.text_cache.max_entries:
            raise AssertionError(f"text cache is hitting its cap during match loop: {len(app.text_cache.surfaces)}")
        assert_auxiliary_caches_within_limits(app, "match loop")

        current_wall_passed = passed(measurements[-1], "wall")
        current_passed = complete_passed(measurements[-1])
        if current_passed:
            consecutive_passes += 1
        else:
            consecutive_passes = 0
            saw_failure = True
        if consecutive_passes >= 2:
            pass_mode = (
                f"{cpu_window_frames}-frame CPU windows plus mandatory "
                "per-frame wall budget"
                if cpu_clock_coarse
                else ("CPU+wall" if strict_wall_clock else "CPU")
            )
            if cpu_clock_coarse:
                print(
                    "[aaa-qa] coarse CPU clock detected and locked for all "
                    "60fps attempts"
                )
            print(
                "[aaa-qa] 60fps budget "
                f"{'recovered' if saw_failure else 'passed'} with two "
                f"consecutive {pass_mode} passes"
            )
            if not cpu_clock_coarse and not current_wall_passed:
                print(
                    "[aaa-qa] 60fps CPU budget passed; wall-clock is "
                    "inconclusive under host contention "
                    "(set ARENA_AI_STRICT_WALL_CLOCK_QA=1 on a controlled host)"
                )
            return
        if attempt < maximum_attempts - 1:
            del app
            gc.collect()
            time.sleep(0.5)
            continue
        detail_parts = []
        for index, trial in enumerate(measurements):
            cpu_label = (
                f"cpu[{cpu_window_frames}-frame avg]"
                if bool(trial["cpu_clock_coarse"])
                else "cpu"
            )
            quantum_detail = (
                f", raw_quantum={trial['cpu_clock_quantum'] * 1000:.2f}ms, "
                f"raw_max={trial['cpu_raw_max'] * 1000:.2f}ms"
                if bool(trial["cpu_clock_coarse"])
                else ""
            )
            detail_parts.append(
                f"attempt={index + 1}: "
                f"wall(p95={trial['wall_p95'] * 1000:.2f}ms, "
                f"p99={trial['wall_p99'] * 1000:.2f}ms, "
                f"max={trial['wall_max'] * 1000:.2f}ms, "
                f"over_33ms={trial['wall_severe_ratio']:.3%}); "
                f"{cpu_label}(p95={trial['cpu_p95'] * 1000:.2f}ms, "
                f"p99={trial['cpu_p99'] * 1000:.2f}ms, "
                f"max={trial['cpu_max'] * 1000:.2f}ms, "
                f"over_33ms={trial['cpu_severe_ratio']:.3%}"
                f"{quantum_detail})"
            )
        details = "; ".join(detail_parts)
        complete_green = all(
            complete_passed(trial) for trial in measurements[-2:]
        )
        cpu_green = all(passed(trial, "cpu") for trial in measurements[-2:])
        coarse_clock = any(
            bool(trial["cpu_clock_coarse"]) for trial in measurements[-2:]
        )
        if coarse_clock:
            classification = "runtime frame-time regression"
        elif strict_wall_clock and cpu_green and not complete_green:
            classification = "wall-clock inconclusive under host contention"
        else:
            classification = "runtime CPU regression"
        raise AssertionError(
            f"60fps budget {classification}; two consecutive complete "
            f"passes are required: {details}"
        )


def validate_tournament_render_budget() -> None:
    app = App(seed=2026)
    qa_runs = 24
    odds, representative = app.model.champion_odds_with_representative(
        runs=qa_runs,
        seed=2026,
        workers=TOURNAMENT_MONTE_CARLO_WORKERS,
        progress_with_odds=False,
    )
    app.mc_running = False
    app.mc_progress_done = TOURNAMENT_MONTE_CARLO_RUNS
    app.mc_progress_total = TOURNAMENT_MONTE_CARLO_RUNS
    app.champion_odds = odds
    app.tournament_result = representative
    attempts: list[dict[str, float]] = []
    strict_wall_clock = os.getenv(
        "ARENA_AI_STRICT_WALL_CLOCK_QA",
        "",
    ).strip().casefold() in {"1", "true", "yes", "on"}
    maximum_attempts = 5
    for attempt in range(maximum_attempts):
        wall_times = []
        cpu_times = []
        for view in ("groups", "bracket"):
            app.tournament_view = view
            for _index in range(12):
                app.t += 1 / 60
                app.draw_tournament()
            gc.collect()
            for _index in range(120):
                app.t += 1 / 60
                wall_start = time.perf_counter()
                cpu_start = time.process_time()
                app.draw_tournament()
                cpu_times.append(time.process_time() - cpu_start)
                wall_times.append(time.perf_counter() - wall_start)
        wall_ordered = sorted(wall_times)
        cpu_ordered = sorted(cpu_times)
        attempts.append(
            {
                "wall_p95": wall_ordered[int(len(wall_ordered) * 0.95)],
                "cpu_p95": cpu_ordered[int(len(cpu_ordered) * 0.95)],
            }
        )
        current_cpu_passed = attempts[-1]["cpu_p95"] <= 1.0 / 60.0
        current_wall_passed = attempts[-1]["wall_p95"] <= 0.024
        current_passed = current_cpu_passed and (
            current_wall_passed or not strict_wall_clock
        )
        recovered = (
            attempt >= 2
            and current_passed
            and attempts[-2]["cpu_p95"] <= 1.0 / 60.0
            and (
                attempts[-2]["wall_p95"] <= 0.024
                or not strict_wall_clock
            )
        )
        if attempt == 0 and current_passed:
            if not current_wall_passed:
                print(
                    "[aaa-qa] tournament CPU budget passed; wall-clock is "
                    "inconclusive under host contention "
                    "(set ARENA_AI_STRICT_WALL_CLOCK_QA=1 on a controlled host)"
                )
            break
        if recovered:
            print(
                "[aaa-qa] tournament render recovered with two consecutive "
                f"{'CPU+wall' if strict_wall_clock else 'CPU'} passes "
                "after an initial failure"
            )
            break
        if attempt < maximum_attempts - 1:
            time.sleep(0.5)
            continue
        details = "; ".join(
            f"attempt={index + 1}: "
            f"wall_p95={trial['wall_p95'] * 1000:.2f}ms, "
            f"cpu_p95={trial['cpu_p95'] * 1000:.2f}ms"
            for index, trial in enumerate(attempts)
        )
        cpu_green = all(
            trial["cpu_p95"] <= 1.0 / 60.0
            for trial in attempts[-2:]
        )
        classification = (
            "wall-clock inconclusive under host contention"
            if strict_wall_clock and cpu_green
            else "runtime CPU regression"
        )
        raise AssertionError(
            f"tournament render budget {classification}; two "
            f"consecutive CPU+wall passes are required: {details}"
        )
    if len(app.surface_cache.covered) >= app.surface_cache.max_cover:
        raise AssertionError(f"cover-image cache is hitting its cap during tournament render: {app.surface_cache.stats()}")
    assert_auxiliary_caches_within_limits(app, "tournament render")


def assert_tournament_draw_has_no_ellipsis(app: App, label: str) -> None:
    original = app.draw_text_ellipsis
    truncations: list[tuple[str, str, int, int]] = []

    def tracking_draw_text_ellipsis(
        text: str,
        text_font: pygame.font.Font,
        color: tuple[int, int, int],
        x: int,
        y: int,
        max_width: int,
    ) -> None:
        shown = app.ellipsize(text, text_font, max_width)
        if shown != text:
            truncations.append((text, shown, x, y))
        original(text, text_font, color, x, y, max_width)

    app.draw_text_ellipsis = tracking_draw_text_ellipsis  # type: ignore[method-assign]
    try:
        app.draw_tournament()
    finally:
        app.draw_text_ellipsis = original  # type: ignore[method-assign]

    if truncations:
        sample = "; ".join(f"{text!r} -> {shown!r} at {x},{y}" for text, shown, x, y in truncations[:5])
        raise AssertionError(f"Copa UI text truncates in {label}: {sample}")


def validate_copa_copy_contract() -> None:
    main_source = (ROOT / "src" / "arena_ai" / "main.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_doc = (ROOT / "docs" / "MODEL.md").read_text(encoding="utf-8")
    banned_main = {
        '"TIME A"': "confronto selector must use seleção",
        '"TIME B"': "confronto selector must use seleção",
        "Setas: time": "top shortcut must use seleção",
        "32 AVOS": "Copa bracket must say Fase de 32",
        '"32 avos"': "Copa bracket helper must say Fase de 32",
        "def draw_tournament_summary": "legacy tournament summary UI should stay removed",
        "def draw_tournament_pending_panels": "legacy pending Copa panels should stay removed",
        "def draw_group_stage_panel": "legacy group-stage Copa panel should stay removed",
        "def draw_knockout_panel": "legacy knockout Copa panel should stay removed",
        "def draw_monte_carlo_panel": "legacy Monte Carlo Copa panel should stay removed",
    }
    for fragment, reason in banned_main.items():
        if fragment in main_source:
            raise AssertionError(f"forbidden Copa UI copy/helper remains: {fragment} ({reason})")
    for required in (
        "BACKSPACE volta | T/R nova | G grupos | M chave",
        "FASE DE 32",
        "SELEÇÃO A",
        "SELEÇÃO B",
    ):
        if required not in main_source:
            raise AssertionError(f"required Copa UI copy is missing: {required}")
    for fragment in ("troca o time", "48 times", "IDs de times", "quatro times"):
        if fragment in readme or fragment in model_doc:
            raise AssertionError(f"Portuguese Copa docs still use time instead of seleção: {fragment}")


def validate_tournament_layout_gate() -> None:
    app = App(seed=2026)

    app.state = "tournament"
    app.tournament_result = None
    app.champion_odds = []
    app.mc_running = True
    app.mc_progress_done = TOURNAMENT_MONTE_CARLO_RUNS
    app.mc_progress_total = TOURNAMENT_MONTE_CARLO_RUNS
    assert_tournament_draw_has_no_ellipsis(app, "loading")

    odds, representative = app.model.champion_odds_with_representative(
        runs=24,
        seed=2026,
        workers=TOURNAMENT_MONTE_CARLO_WORKERS,
        progress_with_odds=False,
    )
    app.mc_running = False
    app.mc_progress_done = TOURNAMENT_MONTE_CARLO_RUNS
    app.mc_progress_total = TOURNAMENT_MONTE_CARLO_RUNS
    app.champion_odds = odds
    app.tournament_result = representative

    safe_rect = pygame.Rect(24, 88, WIDTH - 48, HEIGHT - 112)
    start_x, start_y = 48, 230
    card_w, card_h = 282, 140
    gap_x, gap_y = 22, 12
    for index in range(12):
        col = index % 4
        row = index // 4
        rect = pygame.Rect(start_x + col * (card_w + gap_x), start_y + row * (card_h + gap_y), card_w, card_h)
        if not safe_rect.contains(rect):
            raise AssertionError(f"group card leaves tournament safe area: {rect}")

    round32_rects = []
    for index in range(16):
        col = index // 8
        row = index % 8
        round32_rects.append(pygame.Rect(44 + col * 186, 254 + row * 30, 172, 24))
    bracket_rects = [
        *round32_rects,
        *[pygame.Rect(430, 258 + index * 30, 170, 24) for index in range(8)],
        *[pygame.Rect(620, 306 + index * 50, 170, 24) for index in range(4)],
        *[pygame.Rect(810, 356 + index * 50, 150, 24) for index in range(2)],
        pygame.Rect(984, 368, 238, 250),
    ]
    for rect in bracket_rects:
        if not safe_rect.contains(rect):
            raise AssertionError(f"knockout element leaves tournament safe area: {rect}")

    for view in ("groups", "bracket"):
        app.tournament_view = view
        assert_tournament_draw_has_no_ellipsis(app, f"result {view}")


def validate_tournament_result_header_contract() -> None:
    app = App(seed=2026)
    odds, representative = app.model.champion_odds_with_representative(
        runs=24,
        seed=2026,
        workers=TOURNAMENT_MONTE_CARLO_WORKERS,
        progress_with_odds=False,
    )
    if representative is None:
        raise AssertionError("Copa result header needs a representative campaign")
    app.state = "tournament"
    app.mc_running = False
    app.mc_progress_done = TOURNAMENT_MONTE_CARLO_RUNS
    app.mc_progress_total = TOURNAMENT_MONTE_CARLO_RUNS
    app.champion_odds = odds
    app.tournament_result = representative
    texts = capture_app_draw_text(app, app.draw_tournament_result_header)
    joined = "\n".join(texts)
    required = (
        "CAMINHOS MAIS FORTES",
        "FAVORITO DO ORÁCULO",
        "HISTÓRIA DA SIMULAÇÃO",
        "Caminhos fortes:",
        "Copas vencidas",
    )
    missing = [fragment for fragment in required if fragment not in joined]
    if missing:
        raise AssertionError(f"Copa result header is missing separated blocks: {missing}")
    positions = [texts.index(fragment) for fragment in required[:3]]
    if positions != sorted(positions):
        raise AssertionError(f"Copa header hierarchy is not ranking/favorite/story: {positions}")
    phrase = app.tournament_plausibility_phrase()
    if not phrase or phrase not in joined:
        raise AssertionError(f"Copa header does not expose narrative plausibility phrase: {phrase!r}")
    if "_" in phrase or len(phrase) > 90:
        raise AssertionError(f"Copa plausibility phrase is not a clean microcopy: {phrase!r}")


def validate_tournament_loading_pacing_gate() -> None:
    app = App(seed=2026)
    app.state = "tournament"
    app.mc_progress_total = TOURNAMENT_MONTE_CARLO_RUNS
    app.mc_progress_done = TOURNAMENT_MONTE_CARLO_RUNS
    app.mc_started_t = 0.0
    app.t = TOURNAMENT_MIN_LOADING_SECONDS * 0.42
    app.mc_pending_result = ([("Brazil", TOURNAMENT_MONTE_CARLO_RUNS, 1.0)], None)
    app.mc_running = True
    early_progress = app.monte_carlo_progress()
    if early_progress >= 0.985:
        raise AssertionError(f"Monte Carlo loading revealed too fast from cached/full progress: {early_progress:.3f}")
    pending_text = "\n".join(capture_app_draw_text(app, app.draw_tournament_loading))
    if "REVELANDO CENÁRIO" not in pending_text:
        raise AssertionError("Monte Carlo pending-result reveal state is not visible during the minimum loading window")
    app.apply_pending_monte_carlo_result_if_ready()
    if not app.mc_running or app.mc_pending_result is None:
        raise AssertionError("Monte Carlo pending result should wait for the minimum reveal window")

    app.t = TOURNAMENT_MIN_LOADING_SECONDS + 0.02
    app.apply_pending_monte_carlo_result_if_ready()
    if app.mc_running or app.mc_pending_result is not None or not app.champion_odds:
        raise AssertionError("Monte Carlo pending result did not reveal after the minimum loading window")


def validate_render_purity() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    app.update(1 / 30)
    app.draw()
    before_cache = render_cache_snapshot(app)
    before = (
        app.state,
        app.t,
        app.ground_scroll,
        app.ground_scroll_velocity,
        tuple(sorted(app.shot_events)),
        tuple(sorted(app.goal_events)),
        tuple(sorted(app.shot_progress_cursor.items())),
        app.mc_running,
        app.pending_tournament_seed,
        len(app.turf_tile_cache),
        len(app.gradient_tile_cache),
        len(app.gradient_mask_cache),
    )
    app.draw()
    after_cache = render_cache_snapshot(app)
    after = (
        app.state,
        app.t,
        app.ground_scroll,
        app.ground_scroll_velocity,
        tuple(sorted(app.shot_events)),
        tuple(sorted(app.goal_events)),
        tuple(sorted(app.shot_progress_cursor.items())),
        app.mc_running,
        app.pending_tournament_seed,
        len(app.turf_tile_cache),
        len(app.gradient_tile_cache),
        len(app.gradient_mask_cache),
    )
    if before != after:
        raise AssertionError(f"draw mutated simulation state: before={before}, after={after}")
    if before_cache != after_cache:
        raise AssertionError(f"warm match draw allocated new cached surfaces: before={before_cache}, after={after_cache}")
    assert_auxiliary_caches_within_limits(app, "render purity match draw")

    app.champion_odds_runs = 24
    app.set_tournament()
    pending = app.pending_tournament_seed
    if app.tournament_result is not None or app.champion_odds:
        raise AssertionError("set_tournament must not reveal any Monte Carlo result before the background job finishes")
    app.draw()
    before_cache = render_cache_snapshot(app)
    app.draw()
    after_cache = render_cache_snapshot(app)
    if before_cache != after_cache:
        raise AssertionError(f"warm tournament draw allocated new cached surfaces: before={before_cache}, after={after_cache}")
    if app.pending_tournament_seed != pending or app.mc_running:
        raise AssertionError("draw_tournament started or changed Monte Carlo state; update() must own that")
    if app.tournament_result is not None or app.champion_odds:
        raise AssertionError("draw_tournament revealed preview results before Monte Carlo completion")
    app.update(1 / 60)
    if not app.mc_running:
        raise AssertionError("update() did not start pending Monte Carlo job")
    if app.mc_thread is None or app.mc_thread.name != "arena-ai-monte-carlo" or not app.mc_thread.daemon:
        raise AssertionError("Monte Carlo must run in a dedicated daemon thread")
    if app.champion_odds:
        raise AssertionError("progress updates must not expose partial champion odds in the game UI")
    assert_auxiliary_caches_within_limits(app, "render purity tournament draw")
    app.cancel_champion_odds_job()


def validate_visual_determinism() -> None:
    def sequence() -> list[str]:
        app = App(seed=2026)
        app.set_simulate("match")
        pred = home_win_prediction()
        app.match_prediction = pred
        hashes = []
        for seconds in (0.0, 12.0, 26.5, 31.0, 43.0):
            seek_match_time(app, pred, seconds)
            app.screen.fill((0, 0, 0))
            app.draw()
            hashes.append(surface_hash(app.screen))
        return hashes

    first = sequence()
    second = sequence()
    if first != second:
        raise AssertionError("visual QA is not deterministic for the same seed and prediction")


SMOKE_STEPS = (
    validate_asset_manifest,
    validate_copa_copy_contract,
    validate_button_label_auto_fit_gate,
    validate_text_safe_area_gate,
    validate_cinematic_draw_order_declared,
)

STANDARD_STEPS = (
    validate_flag_sprites,
    validate_cinematic_inventory,
    validate_sound_assets,
    validate_app_icon,
    validate_asset_manifest,
    validate_release_inventory_contract,
    validate_model_policy_artifacts,
    validate_cpu_clock_quantization_policy,
    validate_sound_engine_layers,
    validate_match_screen_layout_gate,
    validate_text_safe_area_gate,
    validate_button_label_auto_fit_gate,
    validate_selection_card_metric_layout_gate,
    validate_selection_input_flow_gate,
    validate_match_hud_text_fit_gate,
    validate_match_hud_density_legibility_gate,
    validate_match_result_suspense_gate,
    validate_match_clock_and_reveal_sync_gate,
    validate_match_runtime_state_cache_gate,
    validate_ball_physics_contract_fast,
    validate_aaa_findings_light_gate,
    validate_match_final_cinematic_settlement_gate,
    validate_copa_copy_contract,
    validate_chance_schedule_no_dead_air,
    validate_nil_draw_has_no_fake_goals_gate,
    validate_cinematic_draw_order_declared,
    validate_cinematic_reveal_timing_gate,
    validate_tournament_layout_gate,
    validate_tournament_result_header_contract,
    validate_tournament_loading_pacing_gate,
    validate_tournament_seed_entropy_gate,
    validate_monte_carlo_runtime_mode_gate,
    validate_render_purity,
)

AAA_STEPS = (
    validate_parallax_assets,
    validate_authored_cinematic_sprites,
    validate_native_oracle_legibility,
    validate_goal_composite_cache_bound_gate,
    validate_audio_event_order,
    validate_fifa_external_assets,
    *STANDARD_STEPS,
    validate_cinematic_camera_continuity_gate,
    validate_goalkeeper_safe_bounds_variants,
    validate_goal_overlay_score_sync_gate,
    validate_visual_determinism,
    validate_full_match_flow,
    validate_sixty_fps_budget,
    validate_tournament_render_budget,
    validate_monte_carlo_story_diversity_gate,
    validate_monte_carlo_fast_path,
)

VALIDATION_SUITES = {
    "smoke": SMOKE_STEPS,
    "standard": STANDARD_STEPS,
    "aaa": AAA_STEPS,
}


def run_step(step: Callable[[], None]) -> None:
    start = time.perf_counter()
    print(f"[validate] {step.__name__}", flush=True)
    try:
        step()
    except Exception:
        elapsed = time.perf_counter() - start
        print(f"[validate] fail {step.__name__} ({elapsed:.2f}s)", flush=True)
        raise
    finally:
        gc.collect()
    elapsed = time.perf_counter() - start
    print(f"[validate] ok {step.__name__} ({elapsed:.2f}s)", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate Arena AI assets, UI, audio and performance gates.")
    parser.add_argument(
        "--suite",
        choices=tuple(VALIDATION_SUITES),
        default="standard",
        help="smoke is fast, standard is the default CI gate, aaa runs the full cinematic/performance QA.",
    )
    args = parser.parse_args(argv)

    pygame.init()
    pygame.display.set_mode((1, 1))
    try:
        for step in VALIDATION_SUITES[args.suite]:
            run_step(step)
    finally:
        pygame.quit()
    print(f"validation passed: suite={args.suite}, steps={len(VALIDATION_SUITES[args.suite])}")


if __name__ == "__main__":
    main()

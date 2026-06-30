from __future__ import annotations

import argparse
import csv
import hashlib
import fnmatch
import gc
import json
import os
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from arena_ai.main import (
    App,
    ALGORITHM_NAMES,
    CINEMATIC_BALL_SIZE,
    CINEMATIC_KEEPER_SCALE,
    CINEMATIC_NEUTRAL_PLAYER_SCALE,
    CINEMATIC_PLAYER_SCALE,
    CINEMATIC_POSE_SIZE,
    CHANCE_EVENT_WINDOW_MINUTES,
    CHANCE_MIN_SPACING_MINUTES,
    DRAW_NEUTRAL_RAMP,
    DRAW_NEUTRAL_START_PROGRESS,
    GOAL_EVENT_WINDOW_MINUTES,
    FIFA_EXTERNAL_IMAGES,
    MATCH_HUD_BANNED_COPY,
    MATCH_HUD_REQUIRED_COPY,
    MATCH_HUD_STATE_COPY,
    MATCH_HUD_TOP_SCORE_COUNT,
    SIMULATION_SECONDS,
    SHOT_FOLLOW_THROUGH_HOLD_END,
    SHOT_GOAL_REVEAL_AT,
    SHOT_KEEPER_FULL_AT,
    SHOT_KEEPER_REVEAL_AT,
    SHOT_KICK_AT,
    SHOT_PLANT_AT,
    SHOT_WHOOSH_AT,
    SHOT_NET_AT,
    SHOT_NET_VISUAL_CONTACT_AT,
    SHOT_REVERB_AT,
    TOURNAMENT_MONTE_CARLO_RUNS,
    TOURNAMENT_MONTE_CARLO_USE_SCENARIO_BANK,
    TOURNAMENT_MONTE_CARLO_WORKERS,
    TOURNAMENT_MIN_LOADING_SECONDS,
    WIDTH,
    HEIGHT,
    KICK_FOOT_ANCHOR,
    RUNNER_FOOT_ANCHORS,
    clamp,
    smoothstep,
)
from arena_ai.cinematic_uniforms import CINEMATIC_UNIFORMS, UNIFORM_CODES
from arena_ai.audio_manifest import AUDIO_RUNTIME_FILES, GOAL_AUDIO_SEQUENCE, REQUIRED_AUDIO_BUSES
from arena_ai.ui import Button
from arena_ai.worldcup_model import Prediction, WorldCupModel


ROOT = Path(__file__).resolve().parents[1]
CINEMATIC_DIR = ROOT / "assets" / "generated" / "cinematic"
CINEMATIC_SOURCE_DIR = ROOT / "assets" / "generated" / "cinematic_sources"
FLAG_DIR = ROOT / "assets" / "generated" / "flags"
SOUND_DIR = ROOT / "assets" / "sounds"
STADIUM_BG = ROOT / "assets" / "generated" / "stadium_parallax_real.png"
APP_ICON = ROOT / "assets" / "generated" / "app_icon_worldcup.png"
PARALLAX_DIR = ROOT / "assets" / "generated" / "parallax"
PARALLAX_SOURCES = ROOT / "assets" / "generated" / "parallax_sources"
BALL_DIR = ROOT / "assets" / "generated" / "balls3d"
BALL_SOURCE = ROOT / "assets" / "generated" / "ball_sources" / "plain_ball_sheet_8frames.png"
ASSET_MANIFEST = ROOT / "assets" / "asset_manifest.json"
FIFA_EXTERNAL_DIR = ROOT / "assets" / "generated" / "fifa_external"
MODEL_PACKAGE = ROOT / "modeling" / "worldcup_2026_ml" / "models" / "model_sota.pkl"
MODEL_REPORT = ROOT / "modeling" / "worldcup_2026_ml" / "reports" / "sota_model_report.json"
MODEL_STATS_REPORT = ROOT / "modeling" / "worldcup_2026_ml" / "reports" / "sota_statistical_report.json"
MODEL_TRAINING_MATCHES = ROOT / "modeling" / "worldcup_2026_ml" / "data" / "processed" / "sota_training_matches.csv"
MODEL_SOTA_PIPELINE = ROOT / "modeling" / "worldcup_2026_ml" / "src" / "sota_pipeline.py"
MODEL_STATS_QA_SCRIPT = ROOT / "scripts" / "model_stats_qa.py"
MODEL_MC_STABILITY_SCRIPT = ROOT / "scripts" / "monte_carlo_stability.py"
MODEL_RAW_DATA_ROOT = ROOT / "modeling" / "worldcup_2026_ml" / "data" / "raw"
TOURNAMENT_SIMULATION = ROOT / "modeling" / "worldcup_2026_ml" / "reports" / "sota_tournament_simulation.csv"
FLAG_SIZE = (172, 108)
ROWS = UNIFORM_CODES
SHORTS_BY_CODE = {uniform.code: uniform.shorts for uniform in CINEMATIC_UNIFORMS}
POSES = ("idle", "run1", "dribble", "kick")
RUNNER_FRAMES = 4
KEEPER_FRAMES = 4
GOAL_NET_FRAMES = 4
POSE_SPRITE_SIZE = (256, 256)
RUNNER_SPRITE_SIZE = (256, 256)
KEEPER_SPRITE_SIZE = (288, 288)
REQUIRED_CINEMATIC_SOURCES = (
    "imagen_oracle_pose_sheet_8rows.png",
    "imagen_oracle_runner_sheet_8rows.png",
    "imagen_oracle_burgundy_pose_row.png",
    "imagen_oracle_burgundy_runner_row.png",
    "imagen_oracle_pose_left_sheet_8rows.png",
    "imagen_oracle_runner_left_sheet_8rows.png",
    "imagen_oracle_burgundy_pose_left_row.png",
    "imagen_oracle_burgundy_runner_left_row.png",
    "plain_keeper_sheet_green.png",
    "plain_goal_net_sheet.png",
)
REQUIRED_SOUNDS = tuple(f"runtime_assets/{filename}" for filename in AUDIO_RUNTIME_FILES)
GOAL_AUDIO_EVENTS = list(GOAL_AUDIO_SEQUENCE)
CINEMATIC_DRAW_ORDER = (
    "draw_cinematic_background",
    "draw_model_flow",
    "draw_cinematic_goal",
    "draw_cinematic_scene",
    "draw_cinematic_goal_overlay",
)
AUXILIARY_CACHE_LIMITS = {
    "turf_tile_cache": 8,
    "gradient_tile_cache": 8,
    "gradient_mask_cache": 8,
    "goal_orientation_cache": 24,
    "surface_bbox_cache": 360,
    "cinematic_overlay_cache": 512,
}


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
        "goal_orientation_cache": len(app.goal_orientation_cache),
        "surface_bbox_cache": len(app.surface_bbox_cache),
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


def assert_between(value: float, lower: float, upper: float, label: str) -> None:
    if not lower <= value <= upper:
        raise AssertionError(f"{label} out of AAA gate: {value:.3f} not in [{lower:.3f}, {upper:.3f}]")


def neutral_sample_second(ramp_position: float) -> float:
    progress = clamp(DRAW_NEUTRAL_START_PROGRESS + DRAW_NEUTRAL_RAMP * ramp_position)
    return progress * SIMULATION_SECONDS


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


def seek_match_time(app: App, pred: Prediction, seconds: float, step: float = 1 / 30) -> None:
    app.t = 0.0
    app.ground_scroll = 0.0
    app.ground_scroll_velocity = 0.0
    app.match_prediction = pred
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


def frame_delta(a: pygame.Surface, b: pygame.Surface) -> int:
    delta = 0
    for y in range(0, a.get_height(), 2):
        for x in range(0, a.get_width(), 2):
            ca = a.get_at((x, y))
            cb = b.get_at((x, y))
            delta += abs(ca.r - cb.r) + abs(ca.g - cb.g) + abs(ca.b - cb.b) + abs(ca.a - cb.a)
    return delta


def alpha_centroid(surface: pygame.Surface) -> tuple[float, float]:
    total = 0
    sx = 0
    sy = 0
    for y in range(surface.get_height()):
        for x in range(surface.get_width()):
            alpha = surface.get_at((x, y)).a
            if alpha <= 40:
                continue
            total += alpha
            sx += x * alpha
            sy += y * alpha
    if total <= 0:
        raise AssertionError("empty alpha centroid")
    return sx / total, sy / total


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


def oracle_mark_geometry(surface: pygame.Surface, code: str, step: int = 1) -> dict[str, float]:
    count, mark = oracle_mark_bounds(surface, code, step)
    if count <= 0 or mark.w <= 0 or mark.h <= 0:
        return {"pixels": 0.0, "width": 0.0, "height": 0.0, "density": 0.0, "columns": 0.0, "rows": 0.0, "gaps": 0.0}
    dark_mark = oracle_mark_color_is_dark(code)
    column_hits = [0] * mark.w
    row_hits = [0] * mark.h
    for y in range(mark.y, mark.bottom, step):
        for x in range(mark.x, mark.right, step):
            color = surface.get_at((x, y))
            if color.a <= 80:
                continue
            if dark_mark:
                active = color.r < 125 and color.g < 125 and color.b < 125
            else:
                active = color.r > 132 and color.g > 132 and color.b > 124
            if active:
                column_hits[x - mark.x] += 1
                row_hits[y - mark.y] += 1
    active_columns = [value > 0 for value in column_hits]
    active_rows = [value > 0 for value in row_hits]
    inner_columns = active_columns[1:-1] if len(active_columns) > 2 else active_columns
    gaps = 0
    in_gap = False
    for active in inner_columns:
        if not active and not in_gap:
            gaps += 1
            in_gap = True
        elif active:
            in_gap = False
    density = count / max(1, mark.w * mark.h)
    return {
        "pixels": float(count),
        "width": float(mark.w),
        "height": float(mark.h),
        "density": density,
        "columns": sum(active_columns) / max(1, len(active_columns)),
        "rows": sum(active_rows) / max(1, len(active_rows)),
        "gaps": float(gaps),
    }


def assert_oracle_wordmark_geometry(surface: pygame.Surface, code: str, path: Path) -> None:
    geometry = oracle_mark_geometry(surface, code)
    width = geometry["width"]
    density = geometry["density"]
    columns = geometry["columns"]
    rows = geometry["rows"]
    if width < 46:
        raise AssertionError(f"{path} ORACLE wordmark is too narrow to read as a chest mark: {geometry}")
    if not 0.035 <= density <= 0.24:
        raise AssertionError(f"{path} ORACLE wordmark has invalid ink density: {geometry}")
    if columns < 0.50 or rows < 0.30:
        raise AssertionError(f"{path} ORACLE wordmark has broken stroke coverage: {geometry}")


def assert_oracle_mark(surface: pygame.Surface, code: str, path: Path, minimum: int) -> None:
    broad_count = oracle_mark_pixel_count(surface, code)
    count, mark = oracle_mark_bounds(surface, code)
    if broad_count < minimum:
        raise AssertionError(f"{path} is missing the only allowed uniform mark, ORACLE on the chest: {broad_count} pixels")
    if minimum >= 30:
        if count < 16:
            raise AssertionError(f"{path} ORACLE mark is not measurable enough for AAA legibility: rect={mark}, pixels={count}")
        density = count / max(1, mark.w * mark.h)
        if mark.w < 18 or mark.h < 5 or density > 0.70:
            raise AssertionError(f"{path} ORACLE mark collapsed into an unreadable blob: rect={mark}, pixels={count}")
        assert_oracle_wordmark_geometry(surface, code, path)


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


def blue_shorts_pixel_count(surface: pygame.Surface, step: int = 1) -> int:
    bbox = alpha_bbox(surface)
    region = pygame.Rect(
        bbox.x + int(bbox.w * 0.10),
        bbox.y + int(bbox.h * 0.40),
        max(1, int(bbox.w * 0.82)),
        max(1, int(bbox.h * 0.42)),
    ).clip(surface.get_rect())
    count = 0
    for y in range(region.y, region.bottom, step):
        for x in range(region.x, region.right, step):
            color = surface.get_at((x, y))
            if color.a <= 80:
                continue
            if color.b > 110 and color.b > color.r + 45 and color.b > color.g + 25:
                count += 1
    return count


def assert_gold_blue_shorts(surface: pygame.Surface, path: Path, minimum: int) -> None:
    count = blue_shorts_pixel_count(surface)
    if count < minimum:
        raise AssertionError(f"{path} must keep Brazil/gold contrast: yellow shirt with blue shorts; blue pixels={count}")


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


def assert_animation_bbox_stability(frames: list[pygame.Surface], label: str) -> None:
    boxes = [alpha_bbox(frame) for frame in frames]
    bottoms = [box.bottom for box in boxes]
    widths = [box.w for box in boxes]
    heights = [box.h for box in boxes]
    if max(bottoms) - min(bottoms) > 14:
        raise AssertionError(f"{label} baseline jumps between animation frames: {bottoms}")
    if max(widths) / max(1, min(widths)) > 1.36:
        raise AssertionError(f"{label} width changes enough to read as crop flicker: {widths}")
    if max(heights) / max(1, min(heights)) > 1.18:
        raise AssertionError(f"{label} height changes enough to read as scale flicker: {heights}")


def average_rgb_in_visible_region(surface: pygame.Surface, region: pygame.Rect, step: int = 2) -> tuple[float, float, float]:
    region = region.clip(surface.get_rect())
    totals = [0.0, 0.0, 0.0]
    samples = 0
    for y in range(region.y, region.bottom, step):
        for x in range(region.x, region.right, step):
            color = surface.get_at((x, y))
            if color.a <= 90:
                continue
            totals[0] += color.r
            totals[1] += color.g
            totals[2] += color.b
            samples += 1
    if samples <= 0:
        return 0.0, 0.0, 0.0
    return totals[0] / samples, totals[1] / samples, totals[2] / samples


def assert_animation_color_stability(frames: list[pygame.Surface], label: str) -> None:
    torso_colors = []
    head_colors = []
    for frame in frames:
        bbox = alpha_bbox(frame)
        torso = pygame.Rect(
            bbox.x + int(bbox.w * 0.24),
            bbox.y + int(bbox.h * 0.36),
            max(1, int(bbox.w * 0.52)),
            max(1, int(bbox.h * 0.28)),
        )
        head = pygame.Rect(
            bbox.x + int(bbox.w * 0.30),
            bbox.y + int(bbox.h * 0.08),
            max(1, int(bbox.w * 0.40)),
            max(1, int(bbox.h * 0.20)),
        )
        torso_colors.append(average_rgb_in_visible_region(frame, torso))
        head_colors.append(average_rgb_in_visible_region(frame, head))

    for colors, limit, part in ((torso_colors, 62.0, "kit"), (head_colors, 78.0, "head/hair")):
        distances = [
            abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
            for a, b in zip(colors, colors[1:])
        ]
        if distances and max(distances) > limit:
            rounded = [tuple(round(channel, 1) for channel in color) for color in colors]
            raise AssertionError(f"{label} {part} color jumps between frames: distances={distances}, colors={rounded}")


def alpha_pixel_count(surface: pygame.Surface, threshold: int = 25, step: int = 1) -> int:
    return sum(
        1
        for y in range(0, surface.get_height(), step)
        for x in range(0, surface.get_width(), step)
        if surface.get_at((x, y)).a > threshold
    )


def assert_actor_feet_and_legs_uncropped(
    surface: pygame.Surface,
    label: str,
    min_lower_height: int,
    min_bottom_margin: int,
    max_bottom_margin: int,
) -> None:
    bbox = alpha_bbox(surface)
    bottom_margin = surface.get_height() - bbox.bottom
    if bottom_margin < min_bottom_margin:
        raise AssertionError(f"{label} feet are visually cut by the sprite bottom edge: bbox={bbox}")
    if bottom_margin > max_bottom_margin:
        raise AssertionError(f"{label} foot baseline drifted too high for the planted anchor: bbox={bbox}")
    lower_start = int(surface.get_height() * 0.52)
    lower_height = surface.get_height() - lower_start - min_bottom_margin
    lower_body = surface.subsurface(pygame.Rect(0, lower_start, surface.get_width(), lower_height)).copy()
    lower_bbox = alpha_bbox(lower_body)
    if lower_bbox.h < min_lower_height:
        raise AssertionError(f"{label} has weak leg/foot visibility: lower_body={lower_bbox}")
    foot_strip = surface.subsurface(
        pygame.Rect(0, max(0, bbox.bottom - 22), surface.get_width(), min(22, bbox.bottom))
    ).copy()
    if alpha_pixel_count(foot_strip, threshold=50, step=2) < 18:
        raise AssertionError(f"{label} does not keep enough visible foot pixels near the planted baseline")


def expected_cinematic_runtime_files() -> set[str]:
    expected: set[str] = set()
    for code in ROWS:
        expected.update(f"{code}_{pose}.png" for pose in POSES)
        expected.update(f"left_{code}_{pose}.png" for pose in POSES)
        expected.update(f"runner_{code}_{index}.png" for index in range(RUNNER_FRAMES))
        expected.update(f"runner_left_{code}_{index}.png" for index in range(RUNNER_FRAMES))
    expected.update(f"keeper_anim_{index}.png" for index in range(KEEPER_FRAMES))
    expected.update(f"goal_net_{index}.png" for index in range(GOAL_NET_FRAMES))
    expected.update(f"goal_front_{index}.png" for index in range(GOAL_NET_FRAMES))
    expected.update(f"goal_impact_{index}.png" for index in range(GOAL_NET_FRAMES))
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
    legacy_sources = sorted(path.name for path in CINEMATIC_SOURCE_DIR.glob("oracle_*.png"))
    if legacy_sources:
        raise AssertionError(f"legacy Oracle cinematic sources still present: {legacy_sources}")
    actual_sources = {path.name for path in CINEMATIC_SOURCE_DIR.glob("*.png")}
    expected_sources = set(REQUIRED_CINEMATIC_SOURCES)
    if actual_sources != expected_sources:
        extra = sorted(actual_sources - expected_sources)
        missing = sorted(expected_sources - actual_sources)
        raise AssertionError(f"cinematic source inventory mismatch; extra={extra}, missing={missing}")
    for filename in REQUIRED_CINEMATIC_SOURCES:
        path = CINEMATIC_SOURCE_DIR / filename
        if not path.exists():
            raise AssertionError(f"missing image_gen cinematic source: {path}")
    if not BALL_SOURCE.exists():
        raise AssertionError(f"missing image_gen ball source: {BALL_SOURCE}")
    expected_runtime = expected_cinematic_runtime_files()
    actual_runtime = {path.name for path in CINEMATIC_DIR.glob("*.png")}
    if actual_runtime != expected_runtime:
        extra = sorted(actual_runtime - expected_runtime)
        missing = sorted(expected_runtime - actual_runtime)
        raise AssertionError(f"cinematic runtime sprite inventory mismatch; extra={extra}, missing={missing}")
    for sample_name, expected_size in (
        ("blue_idle.png", POSE_SPRITE_SIZE),
        ("runner_blue_0.png", RUNNER_SPRITE_SIZE),
        ("keeper_anim_0.png", KEEPER_SPRITE_SIZE),
        ("goal_net_0.png", (360, 240)),
    ):
        path = CINEMATIC_DIR / sample_name
        if not path.exists():
            raise AssertionError(f"missing cinematic sample: {path}")
        image = pygame.image.load(path).convert_alpha()
        if image.get_size() != expected_size:
            raise AssertionError(f"{path} has unexpected size {image.get_size()}")


def validate_cinematic_sprites() -> None:
    validate_cinematic_inventory()
    for code in ROWS:
        frames = {}
        for pose in POSES:
            path = CINEMATIC_DIR / f"{code}_{pose}.png"
            if not path.exists():
                raise AssertionError(f"missing sprite: {path}")
            image = pygame.image.load(path).convert_alpha()
            if image.get_size() != POSE_SPRITE_SIZE:
                raise AssertionError(f"{path} has unexpected size {image.get_size()}")
            max_x = image.get_width() - 1
            max_y = image.get_height() - 1
            if any(image.get_at(point).a > 4 for point in ((0, 0), (max_x, 0), (0, max_y), (max_x, max_y))):
                raise AssertionError(f"{path} has non-transparent corners")
            if chroma_leak_count(image) > 20:
                raise AssertionError(f"{path} still has visible chroma-key leakage")
            if opaque_chroma_artifact_count(image) > 28:
                raise AssertionError(f"{path} has opaque chroma-key artifact blobs")
            bbox = alpha_bbox(image)
            if bbox.x <= 3 or bbox.y <= 3 or bbox.right >= image.get_width() - 3 or bbox.bottom >= image.get_height() - 3:
                raise AssertionError(f"{path} appears clipped: {bbox}")
            if bbox.h < 128 or bbox.w < 58:
                raise AssertionError(f"{path} has too little visible subject: {bbox}")
            assert_actor_feet_and_legs_uncropped(image, str(path), min_lower_height=76, min_bottom_margin=8, max_bottom_margin=24)
            lower_body = image.subsurface(pygame.Rect(0, int(image.get_height() * 0.56), image.get_width(), int(image.get_height() * 0.38))).copy()
            if alpha_bbox(lower_body).h < 34:
                raise AssertionError(f"{path} has weak lower-body/leg visibility")
            if bbox.w > image.get_width() - 18:
                raise AssertionError(f"{path} is too wide; likely has bad crop: {bbox}")
            if not image.get_height() - 28 <= bbox.bottom <= image.get_height() - 4:
                raise AssertionError(f"{path} foot anchor is inconsistent: {bbox}")
            if edge_energy(image) < 8.0:
                raise AssertionError(f"{path} is too soft after SOTA matte/resize: edge={edge_energy(image):.2f}")
            detached = [(size, rect) for size, rect in alpha_components(image)[1:] if size > 80]
            if detached:
                raise AssertionError(f"{path} has detached alpha fragments: {detached}")
            if pose in {"idle", "run1", "dribble", "kick"}:
                assert_oracle_mark(image, code, path, 5)
                assert_green_uniform_alpha(image, code, path)
                if code == "gold":
                    assert_gold_blue_shorts(image, path, 40)
                assert_no_light_short_holes(image, code, path, 140)
            frames[pose] = image
        if frame_delta(frames["run1"], frames["dribble"]) < 26000:
            raise AssertionError(f"{code} run animation is visually static")
        if frame_delta(frames["dribble"], frames["kick"]) < 22000:
            raise AssertionError(f"{code} kick frame is too similar to dribble frame")
        run_cx, run_cy = alpha_centroid(frames["run1"])
        kick_cx, kick_cy = alpha_centroid(frames["kick"])
        if abs(run_cx - kick_cx) > 42 or abs(run_cy - kick_cy) > 34:
            raise AssertionError(f"{code} actor identity/anchor shifts too much between run and kick")
        runner_frames = []
        for index in range(RUNNER_FRAMES):
            path = CINEMATIC_DIR / f"runner_{code}_{index}.png"
            if not path.exists():
                raise AssertionError(f"missing runner sprite: {path}")
            image = pygame.image.load(path).convert_alpha()
            if image.get_size() != RUNNER_SPRITE_SIZE:
                raise AssertionError(f"{path} has unexpected size {image.get_size()}")
            if any(image.get_at(point).a > 4 for point in ((0, 0), (255, 0), (0, 255), (255, 255))):
                raise AssertionError(f"{path} has non-transparent corners")
            if chroma_leak_count(image) > 24:
                raise AssertionError(f"{path} still has visible chroma-key leakage")
            if opaque_chroma_artifact_count(image) > 42:
                raise AssertionError(f"{path} has opaque chroma-key artifact blobs")
            bbox = alpha_bbox(image)
            if bbox.h < 190 or bbox.w < 110:
                raise AssertionError(f"{path} has too little visible runner subject: {bbox}")
            assert_actor_feet_and_legs_uncropped(image, str(path), min_lower_height=96, min_bottom_margin=5, max_bottom_margin=14)
            lower_body = image.subsurface(pygame.Rect(0, int(image.get_height() * 0.52), image.get_width(), int(image.get_height() * 0.42))).copy()
            if alpha_bbox(lower_body).h < 76:
                raise AssertionError(f"{path} has weak runner leg visibility")
            if bbox.x <= 4 or bbox.y <= 4 or bbox.right >= 252 or bbox.bottom >= 252:
                raise AssertionError(f"{path} appears clipped: {bbox}")
            if bbox.bottom < 230:
                raise AssertionError(f"{path} foot anchor is too high: {bbox}")
            detached = [(size, rect) for size, rect in alpha_components(image)[1:] if size > 80]
            if detached:
                raise AssertionError(f"{path} has detached alpha fragments: {detached}")
            assert_oracle_mark(image, code, path, 36)
            assert_green_uniform_alpha(image, code, path)
            if code == "gold":
                assert_gold_blue_shorts(image, path, 120)
            assert_no_light_short_holes(image, code, path, 140)
            runner_frames.append(image)
        assert_animation_bbox_stability(runner_frames, f"{code} runner loop")
        assert_animation_color_stability(runner_frames, f"{code} runner loop")
        runner_deltas = [frame_delta(a, b) for a, b in zip(runner_frames, runner_frames[1:])]
        if min(runner_deltas) < 28000:
            raise AssertionError(f"{code} runner loop is visually static: {runner_deltas}")

        left_pose_frames = []
        for pose in POSES:
            path = CINEMATIC_DIR / f"left_{code}_{pose}.png"
            image = pygame.image.load(path).convert_alpha()
            if image.get_size() != POSE_SPRITE_SIZE:
                raise AssertionError(f"{path} has unexpected size {image.get_size()}")
            if chroma_leak_count(image) > 20:
                raise AssertionError(f"{path} still has visible chroma-key leakage")
            if opaque_chroma_artifact_count(image) > 28:
                raise AssertionError(f"{path} has opaque chroma-key artifact blobs")
            bbox = alpha_bbox(image)
            if bbox.x <= 3 or bbox.y <= 3 or bbox.right >= image.get_width() - 3 or bbox.bottom >= image.get_height() - 3:
                raise AssertionError(f"{path} appears clipped: {bbox}")
            assert_actor_feet_and_legs_uncropped(image, str(path), min_lower_height=76, min_bottom_margin=8, max_bottom_margin=24)
            assert_oracle_mark(image, code, path, 5)
            assert_green_uniform_alpha(image, code, path)
            assert_no_light_short_holes(image, code, path, 140)
            left_pose_frames.append(image)
        if frame_delta(left_pose_frames[1], left_pose_frames[2]) < 22000:
            raise AssertionError(f"{code} left-facing pose animation is visually static")
        left_runner_frames = []
        for index in range(RUNNER_FRAMES):
            path = CINEMATIC_DIR / f"runner_left_{code}_{index}.png"
            image = pygame.image.load(path).convert_alpha()
            if image.get_size() != RUNNER_SPRITE_SIZE:
                raise AssertionError(f"{path} has unexpected size {image.get_size()}")
            if chroma_leak_count(image) > 24:
                raise AssertionError(f"{path} still has visible chroma-key leakage")
            if opaque_chroma_artifact_count(image) > 42:
                raise AssertionError(f"{path} has opaque chroma-key artifact blobs")
            bbox = alpha_bbox(image)
            if bbox.h < 190 or bbox.w < 110:
                raise AssertionError(f"{path} has too little visible left runner subject: {bbox}")
            assert_actor_feet_and_legs_uncropped(image, str(path), min_lower_height=96, min_bottom_margin=5, max_bottom_margin=14)
            if bbox.x <= 4 or bbox.y <= 4 or bbox.right >= 252 or bbox.bottom >= 252:
                raise AssertionError(f"{path} appears clipped: {bbox}")
            assert_oracle_mark(image, code, path, 36)
            assert_green_uniform_alpha(image, code, path)
            assert_no_light_short_holes(image, code, path, 140)
            left_runner_frames.append(image)
        assert_animation_bbox_stability(left_runner_frames, f"{code} left runner loop")
        assert_animation_color_stability(left_runner_frames, f"{code} left runner loop")
        left_runner_deltas = [frame_delta(a, b) for a, b in zip(left_runner_frames, left_runner_frames[1:])]
        if min(left_runner_deltas) < 26000:
            raise AssertionError(f"{code} left runner loop is visually static: {left_runner_deltas}")

    keeper_frames = []
    keeper_bottoms = []
    for index in range(KEEPER_FRAMES):
        path = CINEMATIC_DIR / f"keeper_anim_{index}.png"
        if not path.exists():
            raise AssertionError(f"missing goalkeeper animation frame: {path}")
        image = pygame.image.load(path).convert_alpha()
        if image.get_size() != KEEPER_SPRITE_SIZE:
            raise AssertionError(f"{path} has unexpected size {image.get_size()}")
        if any(image.get_at(point).a > 4 for point in ((0, 0), (287, 0), (0, 287), (287, 287))):
            raise AssertionError(f"{path} has non-transparent corners")
        if opaque_chroma_artifact_count(image) > 20:
            raise AssertionError(f"{path} has opaque chroma-key artifact blobs")
        bbox = alpha_bbox(image)
        if bbox.h < 118 or bbox.w < 98:
            raise AssertionError(f"{path} has too little goalkeeper subject: {bbox}")
        if bbox.x <= 3 or bbox.y <= 3 or bbox.right >= 285 or bbox.bottom >= 285:
            raise AssertionError(f"{path} appears clipped: {bbox}")
        centroid_x, _centroid_y = alpha_centroid(image)
        if abs(centroid_x - image.get_width() / 2) > 30:
            raise AssertionError(f"{path} goalkeeper visual center is off-pivot: cx={centroid_x:.1f}")
        keeper_bottoms.append(bbox.bottom)
        keeper_frames.append(image)
    if max(keeper_bottoms) - min(keeper_bottoms) > 18:
        raise AssertionError(f"goalkeeper feet baseline jumps between animation frames: {keeper_bottoms}")
    if frame_delta(keeper_frames[0], keeper_frames[2]) < 36000:
        raise AssertionError("goalkeeper dive animation is visually static")

    goal_frames = []
    for index in range(GOAL_NET_FRAMES):
        path = CINEMATIC_DIR / f"goal_net_{index}.png"
        if not path.exists():
            raise AssertionError(f"missing goal net animation frame: {path}")
        image = pygame.image.load(path).convert_alpha()
        if image.get_size() != (360, 240):
            raise AssertionError(f"{path} has unexpected size {image.get_size()}")
        if any(image.get_at(point).a > 4 for point in ((0, 0), (359, 0), (0, 239), (359, 239))):
            raise AssertionError(f"{path} has non-transparent corners")
        bbox = alpha_bbox(image)
        if bbox.w < 260 or bbox.h < 145:
            raise AssertionError(f"{path} has too little visible 3D goal subject: {bbox}")
        mesh_sample = image.subsurface(pygame.Rect(72, 54, 216, 112)).copy()
        if bright_pixel_count(mesh_sample) < 170:
            raise AssertionError(f"{path} has too little visible central net mesh")
        goal_frames.append(image)
    if frame_delta(goal_frames[0], goal_frames[2]) < 28000:
        raise AssertionError("goal net animation is visually static")
    front_frames = []
    for index in range(GOAL_NET_FRAMES):
        path = CINEMATIC_DIR / f"goal_front_{index}.png"
        if not path.exists():
            raise AssertionError(f"missing generated goal front frame: {path}")
        image = pygame.image.load(path).convert_alpha()
        if image.get_size() != (360, 240):
            raise AssertionError(f"{path} has unexpected size {image.get_size()}")
        bbox = alpha_bbox(image)
        if bbox.w < 210 or bbox.h < 145:
            raise AssertionError(f"{path} has too little visible generated front-post subject: {bbox}")
        front_alpha = alpha_pixel_count(image)
        net_alpha = alpha_pixel_count(goal_frames[index])
        inner_alpha = alpha_pixel_count(image.subsurface(pygame.Rect(85, 55, 190, 115)).copy())
        if front_alpha / max(1, net_alpha) > 0.46:
            raise AssertionError(f"{path} still behaves like duplicated net: front/net={front_alpha / max(1, net_alpha):.2f}")
        if inner_alpha > 620:
            raise AssertionError(f"{path} leaks too much inner mesh into the front-post layer: inner={inner_alpha}")
        front_frames.append(image)
    impact_frames = []
    for index in range(GOAL_NET_FRAMES):
        path = CINEMATIC_DIR / f"goal_impact_{index}.png"
        if not path.exists():
            raise AssertionError(f"missing generated goal impact frame: {path}")
        image = pygame.image.load(path).convert_alpha()
        if image.get_size() != (240, 180):
            raise AssertionError(f"{path} has unexpected size {image.get_size()}")
        bbox = alpha_bbox(image)
        if bbox.w < 130 or bbox.h < 90:
            raise AssertionError(f"{path} has too little visible generated net impact: {bbox}")
        impact_frames.append(image)
    if frame_delta(impact_frames[0], impact_frames[2]) < 18000:
        raise AssertionError("generated goal impact net animation is visually static")
    actual_balls = {path.name for path in BALL_DIR.glob("*.png")}
    expected_balls = {f"ball_{index}.png" for index in range(8)}
    if actual_balls != expected_balls:
        raise AssertionError(f"ball runtime sprite inventory mismatch; extra={sorted(actual_balls - expected_balls)}, missing={sorted(expected_balls - actual_balls)}")
    ball_areas = []
    ball_centers = []
    for index in range(8):
        path = BALL_DIR / f"ball_{index}.png"
        image = pygame.image.load(path).convert_alpha()
        if image.get_size() != (128, 128):
            raise AssertionError(f"{path} has unexpected size {image.get_size()}")
        bbox = alpha_bbox(image)
        if not (72 <= bbox.w <= 116 and 72 <= bbox.h <= 116):
            raise AssertionError(f"{path} has invalid ball crop: {bbox}")
        ball_areas.append(sum(1 for y in range(image.get_height()) for x in range(image.get_width()) if image.get_at((x, y)).a > 25))
        ball_centers.append(bbox.center)
    if max(ball_areas) / max(1, min(ball_areas)) > 1.12:
        raise AssertionError(f"ball alpha area changes enough to read as flicker: {ball_areas}")
    if max(x for x, _y in ball_centers) - min(x for x, _y in ball_centers) > 4:
        raise AssertionError(f"ball visual center jumps horizontally: {ball_centers}")
    if max(y for _x, y in ball_centers) - min(y for _x, y in ball_centers) > 4:
        raise AssertionError(f"ball visual center jumps vertically: {ball_centers}")


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


def high_score_prediction() -> Prediction:
    return Prediction(
        algorithm="CONFRONTO",
        home=0.56,
        draw=0.10,
        away=0.34,
        home_goals=4.0,
        away_goals=3.0,
        confidence=0.62,
        reason="high-score cinematic collision validation",
        home_advances=0.58,
        away_advances=0.42,
        top_scores=((4, 3, 0.06), (3, 3, 0.05), (4, 2, 0.05)),
        over_25=0.92,
        btts=0.86,
        score_home=4,
        score_away=3,
        outcome_class=0,
        outcome_probability=0.56,
        score_probability=0.05,
        blend_probs=(0.56, 0.10, 0.34),
        poisson_outcome_probs=(0.52, 0.14, 0.34),
    )


def assert_point_in_field(field: pygame.Rect, point: tuple[float, float], label: str) -> None:
    if not field.inflate(-8, 0).collidepoint((int(point[0]), int(point[1]))):
        raise AssertionError(f"{label} outside cinematic field: {point}")


def bright_pixel_count(surface: pygame.Surface) -> int:
    pixels = 0
    for y in range(0, surface.get_height(), 2):
        for x in range(0, surface.get_width(), 2):
            color = surface.get_at((x, y))
            if color.a > 30 and color.r > 210 and color.g > 210 and color.b > 185:
                pixels += 1
    return pixels


def dark_pixel_count(surface: pygame.Surface, step: int = 2) -> int:
    pixels = 0
    for y in range(0, surface.get_height(), step):
        for x in range(0, surface.get_width(), step):
            color = surface.get_at((x, y))
            if color.r < 42 and color.g < 48 and color.b < 42:
                pixels += 1
    return pixels


def logo_pixel_count(surface: pygame.Surface, dark_text: bool, step: int = 1) -> int:
    pixels = 0
    for y in range(0, surface.get_height(), step):
        for x in range(0, surface.get_width(), step):
            color = surface.get_at((x, y))
            if color.a <= 20:
                continue
            if dark_text:
                if color.r < 92 and color.g < 92 and color.b < 92:
                    pixels += 1
            elif color.r > 195 and color.g > 205 and color.b > 210:
                pixels += 1
    return pixels


def logo_geometry(surface: pygame.Surface, dark_text: bool, step: int = 1) -> dict[str, float]:
    points = []
    for y in range(0, surface.get_height(), step):
        for x in range(0, surface.get_width(), step):
            color = surface.get_at((x, y))
            if color.a <= 20:
                continue
            if dark_text:
                active = color.r < 92 and color.g < 92 and color.b < 92
            else:
                active = color.r > 195 and color.g > 205 and color.b > 210
            if active:
                points.append((x, y))
    if not points:
        return {"pixels": 0.0, "width": 0.0, "height": 0.0, "density": 0.0, "columns": 0.0, "rows": 0.0, "gaps": 0.0}
    min_x = min(x for x, _y in points)
    max_x = max(x for x, _y in points)
    min_y = min(y for _x, y in points)
    max_y = max(y for _x, y in points)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    columns = [False] * width
    rows = [False] * height
    for x, y in points:
        columns[x - min_x] = True
        rows[y - min_y] = True
    gaps = 0
    in_gap = False
    inner_columns = columns[1:-1] if len(columns) > 2 else columns
    for active in inner_columns:
        if not active and not in_gap:
            gaps += 1
            in_gap = True
        elif active:
            in_gap = False
    return {
        "pixels": float(len(points)),
        "width": float(width),
        "height": float(height),
        "density": len(points) / max(1, width * height),
        "columns": sum(columns) / max(1, len(columns)),
        "rows": sum(rows) / max(1, len(rows)),
        "gaps": float(gaps),
    }


def assert_runtime_logo_geometry(surface: pygame.Surface, dark_text: bool, label: str) -> None:
    geometry = logo_geometry(surface, dark_text)
    if geometry["width"] < 24:
        raise AssertionError(f"runtime ORACLE mark is too narrow to read for {label}: {geometry}")
    if not 0.015 <= geometry["density"] <= 0.68:
        raise AssertionError(f"runtime ORACLE mark has invalid wordmark density for {label}: {geometry}")
    if geometry["columns"] + geometry["rows"] < 0.40:
        raise AssertionError(f"runtime ORACLE mark has broken stroke coverage for {label}: {geometry}")


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


def vertical_transition_spike_ratio(surface: pygame.Surface, step_y: int = 5) -> float:
    transitions = []
    for x in range(2, surface.get_width() - 2):
        total = 0
        samples = 0
        for y in range(0, surface.get_height(), step_y):
            left = surface.get_at((x - 1, y))
            right = surface.get_at((x, y))
            total += abs(left.r - right.r) + abs(left.g - right.g) + abs(left.b - right.b)
            samples += 1
        transitions.append(total / max(1, samples))
    if not transitions:
        return 0.0
    ordered = sorted(transitions)
    median = ordered[len(ordered) // 2]
    return max(transitions) / max(1.0, median)


def horizontal_transition_spike_ratio(surface: pygame.Surface, step_x: int = 5) -> float:
    transitions = []
    for y in range(2, surface.get_height() - 2):
        total = 0
        samples = 0
        for x in range(0, surface.get_width(), step_x):
            top = surface.get_at((x, y - 1))
            bottom = surface.get_at((x, y))
            total += abs(top.r - bottom.r) + abs(top.g - bottom.g) + abs(top.b - bottom.b)
            samples += 1
        transitions.append(total / max(1, samples))
    if not transitions:
        return 0.0
    ordered = sorted(transitions)
    median = ordered[len(ordered) // 2]
    return max(transitions) / max(1.0, median)


def assert_goal_render_visible(app: App, field: pygame.Rect, pred: Prediction, side: str) -> None:
    first_goal_minute = app.goal_schedule(pred)[0][0]
    app.t = (first_goal_minute + 1) / 90 * 45
    app.screen.fill((0, 0, 0))
    app.draw_field(pred, pred, "CONFRONTO")
    goal = app.cinematic_goal_rect(field, "right" if side == "home" else "left")
    sample_rect = goal.inflate(92, 96).clip(field)
    sample = app.screen.subsurface(sample_rect).copy()
    if bright_pixel_count(sample) < 500:
        raise AssertionError(f"{side} goal render has too little visible net/ball/frame detail")


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
    shot_progress = float(state["shot_progress"])
    team_code = getattr(team, "code")
    frames = app.assets.cinematic_keeper_frames[team_code]
    index, scale, angle = app.cinematic_keeper_animation_state(True, shot_progress, len(frames), flip)
    frame = frames[index]
    if flip:
        frame = pygame.transform.flip(frame, True, False)
    frame = pygame.transform.rotozoom(frame, angle, scale)
    keeper_x, keeper_y = state["keeper_pos"]  # type: ignore[misc]
    return frame, frame.get_rect(center=(int(keeper_x), int(keeper_y)))


def cinematic_goal_layer_target_rect(app: App, goal: pygame.Rect, side: str, ripple: float, as_front: bool) -> pygame.Rect:
    frames = app.assets.goal_front_frames if as_front else app.assets.goal_net_frames
    frame_float = clamp(ripple) * (len(frames) - 1)
    frame_index = int(math.floor(frame_float))
    frame = app.orient_cinematic_goal_frame(frames[frame_index], side)
    aspect = frame.get_width() / max(1, frame.get_height())
    direction = -1 if side == "left" else 1
    wave = math.sin(clamp(ripple) * math.pi)
    scale_pulse = wave * (0.0 if as_front else 0.016)
    target_h = int(round((goal.h * (1.22 + scale_pulse)) / 2) * 2)
    target_w = int(target_h * aspect)
    target = pygame.Rect(0, 0, target_w, target_h)
    target.midbottom = goal.midbottom
    target.move_ip(-18 if side == "right" else 18, -4)
    if ripple > 0.02 and not as_front:
        target.move_ip(int(direction * wave * 4), int(wave))
    return target


def assert_ball_draw_sequence_stable(app: App, field: pygame.Rect, pred: Prediction, times: tuple[float, ...], label: str) -> None:
    original_ball = app.draw_cinematic_ball
    scales = []
    rendered_sizes = []
    phases = []
    visibility = []
    try:
        for time_value in times:
            app.t = time_value
            calls = []

            def spy_ball(*args: object, **kwargs: object) -> object:
                calls.append((args, kwargs))
                return original_ball(*args, **kwargs)

            app.draw_cinematic_ball = spy_ball  # type: ignore[method-assign]
            app.screen.fill((0, 0, 0))
            app.draw_field(pred, pred, "CONFRONTO")
            if len(calls) != 1:
                raise AssertionError(f"{label} ball blink guard expected one draw per frame, got {len(calls)} at t={time_value:.3f}")
            args, kwargs = calls[0]
            pos = args[0]
            if not isinstance(pos, tuple):
                raise AssertionError(f"{label} ball draw did not receive a screen position: {pos}")
            scale = int(kwargs.get("scale", CINEMATIC_BALL_SIZE))
            squash = kwargs.get("squash", (1.0, 1.0))
            if not isinstance(squash, tuple):
                squash = (1.0, 1.0)
            size_x = max(22, int(scale * float(squash[0])))
            size_y = max(22, int(scale * float(squash[1])))
            scales.append(scale)
            rendered_sizes.append((size_x, size_y))
            phases.append(str(kwargs.get("phase", "")))
            sample_rect = pygame.Rect(0, 0, size_x + 12, size_y + 12)
            sample_rect.center = (int(pos[0]), int(pos[1]))
            sample = app.screen.subsurface(sample_rect.clip(field)).copy()
            visibility.append(bright_pixel_count(sample) + dark_pixel_count(sample, step=3))
    finally:
        app.draw_cinematic_ball = original_ball  # type: ignore[method-assign]
    if min(visibility) < 80 or min(visibility) < max(1, int(max(visibility) * 0.30)):
        raise AssertionError(f"{label} ball visibility flickers across frames: {visibility}, phases={phases}")
    scale_steps = [abs(a - b) for a, b in zip(scales, scales[1:])]
    if scale_steps and max(scale_steps) > 3:
        raise AssertionError(f"{label} ball scale changes abruptly between frames: {scales}")
    size_steps = [
        max(abs(ax - bx), abs(ay - by))
        for (ax, ay), (bx, by) in zip(rendered_sizes, rendered_sizes[1:])
    ]
    if size_steps and max(size_steps) > 5:
        raise AssertionError(f"{label} ball rendered size pops between frames: {rendered_sizes}")


def validate_aaa_player_crop_gate() -> None:
    app = App(seed=2026)
    field = pygame.Rect(32, 110, 910, 490)
    ground_y = field.bottom - 54
    neutral_cases = (
        ("neutral-run", "runner", 0),
        ("neutral-run", "runner", 2),
        ("neutral-run", "runner", 3),
        ("neutral-pose", "pose", 1),
        ("neutral-idle", "pose", 0),
    )
    for code in ROWS:
        for prefix, side_label, x in (("", "home/right-facing", field.centerx - 154), ("left_", "away/left-facing", field.centerx + 154)):
            for pose in ("idle", "run1", "kick"):
                path = CINEMATIC_DIR / f"{prefix}{code}_{pose}.png"
                image = pygame.image.load(path).convert_alpha()
                assert_actor_feet_and_legs_uncropped(image, f"{side_label} {code} {pose}", 76, 8, 24)
                target = app.cinematic_actor_target_size(image, CINEMATIC_PLAYER_SCALE)
                rect = pygame.Rect(0, 0, *target)
                rect.midbottom = (x, ground_y)
                if not field.contains(rect):
                    raise AssertionError(f"{side_label} {code} {pose} runtime rect clips feet/legs: {rect} outside {field}")
            for label, frame_kind, index in neutral_cases:
                if frame_kind == "runner":
                    path = CINEMATIC_DIR / f"runner_{prefix}{code}_{index}.png"
                    image = pygame.image.load(path).convert_alpha()
                    assert_actor_feet_and_legs_uncropped(image, f"{side_label} {code} {label}{index}", 96, 5, 14)
                else:
                    path = CINEMATIC_DIR / f"{prefix}{code}_{'run1' if index == 1 else 'idle'}.png"
                    image = pygame.image.load(path).convert_alpha()
                    assert_actor_feet_and_legs_uncropped(image, f"{side_label} {code} {label}", 76, 8, 24)
                target = app.cinematic_actor_target_size(image, CINEMATIC_NEUTRAL_PLAYER_SCALE)
                rect = pygame.Rect(0, 0, *target)
                rect.midbottom = (x, ground_y)
                if not field.contains(rect):
                    raise AssertionError(f"{side_label} {code} {label} runtime neutral rect clips feet/legs: {rect} outside {field}")


def validate_cinematic_scene() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)
    pred = home_win_prediction()
    first_goal_minute = app.goal_schedule(pred)[0][0]

    runner = app.assets.cinematic_runners[app.home.code][0]
    kick = app.assets.cinematic_players[app.home.code][3]
    runner_target = app.cinematic_actor_target_size(runner, CINEMATIC_PLAYER_SCALE)
    kick_target = app.cinematic_actor_target_size(kick, CINEMATIC_PLAYER_SCALE)
    expected_height = int(CINEMATIC_POSE_SIZE * CINEMATIC_PLAYER_SCALE)
    runner_visible = scaled_visible_height(runner, runner_target)
    kick_visible = scaled_visible_height(kick, kick_target)
    if abs(runner_visible - expected_height) > 1.2 or abs(kick_visible - expected_height) > 1.2:
        raise AssertionError(f"runner/kick visible bbox must share the 192 pose standard: {runner_visible:.1f} vs {kick_visible:.1f}")
    if abs(runner_visible - kick_visible) > 1.6:
        raise AssertionError(f"runner and kick visible scale pop: {runner_visible:.1f} vs {kick_visible:.1f}")

    def set_goal_progress(goal_minute: int, progress: float) -> dict[str, object]:
        app.t = (goal_minute - 5.0 + progress * 5.0) / 90.0 * 45.0
        return app.cinematic_scene_state(field, pred)

    def ball_distance_to_kick(state: dict[str, object]) -> float:
        return math.dist(state["ball_pos"], state["kick_pos"])  # type: ignore[arg-type]

    for second in (0, 9, 18, 31, 44):
        app.t = float(second)
        state = app.cinematic_scene_state(field, pred)
        if state["neutral"]:
            raise AssertionError("home-win cinematic should not become neutral")
        assert_point_in_field(field, state["actor_pos"], "attacker")
        assert_point_in_field(field, state["ball_pos"], "ball")
        assert_point_in_field(field, state["keeper_pos"], "keeper")
        goal = state["goal_rect"]
        if not field.contains(goal):
            raise AssertionError(f"goal outside field: {goal}")

    for minute in (2, max(3, first_goal_minute - 8)):
        app.t = minute / 90 * 45
        state = app.cinematic_scene_state(field, pred)
        if float(state["actor_pos"][0]) > field.centerx:
            raise AssertionError(f"attacker crossed midfield before goal movement: {state['actor_pos']}")

    app.t = first_goal_minute / 90 * 45
    state = app.cinematic_scene_state(field, pred)
    if float(state["kick_pos"][0]) <= field.centerx:
        raise AssertionError(f"attacker foot/ball did not cross midfield during goal movement: {state['kick_pos']}")
    assert_goal_render_visible(app, field, pred, "home")

    shot_state = set_goal_progress(first_goal_minute, 0.60)
    kick_x, kick_y = shot_state["kick_pos"]  # type: ignore[misc]
    keeper_x, keeper_y = shot_state["keeper_pos"]  # type: ignore[misc]
    actor_x, actor_y = shot_state["actor_pos"]  # type: ignore[misc]
    kick_frame = app.assets.cinematic_players[app.home.code][3]
    kick_target = app.cinematic_actor_target_size(kick_frame, CINEMATIC_PLAYER_SCALE)
    expected_kick = app.cinematic_actor_anchor_screen(
        (field.centerx + 46 - 68, actor_y),
        kick_target,
        KICK_FOOT_ANCHOR,
        False,
        kick_frame,
    )
    if 0.60 <= SHOT_KICK_AT:
        settle_to_kick = smoothstep((0.60 - SHOT_PLANT_AT) / max(0.001, SHOT_KICK_AT - SHOT_PLANT_AT))
        roll_cycle = math.sin(0.60 * 38.0 + app.match_seed * 0.01)
        dribble_foot = (
            actor_x + 50 + roll_cycle * 4,
            actor_y - 42 + math.cos(0.60 * 32.0) * 2,
        )
        expected_kick = (
            dribble_foot[0] + (expected_kick[0] - dribble_foot[0]) * settle_to_kick,
            dribble_foot[1] + (expected_kick[1] - dribble_foot[1]) * settle_to_kick,
        )
    if math.dist((kick_x, kick_y), expected_kick) > 1.0:
        raise AssertionError(f"kick point is not using the sprite foot anchor: {(kick_x, kick_y)} vs {expected_kick}")
    if not field.centerx + 34 <= kick_x <= field.centerx + 66:
        raise AssertionError(f"home shot should start just past midfield, got kick_x={kick_x:.1f}")
    if math.dist((kick_x, kick_y), (keeper_x, keeper_y)) < 145:
        raise AssertionError("home shot starts too close to the goalkeeper")
    actor_y_samples = []
    actor_x_samples = []
    for progress in (0.46, 0.56, 0.66, 0.70):
        sample_state = set_goal_progress(first_goal_minute, progress)
        actor_x, actor_y = sample_state["actor_pos"]  # type: ignore[misc]
        actor_x_samples.append(actor_x)
        actor_y_samples.append(actor_y)
    if max(actor_y_samples) - min(actor_y_samples) > 8:
        raise AssertionError(f"attacker appears to float during shot: y samples={actor_y_samples}")
    if abs(actor_x_samples[-1] - actor_x_samples[1]) > 14:
        raise AssertionError(f"attacker keeps sprinting after planting foot: x samples={actor_x_samples}")
    if ball_distance_to_kick(set_goal_progress(first_goal_minute, 0.555)) > 14:
        raise AssertionError("ball is not attached to foot before the shot")
    if ball_distance_to_kick(set_goal_progress(first_goal_minute, 0.66)) < 35:
        raise AssertionError("ball did not leave the foot after the shot")
    for progress in (0.62, 0.70, 0.78, 0.86, 0.90, 0.92, 0.94, 0.955):
        sample_state = set_goal_progress(first_goal_minute, progress)
        goal_rect = sample_state["goal_rect"]
        ball_x, ball_y = sample_state["ball_pos"]  # type: ignore[misc]
        if isinstance(goal_rect, pygame.Rect) and goal_rect.collidepoint(int(ball_x), int(ball_y)):
            raise AssertionError(f"ball entered the goal mouth too early in the flight at progress={progress:.2f}: {(ball_x, ball_y)} in {goal_rect}")
    post_net_state = set_goal_progress(first_goal_minute, SHOT_NET_VISUAL_CONTACT_AT)
    post_goal_rect = post_net_state["goal_rect"]
    post_ball_x, post_ball_y = post_net_state["ball_pos"]  # type: ignore[misc]
    if not isinstance(post_goal_rect, pygame.Rect) or not post_goal_rect.collidepoint(int(post_ball_x), int(post_ball_y)):
        raise AssertionError(f"ball does not enter the net after impact: {(post_ball_x, post_ball_y)} in {post_goal_rect}")
    if float(post_net_state.get("net_progress", 0.0)) <= 0.0:
        raise AssertionError("net ripple did not start once the ball entered the net")
    flight_samples = [set_goal_progress(first_goal_minute, progress)["ball_pos"] for progress in (0.62, 0.70, 0.78, 0.86, 0.94)]
    for current, following in zip(flight_samples, flight_samples[1:]):
        if following[0] + 2 < current[0]:
            raise AssertionError(f"home ball flight is not monotonic toward goal: {flight_samples}")
    home_dive_state = set_goal_progress(first_goal_minute, 0.86)
    home_goal = home_dive_state["goal_rect"]
    if not isinstance(home_goal, pygame.Rect):
        raise AssertionError("home goal rect missing during keeper dive")
    home_target = app.cinematic_shot_target(home_goal, 1, first_goal_minute)
    home_keeper_base_x = home_goal.centerx
    home_keeper_x = float(home_dive_state["keeper_pos"][0])  # type: ignore[index]
    if (home_keeper_x - home_keeper_base_x) * (home_target[0] - home_keeper_base_x) <= 0:
        raise AssertionError(f"home goalkeeper dives away from ball: base={home_keeper_base_x:.1f}, keeper={home_keeper_x:.1f}, target={home_target[0]:.1f}")

    away_pred = away_win_prediction()
    first_away_goal = app.goal_schedule(away_pred)[0][0]
    app.t = 2 / 90 * 45
    away_state = app.cinematic_scene_state(field, away_pred)
    if float(away_state["actor_pos"][0]) < field.centerx:
        raise AssertionError(f"away attacker crossed midfield before goal movement: {away_state['actor_pos']}")
    app.t = first_away_goal / 90 * 45
    away_state = app.cinematic_scene_state(field, away_pred)
    if float(away_state["kick_pos"][0]) >= field.centerx:
        raise AssertionError(f"away attacker foot/ball did not cross midfield during goal movement: {away_state['kick_pos']}")
    def set_away_goal_progress(progress: float) -> dict[str, object]:
        app.t = (first_away_goal - 5.0 + progress * 5.0) / 90.0 * 45.0
        return app.cinematic_scene_state(field, away_pred)

    away_shot = set_away_goal_progress(0.60)
    away_kick_x, away_kick_y = away_shot["kick_pos"]  # type: ignore[misc]
    away_keeper_x, away_keeper_y = away_shot["keeper_pos"]  # type: ignore[misc]
    if not field.centerx - 66 <= away_kick_x <= field.centerx - 34:
        raise AssertionError(f"away shot should start just past midfield, got kick_x={away_kick_x:.1f}")
    if math.dist((away_kick_x, away_kick_y), (away_keeper_x, away_keeper_y)) < 145:
        raise AssertionError("away shot starts too close to the goalkeeper")
    away_dive_state = set_away_goal_progress(0.86)
    away_goal = away_dive_state["goal_rect"]
    if not isinstance(away_goal, pygame.Rect):
        raise AssertionError("away goal rect missing during keeper dive")
    away_target = app.cinematic_shot_target(away_goal, -1, first_away_goal)
    away_keeper_base_x = away_goal.centerx
    away_keeper_x = float(away_dive_state["keeper_pos"][0])  # type: ignore[index]
    if (away_keeper_x - away_keeper_base_x) * (away_target[0] - away_keeper_base_x) <= 0:
        raise AssertionError(f"away goalkeeper dives away from ball: base={away_keeper_base_x:.1f}, keeper={away_keeper_x:.1f}, target={away_target[0]:.1f}")
    for goal_minute, side in app.goal_schedule(away_pred):
        target_goal = app.cinematic_goal_rect(field, "right" if side == "home" else "left")
        for minute in (goal_minute, goal_minute + 2):
            app.t = minute / 90 * 45
            state = app.cinematic_scene_state(field, away_pred)
            ball = state["ball_pos"]
            if not target_goal.collidepoint((int(ball[0]), int(ball[1]))):
                raise AssertionError(f"away cinematic ball misses goal at {minute}': {ball}")
    assert_goal_render_visible(app, field, away_pred, "away")

    draw_pred = neutral_prediction()
    for second in (0, 15, 30, 40):
        app.t = float(second)
        if app.cinematic_scene_state(field, draw_pred)["neutral"]:
            raise AssertionError(f"draw cinematic became neutral too early at {second}s")

    for goal_minute, side in app.goal_schedule(pred):
        target_goal = app.cinematic_goal_rect(field, "right" if side == "home" else "left")
        for minute in (goal_minute, goal_minute + 2):
            app.t = minute / 90 * 45
            state = app.cinematic_scene_state(field, pred)
            ball = state["ball_pos"]
            if not target_goal.collidepoint((int(ball[0]), int(ball[1]))):
                raise AssertionError(f"cinematic ball misses goal at {minute}': {ball}")
        samples = []
        for step in range(120):
            minute = goal_minute - 5 + step * 8 / 119
            app.t = minute / 90 * 45
            ball = app.cinematic_scene_state(field, pred)["ball_pos"]
            samples.append(ball)
        for a, b in zip(samples, samples[1:]):
            distance = math.dist(a, b)
            if distance > 44:
                raise AssertionError(f"cinematic ball jumps too far between frames: {distance:.1f}px")

    app.t = neutral_sample_second(0.85)
    neutral = app.cinematic_scene_state(field, draw_pred)
    if not neutral["neutral"]:
        raise AssertionError("draw cinematic should finish neutral")
    ball = neutral["ball_pos"]
    neutral_ground_y = field.bottom - 54
    expected_neutral_ball_y = neutral_ground_y - CINEMATIC_BALL_SIZE * 0.43
    if abs(ball[0] - field.centerx) > 12 or abs(ball[1] - expected_neutral_ball_y) > 18:
        raise AssertionError(f"neutral ball is not centered: {ball}")
    assert_point_in_field(field, neutral["home_pos"], "neutral home player")
    assert_point_in_field(field, neutral["away_pos"], "neutral away player")
    for team in (app.home, app.away):
        neutral_frame = app.neutral_frame_for_phase(team, 0.0, 1.0, False)
        neutral_target = app.cinematic_actor_target_size(neutral_frame, CINEMATIC_NEUTRAL_PLAYER_SCALE)
        neutral_visible_h = scaled_visible_height(neutral_frame, neutral_target)
        if abs(neutral_visible_h - CINEMATIC_POSE_SIZE) > 1.2:
            raise AssertionError(f"neutral player does not render at visible 192px standard: {team.code}={neutral_visible_h:.1f}")

    neutral_frame_indices = []
    neutral_field_samples = []
    for second in tuple(neutral_sample_second(position) for position in (0.14, 0.50, 0.88)):
        app.t = second
        state = app.cinematic_scene_state(field, draw_pred)
        neutral_frame_indices.append(int(float(state["home_stride_phase"])) % 4)
        app.screen.fill((0, 0, 0))
        app.draw_field(draw_pred, draw_pred, "CONFRONTO")
        neutral_field_samples.append(app.screen.subsurface(pygame.Rect(field.centerx - 250, field.y + 236, 500, 270)).copy())
    if len(set(neutral_frame_indices)) < 2:
        raise AssertionError(f"neutral draw cinematic is not changing animation frames: {neutral_frame_indices}")
    if frame_sample_delta(neutral_field_samples[0], neutral_field_samples[-1], step=5) < 18000:
        raise AssertionError("neutral draw cinematic is visually static")

    app.screen.fill((0, 0, 0))
    app.t = (first_goal_minute + 1) / 90 * 45
    ball_calls = 0
    ball_scales = []
    original_ball = app.draw_cinematic_ball

    def count_ball(*args, **kwargs):
        nonlocal ball_calls
        ball_calls += 1
        if "scale" in kwargs:
            ball_scales.append(kwargs["scale"])
        return original_ball(*args, **kwargs)

    app.draw_cinematic_ball = count_ball
    app.draw_field(pred, pred, "CONFRONTO")
    app.draw_cinematic_ball = original_ball
    if ball_calls != 1:
        raise AssertionError(f"cinematic render drew {ball_calls} balls instead of one")
    if not ball_scales or min(ball_scales) < CINEMATIC_BALL_SIZE - 10:
        raise AssertionError(f"cinematic ball is too small: {ball_scales}")
    if max(ball_scales) - min(ball_scales) > 12:
        raise AssertionError(f"cinematic ball changed size abruptly: {ball_scales}")
    if app.screen.subsurface(field).get_bounding_rect().w <= 0:
        raise AssertionError("cinematic render is blank")

    app.screen.fill((0, 0, 0))
    seek_match_time(app, pred, 6.0)
    app.draw_cinematic_background(field, pred)
    early_ground = app.screen.subsurface(pygame.Rect(field.x, field.y + 332, field.w, 120)).copy()
    early_scroll = app.ground_scroll
    app.screen.fill((0, 0, 0))
    seek_match_time(app, pred, 12.0)
    app.draw_cinematic_background(field, pred)
    later_ground = app.screen.subsurface(pygame.Rect(field.x, field.y + 332, field.w, 120)).copy()
    if abs(app.ground_scroll) <= abs(early_scroll) + 8:
        raise AssertionError(f"parallax scroll is not accumulated/eased over time: early={early_scroll:.1f}, later={app.ground_scroll:.1f}")
    if frame_sample_delta(early_ground, later_ground) < 42000:
        raise AssertionError("cinematic parallax turf layer is too static")
    for label, sample in (("early", early_ground), ("later", later_ground)):
        if vertical_transition_spike_ratio(sample) > 12.0:
            raise AssertionError(f"cinematic parallax turf has a visible vertical seam in {label} sample")
        if horizontal_transition_spike_ratio(sample) > 9.0:
            raise AssertionError(f"cinematic parallax turf has a visible horizontal band in {label} sample")


def validate_aaa_cinematic_design_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)
    pred = home_win_prediction()
    first_goal_minute = app.goal_schedule(pred)[0][0]

    runner = app.assets.cinematic_runners[app.home.code][0]
    kick = app.assets.cinematic_players[app.home.code][3]
    runner_target = app.cinematic_actor_target_size(runner, CINEMATIC_PLAYER_SCALE)
    kick_target = app.cinematic_actor_target_size(kick, CINEMATIC_PLAYER_SCALE)
    player_h = scaled_visible_height(runner, runner_target)
    player_w = scaled_visible_width(runner, runner_target)
    kick_h = scaled_visible_height(kick, kick_target)

    assert_between(player_h / field.h, 0.39, 0.46, "player height versus field")
    assert_between(abs(player_h - kick_h), 0.0, 1.6, "run to kick visual scale pop")
    assert_between(CINEMATIC_BALL_SIZE / player_h, 0.24, 0.30, "ball size versus player")

    shadow_w = max(74, int(player_w * 0.84))
    shadow_h = int(13 * CINEMATIC_PLAYER_SCALE)
    assert_between(shadow_w / player_w, 0.78, 1.02, "player shadow width versus body")
    assert_between(shadow_h / player_h, 0.045, 0.085, "player shadow height versus body")

    keeper = app.assets.cinematic_keeper_frames[app.away.code][0]
    keeper_target = (int(keeper.get_width() * CINEMATIC_KEEPER_SCALE), int(keeper.get_height() * CINEMATIC_KEEPER_SCALE))
    keeper_h = scaled_visible_height(keeper, keeper_target)
    assert_between(keeper_h / player_h, 0.82, 1.08, "goalkeeper size versus player")

    goal = app.cinematic_goal_rect(field, "right")
    if not field.contains(goal):
        raise AssertionError(f"AAA goal frame leaves the field: {goal}")
    assert_between(goal.w / player_h, 0.82, 1.02, "goal mouth width versus player")
    assert_between(goal.h / player_h, 0.86, 1.05, "goal mouth height versus player")

    goal_frame = app.assets.goal_net_frames[0]
    target_h = int(goal.h * 1.22)
    target_w = int(goal_frame.get_width() * target_h / goal_frame.get_height())
    goal_sprite_rect = pygame.Rect(0, 0, target_w, target_h)
    goal_sprite_rect.midbottom = goal.midbottom
    goal_sprite_rect.move_ip(-18, -4)
    if not field.contains(goal_sprite_rect):
        raise AssertionError(f"rendered 3D goal sprite is clipped by field: {goal_sprite_rect}")
    assert_between(goal_sprite_rect.w / player_h, 1.05, 1.70, "rendered 3D goal width versus player")
    assert_between(goal_sprite_rect.h / player_h, 1.02, 1.18, "rendered 3D goal height versus player")

    def side_alpha(surface: pygame.Surface, left_side: bool) -> int:
        width, height = surface.get_size()
        x_range = range(0, width // 3) if left_side else range(width * 2 // 3, width)
        total = 0
        for y in range(height):
            for x in x_range:
                if surface.get_at((x, y)).a > 80:
                    total += 1
        return total

    left_goal_frame = app.orient_cinematic_goal_frame(goal_frame, "left")
    right_goal_frame = app.orient_cinematic_goal_frame(goal_frame, "right")
    if side_alpha(left_goal_frame, True) <= side_alpha(left_goal_frame, False):
        raise AssertionError("left-side 3D goal is inverted; depth must point away from midfield")
    if side_alpha(right_goal_frame, False) <= side_alpha(right_goal_frame, True):
        raise AssertionError("right-side 3D goal is inverted; depth must point away from midfield")

    def goal_progress_state(progress: float) -> dict[str, object]:
        app.t = (first_goal_minute - 5.0 + progress * 5.0) / 90.0 * 45.0
        return app.cinematic_scene_state(field, pred)

    for progress in (0.0, 0.40, 0.54, 0.60, 0.72):
        state = goal_progress_state(progress)
        kick_window, stride = app.cinematic_stride_state(progress, float(state["stride_phase"]))
        frame = kick if kick_window else app.assets.cinematic_runners[app.home.code][stride]
        target = app.cinematic_actor_target_size(frame, CINEMATIC_PLAYER_SCALE)
        actor_x, actor_y = state["actor_pos"]  # type: ignore[misc]
        planted_x = app.cinematic_planted_x(actor_x, stride, kick_window, 1, float(state["run_speed"]))
        actor_rect = pygame.Rect(0, 0, *target)
        actor_rect.midbottom = (int(planted_x), int(actor_y))
        if not field.contains(actor_rect):
            raise AssertionError(f"player sprite is clipped during AAA sequence at {progress:.2f}: {actor_rect}")
    for progress in (0.54, 0.62, 0.68, SHOT_FOLLOW_THROUGH_HOLD_END - 0.01):
        impact_kick_window, _stride = app.cinematic_stride_state(progress, 0.0)
        if not impact_kick_window:
            raise AssertionError(f"player must hold kick/follow-through through impact at progress={progress:.3f}")
    for progress in (SHOT_FOLLOW_THROUGH_HOLD_END + 0.03, 0.84, 0.90):
        late_kick_window, _stride = app.cinematic_stride_state(progress, 0.0)
        if late_kick_window:
            raise AssertionError(f"player remains frozen in kick pose during ball flight at progress={progress:.3f}")

    keeper_distances = []
    shot_target = app.cinematic_shot_target(goal, 1, first_goal_minute)
    for progress in (0.52, 0.70, 0.86, SHOT_NET_VISUAL_CONTACT_AT):
        state = goal_progress_state(progress)
        keeper_x, keeper_y = state["keeper_pos"]  # type: ignore[misc]
        _keeper_render, keeper_rect = goalkeeper_render_for_state(app, app.away, state, False)
        if not field.contains(keeper_rect):
            raise AssertionError(f"goalkeeper sprite is clipped during AAA sequence at {progress:.2f}: {keeper_rect}")
        keeper_distances.append(math.dist((keeper_x, keeper_y), shot_target))

        ball_x, ball_y = state["ball_pos"]  # type: ignore[misc]
        ball_size = int(state["ball_scale"])
        ball_rect = pygame.Rect(0, 0, ball_size, ball_size)
        ball_rect.center = (int(ball_x), int(ball_y))
        if not field.contains(ball_rect):
            raise AssertionError(f"ball sprite is clipped during AAA sequence at {progress:.2f}: {ball_rect}")
        if progress >= SHOT_NET_VISUAL_CONTACT_AT and not goal.collidepoint(ball_rect.center):
            raise AssertionError(f"ball does not finish inside the net: ball={ball_rect}, goal={goal}")

    if keeper_distances[0] < 18.0:
        if max(keeper_distances[1:]) > 18.0:
            raise AssertionError(f"central goalkeeper reaction drifts away from the shot target: {keeper_distances}")
    elif min(keeper_distances[1:]) > keeper_distances[0] * 0.70:
        raise AssertionError(f"goalkeeper does not dive toward the shot target: {keeper_distances}")


def validate_aaa_screen_composition_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)
    pred = home_win_prediction()
    first_goal_minute = app.goal_schedule(pred)[0][0]

    def render_goal_progress(prediction: Prediction, goal_minute: int, progress: float) -> dict[str, object]:
        app.match_prediction = prediction
        app.t = (goal_minute - 5.0 + progress * 5.0) / 90.0 * 45.0
        state = app.cinematic_scene_state(field, prediction)
        app.screen.fill((0, 0, 0))
        app.draw_field(prediction, prediction, "CONFRONTO")
        return state

    contact = render_goal_progress(pred, first_goal_minute, SHOT_KICK_AT + 0.005)
    actor_x, actor_y = contact["actor_pos"]  # type: ignore[misc]
    shadow = pygame.Rect(0, 0, int(132 * CINEMATIC_PLAYER_SCALE), int(15 * CINEMATIC_PLAYER_SCALE))
    shadow.center = (int(actor_x), int(actor_y - 1))
    shadow_sample = app.screen.subsurface(shadow.inflate(16, 10).clip(field)).copy()
    if dark_pixel_count(shadow_sample) < 180:
        raise AssertionError("AAA screen gate: player shadow is too weak or detached from the turf")

    ball_x, ball_y = contact["ball_pos"]  # type: ignore[misc]
    ball_rect = pygame.Rect(0, 0, int(contact["ball_scale"]), int(contact["ball_scale"]))
    ball_rect.center = (int(ball_x), int(ball_y))
    player_ball_distance = math.dist(contact["kick_pos"], contact["ball_pos"])  # type: ignore[arg-type]
    if player_ball_distance > 24 or not field.contains(ball_rect):
        raise AssertionError(f"AAA screen gate: ball contact reads wrong, distance={player_ball_distance:.1f}, ball={ball_rect}")

    impact = render_goal_progress(pred, first_goal_minute, SHOT_NET_VISUAL_CONTACT_AT)
    goal = impact["goal_rect"]
    if not isinstance(goal, pygame.Rect):
        raise AssertionError("AAA screen gate: missing goal rect at impact")
    goal_sample = app.screen.subsurface(goal.inflate(118, 112).clip(field)).copy()
    if bright_pixel_count(goal_sample) < 800:
        raise AssertionError("AAA screen gate: generated 3D goal/net is not visually strong enough at impact")
    if dark_pixel_count(goal_sample) < 220:
        raise AssertionError("AAA screen gate: goal scene lacks contact shadows/depth at impact")
    impact_ball_x, impact_ball_y = impact["ball_pos"]  # type: ignore[misc]
    impact_ball = pygame.Rect(0, 0, int(impact["ball_scale"]), int(impact["ball_scale"]))
    impact_ball.center = (int(impact_ball_x), int(impact_ball_y))
    if not goal.collidepoint(impact_ball.center):
        raise AssertionError(f"AAA screen gate: impact ball is not inside the net, ball={impact_ball}, goal={goal}")
    ball_sample = app.screen.subsurface(impact_ball.inflate(12, 12).clip(field)).copy()
    if bright_pixel_count(ball_sample) < 96:
        raise AssertionError(f"AAA screen gate: ball is not independently readable at net impact: {impact_ball}")
    if edge_energy(ball_sample) < 18:
        raise AssertionError(f"AAA screen gate: ball lost panel/texture definition at net impact: {impact_ball}")
    post_sample = app.screen.subsurface(goal.inflate(8, 8).clip(field)).copy()
    if bright_pixel_count(post_sample) < 160 or dark_pixel_count(post_sample) < 760:
        raise AssertionError("AAA screen gate: trave/postes do not keep independent depth at impact")
    local_impact = pygame.Rect(0, 0, 120, 92)
    local_impact.center = impact_ball.center
    local_impact = local_impact.clip(field)
    if bright_pixel_count(app.screen.subsurface(local_impact).copy()) < 160:
        raise AssertionError(f"AAA screen gate: net impact lacks local burst/ripple around the ball: {local_impact}")
    text = app.f_lg.render("GOOOL!", True, (255, 255, 255))
    overlay = pygame.Rect(0, 0, text.get_width() + 54, text.get_height() + 22)
    overlay.center = app.cinematic_goal_overlay_center(field)
    if overlay.colliderect(goal.inflate(90, 80)):
        raise AssertionError(f"AAA screen gate: GOOOL overlay covers the goal/net payoff: overlay={overlay}, goal={goal}")

    order_app = App(seed=2026)
    order_app.set_simulate("match")
    order_app.match_prediction = pred
    order_app.t = (first_goal_minute - 5.0 + SHOT_NET_VISUAL_CONTACT_AT * 5.0) / 90.0 * 45.0
    render_order: list[str] = []
    order_app.draw_cinematic_runner = lambda *args, **kwargs: render_order.append("runner")  # type: ignore[method-assign]
    order_app.draw_cinematic_kick_impact = lambda *args, **kwargs: render_order.append("kick")  # type: ignore[method-assign]
    order_app.draw_cinematic_ball = lambda *args, **kwargs: render_order.append("ball")  # type: ignore[method-assign]
    order_app.draw_cinematic_keeper = lambda *args, **kwargs: render_order.append("keeper")  # type: ignore[method-assign]
    original_goal_3d = order_app.draw_cinematic_goal_3d
    original_goal_front_posts = order_app.draw_cinematic_goal_front_posts
    original_goal_impact = order_app.draw_cinematic_goal_impact

    def spy_goal_3d(*args, **kwargs):
        render_order.append("goal_front" if kwargs.get("front_only") else "goal_back")
        return original_goal_3d(*args, **kwargs)

    def spy_goal_front_posts(*args, **kwargs):
        render_order.append("goal_front")
        return original_goal_front_posts(*args, **kwargs)

    def spy_goal_impact(*args, **kwargs):
        render_order.append("impact")
        return original_goal_impact(*args, **kwargs)

    order_app.draw_cinematic_goal_3d = spy_goal_3d  # type: ignore[method-assign]
    order_app.draw_cinematic_goal_front_posts = spy_goal_front_posts  # type: ignore[method-assign]
    order_app.draw_cinematic_goal_impact = spy_goal_impact  # type: ignore[method-assign]
    order_app.draw_field(pred, pred, "CONFRONTO")
    required_layers = {"goal_back", "runner", "ball", "keeper", "goal_front"}
    missing_layers = sorted(required_layers - set(render_order))
    if missing_layers:
        raise AssertionError(f"AAA screen gate: missing active goal render layers {missing_layers}: {render_order}")
    actor_indexes = [render_order.index(layer) for layer in ("runner", "ball", "keeper")]
    if render_order.index("goal_back") >= min(actor_indexes):
        raise AssertionError(f"AAA screen gate: goal net/back layer must render before actors and ball: {render_order}")
    if render_order.index("goal_front") <= render_order.index("runner"):
        raise AssertionError(
            f"AAA screen gate: front posts must render after the runner enters the goal scene: {render_order}"
        )
    if render_order.index("goal_front") >= render_order.index("keeper"):
        raise AssertionError(
            f"AAA screen gate: goalkeeper must render in front of the goal posts: {render_order}"
        )
    if "impact" in render_order and render_order.index("impact") <= render_order.index("ball"):
        raise AssertionError(
            f"AAA screen gate: net impact should render after the ball reaches the net: {render_order}"
        )

    away_pred = away_win_prediction()
    first_away_goal = app.goal_schedule(away_pred)[0][0]
    away_impact = render_goal_progress(away_pred, first_away_goal, SHOT_NET_VISUAL_CONTACT_AT)
    away_goal = away_impact["goal_rect"]
    if not isinstance(away_goal, pygame.Rect):
        raise AssertionError("AAA screen gate: missing away goal rect at impact")
    away_sample = app.screen.subsurface(away_goal.inflate(118, 112).clip(field)).copy()
    if bright_pixel_count(away_sample) < 800:
        raise AssertionError("AAA screen gate: away-side goal/net loses visual quality when flipped")

    draw_pred = neutral_prediction()
    neutral_samples = []
    for seconds in tuple(neutral_sample_second(position) for position in (0.16, 0.56, 0.92)):
        app.match_prediction = draw_pred
        app.t = seconds
        state = app.cinematic_scene_state(field, draw_pred)
        app.screen.fill((0, 0, 0))
        app.draw_field(draw_pred, draw_pred, "CONFRONTO")
        home_x, home_y = state["home_pos"]  # type: ignore[misc]
        away_x, away_y = state["away_pos"]  # type: ignore[misc]
        if abs(home_y - away_y) > 8:
            raise AssertionError(f"AAA screen gate: neutral players are not grounded consistently: {home_y:.1f} vs {away_y:.1f}")
        neutral_samples.append(app.screen.subsurface(pygame.Rect(field.centerx - 280, field.y + 222, 560, 292)).copy())
    if frame_sample_delta(neutral_samples[0], neutral_samples[-1], step=5) < 19000:
        raise AssertionError("AAA screen gate: final draw cinematic is too static on screen")


def validate_aaa_away_draw_ball_stability_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)
    away_pred = away_win_prediction()
    app.match_prediction = away_pred
    first_away_goal = app.goal_schedule(away_pred)[0][0]

    actor_x_samples = []
    stride_indices = []
    away_samples = []
    for progress in (0.08, 0.18, 0.28, 0.38, 0.48, 0.54):
        app.t = (first_away_goal - 5.0 + progress * 5.0) / 90.0 * SIMULATION_SECONDS
        state = app.cinematic_scene_state(field, away_pred)
        if state["possession"] != "away":
            raise AssertionError(f"away cinematic lost visitor possession at progress={progress:.2f}: {state['possession']}")
        actor_x, _actor_y = state["actor_pos"]  # type: ignore[misc]
        actor_x_samples.append(float(actor_x))
        kick_window, stride = app.cinematic_stride_state(float(state["shot_progress"]), float(state.get("stride_phase", 0.0)))
        if not kick_window:
            stride_indices.append(stride)
        app.screen.fill((0, 0, 0))
        app.draw_field(away_pred, away_pred, "CONFRONTO")
        away_samples.append(app.screen.subsurface(pygame.Rect(field.centerx - 360, field.y + 198, 720, 286)).copy())
    if actor_x_samples[-1] >= actor_x_samples[0] - 34:
        raise AssertionError(f"away attacker does not move fluidly toward the left-side goal: {actor_x_samples}")
    if len(set(stride_indices)) < 2:
        raise AssertionError(f"away runner animation is not cycling through visitor frames: {stride_indices}")
    away_deltas = [frame_sample_delta(a, b, step=5) for a, b in zip(away_samples, away_samples[1:])]
    if min(away_deltas[:3]) < 12000:
        raise AssertionError(f"away visitor animation is visually static before the kick: {away_deltas}")
    if len(away_deltas) >= 4:
        cadence = [abs(current - following) for current, following in zip(away_deltas, away_deltas[1:])]
        if max(cadence[:4]) > max(22000, min(away_deltas[:4]) * 2.8):
            raise AssertionError(f"away runner cadence jumps enough to read as perceptual flicker: deltas={away_deltas}")
    for progress in (0.54, 0.70, SHOT_NET_VISUAL_CONTACT_AT):
        app.t = (first_away_goal - 5.0 + progress * 5.0) / 90.0 * SIMULATION_SECONDS
        state = app.cinematic_scene_state(field, away_pred)
        app.screen.fill((0, 0, 0))
        app.draw_field(away_pred, away_pred, "CONFRONTO")
        actor_x, actor_y = state["actor_pos"]  # type: ignore[misc]
        foot_zone = pygame.Rect(int(actor_x) - 82, int(actor_y) - 78, 164, 82).clip(field)
        if foot_zone.w <= 0 or foot_zone.h <= 0:
            raise AssertionError(f"away visitor foot zone is outside field at progress={progress:.2f}: {foot_zone}")
        foot_sample = app.screen.subsurface(foot_zone).copy()
        if dark_pixel_count(foot_sample, step=2) < 95:
            raise AssertionError(f"away visitor feet/legs are not readable near left-side action at progress={progress:.2f}: {foot_zone}")

    away_flight = []
    for progress in (0.62, 0.70, 0.78, 0.86, 0.94):
        app.t = (first_away_goal - 5.0 + progress * 5.0) / 90.0 * SIMULATION_SECONDS
        away_flight.append(app.cinematic_scene_state(field, away_pred)["ball_pos"])
    for current, following in zip(away_flight, away_flight[1:]):
        if following[0] - 2 > current[0]:
            raise AssertionError(f"away ball flight is not monotonic toward the visitor goal: {away_flight}")

    home_pred = home_win_prediction()
    first_home_goal = app.goal_schedule(home_pred)[0][0]
    shot_progress_samples = (0.46, 0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.78, 0.86, 0.88, 0.90, 0.92, 0.94, SHOT_NET_AT, 1.0)
    home_times = tuple((first_home_goal - 5.0 + progress * 5.0) / 90.0 * SIMULATION_SECONDS for progress in shot_progress_samples)
    away_times = tuple((first_away_goal - 5.0 + progress * 5.0) / 90.0 * SIMULATION_SECONDS for progress in shot_progress_samples)
    draw_pred = neutral_prediction()
    draw_times = tuple(neutral_sample_second(position) for position in (0.08, 0.20, 0.34, 0.48, 0.62, 0.78, 0.94))
    assert_ball_draw_sequence_stable(app, field, home_pred, home_times, "home goal")
    assert_ball_draw_sequence_stable(app, field, away_pred, away_times, "away goal")
    assert_ball_draw_sequence_stable(app, field, draw_pred, draw_times, "draw neutral")

    home_hashes = []
    away_hashes = []
    neutral_field_samples = []
    for seconds in draw_times:
        app.t = seconds
        state = app.cinematic_scene_state(field, draw_pred)
        if not state["neutral"]:
            raise AssertionError(f"draw cinematic should be neutral during final animation at t={seconds:.2f}")
        neutral_progress = float(state.get("neutral_progress", 1.0))
        home_frame = app.neutral_frame_for_phase(app.home, float(state.get("home_stride_phase", 0.0)), neutral_progress, False)
        away_frame = app.neutral_frame_for_phase(app.away, float(state.get("away_stride_phase", 0.0)), neutral_progress, True)
        home_hashes.append(surface_hash(home_frame))
        away_hashes.append(surface_hash(away_frame))
        app.screen.fill((0, 0, 0))
        app.draw_field(draw_pred, draw_pred, "CONFRONTO")
        neutral_field_samples.append(app.screen.subsurface(pygame.Rect(field.centerx - 280, field.y + 222, 560, 292)).copy())
    if len(set(home_hashes)) < 2 or len(set(away_hashes)) < 2:
        raise AssertionError(f"draw tie animation must advance frames for both players: home={home_hashes}, away={away_hashes}")
    neutral_deltas = [frame_sample_delta(a, b, step=5) for a, b in zip(neutral_field_samples, neutral_field_samples[1:])]
    if max(neutral_deltas) < 9000:
        raise AssertionError(f"draw tie screen frames are not visibly animated: {neutral_deltas}")


def validate_aaa_chance_outcome_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)
    predictions = (home_win_prediction(), away_win_prediction(), neutral_prediction())
    chance_events: dict[str, tuple[Prediction, tuple[int, str, str]]] = {}
    for pred in predictions:
        schedule = app.chance_schedule(pred)
        if schedule != sorted(schedule, key=lambda item: item[0]):
            raise AssertionError(f"chance schedule is not chronological: {schedule}")
        for current, following in zip(schedule, schedule[1:]):
            if following[0] - current[0] < CHANCE_MIN_SPACING_MINUTES:
                raise AssertionError(f"chance schedule bunches no-goal lances: {schedule}")
        for event in schedule:
            minute, _side, kind = event
            if kind not in {"save", "wide"}:
                raise AssertionError(f"unknown no-goal chance kind: {event}")
            chance_events.setdefault(kind, (pred, event))
            if not 8 <= minute <= 84:
                raise AssertionError(f"chance event minute outside visual range: {event}")
    if set(chance_events) != {"save", "wide"}:
        raise AssertionError(f"AAA needs both save and wide no-goal lances, got {chance_events}")
    save_minute, save_side, _save_kind = chance_events["save"][1]
    original_match_seed = app.match_seed
    save_variants = set()
    for seed in range(2020, 2030):
        app.match_seed = seed
        save_variants.add(app.cinematic_save_variant(save_minute, save_side))
    app.match_seed = original_match_seed
    if save_variants != {"stand", "dive"}:
        raise AssertionError(f"save chances must alternate standing and diving saves, got {save_variants}")

    for kind, (pred, (minute, side, _kind)) in chance_events.items():
        app.match_prediction = pred
        app.t = (minute - CHANCE_EVENT_WINDOW_MINUTES + 0.98 * CHANCE_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS
        state = app.cinematic_scene_state(field, pred)
        if state.get("active_goal"):
            raise AssertionError(f"{kind} no-goal chance was treated as a goal")
        active_attack = state.get("active_attack")
        if not active_attack or state.get("attack_kind") != kind:
            raise AssertionError(f"{kind} chance did not become the active attack: {state}")
        goal = state["goal_rect"]
        if not isinstance(goal, pygame.Rect):
            raise AssertionError(f"{kind} chance missing goal frame")
        keeper_x, keeper_y = state["keeper_pos"]  # type: ignore[misc]
        if not goal.inflate(72, 74).collidepoint(int(keeper_x), int(keeper_y)):
            raise AssertionError(f"{kind} chance goalkeeper is not staged in the goal: keeper={state['keeper_pos']}, goal={goal}")
        ball_x, ball_y = state["ball_pos"]  # type: ignore[misc]
        ball_rect = pygame.Rect(0, 0, int(state["ball_scale"]), int(state["ball_scale"]))
        ball_rect.center = (int(ball_x), int(ball_y))
        if kind == "save":
            if math.dist((ball_x, ball_y), (keeper_x, keeper_y)) > 130:
                raise AssertionError(f"save chance does not read as a goalkeeper action: ball={ball_rect}, keeper={state['keeper_pos']}")
            glove_pos = state.get("keeper_glove_pos")
            if not isinstance(glove_pos, tuple):
                raise AssertionError(f"save chance must expose a glove contact anchor: {state}")
            glove_x, glove_y = float(glove_pos[0]), float(glove_pos[1])
            glove_contact_distance = math.dist((ball_x, ball_y), (glove_x, glove_y))
            if state.get("save_variant") == "stand":
                index, _scale, angle = app.cinematic_keeper_animation_state(
                    False,
                    float(state["shot_progress"]),
                    4,
                    state.get("possession") == "away",
                    "stand_save",
                )
                if index != 0:
                    raise AssertionError(f"standing save must use the central glove pose at contact, got frame {index}")
                if abs(angle) > 1:
                    raise AssertionError(f"standing save contact should not rotate gloves away from the ball, got angle {angle}")
                if glove_contact_distance > 22:
                    raise AssertionError(
                        f"standing save ball must meet the central gloves: ball={ball_rect.center}, glove={glove_pos}, dist={glove_contact_distance:.1f}"
                    )
            if state.get("save_variant") == "dive":
                index, _scale, _angle = app.cinematic_keeper_animation_state(
                    True,
                    float(state["shot_progress"]),
                    4,
                    state.get("possession") == "away",
                    "dive_save",
                )
                if index != 2:
                    raise AssertionError(f"diving save must use the outstretched-hand pose at contact, got frame {index}")
                if glove_contact_distance > 28:
                    raise AssertionError(
                        f"diving save ball must meet the extended glove: ball={ball_rect.center}, glove={glove_pos}, dist={glove_contact_distance:.1f}"
                    )
        else:
            if goal.contains(ball_rect.inflate(-8, -8)):
                raise AssertionError(f"wide chance must miss the goal mouth: ball={ball_rect}, goal={goal}")
            near_post_pos = state.get("chance_near_post_pos")
            if not isinstance(near_post_pos, tuple):
                raise AssertionError(f"wide chance must expose a near-post anchor: {state}")
            near_post_distance = math.dist((ball_x, ball_y), (float(near_post_pos[0]), float(near_post_pos[1])))
            if not 4 <= near_post_distance <= 30:
                raise AssertionError(
                    f"wide chance must read as raspando a trave: ball={ball_rect.center}, post={near_post_pos}, dist={near_post_distance:.1f}"
                )

        goal_3d_calls: list[tuple[str, float]] = []
        keeper_calls: list[tuple[bool, str]] = []
        original_goal_3d = app.draw_cinematic_goal_3d
        original_keeper = app.draw_cinematic_keeper

        def spy_goal_3d(
            goal_rect: pygame.Rect,
            side_name: str,
            ripple: float,
            impact_pos: object | None = None,
            front_only: bool = False,
            alpha: int = 255,
        ) -> None:
            goal_3d_calls.append((side_name, ripple))
            original_goal_3d(goal_rect, side_name, ripple, impact_pos, front_only, alpha)

        def spy_keeper(*args: object, **kwargs: object) -> None:
            keeper_calls.append((bool(kwargs.get("active_goal", False)), str(kwargs.get("keeper_action", ""))))
            original_keeper(*args, **kwargs)  # type: ignore[arg-type]

        app.draw_cinematic_goal_3d = spy_goal_3d  # type: ignore[method-assign]
        app.draw_cinematic_keeper = spy_keeper  # type: ignore[method-assign]
        app.screen.fill((0, 0, 0))
        try:
            app.draw_field(pred, pred, "CONFRONTO")
        finally:
            app.draw_cinematic_goal_3d = original_goal_3d  # type: ignore[method-assign]
            app.draw_cinematic_keeper = original_keeper  # type: ignore[method-assign]
        if not goal_3d_calls:
            raise AssertionError(f"{kind} chance must show the goal/trave as visual reference")
        if any(ripple > 0.0 for _side, ripple in goal_3d_calls):
            raise AssertionError(f"{kind} no-goal chance rendered goal ripple as if it scored: {goal_3d_calls}")
        if not keeper_calls:
            raise AssertionError(f"{kind} chance must show the goalkeeper in the goal")
        if kind == "save":
            expected_action = f"{state.get('save_variant')}_save"
            if not any(action == expected_action for _active, action in keeper_calls):
                raise AssertionError(f"save chance must use the selected goalkeeper animation: {keeper_calls}, state={state}")
            if state.get("save_variant") == "dive" and not any(active for active, _action in keeper_calls):
                raise AssertionError("diving save must keep the dive animation path active")
            if state.get("save_variant") == "stand" and any(active for active, _action in keeper_calls):
                raise AssertionError("standing save should not reuse the full dive animation path")
        payoff_sample = app.screen.subsurface(ball_rect.inflate(110, 88).clip(field)).copy()
        if bright_pixel_count(payoff_sample) < 120:
            raise AssertionError(f"{kind} chance payoff effect is not visible enough")
        if dark_pixel_count(payoff_sample) < 60:
            raise AssertionError(f"{kind} chance payoff lacks grounding/depth")
        if side != state.get("possession"):
            raise AssertionError(f"{kind} chance changed possession during payoff: {side} vs {state.get('possession')}")


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

    original_goal_3d = app.draw_cinematic_goal_3d
    original_keeper = app.draw_cinematic_keeper
    goal_3d_calls: list[tuple[str, float]] = []
    keeper_calls: list[tuple[bool, str]] = []

    def spy_goal_3d(
        goal: pygame.Rect,
        side: str,
        ripple: float,
        impact_pos: object | None = None,
        front_only: bool = False,
        alpha: int = 255,
    ) -> None:
        goal_3d_calls.append((side, ripple))
        original_goal_3d(goal, side, ripple, impact_pos, front_only, alpha)

    def spy_keeper(*args: object, **kwargs: object) -> None:
        keeper_calls.append((bool(kwargs.get("active_goal", False)), str(kwargs.get("keeper_action", ""))))
        original_keeper(*args, **kwargs)  # type: ignore[arg-type]

    try:
        app.draw_cinematic_goal_3d = spy_goal_3d  # type: ignore[method-assign]
        app.draw_cinematic_keeper = spy_keeper  # type: ignore[method-assign]
        for minute, _side, kind in chance_events:
            app.t = (minute - CHANCE_EVENT_WINDOW_MINUTES + 0.98 * CHANCE_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS
            state = app.cinematic_scene_state(field, pred)
            if state.get("active_goal") or app.active_goal_event(pred) is not None:
                raise AssertionError(f"0 x 0 chance became a real goal event: {minute}, {kind}, {state}")
            if app.score_from_prediction(pred) != (0, 0):
                raise AssertionError(f"0 x 0 live score changed during a non-goal chance: {app.score_from_prediction(pred)}")
            app.screen.fill((0, 0, 0))
            goal_3d_calls.clear()
            keeper_calls.clear()
            app.draw_field(pred, pred, "CONFRONTO")
            if not goal_3d_calls:
                raise AssertionError(f"0 x 0 {kind} chance must keep the goal frame visible")
            if any(ripple > 0.0 for _side, ripple in goal_3d_calls):
                raise AssertionError(f"0 x 0 {kind} chance rendered goal-impact ripple as if it scored: {goal_3d_calls}")
            if not keeper_calls:
                raise AssertionError(f"0 x 0 {kind} chance must keep the goalkeeper visible")
            if kind == "save":
                expected_action = f"{state.get('save_variant')}_save"
                if not any(action == expected_action for _active, action in keeper_calls):
                    raise AssertionError(f"0 x 0 save must use the selected goalkeeper animation: {keeper_calls}, state={state}")
                if state.get("save_variant") == "dive" and not any(active for active, _action in keeper_calls):
                    raise AssertionError("0 x 0 diving save must keep the dive animation path active")
                if state.get("save_variant") == "stand" and any(active for active, _action in keeper_calls):
                    raise AssertionError("0 x 0 standing save should not reuse the full dive animation path")
            if app.active_goal_event(pred) is not None:
                raise AssertionError("0 x 0 non-goal chance exposed GOOOL overlay timing")
    finally:
        app.draw_cinematic_goal_3d = original_goal_3d  # type: ignore[method-assign]
        app.draw_cinematic_keeper = original_keeper  # type: ignore[method-assign]

    app.t = SIMULATION_SECONDS
    if app.score_from_prediction(pred) != (0, 0):
        raise AssertionError(f"0 x 0 final score drifted after full time: {app.score_from_prediction(pred)}")


def validate_cinematic_reveal_timing_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)
    pred = home_win_prediction()
    first_goal = app.goal_schedule(pred)[0][0]
    samples = (
        SHOT_GOAL_REVEAL_AT + 0.03,
        SHOT_KEEPER_REVEAL_AT + 0.06,
        SHOT_KICK_AT,
        SHOT_KEEPER_FULL_AT + 0.02,
    )
    goal_alphas: dict[float, list[int]] = {progress: [] for progress in samples}
    keeper_alphas: dict[float, list[int]] = {progress: [] for progress in samples}
    original_goal = app.draw_cinematic_goal_3d
    original_keeper = app.draw_cinematic_keeper
    current_progress = samples[0]

    def spy_goal(*args, **kwargs):
        goal_alphas[current_progress].append(int(kwargs.get("alpha", 255)))
        return original_goal(*args, **kwargs)

    def spy_keeper(*args, **kwargs):
        keeper_alphas[current_progress].append(int(kwargs.get("alpha", 255)))
        return original_keeper(*args, **kwargs)

    try:
        app.draw_cinematic_goal_3d = spy_goal  # type: ignore[method-assign]
        app.draw_cinematic_keeper = spy_keeper  # type: ignore[method-assign]
        for progress in samples:
            current_progress = progress
            app.t = (first_goal - 5.0 + progress * 5.0) / 90.0 * SIMULATION_SECONDS
            app.screen.fill((0, 0, 0))
            app.draw_field(pred, pred, "CONFRONTO")
    finally:
        app.draw_cinematic_goal_3d = original_goal  # type: ignore[method-assign]
        app.draw_cinematic_keeper = original_keeper  # type: ignore[method-assign]

    if app.cinematic_goal_alpha(SHOT_GOAL_REVEAL_AT - 0.01) != 0:
        raise AssertionError("goal/trave reveal starts before the declared read window")
    if app.cinematic_keeper_alpha(SHOT_KEEPER_REVEAL_AT - 0.01) != 0:
        raise AssertionError("goalkeeper reveal starts before the declared read window")

    early_goal = max(goal_alphas[samples[0]] or [0])
    kick_goal = max(goal_alphas[samples[2]] or [0])
    if not 0 < early_goal <= 70 or kick_goal < 245:
        raise AssertionError(f"goal/trave fade-in lost early readable ramp: {goal_alphas}")

    early_keeper = max(keeper_alphas[samples[0]] or [0])
    read_keeper = max(keeper_alphas[samples[1]] or [0])
    kick_keeper = max(keeper_alphas[samples[2]] or [0])
    full_keeper = max(keeper_alphas[samples[3]] or [0])
    if early_keeper > 10 or not 10 <= read_keeper <= 120 or kick_keeper < 245 or full_keeper < 245:
        raise AssertionError(f"goalkeeper fade-in is either too late or too noisy early: {keeper_alphas}")

    keeper_curve = [app.cinematic_keeper_alpha(progress) for progress in samples]
    if any(later < earlier for earlier, later in zip(keeper_curve, keeper_curve[1:])):
        raise AssertionError(f"goalkeeper alpha regressed during fade-in: {keeper_curve}")


def validate_aaa_cinematic_fade_gate() -> None:
    validate_cinematic_reveal_timing_gate()

    draw_pred = neutral_prediction()
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)
    neutral_progresses = []
    neutral_reveals = []
    neutral_centers = []
    pre_neutral_second = max(0.0, (DRAW_NEUTRAL_START_PROGRESS - 0.02) * SIMULATION_SECONDS)
    for second in (pre_neutral_second, *(neutral_sample_second(position) for position in (0.05, 0.25, 0.55, 0.90))):
        app.t = second
        state = app.cinematic_scene_state(field, draw_pred)
        if second <= pre_neutral_second and state["neutral"]:
            raise AssertionError(f"draw neutral entry starts too early at {second}s")
        if state["neutral"]:
            neutral_progresses.append(float(state.get("neutral_progress", 1.0)))
            neutral_reveals.append(float(state.get("neutral_reveal", 1.0)))
            home_x, home_y = state["home_pos"]  # type: ignore[misc]
            away_x, away_y = state["away_pos"]  # type: ignore[misc]
            neutral_centers.append(((float(home_x) + float(away_x)) * 0.5, (float(home_y) + float(away_y)) * 0.5))
    if len(neutral_progresses) < 4 or neutral_progresses[0] >= 0.15 or neutral_reveals[0] >= 0.35:
        raise AssertionError(f"neutral draw entry is not eased in: progress={neutral_progresses}, reveal={neutral_reveals}")
    if any(later < earlier for earlier, later in zip(neutral_progresses, neutral_progresses[1:])):
        raise AssertionError(f"neutral progress regressed: {neutral_progresses}")
    if any(math.dist(a, b) > 28 for a, b in zip(neutral_centers, neutral_centers[1:])):
        raise AssertionError(f"neutral entry center jumps on screen: {neutral_centers}")


def validate_aaa_goalkeeper_front_layer_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)
    for label, pred in (("home", home_win_prediction()), ("away", away_win_prediction())):
        app.match_prediction = pred
        first_goal = app.goal_schedule(pred)[0][0]
        for progress in (0.70, 0.86, SHOT_NET_VISUAL_CONTACT_AT):
            app.t = (first_goal - 5.0 + progress * 5.0) / 90.0 * SIMULATION_SECONDS
            state = app.cinematic_scene_state(field, pred)
            possession = str(state["possession"])
            keeper_team = app.away if possession == "home" else app.home
            flip_keeper = possession == "away"
            keeper_frame, keeper_rect = goalkeeper_render_for_state(app, keeper_team, state, flip_keeper)
            if not field.contains(keeper_rect):
                raise AssertionError(f"{label} goalkeeper clips before front-layer check: {keeper_rect}")

            goal = state["goal_rect"]
            if not isinstance(goal, pygame.Rect):
                raise AssertionError(f"{label} missing goal rect for front-layer check")
            render_order: list[str] = []
            original_front_posts = app.draw_cinematic_goal_front_posts
            original_keeper = app.draw_cinematic_keeper

            def spy_front_posts(*args, **kwargs):
                render_order.append("goal_front")
                return original_front_posts(*args, **kwargs)

            def spy_keeper(*args, **kwargs):
                render_order.append("keeper")
                return original_keeper(*args, **kwargs)

            app.draw_cinematic_goal_front_posts = spy_front_posts  # type: ignore[method-assign]
            app.draw_cinematic_keeper = spy_keeper  # type: ignore[method-assign]
            app.screen.fill((0, 0, 0))
            try:
                app.draw_field(pred, pred, "CONFRONTO")
            finally:
                app.draw_cinematic_goal_front_posts = original_front_posts  # type: ignore[method-assign]
                app.draw_cinematic_keeper = original_keeper  # type: ignore[method-assign]
            if "goal_front" in render_order and "keeper" in render_order and render_order.index("goal_front") >= render_order.index("keeper"):
                raise AssertionError(f"{label} goalkeeper must render in front of the goal posts: {render_order}")
            keeper_sample = app.screen.subsurface(keeper_rect.clip(field)).copy()
            visible_score = bright_pixel_count(keeper_sample) + dark_pixel_count(keeper_sample, step=3)
            if visible_score < 900 or edge_energy(keeper_sample) < 30:
                raise AssertionError(f"{label} goalkeeper loses visual presence after goal-front draw: visible={visible_score}, edge={edge_energy(keeper_sample):.1f}")


def validate_cinematic_temporal_stability() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)
    pred = home_win_prediction()
    app.match_prediction = pred
    first_goal_minute = app.goal_schedule(pred)[0][0]
    actor_y_samples = []
    ball_visibility = []
    pre_kick_visibility = []
    previous_ball = None
    for index in range(25):
        progress = 0.46 + index * 0.01
        app.t = (first_goal_minute - 5.0 + progress * 5.0) / 90.0 * 45.0
        state = app.cinematic_scene_state(field, pred)
        actor_y_samples.append(float(state["actor_pos"][1]))  # type: ignore[index]
        ball_x, ball_y = state["ball_pos"]  # type: ignore[misc]
        kick_x, kick_y = state["kick_pos"]  # type: ignore[misc]
        if progress <= SHOT_KICK_AT and math.dist((ball_x, ball_y), (kick_x, kick_y)) > 28:
            raise AssertionError(f"ball flickers away from foot before kick at progress={progress:.2f}")
        if previous_ball is not None:
            jump = math.dist(previous_ball, (ball_x, ball_y))
            if progress <= SHOT_KICK_AT and jump > 16:
                raise AssertionError(f"ball jumps while attached to foot: progress={progress:.2f}, jump={jump:.1f}")
            if progress > SHOT_KICK_AT and jump > 34:
                raise AssertionError(f"ball flight has an abrupt visual jump: progress={progress:.2f}, jump={jump:.1f}")
        previous_ball = (ball_x, ball_y)

        app.screen.fill((0, 0, 0))
        app.draw_field(pred, pred, "CONFRONTO")
        ball_size = int(state["ball_scale"])
        sample_rect = pygame.Rect(0, 0, ball_size + 10, ball_size + 10)
        sample_rect.center = (int(ball_x), int(ball_y))
        sample = app.screen.subsurface(sample_rect.clip(field)).copy()
        visible = bright_pixel_count(sample) + dark_pixel_count(sample, step=3)
        ball_visibility.append(visible)
        if progress <= SHOT_KICK_AT:
            pre_kick_visibility.append(visible)
    if max(actor_y_samples) - min(actor_y_samples) > 8:
        raise AssertionError(f"runner vertical baseline flickers during kick: {actor_y_samples}")
    if min(pre_kick_visibility) < max(1, int(max(pre_kick_visibility) * 0.70)):
        raise AssertionError(f"ball visibility flickers near foot/shot: {ball_visibility}")


def validate_aaa_ball_physics_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)

    def state_at(pred: Prediction, goal_minute: int, progress: float) -> dict[str, object]:
        app.match_prediction = pred
        app.t = (goal_minute - 5.0 + progress * 5.0) / 90.0 * 45.0
        return app.cinematic_scene_state(field, pred)

    for label, pred in (("home", home_win_prediction()), ("away", away_win_prediction())):
        goal_minute, _side = app.goal_schedule(pred)[0]
        for progress in (0.50, 0.54, 0.555):
            state = state_at(pred, goal_minute, progress)
            ball = state["ball_pos"]  # type: ignore[assignment]
            kick = state["kick_pos"]  # type: ignore[assignment]
            if math.dist(ball, kick) > 18:
                raise AssertionError(f"{label} dribble ball is not physically attached to the foot at {progress:.3f}: {ball} vs {kick}")

        arc_samples = [state_at(pred, goal_minute, progress) for progress in (SHOT_KICK_AT, 0.66, 0.78, 0.90, 0.955)]
        arc_y = [float(state["ball_pos"][1]) for state in arc_samples]  # type: ignore[index]
        apex_index = min(range(len(arc_y)), key=arc_y.__getitem__)
        if apex_index in {0, len(arc_y) - 1} or arc_y[apex_index] > min(arc_y[0], arc_y[-1]) - 24 or arc_y[-1] < arc_y[apex_index] + 16:
            raise AssertionError(f"{label} shot has no readable loft/descent arc: y={arc_y}")
        if float(arc_samples[0]["ball_spin_rate"]) < 30 or float(arc_samples[2]["ball_spin_rate"]) < 26:  # type: ignore[index]
            raise AssertionError(f"{label} ball spin is too low for a driven shot")

        samples = [state_at(pred, goal_minute, 0.58 + index * 0.035) for index in range(11)]
        direction = 1 if samples[0]["goal_side"] == "right" else -1
        xs = [float(state["ball_pos"][0]) for state in samples]  # type: ignore[index]
        for current, following in zip(xs, xs[1:]):
            if (following - current) * direction < -3:
                raise AssertionError(f"{label} ball path reverses during flight: {xs}")
        scales = [int(state["ball_scale"]) for state in samples]
        for current, following in zip(scales, scales[1:]):
            if abs(following - current) > 3:
                raise AssertionError(f"{label} ball scale pops during flight: {scales}")
        squash_ratios = []
        for state in [*samples, state_at(pred, goal_minute, SHOT_NET_VISUAL_CONTACT_AT), state_at(pred, goal_minute, 1.0)]:
            squash = state.get("ball_squash", (1.0, 1.0))
            if not isinstance(squash, tuple):
                raise AssertionError(f"{label} ball squash metadata is not a tuple: {squash}")
            sx, sy = max(0.001, float(squash[0])), max(0.001, float(squash[1]))
            squash_ratios.append(max(sx / sy, sy / sx))
        if max(squash_ratios) > 1.06:
            raise AssertionError(f"{label} ball becomes visibly oval during shot: ratios={squash_ratios}")

        net_start = state_at(pred, goal_minute, SHOT_NET_AT)
        net_start_goal = net_start["goal_rect"]
        net_start_ball = net_start["ball_pos"]  # type: ignore[assignment]
        if isinstance(net_start_goal, pygame.Rect) and net_start_goal.collidepoint(int(net_start_ball[0]), int(net_start_ball[1])):
            raise AssertionError(f"{label} ball visually enters the net before the synced impact cue")
        synced_impact = state_at(pred, goal_minute, SHOT_NET_VISUAL_CONTACT_AT)
        synced_goal = synced_impact["goal_rect"]
        synced_ball = synced_impact["ball_pos"]  # type: ignore[assignment]
        if not isinstance(synced_goal, pygame.Rect) or not synced_goal.collidepoint(int(synced_ball[0]), int(synced_ball[1])):
            raise AssertionError(f"{label} ball does not enter the net on the synced impact cue")

    draw_pred = neutral_prediction()
    app.match_prediction = draw_pred
    previous_home_x = None
    previous_away_x = None
    previous_ball = None
    for progress in tuple(DRAW_NEUTRAL_START_PROGRESS + DRAW_NEUTRAL_RAMP * position for position in (0.08, 0.26, 0.46, 0.68, 0.92)):
        app.t = progress * SIMULATION_SECONDS
        state = app.cinematic_scene_state(field, draw_pred)
        if not state.get("neutral"):
            raise AssertionError(f"draw cinematic should be neutral during final settle: progress={progress}")
        home_x = float(state["home_pos"][0])  # type: ignore[index]
        away_x = float(state["away_pos"][0])  # type: ignore[index]
        ball = state["ball_pos"]  # type: ignore[assignment]
        if previous_home_x is not None:
            if home_x + 1 < previous_home_x or away_x - 1 > previous_away_x:  # type: ignore[operator]
                raise AssertionError("draw cinematic players reverse instead of easing toward the center")
            if math.dist(previous_ball, ball) > 18:  # type: ignore[arg-type]
                raise AssertionError("draw cinematic ball jumps instead of rolling into the center")
        previous_home_x = home_x
        previous_away_x = away_x
        previous_ball = ball


def validate_ball_physics_contract_fast() -> None:
    field = pygame.Rect(32, 110, 910, 490)

    target_zones: set[tuple[int, int, int]] = set()
    target_app = App(seed=2026)
    target_app.set_simulate("match")
    for seed in range(2026, 2040):
        target_app.match_seed = seed
        for direction in (-1, 1):
            side = "right" if direction > 0 else "left"
            goal = target_app.cinematic_goal_rect(field, side)
            for minute in (18, 31, 47, 66, 82):
                profile = target_app.cinematic_shot_profile(goal, direction, minute)
                target_x, target_y = profile.target
                ball_margin = CINEMATIC_BALL_SIZE * 0.46
                if not goal.inflate(-int(ball_margin * 2), -int(ball_margin * 2)).collidepoint(int(target_x), int(target_y)):
                    raise AssertionError(f"shot target is unsafe for full ball read: seed={seed}, target={profile.target}, goal={goal}")
                zone_x = int(clamp((target_x - goal.left) / max(1, goal.w), 0, 0.999) * 4)
                zone_y = int(clamp((target_y - goal.top) / max(1, goal.h), 0, 0.999) * 4)
                target_zones.add((direction, zone_x, zone_y))
                if not (70.0 <= abs(profile.entry[0] - profile.target[0]) <= 170.0):
                    raise AssertionError(f"shot entry depth is not plausible: {profile}")
    if len(target_zones) < 9:
        raise AssertionError(f"shot targets are not diverse enough: zones={sorted(target_zones)}")

    def state_at(app: App, pred: Prediction, goal_minute: int, progress: float) -> dict[str, object]:
        app.match_prediction = pred
        app.t = (goal_minute - GOAL_EVENT_WINDOW_MINUTES + progress * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS
        return app.cinematic_scene_state(field, pred)

    for label, app, pred in (
        ("home", App(seed=2026), home_win_prediction()),
        ("away", App(seed=2031), away_win_prediction()),
    ):
        app.set_simulate("match")
        goal_minute, _side = app.goal_schedule(pred)[0]
        release_state = state_at(app, pred, goal_minute, SHOT_KICK_AT)
        release = release_state["ball_pos"]  # type: ignore[assignment]
        target = release_state.get("shot_target")
        if not isinstance(target, tuple):
            raise AssertionError(f"{label} shot target missing from cinematic state")
        vx = float(target[0]) - float(release[0])
        vy = float(target[1]) - float(release[1])
        denom = max(1.0, vx * vx + vy * vy)
        prev_progress = -0.05
        prev_ball = release
        lateral_offsets: list[float] = []
        for progress in (0.58, 0.62, 0.66, 0.70, 0.76, 0.82, 0.88, 0.93, 0.955, 0.970, SHOT_NET_VISUAL_CONTACT_AT):
            state = state_at(app, pred, goal_minute, progress)
            ball = state["ball_pos"]  # type: ignore[assignment]
            bx = float(ball[0]) - float(release[0])
            by = float(ball[1]) - float(release[1])
            projected = (bx * vx + by * vy) / denom
            if projected + 0.025 < prev_progress:
                raise AssertionError(f"{label} shot regresses toward target: progress={progress:.3f}, projected={projected:.3f}, prev={prev_progress:.3f}")
            prev_progress = projected
            if math.dist(prev_ball, ball) > 52:
                raise AssertionError(f"{label} shot teleports between dense samples at {progress:.3f}: {prev_ball} -> {ball}")
            prev_ball = ball
            cross = abs((bx * vy - by * vx) / math.sqrt(denom))
            lateral_offsets.append(cross)
            scale = int(state["ball_scale"])
            squash = state.get("ball_squash", (1.0, 1.0))
            sx, sy = float(squash[0]), float(squash[1])  # type: ignore[index]
            if abs(sx - sy) > 0.015 or not (40 <= scale <= 52):
                raise AssertionError(f"{label} ball should stay round and stable: scale={scale}, squash={squash}")
        if max(lateral_offsets) < 8:
            raise AssertionError(f"{label} shot has no readable bend/loft deviation: offsets={lateral_offsets}")

        before_net = state_at(app, pred, goal_minute, SHOT_NET_AT)
        before_ball = before_net["ball_pos"]  # type: ignore[assignment]
        before_scale = int(before_net["ball_scale"])
        before_rect = pygame.Rect(0, 0, before_scale, before_scale)
        before_rect.center = (int(before_ball[0]), int(before_ball[1]))
        goal = before_net["goal_rect"]
        if isinstance(goal, pygame.Rect) and goal.contains(before_rect.inflate(-before_scale // 3, -before_scale // 3)):
            raise AssertionError(f"{label} ball enters the net before visual impact: ball={before_rect}, goal={goal}")

        impact = state_at(app, pred, goal_minute, SHOT_NET_VISUAL_CONTACT_AT)
        impact_ball = impact["ball_pos"]  # type: ignore[assignment]
        impact_scale = int(impact["ball_scale"])
        impact_rect = pygame.Rect(0, 0, impact_scale, impact_scale)
        impact_rect.center = (int(impact_ball[0]), int(impact_ball[1]))
        goal = impact["goal_rect"]
        if not isinstance(goal, pygame.Rect) or not goal.inflate(-6, -6).colliderect(impact_rect):
            raise AssertionError(f"{label} ball does not read inside the net at impact: ball={impact_rect}, goal={goal}")


def validate_aaa_findings_light_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)

    def time_for_progress(goal_minute: int, progress: float) -> float:
        return (goal_minute - GOAL_EVENT_WINDOW_MINUTES + progress * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS

    def state_at(pred: Prediction, goal_minute: int, progress: float) -> dict[str, object]:
        app.match_prediction = pred
        app.t = time_for_progress(goal_minute, progress)
        return app.cinematic_scene_state(field, pred)

    high_pred = high_score_prediction()
    app.match_prediction = high_pred
    high_goals = app.goal_schedule(high_pred)
    final_score = app.final_score_from_prediction(high_pred)
    if len(high_goals) != sum(final_score):
        raise AssertionError(f"high-score cinematic schedule lost goals: score={final_score}, schedule={high_goals}")
    attack_windows: list[tuple[float, float, str, int, str]] = []
    for minute, side in high_goals:
        start = minute - GOAL_EVENT_WINDOW_MINUTES
        impact = start + SHOT_NET_VISUAL_CONTACT_AT * GOAL_EVENT_WINDOW_MINUTES
        attack_windows.append((start, impact, "goal", minute, side))
    for minute, side, kind in app.chance_schedule(high_pred):
        start = minute - CHANCE_EVENT_WINDOW_MINUTES
        impact = start + SHOT_NET_VISUAL_CONTACT_AT * CHANCE_EVENT_WINDOW_MINUTES
        attack_windows.append((start, impact, kind, minute, side))
    attack_windows.sort(key=lambda item: item[0])
    for current, following in zip(attack_windows, attack_windows[1:]):
        if following[0] <= current[1] + 0.20:
            raise AssertionError(f"high-score cinematic attack windows collide: {current} -> {following}")
    for start, _impact, kind, minute, side in attack_windows:
        app.t = (start + 0.05) / 90.0 * SIMULATION_SECONDS
        active = app.active_attack_event(high_pred)
        if active is None:
            raise AssertionError(f"high-score {kind} attack is not active at its own cinematic start: {(minute, side)}")
        if (active.minute, active.side, active.kind) != (minute, side, kind):
            raise AssertionError(
                f"high-score cinematic start was stolen by another event: expected={(minute, side, kind)} active={active}"
            )

    for label, pred in (("home", home_win_prediction()), ("away", away_win_prediction())):
        goal_minute, _side = app.goal_schedule(pred)[0]
        phases = []
        actor_xs = []
        for progress in (
            SHOT_KICK_AT + 0.005,
            0.64,
            SHOT_FOLLOW_THROUGH_HOLD_END - 0.01,
            SHOT_FOLLOW_THROUGH_HOLD_END + 0.04,
            0.90,
            SHOT_NET_VISUAL_CONTACT_AT,
        ):
            state = state_at(pred, goal_minute, progress)
            phases.append(str(state.get("shot_phase", "")))
            actor_xs.append(float(state["actor_pos"][0]))  # type: ignore[index]
            kick_window, _stride = app.cinematic_stride_state(progress, float(state.get("stride_phase", 0.0)))
            if progress <= SHOT_FOLLOW_THROUGH_HOLD_END - 0.005 and not kick_window:
                raise AssertionError(f"{label} shooter drops the follow-through before the hold window: progress={progress:.3f}")
            if SHOT_FOLLOW_THROUGH_HOLD_END + 0.03 <= progress < SHOT_NET_VISUAL_CONTACT_AT - 0.02 and kick_window:
                raise AssertionError(f"{label} shooter remains frozen instead of recovering before impact: progress={progress:.3f}")
        if len(set(phases)) < 3:
            raise AssertionError(f"{label} shooter does not expose readable follow-through/recovery phases: {phases}")
        if max(actor_xs) - min(actor_xs) > 24:
            raise AssertionError(f"{label} shooter drifts across the shot/recovery window before impact: {actor_xs}")

    # If main.py later exposes explicit depth/z metadata, this gate asserts it.
    # Current runtime falls back to render-order inspection so the check stays compatible.
    pred = home_win_prediction()
    goal_minute, _side = app.goal_schedule(pred)[0]
    impact_state = state_at(pred, goal_minute, SHOT_NET_VISUAL_CONTACT_AT)
    depth_keys = ("goal_back_depth", "ball_depth", "front_post_depth")
    if all(key in impact_state for key in depth_keys):
        goal_back_depth = float(impact_state["goal_back_depth"])
        ball_depth = float(impact_state["ball_depth"])
        front_post_depth = float(impact_state["front_post_depth"])
        if not goal_back_depth <= ball_depth <= front_post_depth:
            raise AssertionError(
                f"ball/post depth metadata is not ordered back-to-front: "
                f"back={goal_back_depth}, ball={ball_depth}, front={front_post_depth}"
            )
    else:
        render_order: list[str] = []
        original_goal_3d = app.draw_cinematic_goal_3d
        original_front_posts = app.draw_cinematic_goal_front_posts
        original_ball = app.draw_cinematic_ball
        original_impact = app.draw_cinematic_goal_impact

        def spy_goal_3d(*args: object, **kwargs: object) -> object:
            render_order.append("goal_front" if kwargs.get("front_only") else "goal_back")
            return original_goal_3d(*args, **kwargs)

        def spy_front_posts(*args: object, **kwargs: object) -> object:
            render_order.append("goal_front")
            return original_front_posts(*args, **kwargs)

        def spy_ball(*args: object, **kwargs: object) -> object:
            render_order.append("ball")
            return original_ball(*args, **kwargs)

        def spy_impact(*args: object, **kwargs: object) -> object:
            render_order.append("impact")
            return original_impact(*args, **kwargs)

        try:
            app.draw_cinematic_goal_3d = spy_goal_3d  # type: ignore[method-assign]
            app.draw_cinematic_goal_front_posts = spy_front_posts  # type: ignore[method-assign]
            app.draw_cinematic_ball = spy_ball  # type: ignore[method-assign]
            app.draw_cinematic_goal_impact = spy_impact  # type: ignore[method-assign]
            app.screen.fill((0, 0, 0))
            app.draw_field(pred, pred, "CONFRONTO")
        finally:
            app.draw_cinematic_goal_3d = original_goal_3d  # type: ignore[method-assign]
            app.draw_cinematic_goal_front_posts = original_front_posts  # type: ignore[method-assign]
            app.draw_cinematic_ball = original_ball  # type: ignore[method-assign]
            app.draw_cinematic_goal_impact = original_impact  # type: ignore[method-assign]
        required = {"goal_back", "goal_front", "ball", "impact"}
        missing = sorted(required - set(render_order))
        if missing:
            raise AssertionError(f"cinematic z-order fallback missing layers {missing}: {render_order}")
        if render_order.index("goal_back") >= render_order.index("ball"):
            raise AssertionError(f"goal back/net layer must render before the ball: {render_order}")
        if render_order.index("impact") <= render_order.index("ball"):
            raise AssertionError(f"net impact burst must render after the ball reaches the net: {render_order}")

    for label, pred in (("home", home_win_prediction()), ("away", away_win_prediction())):
        goal_minute, _side = app.goal_schedule(pred)[0]
        before = state_at(pred, goal_minute, SHOT_NET_VISUAL_CONTACT_AT - 0.012)
        contact = state_at(pred, goal_minute, min(1.0, SHOT_NET_VISUAL_CONTACT_AT + 0.004))
        peak = state_at(pred, goal_minute, 1.0)
        mid_decay = state_at(pred, goal_minute, SHOT_NET_VISUAL_CONTACT_AT + 0.25)
        late_decay = state_at(pred, goal_minute, SHOT_NET_VISUAL_CONTACT_AT + 0.50)
        if float(before.get("net_progress", 0.0)) != 0.0:
            raise AssertionError(f"{label} net ripple starts before visual impact")
        if float(contact.get("net_progress", 0.0)) <= 0.0:
            raise AssertionError(f"{label} net ripple does not start at impact")
        if float(peak.get("net_progress", 0.0)) < 0.85:
            raise AssertionError(f"{label} net ripple never reaches a readable peak: {peak.get('net_progress')}")
        peak_progress = float(peak.get("net_progress", 0.0))
        mid_progress = float(mid_decay.get("net_progress", 0.0))
        late_progress = float(late_decay.get("net_progress", 0.0))
        if mid_progress >= peak_progress * 0.86:
            raise AssertionError(f"{label} net stays stretched too long after impact: peak={peak_progress:.3f}, mid={mid_progress:.3f}")
        if late_progress >= mid_progress * 0.58:
            raise AssertionError(f"{label} net does not dissipate elastically: mid={mid_progress:.3f}, late={late_progress:.3f}")
        ripple_alpha_keys = ("net_ripple_alpha", "net_decay", "net_ripple_decay")
        decay_values = [late_decay.get(key) for key in ripple_alpha_keys if isinstance(late_decay.get(key), (int, float))]
        if decay_values and any(float(value) >= float(peak.get("net_progress", 0.0)) for value in decay_values):
            raise AssertionError(f"{label} explicit net decay metadata does not decay: {decay_values}")
        side = str(peak["goal_side"])
        def burst_alpha_weight(ripple: float) -> int:
            burst = app.cached_goal_impact_burst(side, ripple, 255)
            return sum(
                burst.get_at((x, y)).a
                for y in range(0, burst.get_height(), 2)
                for x in range(0, burst.get_width(), 2)
            )

        early_ripple = burst_alpha_weight(0.18)
        peak_ripple = burst_alpha_weight(0.62)
        late_ripple = burst_alpha_weight(1.0)
        if peak_ripple <= early_ripple * 1.20:
            raise AssertionError(f"{label} net burst does not visibly expand: early={early_ripple}, peak={peak_ripple}")
        if late_ripple >= peak_ripple * 0.72:
            raise AssertionError(f"{label} net burst does not visibly decay: peak={peak_ripple}, late={late_ripple}")

        app.t = time_for_progress(goal_minute, SHOT_NET_VISUAL_CONTACT_AT - 0.012)
        if app.active_goal_event(pred) is not None:
            raise AssertionError(f"{label} GOOOL overlay can activate before visual impact")
        if app.score_from_prediction(pred) != (0, 0):
            raise AssertionError(f"{label} score changes before visual impact: {app.score_from_prediction(pred)}")
        app.t = time_for_progress(goal_minute, min(1.0, SHOT_NET_VISUAL_CONTACT_AT + 0.004))
        if app.active_goal_event(pred) is None:
            raise AssertionError(f"{label} GOOOL overlay is not active at visual impact")


def validate_runtime_oracle_legibility() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    field = pygame.Rect(32, 110, 910, 490)
    pred = neutral_prediction()
    app.match_prediction = pred
    app.t = neutral_sample_second(0.85)
    state = app.cinematic_scene_state(field, pred)
    app.screen.fill((0, 0, 0))
    app.draw_field(pred, pred, "CONFRONTO")
    neutral_progress = float(state.get("neutral_progress", 1.0))
    cases = (
        (app.home, state["home_pos"], False, float(state.get("home_stride_phase", 0.0))),
        (app.away, state["away_pos"], True, float(state.get("away_stride_phase", 0.0))),
    )
    for team, pos, flip, stride_phase in cases:
        frame = app.neutral_frame_for_phase(team, stride_phase, neutral_progress, flip)
        target = app.cinematic_actor_target_size(frame, CINEMATIC_NEUTRAL_PLAYER_SCALE)
        x, ground_y = pos  # type: ignore[misc]
        actor_rect = pygame.Rect(0, 0, *target)
        actor_rect.midbottom = (int(x), int(ground_y))
        chest_rect = pygame.Rect(
            int(actor_rect.x + actor_rect.w * 0.22),
            int(actor_rect.y + actor_rect.h * 0.38),
            max(1, int(actor_rect.w * 0.58)),
            max(1, int(actor_rect.h * 0.22)),
        )
        crop = app.screen.subsurface(chest_rect.clip(field)).copy()
        code = app.assets.cinematic_source_code(team)
        dark_text = code in {"white", "sky", "gold", "orange"}
        count = logo_pixel_count(crop, dark_text)
        if count < 18:
            raise AssertionError(f"runtime ORACLE mark is not legible for flipped/neutral {team.code}: pixels={count}")
        assert_runtime_logo_geometry(crop, dark_text, team.code)


def validate_audio_event_order() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    goal_minute = app.goal_schedule(pred)[0][0]
    played: list[str] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        played.append(name)

    app.sound.play = spy  # type: ignore[method-assign]
    for progress in (
        SHOT_KICK_AT + 0.03,
        SHOT_WHOOSH_AT + 0.01,
        SHOT_NET_VISUAL_CONTACT_AT,
        SHOT_REVERB_AT + 0.01,
        SHOT_REVERB_AT + 0.01,
        SHOT_REVERB_AT + 0.01,
    ):
        app.t = (goal_minute - 5.0 + progress * 5.0) / 90.0 * 45.0
        app.update_soundscape(1 / 60)
        app.flush_queued_match_audio()
    if played != GOAL_AUDIO_EVENTS:
        raise AssertionError(f"unexpected audio event order/dedup: {played}")

    app.screen.fill((0, 0, 0))
    app.draw_score_panel({"CONFRONTO": pred}, "CONFRONTO", pred)
    if played != GOAL_AUDIO_EVENTS:
        raise AssertionError(f"draw_score_panel must not trigger audio events: {played}")
    app.t = (goal_minute - 5.0 + SHOT_NET_VISUAL_CONTACT_AT * 5.0) / 90.0 * 45.0
    app.draw_simulate()
    if played != GOAL_AUDIO_EVENTS:
        raise AssertionError(f"draw_simulate must not trigger audio events: {played}")

    app = App(seed=2026)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    goal_minute = app.goal_schedule(pred)[0][0]
    played = []
    app.sound.play = spy  # type: ignore[method-assign]
    app.t = (goal_minute - 5.0 + (SHOT_KICK_AT - 0.04) * 5.0) / 90.0 * 45.0
    app.update_soundscape(1 / 60)
    app.t = (goal_minute - 5.0 + (SHOT_REVERB_AT + 0.02) * 5.0) / 90.0 * 45.0
    app.update_soundscape(1 / 8)
    app.flush_queued_match_audio()
    if played != ["net", "bass", "cheer"]:
        raise AssertionError(f"stuttered audio update should keep only synchronized impact cues: {played}")
    for _index in range(6):
        app.update_soundscape(1 / 60)
        app.flush_queued_match_audio()
    if played != ["net", "bass", "cheer", "reverb"]:
        raise AssertionError(f"stuttered audio cues did not recover impact tail in order: {played}")


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

    bundle_candidates = sorted(
        path
        for path in (*exact_assets, *allowed_orphans, *runtime_globs)
        if "/candidates/" in path or path.startswith("assets/sounds/candidates/")
    )
    if bundle_candidates:
        raise AssertionError(f"runtime bundle manifest must not package candidate assets: {bundle_candidates}")
    raw_runtime_globs = sorted(
        pattern
        for pattern in runtime_globs
        if "/cinematic_sources/" in pattern
        or "/parallax_sources/" in pattern
        or "/ball_sources/" in pattern
        or pattern.endswith("_sources/*.png")
    )
    if raw_runtime_globs:
        raise AssertionError(f"runtime globs must not point at generated source sheets: {raw_runtime_globs}")
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
        raise AssertionError(f"manifest allowlist references missing legacy assets: {missing_allowed}")
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


def validate_cinematic_draw_order_declared() -> None:
    source = (ROOT / "src" / "arena_ai" / "main.py").read_text(encoding="utf-8")
    marker = "    def draw_field("
    start = source.find(marker)
    if start < 0:
        raise AssertionError("draw_field implementation not found for z-order validation")
    next_method = source.find("\n    def ", start + len(marker))
    body = source[start:next_method if next_method > start else len(source)]
    positions = []
    for call in CINEMATIC_DRAW_ORDER:
        position = body.find(f"self.{call}(")
        if position < 0:
            raise AssertionError(f"draw_field is missing declared cinematic z-order call: {call}")
        positions.append(position)
    if positions != sorted(positions):
        raise AssertionError(f"draw_field cinematic z-order drifted: {dict(zip(CINEMATIC_DRAW_ORDER, positions))}")


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
            if first_goal - 5.0 > 32.0:
                raise AssertionError(f"{label} first chance starts too late: {schedule}")
            if 90.0 - schedule[-1][0] > 38.0 and final_home != final_away:
                raise AssertionError(f"{label} winner has no late payoff/chance pressure: {schedule}")
        active_samples = 0
        moving_samples = 0
        previous_ball: tuple[float, float] | None = None
        for minute in range(4, 89, 4):
            app.t = minute / 90.0 * SIMULATION_SECONDS
            state = app.cinematic_scene_state(field, pred)
            if state.get("active_goal") or float(state.get("run_speed", 0.0)) > 0.38:
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
        if goal_overlay.colliderect(narrator.inflate(8, 8)):
            raise AssertionError(f"goal overlay collides with narrator {possession}: {goal_overlay} vs {narrator}")
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
        seconds.append((first_goal - 5.0 + (SHOT_KICK_AT + 0.005) * 5.0) / 90.0 * SIMULATION_SECONDS)
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
    focus_second = (first_goal_minute - 5.0 + 0.52 * 5.0) / 90.0 * SIMULATION_SECONDS
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
    focus_second = (first_goal_minute - 5.0 + 0.52 * 5.0) / 90.0 * SIMULATION_SECONDS
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
    first_goal_minute = app.goal_schedule(pred)[0][0]
    visible_goal_minute = first_goal_minute - GOAL_EVENT_WINDOW_MINUTES + SHOT_NET_VISUAL_CONTACT_AT * GOAL_EVENT_WINDOW_MINUTES
    app.t = visible_goal_minute / 90.0 * SIMULATION_SECONDS
    if app.score_from_prediction(pred) == (0, 0):
        raise AssertionError("score did not update when the goal visually reached the net")
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
            minute = first_goal_minute + 3.55 + step * 0.10
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
            app.t = (target_goal - 5.0 + progress * 5.0) / 90.0 * SIMULATION_SECONDS
            state = app.cinematic_scene_state(field, pred)
            keeper_team = app.away if str(state.get("possession")) == "home" else app.home
            _render, rect = goalkeeper_render_for_state(app, keeper_team, state, str(state.get("possession")) == "away")
            if not field.contains(rect):
                raise AssertionError(f"goalkeeper clips in {home_code}/FRA at progress {progress:.2f}: {rect} outside {field}")


def validate_goal_overlay_score_sync_gate() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    first_goal_minute = app.goal_schedule(pred)[0][0]
    app.t = (first_goal_minute - 5.0 + SHOT_NET_VISUAL_CONTACT_AT * 5.0) / 90.0 * SIMULATION_SECONDS
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
    app.match_prediction = pred
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


def validate_sixty_fps_budget() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = home_win_prediction()
    app.match_prediction = pred
    warmup_frames = 120
    for _index in range(warmup_frames):
        app.update(1 / 60)
        app.draw()
    gc.collect()
    frame_times = []
    frames = int(SIMULATION_SECONDS / (1 / 60)) + 1
    for _index in range(frames):
        start = time.perf_counter()
        app.update(1 / 60)
        app.draw()
        frame_times.append(time.perf_counter() - start)
    ordered = sorted(frame_times)
    p95 = ordered[int(len(ordered) * 0.95)]
    if p95 > 0.024:
        raise AssertionError(f"60fps visual budget failed: p95={p95 * 1000:.2f}ms")
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
    frame_times = []
    for view in ("groups", "bracket"):
        app.tournament_view = view
        for _index in range(12):
            app.t += 1 / 60
            app.draw_tournament()
        gc.collect()
        for _index in range(120):
            app.t += 1 / 60
            start = time.perf_counter()
            app.draw_tournament()
            frame_times.append(time.perf_counter() - start)
    ordered = sorted(frame_times)
    p95 = ordered[int(len(ordered) * 0.95)]
    if p95 > 0.024:
        raise AssertionError(f"tournament render budget failed: p95={p95 * 1000:.2f}ms")
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
        len(app.goal_orientation_cache),
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
        len(app.goal_orientation_cache),
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
    validate_model_policy_artifacts,
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
    validate_flag_sprites,
    validate_parallax_assets,
    validate_cinematic_sprites,
    validate_aaa_player_crop_gate,
    validate_cinematic_scene,
    validate_aaa_cinematic_design_gate,
    validate_aaa_screen_composition_gate,
    validate_aaa_away_draw_ball_stability_gate,
    validate_aaa_chance_outcome_gate,
    validate_nil_draw_has_no_fake_goals_gate,
    validate_aaa_cinematic_fade_gate,
    validate_aaa_goalkeeper_front_layer_gate,
    validate_cinematic_temporal_stability,
    validate_aaa_ball_physics_gate,
    validate_aaa_findings_light_gate,
    validate_runtime_oracle_legibility,
    validate_audio_event_order,
    validate_sound_assets,
    validate_app_icon,
    validate_fifa_external_assets,
    validate_asset_manifest,
    validate_model_policy_artifacts,
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
    validate_copa_copy_contract,
    validate_chance_schedule_no_dead_air,
    validate_cinematic_draw_order_declared,
    validate_cinematic_camera_continuity_gate,
    validate_goalkeeper_safe_bounds_variants,
    validate_goal_overlay_score_sync_gate,
    validate_render_purity,
    validate_visual_determinism,
    validate_full_match_flow,
    validate_sixty_fps_budget,
    validate_tournament_render_budget,
    validate_tournament_layout_gate,
    validate_tournament_result_header_contract,
    validate_tournament_loading_pacing_gate,
    validate_tournament_seed_entropy_gate,
    validate_monte_carlo_runtime_mode_gate,
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

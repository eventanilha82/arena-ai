#!/usr/bin/env python3
"""Validate the promoted cinematic contract in the production game runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (str(SRC), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pygame

from arena_ai.cinematic_poc_runtime import (
    POC_BALL_CANVAS_SIZE,
    POC_GROUND_Y,
    POC_RUNNER_CANVAS_SIZE,
    POC_RUNNER_ROOT,
    PocSequence,
    PocSequenceBank,
    PocViewport,
)
from arena_ai.cinematic_uniforms import CINEMATIC_UNIFORMS
from arena_ai.main import (
    App,
    BG,
    CHANCE_EVENT_WINDOW_MINUTES,
    CHANCE_PAYOFF_MINUTES,
    CinematicAttackEvent,
    GOAL_EVENT_WINDOW_MINUTES,
    GOAL_PAYOFF_MINUTES,
    MatchRuntimeState,
    SIMULATION_SECONDS,
)
from arena_ai.worldcup_model import Prediction
CONTRACT = (
    ROOT
    / "assets"
    / "generated"
    / "cinematic"
    / "poc7_runtime_contract.json"
)
MANIFEST = ROOT / "assets" / "asset_manifest.json"
PROFILES = ("high", "mid", "low")
OUTCOMES = ("goal", "save", "wide")
DIRECTIONS = ("right", "left")
RAW_STAGES = (0.45, 0.82, 1.0, 1.25)
CLEARANCE_RELEASE_OFFSETS = tuple(
    frame / 60.0
    for frame in range(9)
)
EXPECTED_SEQUENCE_COUNT = 30
EXPECTED_RENDERER_CASES = 150
EXPECTED_ACTOR_PROGRESSION_CASES = 18
EXPECTED_CLEARANCE_CASES = 2_430
EXPECTED_E2E_CASES = 6
EXPECTED_FRAMEBUFFER_CASES = 8
EXPECTED_PRELOAD_ASSETS = 58
EXPECTED_SEQUENTIAL_NET_CASES = 2_502
EVIDENCE_SCHEMA_VERSION = 13
EXPECTED_RUNTIME_ASSETS = 181
EVIDENCE_ARTIFACT = "approved_cinematic_runtime_renderer"
EVIDENCE_CAPTURE_POLICY = (
    "30 promoted sequences x 5 renderer checkpoints "
    "(including terminal); fresh current-runtime capture"
)
CLEARANCE_MAX_ALPHA_OVERLAP_RATIO = 0.03
CLEARANCE_RUNTIME_ROUNDING_SAFETY_PX = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sequence_seed(
    bank: PocSequenceBank,
    sequence: PocSequence,
) -> int:
    direction = sequence.attack_direction
    profile = sequence.profile
    outcome = sequence.outcome
    side = "home" if direction == "right" else "away"
    for seed in range(1, 100_000):
        if (
            PocSequenceBank.select_profile(
                seed,
                50,
                side,
                outcome,
            )
            == profile
            and bank.select_sequence(
                attack_direction=direction,
                profile=profile,
                outcome=outcome,
                match_seed=seed,
                event_minute=50,
                side=side,
            ).key
            == sequence.key
        ):
            return seed
    raise RuntimeError(
        f"could not select {sequence.key}"
    )


def expected_audio_names(outcome: str) -> tuple[str, ...]:
    if outcome == "goal":
        return ("kick", "whoosh", "net", "bass", "cheer", "reverb")
    if outcome == "save":
        return ("kick", "whoosh", "save")
    return ("kick", "whoosh", "near_miss")


def renderer_stages(
    sequence: PocSequence,
    event_seconds: float,
) -> tuple[tuple[str, float], ...]:
    terminal_raw = (
        1.0
        + (
            sequence.duration_seconds
            - sequence.impact_seconds
        )
        / max(1e-9, event_seconds)
    )
    return (
        *((f"{raw:.2f}", raw) for raw in RAW_STAGES),
        ("terminal", terminal_raw),
    )


def validate_provenance(
    payload: dict[str, object],
    failures: list[str],
) -> None:
    if (
        payload.get("artifact")
        != "arena_cinematic_runtime_contract"
        or payload.get("status") != "promoted"
    ):
        failures.append(
            "invalid promoted cinematic runtime contract identity"
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract_pattern = (
        "assets/generated/cinematic/poc7_runtime_contract.json"
    )
    net_patterns = {
        "assets/generated/cinematic/poc7_net/*.png": 4,
        "assets/generated/cinematic/poc7_net/*/*.png": 54,
    }
    for key in ("generated_runtime_globs", "release_runtime_globs"):
        if contract_pattern not in manifest.get(key, []):
            failures.append(
                f"runtime contract missing from manifest {key}"
            )
        for pattern in net_patterns:
            if pattern not in manifest.get(key, []):
                failures.append(
                    f"POC net assets missing from manifest {key}: {pattern}"
                )
    expected_counts = manifest.get(
        "release_runtime_glob_expected_counts",
        {},
    )
    if expected_counts.get(contract_pattern) != 1:
        failures.append(
            "runtime contract manifest count is not one"
        )
    for pattern, count in net_patterns.items():
        if expected_counts.get(pattern) != count:
            failures.append(
                f"POC net manifest count drift: {pattern}"
            )


def validate_sequence(
    bank: PocSequenceBank,
    sequence: PocSequence,
    failures: list[str],
) -> None:
    expected_keeper = (
        "left"
        if sequence.attack_direction == "right"
        else "right"
    )
    expected_goal = sequence.attack_direction
    if sequence.keeper_direction != expected_keeper:
        failures.append(
            f"{sequence.key}: goalkeeper direction is not independent/opposite"
        )
    if sequence.goal_side != expected_goal:
        failures.append(
            f"{sequence.key}: goal is on the wrong attack side"
        )
    if tuple(name for name, _seconds in sequence.audio_cues) != (
        expected_audio_names(sequence.outcome)
    ):
        failures.append(
            f"{sequence.key}: audio timeline differs from the frozen contract"
        )
    cue_seconds = [seconds for _name, seconds in sequence.audio_cues]
    if cue_seconds != sorted(cue_seconds):
        failures.append(
            f"{sequence.key}: audio cues are not chronological"
        )
    expected_samples = math.ceil(
        sequence.duration_seconds * bank.sample_hz
    ) + 1
    if len(sequence.samples) != expected_samples:
        failures.append(
            f"{sequence.key}: incomplete 60 Hz sample inventory"
        )
    if not 0 < sequence.weight_bps <= 10_000:
        failures.append(
            f"{sequence.key}: invalid medoid weight"
        )
    if sequence.outcome == "goal":
        if (
            not sequence.net_static_back
            or not sequence.net_static_back_sha256
            or len(sequence.net_keyframes)
            != bank.NET_KEYFRAME_COUNT
            or not sequence.net_contact_frames
        ):
            failures.append(
                f"{sequence.key}: incomplete localized net animation"
            )
        previous_net_seconds = -1.0
        verified_assets: set[
            tuple[str, str]
        ] = set()
        static_path = ROOT / str(sequence.net_static_back)
        if (
            not static_path.is_file()
            or sha256(static_path)
            != sequence.net_static_back_sha256
        ):
            failures.append(
                f"{sequence.key}: static net asset drift "
                f"{sequence.net_static_back}"
            )
        previous_source_elapsed = -1.0
        for frame in sequence.net_keyframes:
            if (
                previous_net_seconds >= 0.0
                and frame.seconds_after_impact
                - previous_net_seconds
                > 1.0 / bank.NET_KEYFRAME_HZ + 1e-5
            ):
                failures.append(
                    f"{sequence.key}: net cadence is below "
                    f"{bank.NET_KEYFRAME_HZ} Hz"
                )
            previous_net_seconds = frame.seconds_after_impact
            if (
                frame.source_elapsed_seconds
                <= previous_source_elapsed
            ):
                failures.append(
                    f"{sequence.key}: POC net source timeline "
                    "is not sequential"
                )
            previous_source_elapsed = (
                frame.source_elapsed_seconds
            )
            for relative_path, expected_sha in (
                (frame.back_roi, frame.back_roi_sha256),
            ):
                asset_identity = (
                    relative_path,
                    str(expected_sha),
                )
                if asset_identity in verified_assets:
                    continue
                verified_assets.add(asset_identity)
                path = ROOT / relative_path
                if (
                    not path.is_file()
                    or expected_sha is None
                    or sha256(path) != expected_sha
                ):
                    failures.append(
                        f"{sequence.key}: net asset drift {relative_path}"
                    )
        previous_contact_seconds = -1.0
        for frame in sequence.net_contact_frames:
            if (
                previous_contact_seconds >= 0.0
                and frame.seconds_after_impact
                - previous_contact_seconds
                > 1.0 / bank.NET_CONTACT_SAMPLE_HZ + 1e-5
            ):
                failures.append(
                    f"{sequence.key}: contact cadence is below "
                    f"{bank.NET_CONTACT_SAMPLE_HZ} Hz"
                )
            if (
                frame.seconds_after_impact
                >= bank.NET_CONTACT_END_SECONDS + 1e-5
            ):
                failures.append(
                    f"{sequence.key}: stale contact frame after fade"
                )
            previous_contact_seconds = (
                frame.seconds_after_impact
            )
            asset_identity = (
                frame.front_contact,
                frame.front_contact_sha256,
            )
            if asset_identity in verified_assets:
                continue
            verified_assets.add(asset_identity)
            path = ROOT / frame.front_contact
            if (
                not path.is_file()
                or sha256(path)
                != frame.front_contact_sha256
            ):
                failures.append(
                    f"{sequence.key}: contact asset drift "
                    f"{frame.front_contact}"
                )
    elif (
        sequence.net_static_back
        or sequence.net_keyframes
        or sequence.net_contact_frames
    ):
        failures.append(
            f"{sequence.key}: non-goal contains net assets"
        )

    previous_elapsed = -1.0
    net_peak = 0.0
    ball_after_keeper = False
    initial_goal_x = None
    final_goal_x = None
    for raw_sample in sequence.samples:
        sample = bank.sample(
            sequence,
            float(raw_sample[0]),
        )
        if sample.elapsed + 1e-6 < previous_elapsed:
            failures.append(
                f"{sequence.key}: non-monotonic timeline"
            )
            break
        previous_elapsed = sample.elapsed
        if abs(sample.actor_ground_y - POC_GROUND_Y) > 1e-6:
            failures.append(
                f"{sequence.key}: actor ground drift"
            )
            break
        if (
            abs(sample.goal_w - 993.0) > 1e-6
            or abs(sample.goal_h - 497.0) > 1e-6
        ):
            failures.append(
                f"{sequence.key}: goal source geometry drift"
            )
            break
        if not 0 <= sample.keeper_frame < 16:
            failures.append(
                f"{sequence.key}: invalid goalkeeper frame"
            )
            break
        if not 0 <= sample.ball_phase < len(bank.ball_phase_labels):
            failures.append(
                f"{sequence.key}: invalid ball phase"
            )
            break
        expected_ball_after_keeper = (
            sequence.outcome in {"goal", "wide"}
            and sample.elapsed > sequence.line_seconds
        )
        if sample.ball_after_keeper != expected_ball_after_keeper:
            failures.append(
                f"{sequence.key}: invalid ball/keeper z-order "
                f"at {sample.elapsed:.3f}s"
            )
            break
        net_peak = max(net_peak, sample.net_strength)
        ball_after_keeper = (
            ball_after_keeper or sample.ball_after_keeper
        )
        if initial_goal_x is None:
            initial_goal_x = sample.goal_x
        final_goal_x = sample.goal_x

    if sequence.outcome == "goal" and net_peak < 0.65:
        failures.append(
            f"{sequence.key}: approved net impulse was lost"
        )
    if sequence.outcome != "goal" and net_peak > 1e-6:
        failures.append(
            f"{sequence.key}: non-goal deforms the net"
        )
    if sequence.outcome in {"goal", "wide"} and not ball_after_keeper:
        failures.append(
            f"{sequence.key}: scored/wide ball never crosses goalkeeper depth"
        )
    if sequence.outcome == "save" and ball_after_keeper:
        failures.append(
            f"{sequence.key}: unexpected post-keeper ball layer"
        )
    terminal = bank.sample(
        sequence,
        sequence.duration_seconds,
    )
    raw_terminal = sequence.samples[-1]
    if (
        abs(terminal.elapsed - float(raw_terminal[0])) > 1e-6
        or abs(terminal.ball_x - float(raw_terminal[12])) > 1e-6
        or abs(terminal.ball_y - float(raw_terminal[13])) > 1e-6
        or terminal.keeper_frame != int(raw_terminal[23])
    ):
        failures.append(
            f"{sequence.key}: terminal nonuniform sample was not reproduced"
        )
    if (
        sequence.outcome in {"save", "wide"}
        and terminal.keeper_frame != 15
    ):
        failures.append(
            f"{sequence.key}: non-goal keeper does not recover/reset"
        )
    if sequence.outcome == "goal" and terminal.keeper_frame >= 14:
        failures.append(
            f"{sequence.key}: beaten keeper incorrectly recovers"
        )
    if initial_goal_x is not None and final_goal_x is not None:
        camera_delta = final_goal_x - initial_goal_x
        expected_sign = -1.0 if sequence.attack_direction == "right" else 1.0
        if camera_delta * expected_sign < 250.0:
            failures.append(
                f"{sequence.key}: POC approach camera was not preserved"
            )


def validate_profile_ordering(
    bank: PocSequenceBank,
    failures: list[str],
) -> None:
    for direction in DIRECTIONS:
        for outcome in ("goal", "wide"):
            impact_y = []
            for profile in PROFILES:
                variants = bank.variants(
                    direction,
                    profile,
                    outcome,
                )
                impact_y.append(
                    sum(
                        bank.sample(
                            sequence,
                            sequence.impact_seconds,
                        ).ball_y
                        for sequence in variants
                    )
                    / len(variants)
                )
            ordered = (
                impact_y[0] < impact_y[1] < impact_y[2]
                if outcome == "goal"
                else (
                    impact_y[0] < 300.0
                    and impact_y[1] < 300.0
                    and impact_y[2] > 600.0
                )
            )
            if not ordered:
                failures.append(
                    f"{direction}/{outcome}: high-mid-low trajectory order drift"
                )


def surface_rgba_sha256(surface: pygame.Surface) -> str:
    return hashlib.sha256(
        pygame.image.tostring(surface, "RGBA")
    ).hexdigest()


def validate_sequential_net_contract(
    bank: PocSequenceBank,
    payload: dict[str, object],
    failures: list[str],
) -> int:
    if not pygame.get_init():
        pygame.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))
    raw_sequences = payload.get("sequences")
    if not isinstance(raw_sequences, dict):
        failures.append(
            "contract has no raw sequence payload for net validation"
        )
        return 0
    atlas_cache: dict[str, pygame.Surface] = {}
    verified_hashes: set[tuple[str, str]] = set()
    cases = 0
    goal_sequences = sorted(
        (
            sequence
            for sequence in bank.sequences.values()
            if sequence.outcome == "goal"
        ),
        key=lambda sequence: sequence.key,
    )
    for sequence in goal_sequences:
        entry = raw_sequences.get(sequence.key)
        if not isinstance(entry, dict):
            failures.append(
                f"{sequence.key}: missing raw net payload"
            )
            continue
        if not isinstance(entry.get("shot_plan"), dict):
            failures.append(
                f"{sequence.key}: missing frozen shot plan"
            )
        if sequence.net_static_back is None:
            failures.append(
                f"{sequence.key}: static net frame missing"
            )
        else:
            static_path = ROOT / sequence.net_static_back
            if (
                not static_path.is_file()
                or sha256(static_path) != sequence.net_static_back_sha256
            ):
                failures.append(
                    f"{sequence.key}: static net hash drift"
                )
        source_times = [
            frame.source_elapsed_seconds
            for frame in sequence.net_keyframes
        ]
        if (
            not source_times
            or source_times[0] + 1e-9
            < sequence.impact_seconds
            or source_times[0]
            > sequence.impact_seconds
            + 1.0 / bank.NET_KEYFRAME_HZ
            + 5e-6
            or any(
                following <= previous
                for previous, following in zip(
                    source_times,
                    source_times[1:],
                )
            )
        ):
            failures.append(
                f"{sequence.key}: net source timeline drift"
            )
        for index, frame in enumerate(sequence.net_keyframes):
            expected_seconds = index / bank.NET_KEYFRAME_HZ
            if (
                abs(frame.seconds_after_impact - expected_seconds)
                > 5e-6
                or abs(
                    (
                        frame.source_elapsed_seconds
                        - sequence.impact_seconds
                    )
                    - expected_seconds
                )
                > 1.0 / bank.sample_hz + 5e-6
            ):
                failures.append(
                    f"{sequence.key}: net cadence drift at frame {index}"
                )
                break
        contact_seconds = [
            frame.seconds_after_impact
            for frame in sequence.net_contact_frames
        ]
        if (
            contact_seconds != sorted(contact_seconds)
            or not contact_seconds
            or contact_seconds[0] < -1e-9
            or contact_seconds[-1]
            > bank.NET_CONTACT_END_SECONDS + 1e-9
        ):
            failures.append(
                f"{sequence.key}: contact cadence drift"
            )
        visible_contact_frames = 0
        for contact_frame in sequence.net_contact_frames:
            contact_path = ROOT / contact_frame.front_contact
            contact_digest_key = (
                contact_frame.front_contact,
                contact_frame.front_contact_sha256,
            )
            if contact_digest_key not in verified_hashes:
                if (
                    not contact_path.is_file()
                    or sha256(contact_path)
                    != contact_frame.front_contact_sha256
                ):
                    failures.append(
                        f"{sequence.key}: contact atlas hash drift"
                    )
                verified_hashes.add(contact_digest_key)
            contact_atlas = atlas_cache.get(
                contact_frame.front_contact
            )
            if contact_atlas is None:
                contact_atlas = pygame.image.load(
                    contact_path
                ).convert_alpha()
                atlas_cache[
                    contact_frame.front_contact
                ] = contact_atlas
            contact_rect = pygame.Rect(
                contact_frame.front_contact_source_rect
            )
            if (
                not contact_atlas.get_rect().contains(contact_rect)
                or contact_rect.size
                != tuple(contact_frame.front_contact_rect[2:])
            ):
                failures.append(
                    f"{sequence.key}: invalid contact atlas frame"
                )
                continue
            if pygame.mask.from_surface(
                contact_atlas.subsurface(contact_rect),
                1,
            ).count():
                visible_contact_frames += 1
        if not visible_contact_frames:
            failures.append(
                f"{sequence.key}: contact animation has no visible frame"
            )

        reported: set[str] = set()
        for elapsed in sequence.sample_times:
            if elapsed + 1e-9 < sequence.impact_seconds:
                continue
            cases += 1
            frame = bank.nearest_net_keyframe(
                sequence,
                elapsed,
            )
            local = max(
                0.0,
                elapsed - sequence.impact_seconds,
            )
            expected_index = max(
                0,
                min(
                    len(sequence.net_keyframes) - 1,
                    round(local * bank.NET_KEYFRAME_HZ),
                ),
            )
            if frame is None:
                if "back_missing" not in reported:
                    failures.append(
                        f"{sequence.key}: runtime net frame missing"
                    )
                    reported.add("back_missing")
                continue
            if (
                frame
                != sequence.net_keyframes[expected_index]
                and "back_selection" not in reported
            ):
                failures.append(
                    f"{sequence.key}: invalid net frame selection"
                )
                reported.add("back_selection")
            atlas_path = ROOT / frame.back_roi
            digest_key = (frame.back_roi, frame.back_roi_sha256)
            if digest_key not in verified_hashes:
                if (
                    not atlas_path.is_file()
                    or sha256(atlas_path) != frame.back_roi_sha256
                ):
                    failures.append(
                        f"{sequence.key}: net atlas hash drift"
                    )
                verified_hashes.add(digest_key)
            atlas = atlas_cache.get(frame.back_roi)
            if atlas is None:
                atlas = pygame.image.load(atlas_path).convert_alpha()
                atlas_cache[frame.back_roi] = atlas
            source_rect = pygame.Rect(frame.back_roi_source_rect)
            if (
                not atlas.get_rect().contains(source_rect)
                or source_rect.size
                != tuple(frame.back_roi_rect[2:])
                or pygame.mask.from_surface(
                    atlas.subsurface(source_rect),
                    1,
                ).count()
                == 0
            ) and "back_pixels" not in reported:
                failures.append(
                    f"{sequence.key}: invalid net atlas frame"
                )
                reported.add("back_pixels")

            contact_frame = bank.nearest_net_contact_frame(
                sequence,
                elapsed,
            )
            expected_contact = (
                None
                if local > bank.NET_CONTACT_END_SECONDS
                else min(
                    sequence.net_contact_frames,
                    key=lambda candidate: abs(
                        candidate.seconds_after_impact - local
                    ),
                )
            )
            if (
                contact_frame != expected_contact
                and "contact_selection" not in reported
            ):
                failures.append(
                    f"{sequence.key}: invalid contact frame selection"
                )
                reported.add("contact_selection")
    return cases


def validate_renderer(
    bank: PocSequenceBank,
    failures: list[str],
    capture_dir: Path | None,
) -> int:
    app = App(seed=20260728)
    field = app.match_field_rect()
    viewport = PocViewport.fit(field)
    expected_goal_size = viewport.size(993.0, 497.0)
    rendered = 0
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=True)
    for expected_sequence in bank.sequences.values():
            app.poc_layer_frame_cache.clear()
            app.poc_layer_cache.clear()
            app.surface_cache.scaled.clear()
            direction = expected_sequence.attack_direction
            profile = expected_sequence.profile
            outcome = expected_sequence.outcome
            side = "home" if direction == "right" else "away"
            event_window = (
                GOAL_EVENT_WINDOW_MINUTES
                if outcome == "goal"
                else CHANCE_EVENT_WINDOW_MINUTES
            )
            event_seconds = (
                event_window / 90.0 * SIMULATION_SECONDS
            )
            app.match_seed = sequence_seed(
                bank,
                expected_sequence,
            )
            app.preload_cinematic_poc_sequence(
                expected_sequence,
            )
            layer_cache_misses: list[str] = []
            original_layer_load = app.load_cinematic_poc_layer

            def layer_load_spy(
                relative_path: str,
                expected_sha256: str,
            ) -> pygame.Surface:
                if relative_path not in app.poc_layer_cache:
                    layer_cache_misses.append(relative_path)
                return original_layer_load(
                    relative_path,
                    expected_sha256,
                )

            app.load_cinematic_poc_layer = layer_load_spy  # type: ignore[method-assign]
            event = CinematicAttackEvent(
                50,
                side,
                outcome == "goal",
                outcome,
            )
            for stage_label, raw_progress in renderer_stages(
                expected_sequence,
                event_seconds,
            ):
                state = app.cinematic_poc_scene_state(
                    field,
                    event,
                    raw_progress,
                )
                sequence = state["poc_contract_sequence"]
                sample = state["poc_contract_sample"]
                if (
                    not isinstance(sequence, PocSequence)
                    or sequence.key != expected_sequence.key
                ):
                    failures.append(
                        f"{direction}/{profile}/{outcome}: runtime selected wrong sequence"
                    )
                    continue
                if (
                    stage_label == "terminal"
                    and abs(
                        float(sample.elapsed)  # type: ignore[union-attr]
                        - expected_sequence.duration_seconds
                    )
                    > 1e-5
                ):
                    failures.append(
                        f"{sequence.key}: terminal renderer "
                        "checkpoint did not reach the terminal sample"
                    )
                if state["goal_rect"].size != expected_goal_size:  # type: ignore[union-attr]
                    failures.append(
                        f"{sequence.key}: runtime goal scale differs from POC"
                    )
                expected_keeper_frames = (
                    app.assets.cinematic_keeper_frames_left
                    if sequence.keeper_direction == "left"
                    else app.assets.cinematic_keeper_frames
                )
                keeper_team = (
                    app.away if side == "home" else app.home
                )
                expected_keeper = expected_keeper_frames[
                    keeper_team.code
                ][sample.keeper_frame]  # type: ignore[union-attr]
                expected_actor = app.cinematic_poc_actor_material(
                    sequence,
                    sample,  # type: ignore[arg-type]
                    app.home if side == "home" else app.away,
                )[0]
                ball_frame_index = (
                    int(
                        (
                            float(sample.ball_rotation)  # type: ignore[union-attr]
                            % 360.0
                        )
                        / 360.0
                        * len(app.assets.balls)
                    )
                    % len(app.assets.balls)
                )
                expected_ball = app.assets.balls[
                    ball_frame_index
                ]
                front_asset = bank.goal_front_layers[
                    expected_sequence.goal_side
                ]
                expected_front = original_layer_load(
                    front_asset.file,
                    front_asset.sha256,
                )
                expected_goal_sources = [expected_front]
                if (
                    expected_sequence.outcome != "goal"
                    or sample.elapsed  # type: ignore[union-attr]
                    < expected_sequence.impact_seconds
                ):
                    base_asset = bank.goal_base_layers[
                        expected_sequence.goal_side
                    ]
                    expected_goal_sources.append(
                        original_layer_load(
                            base_asset.file,
                            base_asset.sha256,
                        )
                    )
                else:
                    if (
                        expected_sequence.net_static_back is None
                        or expected_sequence.net_static_back_sha256
                        is None
                    ):
                        failures.append(
                            f"{expected_sequence.key}: missing net back"
                        )
                        continue
                    expected_goal_sources.append(
                        original_layer_load(
                            expected_sequence.net_static_back,
                            expected_sequence.net_static_back_sha256,
                        )
                    )
                    net_frame = bank.nearest_net_keyframe(
                        expected_sequence,
                        sample.elapsed,  # type: ignore[union-attr]
                    )
                    if net_frame is not None:
                        expected_goal_sources.append(
                            app.load_cinematic_poc_atlas_frame(
                                net_frame.back_roi,
                                net_frame.back_roi_sha256,
                                net_frame.back_roi_source_rect,
                            )
                        )
                    contact_frame = (
                        bank.nearest_net_contact_frame(
                            expected_sequence,
                            sample.elapsed,  # type: ignore[union-attr]
                        )
                    )
                    if contact_frame is not None:
                        expected_goal_sources.append(
                            app.load_cinematic_poc_atlas_frame(
                                contact_frame.front_contact,
                                contact_frame.front_contact_sha256,
                                contact_frame.front_contact_source_rect,
                            )
                        )
                scaled_sources: list[int] = []
                original_scale = app.cached_smoothscale
                original_ball_draw = app.draw_cinematic_poc_ball
                original_keeper_draw = app.draw_cinematic_poc_keeper
                actor_layer_order: list[str] = []

                def scale_spy(
                    image: pygame.Surface,
                    size: tuple[int, int],
                ) -> pygame.Surface:
                    scaled_sources.append(id(image))
                    return original_scale(image, size)

                def ball_draw_spy(
                    active_state: dict[str, object],
                ) -> None:
                    actor_layer_order.append("ball")
                    original_ball_draw(active_state)

                def keeper_draw_spy(
                    active_state: dict[str, object],
                ) -> None:
                    actor_layer_order.append("keeper")
                    original_keeper_draw(active_state)

                app.cached_smoothscale = scale_spy  # type: ignore[method-assign]
                app.draw_cinematic_poc_ball = ball_draw_spy  # type: ignore[method-assign]
                app.draw_cinematic_poc_keeper = keeper_draw_spy  # type: ignore[method-assign]
                try:
                    app.screen.fill(BG)
                    old_clip = app.screen.get_clip()
                    app.screen.set_clip(field)
                    app.draw_cinematic_poc_sequence(
                        field,
                        state,
                    )
                    app.screen.set_clip(old_clip)
                finally:
                    app.cached_smoothscale = original_scale  # type: ignore[method-assign]
                    app.draw_cinematic_poc_ball = original_ball_draw  # type: ignore[method-assign]
                    app.draw_cinematic_poc_keeper = original_keeper_draw  # type: ignore[method-assign]
                if id(expected_keeper) not in scaled_sources:
                    failures.append(
                        f"{sequence.key}: renderer used wrong goalkeeper direction/frame"
                    )
                if (
                    sample.actor_visible  # type: ignore[union-attr]
                    and id(expected_actor) not in scaled_sources
                ):
                    failures.append(
                        f"{sequence.key}: renderer used wrong authored player frame"
                    )
                if (
                    sample.ball_visible  # type: ignore[union-attr]
                    and id(expected_ball) not in scaled_sources
                ):
                    failures.append(
                        f"{sequence.key}: renderer did not draw the approved ball"
                    )
                for goal_source in expected_goal_sources:
                    if id(goal_source) not in scaled_sources:
                        failures.append(
                            f"{sequence.key}: renderer omitted an approved goal layer"
                        )
                expected_layer_order = (
                    ["keeper", "ball"]
                    if sample.ball_after_keeper  # type: ignore[union-attr]
                    else ["ball", "keeper"]
                )
                if actor_layer_order != expected_layer_order:
                    failures.append(
                        f"{sequence.key}: ball/keeper draw order drift "
                        f"{actor_layer_order}/{expected_layer_order}"
                    )
                if capture_dir is not None:
                    filename = (
                        f"{direction}_{profile}_{outcome}_"
                        f"v{expected_sequence.variant}_"
                        f"{stage_label}.png"
                    )
                    pygame.image.save(
                        app.screen.subsurface(field),
                        capture_dir / filename,
                    )
                rendered += 1
            app.load_cinematic_poc_layer = original_layer_load  # type: ignore[method-assign]
            if layer_cache_misses:
                failures.append(
                    f"{expected_sequence.key}: renderer performed "
                    f"uncached layer I/O {sorted(set(layer_cache_misses))}"
                )
    return rendered


def assert_source_materialized(
    *,
    actual: pygame.Surface,
    source: pygame.Surface,
    destination: pygame.Rect,
    background: tuple[int, int, int],
    label: str,
    failures: list[str],
) -> None:
    expected_source = pygame.transform.smoothscale(
        source,
        destination.size,
    )
    clipped = destination.clip(actual.get_rect())
    if clipped.width <= 0 or clipped.height <= 0:
        failures.append(
            f"{label}: approved source is outside framebuffer"
        )
        return
    source_clip = pygame.Rect(
        clipped.x - destination.x,
        clipped.y - destination.y,
        clipped.w,
        clipped.h,
    )
    expected_visible = expected_source.subsurface(source_clip)
    alpha = pygame.surfarray.array_alpha(expected_visible)
    mask = alpha >= 32
    if int(mask.sum()) < 12:
        failures.append(
            f"{label}: approved source has no testable visible pixels"
        )
        return
    expected = pygame.Surface(clipped.size)
    expected.fill(background)
    expected.blit(expected_visible, (0, 0))
    actual_crop = actual.subsurface(clipped)
    actual_rgb = pygame.surfarray.array3d(actual_crop)
    expected_rgb = pygame.surfarray.array3d(expected)
    matches = (
        actual_rgb[mask] == expected_rgb[mask]
    ).all(axis=1)
    match_ratio = float(matches.mean())
    visible_ratio = float(
        (
            actual_rgb[mask]
            != background
        ).any(axis=1).mean()
    )
    if match_ratio < 0.97 or visible_ratio < 0.995:
        failures.append(
            f"{label}: approved pixels did not reach framebuffer "
            f"(exact={match_ratio:.3f}, visible={visible_ratio:.3f})"
        )


def assert_canvas_region_equal(
    *,
    actual: pygame.Surface,
    expected: pygame.Surface,
    region: pygame.Rect,
    label: str,
    failures: list[str],
) -> None:
    region = region.clip(actual.get_rect())
    if region.width <= 0 or region.height <= 0:
        failures.append(
            f"{label}: approved composite is outside framebuffer"
        )
        return
    actual_rgb = pygame.surfarray.array3d(
        actual.subsurface(region)
    )
    expected_rgb = pygame.surfarray.array3d(
        expected.subsurface(region)
    )
    if not bool((actual_rgb == expected_rgb).all()):
        failures.append(
            f"{label}: approved composite did not reach framebuffer"
        )


def validate_framebuffer_materialization(
    bank: PocSequenceBank,
    failures: list[str],
) -> int:
    app = App(seed=20260728)
    field = app.match_field_rect()
    background = (17, 29, 43)
    cases = 0
    for direction in DIRECTIONS:
        sequence = bank.sequence(
            direction,
            "mid",
            "goal",
            0,
        )
        side = "home" if direction == "right" else "away"
        app.match_seed = sequence_seed(bank, sequence)
        app.preload_cinematic_poc_sequence(sequence)
        event = CinematicAttackEvent(50, side, True, "goal")
        event_seconds = (
            GOAL_EVENT_WINDOW_MINUTES
            / 90.0
            * SIMULATION_SECONDS
        )

        actor_state = app.cinematic_poc_scene_state(
            field,
            event,
            0.45,
        )
        actor_sample = actor_state["poc_contract_sample"]
        viewport = actor_state["poc_viewport"]
        if not hasattr(actor_sample, "actor_ground_y"):
            failures.append(
                f"{sequence.key}: invalid actor framebuffer state"
            )
            continue
        team = app.home if side == "home" else app.away
        (
            actor_source,
            _actor_index,
            _metadata,
            actor_scale,
        ) = app.cinematic_poc_actor_material(
            sequence,
            actor_sample,  # type: ignore[arg-type]
            team,
        )
        actor_scale *= viewport.scale  # type: ignore[union-attr]
        actor_size = max(
            1,
            round(POC_RUNNER_CANVAS_SIZE * actor_scale),
        )
        actor_root = viewport.point(  # type: ignore[union-attr]
            float(actor_state["poc_actor_x"]),
            actor_sample.actor_ground_y,  # type: ignore[union-attr]
        )
        actor_rect = pygame.Rect(
            round(
                actor_root[0]
                - POC_RUNNER_ROOT[0] * actor_scale
            ),
            round(
                actor_root[1]
                - POC_RUNNER_ROOT[1] * actor_scale
            ),
            actor_size,
            actor_size,
        )
        app.screen.fill(background)
        app.draw_cinematic_poc_actor(actor_state)
        assert_source_materialized(
            actual=app.screen,
            source=actor_source,
            destination=actor_rect,
            background=background,
            label=f"{sequence.key}/actor",
            failures=failures,
        )
        cases += 1

        impact_state = app.cinematic_poc_scene_state(
            field,
            event,
            1.0,
        )
        impact_sample = impact_state["poc_contract_sample"]
        impact_viewport = impact_state["poc_viewport"]
        ball_size = max(
            1,
            round(
                POC_BALL_CANVAS_SIZE
                * impact_viewport.scale  # type: ignore[union-attr]
            ),
        )
        expected_ball = pygame.Surface(app.screen.get_size())
        expected_ball.fill(background)
        ball_region: pygame.Rect | None = None
        for layer_sample, alpha in app.cinematic_poc_ball_layers(
            sequence,
            impact_sample,  # type: ignore[arg-type]
            float(impact_state["poc_elapsed"]),
            team,
        ):
            ball_frame_index = (
                int(
                    (layer_sample.ball_rotation % 360.0)
                    / 360.0
                    * len(app.assets.balls)
                )
                % len(app.assets.balls)
            )
            ball_layer = pygame.transform.smoothscale(
                app.assets.balls[ball_frame_index],
                (ball_size, ball_size),
            )
            if alpha < 255:
                ball_layer.set_alpha(alpha)
            ball_x = app.cinematic_poc_ball_x(
                sequence,
                layer_sample,
                team,
                impact_viewport,  # type: ignore[arg-type]
                impact_sample,  # type: ignore[arg-type]
            )
            ball_center = impact_viewport.point(  # type: ignore[union-attr]
                ball_x,
                layer_sample.ball_y,
            )
            ball_rect = ball_layer.get_rect(
                center=(
                    round(ball_center[0]),
                    round(ball_center[1]),
                )
            )
            expected_ball.blit(ball_layer, ball_rect)
            ball_region = (
                ball_rect.copy()
                if ball_region is None
                else ball_region.union(ball_rect)
            )
        app.screen.fill(background)
        app.draw_cinematic_poc_ball(impact_state)
        assert_canvas_region_equal(
            actual=app.screen,
            expected=expected_ball,
            region=ball_region or pygame.Rect(0, 0, 0, 0),
            label=f"{sequence.key}/ball",
            failures=failures,
        )
        cases += 1

        keeper_team = app.away if side == "home" else app.home
        keeper_frames = (
            app.assets.cinematic_keeper_frames_left
            if sequence.keeper_direction == "left"
            else app.assets.cinematic_keeper_frames
        )[keeper_team.code]
        keeper_source = keeper_frames[
            impact_sample.keeper_frame  # type: ignore[union-attr]
        ]
        keeper_size = impact_viewport.size(  # type: ignore[union-attr]
            340.0,
            340.0,
        )
        keeper_top_left = impact_viewport.point(  # type: ignore[union-attr]
            impact_sample.keeper_x,  # type: ignore[union-attr]
            impact_sample.keeper_y,  # type: ignore[union-attr]
        )
        keeper_rect = pygame.Rect(
            round(keeper_top_left[0]),
            round(keeper_top_left[1]),
            *keeper_size,
        )
        app.screen.fill(background)
        app.draw_cinematic_poc_keeper(impact_state)
        assert_source_materialized(
            actual=app.screen,
            source=keeper_source,
            destination=keeper_rect,
            background=background,
            label=f"{sequence.key}/keeper",
            failures=failures,
        )
        cases += 1

        contact_raw = (
            1.0 + 0.08 / max(1e-9, event_seconds)
        )
        goal_state = app.cinematic_poc_scene_state(
            field,
            event,
            contact_raw,
        )
        goal_sample = goal_state["poc_contract_sample"]
        goal_rect = goal_state["goal_rect"]
        if not isinstance(goal_rect, pygame.Rect):
            failures.append(
                f"{sequence.key}: invalid goal framebuffer state"
            )
            continue
        expected = pygame.Surface(app.screen.get_size())
        expected.fill(background)
        if (
            sequence.net_static_back is None
            or sequence.net_static_back_sha256 is None
        ):
            failures.append(
                f"{sequence.key}: missing goal framebuffer source"
            )
            continue
        static_back = app.load_cinematic_poc_layer(
            sequence.net_static_back,
            sequence.net_static_back_sha256,
        )
        expected.blit(
            pygame.transform.smoothscale(
                static_back,
                goal_rect.size,
            ),
            goal_rect,
        )
        net_frame = bank.nearest_net_keyframe(
            sequence,
            goal_sample.elapsed,  # type: ignore[union-attr]
        )
        if net_frame is None:
            failures.append(
                f"{sequence.key}: missing net framebuffer frame"
            )
            continue
        net_roi = app.load_cinematic_poc_atlas_frame(
            net_frame.back_roi,
            net_frame.back_roi_sha256,
            net_frame.back_roi_source_rect,
        )
        net_rect = app.cinematic_poc_goal_subrect(
            goal_rect,
            net_frame.back_roi_rect,
        )
        expected.blit(
            pygame.transform.smoothscale(
                net_roi,
                net_rect.size,
            ),
            net_rect,
        )
        contact_frame = bank.nearest_net_contact_frame(
            sequence,
            goal_sample.elapsed,  # type: ignore[union-attr]
        )
        if contact_frame is not None:
            contact = app.load_cinematic_poc_atlas_frame(
                contact_frame.front_contact,
                contact_frame.front_contact_sha256,
                contact_frame.front_contact_source_rect,
            )
            contact_rect = app.cinematic_poc_goal_subrect(
                goal_rect,
                contact_frame.front_contact_rect,
            )
            expected.blit(
                pygame.transform.smoothscale(
                    contact,
                    contact_rect.size,
                ),
                contact_rect,
            )
        front_asset = bank.goal_front_layers[
            sequence.goal_side
        ]
        front = app.load_cinematic_poc_layer(
            front_asset.file,
            front_asset.sha256,
        )
        expected.blit(
            pygame.transform.smoothscale(
                front,
                goal_rect.size,
            ),
            goal_rect,
        )
        app.screen.fill(background)
        app.draw_cinematic_poc_goal_layer(
            goal_state,
            front=False,
        )
        app.draw_cinematic_poc_goal_layer(
            goal_state,
            front=True,
        )
        assert_canvas_region_equal(
            actual=app.screen,
            expected=expected,
            region=goal_rect,
            label=f"{sequence.key}/goal",
            failures=failures,
        )
        cases += 1
    return cases


def validate_async_preload(
    bank: PocSequenceBank,
    failures: list[str],
) -> int:
    app = App(seed=20260728)
    sequences = tuple(
        bank.sequences[key]
        for key in sorted(bank.sequences)
    )
    assets = app.cinematic_poc_preload_assets(sequences)
    expected_paths = {path for path, _digest in assets}
    if len(assets) != EXPECTED_PRELOAD_ASSETS:
        failures.append(
            "approved POC preload inventory drift: "
            f"{len(assets)}/{EXPECTED_PRELOAD_ASSETS}"
        )

    app.state = "simulate"
    app.t = 0.25
    app.poc_preload_ready = False
    app.poc_preload_pending = 1
    app.update(1.0 / 60.0)
    if abs(app.t - 0.25) > 1e-9:
        failures.append(
            "match timeline advanced before approved POC preload"
        )
    app.state = "menu"
    app.cancel_cinematic_poc_preload()

    original_sync_preload = app.preload_cinematic_poc_sequence
    sync_calls: list[str] = []

    def forbidden_sync_preload(sequence: PocSequence) -> None:
        sync_calls.append(sequence.key)
        raise AssertionError(
            "synchronous POC preload reached match startup"
        )

    app.preload_cinematic_poc_sequence = forbidden_sync_preload  # type: ignore[method-assign]
    try:
        app.start_cinematic_poc_preload(sequences)
    finally:
        app.preload_cinematic_poc_sequence = original_sync_preload  # type: ignore[method-assign]
    if sync_calls:
        failures.append(
            "match preload called the synchronous sequence loader"
        )
    if (
        app.poc_preload_pending != len(assets)
        or app.poc_preload_completed != 0
        or app.poc_preload_ready
    ):
        failures.append(
            "approved POC preload did not return in pending state"
        )

    deadline = time.perf_counter() + 15.0
    while (
        not app.poc_preload_ready
        and not app.poc_preload_error
        and time.perf_counter() < deadline
    ):
        completed_before = app.poc_preload_completed
        app.drain_cinematic_poc_preload(max_assets=1)
        if app.poc_preload_completed - completed_before > 1:
            failures.append(
                "approved POC preload installed more than one asset per frame"
            )
            break
        if app.poc_preload_completed == completed_before:
            time.sleep(0.001)

    if app.poc_preload_error:
        failures.append(
            f"approved POC preload failed: {app.poc_preload_error}"
        )
    if not app.poc_preload_ready:
        failures.append(
            "approved POC preload did not complete asynchronously"
        )
    if app.poc_preload_completed != len(assets):
        failures.append(
            "approved POC preload completion drift: "
            f"{app.poc_preload_completed}/{len(assets)}"
        )
    missing_cache = expected_paths - set(app.poc_layer_cache)
    if missing_cache:
        failures.append(
            "approved POC preload cache is incomplete: "
            f"{sorted(missing_cache)}"
        )

    cache_misses: list[str] = []
    original_layer_load = app.load_cinematic_poc_layer

    def cache_only_layer(
        relative_path: str,
        expected_sha256: str,
    ) -> pygame.Surface:
        if relative_path not in app.poc_layer_cache:
            cache_misses.append(relative_path)
        return original_layer_load(
            relative_path,
            expected_sha256,
        )

    app.load_cinematic_poc_layer = cache_only_layer  # type: ignore[method-assign]
    try:
        for sequence in sequences:
            original_sync_preload(sequence)
    finally:
        app.load_cinematic_poc_layer = original_layer_load  # type: ignore[method-assign]
        app.cancel_cinematic_poc_preload()
    if cache_misses:
        failures.append(
            "renderer needed uncached POC assets after async preload: "
            f"{sorted(set(cache_misses))}"
        )
    app.start_cinematic_poc_preload(sequences)
    first_generation_thread = app.poc_preload_thread
    app.start_cinematic_poc_preload(sequences)
    if (
        first_generation_thread is not None
        and first_generation_thread.is_alive()
        and first_generation_thread not in app.poc_preload_threads
    ):
        failures.append(
            "replaced POC preload thread escaped runtime tracking"
        )
    if not app.shutdown_cinematic_poc_preloads(timeout=5.0):
        failures.append(
            "POC preload workers remained alive after bounded shutdown"
        )
    return len(assets)


def alpha_overlap_fraction(
    subject: pygame.Surface,
    subject_rect: pygame.Rect,
    occluder: pygame.Surface,
    occluder_rect: pygame.Rect,
) -> float:
    if not subject_rect.colliderect(occluder_rect):
        return 0.0
    subject_mask = pygame.mask.from_surface(subject, 30)
    subject_area = subject_mask.count()
    if subject_area <= 0:
        return 0.0
    occluder_mask = pygame.mask.from_surface(occluder, 30)
    offset = (
        occluder_rect.x - subject_rect.x,
        occluder_rect.y - subject_rect.y,
    )
    return subject_mask.overlap_area(
        occluder_mask,
        offset,
    ) / subject_area


def validate_authored_actor_progression(
    bank: PocSequenceBank,
    failures: list[str],
) -> int:
    app = App(seed=20260728)
    original_home = app.teams[app.home_idx]
    original_away = app.teams[app.away_idx]
    cases = 0
    try:
        for direction in DIRECTIONS:
            side = "home" if direction == "right" else "away"
            for uniform in CINEMATIC_UNIFORMS:
                profile = replace(
                    original_home if side == "home" else original_away,
                    key=f"qa-progress-{uniform.code}-{direction}",
                    code=f"QA_PROGRESS_{uniform.code}_{direction}",
                    kit=uniform.primary,
                )
                visited = {0: set(), 1: set()}
                grounding_failures: set[tuple[int, int]] = set()
                for sequence in bank.sequences.values():
                    if sequence.attack_direction != direction:
                        continue
                    for raw_sample in sequence.samples:
                        sample = bank.sample(
                            sequence,
                            float(raw_sample[0]),
                        )
                        if not sample.actor_visible:
                            continue
                        frame, frame_index, _metadata, actor_scale = (
                            app.cinematic_poc_actor_material(
                                sequence,
                                sample,
                                profile,
                            )
                        )
                        visited[sample.actor_source].add(frame_index)
                        grounding_key = (
                            sample.actor_source,
                            frame_index,
                        )
                        if grounding_key in grounding_failures:
                            continue
                        target_size = max(
                            1,
                            round(
                                POC_RUNNER_CANVAS_SIZE
                                * actor_scale
                            ),
                        )
                        rendered_frame = pygame.transform.smoothscale(
                            frame,
                            (target_size, target_size),
                        )
                        visible = rendered_frame.get_bounding_rect(
                            min_alpha=30
                        )
                        root_y = round(
                            POC_RUNNER_ROOT[1]
                            * actor_scale
                        )
                        ground_gap = root_y - visible.bottom
                        if not -3 <= ground_gap <= 5:
                            failures.append(
                                f"{uniform.code}/{direction}/"
                                f"{sample.actor_source}:{frame_index}: "
                                f"pixel ground gap {ground_gap}px"
                            )
                            grounding_failures.add(
                                grounding_key
                            )
                expected = set(range(16))
                for source, label in ((0, "run"), (1, "kick")):
                    if visited[source] != expected:
                        missing = sorted(expected - visited[source])
                        failures.append(
                            f"{uniform.code}/{direction}/{label}: "
                            f"authored frames not executed {missing}"
                        )
                cases += 1
    finally:
        app.teams[app.home_idx] = original_home
        app.teams[app.away_idx] = original_away
    return cases


def validate_all_uniform_ball_clearance(
    bank: PocSequenceBank,
    failures: list[str],
) -> tuple[int, float]:
    app = App(seed=20260728)
    field = app.match_field_rect()
    cases = 0
    max_overlap = 0.0
    original_home = app.teams[app.home_idx]
    original_away = app.teams[app.away_idx]
    try:
        for expected_sequence in bank.sequences.values():
            direction = expected_sequence.attack_direction
            side = "home" if direction == "right" else "away"
            for uniform in CINEMATIC_UNIFORMS:
                profile = replace(
                    original_home if side == "home" else original_away,
                    key=f"qa-{uniform.code}-{direction}",
                    code=f"QA_{uniform.code}_{direction}",
                    kit=uniform.primary,
                )
                if side == "home":
                    app.teams[app.home_idx] = profile
                    app.teams[app.away_idx] = original_away
                else:
                    app.teams[app.home_idx] = original_home
                    app.teams[app.away_idx] = profile
                app.match_seed = sequence_seed(
                    bank,
                    expected_sequence,
                )
                event = CinematicAttackEvent(
                    50,
                    side,
                    expected_sequence.outcome == "goal",
                    expected_sequence.outcome,
                )
                sequence = app.cinematic_poc_sequence_for_event(event)
                if sequence.key != expected_sequence.key:
                    failures.append(
                        f"{expected_sequence.key}/{uniform.code}: "
                        f"selected {sequence.key}"
                    )
                    continue
                for seconds_after_release in CLEARANCE_RELEASE_OFFSETS:
                    elapsed = (
                        sequence.release_seconds
                        + seconds_after_release
                    )
                    raw_progress = min(
                        1.0,
                        elapsed / sequence.impact_seconds,
                    )
                    state = app.cinematic_poc_scene_state(
                        field,
                        event,
                        raw_progress,
                    )
                    sample = state["poc_contract_sample"]
                    viewport = state["poc_viewport"]
                    if not hasattr(sample, "actor_ground_y"):
                        failures.append(
                            f"{uniform.code}/{direction}: missing actor sample"
                        )
                        continue
                    team = app.home if side == "home" else app.away
                    source_code = app.assets.cinematic_source_code(team)
                    if source_code != uniform.code:
                        failures.append(
                            f"{uniform.code}/{direction}: selected {source_code} assets"
                        )
                    actor, _index, _metadata, actor_scale = (
                        app.cinematic_poc_actor_material(
                            sequence,
                            sample,
                            team,
                        )
                    )
                    actor_scale *= viewport.scale
                    actor_size = (
                        max(
                            1,
                            round(
                                POC_RUNNER_CANVAS_SIZE
                                * actor_scale
                            ),
                        ),
                    ) * 2
                    actor_render = pygame.transform.smoothscale(
                        actor,
                        actor_size,
                    )
                    root = viewport.point(
                        float(state["poc_actor_x"]),
                        sample.actor_ground_y,
                    )
                    actor_rect = pygame.Rect(
                        round(
                            root[0]
                            - POC_RUNNER_ROOT[0] * actor_scale
                        ),
                        round(
                            root[1]
                            - POC_RUNNER_ROOT[1] * actor_scale
                        ),
                        *actor_size,
                    )
                    ball_size = int(state["ball_scale"])
                    ball_render = (
                        app.cached_cinematic_ball_material(
                            ball_size,
                            float(state["ball_rotation_degrees"]),
                        )
                    )
                    ball_rect = ball_render.get_rect(
                        center=(
                            round(float(state["ball_pos"][0])),
                            round(float(state["ball_pos"][1])),
                        )
                    )
                    overlap = alpha_overlap_fraction(
                        ball_render,
                        ball_rect,
                        actor_render,
                        actor_rect,
                    )
                    max_overlap = max(max_overlap, overlap)
                    if (
                        overlap
                        > CLEARANCE_MAX_ALPHA_OVERLAP_RATIO + 1e-9
                    ):
                        failures.append(
                            f"{sequence.key}/{uniform.code}/"
                            f"+{seconds_after_release:.4f}s: "
                            f"main ball/player alpha overlap {overlap:.3%}"
                        )
                    ball_layers = app.cinematic_poc_ball_layers(
                        sequence,
                        sample,
                        elapsed,
                        team,
                    )
                    layer_rects: list[
                        tuple[pygame.Surface, pygame.Rect]
                    ] = []
                    bounds: pygame.Rect | None = None
                    for layer_sample, alpha in ball_layers:
                        layer_ball = (
                            app.cached_cinematic_ball_material(
                                ball_size,
                                layer_sample.ball_rotation,
                            ).copy()
                        )
                        layer_ball.set_alpha(alpha)
                        layer_x = app.cinematic_poc_ball_x(
                            sequence,
                            layer_sample,
                            team,
                            viewport,
                            sample,
                        )
                        layer_center = viewport.point(
                            layer_x,
                            layer_sample.ball_y,
                        )
                        layer_rect = layer_ball.get_rect(
                            center=(
                                round(layer_center[0]),
                                round(layer_center[1]),
                            )
                        )
                        layer_rects.append((layer_ball, layer_rect))
                        bounds = (
                            layer_rect.copy()
                            if bounds is None
                            else bounds.union(layer_rect)
                        )
                    if bounds is None:
                        failures.append(
                            f"{sequence.key}/{uniform.code}: "
                            "renderer produced no ball layer"
                        )
                        continue
                    composite = pygame.Surface(
                        bounds.size,
                        pygame.SRCALPHA,
                    )
                    for layer_ball, layer_rect in layer_rects:
                        composite.blit(
                            layer_ball,
                            (
                                layer_rect.x - bounds.x,
                                layer_rect.y - bounds.y,
                            ),
                        )
                    layer_overlap = alpha_overlap_fraction(
                        composite,
                        bounds,
                        actor_render,
                        actor_rect,
                    )
                    max_overlap = max(max_overlap, layer_overlap)
                    if (
                        layer_overlap
                        > CLEARANCE_MAX_ALPHA_OVERLAP_RATIO + 1e-9
                    ):
                        failures.append(
                            f"{sequence.key}/{uniform.code}/"
                            f"+{seconds_after_release:.4f}s: "
                            "rendered ball-layer/player alpha overlap "
                            f"{layer_overlap:.3%}"
                        )
                    cases += 1
    finally:
        app.teams[app.home_idx] = original_home
        app.teams[app.away_idx] = original_away
    if max_overlap <= CLEARANCE_MAX_ALPHA_OVERLAP_RATIO:
        print(
            f"[poc-clearance] cases={cases} "
            f"max_overlap={max_overlap:.3%}",
            flush=True,
        )
    return cases, max_overlap


def e2e_prediction(
    direction: str,
    outcome: str,
) -> Prediction:
    is_goal = outcome == "goal"
    home_score = int(is_goal and direction == "right")
    away_score = int(is_goal and direction == "left")
    return Prediction(
        algorithm="CONFRONTO",
        home=0.55,
        draw=0.23,
        away=0.22,
        home_goals=float(home_score),
        away_goals=float(away_score),
        confidence=0.72,
        reason="POC runtime end-to-end parity",
        score_home=home_score,
        score_away=away_score,
        outcome_class=0 if home_score else 2 if away_score else 1,
        outcome_probability=0.55,
        score_probability=0.18,
        blend_probs=(0.55, 0.23, 0.22),
        poisson_outcome_probs=(0.52, 0.25, 0.23),
    )


def validate_end_to_end(
    bank: PocSequenceBank,
    failures: list[str],
) -> int:
    app = App(seed=20260728)
    event_minute = 50
    cases: list[PocSequence] = []
    for direction_index, direction in enumerate(DIRECTIONS):
        for outcome_index, outcome in enumerate(OUTCOMES):
            profile = PROFILES[
                (direction_index + outcome_index) % len(PROFILES)
            ]
            variant = (
                (direction_index + outcome_index)
                % bank.GOAL_VARIANTS
                if outcome == "goal"
                else 0
            )
            cases.append(
                bank.sequence(
                    direction,
                    profile,
                    outcome,
                    variant,
                )
            )

    original_draw = app.draw_cinematic_poc_sequence
    original_goal_layer = app.draw_cinematic_poc_goal_layer
    original_queue = app.queue_match_audio_event
    original_play = app.sound.play
    current_frame = 0
    rendered_keys: list[str] = []
    queued_frames: dict[str, int] = {}
    queued_pans: dict[str, float] = {}
    played_frames: dict[str, int] = {}
    played_pans: dict[str, float] = {}
    played_armed: dict[str, bool] = {}
    visual_frames: dict[str, int] = {}

    def draw_spy(
        field: pygame.Rect,
        state: dict[str, object],
    ) -> None:
        sequence = state.get("poc_contract_sequence")
        sample = state.get("poc_contract_sample")
        before = surface_rgba_sha256(
            app.screen.subsurface(field)
        )
        original_draw(field, state)
        after = surface_rgba_sha256(
            app.screen.subsurface(field)
        )
        if isinstance(sequence, PocSequence):
            rendered_keys.append(sequence.key)
            if hasattr(sample, "elapsed") and before != after:
                for name, seconds in sequence.audio_cues:
                    if (
                        name != "net"
                        and name not in visual_frames
                        and float(sample.elapsed) + 1e-9 >= seconds
                    ):
                        visual_frames[name] = current_frame

    def goal_layer_spy(
        state: dict[str, object],
        *,
        front: bool,
    ) -> None:
        sequence = state.get("poc_contract_sequence")
        sample = state.get("poc_contract_sample")
        goal_rect = state.get("goal_rect")
        visible_goal = (
            goal_rect.clip(app.screen.get_rect())
            if isinstance(goal_rect, pygame.Rect)
            else pygame.Rect(0, 0, 0, 0)
        )
        before = (
            surface_rgba_sha256(
                app.screen.subsurface(visible_goal)
            )
            if visible_goal.width > 0 and visible_goal.height > 0
            else ""
        )
        original_goal_layer(state, front=front)
        after = (
            surface_rgba_sha256(
                app.screen.subsurface(visible_goal)
            )
            if visible_goal.width > 0 and visible_goal.height > 0
            else ""
        )
        if (
            isinstance(sequence, PocSequence)
            and hasattr(sample, "elapsed")
            and sequence.outcome == "goal"
            and before != after
        ):
            net_seconds = dict(sequence.audio_cues).get("net")
            if (
                net_seconds is not None
                and float(sample.elapsed) + 1e-9 >= net_seconds
            ):
                visual_frames.setdefault("net", current_frame)

    def queue_spy(name: str, pan: float) -> None:
        queued_frames.setdefault(name, current_frame)
        queued_pans.setdefault(name, pan)
        original_queue(name, pan)

    def play_spy(
        name: str,
        *_args: object,
        **kwargs: object,
    ) -> None:
        played_frames.setdefault(name, current_frame)
        played_pans.setdefault(
            name,
            float(kwargs.get("pan", 0.0)),
        )
        played_armed.setdefault(
            name,
            bool(kwargs.get("already_armed", False)),
        )

    app.draw_cinematic_poc_sequence = draw_spy  # type: ignore[method-assign]
    app.draw_cinematic_poc_goal_layer = goal_layer_spy  # type: ignore[method-assign]
    app.queue_match_audio_event = queue_spy  # type: ignore[method-assign]
    app.sound.play = play_spy  # type: ignore[method-assign]
    try:
        for expected in cases:
            direction = expected.attack_direction
            side = "home" if direction == "right" else "away"
            outcome = expected.outcome
            prediction = e2e_prediction(direction, outcome)
            app.match_seed = sequence_seed(bank, expected)
            runtime_key = app.match_runtime_key(prediction)
            goals = (
                ((event_minute, side),)
                if outcome == "goal"
                else ()
            )
            chances = (
                ()
                if outcome == "goal"
                else ((event_minute, side, outcome),)
            )
            app.match_runtime_state_cache = {
                runtime_key: MatchRuntimeState(
                    key=runtime_key,
                    goals=goals,
                    chances=chances,
                )
            }
            app.state = "simulate"
            app.match_prediction = prediction
            app.match_intro_audio_pending = False
            app.final_whistle_played = False
            app.goal_events.clear()
            app.shot_events.clear()
            app.shot_progress_cursor.clear()
            app.match_audio_frame_queue.clear()
            rendered_keys.clear()
            queued_frames.clear()
            queued_pans.clear()
            played_frames.clear()
            played_pans.clear()
            played_armed.clear()
            visual_frames.clear()
            current_frame = 0
            event_window = (
                GOAL_EVENT_WINDOW_MINUTES
                if outcome == "goal"
                else CHANCE_EVENT_WINDOW_MINUTES
            )
            payoff = (
                GOAL_PAYOFF_MINUTES
                if outcome == "goal"
                else CHANCE_PAYOFF_MINUTES
            )
            start_minute = event_minute - event_window - 0.04
            end_minute = event_minute + payoff + 0.04
            app.t = (
                start_minute / 90.0 * SIMULATION_SECONDS
            )
            score_before = app.score_from_prediction(prediction)
            frame_count = math.ceil(
                (
                    end_minute - start_minute
                )
                / 90.0
                * SIMULATION_SECONDS
                * 60.0
            )
            score_change_frame: int | None = None
            for frame_index in range(frame_count):
                current_frame = frame_index
                app.update(1.0 / 60.0)
                app.draw()
                app.flush_queued_match_audio()
                if app.match_audio_frame_queue:
                    failures.append(
                        f"{expected.key}: audio queue remained populated after flush"
                    )
                    app.match_audio_frame_queue.clear()
                live_score = app.score_from_prediction(prediction)
                if (
                    score_change_frame is None
                    and live_score != score_before
                ):
                    score_change_frame = frame_index
            score_after = app.score_from_prediction(prediction)

            if expected.key not in rendered_keys:
                failures.append(
                    f"{expected.key}: App.update/draw did not execute POC renderer"
                )
            unexpected = set(rendered_keys) - {expected.key}
            if unexpected:
                failures.append(
                    f"{expected.key}: E2E selected unexpected sequences {sorted(unexpected)}"
                )
            expected_audio = expected_audio_names(outcome)
            if tuple(
                name
                for name in expected_audio
                if name in queued_frames
            ) != expected_audio:
                failures.append(
                    f"{expected.key}: E2E audio inventory/order drift"
                )
            if tuple(
                name
                for name in expected_audio
                if name in played_frames
            ) != expected_audio:
                failures.append(
                    f"{expected.key}: E2E audio flush/play order drift"
                )
            for name in expected_audio:
                if (
                    name in queued_frames
                    and name in visual_frames
                    and abs(
                        queued_frames[name] - visual_frames[name]
                    )
                    > 1
                ):
                    failures.append(
                        f"{expected.key}: {name} differs from visual cue by more than one frame"
                    )
                if (
                    name not in played_frames
                    or played_frames[name] != queued_frames.get(name)
                    or not played_armed.get(name, False)
                    or abs(
                        played_pans.get(name, 0.0)
                        - queued_pans.get(name, 0.0)
                    )
                    > 1e-9
                ):
                    failures.append(
                        f"{expected.key}: {name} was not flushed "
                        "once on its queued frame with the armed pan"
                    )
            expected_score = (
                (1, 0)
                if outcome == "goal" and side == "home"
                else (0, 1)
                if outcome == "goal"
                else (0, 0)
            )
            if score_before != (0, 0) or score_after != expected_score:
                failures.append(
                    f"{expected.key}: E2E score transition drift "
                    f"{score_before}->{score_after}"
                )
            if outcome == "goal":
                impact_frame = visual_frames.get("net")
                if (
                    score_change_frame is None
                    or impact_frame is None
                    or abs(score_change_frame - impact_frame) > 1
                ):
                    failures.append(
                        f"{expected.key}: score did not change on "
                        "the visual impact frame"
                    )
            elif score_change_frame is not None:
                failures.append(
                    f"{expected.key}: non-goal changed the score"
                )
    finally:
        app.draw_cinematic_poc_sequence = original_draw  # type: ignore[method-assign]
        app.draw_cinematic_poc_goal_layer = original_goal_layer  # type: ignore[method-assign]
        app.queue_match_audio_event = original_queue  # type: ignore[method-assign]
        app.sound.play = original_play  # type: ignore[method-assign]
    return len(cases)


def evidence_source_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                ROOT / "src" / "arena_ai" / "main.py",
                ROOT / "src" / "arena_ai" / "audio.py",
                ROOT
                / "src"
                / "arena_ai"
                / "cinematic_poc_runtime.py",
                ROOT
                / "src"
                / "arena_ai"
                / "cinematic_uniforms.py",
                Path(__file__).resolve(),
            }
        )
    )


def evidence_runtime_asset_paths() -> tuple[Path, ...]:
    cinematic = (
        ROOT
        / "assets"
        / "generated"
        / "cinematic"
    )
    paths = {
        cinematic / "runner_motion.json",
        cinematic / "keeper_motion.json",
        ROOT
        / "assets"
        / "generated"
        / "stadium_parallax_real.png",
        ROOT
        / "assets"
        / "generated"
        / "parallax"
        / "turf_mid_strip.png",
        ROOT
        / "assets"
        / "generated"
        / "parallax"
        / "turf_near_strip.png",
    }
    for index in range(32):
        paths.add(
            ROOT
            / "assets"
            / "generated"
            / "balls3d"
            / f"ball_{index}.png"
        )
    for direction in DIRECTIONS:
        for index in range(16):
            paths.add(
                cinematic
                / f"keeper_anim_{direction}_{index}.png"
            )
    for uniform in CINEMATIC_UNIFORMS:
        for stem in (
            "runner_smooth",
            "runner_left_smooth",
            "runner_kick_smooth",
            "runner_left_kick_smooth",
            "runner_stop_smooth",
            "runner_left_stop_smooth",
        ):
            paths.add(
                cinematic / f"{stem}_{uniform.code}.png"
            )
    paths.update(
        path
        for path in (cinematic / "poc7_net").rglob("*.png")
        if path.is_file()
    )
    missing = sorted(
        path.relative_to(ROOT).as_posix()
        for path in paths
        if not path.is_file()
    )
    if missing:
        raise RuntimeError(
            f"missing approved cinematic runtime assets: {missing}"
        )
    ordered = tuple(sorted(paths))
    if len(ordered) != EXPECTED_RUNTIME_ASSETS:
        raise RuntimeError(
            "approved cinematic runtime asset inventory drift: "
            f"{len(ordered)}/{EXPECTED_RUNTIME_ASSETS}"
        )
    return ordered


def manifest_entry_map(
    entries: object,
    *,
    label: str,
    failures: list[str],
) -> dict[str, dict[str, object]]:
    if not isinstance(entries, list):
        failures.append(
            f"cinematic evidence has invalid {label} inventory"
        )
        return {}
    mapped: dict[str, dict[str, object]] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            failures.append(
                f"cinematic evidence has malformed {label} entry"
            )
            continue
        path = str(raw.get("path", ""))
        if not path or path in mapped:
            failures.append(
                f"cinematic evidence has duplicate/empty {label} path"
            )
            continue
        mapped[path] = raw
    return mapped


def verify_evidence_file_entries(
    evidence_root: Path,
    entries: dict[str, dict[str, object]],
    expected_paths: tuple[Path, ...],
    *,
    label: str,
    failures: list[str],
) -> None:
    expected = {
        path.relative_to(ROOT).as_posix(): path
        for path in expected_paths
    }
    if set(entries) != set(expected):
        failures.append(
            f"cinematic evidence {label} inventory drift: "
            f"missing={sorted(set(expected) - set(entries))} "
            f"extra={sorted(set(entries) - set(expected))}"
        )
    for relative_path in sorted(set(entries) & set(expected)):
        path = expected[relative_path]
        if entries[relative_path].get("sha256") != sha256(path):
            failures.append(
                f"cinematic evidence stale {label}: {relative_path}"
            )


def expected_recorded_report_metrics(
    bank: PocSequenceBank,
    *,
    renderer_cases: int,
    framebuffer_cases: int,
    preload_assets: int,
    actor_progression_cases: int,
    clearance_cases: int,
    clearance_max_overlap: float,
    e2e_cases: int,
    sequential_net_cases: int,
) -> dict[str, object]:
    return {
        "passed": True,
        "failures": [],
        "contract": CONTRACT.relative_to(ROOT).as_posix(),
        "contract_sha256": bank.sha256,
        "contract_version": bank.VERSION,
        "sequence_count": len(bank.sequences),
        "sample_hz": bank.sample_hz,
        "renderer_cases": renderer_cases,
        "expected_renderer_cases": EXPECTED_RENDERER_CASES,
        "framebuffer_cases": framebuffer_cases,
        "expected_framebuffer_cases": EXPECTED_FRAMEBUFFER_CASES,
        "preload_assets": preload_assets,
        "expected_preload_assets": EXPECTED_PRELOAD_ASSETS,
        "actor_progression_cases": actor_progression_cases,
        "expected_actor_progression_cases": (
            EXPECTED_ACTOR_PROGRESSION_CASES
        ),
        "uniform_clearance_cases": clearance_cases,
        "expected_uniform_clearance_cases": (
            EXPECTED_CLEARANCE_CASES
        ),
        "uniform_clearance_max_overlap_ratio": round(
            clearance_max_overlap,
            8,
        ),
        "uniform_clearance_limit_ratio": (
            CLEARANCE_MAX_ALPHA_OVERLAP_RATIO
        ),
        "uniform_clearance_rounding_safety_px": (
            CLEARANCE_RUNTIME_ROUNDING_SAFETY_PX
        ),
        "e2e_cases": e2e_cases,
        "expected_e2e_cases": EXPECTED_E2E_CASES,
        "evidence_check_requested": False,
        "evidence_replay_frames": 0,
        "expected_evidence_replay_frames": 0,
        "sequential_net_cases": sequential_net_cases,
        "expected_sequential_net_cases": (
            EXPECTED_SEQUENTIAL_NET_CASES
        ),
    }


def validate_recorded_evidence(
    evidence_root: Path,
    bank: PocSequenceBank,
    expected_report_metrics: dict[str, object],
    failures: list[str],
) -> int:
    manifest_path = evidence_root / "manifest.json"
    if not manifest_path.is_file():
        failures.append(
            f"cinematic evidence manifest is missing: {manifest_path}"
        )
        return 0
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        failures.append(
            "cinematic evidence manifest schema drift: "
            f"{manifest.get('schema_version')}/{EVIDENCE_SCHEMA_VERSION}"
        )
    if manifest.get("artifact") != EVIDENCE_ARTIFACT:
        failures.append(
            "cinematic evidence artifact identity drift"
        )
    if manifest.get("capture_policy") != EVIDENCE_CAPTURE_POLICY:
        failures.append(
            "cinematic evidence capture policy drift"
        )
    contract = manifest.get("contract")
    expected_contract = {
        "path": CONTRACT.relative_to(ROOT).as_posix(),
        "sha256": bank.sha256,
        "version": bank.VERSION,
    }
    if contract != expected_contract:
        failures.append(
            "cinematic evidence contract provenance is stale"
        )

    source_entries = manifest_entry_map(
        manifest.get("runtime_sources"),
        label="runtime source",
        failures=failures,
    )
    verify_evidence_file_entries(
        evidence_root,
        source_entries,
        evidence_source_paths(),
        label="runtime source",
        failures=failures,
    )
    asset_entries = manifest_entry_map(
        manifest.get("runtime_assets"),
        label="runtime asset",
        failures=failures,
    )
    verify_evidence_file_entries(
        evidence_root,
        asset_entries,
        evidence_runtime_asset_paths(),
        label="runtime asset",
        failures=failures,
    )

    report_entry = manifest.get("integration_report")
    if not isinstance(report_entry, dict):
        failures.append(
            "cinematic evidence integration report entry is missing"
        )
    else:
        report_relative = str(report_entry.get("path", ""))
        report_path = (evidence_root / report_relative).resolve()
        if (
            evidence_root.resolve() not in report_path.parents
            or not report_path.is_file()
        ):
            failures.append(
                "cinematic evidence integration report is missing"
            )
        else:
            if report_entry.get("sha256") != sha256(report_path):
                failures.append(
                    "cinematic evidence integration report hash is stale"
                )
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            mismatches = {
                key: {
                    "recorded": report.get(key),
                    "fresh": expected,
                }
                for key, expected in expected_report_metrics.items()
                if report.get(key) != expected
            }
            if mismatches:
                failures.append(
                    "cinematic evidence integration report did not pass "
                    f"the current contract: {mismatches}"
                )

    frame_entries = manifest_entry_map(
        manifest.get("frames"),
        label="frame",
        failures=failures,
    )
    recorded_files = {
        path.relative_to(evidence_root).as_posix()
        for path in (evidence_root / "frames").glob("*.png")
        if path.is_file()
    }
    if (
        len(frame_entries) != EXPECTED_RENDERER_CASES
        or set(frame_entries) != recorded_files
    ):
        failures.append(
            "cinematic evidence frame inventory drift: "
            f"manifest={len(frame_entries)} files={len(recorded_files)} "
            f"expected={EXPECTED_RENDERER_CASES}"
        )
    for relative_path in sorted(
        set(frame_entries) & recorded_files
    ):
        frame_path = evidence_root / relative_path
        entry = frame_entries[relative_path]
        if entry.get("sha256") != sha256(frame_path):
            failures.append(
                f"cinematic evidence frame hash is stale: {relative_path}"
            )
        image = pygame.image.load(frame_path)
        if (
            entry.get("rgba_sha256")
            != surface_rgba_sha256(image)
        ):
            failures.append(
                f"cinematic evidence frame pixels are stale: {relative_path}"
            )

    with tempfile.TemporaryDirectory(
        prefix="arena-poc7-evidence-"
    ) as temp_dir:
        replay_dir = Path(temp_dir) / "frames"
        replay_failures: list[str] = []
        replay_count = validate_renderer(
            bank,
            replay_failures,
            replay_dir,
        )
        if replay_failures:
            failures.extend(
                f"evidence replay: {failure}"
                for failure in replay_failures
            )
        replay_files = {
            path.relative_to(replay_dir).as_posix(): path
            for path in replay_dir.glob("*.png")
            if path.is_file()
        }
        recorded_names = {
            Path(path).relative_to("frames").as_posix()
            for path in frame_entries
            if Path(path).parts[:1] == ("frames",)
        }
        if (
            replay_count != EXPECTED_RENDERER_CASES
            or set(replay_files) != recorded_names
        ):
            failures.append(
                "cinematic evidence replay inventory drift: "
                f"{replay_count}/{EXPECTED_RENDERER_CASES}"
            )
        for name in sorted(set(replay_files) & recorded_names):
            replay_image = pygame.image.load(replay_files[name])
            recorded_entry = frame_entries[f"frames/{name}"]
            if (
                surface_rgba_sha256(replay_image)
                != recorded_entry.get("rgba_sha256")
            ):
                failures.append(
                    f"cinematic evidence replay pixel drift: {name}"
                )
        return replay_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida o contrato cinematografico no runtime real."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--check-evidence", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.capture_dir is not None
        and args.check_evidence is not None
    ):
        raise RuntimeError(
            "capture and evidence replay must be separate operations"
        )
    started = time.perf_counter()
    failures: list[str] = []
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    bank = PocSequenceBank(CONTRACT)
    validate_provenance(payload, failures)
    expected_keys = PocSequenceBank.expected_keys()
    if (
        len(expected_keys) != EXPECTED_SEQUENCE_COUNT
        or set(bank.sequences) != expected_keys
    ):
        failures.append("contract does not contain the exact 30-case matrix")
    for key in sorted(expected_keys & set(bank.sequences)):
        validate_sequence(bank, bank.sequences[key], failures)
    validate_profile_ordering(bank, failures)
    sequential_net_cases = validate_sequential_net_contract(
        bank,
        payload,
        failures,
    )
    capture_dir = args.capture_dir
    if capture_dir is not None and not capture_dir.is_absolute():
        capture_dir = ROOT / capture_dir
    if capture_dir is not None:
        evidence_root = capture_dir.parent.resolve()
        allowed_root = (
            ROOT
            / "artifacts"
            / "cinematic_game_qa"
        ).resolve()
        if (
            evidence_root.name != "current"
            or allowed_root not in evidence_root.parents
        ):
            raise RuntimeError(
                f"unsafe cinematic evidence root: {evidence_root}"
            )
        if evidence_root.exists():
            shutil.rmtree(evidence_root)
        capture_dir.mkdir(parents=True, exist_ok=True)
    rendered = validate_renderer(
        bank,
        failures,
        capture_dir,
    )
    framebuffer_cases = validate_framebuffer_materialization(
        bank,
        failures,
    )
    preload_assets = validate_async_preload(
        bank,
        failures,
    )
    actor_progression_cases = validate_authored_actor_progression(
        bank,
        failures,
    )
    clearance_cases, clearance_max_overlap = (
        validate_all_uniform_ball_clearance(
            bank,
            failures,
        )
    )
    e2e_cases = validate_end_to_end(
        bank,
        failures,
    )
    evidence_replay_frames = 0
    if args.check_evidence is not None:
        evidence_root = args.check_evidence
        if not evidence_root.is_absolute():
            evidence_root = ROOT / evidence_root
        evidence_root = evidence_root.resolve()
        allowed_root = (
            ROOT
            / "artifacts"
            / "cinematic_game_qa"
        ).resolve()
        if (
            evidence_root.name != "current"
            or allowed_root not in evidence_root.parents
        ):
            raise RuntimeError(
                f"unsafe cinematic evidence check root: {evidence_root}"
            )
        evidence_replay_frames = validate_recorded_evidence(
            evidence_root,
            bank,
            expected_recorded_report_metrics(
                bank,
                renderer_cases=rendered,
                framebuffer_cases=framebuffer_cases,
                preload_assets=preload_assets,
                actor_progression_cases=actor_progression_cases,
                clearance_cases=clearance_cases,
                clearance_max_overlap=clearance_max_overlap,
                e2e_cases=e2e_cases,
                sequential_net_cases=sequential_net_cases,
            ),
            failures,
        )
    actual_counts = {
        "sequence": len(bank.sequences),
        "renderer": rendered,
        "framebuffer": framebuffer_cases,
        "preload assets": preload_assets,
        "actor progression": actor_progression_cases,
        "clearance": clearance_cases,
        "E2E": e2e_cases,
        "sequential net": sequential_net_cases,
    }
    expected_counts = {
        "sequence": EXPECTED_SEQUENCE_COUNT,
        "renderer": EXPECTED_RENDERER_CASES,
        "framebuffer": EXPECTED_FRAMEBUFFER_CASES,
        "preload assets": EXPECTED_PRELOAD_ASSETS,
        "actor progression": EXPECTED_ACTOR_PROGRESSION_CASES,
        "clearance": EXPECTED_CLEARANCE_CASES,
        "E2E": EXPECTED_E2E_CASES,
        "sequential net": EXPECTED_SEQUENTIAL_NET_CASES,
    }
    for label, expected_count in expected_counts.items():
        if actual_counts[label] != expected_count:
            failures.append(
                f"{label} matrix count drift: "
                f"{actual_counts[label]}/{expected_count}"
            )
    report = {
        "passed": not failures,
        "contract": CONTRACT.relative_to(ROOT).as_posix(),
        "contract_sha256": bank.sha256,
        "contract_version": bank.VERSION,
        "sequence_count": len(bank.sequences),
        "sample_hz": bank.sample_hz,
        "renderer_cases": rendered,
        "expected_renderer_cases": EXPECTED_RENDERER_CASES,
        "framebuffer_cases": framebuffer_cases,
        "expected_framebuffer_cases": (
            EXPECTED_FRAMEBUFFER_CASES
        ),
        "preload_assets": preload_assets,
        "expected_preload_assets": EXPECTED_PRELOAD_ASSETS,
        "actor_progression_cases": actor_progression_cases,
        "expected_actor_progression_cases": (
            EXPECTED_ACTOR_PROGRESSION_CASES
        ),
        "uniform_clearance_cases": clearance_cases,
        "expected_uniform_clearance_cases": (
            EXPECTED_CLEARANCE_CASES
        ),
        "uniform_clearance_max_overlap_ratio": round(
            clearance_max_overlap,
            8,
        ),
        "uniform_clearance_limit_ratio": (
            CLEARANCE_MAX_ALPHA_OVERLAP_RATIO
        ),
        "uniform_clearance_rounding_safety_px": (
            CLEARANCE_RUNTIME_ROUNDING_SAFETY_PX
        ),
        "e2e_cases": e2e_cases,
        "expected_e2e_cases": EXPECTED_E2E_CASES,
        "evidence_check_requested": (
            args.check_evidence is not None
        ),
        "evidence_replay_frames": evidence_replay_frames,
        "expected_evidence_replay_frames": (
            EXPECTED_RENDERER_CASES
            if args.check_evidence is not None
            else 0
        ),
        "sequential_net_cases": sequential_net_cases,
        "expected_sequential_net_cases": (
            EXPECTED_SEQUENTIAL_NET_CASES
        ),
        "failures": failures,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    encoded = json.dumps(
        report,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    output = args.output
    if output is None and capture_dir is not None:
        output = capture_dir.parent / "integration_report.json"
    if output is not None:
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    if capture_dir is not None:
        if output is None:
            raise RuntimeError("missing cinematic integration report")
        frames = sorted(capture_dir.glob("*.png"))
        if len(frames) != EXPECTED_RENDERER_CASES:
            raise RuntimeError(
                "incomplete cinematic evidence frame inventory: "
                f"{len(frames)}/{EXPECTED_RENDERER_CASES}"
            )
        evidence_root = capture_dir.parent
        source_files = evidence_source_paths()
        runtime_assets = evidence_runtime_asset_paths()
        manifest = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "artifact": EVIDENCE_ARTIFACT,
            "capture_policy": EVIDENCE_CAPTURE_POLICY,
            "contract": {
                "path": CONTRACT.relative_to(ROOT).as_posix(),
                "sha256": bank.sha256,
                "version": bank.VERSION,
            },
            "runtime_sources": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(path),
                }
                for path in source_files
            ],
            "runtime_assets": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(path),
                }
                for path in runtime_assets
            ],
            "integration_report": {
                "path": output.relative_to(evidence_root).as_posix(),
                "sha256": sha256(output),
            },
            "frames": [
                {
                    "path": frame.relative_to(evidence_root).as_posix(),
                    "sha256": sha256(frame),
                    "rgba_sha256": surface_rgba_sha256(
                        pygame.image.load(frame)
                    ),
                }
                for frame in frames
            ],
        }
        (evidence_root / "manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(encoded)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import hashlib
import io
import math
import os
import queue
import random
import sys
import threading
import time
from bisect import bisect_left
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from arena_ai.audio import AudioEngine, pre_init_mixer
from arena_ai.audio_manifest import CUP_PROGRESS_MARKERS
from arena_ai.cinematic_dribble_runtime import (
    Poc2DribbleRuntime,
    Poc2DribbleSample,
)
from arena_ai.cinematic_poc_runtime import (
    POC_APPROVED_REFERENCE_VISIBLE_HEIGHT,
    POC_BALL_CANVAS_SIZE,
    POC_GROUND_Y,
    POC_HEADER_HEIGHT,
    POC_RUNNER_CANVAS_SIZE,
    POC_RUNNER_ROOT,
    PocNetKeyframe,
    PocSequence,
    PocSequenceBank,
    PocSequenceSample,
    PocViewport,
)
from arena_ai.cinematic_uniforms import CINEMATIC_UNIFORMS, TEAM_UNIFORM_OVERRIDES
from arena_ai.rendering import SurfaceCache, TextCache
from arena_ai.ui import Button
from arena_ai.worldcup_model import MatchAnalysis, Prediction, TeamProfile, WorldCupModel, effective_monte_carlo_workers


WIDTH = 1280
HEIGHT = 760
FPS = 60
MAX_FRAME_DT = 1.0 / 20.0
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
ASSETS = ROOT / "assets"

BG = (5, 13, 19)
PANEL = (11, 25, 35)
PANEL_2 = (18, 39, 51)
WHITE = (242, 247, 250)
MUTED = (150, 174, 187)
LINE = (219, 235, 226)
GOLD = (250, 195, 67)
BLUE = (72, 153, 255)
RED = (234, 76, 82)
GREEN = (74, 214, 122)
CYAN = (82, 226, 255)
PURPLE = (181, 132, 255)
BLACK = (4, 8, 11)

ALGORITHMS = ["CONFRONTO"]
SIMULATION_SECONDS = 45.0
FINAL_ACTION_START = 0.76
CINEMATIC_POSE_SIZE = 192
CINEMATIC_PLAYER_SCALE = 0.90
CINEMATIC_NEUTRAL_PLAYER_SCALE = 0.90
CINEMATIC_KEEPER_SCALE = 0.84
CINEMATIC_KEEPER_GROUND_OFFSET = 72.0
CINEMATIC_GOAL_EDGE_INSET = 23
CINEMATIC_GOAL_WIDTH = 442
CINEMATIC_GOAL_HEIGHT = 244
CINEMATIC_GOAL_BOTTOM_INSET = 74
CINEMATIC_GOAL_LANE_SHIFT = 18
CINEMATIC_GOAL_ENTRY_DEPTH = 148.0
CINEMATIC_RUNNER_EDGE_INSET = 180.0
CINEMATIC_KEEPER_POC7_DIVE_OFFSET = 15.5
CINEMATIC_GOAL_SOURCE_WIDTH = 1774.0
CINEMATIC_GOAL_SOURCE_HEIGHT = 887.0
CINEMATIC_GOAL_SOURCE_POST_X = (230.0, 1342.0)
CINEMATIC_GOAL_SOURCE_POST_BOTTOM = 735.0
CINEMATIC_NET_TARGET_DEPTH_MIN_RATIO = 0.16
CINEMATIC_NET_TARGET_DEPTH_MAX_RATIO = 0.22
CINEMATIC_GOAL_BALL_REST_DEPTH = 296.0
CINEMATIC_GOAL_BALL_REST_DEPTH_VARIATION = 24
CINEMATIC_GOAL_COMPOSITE_CACHE_LIMIT = 96
CINEMATIC_POC_NET_SAFE_FRAME_REMAP = {4: 3, 5: 3, 6: 7}
CINEMATIC_SHOT_SPIN_DISTANCE_DIVISOR = 64.0
CINEMATIC_BALL_SIZE = 38
CINEMATIC_SHOT_BALL_SIZE = 32
CINEMATIC_NET_BALL_SIZE = 30
CINEMATIC_BALL_GROUND_RADIUS_RATIO = 0.43
CINEMATIC_BALL_MATERIAL_FRAME_COUNT = 32
CINEMATIC_BALL_MATERIAL_BLEND_STEPS = 1
CINEMATIC_BALL_VELOCITY_SAMPLE_HZ = 120.0
CINEMATIC_RUNNER_FRAME_COUNT = 16
CINEMATIC_KICK_FRAME_COUNT = 16
CINEMATIC_STOP_FRAME_COUNT = 8
CINEMATIC_KEEPER_FRAME_COUNT = 16
CINEMATIC_POC2_SOURCE_CYCLE_SECONDS = 0.8
CINEMATIC_POC2_PLAYBACK_SPEED = 0.5
CINEMATIC_POC2_CYCLE_SECONDS = (
    CINEMATIC_POC2_SOURCE_CYCLE_SECONDS
    / CINEMATIC_POC2_PLAYBACK_SPEED
)
CINEMATIC_POC2_FRAME_DURATIONS = (
    0.11,
    0.19,
    0.04,
    0.06,
    0.13,
    0.18,
    0.04,
    0.05,
)
CINEMATIC_POC2_CONTACT_OFFSETS_RIGHT = (118.0, 128.0)
CINEMATIC_POC2_CONTACT_OFFSETS_LEFT = (121.0, 123.0)
CINEMATIC_POC2_SOURCE_CYCLE_DISTANCE = 300.0
CINEMATIC_POC2_RENDER_SCALE = (
    CINEMATIC_POSE_SIZE
    * CINEMATIC_PLAYER_SCALE
    / POC_APPROVED_REFERENCE_VISIBLE_HEIGHT
)
CINEMATIC_RUNNER_STRIDE_DISTANCE = (
    CINEMATIC_POC2_SOURCE_CYCLE_DISTANCE
    * CINEMATIC_POC2_RENDER_SCALE
)
CINEMATIC_TURF_FOREGROUND_PARALLAX = 0.78
CINEMATIC_TURF_SPEED = (
    CINEMATIC_RUNNER_STRIDE_DISTANCE
    / CINEMATIC_POC2_CYCLE_SECONDS
    / CINEMATIC_TURF_FOREGROUND_PARALLAX
)
CINEMATIC_DRIBBLE_CONTROL_OFFSET = 52.0
CINEMATIC_DRIBBLE_CONTACT_GAP = 0.8
CINEMATIC_DRIBBLE_TOUCH_PHASE_OFFSET = 0.5
CINEMATIC_DRIBBLE_TOUCH_PHASE_OFFSET_LEFT = 0.25
CINEMATIC_DRIBBLE_BALL_EXCURSION_RIGHT = 17.0
CINEMATIC_DRIBBLE_BALL_EXCURSION_LEFT = 17.5
CINEMATIC_DRIBBLE_FOOT_RETREAT = 1.8
CINEMATIC_DRIBBLE_PUSH_PEAK = 0.46
CINEMATIC_DRIBBLE_MAX_GAP = 22.0
CINEMATIC_KICK_CONTACT_GAP = -3.0
CINEMATIC_LEFT_ATTACK_CONTACT_ADVANCE = 10.0
CINEMATIC_BALL_CORRIDOR_START = 0.20
CINEMATIC_KICK_APPROACH_START = 0.48
CINEMATIC_KICK_APPROACH_DISTANCE = 34.0
CINEMATIC_CONTACT_CATCHUP_MAX = 12.0
CINEMATIC_SHOT_CONTACT_DISTANCE_RATIO = 0.402
CINEMATIC_WIDE_POST_OFFSET = 16.5
SHOT_KICK_AT = 0.80
SHOT_WHOOSH_AT = 0.828
SHOT_NET_AT = 0.955
SHOT_NET_VISUAL_CONTACT_AT = 0.995
SHOT_NET_PENETRATION_RATIO = 1.0
SHOT_KICK_AUDIO_AT = SHOT_KICK_AT - 0.006
SHOT_WHOOSH_AUDIO_AT = SHOT_WHOOSH_AT - 0.006
SHOT_NET_AUDIO_AT = SHOT_NET_VISUAL_CONTACT_AT - 0.004
SHOT_BASS_AUDIO_AT = SHOT_NET_VISUAL_CONTACT_AT - 0.001
SHOT_CHEER_AUDIO_AT = SHOT_NET_VISUAL_CONTACT_AT + 0.002
SHOT_REVERB_AT = SHOT_NET_VISUAL_CONTACT_AT + 0.004
CHANCE_CONTACT_VISUAL_AT = SHOT_NET_AT
WIDE_CONTACT_VISUAL_AT = CHANCE_CONTACT_VISUAL_AT + 0.003
CHANCE_CONTACT_AUDIO_AT = CHANCE_CONTACT_VISUAL_AT - 0.008
SHOT_PLANT_AT = 0.70
SHOT_CONTACT_FREEZE_END = 0.83
SHOT_KICK_POSE_AT = SHOT_KICK_AT
SHOT_RELEASE_END = 0.88
SHOT_RECOVERY_AT = 1.010
SHOT_STOP_BLEND_END = SHOT_RECOVERY_AT + 0.12
SHOT_FOLLOW_THROUGH_HOLD_END = 1.001
SHOT_RUN_TO_PLANT_AT = SHOT_PLANT_AT - 0.08
SHOT_FOLLOW_THROUGH_AT = 0.90
SHOT_RECOVERY_BLEND_AT = 0.94
SHOT_GOAL_REVEAL_AT = 0.06
SHOT_GOAL_FULL_AT = 0.20
SHOT_KEEPER_REVEAL_AT = 0.14
SHOT_KEEPER_FULL_AT = 0.32
SHOT_KEEPER_READ_AT = 0.50
SHOT_KEEPER_DIVE_AT = 0.74
SHOT_NET_SETTLE_PROGRESS = 0.535
DRAW_NEUTRAL_START_PROGRESS = 0.958
DRAW_NEUTRAL_RAMP = 0.036
SHOT_PHASE_APPROACH = "approach"
SHOT_PHASE_PLANT = "plant"
SHOT_PHASE_CONTACT_FREEZE = "contact_freeze"
SHOT_PHASE_RELEASE = "release"
SHOT_PHASE_FOLLOW_THROUGH = "follow_through"
SHOT_PHASE_NET_IMPACT = "net_impact"
SHOT_PHASE_RECOVERY = "recovery"
GOAL_IMPACT_AUDIO_EVENTS = {"net", "bass", "cheer"}
GOAL_EVENT_WINDOW_MINUTES = 6.0
GOAL_PAYOFF_MINUTES = 4.6
GOAL_MIN_SPACING_MINUTES = 11
CHANCE_EVENT_WINDOW_MINUTES = 6.0
CHANCE_PAYOFF_MINUTES = 4.6
CHANCE_MIN_SPACING_MINUTES = 13
KICK_FOOT_ANCHOR = (0.84, 0.67)
UNIFORMS_BY_CODE = {uniform.code: uniform for uniform in CINEMATIC_UNIFORMS}
TOURNAMENT_MONTE_CARLO_RUNS = max(1000, int(os.environ.get("ARENA_AI_TOURNAMENT_MC_RUNS", "1000")))
TOURNAMENT_MONTE_CARLO_WORKERS = effective_monte_carlo_workers(int(os.environ.get("ARENA_AI_TOURNAMENT_MC_WORKERS", "8")))


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_mc_fresh_override = os.environ.get("ARENA_AI_TOURNAMENT_MC_FRESH")
TOURNAMENT_MONTE_CARLO_USE_SCENARIO_BANK = (
    _mc_fresh_override.strip().lower() not in {"1", "true", "yes", "on"}
    if _mc_fresh_override is not None
    else env_flag("ARENA_AI_TOURNAMENT_MC_BOOTSTRAP", False)
)
TOURNAMENT_MIN_LOADING_SECONDS = max(1.5, float(os.environ.get("ARENA_AI_TOURNAMENT_MIN_LOADING_SECONDS", "3.2")))
ALGORITHM_COLORS = {"CONFRONTO": GOLD}
ALGORITHM_NAMES = {"CONFRONTO": "XGBoost + Poisson/DC"}
MATCH_HUD_STATE_COPY = {
    "live": ("JOGO EM ABERTO", "Nada decidido", "Final só no apito"),
    "focus": ("PRESSÃO NA ÁREA", "Lance vivo", "Final só no apito"),
    "closed": ("APITO FINAL", "Resultado revelado", "Leitura completa"),
}
MATCH_HUD_BANNED_COPY = (
    "Pacote:",
    "Peso do placar:",
    "Força do placar:",
    "Mata-mata:",
    "45 s reais",
    "Motores em sinergia",
    "drama vivo",
    "Auditoria ML",
    "Auditoria no apito",
    "Painel do modelo",
    "MODELO FECHADO",
    "Auditoria concluída",
    "Placares ocultos",
    "Matriz de gols",
    "Mix final",
    "Sorteio híbrido",
    "Amostra travada no apito.",
)
MATCH_HUD_REQUIRED_COPY = tuple(copy for state_copy in MATCH_HUD_STATE_COPY.values() for copy in state_copy) + ("Sorteio da Copa",)
MATCH_HUD_TOP_SCORE_COUNT = 5
FIELD_NARRATION_COPY = {
    "home": ("Ataque da casa", "A bola entra no terço final."),
    "away": ("Resposta visitante", "O contra-ataque ganha campo."),
}
FIFA_EXTERNAL_IMAGES = {
    "maple": "fifa_maple.jpg",
    "zayu": "fifa_zayu.jpg",
    "clutch": "fifa_clutch.jpg",
    "club_final": "fifa_club_world_cup_final_2025.jpg",
    "mexico_opening": "fifa_mexico_opening_ceremony_clean.png",
    "detail": "fifa_detail_image_03.jpg",
    "ecomm": "fwc26_ecomm_photo_update_b.jpg",
}
TEAM_DISPLAY_NAMES_PT = {
    "ALG": "Argélia",
    "ARG": "Argentina",
    "AUS": "Austrália",
    "AUT": "Áustria",
    "BEL": "Bélgica",
    "BIH": "Bósnia e Herzegovina",
    "BRA": "Brasil",
    "CAN": "Canadá",
    "CIV": "Costa do Marfim",
    "COD": "RD Congo",
    "COL": "Colômbia",
    "CPV": "Cabo Verde",
    "CRO": "Croácia",
    "CUR": "Curaçao",
    "CZE": "Tchéquia",
    "ECU": "Equador",
    "EGY": "Egito",
    "ENG": "Inglaterra",
    "ESP": "Espanha",
    "FRA": "França",
    "GER": "Alemanha",
    "GHA": "Gana",
    "HAI": "Haiti",
    "IRN": "Irã",
    "IRQ": "Iraque",
    "JOR": "Jordânia",
    "JPN": "Japão",
    "KOR": "Coreia do Sul",
    "KSA": "Arábia Saudita",
    "MAR": "Marrocos",
    "MEX": "México",
    "NED": "Países Baixos",
    "NOR": "Noruega",
    "NZL": "Nova Zelândia",
    "PAN": "Panamá",
    "PAR": "Paraguai",
    "POR": "Portugal",
    "QAT": "Catar",
    "RSA": "África do Sul",
    "SCO": "Escócia",
    "SEN": "Senegal",
    "SUI": "Suíça",
    "SWE": "Suécia",
    "TUN": "Tunísia",
    "TUR": "Turquia",
    "URU": "Uruguai",
    "USA": "Estados Unidos",
    "UZB": "Uzbequistão",
}
_FONT_CACHE: dict[tuple[int, bool], pygame.font.Font] = {}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(value: float) -> float:
    value = clamp(value)
    return value * value * (3 - 2 * value)


def cinematic_dribble_touch_phase_offset(direction: int) -> float:
    return (
        CINEMATIC_DRIBBLE_TOUCH_PHASE_OFFSET
        if direction > 0
        else CINEMATIC_DRIBBLE_TOUCH_PHASE_OFFSET_LEFT
    )


def cinematic_dribble_touch_phase(stride_phase: float, direction: int = 1) -> float:
    return ((stride_phase - cinematic_dribble_touch_phase_offset(direction)) % 2.0) / 2.0


def cinematic_poc2_runtime_frame_position(stride_phase: float) -> float:
    cycle_elapsed = (
        (stride_phase % 4.0)
        / 4.0
        * CINEMATIC_POC2_SOURCE_CYCLE_SECONDS
    )
    frame_start = 0.0
    for frame_index, duration in enumerate(
        CINEMATIC_POC2_FRAME_DURATIONS
    ):
        frame_end = frame_start + duration
        if cycle_elapsed < frame_end:
            local = (
                cycle_elapsed - frame_start
            ) / max(1e-9, duration)
            return frame_index + local
        frame_start = frame_end
    return 0.0


def cubic_bezier(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
    value: float,
) -> tuple[float, float]:
    value = clamp(value)
    inverse = 1.0 - value
    return (
        inverse**3 * a[0]
        + 3.0 * inverse * inverse * value * b[0]
        + 3.0 * inverse * value * value * c[0]
        + value**3 * d[0],
        inverse**3 * a[1]
        + 3.0 * inverse * inverse * value * b[1]
        + 3.0 * inverse * value * value * c[1]
        + value**3 * d[1],
    )


def cubic_bezier_arc_table(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
    steps: int = 96,
) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...], float]:
    points = tuple(cubic_bezier(a, b, c, d, index / steps) for index in range(steps + 1))
    cumulative = [0.0]
    for previous, current in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(previous, current))
    return points, tuple(cumulative), cumulative[-1]


def cubic_bezier_arc_sample(
    table: tuple[tuple[tuple[float, float], ...], tuple[float, ...], float],
    distance_fraction: float,
) -> tuple[float, float]:
    points, cumulative, path_length = table
    if path_length <= 1e-6:
        return points[-1]

    target_distance = clamp(distance_fraction) * path_length
    index = min(len(cumulative) - 1, max(1, bisect_left(cumulative, target_distance)))
    segment_length = max(1e-9, cumulative[index] - cumulative[index - 1])
    segment_progress = (target_distance - cumulative[index - 1]) / segment_length
    return (
        lerp(points[index - 1][0], points[index][0], segment_progress),
        lerp(points[index - 1][1], points[index][1], segment_progress),
    )


def ease_out_cubic(value: float) -> float:
    value = clamp(value)
    return 1 - (1 - value) ** 3


def font(size: int, bold: bool = True) -> pygame.font.Font:
    cache_key = (size, bold)
    cached = _FONT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    local_font = ASSETS / "fonts" / "Oxanium.ttf"
    if local_font.exists():
        selected = pygame.font.Font(local_font, size)
        selected.set_bold(bold)
    else:
        name = pygame.font.match_font("Avenir Next") or pygame.font.match_font("Helvetica Neue") or pygame.font.match_font("Arial")
        selected = pygame.font.Font(name, size) if name else pygame.font.SysFont("arial", size, bold=bold)
    _FONT_CACHE[cache_key] = selected
    return selected


Sound = AudioEngine


@dataclass(frozen=True)
class CinematicAttackEvent:
    minute: int
    side: str
    is_goal: bool
    kind: str = "goal"


@dataclass(frozen=True)
class MatchRuntimeState:
    key: tuple[object, ...]
    goals: tuple[tuple[int, str], ...]
    chances: tuple[tuple[int, str, str], ...]


@dataclass(frozen=True)
class ShotProfile:
    zone: str
    target: tuple[float, float]
    entry: tuple[float, float]
    mouth: tuple[float, float]
    bend: float
    loft: float
    dip: float
    speed: float
    spin: float


@dataclass(frozen=True)
class CinematicShotPlan:
    key: tuple[int, int, str, str]
    profile: ShotProfile
    profile_band: str
    target: tuple[float, float]
    save_variant: str


SHOT_PROFILE_BANDS = {
    "alto firme": "high",
    "angulo seco": "high",
    "meia altura": "mid",
    "central forte": "mid",
    "baixo cruzado": "low",
    "rasteiro forte": "low",
}


@dataclass(frozen=True)
class BallKinematics:
    position: tuple[float, float]
    ground_position: tuple[float, float]
    phase: str
    scale: int
    squash: tuple[float, float]
    rotation_degrees: float
    depth: float


@dataclass(frozen=True)
class DribbleKinematics:
    control_position: tuple[float, float]
    visible_foot_position: tuple[float, float]
    ball: BallKinematics
    touch_phase: float
    contact_gap: float


class AssetFactory:
    def __init__(self, profiles: list[TeamProfile]):
        self.profiles = profiles
        self._cinematic_poc2_runner_cache: dict[
            tuple[str, bool],
            list[pygame.Surface],
        ] = {}
        self._cinematic_runner_cache: dict[tuple[str, bool], list[pygame.Surface]] = {}
        self._cinematic_kick_cache: dict[tuple[str, bool], list[pygame.Surface]] = {}
        self._cinematic_stop_cache: dict[tuple[str, bool], list[pygame.Surface]] = {}
        self._cinematic_keeper_cache: dict[bool, list[pygame.Surface]] = {}
        self.cinematic_poc2_motion = self.load_cinematic_poc2_motion()
        self.cinematic_runner_motion = self.load_cinematic_runner_motion()
        self.cinematic_keeper_motion = self.load_cinematic_keeper_motion()
        self.cinematic_poc2_runners: dict[str, list[pygame.Surface]] = {}
        self.cinematic_poc2_runners_left: dict[str, list[pygame.Surface]] = {}
        self.cinematic_runners: dict[str, list[pygame.Surface]] = {}
        self.cinematic_runners_left: dict[str, list[pygame.Surface]] = {}
        self.cinematic_kicks: dict[str, list[pygame.Surface]] = {}
        self.cinematic_kicks_left: dict[str, list[pygame.Surface]] = {}
        self.cinematic_stops: dict[str, list[pygame.Surface]] = {}
        self.cinematic_stops_left: dict[str, list[pygame.Surface]] = {}
        self.cinematic_keepers: dict[str, pygame.Surface] = {}
        self.cinematic_keeper_frames: dict[str, list[pygame.Surface]] = {}
        self.cinematic_keeper_frames_left: dict[str, list[pygame.Surface]] = {}
        self.flags: dict[str, pygame.Surface] = {}
        self.balls: list[pygame.Surface] = []
        self.field: pygame.Surface | None = None
        self.stadium_bg: pygame.Surface | None = None
        self.turf_mid_strip: pygame.Surface | None = None
        self.turf_near_strip: pygame.Surface | None = None
        self.fifa_images: dict[str, pygame.Surface] = {}
        self.generate_all()

    def load_cinematic_poc2_motion(self) -> dict[str, object]:
        path = ASSETS / "generated" / "cinematic" / "poc2_runner_motion.json"
        if not path.exists():
            raise RuntimeError(f"missing approved POC 2 runner metadata: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_codes = {uniform.code for uniform in CINEMATIC_UNIFORMS}
        uniforms = payload.get("uniforms")
        if (
            payload.get("version") != 1
            or payload.get("artifact")
            != "arena_poc2_dribble_motion_contract"
            or payload.get("status") != "promoted"
            or payload.get("wordmark_provenance")
            != "native_gpt_image_jersey_pixels_no_overlay"
            or payload.get("animation")
            != "approved_poc2_variable_timing_no_morph_no_crossfade"
            or payload.get("frame_count") != 8
            or payload.get("canvas_size") != 320
            or payload.get("canvas_ground_y") != 306
            or payload.get("sheet_columns") != 4
            or payload.get("sheet_rows") != 2
            or not isinstance(uniforms, dict)
            or set(uniforms) != expected_codes
        ):
            raise RuntimeError(f"invalid approved POC 2 runner metadata: {path}")
        for code in expected_codes:
            directions = uniforms[code].get("directions")
            if not isinstance(directions, dict) or set(directions) != {
                "right",
                "left",
            }:
                raise RuntimeError(
                    f"missing {code} POC 2 direction metadata: {path}"
                )
            for direction in ("right", "left"):
                entry = directions[direction]
                if (
                    not isinstance(entry, dict)
                    or len(entry.get("frames", [])) != 8
                    or len(entry.get("contact_offsets_px", [])) != 2
                    or not isinstance(entry.get("sheet"), str)
                    or not isinstance(entry.get("sheet_sha256"), str)
                ):
                    raise RuntimeError(
                        f"incomplete {code} {direction} POC 2 metadata: {path}"
                    )
        return payload

    def load_cinematic_runner_motion(self) -> dict[str, object]:
        path = ASSETS / "generated" / "cinematic" / "runner_motion.json"
        if not path.exists():
            raise RuntimeError(f"missing grounded runner motion metadata: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("version") != 8
            or payload.get("artifact")
            != "arena_runner_motion_contract"
            or payload.get("status") != "promoted"
            or payload.get("artwork_provenance")
            != "gpt_image_authored_per_uniform_per_direction_per_action"
            or payload.get("frame_count")
            != CINEMATIC_RUNNER_FRAME_COUNT
        ):
            raise RuntimeError(f"invalid grounded runner motion metadata: {path}")
        if payload.get("kick_frame_count") != CINEMATIC_KICK_FRAME_COUNT:
            raise RuntimeError(f"invalid grounded runner kick metadata: {path}")
        if payload.get("stop_frame_count") != CINEMATIC_STOP_FRAME_COUNT:
            raise RuntimeError(f"invalid grounded runner stop metadata: {path}")
        if payload.get("temporal_interpolation") != "none_gpt_image_authored_frames":
            raise RuntimeError(f"invalid grounded runner interpolation metadata: {path}")
        if payload.get("kick_temporal_interpolation") != "none_gpt_image_authored_frames":
            raise RuntimeError(f"invalid grounded runner kick interpolation metadata: {path}")
        if payload.get("left_runtime_derivation") != "separately_generated_gpt_image":
            raise RuntimeError(f"invalid grounded runner direction derivation metadata: {path}")
        if payload.get("wordmark_provenance") != "native_gpt_image_jersey_pixels_no_overlay":
            raise RuntimeError(f"invalid grounded runner wordmark provenance: {path}")
        clearance_contract = payload.get("ball_clearance_contract")
        if (
            not isinstance(clearance_contract, dict)
            or clearance_contract.get("field") != "ball_clearance_offset_px"
            or clearance_contract.get("monotonic_field") != "ball_corridor_offset_px"
            or clearance_contract.get("run_approach_field")
            != "ball_approach_corridor_offset_px"
            or clearance_contract.get("covered_actions") != ["run", "kick"]
            or clearance_contract.get("ball_rotation_count")
            != CINEMATIC_BALL_MATERIAL_FRAME_COUNT
            or float(clearance_contract.get("max_alpha_overlap_ratio", -1.0)) != 0.03
            or clearance_contract.get("runtime_rounding_safety_px") != 3
        ):
            raise RuntimeError(f"invalid runner ball-clearance contract: {path}")
        frame_key_indices = payload.get("frame_key_indices")
        if not isinstance(frame_key_indices, list) or len(frame_key_indices) != CINEMATIC_RUNNER_FRAME_COUNT:
            raise RuntimeError(f"invalid grounded runner gait timeline: {path}")
        uniforms = payload.get("uniforms")
        expected_codes = {uniform.code for uniform in CINEMATIC_UNIFORMS}
        if not isinstance(uniforms, dict) or set(uniforms) != expected_codes:
            raise RuntimeError(f"missing per-avatar runner metadata: {path}")
        for code in expected_codes:
            directions = uniforms[code].get("directions")
            if not isinstance(directions, dict):
                raise RuntimeError(f"missing {code} runner direction metadata: {path}")
            for direction in ("right", "left"):
                entry = directions.get(direction)
                if (
                    not isinstance(entry, dict)
                    or len(entry.get("frames", [])) != CINEMATIC_RUNNER_FRAME_COUNT
                    or len(entry.get("kick_frames", [])) != CINEMATIC_KICK_FRAME_COUNT
                    or len(entry.get("stop_frames", [])) != CINEMATIC_STOP_FRAME_COUNT
                    or not 0 <= int(entry.get("kick_contact_frame", -1)) < CINEMATIC_KICK_FRAME_COUNT
                    or not isinstance(entry.get("kick_contact_gap_px"), (int, float))
                    or not isinstance(
                        entry.get("kick_contact_root_correction_px"),
                        (int, float),
                    )
                    or not isinstance(
                        entry.get("kick_entry_ball_corridor_offset_px"),
                        (int, float),
                    )
                ):
                    raise RuntimeError(f"incomplete {code} {direction} runner metadata: {path}")
                clearance_frames = [
                    *entry["frames"],
                    *entry["kick_frames"],
                ]
                if any(
                    not isinstance(frame.get("ball_clearance_offset_px"), (int, float))
                    or float(frame["ball_clearance_offset_px"]) <= 0.0
                    for frame in clearance_frames
                ):
                    raise RuntimeError(
                        f"incomplete {code} {direction} ball-clearance metadata: {path}"
                    )
                if any(
                    not isinstance(
                        frame.get("ball_approach_corridor_offset_px"),
                        (int, float),
                    )
                    or float(frame["ball_approach_corridor_offset_px"]) + 1e-6
                    < float(frame["ball_clearance_offset_px"])
                    for frame in entry["frames"]
                ):
                    raise RuntimeError(
                        f"incomplete {code} {direction} run-approach corridor: {path}"
                    )
                previous_corridor = 0.0
                for frame in entry["kick_frames"]:
                    corridor = frame.get("ball_corridor_offset_px")
                    if (
                        not isinstance(corridor, (int, float))
                        or float(corridor) + 1e-6
                        < float(frame["ball_clearance_offset_px"])
                        or float(corridor) + 1e-6 < previous_corridor
                    ):
                        raise RuntimeError(
                            f"invalid {code} {direction} monotonic ball corridor: {path}"
                        )
                    previous_corridor = float(corridor)
        return payload

    def load_cinematic_keeper_motion(self) -> dict[str, object]:
        path = ASSETS / "generated" / "cinematic" / "keeper_motion.json"
        if not path.exists():
            raise RuntimeError(f"missing goalkeeper motion metadata: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime_frame_order = payload.get("runtime_frame_order")
        directions = payload.get("directions")
        if (
            payload.get("version") != 4
            or payload.get("artifact")
            != "arena_keeper_motion_contract"
            or payload.get("status") != "promoted"
            or payload.get("artwork_provenance")
            != "gpt_image_authored_directional_frames"
            or payload.get("authored_frame_count") != 16
            or payload.get("frame_count") != CINEMATIC_KEEPER_FRAME_COUNT
            or not isinstance(runtime_frame_order, list)
            or len(runtime_frame_order) != 16
            or runtime_frame_order
            != list(range(CINEMATIC_KEEPER_FRAME_COUNT))
            or payload.get("temporal_interpolation") != "none_gpt_image_authored_frames"
            or payload.get("left_runtime_derivation") != "separately_generated_gpt_image"
            or not isinstance(directions, dict)
            or set(directions) != {"right", "left"}
        ):
            raise RuntimeError(f"invalid goalkeeper motion metadata: {path}")
        return payload

    def generate_all(self) -> None:
        flags_dir = ASSETS / "generated" / "flags"
        for index, profile in enumerate(self.profiles):
            flag_path = flags_dir / f"{profile.code.lower()}.png"
            if not flag_path.exists():
                raise RuntimeError(f"missing generated image_gen flag sprite: {flag_path}")
            self.flags[profile.code] = pygame.image.load(flag_path).convert_alpha()
            poc2_runner_frames = self.load_cinematic_poc2_runner_frames(
                profile
            )
            poc2_runner_left_frames = (
                self.load_cinematic_poc2_runner_frames(
                    profile,
                    left=True,
                )
            )
            runner_frames = self.load_cinematic_runner_frames(profile)
            runner_left_frames = self.load_cinematic_runner_frames(profile, left=True)
            kick_frames = self.load_cinematic_kick_frames(profile)
            kick_left_frames = self.load_cinematic_kick_frames(profile, left=True)
            stop_frames = self.load_cinematic_stop_frames(profile)
            stop_left_frames = self.load_cinematic_stop_frames(profile, left=True)
            keeper_frames = self.load_cinematic_keeper_frames(profile)
            keeper_left_frames = self.load_cinematic_keeper_frames(profile, left=True)
            if not poc2_runner_frames or not poc2_runner_left_frames:
                raise RuntimeError(
                    f"missing generated POC 2 runner sprites for {profile.code}"
                )
            if not runner_frames:
                raise RuntimeError(f"missing generated cinematic runner sprites for {profile.code}")
            if not runner_left_frames:
                raise RuntimeError(f"missing generated left-facing cinematic runner sprites for {profile.code}")
            if not kick_frames or not kick_left_frames:
                raise RuntimeError(f"missing generated cinematic kick sprites for {profile.code}")
            if not stop_frames or not stop_left_frames:
                raise RuntimeError(f"missing generated cinematic stop sprites for {profile.code}")
            if not keeper_frames or not keeper_left_frames:
                raise RuntimeError(f"missing generated cinematic goalkeeper animation for {profile.code}")
            self.cinematic_poc2_runners[profile.code] = (
                poc2_runner_frames
            )
            self.cinematic_poc2_runners_left[profile.code] = (
                poc2_runner_left_frames
            )
            self.cinematic_runners[profile.code] = runner_frames
            self.cinematic_runners_left[profile.code] = runner_left_frames
            self.cinematic_kicks[profile.code] = kick_frames
            self.cinematic_kicks_left[profile.code] = kick_left_frames
            self.cinematic_stops[profile.code] = stop_frames
            self.cinematic_stops_left[profile.code] = stop_left_frames
            self.cinematic_keeper_frames[profile.code] = keeper_frames
            self.cinematic_keeper_frames_left[profile.code] = keeper_left_frames
            self.cinematic_keepers[profile.code] = keeper_frames[0]
        self.balls = self.load_ball_frames()
        if not self.balls:
            raise RuntimeError("missing generated ball sprites")
        self.stadium_bg = self.load_stadium_background((1520, 472))
        if not self.stadium_bg:
            raise RuntimeError("missing generated stadium parallax background")
        self.turf_mid_strip = self.load_turf_strip("turf_mid_strip.png")
        self.turf_near_strip = self.load_turf_strip("turf_near_strip.png")
        if not self.turf_mid_strip or not self.turf_near_strip:
            raise RuntimeError("missing generated parallax turf strips")
        self.fifa_images = self.load_fifa_external_images()

    def load_fifa_external_images(self) -> dict[str, pygame.Surface]:
        base = ASSETS / "generated" / "fifa_external"
        images: dict[str, pygame.Surface] = {}
        for key, filename in FIFA_EXTERNAL_IMAGES.items():
            path = base / filename
            if path.exists():
                images[key] = pygame.image.load(path).convert_alpha()
        return images

    def load_stadium_background(self, size: tuple[int, int]) -> pygame.Surface | None:
        path = ASSETS / "generated" / "stadium_parallax_real.png"
        if not path.exists():
            return None
        image = pygame.image.load(path).convert_alpha()
        scale = max(size[0] / image.get_width(), size[1] / image.get_height())
        scaled = pygame.transform.smoothscale(image, (int(image.get_width() * scale), int(image.get_height() * scale)))
        x = max(0, (scaled.get_width() - size[0]) // 2)
        y = max(0, (scaled.get_height() - size[1]) // 2)
        result = pygame.Surface(size, pygame.SRCALPHA)
        result.blit(scaled, (0, 0), (x, y, size[0], size[1]))
        return result

    def cinematic_source_code(self, profile: TeamProfile) -> str:
        if profile.code in TEAM_UNIFORM_OVERRIDES:
            return TEAM_UNIFORM_OVERRIDES[profile.code]
        palette = {uniform.code: uniform.primary for uniform in CINEMATIC_UNIFORMS}
        kit = profile.kit
        return min(
            palette,
            key=lambda name: sum((int(kit[i]) - palette[name][i]) ** 2 for i in range(3)),
        )

    def cinematic_runner_frames_for_uniform(
        self,
        uniform_code: str,
        *,
        left: bool,
    ) -> list[pygame.Surface]:
        return self._cinematic_runner_cache[(uniform_code, left)]

    def cinematic_poc2_frames_for_uniform(
        self,
        uniform_code: str,
        *,
        left: bool,
    ) -> list[pygame.Surface]:
        return self._cinematic_poc2_runner_cache[
            (uniform_code, left)
        ]

    def load_cinematic_poc2_runner_frames(
        self,
        profile: TeamProfile,
        left: bool = False,
    ) -> list[pygame.Surface] | None:
        code = self.cinematic_source_code(profile)
        cache_key = (code, left)
        if cache_key in self._cinematic_poc2_runner_cache:
            return self._cinematic_poc2_runner_cache[cache_key]
        direction = "left" if left else "right"
        uniforms = self.cinematic_poc2_motion["uniforms"]
        entry = uniforms[code]["directions"][direction]  # type: ignore[index]
        path = (
            ASSETS
            / "generated"
            / "cinematic"
            / str(entry["sheet"])
        )
        if not path.exists():
            return None
        expected_sha = str(entry["sheet_sha256"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise RuntimeError(f"stale POC 2 runner sheet: {path}")
        sheet = pygame.image.load(path).convert_alpha()
        frame_size = int(self.cinematic_poc2_motion["canvas_size"])
        columns = int(self.cinematic_poc2_motion["sheet_columns"])
        rows = int(self.cinematic_poc2_motion["sheet_rows"])
        frame_count = int(self.cinematic_poc2_motion["frame_count"])
        if sheet.get_size() != (
            frame_size * columns,
            frame_size * rows,
        ):
            raise RuntimeError(
                f"invalid POC 2 runner sheet: {path}={sheet.get_size()}"
            )
        frames = [
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
        frame_metadata = entry["frames"]
        for index, frame in enumerate(frames):
            expected_frame_sha = str(frame_metadata[index]["sha256"])
            actual_frame_sha = hashlib.sha256(
                pygame.image.tostring(frame, "RGBA")
            ).hexdigest()
            if actual_frame_sha != expected_frame_sha:
                raise RuntimeError(
                    "stale POC 2 runner frame: "
                    f"{path} frame={index}"
                )
        self._cinematic_poc2_runner_cache[cache_key] = frames
        return frames

    def cinematic_kick_frames_for_uniform(
        self,
        uniform_code: str,
        *,
        left: bool,
    ) -> list[pygame.Surface]:
        return self._cinematic_kick_cache[(uniform_code, left)]

    def load_cinematic_runner_frames(self, profile: TeamProfile, left: bool = False) -> list[pygame.Surface] | None:
        code = self.cinematic_source_code(profile)
        cache_key = (code, left)
        if cache_key in self._cinematic_runner_cache:
            return self._cinematic_runner_cache[cache_key]
        base = ASSETS / "generated" / "cinematic"
        stem = "runner_left" if left else "runner"
        smooth_path = base / f"{stem}_smooth_{code}.png"
        if smooth_path.exists():
            sheet = pygame.image.load(smooth_path).convert_alpha()
            frame_size = int(self.cinematic_runner_motion["frame_size"])
            columns = int(self.cinematic_runner_motion["sheet_columns"])
            rows = int(self.cinematic_runner_motion["sheet_rows"])
            if columns * rows < CINEMATIC_RUNNER_FRAME_COUNT:
                raise RuntimeError("smooth runner sheet layout cannot hold every frame")
            if sheet.get_size() != (frame_size * columns, frame_size * rows):
                raise RuntimeError(f"invalid smooth runner sheet: {smooth_path}={sheet.get_size()}")
            frames = [
                sheet.subsurface(
                    pygame.Rect(
                        (index % columns) * frame_size,
                        (index // columns) * frame_size,
                        frame_size,
                        frame_size,
                    )
                ).copy()
                for index in range(CINEMATIC_RUNNER_FRAME_COUNT)
            ]
            self._cinematic_runner_cache[cache_key] = frames
            return frames
        return None

    def load_cinematic_kick_frames(self, profile: TeamProfile, left: bool = False) -> list[pygame.Surface] | None:
        code = self.cinematic_source_code(profile)
        cache_key = (code, left)
        if cache_key in self._cinematic_kick_cache:
            return self._cinematic_kick_cache[cache_key]
        base = ASSETS / "generated" / "cinematic"
        stem = "runner_left_kick_smooth" if left else "runner_kick_smooth"
        path = base / f"{stem}_{code}.png"
        if not path.exists():
            return None
        sheet = pygame.image.load(path).convert_alpha()
        frame_size = int(self.cinematic_runner_motion["frame_size"])
        columns = int(self.cinematic_runner_motion["kick_sheet_columns"])
        rows = int(self.cinematic_runner_motion["kick_sheet_rows"])
        if columns * rows < CINEMATIC_KICK_FRAME_COUNT:
            raise RuntimeError("smooth kick sheet layout cannot hold every frame")
        if sheet.get_size() != (frame_size * columns, frame_size * rows):
            raise RuntimeError(f"invalid smooth kick sheet: {path}={sheet.get_size()}")
        frames = [
            sheet.subsurface(
                pygame.Rect(
                    (index % columns) * frame_size,
                    (index // columns) * frame_size,
                    frame_size,
                    frame_size,
                )
            ).copy()
            for index in range(CINEMATIC_KICK_FRAME_COUNT)
        ]
        self._cinematic_kick_cache[cache_key] = frames
        return frames

    def load_cinematic_stop_frames(self, profile: TeamProfile, left: bool = False) -> list[pygame.Surface] | None:
        code = self.cinematic_source_code(profile)
        cache_key = (code, left)
        if cache_key in self._cinematic_stop_cache:
            return self._cinematic_stop_cache[cache_key]
        base = ASSETS / "generated" / "cinematic"
        stem = "runner_left_stop_smooth" if left else "runner_stop_smooth"
        path = base / f"{stem}_{code}.png"
        if not path.exists():
            return None
        sheet = pygame.image.load(path).convert_alpha()
        frame_size = int(self.cinematic_runner_motion["frame_size"])
        columns = int(self.cinematic_runner_motion["stop_sheet_columns"])
        rows = int(self.cinematic_runner_motion["stop_sheet_rows"])
        if columns * rows < CINEMATIC_STOP_FRAME_COUNT:
            raise RuntimeError("smooth stop sheet layout cannot hold every frame")
        if sheet.get_size() != (frame_size * columns, frame_size * rows):
            raise RuntimeError(f"invalid smooth stop sheet: {path}={sheet.get_size()}")
        frames = [
            sheet.subsurface(
                pygame.Rect(
                    (index % columns) * frame_size,
                    (index // columns) * frame_size,
                    frame_size,
                    frame_size,
                )
            ).copy()
            for index in range(CINEMATIC_STOP_FRAME_COUNT)
        ]
        self._cinematic_stop_cache[cache_key] = frames
        return frames

    def load_cinematic_keeper_frames(self, profile: TeamProfile, left: bool = False) -> list[pygame.Surface] | None:
        if left in self._cinematic_keeper_cache:
            return self._cinematic_keeper_cache[left]
        base = ASSETS / "generated" / "cinematic"
        direction = "left" if left else "right"
        paths = [base / f"keeper_anim_{direction}_{index}.png" for index in range(CINEMATIC_KEEPER_FRAME_COUNT)]
        if all(path.exists() for path in paths):
            self._cinematic_keeper_cache[left] = [pygame.image.load(path).convert_alpha() for path in paths]
            return self._cinematic_keeper_cache[left]
        return None

    def load_turf_strip(self, filename: str) -> pygame.Surface | None:
        path = ASSETS / "generated" / "parallax" / filename
        if path.exists():
            return pygame.image.load(path).convert_alpha()
        return None

    def load_ball_frames(self) -> list[pygame.Surface] | None:
        base = ASSETS / "generated" / "balls3d"
        paths = [
            base / f"ball_{i}.png"
            for i in range(CINEMATIC_BALL_MATERIAL_FRAME_COUNT)
        ]
        if not all(path.exists() for path in paths):
            return None
        return [pygame.image.load(path).convert_alpha() for path in paths]

    def generate_flag(self, team: TeamProfile, size: tuple[int, int]) -> pygame.Surface:
        surf = pygame.Surface(size, pygame.SRCALPHA)
        w, h = size
        code = team.code

        def horizontal(colors: tuple[tuple[int, int, int], ...]) -> None:
            for i, color in enumerate(colors):
                pygame.draw.rect(surf, color, (0, i * h // len(colors), w, h // len(colors) + 1))

        def vertical(colors: tuple[tuple[int, int, int], ...]) -> None:
            for i, color in enumerate(colors):
                pygame.draw.rect(surf, color, (i * w // len(colors), 0, w // len(colors) + 1, h))

        if code in {"FRA", "ITA", "BEL", "IRL", "MEX", "PRT", "POR", "SEN", "CMR", "NGA"}:
            vertical(team.flag)
        elif code == "BRA":
            surf.fill(team.flag[0])
            pygame.draw.polygon(surf, team.flag[1], [(w // 2, 10), (w - 18, h // 2), (w // 2, h - 10), (18, h // 2)])
            pygame.draw.circle(surf, team.flag[2], (w // 2, h // 2), h // 5)
        elif code == "ENG":
            surf.fill(team.flag[0])
            pygame.draw.rect(surf, team.flag[1], (w // 2 - 8, 0, 16, h))
            pygame.draw.rect(surf, team.flag[1], (0, h // 2 - 8, w, 16))
        elif code in {"DNK", "DEN", "SWE"}:
            surf.fill(team.flag[0])
            cross = team.flag[1]
            pygame.draw.rect(surf, cross, (w // 3 - 7, 0, 14, h))
            pygame.draw.rect(surf, cross, (0, h // 2 - 7, w, 14))
        elif code == "JPN":
            surf.fill(team.flag[0])
            pygame.draw.circle(surf, team.flag[1], (w // 2, h // 2), h // 4)
        elif code == "KOR":
            surf.fill(team.flag[0])
            pygame.draw.circle(surf, team.flag[1], (w // 2, h // 2 - 5), h // 5)
            pygame.draw.circle(surf, team.flag[2], (w // 2, h // 2 + 5), h // 5)
            pygame.draw.rect(surf, BLACK, (24, 22, 34, 5))
            pygame.draw.rect(surf, BLACK, (w - 58, h - 27, 34, 5))
        elif code == "USA":
            for i in range(7):
                color = team.flag[0] if i % 2 == 0 else team.flag[1]
                pygame.draw.rect(surf, color, (0, i * h // 7, w, h // 7 + 1))
            pygame.draw.rect(surf, team.flag[2], (0, 0, 72, 58))
        elif code in {"URY", "URU"}:
            for i in range(7):
                pygame.draw.rect(surf, team.flag[0] if i % 2 == 0 else team.flag[1], (0, i * h // 7, w, h // 7 + 1))
            pygame.draw.rect(surf, team.flag[0], (0, 0, 62, 54))
            pygame.draw.circle(surf, team.flag[2], (31, 27), 13)
        elif code == "TUR":
            surf.fill(team.flag[0])
            pygame.draw.circle(surf, team.flag[1], (w // 2 - 16, h // 2), 24)
            pygame.draw.circle(surf, team.flag[0], (w // 2 - 7, h // 2), 19)
            pygame.draw.polygon(surf, team.flag[1], [(w // 2 + 31, h // 2 - 13), (w // 2 + 37, h // 2 + 8), (w // 2 + 18, h // 2 - 4), (w // 2 + 44, h // 2 - 4), (w // 2 + 25, h // 2 + 8)])
        elif code == "CHN":
            surf.fill(team.flag[0])
            pygame.draw.circle(surf, team.flag[1], (40, 32), 12)
            for dx, dy in [(70, 18), (84, 36), (82, 58), (64, 72)]:
                pygame.draw.circle(surf, team.flag[1], (dx, dy), 4)
        elif code == "CRI":
            bands = [team.flag[0], team.flag[1], team.flag[2], team.flag[1], team.flag[0]]
            heights = [18, 18, 36, 18, 18]
            y = 0
            for color, band_h in zip(bands, heights):
                pygame.draw.rect(surf, color, (0, y, w, band_h))
                y += band_h
        elif code in {"ZAF", "RSA"}:
            surf.fill(team.flag[0])
            pygame.draw.polygon(surf, team.flag[1], [(0, 0), (w // 2, h // 2), (0, h)])
            pygame.draw.polygon(surf, BLACK, [(0, 14), (w // 3, h // 2), (0, h - 14)])
        elif code == "TUN":
            surf.fill(team.flag[0])
            pygame.draw.circle(surf, team.flag[1], (w // 2, h // 2), 25)
            pygame.draw.circle(surf, team.flag[0], (w // 2 - 6, h // 2), 12)
            pygame.draw.circle(surf, team.flag[1], (w // 2, h // 2), 9)
        else:
            horizontal(team.flag)
        pygame.draw.rect(surf, WHITE, surf.get_rect(), 3, border_radius=12)
        return surf

    def generate_player(self, team: TeamProfile, frame: int) -> pygame.Surface:
        surf = pygame.Surface((76, 76), pygame.SRCALPHA)
        cx, cy = 38, 38
        bob = math.sin(frame / 8 * math.tau) * 2
        leg = math.sin(frame / 8 * math.tau) * 7
        kit = team.kit
        outline = (2, 8, 12)
        pygame.draw.ellipse(surf, (0, 0, 0, 95), (18, 56, 40, 12))
        pygame.draw.line(surf, outline, (cx - 9, cy + 10), (cx - 17, cy + 25 + leg * 0.25), 9)
        pygame.draw.line(surf, outline, (cx + 9, cy + 10), (cx + 17, cy + 25 - leg * 0.25), 9)
        pygame.draw.line(surf, kit, (cx - 9, cy + 10), (cx - 17, cy + 25 + leg * 0.25), 5)
        pygame.draw.line(surf, kit, (cx + 9, cy + 10), (cx + 17, cy + 25 - leg * 0.25), 5)
        pygame.draw.line(surf, outline, (cx - 17, cy - 5), (cx - 27, cy + 10 - leg * 0.16), 8)
        pygame.draw.line(surf, outline, (cx + 17, cy - 5), (cx + 27, cy + 10 + leg * 0.16), 8)
        pygame.draw.line(surf, kit, (cx - 17, cy - 5), (cx - 27, cy + 10 - leg * 0.16), 4)
        pygame.draw.line(surf, kit, (cx + 17, cy - 5), (cx + 27, cy + 10 + leg * 0.16), 4)
        pygame.draw.polygon(surf, outline, [(cx, cy - 18 + bob), (cx - 22, cy - 2), (cx - 13, cy + 18), (cx + 13, cy + 18), (cx + 22, cy - 2)])
        pygame.draw.polygon(surf, kit, [(cx, cy - 14 + bob), (cx - 17, cy), (cx - 10, cy + 14), (cx + 10, cy + 14), (cx + 17, cy)])
        pygame.draw.circle(surf, outline, (cx, int(cy - 25 + bob)), 13)
        pygame.draw.circle(surf, (232, 185, 141), (cx, int(cy - 25 + bob)), 10)
        pygame.draw.arc(surf, (45, 35, 28), (cx - 10, cy - 35 + bob, 20, 16), math.pi, math.tau, 4)
        pygame.draw.rect(surf, WHITE if sum(kit) < 360 else BLACK, (cx - 8, cy - 3, 16, 4), border_radius=2)
        return surf

    def generate_ball(self, frame: int) -> pygame.Surface:
        surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        angle = frame / 12 * math.tau
        pygame.draw.circle(surf, (0, 0, 0, 80), (22, 24), 15)
        pygame.draw.circle(surf, WHITE, (20, 20), 15)
        pygame.draw.circle(surf, BLACK, (20, 20), 15, 2)
        for i in range(5):
            a = angle + i * math.tau / 5
            x = 20 + math.cos(a) * 8
            y = 20 + math.sin(a) * 8
            pygame.draw.line(surf, BLACK, (20, 20), (x, y), 2)
            pygame.draw.circle(surf, BLACK, (int(x), int(y)), 2)
        return surf

    def generate_field(self, size: tuple[int, int]) -> pygame.Surface:
        surf = pygame.Surface(size, pygame.SRCALPHA)
        w, h = size
        for i in range(12):
            color = (20, 110, 58) if i % 2 == 0 else (15, 86, 47)
            pygame.draw.rect(surf, color, (i * w // 12, 0, w // 12 + 1, h))
        crowd_h = 72
        pygame.draw.rect(surf, (19, 28, 34), (0, 0, w, crowd_h))
        for row in range(5):
            for col in range(45):
                px = 18 + col * 20 + (row % 2) * 8
                py = 10 + row * 12
                shade = 85 + ((row * 23 + col * 17) % 80)
                pygame.draw.circle(surf, (shade, shade + 12, shade + 22), (px, py), 3)
        pitch = pygame.Rect(48, 88, w - 96, h - 116)
        pygame.draw.rect(surf, LINE, pitch, 2)
        pygame.draw.line(surf, LINE, (pitch.centerx, pitch.y), (pitch.centerx, pitch.bottom), 2)
        pygame.draw.circle(surf, LINE, pitch.center, 58, 2)
        pygame.draw.circle(surf, LINE, pitch.center, 4)
        pygame.draw.rect(surf, LINE, (pitch.x, pitch.centery - 75, 110, 150), 2)
        pygame.draw.rect(surf, LINE, (pitch.right - 110, pitch.centery - 75, 110, 150), 2)
        pygame.draw.rect(surf, LINE, (pitch.x, pitch.centery - 35, 45, 70), 2)
        pygame.draw.rect(surf, LINE, (pitch.right - 45, pitch.centery - 35, 45, 70), 2)
        pygame.draw.rect(surf, WHITE, (pitch.x - 12, pitch.centery - 38, 12, 76), 2)
        pygame.draw.rect(surf, WHITE, (pitch.right, pitch.centery - 38, 12, 76), 2)
        return surf


class App:
    def __init__(self, seed: int | None = None):
        pre_init_mixer()
        pygame.init()
        pygame.display.set_caption("Oráculo da Copa")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.surface_cache = SurfaceCache()
        self.text_cache = TextCache()
        self.app_icon = self.load_image("generated/app_icon_worldcup.png")
        self.menu_icon = self.cached_smoothscale(self.app_icon, (112, 112)) if self.app_icon else None
        self.top_icon = self.cached_smoothscale(self.app_icon, (46, 46)) if self.app_icon else None
        self.trophy_icon = self.cached_smoothscale(self.app_icon, (132, 132)) if self.app_icon else None
        if self.app_icon:
            pygame.display.set_icon(self.cached_smoothscale(self.app_icon, (128, 128)))
        self.clock = pygame.time.Clock()
        self.rng = random.Random(seed)
        self.tournament_rng = random.Random(seed) if seed is not None else random.SystemRandom()
        self.sound = Sound(seed)
        self.model = WorldCupModel()
        self.teams = self.model.profiles()
        self.assets = AssetFactory(self.teams)
        self.poc2_dribble = Poc2DribbleRuntime.load(
            ASSETS
            / "generated"
            / "cinematic"
            / "poc2_runner_motion.json"
        )
        self.poc_sequences = PocSequenceBank(
            ASSETS
            / "generated"
            / "cinematic"
            / "poc7_runtime_contract.json"
        )
        self.poc_layer_cache: dict[str, pygame.Surface] = {}
        self.poc_layer_frame_cache: dict[
            tuple[str, tuple[int, int, int, int]],
            pygame.Surface,
        ] = {}
        self.poc_goal_composite_cache: OrderedDict[
            tuple[object, ...],
            pygame.Surface,
        ] = OrderedDict()
        self.poc_preload_generation = 0
        self.poc_preload_cancel_event = threading.Event()
        self.poc_preload_queue: queue.Queue[
            tuple[object, ...]
        ] = queue.Queue(maxsize=2)
        self.poc_preload_thread: threading.Thread | None = None
        self.poc_preload_threads: list[threading.Thread] = []
        self.poc_preload_pending = 0
        self.poc_preload_completed = 0
        self.poc_preload_ready = True
        self.poc_preload_error = ""
        self.turf_tile_cache: dict[tuple[int, int, int, int], pygame.Surface] = {}
        self.gradient_mask_cache: dict[tuple[int, int, int, int], pygame.Surface] = {}
        self.gradient_tile_cache: dict[tuple[int, int, int, int, int], pygame.Surface] = {}
        self.surface_bbox_cache: OrderedDict[
            pygame.Surface,
            pygame.Rect,
        ] = OrderedDict()
        self.surface_alpha_centroid_cache: OrderedDict[
            pygame.Surface,
            tuple[float, float],
        ] = OrderedDict()
        self.surface_toe_anchor_cache: dict[tuple[int, int], tuple[float, float]] = {}
        self.keeper_glove_offset_cache: dict[tuple[bool, str], tuple[float, float]] = {}
        self.runner_reference_height_cache: dict[tuple[str, str], float] = {}
        self.cinematic_ball_corridor_cache: dict[
            tuple[object, ...],
            tuple[tuple[float, ...], float],
        ] = {}
        self.ball_net_path_cache: dict[
            tuple[float, ...],
            tuple[tuple[tuple[float, float], ...], tuple[float, ...], float],
        ] = {}
        self.cinematic_ball_history_cache: dict[
            tuple[object, ...],
            tuple[CinematicAttackEvent | None, tuple[float, float]],
        ] = {}
        self.scaled_surface_cache = self.surface_cache.scaled
        self.flipped_surface_cache = self.surface_cache.flipped
        self.roto_surface_cache = self.surface_cache.roto
        self.cinematic_overlay_cache: dict[tuple[object, ...], pygame.Surface] = {}
        self.prepare_turf_tile_cache()
        self.title_bg = self.load_image("generated/title_stadium_ai.png", (WIDTH, HEIGHT))
        self.state = "menu"
        self.home_idx = self.team_index("BRA", 0)
        self.away_idx = self.team_index("FRA", 1)
        self.mode = "single"
        self.t = 0.0
        self.ground_scroll = 0.0
        self.ground_scroll_velocity = 0.0
        self.ground_travel_distance = 0.0
        self.cinematic_render_dt = 1.0 / FPS
        self.cinematic_attack_phase_anchors: dict[tuple[int, str, bool, str], float] = {}
        self.segment_started = -1
        self.goal_events: set[tuple[str, int, str]] = set()
        self.shot_events: set[tuple[str, int, str, str]] = set()
        self.shot_progress_cursor: dict[tuple[str, int, str], float] = {}
        self.match_audio_frame_queue: list[tuple[str, float]] = []
        self.final_whistle_played = False
        self.match_intro_audio_pending = False
        self.cup_audio_markers: set[int] = set()
        self.cup_start_audio_pending = False
        self.cup_reveal_audio_pending = False
        self.cup_reveal_audio_played = False
        self.match_seed = self.rng.randint(1, 999999)
        self.match_prediction: Prediction | None = None
        self.match_analysis: MatchAnalysis | None = None
        self.match_runtime_state_cache: dict[tuple[object, ...], MatchRuntimeState] = {}
        self.cinematic_shot_plan_cache: dict[
            tuple[int, int, str, str],
            CinematicShotPlan,
        ] = {}
        self.tournament_result: dict[str, object] | None = None
        self.champion_odds: list[tuple[str, int, float]] = []
        self.champion_odds_runs = TOURNAMENT_MONTE_CARLO_RUNS
        self.champion_odds_workers = TOURNAMENT_MONTE_CARLO_WORKERS
        self.mc_progress_done = 0
        self.mc_progress_total = TOURNAMENT_MONTE_CARLO_RUNS
        self.mc_running = False
        self.mc_error = ""
        self.mc_generation = 0
        self.mc_thread: threading.Thread | None = None
        self.mc_cancel_event = threading.Event()
        self.mc_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
        self.mc_seed = 2026
        self.pending_tournament_seed: int | None = None
        self.mc_started_t = 0.0
        self.mc_pending_result: tuple[list[tuple[str, int, float]], dict[str, object] | None] | None = None
        self.tournament_view = "groups"
        self.tournament_reveal_t = 0.0
        self.mouse = (0, 0)
        self.f_tiny = font(13)
        self.f_xs = font(16)
        self.f_sm = font(20)
        self.f_body = font(22)
        self.f_md = font(27)
        self.f_lg = font(42)
        self.f_xl = font(68)
        self.start_button = Button(pygame.Rect(86, 608, 310, 64), "COMEÇAR", CYAN)
        self.single_button = Button(pygame.Rect(292, 628, 290, 58), "SIMULAR CONFRONTO", BLUE)
        self.cup_button = Button(pygame.Rect(698, 628, 250, 58), "SIMULAR COPA", GREEN)
        self.back_button = Button(pygame.Rect(32, 28, 138, 44), "VOLTAR", CYAN)
        self.group_tab_rect = pygame.Rect(772, 30, 178, 38)
        self.bracket_tab_rect = pygame.Rect(960, 30, 230, 38)

    @property
    def home(self) -> TeamProfile:
        return self.teams[self.home_idx]

    @property
    def away(self) -> TeamProfile:
        return self.teams[self.away_idx]

    def team_index(self, code: str, fallback: int) -> int:
        return next((index for index, team in enumerate(self.teams) if team.code == code), fallback)

    def load_image(self, rel: str, size: tuple[int, int] | None = None) -> pygame.Surface | None:
        path = ASSETS / rel
        if not path.exists():
            return None
        image = pygame.image.load(path).convert_alpha()
        return self.cached_smoothscale(image, size) if size else image

    def cached_smoothscale(self, image: pygame.Surface, size: tuple[int, int]) -> pygame.Surface:
        return self.surface_cache.smoothscale(image, size)

    def cached_flip(self, image: pygame.Surface) -> pygame.Surface:
        return self.surface_cache.flip(image)

    def cached_rotozoom(self, image: pygame.Surface, angle: float, scale: float) -> pygame.Surface:
        return self.surface_cache.rotozoom(image, angle, scale)

    def cached_alpha(self, image: pygame.Surface, alpha: int, step: int = 4) -> pygame.Surface:
        return self.surface_cache.with_alpha(image, alpha, step)

    def cached_filled_overlay(self, cache_key: tuple[object, ...], size: tuple[int, int], color: tuple[int, int, int, int]) -> pygame.Surface:
        key = (*cache_key, int(size[0]), int(size[1]), color)
        surface = self.cinematic_overlay_cache.get(key)
        if surface is None:
            surface = pygame.Surface(size, pygame.SRCALPHA)
            surface.fill(color)
            self.cinematic_overlay_cache[key] = surface
        return surface

    def visible_bbox(self, image: pygame.Surface) -> pygame.Rect:
        cached = self.surface_bbox_cache.get(image)
        if cached is None:
            cached = image.get_bounding_rect()
            self.surface_bbox_cache[image] = cached
            while len(self.surface_bbox_cache) > 360:
                self.surface_bbox_cache.popitem(last=False)
        else:
            self.surface_bbox_cache.move_to_end(image)
        return cached.copy()

    def visible_alpha_centroid(self, image: pygame.Surface) -> tuple[float, float]:
        cached = self.surface_alpha_centroid_cache.get(image)
        if cached is None:
            alpha = pygame.surfarray.array_alpha(image)
            xs, ys = np.nonzero(alpha > 48)
            if len(xs):
                cached = float(xs.mean()), float(ys.mean())
            else:
                cached = (
                    image.get_width() * 0.5,
                    image.get_height() * 0.5,
                )
            self.surface_alpha_centroid_cache[image] = cached
            while len(self.surface_alpha_centroid_cache) > 360:
                self.surface_alpha_centroid_cache.popitem(last=False)
        else:
            self.surface_alpha_centroid_cache.move_to_end(image)
        return cached

    def prepare_turf_tile_cache(self) -> None:
        ground_size = (910, 220)
        for image, alpha in (
            (self.assets.turf_mid_strip, 212),
            (self.assets.turf_near_strip, 86),
        ):
            if image is None:
                continue
            width, height = ground_size
            scaled_h = max(1, height)
            scaled_w = max(int(width * 1.32), int(image.get_width() * scaled_h / max(1, image.get_height())))
            tile = pygame.transform.smoothscale(image, (scaled_w, scaled_h)).convert_alpha()
            tile.set_alpha(alpha)
            self.turf_tile_cache[(id(image), scaled_w, scaled_h, alpha)] = tile
        if self.assets.turf_near_strip:
            self.cached_gradient_turf_tile(self.assets.turf_near_strip, ground_size, 104, 1.65)

    def draw_text(self, text: str, text_font: pygame.font.Font, color: tuple[int, int, int], x: int, y: int) -> None:
        self.screen.blit(self.text_cache.render(text_font, text, color), (x, y))

    def draw_text_centered(self, text: str, text_font: pygame.font.Font, color: tuple[int, int, int], center: tuple[int, int]) -> None:
        rendered = self.text_cache.render(text_font, text, color)
        self.screen.blit(rendered, rendered.get_rect(center=center))

    def draw_text_right(self, text: str, text_font: pygame.font.Font, color: tuple[int, int, int], right: int, y: int) -> None:
        rendered = self.text_cache.render(text_font, text, color)
        self.screen.blit(rendered, (right - rendered.get_width(), y))

    def draw_text_midleft(self, text: str, text_font: pygame.font.Font, color: tuple[int, int, int], midleft: tuple[int, int]) -> None:
        rendered = self.text_cache.render(text_font, text, color)
        self.screen.blit(rendered, rendered.get_rect(midleft=midleft))

    def draw_text_midright(self, text: str, text_font: pygame.font.Font, color: tuple[int, int, int], midright: tuple[int, int]) -> None:
        rendered = self.text_cache.render(text_font, text, color)
        self.screen.blit(rendered, rendered.get_rect(midright=midright))

    def draw_hud_bar(
        self,
        rect: pygame.Rect,
        value: float,
        color: tuple[int, int, int],
        bg: tuple[int, int, int] = (44, 57, 65),
    ) -> None:
        pygame.draw.rect(self.screen, bg, rect, border_radius=max(1, rect.h // 2))
        fill = pygame.Rect(rect.x, rect.y, int(rect.w * clamp(value)), rect.h)
        if fill.w > 0:
            pygame.draw.rect(self.screen, color, fill, border_radius=max(1, rect.h // 2))

    def draw_probability_strip(
        self,
        rect: pygame.Rect,
        values: tuple[float, float, float],
        colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    ) -> None:
        pygame.draw.rect(self.screen, (44, 57, 65), rect, border_radius=max(1, rect.h // 2))
        total = max(0.001, sum(max(0.0, value) for value in values))
        x = rect.x
        for index, (value, color) in enumerate(zip(values, colors)):
            if value <= 0:
                continue
            if index == len(values) - 1:
                width = max(0, rect.right - x)
            else:
                width = int(rect.w * (value / total))
            if width > 0:
                pygame.draw.rect(self.screen, color, pygame.Rect(x, rect.y, width, rect.h), border_radius=max(1, rect.h // 2))
                x += width

    def fmt_pct(self, value: float, digits: int = 0) -> str:
        return f"{value * 100:.{digits}f}%".replace(".", ",")

    def fmt_num(self, value: float, digits: int = 2) -> str:
        return f"{value:.{digits}f}".replace(".", ",")

    def ellipsize(self, text: str, text_font: pygame.font.Font, max_width: int) -> str:
        if text_font.size(text)[0] <= max_width:
            return text
        ellipsis = "..."
        trimmed = text
        while trimmed and text_font.size(trimmed + ellipsis)[0] > max_width:
            trimmed = trimmed[:-1]
        return (trimmed.rstrip() + ellipsis) if trimmed else ellipsis

    def draw_text_ellipsis(self, text: str, text_font: pygame.font.Font, color: tuple[int, int, int], x: int, y: int, max_width: int) -> None:
        self.draw_text(self.ellipsize(text, text_font, max_width), text_font, color, x, y)

    def draw_cover_image(self, image: pygame.Surface, rect: pygame.Rect, alpha: int = 255) -> None:
        self.screen.blit(self.surface_cache.cover(image, rect.size, alpha), rect.topleft)

    def fit_font(self, text: str, start_size: int, max_width: int, min_size: int = 16, bold: bool = True) -> pygame.font.Font:
        size = start_size
        while size > min_size:
            candidate = font(size, bold)
            if candidate.size(text)[0] <= max_width:
                return candidate
            size -= 2
        return font(min_size, bold)

    def team_arrow_rects(self, rect: pygame.Rect) -> tuple[pygame.Rect, pygame.Rect]:
        y = rect.y + 34
        return pygame.Rect(rect.right - 84, y, 32, 30), pygame.Rect(rect.right - 44, y, 32, 30)

    def draw_arrow_button(self, rect: pygame.Rect, label: str) -> None:
        hover = rect.collidepoint(self.mouse)
        pygame.draw.rect(self.screen, (25, 65, 78) if hover else (17, 43, 55), rect, border_radius=8)
        pygame.draw.rect(self.screen, CYAN if hover else (72, 111, 127), rect, 1, border_radius=8)
        rendered = self.text_cache.render(self.f_sm, label, WHITE)
        self.screen.blit(rendered, rendered.get_rect(center=rect.center))

    def draw_menu(self) -> None:
        if self.title_bg:
            self.screen.blit(self.title_bg, (0, 0))
        else:
            self.screen.fill(BG)
        overlay_key = ("menu_overlay", WIDTH, HEIGHT)
        overlay = self.cinematic_overlay_cache.get(overlay_key)
        if overlay is None:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 9, 14, 95))
            pygame.draw.rect(overlay, (0, 0, 0, 140), (0, 0, 510, HEIGHT))
            self.cinematic_overlay_cache[overlay_key] = overlay
        self.screen.blit(overlay, (0, 0))
        menu_x = self.start_button.rect.x
        self.draw_text("ORÁCULO", self.f_xl, WHITE, menu_x, 104)
        self.draw_text("DA COPA", self.f_xl, GOLD, menu_x, 168)
        if self.menu_icon:
            title_w = max(self.f_xl.size("ORÁCULO")[0], self.f_xl.size("DA COPA")[0])
            icon_rect = self.menu_icon.get_rect(midleft=(menu_x + title_w + 18, 170))
            glow_key = ("menu_icon_glow", icon_rect.w, icon_rect.h)
            glow = self.cinematic_overlay_cache.get(glow_key)
            if glow is None:
                glow = pygame.Surface((icon_rect.w + 18, icon_rect.h + 18), pygame.SRCALPHA)
                pygame.draw.rect(glow, (82, 226, 255, 16), glow.get_rect(), border_radius=28)
                pygame.draw.rect(glow, (250, 195, 67, 42), glow.get_rect().inflate(-6, -6), 2, border_radius=24)
                self.cinematic_overlay_cache[glow_key] = glow
            self.screen.blit(glow, glow.get_rect(center=icon_rect.center))
            self.screen.blit(self.menu_icon, icon_rect)
        self.draw_text("Copa do Mundo 2026", self.f_md, WHITE, menu_x, 264)
        self.draw_text("Escolha um duelo.", self.f_sm, MUTED, menu_x, 312)
        self.draw_text("Compare forma e elenco.", self.f_sm, MUTED, menu_x, 340)
        self.draw_text("Simule a Copa em tempo real.", self.f_sm, MUTED, menu_x, 368)
        self.start_button.draw(self.screen, self.f_md, self.mouse)
        self.draw_text("ENTER / ESPAÇO inicia", self.f_sm, MUTED, menu_x, 690)

    def draw_top(self, title: str, hint: str = "") -> None:
        pygame.draw.rect(self.screen, (4, 12, 18), (0, 0, WIDTH, 88))
        self.back_button.draw(self.screen, self.f_sm, self.mouse)
        title_x = 210
        title_y = 44
        if self.top_icon:
            icon_rect = self.top_icon.get_rect(midleft=(184, title_y))
            self.screen.blit(self.top_icon, icon_rect)
            title_x = 244
        title_font = self.fit_font(title, 42, 570 if hint else WIDTH - title_x - 42, min_size=32)
        self.draw_text_midleft(title, title_font, WHITE, (title_x, title_y))
        if hint:
            title_width = title_font.size(title)[0]
            hint_x = max(650, title_x + title_width + 44)
            hint_width = WIDTH - hint_x - 30
            if hint_width >= 140:
                hint_font = self.fit_font(hint, 20, hint_width, min_size=13)
                self.draw_text_ellipsis(hint, hint_font, MUTED, hint_x, 35, hint_width)

    def draw_select(self) -> None:
        self.screen.fill(BG)
        self.draw_select_background()
        self.draw_top("Escolha o confronto", "Setas: seleção A | A/D: seleção B | Espaço: confronto | T: Copa")
        self.draw_team_card(self.home, pygame.Rect(56, 118, 420, 460), "SELEÇÃO A", "LEFT/RIGHT")
        self.draw_team_card(self.away, pygame.Rect(804, 118, 420, 460), "SELEÇÃO B", "A / D")
        self.draw_vs()
        self.draw_engine_summary()
        self.single_button.draw(self.screen, self.f_sm, self.mouse)
        self.cup_button.draw(self.screen, self.f_sm, self.mouse)
        self.draw_text_centered(
            f"Base do jogo: {self.model.training_rows} partidas históricas + 48 seleções da Copa 2026.",
            self.f_sm,
            MUTED,
            (WIDTH // 2, 720),
        )

    def draw_select_background(self) -> None:
        background_rect = pygame.Rect(0, 88, WIDTH, 592)
        image = self.assets.fifa_images.get("mexico_opening")
        if image:
            self.draw_cover_image(image, background_rect, alpha=215)
        elif self.assets.stadium_bg:
            self.screen.blit(self.cached_smoothscale(self.assets.stadium_bg, background_rect.size), background_rect.topleft)
        shade_key = ("select_background_shade", background_rect.w, background_rect.h)
        shade = self.cinematic_overlay_cache.get(shade_key)
        if shade is None:
            shade = pygame.Surface(background_rect.size, pygame.SRCALPHA)
            shade.fill((0, 8, 13, 150))
            for y in range(background_rect.h):
                edge = max(0.0, abs(y / max(1, background_rect.h - 1) - 0.5) * 2.0)
                alpha = int(24 + 82 * edge)
                pygame.draw.line(shade, (0, 0, 0, alpha), (0, y), (background_rect.w, y))
            self.cinematic_overlay_cache[shade_key] = shade
        self.screen.blit(shade, background_rect.topleft)

    def draw_team_card(self, team: TeamProfile, rect: pygame.Rect, label: str, control: str) -> None:
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=22)
        pygame.draw.rect(self.screen, team.kit, rect, 3, border_radius=22)
        pad = 28
        flag_pos = (rect.x + pad, rect.y + 34)
        self.screen.blit(self.assets.flags[team.code], flag_pos)
        info_x = rect.x + 218
        content_right = rect.right - pad
        self.draw_text(label, self.f_sm, MUTED, info_x, rect.y + 38)
        prev_rect, next_rect = self.team_arrow_rects(rect)
        self.draw_arrow_button(prev_rect, "<")
        self.draw_arrow_button(next_rect, ">")
        self.draw_text(team.code, self.f_xl, WHITE, info_x, rect.y + 64)
        display_name = self.team_name(team.key)
        name_width = content_right - info_x
        name_font = self.fit_font(display_name, 38, name_width, min_size=18)
        self.draw_text_ellipsis(display_name, name_font, GOLD, info_x, rect.y + 132, name_width)
        rows = [
            ("ELO", team.elo, 2100, CYAN, f"{team.elo:.0f}"),
            ("Gols feitos", team.goals_for, 2.8, GREEN, self.fmt_num(team.goals_for)),
            ("Defesa", 2.4 - team.goals_against, 2.0, BLUE, f"{self.fmt_num(team.goals_against)} GC"),
            ("Vitórias", team.win_rate, 0.75, GOLD, self.fmt_pct(team.win_rate)),
            ("Elenco", team.squad_rating, 90.0, PURPLE, self.fmt_num(team.squad_rating, 1)),
        ]
        label_x = rect.x + pad
        bar_x = rect.x + 160
        bar_w = 150
        value_right = rect.right - 30
        for i, (name, value, maxv, color, shown) in enumerate(rows):
            y = rect.y + 228 + i * 41
            self.draw_text_midleft(name, self.f_sm, WHITE, (label_x, y + 7))
            self.draw_hud_bar(pygame.Rect(bar_x, y, bar_w, 14), float(value) / maxv, color, bg=(45, 58, 66))
            self.draw_text_midright(shown, self.f_xs, color, (value_right, y + 7))
        footer_y = rect.y + 420
        self.draw_text_ellipsis(f"Grupo {team.group} | ranking FIFA {team.fifa_rank}", self.f_sm, MUTED, label_x, footer_y, 260)
        self.draw_text_right("SETAS" if control == "LEFT/RIGHT" else control, self.f_sm, WHITE, content_right, footer_y)

    def draw_vs(self) -> None:
        center = (WIDTH // 2, 348)
        pygame.draw.circle(self.screen, (10, 22, 30), center, 78)
        pygame.draw.circle(self.screen, GOLD, center, 78, 3)
        self.draw_text_centered("VS", self.f_xl, WHITE, center)

    def draw_engine_summary(self) -> None:
        card_w = 304
        card_h = 50
        card_gap = 56
        x = WIDTH // 2 - card_w // 2
        y = 430
        cards = (
            ("1", "XGBoost 1X2", "classifica vitória, empate e derrota", CYAN),
            ("2", "Poisson/DC", "distribui os placares possíveis", GOLD),
            ("3", "Monte Carlo", "sorteia mil Copas possíveis", GREEN),
        )
        for index, (step, label, detail, color) in enumerate(cards):
            card = pygame.Rect(x, y + index * card_gap, card_w, card_h)
            pygame.draw.rect(self.screen, (7, 22, 31, 214), card, border_radius=11)
            pygame.draw.rect(self.screen, color, card, 2, border_radius=11)
            badge = pygame.Rect(card.x + 12, card.y + 11, 28, 28)
            pygame.draw.rect(self.screen, color, badge, border_radius=7)
            self.draw_text_centered(step, self.f_xs, BLACK, badge.center)
            self.draw_text_ellipsis(label, self.f_sm, color, card.x + 52, card.y + 6, card.w - 64)
            self.draw_text_ellipsis(detail, self.f_tiny, MUTED, card.x + 52, card.y + 31, card.w - 64)

    def active_sequence(self) -> list[str]:
        return ["CONFRONTO"]

    def simulation_progress(self) -> float:
        return clamp(self.t / SIMULATION_SECONDS)

    def match_minute_float(self) -> float:
        return self.simulation_progress() * 90.0

    def match_minute(self) -> int:
        minute = self.match_minute_float()
        if minute <= 0:
            return 0
        return min(90, int(math.ceil(minute - 1e-6)))

    def match_result_revealed(self) -> bool:
        return self.t >= SIMULATION_SECONDS

    def display_match_minute(self) -> int:
        minute = self.match_minute()
        if not self.match_result_revealed() and minute >= 90:
            return 89
        return minute

    def segment_duration(self) -> float:
        return SIMULATION_SECONDS / len(self.active_sequence())

    def current_algorithm(self) -> str:
        seq = self.active_sequence()
        idx = min(len(seq) - 1, int(self.t // self.segment_duration()))
        return seq[idx]

    def segment_progress(self) -> float:
        return clamp((self.t % self.segment_duration()) / self.segment_duration())

    def current_segment(self) -> int:
        return int(self.t // self.segment_duration())

    def elapsed_label(self) -> str:
        return f"{self.display_match_minute():02d}' / 90'"

    def outcome_label(self, pred: Prediction) -> str:
        if pred.outcome_class == 0:
            return f"Vitória {self.home.code}"
        if pred.outcome_class == 2:
            return f"Vitória {self.away.code}"
        return "Empate"

    def model_working_text(self, pred: Prediction) -> str:
        return "A IA lê o ritmo, pesa a camisa e segura o placar até o apito."

    def match_context_text(self) -> str:
        elo_delta = int(round(self.home.elo - self.away.elo))
        return f"Forma {self.fmt_num(self.home.form)}x{self.fmt_num(self.away.form)} | ELO {elo_delta:+d}"

    def live_probs(self, pred: Prediction) -> tuple[float, float, float]:
        reveal = smoothstep((self.simulation_progress() - 0.08) / 0.86)
        uncertainty = 1 - reveal
        pulse = math.sin(self.t * 1.7) * 0.018 * uncertainty
        home = lerp(0.34, pred.home, reveal) + pulse
        draw = lerp(0.33, pred.draw, reveal) - pulse * 0.5
        away = lerp(0.33, pred.away, reveal) - pulse * 0.5
        total = max(0.001, home + draw + away)
        return home / total, draw / total, away / total

    def score_from_prediction(self, pred: Prediction) -> tuple[int, int]:
        return self.score_until_minute(pred, self.match_minute_float())

    def score_until_minute(self, pred: Prediction, minute: float) -> tuple[int, int]:
        home_score = 0
        away_score = 0
        for goal_minute, side in self.goal_schedule(pred):
            visual_goal_minute = float(goal_minute)
            if minute + 0.01 < visual_goal_minute:
                continue
            if side == "home":
                home_score += 1
            else:
                away_score += 1
        return home_score, away_score

    def final_score_from_prediction(self, pred: Prediction) -> tuple[int, int]:
        if pred.score_home is not None and pred.score_away is not None:
            return pred.score_home, pred.score_away
        home_xg = self.score_intensity("home", pred)
        away_xg = self.score_intensity("away", pred)
        home_score = max(0, int(round(home_xg)))
        away_score = max(0, int(round(away_xg)))
        if pred.draw > max(pred.home, pred.away):
            level = max(0, int(round((home_xg + away_xg) / 2)))
            if level == 0 and (home_xg + away_xg) / 2 >= 0.65:
                level = 1
            return min(level, 5), min(level, 5)
        if pred.home > pred.away:
            home_score = max(1, home_score)
            if home_score <= away_score:
                home_score = away_score + 1
        else:
            away_score = max(1, away_score)
            if away_score <= home_score:
                away_score = home_score + 1
        return min(home_score, 5), min(away_score, 5)

    def statistical_pressure(self, side: str, pred: Prediction) -> float:
        team = self.home if side == "home" else self.away
        opponent = self.away if side == "home" else self.home
        prob = pred.home if side == "home" else pred.away
        xg = pred.home_goals if side == "home" else pred.away_goals
        attack = team.goals_for / max(0.1, team.goals_for + opponent.goals_against)
        opponent_leak = opponent.goals_against / max(0.1, team.goals_for + opponent.goals_against)
        elo_edge = 1 / (1 + 10 ** ((opponent.elo - team.elo) / 400))
        experience = math.log1p(team.matches) / max(0.1, math.log1p(team.matches) + math.log1p(opponent.matches))
        return (
            0.34 * xg
            + 0.22 * prob
            + 0.14 * attack
            + 0.08 * opponent_leak
            + 0.08 * team.win_rate
            + 0.07 * team.form
            + 0.05 * elo_edge
            + 0.02 * experience
        )

    def score_intensity(self, side: str, pred: Prediction) -> float:
        xg = pred.home_goals if side == "home" else pred.away_goals
        pressure = self.statistical_pressure(side, pred)
        return max(0.05, xg * 0.72 + pressure * 0.56 - pred.draw * 0.22)

    def match_runtime_key(self, pred: Prediction) -> tuple[object, ...]:
        return (
            self.home.code,
            self.away.code,
            self.match_seed,
            pred.algorithm,
            pred.outcome_class,
            pred.score_home,
            pred.score_away,
            round(pred.home, 6),
            round(pred.draw, 6),
            round(pred.away, 6),
            round(pred.home_goals, 6),
            round(pred.away_goals, 6),
        )

    def match_runtime_state(self, pred: Prediction) -> MatchRuntimeState:
        key = self.match_runtime_key(pred)
        cached = self.match_runtime_state_cache.get(key)
        if cached is not None:
            return cached
        goals = tuple(self.build_goal_schedule(pred))
        chances = tuple(self.build_chance_schedule(pred, goals))
        state = MatchRuntimeState(key=key, goals=goals, chances=chances)
        self.match_runtime_state_cache[key] = state
        field = self.match_field_rect()
        for minute, side in goals:
            goal_side = self.cinematic_goal_side(side)
            self.cinematic_shot_plan(
                self.cinematic_goal_rect(field, goal_side),
                minute,
                side,
                "goal",
            )
        for minute, side, outcome in chances:
            goal_side = self.cinematic_goal_side(side)
            self.cinematic_shot_plan(
                self.cinematic_goal_rect(field, goal_side),
                minute,
                side,
                outcome,
            )
        return state

    def goal_schedule(self, pred: Prediction) -> list[tuple[int, str]]:
        return list(self.match_runtime_state(pred).goals)

    def chance_schedule(self, pred: Prediction) -> list[tuple[int, str, str]]:
        return list(self.match_runtime_state(pred).chances)

    def build_goal_schedule(self, pred: Prediction) -> list[tuple[int, str]]:
        home_score, away_score = self.final_score_from_prediction(pred)
        if home_score + away_score == 0:
            return []
        home_pressure = self.statistical_pressure("home", pred)
        away_pressure = self.statistical_pressure("away", pred)
        goals = [("home", home_pressure)] * home_score + [("away", away_pressure)] * away_score
        goals.sort(key=lambda item: (-item[1], item[0]))
        seed = sum(ord(ch) for ch in f"{pred.algorithm}:{self.home.code}:{self.away.code}:{home_score}:{away_score}:{pred.outcome_class}")
        total = len(goals)
        preferred: list[tuple[float, int, str]] = []
        for index, (side, pressure) in enumerate(goals):
            opponent_pressure = away_pressure if side == "home" else home_pressure
            base_minute = (index + 1) * 90 / (total + 1)
            edge = clamp(pressure - opponent_pressure, -1.0, 1.0)
            wave = math.sin(seed * 0.17 + index * 1.91) * 4.0
            minute = int(round(clamp(base_minute - edge * 8.0 + wave, 7.0, 88.0)))
            preferred.append((float(minute), index, side))
        if total == 1:
            slots = [int(round(clamp(preferred[0][0], 34.0, 56.0)))]
        elif total == 2:
            slots = [30, 63]
        else:
            start = max(14, 32 - total * 3)
            end = min(86, 78 + total)
            step = (end - start) / max(1, total - 1)
            slots = [int(round(start + index * step)) for index in range(total)]
        schedule = [(slot, side) for slot, (_preferred, _index, side) in zip(slots, sorted(preferred))]
        return sorted(schedule, key=lambda item: item[0])

    def build_chance_schedule(
        self,
        pred: Prediction,
        goals: tuple[tuple[int, str], ...],
    ) -> list[tuple[int, str, str]]:
        goal_minutes = [minute for minute, _side in goals]
        seed = sum(ord(ch) for ch in f"chance:{pred.algorithm}:{self.home.code}:{self.away.code}:{pred.home:.3f}:{pred.draw:.3f}:{pred.away:.3f}:{self.match_seed}")
        home_pressure = self.statistical_pressure("home", pred)
        away_pressure = self.statistical_pressure("away", pred)
        total_xg = max(0.4, pred.home_goals + pred.away_goals)
        chance_count = int(clamp(round(1.0 + total_xg + (1.0 - abs(pred.home - pred.away)) * 1.4), 2, 4))
        schedule: list[tuple[int, str, str]] = []
        for index in range(chance_count):
            wave = math.sin(seed * 0.071 + index * 1.73)
            base_minute = 12 + (index + 0.62) * (70 / max(1, chance_count))
            minute = int(round(clamp(base_minute + wave * 6.0, 8.0, 84.0)))
            side_seed = math.sin(seed * 0.113 + index * 2.11)
            pressure_edge = home_pressure - away_pressure
            side = "home" if pressure_edge + side_seed * 0.34 >= 0 else "away"
            kind = "save" if math.sin(seed * 0.19 + index * 0.91) >= -0.25 else "wide"
            attempts = 0
            while (
                any(abs(minute - existing) < CHANCE_MIN_SPACING_MINUTES for existing, _side, _kind in schedule)
                or any(abs(minute - goal_minute) < GOAL_MIN_SPACING_MINUTES for goal_minute in goal_minutes)
            ) and attempts < 12:
                minute = int(clamp(minute + (CHANCE_MIN_SPACING_MINUTES if minute <= 74 else -CHANCE_MIN_SPACING_MINUTES), 8, 84))
                attempts += 1
            blocked = (
                any(abs(minute - existing) < CHANCE_MIN_SPACING_MINUTES for existing, _side, _kind in schedule)
                or any(abs(minute - goal_minute) < GOAL_MIN_SPACING_MINUTES for goal_minute in goal_minutes)
            )
            if blocked or any(abs(minute - goal_minute) < 7 for goal_minute in goal_minutes):
                continue
            schedule.append((minute, side, kind))
        if len(schedule) >= 2:
            kinds = {kind for _minute, _side, kind in schedule}
            if "save" not in kinds:
                minute, side, _kind = schedule[0]
                schedule[0] = (minute, side, "save")
            if "wide" not in kinds:
                minute, side, _kind = schedule[-1]
                schedule[-1] = (minute, side, "wide")
        return sorted(schedule, key=lambda item: item[0])

    def active_goal_event(self, pred: Prediction) -> tuple[int, str] | None:
        minute = self.match_minute_float()
        for goal_minute, side in self.goal_schedule(pred):
            visual_goal_minute = float(goal_minute)
            if visual_goal_minute - 0.01 <= minute < goal_minute + GOAL_PAYOFF_MINUTES:
                return goal_minute, side
        return None

    def active_chance_event(self, pred: Prediction) -> tuple[int, str, str] | None:
        if self.ball_goal_event(pred):
            return None
        minute = self.match_minute_float()
        for chance_minute, side, kind in self.chance_schedule(pred):
            start = chance_minute - CHANCE_EVENT_WINDOW_MINUTES
            payoff_end = chance_minute + CHANCE_PAYOFF_MINUTES
            if start <= minute <= payoff_end:
                return chance_minute, side, kind
        return None

    def active_attack_event(self, pred: Prediction) -> CinematicAttackEvent | None:
        goal = self.ball_goal_event(pred)
        if goal:
            return CinematicAttackEvent(goal[0], goal[1], True, "goal")
        chance = self.active_chance_event(pred)
        if chance:
            return CinematicAttackEvent(chance[0], chance[1], False, chance[2])
        return None

    def ball_goal_event(self, pred: Prediction) -> tuple[int, str] | None:
        minute = self.match_minute_float()
        candidates: list[tuple[int, float, int, str]] = []
        for goal_minute, side in self.goal_schedule(pred):
            start = goal_minute - GOAL_EVENT_WINDOW_MINUTES
            payoff_end = goal_minute + GOAL_PAYOFF_MINUTES
            if not start <= minute <= payoff_end:
                continue
            if minute < goal_minute:
                priority = 3
            elif minute <= goal_minute + 4.0:
                priority = 2
            else:
                priority = 1
            candidates.append((priority, start, goal_minute, side))
        if not candidates:
            return None
        _priority, _start, goal_minute, side = max(candidates, key=lambda item: (item[0], item[1]))
        return goal_minute, side

    def match_cinematic_focus(self, pred: Prediction) -> bool:
        active_goal = self.ball_goal_event(pred)
        if not active_goal:
            return False
        goal_minute, _side = active_goal
        shot_progress = clamp(
            (
                self.match_minute_float()
                - (goal_minute - GOAL_EVENT_WINDOW_MINUTES)
            )
            / GOAL_EVENT_WINDOW_MINUTES
        )
        return shot_progress >= 0.48

    def match_hud_state_key(self, cinematic_focus: bool = False) -> str:
        if self.match_result_revealed():
            return "closed"
        if cinematic_focus:
            return "focus"
        return "live"

    def match_hud_state_copy(self, cinematic_focus: bool = False) -> tuple[str, str, str]:
        return MATCH_HUD_STATE_COPY[self.match_hud_state_key(cinematic_focus)]

    def draw_simulate(self) -> None:
        self.screen.fill(BG)
        self.draw_top(f"{self.home.code} x {self.away.code}", "BACKSPACE volta | ESPAÇO/R reinicia | T abre Copa")
        pred = self.match_prediction
        if pred is None:
            self.draw_text("Preparando confronto...", self.f_md, MUTED, 58, 136)
            return
        cinematic_focus = self.match_cinematic_focus(pred)
        self.draw_field(pred, pred, "CONFRONTO")
        self.draw_side_panel(pred, cinematic_focus=cinematic_focus)
        self.draw_score_panel({"CONFRONTO": pred}, "CONFRONTO", pred, cinematic_focus=cinematic_focus)

    def match_field_rect(self) -> pygame.Rect:
        return pygame.Rect(32, 110, 910, 490)

    def match_side_panel_rect(self) -> pygame.Rect:
        return pygame.Rect(970, 110, 278, 490)

    def match_score_panel_rect(self) -> pygame.Rect:
        return pygame.Rect(32, 610, 1216, 126)

    def match_clock_rect(self, field: pygame.Rect) -> pygame.Rect:
        return pygame.Rect(field.right - 150, field.y + 20, 116, 38)

    def match_narrator_rect(self, field: pygame.Rect, possession: str) -> pygame.Rect:
        narrator_width = 384
        clock = self.match_clock_rect(field)
        if possession == "away":
            x = min(clock.x - narrator_width - 24, field.right - narrator_width - 24)
        else:
            x = field.x + 22
        return pygame.Rect(max(field.x + 18, x), field.y + 20, narrator_width, 78)

    def score_until_now(self, predictions: dict[str, Prediction]) -> tuple[int, int]:
        pred = predictions["CONFRONTO"]
        return self.score_from_prediction(pred)

    def draw_field(self, pred: Prediction, result_pred: Prediction, algo: str) -> None:
        rect = self.match_field_rect()
        pygame.draw.rect(self.screen, (2, 8, 11), rect.inflate(20, 20), border_radius=20)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(rect)
        state = self.cinematic_scene_state(rect, pred)
        if state.get("poc_sequence"):
            self.draw_cinematic_poc_sequence(rect, state)
        else:
            viewport = PocViewport.fit(rect)
            self.draw_cinematic_poc_background(
                rect,
                viewport,
                float(
                    state.get(
                        "poc_scroll",
                        self.ground_scroll
                        / max(1e-6, viewport.scale),
                    )
                ),
            )
            if not state.get("active_attack") and self.simulation_progress() < 0.88:
                self.draw_model_flow(rect, pred, algo)
            self.draw_cinematic_scene(rect, pred, state)
        self.draw_cinematic_goal_overlay(rect, pred)
        self.screen.set_clip(old_clip)
        possession = str(state.get("possession", "home"))
        shot_progress = float(state.get("shot_progress", 0.0))
        cinematic_focus = bool(state.get("active_goal")) and shot_progress >= 0.48
        chance_focus = bool(state.get("active_attack")) and not state.get("active_goal") and shot_progress >= 0.50
        if cinematic_focus:
            self.draw_cinematic_focus_tag(rect, pred)
        elif chance_focus:
            self.draw_cinematic_chance_tag(rect, pred, state)
        else:
            narrator = self.match_narrator_rect(rect, possession)
            pygame.draw.rect(self.screen, (5, 17, 24), narrator, border_radius=15)
            narration_title, narration_body = FIELD_NARRATION_COPY.get(possession, FIELD_NARRATION_COPY["home"])
            self.draw_text(narration_title, self.f_xs, CYAN if possession == "home" else GOLD, narrator.x + 20, narrator.y + 10)
            self.draw_text_ellipsis(narration_body, self.f_sm, WHITE, narrator.x + 20, narrator.y + 31, narrator.w - 40)
            self.draw_text("Jogo corrido, sem placar antecipado.", self.f_xs, MUTED, narrator.x + 20, narrator.y + 54)
        self.draw_clock(rect)

    def draw_cinematic_focus_tag(self, field: pygame.Rect, pred: Prediction) -> None:
        tag = pygame.Rect(field.x + 22, field.y + 22, 410, 54)
        cache_key = ("focus_tag_panel", tag.w, tag.h)
        panel = self.cinematic_overlay_cache.get(cache_key)
        if panel is None:
            panel = pygame.Surface(tag.size, pygame.SRCALPHA)
            pygame.draw.rect(panel, (2, 9, 13, 214), panel.get_rect(), border_radius=14)
            pygame.draw.rect(panel, (*CYAN, 120), panel.get_rect(), 1, border_radius=14)
            self.cinematic_overlay_cache[cache_key] = panel
        self.screen.blit(panel, tag.topleft)
        self.draw_text_ellipsis("Cheiro de gol", self.f_xs, CYAN, tag.x + 18, tag.y + 8, tag.w - 36)
        self.draw_text_ellipsis("A jogada está viva.", self.f_xs, WHITE, tag.x + 18, tag.y + 30, tag.w - 36)

    def draw_cinematic_chance_tag(self, field: pygame.Rect, pred: Prediction, state: dict[str, object]) -> None:
        tag = pygame.Rect(field.x + 22, field.y + 22, 410, 54)
        cache_key = ("chance_tag_panel", tag.w, tag.h)
        panel = self.cinematic_overlay_cache.get(cache_key)
        if panel is None:
            panel = pygame.Surface(tag.size, pygame.SRCALPHA)
            pygame.draw.rect(panel, (2, 9, 13, 214), panel.get_rect(), border_radius=14)
            pygame.draw.rect(panel, (*GOLD, 120), panel.get_rect(), 1, border_radius=14)
            self.cinematic_overlay_cache[cache_key] = panel
        self.screen.blit(panel, tag.topleft)
        side = str(state.get("possession", "home"))
        code = self.home.code if side == "home" else self.away.code
        chance_kind = "defesa do goleiro" if state.get("attack_kind") == "save" else "passa raspando a trave"
        self.draw_text_ellipsis("Quase gol", self.f_xs, GOLD, tag.x + 18, tag.y + 8, tag.w - 36)
        self.draw_text_ellipsis(f"{code} cria perigo: {chance_kind}", self.f_xs, WHITE, tag.x + 18, tag.y + 30, tag.w - 36)

    def draw_clock(self, field: pygame.Rect) -> None:
        box = self.match_clock_rect(field)
        pygame.draw.rect(self.screen, (3, 12, 18), box, border_radius=12)
        pygame.draw.rect(self.screen, CYAN, box, 1, border_radius=12)
        self.draw_text_centered(self.elapsed_label(), self.f_sm, WHITE, box.center)
        pygame.draw.rect(self.screen, (45, 58, 66), (box.x, box.bottom + 8, box.w, 5), border_radius=4)
        pygame.draw.rect(self.screen, CYAN, (box.x, box.bottom + 8, int(box.w * self.simulation_progress()), 5), border_radius=4)

    def cinematic_possession_side(self, pred: Prediction) -> str:
        active_attack = self.active_attack_event(pred)
        if self.simulation_progress() >= 1.0:
            active_attack = None
        if active_attack:
            return active_attack.side
        final_home, final_away = self.final_score_from_prediction(pred)
        if final_home == final_away and self.simulation_progress() > DRAW_NEUTRAL_START_PROGRESS:
            return "neutral"
        home_score, away_score = self.score_from_prediction(pred)
        if home_score > away_score:
            return "home"
        if away_score > home_score:
            return "away"
        if final_home > final_away:
            return "home"
        if final_away > final_home:
            return "away"
        home, draw, away = self.live_probs(pred)
        if draw >= max(home, away):
            return "home" if math.sin(self.t * 0.55 + self.match_seed * 0.001) >= 0 else "away"
        return "home" if home >= away else "away"

    def cinematic_goal_side(self, possession: str) -> str:
        return "left" if possession == "away" else "right"

    def cinematic_goal_rect(self, field: pygame.Rect, side: str) -> pygame.Rect:
        bottom = field.bottom - CINEMATIC_GOAL_BOTTOM_INSET
        if side == "left":
            return pygame.Rect(
                field.x + CINEMATIC_GOAL_EDGE_INSET,
                bottom - CINEMATIC_GOAL_HEIGHT,
                CINEMATIC_GOAL_WIDTH,
                CINEMATIC_GOAL_HEIGHT,
            )
        return pygame.Rect(
            field.right - CINEMATIC_GOAL_EDGE_INSET - CINEMATIC_GOAL_WIDTH,
            bottom - CINEMATIC_GOAL_HEIGHT,
            CINEMATIC_GOAL_WIDTH,
            CINEMATIC_GOAL_HEIGHT,
        )

    def cinematic_goal_layer_target_rect(
        self,
        goal: pygame.Rect,
        side: str,
        frame: pygame.Surface,
    ) -> pygame.Rect:
        mouth_ratio = (
            CINEMATIC_GOAL_SOURCE_POST_X[1]
            - CINEMATIC_GOAL_SOURCE_POST_X[0]
        ) / CINEMATIC_GOAL_SOURCE_WIDTH
        target_w = int(round(goal.w / mouth_ratio))
        aspect = frame.get_width() / max(1, frame.get_height())
        target_h = int(round(target_w / aspect))
        if side == "right":
            left_post_ratio = (
                CINEMATIC_GOAL_SOURCE_POST_X[0]
                / CINEMATIC_GOAL_SOURCE_WIDTH
            )
        else:
            left_post_ratio = (
                CINEMATIC_GOAL_SOURCE_WIDTH
                - CINEMATIC_GOAL_SOURCE_POST_X[1]
            ) / CINEMATIC_GOAL_SOURCE_WIDTH
        target = pygame.Rect(0, 0, target_w, target_h)
        target.x = int(round(goal.left - target_w * left_post_ratio))
        target.y = int(
            round(
                goal.bottom
                - target_h
                * CINEMATIC_GOAL_SOURCE_POST_BOTTOM
                / CINEMATIC_GOAL_SOURCE_HEIGHT
            )
        )
        return target

    @staticmethod
    def cinematic_goal_entry_x(goal: pygame.Rect, direction: int) -> float:
        return (
            float(goal.centerx)
            - float(direction) * CINEMATIC_GOAL_ENTRY_DEPTH
        )

    def cinematic_camera_progress(self, pred: Prediction) -> float:
        active_attack = self.active_attack_event(pred)
        minute = self.match_minute_float()
        if active_attack:
            event_window = GOAL_EVENT_WINDOW_MINUTES if active_attack.is_goal else CHANCE_EVENT_WINDOW_MINUTES
            event_minute = active_attack.minute
            attack_progress = smoothstep(clamp((minute - (event_minute - event_window)) / event_window))
            cruise_progress = self.cinematic_cruise_camera_progress(pred, minute)
            attack_entry_progress = self.cinematic_cruise_camera_progress(pred, event_minute - event_window)
            attack_target = 1.0 if active_attack.side == "home" else 0.0
            settle_duration = 2.60
            payoff_duration = (
                GOAL_PAYOFF_MINUTES
                if active_attack.is_goal
                else CHANCE_PAYOFF_MINUTES
            )
            settle_after = max(0.0, payoff_duration - settle_duration)
            settle = smoothstep(
                (minute - (event_minute + settle_after)) / settle_duration
            )
            attack_camera = lerp(attack_entry_progress, attack_target, attack_progress)
            progress = lerp(attack_camera, cruise_progress, settle)
        else:
            progress = self.cinematic_cruise_camera_progress(pred, minute)
        return progress

    def cinematic_cruise_camera_progress(self, pred: Prediction, minute: float) -> float:
        schedule = self.goal_schedule(pred)
        if schedule:
            first_attack_start = max(6.0, float(schedule[0][0]) - GOAL_EVENT_WINDOW_MINUTES)
        else:
            chance_schedule = self.chance_schedule(pred)
            first_attack_start = max(6.0, float(chance_schedule[0][0]) - CHANCE_EVENT_WINDOW_MINUTES) if chance_schedule else 60.0
        return 0.18 + 0.64 * smoothstep(clamp(minute / first_attack_start))

    def cinematic_motion_state(self, pred: Prediction) -> dict[str, float]:
        active_attack = self.active_attack_event(pred)
        possession = active_attack.side if active_attack else self.cinematic_possession_side(pred)
        direction = -1.0 if possession == "away" else 1.0
        minute = self.match_minute_float()
        schedule = self.goal_schedule(pred)
        if schedule:
            first_attack_start = max(6.0, float(schedule[0][0]) - GOAL_EVENT_WINDOW_MINUTES)
        else:
            chance_schedule = self.chance_schedule(pred)
            first_attack_start = max(6.0, float(chance_schedule[0][0]) - CHANCE_EVENT_WINDOW_MINUTES) if chance_schedule else 60.0
        approach_speed = smoothstep(clamp(minute / first_attack_start))
        shot_speed = 0.0
        if active_attack:
            event_window = GOAL_EVENT_WINDOW_MINUTES if active_attack.is_goal else CHANCE_EVENT_WINDOW_MINUTES
            event_minute = active_attack.minute
            shot_progress = clamp((minute - (event_minute - event_window)) / event_window)
            shot_speed = smoothstep(shot_progress)
        run_speed = clamp(0.46 + approach_speed * 0.44 + shot_speed * 0.14, 0.46, 1.04)
        camera = self.cinematic_camera_progress(pred)
        travel_distance = self.ground_travel_distance
        stride_phase = (travel_distance / CINEMATIC_RUNNER_STRIDE_DISTANCE * 4.0) % 4.0
        desired_scroll_velocity = direction * CINEMATIC_TURF_SPEED
        return {
            "direction": direction,
            "camera": camera,
            "run_speed": run_speed,
            "stride_phase": stride_phase,
            "travel_distance": travel_distance,
            "ground_scroll": self.ground_scroll,
            "desired_scroll_velocity": desired_scroll_velocity,
        }

    def cinematic_shot_plan(
        self,
        goal_rect: pygame.Rect,
        event_minute: int,
        side: str,
        outcome: str,
    ) -> CinematicShotPlan:
        if side not in {"home", "away"}:
            raise ValueError(f"invalid cinematic side: {side}")
        if outcome not in {"goal", "save", "wide"}:
            raise ValueError(f"invalid cinematic outcome: {outcome}")
        key = (int(self.match_seed), int(event_minute), side, outcome)
        cached = self.cinematic_shot_plan_cache.get(key)
        if cached is not None:
            return cached

        direction = 1 if side == "home" else -1
        seed = (
            int(self.match_seed) * 1_000_003
            + int(event_minute) * 9_176
            + (0 if direction > 0 else 4_699)
        )
        rng = random.Random(seed)
        profiles = (
            ("alto firme", -0.54, 48.0, 0.20, 31.0, 1.34, 40.0),
            ("baixo cruzado", 0.54, 118.0, -0.14, 22.0, 1.48, 36.0),
            ("meia altura", -0.06, 76.0, 0.08, 26.0, 1.40, 38.0),
            ("angulo seco", -0.46, 135.0, -0.18, 34.0, 1.28, 42.0),
            ("rasteiro forte", 0.68, 44.0, 0.08, 18.0, 1.56, 34.0),
            ("central forte", 0.16, 94.0, -0.06, 24.0, 1.46, 37.0),
        )
        index = rng.randrange(len(profiles))
        zone, y_ratio, depth, bend_base, loft_base, speed_base, spin = profiles[index]
        target_jitter_x = rng.uniform(-4.0, 4.0)
        target_jitter_y = rng.uniform(-7.0, 7.0)
        target_y = clamp(
            goal_rect.centery + y_ratio * goal_rect.h * 0.39 + target_jitter_y,
            goal_rect.y + CINEMATIC_SHOT_BALL_SIZE * 0.66,
            goal_rect.bottom - CINEMATIC_SHOT_BALL_SIZE * 0.66,
        )
        mouth_x = self.cinematic_goal_entry_x(goal_rect, direction)
        depth_ratio = clamp((depth - 44.0) / (135.0 - 44.0))
        target_depth = lerp(
            goal_rect.w * CINEMATIC_NET_TARGET_DEPTH_MAX_RATIO,
            goal_rect.w * CINEMATIC_NET_TARGET_DEPTH_MIN_RATIO,
            depth_ratio,
        ) + target_jitter_x * 0.5
        target_x = mouth_x + direction * target_depth
        target_x = clamp(
            target_x,
            goal_rect.left + CINEMATIC_SHOT_BALL_SIZE * 0.66,
            goal_rect.right - CINEMATIC_SHOT_BALL_SIZE * 0.66,
        )
        mouth_y = target_y - rng.uniform(0.0, 5.0)
        entry = (mouth_x, mouth_y)
        mouth = (mouth_x, mouth_y)
        bend = direction * (bend_base * 9.0 + rng.uniform(-1.8, 1.8))
        profile = ShotProfile(
            zone=zone,
            target=(target_x, target_y),
            entry=entry,
            mouth=mouth,
            bend=bend,
            loft=loft_base,
            dip=rng.uniform(4.0, 9.0),
            speed=speed_base,
            spin=spin + rng.uniform(-2.0, 3.0),
        )
        save_variant = (
            self.cinematic_save_variant(event_minute, side)
            if outcome == "save"
            else ""
        )
        target = profile.target
        if outcome != "goal":
            target = self.cinematic_chance_target_from_profile(
                profile,
                goal_rect,
                direction,
                outcome,
                save_variant,
            )
        plan = CinematicShotPlan(
            key=key,
            profile=profile,
            profile_band=SHOT_PROFILE_BANDS[profile.zone],
            target=target,
            save_variant=save_variant,
        )
        self.cinematic_shot_plan_cache[key] = plan
        return plan

    def cinematic_shot_profile(
        self,
        goal_rect: pygame.Rect,
        direction: int,
        goal_minute: int,
    ) -> ShotProfile:
        side = "home" if direction > 0 else "away"
        return self.cinematic_shot_plan(
            goal_rect,
            goal_minute,
            side,
            "goal",
        ).profile

    def cinematic_shot_target(self, goal_rect: pygame.Rect, direction: int, goal_minute: int) -> tuple[float, float]:
        return self.cinematic_shot_profile(goal_rect, direction, goal_minute).target

    def cinematic_net_path_table(
        self,
        entry: tuple[float, float],
        control_a: tuple[float, float],
        control_b: tuple[float, float],
        rest: tuple[float, float],
    ) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...], float]:
        cache_key = tuple(
            round(value, 3)
            for point in (entry, control_a, control_b, rest)
            for value in point
        )
        cached = self.ball_net_path_cache.get(cache_key)
        if cached is not None:
            return cached
        if len(self.ball_net_path_cache) >= 128:
            self.ball_net_path_cache.clear()
        table = cubic_bezier_arc_table(entry, control_a, control_b, rest)
        self.ball_net_path_cache[cache_key] = table
        return table

    @staticmethod
    def cinematic_poc2_contact_offsets(
        direction: int,
    ) -> tuple[float, float]:
        source_offsets = (
            CINEMATIC_POC2_CONTACT_OFFSETS_RIGHT
            if direction > 0
            else CINEMATIC_POC2_CONTACT_OFFSETS_LEFT
        )
        return (
            source_offsets[0] * CINEMATIC_POC2_RENDER_SCALE,
            source_offsets[1] * CINEMATIC_POC2_RENDER_SCALE,
        )

    def cinematic_dribble_kinematics(
        self,
        actor_pos: tuple[float, float],
        direction: int,
        stride_phase: float,
        time_value: float,
        shot_progress: float = 0.0,
        kick_contact_position: tuple[float, float] | None = None,
        foot_position: tuple[float, float] | None = None,
        visible_foot_position: tuple[float, float] | None = None,
        contact_foot_position: tuple[float, float] | None = None,
        travel_distance: float | None = None,
        kick_contact_gap: float = CINEMATIC_KICK_CONTACT_GAP,
        minimum_directed_ball_x: float | None = None,
        planned_directed_ball_x: float | None = None,
        poc2_contact_offsets: tuple[float, float] | None = None,
    ) -> DribbleKinematics:
        if (
            poc2_contact_offsets is not None
            and kick_contact_position is None
        ):
            cycle_phase = (stride_phase % 4.0) / 4.0
            touch_position = cycle_phase * 2.0
            touch_slot = min(1, int(touch_position))
            touch_phase = touch_position - touch_slot
            contact_start = poc2_contact_offsets[touch_slot]
            contact_end = poc2_contact_offsets[
                (touch_slot + 1) % 2
            ]
            contact_track = lerp(
                contact_start,
                contact_end,
                touch_phase,
            )
            free_roll = math.sin(math.pi * touch_phase)
            excursion = (
                22.0
                * CINEMATIC_POC2_RENDER_SCALE
            )
            relative_offset = (
                contact_track
                + excursion * free_roll
            )
            ball_radius = (
                21.0
                * CINEMATIC_POC2_RENDER_SCALE
            )
            contact_gap = (
                2.0
                * CINEMATIC_POC2_RENDER_SCALE
            )
            ball_x = (
                float(actor_pos[0])
                + direction * relative_offset
            )
            ground_y = float(actor_pos[1])
            center_y = ground_y - ball_radius
            control_x = (
                ball_x
                - direction
                * (ball_radius + contact_gap)
            )
            visible_foot_x = (
                control_x
                if visible_foot_position is None
                else float(visible_foot_position[0])
            )
            visible_foot_y = (
                center_y
                if visible_foot_position is None
                else float(visible_foot_position[1])
            )
            travel = (
                time_value
                * CINEMATIC_RUNNER_STRIDE_DISTANCE
                / CINEMATIC_POC2_CYCLE_SECONDS
                if travel_distance is None
                else travel_distance
            )
            roll_radius = (
                27.0
                * CINEMATIC_POC2_RENDER_SCALE
            )
            rotation = (
                -direction
                * (
                    travel
                    + relative_offset
                    - poc2_contact_offsets[0]
                )
                / max(1e-6, roll_radius)
                * (180.0 / math.pi)
            )
            ball = BallKinematics(
                position=(ball_x, center_y),
                ground_position=(ball_x, ground_y - 1.0),
                phase="drible",
                scale=CINEMATIC_BALL_SIZE,
                squash=(1.0, 1.0),
                rotation_degrees=rotation,
                depth=0.0,
            )
            return DribbleKinematics(
                control_position=(control_x, center_y),
                visible_foot_position=(
                    visible_foot_x,
                    visible_foot_y,
                ),
                ball=ball,
                touch_phase=touch_phase,
                contact_gap=direction
                * (ball_x - visible_foot_x)
                - ball_radius,
            )

        shadow_y = float(actor_pos[1]) - 2.0
        center_y = shadow_y - CINEMATIC_BALL_SIZE * CINEMATIC_BALL_GROUND_RADIUS_RATIO
        contact_radius = CINEMATIC_BALL_SIZE * 0.46
        touch_phase = cinematic_dribble_touch_phase(stride_phase, direction)
        push_peak = CINEMATIC_DRIBBLE_PUSH_PEAK
        if touch_phase <= push_peak:
            free_roll = smoothstep(touch_phase / push_peak)
        else:
            free_roll = 1.0 - smoothstep((touch_phase - push_peak) / (1.0 - push_peak))
        nominal_control_x = (
            float(actor_pos[0])
            + direction * (CINEMATIC_DRIBBLE_CONTROL_OFFSET - CINEMATIC_DRIBBLE_FOOT_RETREAT * free_roll)
        )
        visible_foot_x = (
            nominal_control_x
            if visible_foot_position is None
            else float(visible_foot_position[0])
        )
        visible_foot_y = center_y if visible_foot_position is None else float(visible_foot_position[1])
        control_x = nominal_control_x
        control_y = center_y
        excursion = (
            CINEMATIC_DRIBBLE_BALL_EXCURSION_RIGHT
            if direction > 0
            else CINEMATIC_DRIBBLE_BALL_EXCURSION_LEFT
        )
        if kick_contact_position is not None and shot_progress > 0.0:
            attack_cushion = lerp(1.0, 0.60, smoothstep(shot_progress / 0.12))
            excursion *= attack_cushion
        surface_lead = CINEMATIC_DRIBBLE_CONTACT_GAP + excursion * free_roll
        contact_anchor_x = (
            float(contact_foot_position[0])
            if contact_foot_position is not None
            else (float(foot_position[0]) if foot_position is not None else visible_foot_x)
        )
        touch_lock = smoothstep(1.0 - free_roll)
        control_x = lerp(control_x, contact_anchor_x, touch_lock)
        contact_foot_x = control_x

        plant = 0.0
        if kick_contact_position is not None:
            plant_start = SHOT_PLANT_AT - 0.10
            plant = smoothstep((shot_progress - plant_start) / max(0.001, SHOT_KICK_AT - plant_start))
            target_control_x = float(kick_contact_position[0])
            target_control_y = float(kick_contact_position[1])
            control_x = lerp(control_x, target_control_x, plant)
            control_y = lerp(control_y, target_control_y, plant)
            visible_foot_x = lerp(visible_foot_x, target_control_x, plant)
            visible_foot_y = lerp(visible_foot_y, target_control_y, plant)
            contact_foot_x = lerp(contact_foot_x, target_control_x, plant)
            surface_lead = lerp(surface_lead, kick_contact_gap, plant)

        ball_x = control_x + direction * (contact_radius + surface_lead)
        if (
            kick_contact_position is not None
            and shot_progress >= CINEMATIC_KICK_APPROACH_START
        ):
            target_ball_x = float(kick_contact_position[0]) + direction * (
                contact_radius + kick_contact_gap
            )
            directed_target_x = direction * target_ball_x
            approach = smoothstep(
                (shot_progress - CINEMATIC_KICK_APPROACH_START)
                / max(0.001, SHOT_KICK_AT - CINEMATIC_KICK_APPROACH_START)
            )
            monotonic_approach_x = lerp(
                directed_target_x - CINEMATIC_KICK_APPROACH_DISTANCE,
                directed_target_x,
                approach,
            )
            directed_ball_x = monotonic_approach_x
            if minimum_directed_ball_x is not None:
                directed_ball_x = max(
                    directed_ball_x,
                    minimum_directed_ball_x,
                )
            ball_x = direction * directed_ball_x
        elif minimum_directed_ball_x is not None:
            directed_ball_x = max(
                direction * ball_x,
                minimum_directed_ball_x,
            )
            ball_x = direction * directed_ball_x
        if planned_directed_ball_x is not None:
            ball_x = direction * planned_directed_ball_x

        contact_gap = direction * (ball_x - contact_foot_x) - contact_radius
        forward_distance = (
            (time_value * CINEMATIC_TURF_SPEED * 0.82 if travel_distance is None else travel_distance)
            + direction * ball_x
        )
        rolling_diameter = max(12.0, CINEMATIC_BALL_SIZE * 0.86)
        rotation = direction * forward_distance / (math.pi * rolling_diameter) * 360.0
        ball = BallKinematics(
            position=(ball_x, center_y),
            ground_position=(ball_x, shadow_y),
            phase="drible",
            scale=CINEMATIC_BALL_SIZE,
            squash=(1.0, 1.0),
            rotation_degrees=rotation,
            depth=0.0,
        )
        return DribbleKinematics(
            control_position=(control_x, control_y),
            visible_foot_position=(visible_foot_x, visible_foot_y),
            ball=ball,
            touch_phase=touch_phase,
            contact_gap=contact_gap,
        )

    @staticmethod
    def cinematic_ballistic_ground_motion(
        start_y: float,
        initial_velocity_y: float,
        elapsed_seconds: float,
        ground_center_y: float,
        bounce_velocity: float,
        gravity: float = 520.0,
    ) -> tuple[float, str]:
        start_y = min(start_y, ground_center_y)
        discriminant = max(
            0.0,
            initial_velocity_y * initial_velocity_y
            - 2.0 * gravity * (start_y - ground_center_y),
        )
        landing_time = (
            -initial_velocity_y + math.sqrt(discriminant)
        ) / gravity
        if elapsed_seconds < landing_time:
            y = (
                start_y
                + initial_velocity_y * elapsed_seconds
                + 0.5 * gravity * elapsed_seconds * elapsed_seconds
            )
            return min(y, ground_center_y), "chute"

        bounce_elapsed = elapsed_seconds - landing_time
        for rebound in (bounce_velocity, bounce_velocity * 0.34):
            bounce_duration = 2.0 * rebound / gravity
            if bounce_elapsed <= bounce_duration:
                y = (
                    ground_center_y
                    - rebound * bounce_elapsed
                    + 0.5 * gravity * bounce_elapsed * bounce_elapsed
                )
                return min(y, ground_center_y), "quique"
            bounce_elapsed -= bounce_duration
        return ground_center_y, "rolagem"

    def cinematic_ball_for_progress(
        self,
        foot: tuple[float, float],
        target: tuple[float, float],
        direction: int,
        shot_progress: float,
        time_value: float,
        goal_minute: int,
        is_goal: bool = True,
        shot_profile: ShotProfile | None = None,
        ground_y: float | None = None,
        attack_kind: str = "goal",
        release_position: tuple[float, float] | None = None,
        release_rotation_degrees: float | None = None,
    ) -> BallKinematics:
        ground_y = float(ground_y if ground_y is not None else foot[1] + 52.0)
        contact_radius = CINEMATIC_BALL_SIZE * 0.46
        shadow_ground_y = ground_y - 2.0
        dribble_ground_center_y = (
            shadow_ground_y
            - CINEMATIC_BALL_SIZE * CINEMATIC_BALL_GROUND_RADIUS_RATIO
        )
        net_ground_center_y = (
            shadow_ground_y
            - CINEMATIC_NET_BALL_SIZE * CINEMATIC_BALL_GROUND_RADIUS_RATIO
        )
        release = release_position or (
            foot[0] + direction * (contact_radius + 1.7),
            dribble_ground_center_y,
        )
        event_window = GOAL_EVENT_WINDOW_MINUTES if is_goal else CHANCE_EVENT_WINDOW_MINUTES
        kick_time_value = (
            goal_minute - event_window + SHOT_KICK_AT * event_window
        ) / 90.0 * SIMULATION_SECONDS
        rolling_diameter = max(12.0, CINEMATIC_BALL_SIZE * 0.86)
        kick_rotation = (
            release_rotation_degrees
            if release_rotation_degrees is not None
            else direction
            * kick_time_value
            * CINEMATIC_TURF_SPEED
            / (math.pi * rolling_diameter)
            * 360.0
        )
        if shot_profile is None:
            shot_profile = ShotProfile(
                zone="legado",
                target=target,
                entry=target,
                mouth=target,
                bend=direction * math.sin(self.match_seed * 0.013 + goal_minute) * 20.0,
                loft=1.0,
                dip=13.0,
                speed=1.0,
                spin=34.0,
            )
        profile_band = SHOT_PROFILE_BANDS.get(shot_profile.zone, "mid")
        if shot_progress <= SHOT_KICK_AT:
            rotation = (
                release_rotation_degrees
                if release_rotation_degrees is not None
                else direction
                * time_value
                * CINEMATIC_TURF_SPEED
                / (math.pi * rolling_diameter)
                * 360.0
            )
            return BallKinematics(
                position=release,
                ground_position=(release[0], shadow_ground_y),
                phase="drible",
                scale=CINEMATIC_BALL_SIZE,
                squash=(1.0, 1.0),
                rotation_degrees=rotation,
                depth=0.0,
            )

        contact_progress = (
            SHOT_NET_AT
            if is_goal
            else WIDE_CONTACT_VISUAL_AT
            if attack_kind == "wide"
            else CHANCE_CONTACT_VISUAL_AT
        )
        flight_duration = max(0.001, contact_progress - SHOT_KICK_AT)
        flight = clamp((shot_progress - SHOT_KICK_AT) / flight_duration)
        speed_slope = (
            1.05
            if is_goal
            else 0.56
            if attack_kind == "save"
            else 0.44
        )
        speed_ceiling = 1.24 if is_goal else 1.13
        speed_ratio = clamp(
            1.0 + (shot_profile.speed - 1.34) * speed_slope,
            1.0,
            speed_ceiling,
        )
        if is_goal:
            speed_ratio *= 1.07
        elif attack_kind == "save":
            speed_ratio *= {
                "high": 0.98,
                "mid": 1.02,
                "low": 1.06,
            }[profile_band]
        elif attack_kind == "wide":
            speed_ratio *= 0.6295 * {
                "high": 0.94,
                "mid": 0.975,
                "low": 0.998,
            }[profile_band]
        contact_speed_ratio = clamp(
            0.94 + (shot_profile.speed - 1.34) * 0.20,
            0.94,
            0.985,
        )
        if attack_kind == "wide":
            contact_speed_ratio *= 0.57
        inverse_flight = 1.0 - flight
        flight_motion = clamp(
            flight * flight * (3.0 - 2.0 * flight)
            + speed_ratio * flight * inverse_flight * inverse_flight
            - contact_speed_ratio * flight * flight * inverse_flight
        )
        spin_degrees_per_pixel = (
            shot_profile.spin / CINEMATIC_SHOT_SPIN_DISTANCE_DIVISOR
        )
        if attack_kind == "wide":
            spin_degrees_per_pixel *= {
                "high": 0.8675,
                "mid": 0.875,
                "low": 0.88,
            }[profile_band]
        entry = shot_profile.entry if is_goal else target
        delta_x = entry[0] - release[0]
        arc_distance_factor = (
            0.040
            if is_goal
            else 0.050
            if attack_kind == "wide"
            else 0.052
        )
        arc_height = clamp(
            shot_profile.loft + abs(delta_x) * arc_distance_factor,
            20.0,
            58.0,
        )
        if is_goal:
            control_a_factor = 0.408
            control_b_factor = 0.458
        elif attack_kind == "save":
            control_a_factor = 0.43
            control_b_factor = 0.37
        else:
            control_a_factor = 0.18
            control_b_factor = 0.195
        control_a = (
            release[0] + delta_x * control_a_factor,
            release[1] - arc_height,
        )
        control_b = (
            entry[0] - delta_x * control_b_factor + shot_profile.bend,
            entry[1] - arc_height * 0.60 - shot_profile.dip,
        )
        flight_path_table = self.cinematic_net_path_table(
            release,
            control_a,
            control_b,
            entry,
        )
        flight_cumulative = flight_path_table[1]
        flight_table_position = flight_motion * (
            len(flight_cumulative) - 1
        )
        flight_table_index = min(
            len(flight_cumulative) - 2,
            int(math.floor(flight_table_position)),
        )
        flight_table_blend = (
            flight_table_position - flight_table_index
        )
        flight_path_distance = lerp(
            flight_cumulative[flight_table_index],
            flight_cumulative[flight_table_index + 1],
            flight_table_blend,
        )
        if shot_progress < contact_progress:
            position = cubic_bezier_arc_sample(
                flight_path_table,
                flight_motion,
            )
            scale = int(round(lerp(CINEMATIC_BALL_SIZE, CINEMATIC_SHOT_BALL_SIZE, smoothstep(flight))))
            rotation = (
                kick_rotation
                + direction
                * flight_path_distance
                * spin_degrees_per_pixel
            )
            return BallKinematics(
                position=position,
                ground_position=(position[0] - 7.0, ground_y - 2.0),
                phase="chute",
                scale=scale,
                squash=(1.0, 1.0),
                rotation_degrees=rotation,
                depth=0.0,
            )

        if not is_goal:
            elapsed = max(0.0, shot_progress - contact_progress)
            post_contact_seconds = elapsed * event_window / 90.0 * SIMULATION_SECONDS
            profile_band = SHOT_PROFILE_BANDS.get(shot_profile.zone, "mid")
            contact_rotation = (
                kick_rotation
                + direction
                * flight_path_table[2]
                * spin_degrees_per_pixel
            )
            initial_vertical_velocity = {
                "high": -82.0,
                "mid": -38.0,
                "low": 8.0,
            }[profile_band]
            bounce_velocity = {
                "high": 70.0,
                "mid": 50.0,
                "low": 30.0,
            }[profile_band]
            vertical_position, vertical_phase = self.cinematic_ballistic_ground_motion(
                entry[1],
                initial_vertical_velocity,
                post_contact_seconds,
                net_ground_center_y,
                bounce_velocity,
            )
            if attack_kind == "wide":
                field = self.match_field_rect()
                ball_radius = CINEMATIC_SHOT_BALL_SIZE * 0.5
                visible_edge = (
                    field.right - ball_radius - 6.0
                    if direction > 0
                    else field.left + ball_radius + 6.0
                )
                visible_run = max(0.0, direction * (visible_edge - entry[0]))
                initial_horizontal_speed = 360.0 * speed_ratio
                horizontal_deceleration = (
                    initial_horizontal_speed * initial_horizontal_speed
                    / max(2.0, 2.0 * visible_run)
                )
                stop_time = (
                    initial_horizontal_speed / horizontal_deceleration
                    if visible_run > 0.0
                    else 0.0
                )
                movement_time = min(post_contact_seconds, stop_time)
                continuation_x = min(
                    visible_run,
                    initial_horizontal_speed * movement_time
                    - 0.5
                    * horizontal_deceleration
                    * movement_time
                    * movement_time,
                )
                position = (
                    entry[0] + direction * continuation_x,
                    vertical_position,
                )
                phase = vertical_phase
                depth = 0.0
                post_contact_distance = math.dist(entry, position)
                rotation = (
                    contact_rotation
                    + direction
                    * post_contact_distance
                    * spin_degrees_per_pixel
                )
            else:
                parry_speed = 165.0 * speed_ratio
                parry_stop_time = 0.78
                movement_time = min(post_contact_seconds, parry_stop_time)
                parry_deceleration = parry_speed / parry_stop_time
                parry = (
                    parry_speed * movement_time
                    - 0.5
                    * parry_deceleration
                    * movement_time
                    * movement_time
                )
                position = (
                    entry[0] - direction * parry,
                    vertical_position,
                )
                phase = vertical_phase
                depth = 0.0
                post_contact_distance = math.dist(entry, position)
                rotation = (
                    contact_rotation
                    - direction
                    * post_contact_distance
                    * spin_degrees_per_pixel
                )
            scale_progress = smoothstep(clamp(post_contact_seconds / 0.48))
            scale = int(
                round(
                    lerp(
                        CINEMATIC_SHOT_BALL_SIZE,
                        CINEMATIC_NET_BALL_SIZE,
                        scale_progress,
                    )
                )
            )
            return BallKinematics(
                position=position,
                ground_position=(position[0], ground_y - 2.0),
                phase=phase,
                scale=scale,
                squash=(1.0, 1.0),
                rotation_degrees=rotation,
                depth=depth,
            )

        entry_tangent = (entry[0] - control_b[0], entry[1] - control_b[1])
        entry_tangent_length = max(1e-6, math.hypot(*entry_tangent))
        net_target = (
            lerp(entry[0], target[0], SHOT_NET_PENETRATION_RATIO),
            lerp(entry[1], target[1], SHOT_NET_PENETRATION_RATIO),
        )
        target_distance = max(1e-6, math.dist(entry, net_target))
        target_vector = (
            net_target[0] - entry[0],
            net_target[1] - entry[1],
        )
        target_unit = (
            target_vector[0] / target_distance,
            target_vector[1] / target_distance,
        )
        entry_handle = min(target_distance * 0.38, 34.0)
        target_handle = min(
            target_distance * 0.28,
            {
                "high": 20.0,
                "mid": 26.0,
                "low": 21.0,
            }[profile_band],
        )
        impact_control_a = (
            entry[0]
            + entry_tangent[0] / entry_tangent_length * entry_handle,
            entry[1]
            + entry_tangent[1] / entry_tangent_length * entry_handle,
        )
        impact_control_b = (
            net_target[0] - target_unit[0] * target_handle,
            net_target[1] - target_unit[1] * target_handle,
        )
        impact_path_table = self.cinematic_net_path_table(
            entry,
            impact_control_a,
            impact_control_b,
            net_target,
        )
        impact_duration = max(
            0.001,
            SHOT_NET_VISUAL_CONTACT_AT - SHOT_NET_AT,
        )
        impact_progress = clamp(
            (shot_progress - SHOT_NET_AT) / impact_duration
        )
        impact_ease_bias = {
            "high": 0.12,
            "mid": -0.14,
            "low": 0.08,
        }[profile_band]
        impact_distance_fraction = clamp(
            impact_progress
            + impact_ease_bias
            * impact_progress
            * (1.0 - impact_progress)
        )
        if shot_progress <= SHOT_NET_VISUAL_CONTACT_AT:
            position = cubic_bezier_arc_sample(
                impact_path_table,
                impact_distance_fraction,
            )
            distance = (
                flight_path_table[2]
                + impact_path_table[2] * impact_distance_fraction
            )
            return BallKinematics(
                position=position,
                ground_position=(position[0] - 7.0, ground_y - 2.0),
                phase="rede",
                scale=int(
                    round(
                        lerp(
                            CINEMATIC_SHOT_BALL_SIZE,
                            CINEMATIC_NET_BALL_SIZE,
                            smoothstep(impact_progress),
                        )
                    )
                ),
                squash=(1.0, 1.0),
                rotation_degrees=(
                    kick_rotation
                    + direction
                    * distance
                    * spin_degrees_per_pixel
                ),
                depth=smoothstep(impact_progress),
            )

        rest_depth_from_mouth = (
            CINEMATIC_GOAL_BALL_REST_DEPTH
            + float(
                (
                    int(self.match_seed) * 17
                    + int(goal_minute) * 13
                )
                % (CINEMATIC_GOAL_BALL_REST_DEPTH_VARIATION + 1)
            )
        )
        rest_target_x = (
            shot_profile.mouth[0]
            + direction * rest_depth_from_mouth
        )
        rest_depth = max(
            1.0,
            direction * (rest_target_x - net_target[0]),
        )
        settle_progress = clamp(
            (shot_progress - SHOT_NET_VISUAL_CONTACT_AT)
            / SHOT_NET_SETTLE_PROGRESS
        )
        post_contact_seconds = (
            max(0.0, shot_progress - SHOT_NET_VISUAL_CONTACT_AT)
            * event_window
            / 90.0
            * SIMULATION_SECONDS
        )
        penetration_seconds = {
            "high": 0.15,
            "mid": 0.13,
            "low": 0.11,
        }[profile_band]
        event_duration_seconds = (
            event_window / 90.0 * SIMULATION_SECONDS
        )
        impact_duration_seconds = max(
            1e-6,
            impact_duration * event_duration_seconds,
        )
        impact_exit_speed_px_s = (
            impact_path_table[2] * 0.74
            / impact_duration_seconds
        )
        capture_duration = (
            2.0
            * rest_depth
            / max(1.0, impact_exit_speed_px_s)
        )
        capture_progress = clamp(
            post_contact_seconds / max(1e-6, capture_duration)
        )
        capture_progress_sq = capture_progress * capture_progress
        capture_progress_cu = capture_progress_sq * capture_progress
        captured_depth = (
            (
                capture_progress_cu
                - 2.0 * capture_progress_sq
                + capture_progress
            )
            * impact_exit_speed_px_s
            * capture_duration
            + (
                -2.0 * capture_progress_cu
                + 3.0 * capture_progress_sq
            )
            * rest_depth
        )

        drop_start = penetration_seconds + {
            "high": 0.16,
            "mid": 0.10,
            "low": 0.12,
        }[profile_band]
        drop_elapsed = max(0.0, post_contact_seconds - drop_start)
        drop_gravity = {
            "high": 426.0,
            "mid": 484.0,
            "low": 541.0,
        }[profile_band]
        drop_terminal_speed = {
            "high": 170.0,
            "mid": 178.0,
            "low": 156.0,
        }[profile_band]
        terminal_time = drop_terminal_speed / drop_gravity
        if drop_elapsed <= terminal_time:
            drop_distance = (
                0.5
                * drop_gravity
                * drop_elapsed
                * drop_elapsed
            )
        else:
            terminal_distance = (
                0.5
                * drop_gravity
                * terminal_time
                * terminal_time
            )
            drop_distance = (
                terminal_distance
                + drop_terminal_speed
                * (drop_elapsed - terminal_time)
            )
        position = (
            net_target[0] + direction * captured_depth,
            min(net_ground_center_y, net_target[1] + drop_distance),
        )
        distance = (
            flight_path_table[2]
            + impact_path_table[2]
            + captured_depth
            + max(0.0, position[1] - net_target[1])
        )
        scale = CINEMATIC_NET_BALL_SIZE
        return BallKinematics(
            position=position,
            ground_position=(position[0] - 7.0, ground_y - 2.0),
            phase="rede" if settle_progress < 0.72 else "rolagem_rede",
            scale=scale,
            squash=(1.0, 1.0),
            rotation_degrees=(
                kick_rotation
                + direction
                * distance
                * spin_degrees_per_pixel
            ),
            depth=1.0,
        )

    def cinematic_save_variant(self, chance_minute: int, side: str) -> str:
        seed = int(self.match_seed) + int(chance_minute) * 31 + (17 if side == "away" else 0)
        return "stand" if seed % 2 == 0 else "dive"

    def cinematic_chance_target_from_profile(
        self,
        profile: ShotProfile,
        goal_rect: pygame.Rect,
        direction: int,
        kind: str,
        save_variant: str = "",
    ) -> tuple[float, float]:
        band = SHOT_PROFILE_BANDS.get(profile.zone, "mid")
        if kind == "wide":
            outside_x = (
                goal_rect.right + CINEMATIC_WIDE_POST_OFFSET
                if direction > 0
                else goal_rect.left - CINEMATIC_WIDE_POST_OFFSET
            )
            target_y = {
                "high": goal_rect.top - 12.0,
                "mid": goal_rect.centery - 4.0,
                "low": goal_rect.bottom - 24.0,
            }[band]
            return outside_x, target_y
        if kind == "save":
            goal_line_x = self.cinematic_goal_entry_x(goal_rect, direction)
            contact_x = goal_line_x - direction * (CINEMATIC_SHOT_BALL_SIZE * 0.5 + 4.0)
            contact_y = {
                "high": goal_rect.centery - 42.0,
                "mid": goal_rect.centery - 6.0,
                "low": goal_rect.centery + 24.0,
            }[band]
            return contact_x, contact_y
        raise ValueError(f"invalid cinematic chance outcome: {kind}")

    def cinematic_keeper_glove_offset(
        self,
        flip: bool,
        keeper_action: str,
    ) -> tuple[float, float]:
        key = (bool(flip), keeper_action)
        cached = self.keeper_glove_offset_cache.get(key)
        if cached is not None:
            return cached

        # Measured from the authored 16-frame keeper sheets after runtime scale.
        # The directional sheets are independently authored, so their offsets
        # intentionally are not mirrored constants.
        if keeper_action == "stand_save":
            offset = (-55.0, 21.0) if flip else (55.0, 34.0)
        else:
            offset = (-69.0, -26.0) if flip else (94.0, -20.0)
        self.keeper_glove_offset_cache[key] = offset
        return offset

    def cinematic_chance_target(
        self,
        goal_rect: pygame.Rect,
        direction: int,
        chance_minute: int,
        kind: str,
        save_variant: str = "",
    ) -> tuple[float, float]:
        side = "home" if direction > 0 else "away"
        plan = self.cinematic_shot_plan(
            goal_rect,
            chance_minute,
            side,
            kind,
        )
        if kind == "save" and save_variant and save_variant != plan.save_variant:
            return self.cinematic_chance_target_from_profile(
                plan.profile,
                goal_rect,
                direction,
                kind,
                save_variant,
            )
        return plan.target

    def cinematic_poc_scene_state(
        self,
        field: pygame.Rect,
        active_attack: CinematicAttackEvent,
        raw_shot_progress: float,
    ) -> dict[str, object]:
        sequence = self.cinematic_poc_sequence_for_event(active_attack)
        profile = sequence.profile
        if self.poc_sequences is None:
            raise RuntimeError("approved POC 7 runtime contract is unavailable")
        attack_direction = sequence.attack_direction
        event_window = (
            GOAL_EVENT_WINDOW_MINUTES
            if active_attack.is_goal
            else CHANCE_EVENT_WINDOW_MINUTES
        )
        event_seconds = event_window / 90.0 * SIMULATION_SECONDS
        elapsed = self.poc_sequences.event_elapsed(
            raw_shot_progress,
            sequence,
            event_seconds,
        )
        sample = self.poc_sequences.sample(sequence, elapsed)
        viewport = PocViewport.fit(field)
        possession_team = (
            self.home
            if active_attack.side == "home"
            else self.away
        )
        actor_x = self.cinematic_poc_actor_x(
            sequence,
            sample,
            possession_team,
        )
        ball_x = self.cinematic_poc_ball_x(
            sequence,
            sample,
            possession_team,
        )
        poc2_runtime_sample = (
            self.cinematic_poc2_sequence_sample(
                sequence,
                sample,
                possession_team,
            )
            if sample.actor_source == 0
            else None
        )
        ball_y = (
            poc2_runtime_sample.ball.scene_center_y
            if poc2_runtime_sample is not None
            else sample.ball_y
        )
        ball_ground_y = (
            poc2_runtime_sample.ball.scene_ground_y
            if poc2_runtime_sample is not None
            else sample.ball_ground_y
        )
        keeper_x = sample.keeper_x
        keeper_y = sample.keeper_y
        actor_pos = viewport.point(
            actor_x,
            sample.actor_ground_y,
        )
        ball_pos = viewport.point(ball_x, ball_y)
        ball_ground_pos = viewport.point(
            ball_x,
            ball_ground_y,
        )
        previous_sample = self.poc_sequences.previous_sample(
            sequence,
            elapsed,
            1.0 / self.poc_sequences.sample_hz,
        )
        previous_ball_x = self.cinematic_poc_ball_x(
            sequence,
            previous_sample,
            possession_team,
        )
        previous_poc2_sample = (
            self.cinematic_poc2_sequence_sample(
                sequence,
                previous_sample,
                possession_team,
            )
            if previous_sample.actor_source == 0
            else None
        )
        previous_ball_y = (
            previous_poc2_sample.ball.scene_center_y
            if previous_poc2_sample is not None
            else previous_sample.ball_y
        )
        previous_ball_pos = viewport.point(
            previous_ball_x,
            previous_ball_y,
        )
        velocity_seconds = max(
            1.0 / self.poc_sequences.sample_hz,
            elapsed - previous_sample.elapsed,
        )
        ball_velocity = (
            (ball_pos[0] - previous_ball_pos[0])
            / velocity_seconds,
            (ball_pos[1] - previous_ball_pos[1])
            / velocity_seconds,
        )
        impact_sample = self.poc_sequences.sample(
            sequence,
            sequence.impact_seconds,
        )
        impact_ball_x = self.cinematic_poc_ball_x(
            sequence,
            impact_sample,
            possession_team,
        )
        shot_target = viewport.point(
            impact_ball_x,
            impact_sample.ball_y,
        )
        keeper_pos = viewport.point(keeper_x, keeper_y)
        goal_rect = pygame.Rect(
            *viewport.rect(
                sample.goal_x,
                sample.goal_y,
                sample.goal_w,
                sample.goal_h,
            )
        )
        actor_start_x = -150.0 if attack_direction == "right" else 1430.0
        travel_direction = 1.0 if attack_direction == "right" else -1.0
        scroll = (
            abs(sample.actor_x - actor_start_x)
            * travel_direction
        )
        phase_index = max(
            0,
            min(
                len(self.poc_sequences.ball_phase_labels) - 1,
                sample.ball_phase,
            ),
        )
        return {
            "neutral": False,
            "poc_sequence": True,
            "poc_viewport": viewport,
            "poc_contract_sequence": sequence,
            "poc_contract_sample": sample,
            "poc2_dribble_sample": poc2_runtime_sample,
            "poc_elapsed": elapsed,
            "poc_profile": profile,
            "poc_actor_x": actor_x,
            "poc_ball_x": ball_x,
            "poc_keeper_x": keeper_x,
            "poc_keeper_y": keeper_y,
            "poc_scroll": scroll,
            "possession": active_attack.side,
            "active_attack": active_attack,
            "active_goal": (
                (active_attack.minute, active_attack.side)
                if active_attack.is_goal
                else None
            ),
            "attack_kind": active_attack.kind,
            "goal_side": sequence.goal_side,
            "goal_rect": goal_rect,
            "actor_pos": actor_pos,
            "ball_pos": ball_pos,
            "ball_prev_pos": previous_ball_pos,
            "ball_velocity_px_s": ball_velocity,
            "ball_ground_pos": ball_ground_pos,
            "ball_phase": self.poc_sequences.ball_phase_labels[phase_index],
            "ball_depth": clamp(
                sample.ball_trajectory_progress
            ),
            "ball_rotation_degrees": (
                self.cinematic_poc2_ball_rotation(
                    sequence,
                    sample,
                    possession_team,
                )
            ),
            "ball_scale": max(
                1,
                round(
                    POC_BALL_CANVAS_SIZE * viewport.scale
                ),
            ),
            "ball_squash": (1.0, 1.0),
            "keeper_pos": keeper_pos,
            "shot_progress": clamp(raw_shot_progress),
            "raw_shot_progress": raw_shot_progress,
            "net_progress": sample.net_strength,
            "shot_profile_band": profile,
            "shot_target": shot_target,
            "goal_impact_pos": shot_target,
            "rendered_player_kind": (
                "runner"
                if sample.actor_source == 0
                else "kick"
            ),
            "runner_pose": {
                "frame_index": sample.actor_frame,
                "render_frame_index": (
                    self.cinematic_poc_actor_material(
                        sequence,
                        sample,
                        possession_team,
                    )[1]
                ),
                "target_size": (
                    max(
                        1,
                        round(
                            (
                                int(
                                    self.assets.cinematic_poc2_motion[
                                        "canvas_size"
                                    ]
                                )
                                if sample.actor_source == 0
                                else POC_RUNNER_CANVAS_SIZE
                            )
                            * viewport.scale
                        ),
                    ),
                )
                * 2,
            },
        }

    def cinematic_poc_sequence_for_event(
        self,
        active_attack: CinematicAttackEvent,
    ) -> PocSequence:
        if self.poc_sequences is None:
            raise RuntimeError("approved POC 7 runtime contract is unavailable")
        attack_direction = (
            "right"
            if active_attack.side == "home"
            else "left"
        )
        profile = self.poc_sequences.select_profile(
            self.match_seed,
            active_attack.minute,
            active_attack.side,
            active_attack.kind,
        )
        return self.poc_sequences.select_sequence(
            attack_direction=attack_direction,
            profile=profile,
            outcome=active_attack.kind,
            match_seed=self.match_seed,
            event_minute=active_attack.minute,
            side=active_attack.side,
        )

    def cinematic_poc_audio_thresholds(
        self,
        active_attack: CinematicAttackEvent,
    ) -> dict[str, float]:
        sequence = self.cinematic_poc_sequence_for_event(
            active_attack,
        )
        return {
            name: seconds / sequence.impact_seconds
            for name, seconds in sequence.audio_cues
        }

    def cinematic_poc2_cruise_state(
        self,
        field: pygame.Rect,
        possession: str,
        goal_side: str,
        goal_rect: pygame.Rect,
    ) -> dict[str, object]:
        team = self.home if possession == "home" else self.away
        left = possession == "away"
        direction = -1 if left else 1
        viewport = PocViewport.fit(field)
        ground_y = viewport.point(0.0, POC_GROUND_Y)[1]
        lane_x = (
            field.x + CINEMATIC_RUNNER_EDGE_INSET
            if direction > 0
            else field.right - CINEMATIC_RUNNER_EDGE_INSET
        )
        uniform_code = self.assets.cinematic_source_code(team)
        elapsed = max(0.0, self.t)
        sample = self.poc2_dribble.sample(
            uniform_code,
            left,
            elapsed,
            float(lane_x),
            float(ground_y),
            viewport.scale,
        )
        settled = self.simulation_progress() >= 0.985
        if settled:
            lane_x = (
                float(field.centerx)
                - sample.ball.signed_relative_offset_px
                * viewport.scale
            )
            sample = self.poc2_dribble.sample(
                uniform_code,
                left,
                elapsed,
                float(lane_x),
                float(ground_y),
                viewport.scale,
            )
        previous_elapsed = max(0.0, elapsed - 1.0 / FPS)
        previous = self.poc2_dribble.sample(
            uniform_code,
            left,
            previous_elapsed,
            float(lane_x),
            float(ground_y),
            viewport.scale,
        )
        ball_pos = (
            float(field.centerx),
            sample.ball.scene_center_y,
        ) if settled else (
            sample.ball.scene_center_x,
            sample.ball.scene_center_y,
        )
        previous_ball_pos = (
            ball_pos
            if settled
            else (
                previous.ball.scene_center_x,
                previous.ball.scene_center_y,
            )
        )
        sample_dt = max(1e-6, elapsed - previous_elapsed)
        ball_velocity = (
            0.0,
            0.0,
        ) if settled else (
            (ball_pos[0] - previous_ball_pos[0]) / sample_dt,
            (ball_pos[1] - previous_ball_pos[1]) / sample_dt,
        )
        ball_ground_pos = (
            ball_pos[0],
            sample.ball.scene_ground_y - viewport.scale,
        )
        return {
            "neutral": False,
            "poc2_dribble": True,
            "poc2_dribble_sample": sample,
            "poc2_scale": viewport.scale,
            "poc_scroll": (
                self.ground_scroll
                / max(1e-6, viewport.scale)
            ),
            "possession": possession,
            "goal_side": goal_side,
            "goal_rect": goal_rect,
            "actor_pos": (
                sample.player.scene_center_x,
                sample.player.scene_ground_y,
            ),
            "ball_pos": ball_pos,
            "ball_prev_pos": previous_ball_pos,
            "ball_velocity_px_s": ball_velocity,
            "ball_ground_pos": ball_ground_pos,
            "ball_depth": 0.0,
            "ball_rotation_degrees": (
                0.0
                if settled
                else sample.ball.rotation_degrees
            ),
            "ball_phase": "neutro" if settled else "drible",
            "ball_scale": max(
                1,
                round(
                    self.poc2_dribble.metadata.ball_canvas_size_px
                    * viewport.scale
                ),
            ),
            "ball_squash": (1.0, 1.0),
            "shot_progress": 0.0,
            "raw_shot_progress": 0.0,
            "run_speed": 0.0 if settled else 1.0,
            "rendered_player_kind": "stop" if settled else "run",
            "active_goal": None,
            "active_attack": None,
            "attack_kind": "",
            "settled": settled,
        }

    def cinematic_scene_state(
        self,
        field: pygame.Rect,
        pred: Prediction,
        _include_previous_frame: bool = True,
    ) -> dict[str, object]:
        possession = self.cinematic_possession_side(pred)
        neutral = possession == "neutral"
        minute = self.match_minute_float()
        active_attack = self.active_attack_event(pred)
        if self.simulation_progress() >= 1.0:
            active_attack = None
        active_goal = (active_attack.minute, active_attack.side) if active_attack and active_attack.is_goal else None
        goal_minute = active_attack.minute if active_attack else 0
        scoring_side = active_attack.side if active_attack else possession
        goal_side = self.cinematic_goal_side(scoring_side)
        goal_rect = self.cinematic_goal_rect(field, goal_side)
        direction = 1 if scoring_side != "away" else -1
        poc_viewport = PocViewport.fit(field)
        poc_ground_y = poc_viewport.point(
            0.0,
            POC_GROUND_Y,
        )[1]
        shot_progress = 0.0
        raw_shot_progress = 0.0
        if active_attack:
            event_window = GOAL_EVENT_WINDOW_MINUTES if active_attack.is_goal else CHANCE_EVENT_WINDOW_MINUTES
            raw_shot_progress = (minute - (goal_minute - event_window)) / event_window
            shot_progress = clamp(raw_shot_progress)
            return self.cinematic_poc_scene_state(
                field,
                active_attack,
                raw_shot_progress,
            )

        if neutral:
            neutral_progress = smoothstep((self.simulation_progress() - DRAW_NEUTRAL_START_PROGRESS) / DRAW_NEUTRAL_RAMP)
            neutral_reveal = smoothstep((neutral_progress - 0.04) / 0.38)
            ground_y = poc_ground_y
            entry = smoothstep(neutral_progress / 0.52)
            approach = 1.0 - entry
            home_final_x = field.centerx - 158
            away_final_x = field.centerx + 158
            home_pos = (
                lerp(field.centerx - 238, home_final_x, entry),
                ground_y,
            )
            away_pos = (
                lerp(field.centerx + 238, away_final_x, entry),
                ground_y,
            )
            ball_roll = smoothstep(neutral_progress / 0.66)
            ball_start_x = field.centerx - 18.0
            ball_x = lerp(ball_start_x, float(field.centerx), ball_roll)
            ball_y = ground_y - 2.0 - CINEMATIC_BALL_SIZE * CINEMATIC_BALL_GROUND_RADIUS_RATIO
            ball_pos = (
                ball_x,
                ball_y,
            )
            previous_simulation_progress = clamp(max(0.0, self.t - 1.0 / FPS) / SIMULATION_SECONDS)
            previous_neutral_progress = smoothstep(
                (previous_simulation_progress - DRAW_NEUTRAL_START_PROGRESS) / DRAW_NEUTRAL_RAMP
            )
            previous_ball_roll = smoothstep(previous_neutral_progress / 0.66)
            previous_ball_x = lerp(ball_start_x, float(field.centerx), previous_ball_roll)
            ball_prev_pos = (previous_ball_x, ball_y)
            rolling_diameter = max(12.0, CINEMATIC_BALL_SIZE * 0.86)
            return {
                "neutral": True,
                "possession": possession,
                "goal_side": goal_side,
                "goal_rect": goal_rect,
                "ball_pos": ball_pos,
                "home_pos": home_pos,
                "away_pos": away_pos,
                "keeper_pos": None,
                "shot_progress": shot_progress,
                "raw_shot_progress": raw_shot_progress,
                "stride_phase": (
                    self.t
                    / CINEMATIC_POC2_CYCLE_SECONDS
                    * 4.0
                )
                % 4.0,
                "home_stride_phase": (
                    self.t
                    / CINEMATIC_POC2_CYCLE_SECONDS
                    * 4.0
                )
                % 4.0,
                "away_stride_phase": (
                    self.t
                    / CINEMATIC_POC2_CYCLE_SECONDS
                    * 4.0
                    + 2.0
                )
                % 4.0,
                "run_speed": 0.34 + approach * 0.30,
                "neutral_progress": neutral_progress,
                "neutral_reveal": neutral_reveal,
                "ball_prev_pos": ball_prev_pos,
                "ball_velocity_px_s": ((ball_x - previous_ball_x) * FPS, 0.0),
                "ball_ground_pos": (ball_x, ground_y - 2.0),
                "ball_depth": 0.0,
                "ball_rotation_degrees": (ball_x - ball_start_x) / (math.pi * rolling_diameter) * 360.0,
                "ball_phase": "neutro",
                "ball_scale": CINEMATIC_BALL_SIZE,
                "ball_squash": (1.0, 1.0),
                "keeper_phase": 0.0,
                "net_progress": 0.0,
                "shot_profile_band": "",
                "goal_impact_pos": ball_pos,
                "active_goal": active_goal,
                "active_attack": active_attack,
                "attack_kind": active_attack.kind if active_attack else "",
            }

        return self.cinematic_poc2_cruise_state(
            field,
            possession,
            goal_side,
            goal_rect,
        )


    def quadratic_bezier(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        t: float,
    ) -> tuple[float, float]:
        t = clamp(t)
        x = (1 - t) * (1 - t) * a[0] + 2 * (1 - t) * t * b[0] + t * t * c[0]
        y = (1 - t) * (1 - t) * a[1] + 2 * (1 - t) * t * b[1] + t * t * c[1]
        return x, y

    def draw_cinematic_background(self, field: pygame.Rect, pred: Prediction) -> None:
        old_clip = self.screen.get_clip()
        self.screen.set_clip(field)
        motion = self.cinematic_motion_state(pred)
        camera = motion["camera"]
        horizon = 252
        if self.assets.stadium_bg:
            max_offset = max(0, self.assets.stadium_bg.get_width() - field.w)
            back_pan = int(max_offset * clamp(0.24 + camera * 0.24))
            self.screen.blit(self.assets.stadium_bg, field.topleft, pygame.Rect(back_pan, 0, field.w, horizon))
            ground = pygame.Rect(field.x, field.y + horizon, field.w, field.h - horizon)
            pygame.draw.rect(self.screen, (13, 68, 35), ground)
        else:
            pygame.draw.rect(self.screen, (7, 19, 24), field, border_radius=12)
        blend_key = ("cinematic_horizon_blend", field.w)
        blend = self.cinematic_overlay_cache.get(blend_key)
        if blend is None:
            blend = pygame.Surface((field.w, 92), pygame.SRCALPHA)
            for y in range(blend.get_height()):
                t = y / max(1, blend.get_height() - 1)
                alpha = int(82 * (1.0 - abs(t - 0.48) * 1.55))
                alpha = max(0, alpha)
                if y < 42:
                    color = (1, 7, 11, alpha)
                else:
                    color = (14, 66, 34, alpha)
                pygame.draw.line(blend, color, (0, y), (field.w, y))
            self.cinematic_overlay_cache[blend_key] = blend
        self.screen.blit(blend, (field.x, field.y + horizon - 44))
        self.draw_cinematic_turf_layers(field, motion)
        vignette_key = ("cinematic_vignette", field.w, field.h)
        vignette = self.cinematic_overlay_cache.get(vignette_key)
        if vignette is None:
            vignette = pygame.Surface(field.size, pygame.SRCALPHA)
            pygame.draw.rect(vignette, (0, 0, 0, 28), vignette.get_rect(), border_radius=12)
            pygame.draw.rect(vignette, (0, 0, 0, 80), (0, 0, field.w, 42))
            self.cinematic_overlay_cache[vignette_key] = vignette
        self.screen.blit(vignette, field.topleft)
        self.screen.set_clip(old_clip)

    def cached_turf_tile(self, image: pygame.Surface, size: tuple[int, int], alpha: int) -> pygame.Surface:
        width, height = size
        scaled_h = max(1, height)
        scaled_w = max(int(width * 1.32), int(image.get_width() * scaled_h / max(1, image.get_height())))
        cache_key = (id(image), scaled_w, scaled_h, alpha)
        tile = self.turf_tile_cache.get(cache_key)
        if tile is None:
            tile = pygame.transform.smoothscale(image, (scaled_w, scaled_h)).convert_alpha()
            tile.set_alpha(alpha)
            self.turf_tile_cache[cache_key] = tile
        return tile

    def blit_tiled_tile(self, tile: pygame.Surface, dest: pygame.Rect, offset: float) -> None:
        old_clip = self.screen.get_clip()
        self.screen.set_clip(old_clip.clip(dest) if old_clip else dest)
        period = tile.get_width()
        start_x = -int(offset % period)
        x = start_x
        while x < dest.w:
            self.screen.blit(tile, (dest.x + x, dest.y))
            x += period
        self.screen.set_clip(old_clip)

    def draw_tiled_surface(self, image: pygame.Surface, dest: pygame.Rect, offset: float, alpha: int) -> None:
        if dest.w <= 0 or dest.h <= 0:
            return
        self.blit_tiled_tile(self.cached_turf_tile(image, dest.size, alpha), dest, offset)

    def draw_tiled_surface_gradient(self, image: pygame.Surface, dest: pygame.Rect, offset: float, max_alpha: int, power: float = 1.65) -> None:
        if dest.w <= 0 or dest.h <= 0:
            return
        tile = self.cached_gradient_turf_tile(image, dest.size, max_alpha, power)
        self.blit_tiled_tile(tile, dest, offset)

    def cached_gradient_turf_tile(self, image: pygame.Surface, size: tuple[int, int], max_alpha: int, power: float) -> pygame.Surface:
        width, height = size
        power_key = int(power * 1000)
        scaled_h = max(1, height)
        scaled_w = max(int(width * 1.32), int(image.get_width() * scaled_h / max(1, image.get_height())))
        tile_key = (id(image), scaled_w, scaled_h, max_alpha, power_key)
        tile = self.gradient_tile_cache.get(tile_key)
        if tile is None:
            tile = pygame.transform.smoothscale(image, (scaled_w, scaled_h)).convert_alpha()
            mask_key = (scaled_w, scaled_h, max_alpha, power_key)
            mask = self.gradient_mask_cache.get(mask_key)
            if mask is None:
                mask = pygame.Surface((scaled_w, scaled_h), pygame.SRCALPHA)
                for y in range(scaled_h):
                    amount = (y / max(1, scaled_h - 1)) ** power
                    alpha = int(max_alpha * amount)
                    pygame.draw.line(mask, (255, 255, 255, alpha), (0, y), (scaled_w, y))
                self.gradient_mask_cache[mask_key] = mask
            tile.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            self.gradient_tile_cache[tile_key] = tile
        return tile

    def draw_cinematic_turf_layers(self, field: pygame.Rect, motion: dict[str, float]) -> None:
        ground_top = field.y + 252
        if not self.assets.turf_mid_strip and not self.assets.turf_near_strip:
            return
        scroll = motion["ground_scroll"]
        ground = pygame.Rect(field.x, ground_top, field.w, field.bottom - ground_top)
        if self.assets.turf_mid_strip:
            self.draw_tiled_surface(
                self.assets.turf_mid_strip,
                ground,
                scroll * 0.30,
                212,
            )
        if self.assets.turf_near_strip:
            self.draw_tiled_surface_gradient(
                self.assets.turf_near_strip,
                ground,
                scroll * CINEMATIC_TURF_FOREGROUND_PARALLAX,
                104,
            )
        shade_key = ("cinematic_ground_shade", ground.w, ground.h)
        bottom_shade = self.cinematic_overlay_cache.get(shade_key)
        if bottom_shade is None:
            bottom_shade = pygame.Surface(ground.size, pygame.SRCALPHA)
            for y in range(ground.h):
                alpha = int(12 + 44 * (y / max(1, ground.h - 1)) ** 1.7)
                pygame.draw.line(bottom_shade, (0, 0, 0, alpha), (0, y), (ground.w, y))
            self.cinematic_overlay_cache[shade_key] = bottom_shade
        self.screen.blit(bottom_shade, ground.topleft)

    def draw_cinematic_poc_sequence(
        self,
        field: pygame.Rect,
        state: dict[str, object],
    ) -> None:
        sequence = state.get("poc_contract_sequence")
        sample = state.get("poc_contract_sample")
        viewport = state.get("poc_viewport")
        if (
            not isinstance(sequence, PocSequence)
            or not isinstance(sample, PocSequenceSample)
            or not isinstance(viewport, PocViewport)
        ):
            raise RuntimeError("invalid approved POC 7 runtime state")

        self.draw_cinematic_poc_background(
            field,
            viewport,
            float(state["poc_scroll"]),
        )
        self.draw_cinematic_poc_goal_layer(
            state,
            front=False,
        )
        if self.cinematic_poc_ball_shadow_is_visible(
            sequence,
            sample,
        ):
            poc2_sample = state.get("poc2_dribble_sample")
            if (
                sample.actor_source == 0
                and isinstance(poc2_sample, Poc2DribbleSample)
            ):
                self.draw_cinematic_poc_shadow(
                    viewport,
                    float(state["poc_ball_x"]),
                    poc2_sample.ball.scene_ground_y - 1.0,
                    poc2_sample.ball.canvas_diameter_px * 1.1,
                    max(
                        8.0,
                        poc2_sample.ball.canvas_diameter_px * 0.22,
                    ),
                    64.0,
                )
            else:
                self.draw_cinematic_poc_shadow(
                    viewport,
                    sample.ball_shadow_x
                    + float(state["poc_ball_x"])
                    - sample.ball_x,
                    sample.ball_shadow_y,
                    sample.ball_shadow_w,
                    sample.ball_shadow_h,
                    sample.ball_shadow_alpha,
                )
        self.draw_cinematic_poc_actor(state)
        if sample.ball_after_keeper:
            self.draw_cinematic_poc_ball(state)
            self.draw_cinematic_poc_keeper(state)
        else:
            self.draw_cinematic_poc_keeper(state)
            self.draw_cinematic_poc_ball(state)
        self.draw_cinematic_poc_goal_layer(
            state,
            front=True,
        )

    def draw_cinematic_poc_background(
        self,
        field: pygame.Rect,
        viewport: PocViewport,
        scroll: float,
    ) -> None:
        pygame.draw.rect(self.screen, BG, field)
        content_left = round(viewport.content_x)
        content_width = max(1, round(1280.0 * viewport.scale))
        content_bottom = round(
            viewport.point(0.0, 760.0)[1]
        )
        stadium_top = round(
            viewport.point(0.0, POC_HEADER_HEIGHT)[1]
        )
        stadium_bottom = round(
            viewport.point(0.0, 354.0)[1]
        )
        stadium = pygame.Rect(
            content_left,
            stadium_top,
            content_width,
            max(1, stadium_bottom - stadium_top),
        )
        if self.assets.stadium_bg is None:
            pygame.draw.rect(self.screen, (7, 19, 24), stadium)
        else:
            source = self.assets.stadium_bg
            canonical_key = (
                "poc7_stadium_canonical",
                id(source),
            )
            canonical = self.cinematic_overlay_cache.get(
                canonical_key
            )
            if canonical is None:
                canonical = pygame.transform.smoothscale(
                    source,
                    (1280, 278),
                ).convert()
                self.cinematic_overlay_cache[
                    canonical_key
                ] = canonical
            layer_size = (
                max(1, round(1280.0 * viewport.scale)),
                max(1, round(278.0 * viewport.scale)),
            )
            layer = self.cached_smoothscale(
                canonical,
                layer_size,
            )
            offset = scroll * 0.15 * viewport.scale
            start_x = stadium.x - int(offset % layer.get_width())
            old_clip = self.screen.get_clip()
            self.screen.set_clip(
                old_clip.clip(stadium)
                if old_clip
                else stadium
            )
            x = start_x
            while x < stadium.right:
                self.screen.blit(layer, (x, stadium.y))
                x += layer.get_width()
            self.screen.set_clip(old_clip)

        horizon_height = max(1, round(118.0 * viewport.scale))
        horizon_key = (
            "poc7_horizon",
            content_width,
            horizon_height,
        )
        horizon = self.cinematic_overlay_cache.get(horizon_key)
        if horizon is None:
            horizon = pygame.Surface(
                (content_width, horizon_height),
                pygame.SRCALPHA,
            )
            for y in range(horizon_height):
                amount = y / max(1, horizon_height - 1)
                alpha = int(118 * (1.0 - amount))
                pygame.draw.line(
                    horizon,
                    (3, 13, 17, alpha),
                    (0, y),
                    (content_width, y),
                )
            self.cinematic_overlay_cache[horizon_key] = horizon
        horizon_y = round(
            viewport.point(0.0, 354.0 - 76.0)[1]
        )
        self.screen.blit(horizon, (content_left, horizon_y))

        ground = pygame.Rect(
            content_left,
            stadium_bottom,
            content_width,
            max(1, content_bottom - stadium_bottom),
        )
        pygame.draw.rect(self.screen, (15, 83, 43), ground)
        if self.assets.turf_mid_strip is not None:
            self.draw_tiled_surface(
                self.assets.turf_mid_strip,
                ground,
                scroll * 0.60 * viewport.scale,
                228,
            )
        if self.assets.turf_near_strip is not None:
            self.draw_tiled_surface_gradient(
                self.assets.turf_near_strip,
                ground,
                scroll * viewport.scale,
                118,
            )

        shade_key = (
            "poc7_field_shade",
            ground.w,
            ground.h,
        )
        field_shade = self.cinematic_overlay_cache.get(shade_key)
        if field_shade is None:
            field_shade = pygame.Surface(
                ground.size,
                pygame.SRCALPHA,
            )
            for y in range(ground.h):
                amount = y / max(1, ground.h - 1)
                pygame.draw.line(
                    field_shade,
                    (0, 0, 0, int(8 + 40 * amount * amount)),
                    (0, y),
                    (ground.w, y),
                )
            self.cinematic_overlay_cache[shade_key] = field_shade
        self.screen.blit(field_shade, ground.topleft)

    def draw_cinematic_poc_shadow(
        self,
        viewport: PocViewport,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        alpha: float,
    ) -> None:
        if width <= 0.0 or height <= 0.0 or alpha <= 0.0:
            return
        center = viewport.point(center_x, center_y)
        size = viewport.size(width, height)
        shadow = pygame.Rect(0, 0, *size)
        shadow.center = (
            round(center[0]),
            round(center[1]),
        )
        self.draw_soft_shadow(
            shadow,
            max(0, min(255, round(alpha))),
        )

    @staticmethod
    def cinematic_poc2_run_window(
        sequence: PocSequence,
    ) -> tuple[float, float]:
        visible_run_starts = [
            segment.start_seconds
            for segment in sequence.actor_segments
            if segment.source == 0 and segment.visible
        ]
        kick_starts = [
            segment.start_seconds
            for segment in sequence.actor_segments
            if segment.source == 1
        ]
        if not visible_run_starts or not kick_starts:
            raise RuntimeError(
                f"incomplete POC 2 handoff window: {sequence.key}"
            )
        return min(visible_run_starts), min(kick_starts)

    def cinematic_poc2_sequence_sample(
        self,
        sequence: PocSequence,
        sample: PocSequenceSample,
        team: TeamProfile,
    ) -> Poc2DribbleSample:
        run_start, handoff = self.cinematic_poc2_run_window(
            sequence
        )
        run_elapsed = clamp(
            sample.elapsed - run_start,
            0.0,
            handoff - run_start,
        )
        uniform_code = self.assets.cinematic_source_code(team)
        return self.poc2_dribble.sample(
            uniform_code,
            sequence.attack_direction == "left",
            run_elapsed,
            sample.actor_x,
            sample.actor_ground_y,
            1.0,
        )

    def cinematic_poc2_handoff_ball_delta(
        self,
        sequence: PocSequence,
        team: TeamProfile,
    ) -> float:
        if self.poc_sequences is None:
            return 0.0
        _run_start, handoff = self.cinematic_poc2_run_window(
            sequence
        )
        before = self.poc_sequences.sample(
            sequence,
            max(0.0, handoff - 1.0 / self.poc_sequences.sample_hz),
        )
        at_handoff = self.poc_sequences.sample(
            sequence,
            handoff,
        )
        approved = self.cinematic_poc2_sequence_sample(
            sequence,
            before,
            team,
        )
        return (
            approved.ball.scene_center_x
            - at_handoff.ball_x
        )

    def cinematic_poc2_ball_rotation(
        self,
        sequence: PocSequence,
        sample: PocSequenceSample,
        team: TeamProfile,
    ) -> float:
        if sample.actor_source == 0:
            return self.cinematic_poc2_sequence_sample(
                sequence,
                sample,
                team,
            ).ball.rotation_degrees
        if self.poc_sequences is None:
            return sample.ball_rotation
        _run_start, handoff = self.cinematic_poc2_run_window(
            sequence
        )
        handoff_sample = self.poc_sequences.sample(
            sequence,
            handoff,
        )
        before = self.poc_sequences.sample(
            sequence,
            max(0.0, handoff - 1.0 / self.poc_sequences.sample_hz),
        )
        approved_rotation = self.cinematic_poc2_sequence_sample(
            sequence,
            before,
            team,
        ).ball.rotation_degrees
        rotation_delta = (
            approved_rotation
            - handoff_sample.ball_rotation
        )
        return sample.ball_rotation + rotation_delta

    def cinematic_poc_actor_material(
        self,
        sequence: PocSequence,
        sample: PocSequenceSample,
        team: TeamProfile,
    ) -> tuple[pygame.Surface, int, dict[str, object], float]:
        left = sequence.attack_direction == "left"
        direction_name = "left" if left else "right"
        uniform_code = self.assets.cinematic_source_code(team)
        actor_elapsed = sample.elapsed
        if sample.actor_source == 1:
            _run_start, handoff = self.cinematic_poc2_run_window(
                sequence
            )
            actor_elapsed = max(actor_elapsed, handoff)
        poc_frame = (
            self.poc_sequences.actor_frame_position(
                sequence,
                actor_elapsed,
            )
            if self.poc_sequences is not None
            else float(sample.actor_frame)
        )
        if sample.actor_source == 0:
            frames = self.assets.cinematic_poc2_frames_for_uniform(
                uniform_code,
                left=left,
            )
            frame_index = self.cinematic_poc2_sequence_sample(
                sequence,
                sample,
                team,
            ).frame_index
            uniforms = self.assets.cinematic_poc2_motion["uniforms"]
            uniform = uniforms.get(
                uniform_code,
                uniforms["gold"],
            )
            direction_payload = uniform["directions"][direction_name]
            metadata = direction_payload["frames"][frame_index]
            return frames[frame_index], frame_index, metadata, 1.0
        else:
            uniforms = self.assets.cinematic_runner_motion["uniforms"]
            uniform = uniforms.get(
                uniform_code,
                uniforms["gold"],
            )
            direction_payload = uniform["directions"][direction_name]
            frames = self.assets.cinematic_kick_frames_for_uniform(
                uniform_code,
                left=left,
            )
            contact_frame = int(
                direction_payload["kick_contact_frame"]
            )
            poc_frame = max(0.0, min(7.0, poc_frame))
            if poc_frame <= 4:
                frame_position = (
                    poc_frame / 4.0 * contact_frame
                )
            else:
                frame_position = (
                    contact_frame
                    + (poc_frame - 4) / 3.0
                    * (len(frames) - 1 - contact_frame)
                )
            frame_index = max(
                0,
                min(
                    len(frames) - 1,
                    int(math.floor(frame_position + 1e-3)),
                ),
            )
            metadata = direction_payload["kick_frames"][frame_index]
        reference_height = self.cinematic_runner_reference_height(
            uniform_code,
            -1 if left else 1,
        )
        actor_scale = (
            POC_APPROVED_REFERENCE_VISIBLE_HEIGHT
            / max(1.0, reference_height)
        )
        return frames[frame_index], frame_index, metadata, actor_scale

    def cinematic_poc_actor_x(
        self,
        sequence: PocSequence,
        sample: PocSequenceSample,
        team: TeamProfile,
    ) -> float:
        if sample.actor_source == 0:
            return sample.actor_x
        _frame, _index, metadata, actor_scale = (
            self.cinematic_poc_actor_material(
                sequence,
                sample,
                team,
            )
        )
        direction = (
            1.0
            if sequence.attack_direction == "right"
            else -1.0
        )
        direction_name = (
            "right"
            if direction > 0
            else "left"
        )
        uniform_code = self.assets.cinematic_source_code(team)
        uniforms = self.assets.cinematic_runner_motion["uniforms"]
        direction_payload = uniforms.get(
            uniform_code,
            uniforms["gold"],
        )["directions"][direction_name]
        correction = float(
            direction_payload["kick_contact_root_correction_px"]
        )
        corridor = float(metadata["ball_corridor_offset_px"])
        clearance = float(metadata["ball_clearance_offset_px"])
        available = max(0.0, corridor - clearance)
        poc_frame = max(
            0.0,
            min(
                7.0,
                (
                    self.poc_sequences.actor_frame_position(
                        sequence,
                        sample.elapsed,
                    )
                    if self.poc_sequences is not None
                    else float(sample.actor_frame)
                ),
            ),
        )
        if poc_frame <= 4.0:
            weight = smoothstep((poc_frame - 1.0) / 3.0)
        else:
            weight = 1.0 - smoothstep((poc_frame - 4.0) / 3.0)
        return (
            sample.actor_x
            + direction
            * min(correction, available)
            * actor_scale
            * weight
        )

    def cinematic_poc_corridor_offset(
        self,
        sequence: PocSequence,
        sample: PocSequenceSample,
        team: TeamProfile,
    ) -> float:
        if self.poc_sequences is None:
            return 0.0
        direction_name = sequence.attack_direction
        uniform_code = self.assets.cinematic_source_code(team)
        uniforms = self.assets.cinematic_runner_motion["uniforms"]
        direction_payload = uniforms.get(
            uniform_code,
            uniforms["gold"],
        )["directions"][direction_name]
        poc_frame = self.poc_sequences.actor_frame_position(
            sequence,
            sample.elapsed,
        )
        if sample.actor_source == 0:
            metadata_frames = direction_payload["frames"]
            frame_position = poc_frame % 8.0
            corridor_key = "ball_approach_corridor_offset_px"
            cyclic = True
        else:
            metadata_frames = direction_payload["kick_frames"]
            contact_frame = int(
                direction_payload["kick_contact_frame"]
            )
            clamped_frame = max(0.0, min(7.0, poc_frame))
            if clamped_frame <= 4.0:
                frame_position = (
                    clamped_frame / 4.0 * contact_frame
                )
            else:
                frame_position = (
                    contact_frame
                    + (clamped_frame - 4.0)
                    / 3.0
                    * (len(metadata_frames) - 1 - contact_frame)
                )
            corridor_key = "ball_corridor_offset_px"
            cyclic = False

        position_floor = math.floor(frame_position + 1e-6)
        local = clamp(frame_position - position_floor)
        if cyclic:
            current_index = int(position_floor) % len(metadata_frames)
            previous_index = (current_index - 1) % len(metadata_frames)
            next_index = (current_index + 1) % len(metadata_frames)
        else:
            current_index = max(
                0,
                min(len(metadata_frames) - 1, int(position_floor)),
            )
            previous_index = max(0, current_index - 1)
            next_index = min(
                len(metadata_frames) - 1,
                current_index + 1,
            )

        previous = float(
            metadata_frames[previous_index][corridor_key]
        )
        current = float(
            metadata_frames[current_index][corridor_key]
        )
        following = float(
            metadata_frames[next_index][corridor_key]
        )
        if local < 0.5:
            return lerp(
                max(previous, current),
                current,
                smoothstep(local / 0.5),
            )
        return lerp(
            current,
            max(current, following),
            smoothstep((local - 0.5) / 0.5),
        )

    def cinematic_poc_pre_release_ball_target_x(
        self,
        sequence: PocSequence,
        sample: PocSequenceSample,
        team: TeamProfile,
        *,
        safety_offset: float = 3.0,
    ) -> float:
        if not sample.actor_visible:
            return sample.ball_x
        if sample.actor_source == 0:
            return self.cinematic_poc2_sequence_sample(
                sequence,
                sample,
                team,
            ).ball.scene_center_x
        _frame, _index, _metadata, actor_scale = (
            self.cinematic_poc_actor_material(
                sequence,
                sample,
                team,
            )
        )
        offset = (
            self.cinematic_poc_corridor_offset(
                sequence,
                sample,
                team,
            )
            + safety_offset
        )
        actor_x = self.cinematic_poc_actor_x(
            sequence,
            sample,
            team,
        )
        direction = (
            1.0
            if sequence.attack_direction == "right"
            else -1.0
        )
        raw = direction * sample.ball_x
        bound = direction * actor_x + offset * actor_scale
        softness = 2.0
        adjusted = 0.5 * (
            raw
            + bound
            + math.sqrt(
                (raw - bound) * (raw - bound)
                + softness * softness
            )
        )
        return direction * adjusted

    def cinematic_poc_pre_release_ball_x(
        self,
        sequence: PocSequence,
        sample: PocSequenceSample,
        team: TeamProfile,
    ) -> float:
        if self.poc_sequences is None:
            return sample.ball_x
        if sample.actor_source == 0:
            return self.cinematic_poc2_sequence_sample(
                sequence,
                sample,
                team,
            ).ball.scene_center_x
        handoff_seconds = next(
            (
                segment.start_seconds
                for segment in sequence.actor_segments
                if segment.source == 1
            ),
            sequence.release_seconds,
        )
        blend_seconds = 0.08
        blend_start = max(
            0.0,
            handoff_seconds - blend_seconds,
        )
        target = self.cinematic_poc_pre_release_ball_target_x(
            sequence,
            sample,
            team,
        )
        if sample.elapsed <= blend_start:
            return target
        handoff_delta = self.cinematic_poc2_handoff_ball_delta(
            sequence,
            team,
        )
        continued = sample.ball_x + handoff_delta
        if sample.elapsed <= handoff_seconds:
            return continued
        corridor_lock_seconds = min(
            sequence.release_seconds,
            handoff_seconds + 0.16,
        )
        if sample.elapsed >= corridor_lock_seconds:
            return target
        return lerp(
            continued,
            target,
            smoothstep(
                (sample.elapsed - handoff_seconds)
                / max(
                    1e-6,
                    corridor_lock_seconds - handoff_seconds,
                )
            ),
        )

    def cinematic_poc_ball_x(
        self,
        sequence: PocSequence,
        sample: PocSequenceSample,
        team: TeamProfile,
    ) -> float:
        if self.poc_sequences is None:
            return sample.ball_x
        if sample.elapsed <= sequence.release_seconds:
            return self.cinematic_poc_pre_release_ball_x(
                sequence,
                sample,
                team,
            )
        release_sample = self.poc_sequences.sample(
            sequence,
            sequence.release_seconds,
        )
        adjusted_release = self.cinematic_poc_pre_release_ball_x(
            sequence,
            release_sample,
            team,
        )
        release_delta = adjusted_release - release_sample.ball_x
        flight_progress = clamp(
            (sample.elapsed - sequence.release_seconds)
            / max(
                1e-6,
                sequence.impact_seconds - sequence.release_seconds,
            )
        )
        candidate = (
            sample.ball_x
            + release_delta * (1.0 - smoothstep(flight_progress))
        )
        if not sample.actor_visible:
            return candidate
        target = self.cinematic_poc_pre_release_ball_target_x(
            sequence,
            sample,
            team,
        )
        direction = (
            1.0
            if sequence.attack_direction == "right"
            else -1.0
        )
        raw = direction * candidate
        bound = direction * target
        if sample.elapsed > sequence.release_seconds:
            _frame, _index, metadata, actor_scale = (
                self.cinematic_poc_actor_material(
                    sequence,
                    sample,
                    team,
                )
            )
            visible_bbox = metadata.get("visible_bbox")
            if (
                isinstance(visible_bbox, list)
                and len(visible_bbox) == 4
            ):
                bbox_x, _bbox_y, bbox_width, _bbox_height = (
                    float(value) for value in visible_bbox
                )
                forward_extent = (
                    bbox_x + bbox_width - POC_RUNNER_ROOT[0]
                    if direction > 0.0
                    else POC_RUNNER_ROOT[0] - bbox_x
                )
                actor_x = self.cinematic_poc_actor_x(
                    sequence,
                    sample,
                    team,
                )
                silhouette_bound = (
                    direction * actor_x
                    + max(0.0, forward_extent) * actor_scale
                    + POC_BALL_CANVAS_SIZE * 0.5
                    + 3.0
                )
                impulse_weight = smoothstep(
                    (sample.elapsed - sequence.release_seconds)
                    / (1.0 / 120.0)
                )
                bound = max(
                    bound,
                    lerp(
                        bound,
                        silhouette_bound,
                        impulse_weight,
                    ),
                )
        softness = 2.0
        return direction * 0.5 * (
            raw
            + bound
            + math.sqrt(
                (raw - bound) * (raw - bound)
                + softness * softness
            )
        )

    def draw_cinematic_poc_actor(
        self,
        state: dict[str, object],
    ) -> None:
        sequence = state["poc_contract_sequence"]
        sample = state["poc_contract_sample"]
        viewport = state["poc_viewport"]
        if (
            not isinstance(sequence, PocSequence)
            or not isinstance(sample, PocSequenceSample)
            or not isinstance(viewport, PocViewport)
            or not sample.actor_visible
        ):
            return
        team = (
            self.home
            if str(state["possession"]) == "home"
            else self.away
        )
        frame, _frame_index, _metadata, actor_scale = (
            self.cinematic_poc_actor_material(
                sequence,
                sample,
                team,
            )
        )
        actor_scale *= viewport.scale
        recovery_start = sequence.release_seconds + 0.30
        recovering = (
            sample.actor_source == 1
            and sample.elapsed >= recovery_start
        )
        if recovering:
            stop_frames = (
                self.assets.cinematic_stops_left[team.code]
                if sequence.attack_direction == "left"
                else self.assets.cinematic_stops[team.code]
            )
            recovery_progress = smoothstep(
                (sample.elapsed - recovery_start) / 0.34
            )
            recovery_index = min(
                len(stop_frames) - 1,
                round(
                    recovery_progress
                    * (len(stop_frames) - 1)
                ),
            )
            frame = stop_frames[recovery_index]
        source_canvas_size = (
            int(self.assets.cinematic_poc2_motion["canvas_size"])
            if sample.actor_source == 0
            else POC_RUNNER_CANVAS_SIZE
        )
        target = (
            max(1, round(source_canvas_size * actor_scale)),
            max(1, round(source_canvas_size * actor_scale)),
        )
        actor_x = float(state["poc_actor_x"])
        root = viewport.point(
            actor_x,
            sample.actor_ground_y,
        )
        rendered_actor = self.cached_smoothscale(frame, target)
        visible_actor = self.visible_bbox(rendered_actor)
        if recovering:
            rect = pygame.Rect(
                round(root[0] - visible_actor.centerx),
                round(root[1] - visible_actor.bottom),
                *target,
            )
        elif sample.actor_source == 0:
            canvas_scale = actor_scale
            canvas_center_x = (
                float(
                    self.assets.cinematic_poc2_motion[
                        "canvas_size"
                    ]
                )
                * 0.5
            )
            canvas_ground_y = float(
                self.assets.cinematic_poc2_motion[
                    "canvas_ground_y"
                ]
            )
            rect = pygame.Rect(
                round(root[0] - canvas_center_x * canvas_scale),
                round(root[1] - canvas_ground_y * canvas_scale),
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
            submersion = (
                rect.top
                + visible_actor.bottom
                - round(root[1])
            )
            if submersion > 0:
                rect.y -= submersion
        if sample.actor_source == 0:
            runtime_sample = self.cinematic_poc2_sequence_sample(
                sequence,
                sample,
                team,
            )
            self.draw_cinematic_poc_shadow(
                viewport,
                actor_x,
                sample.actor_ground_y - 1.0,
                105.0 if runtime_sample.flight else 122.0,
                15.0,
                92.0 if runtime_sample.flight else 124.0,
            )
        else:
            self.draw_cinematic_poc_shadow(
                viewport,
                sample.actor_shadow_x
                + (actor_x - sample.actor_x),
                sample.actor_shadow_y,
                sample.actor_shadow_w,
                sample.actor_shadow_h,
                sample.actor_shadow_alpha,
            )
        self.screen.blit(
            rendered_actor,
            rect,
        )

    def cinematic_poc_composite_goal_layer(
        self,
        *,
        cache_key: tuple[object, ...],
        canvas_size: tuple[int, int],
        patch: pygame.Surface,
        patch_rect: tuple[int, int, int, int],
        base: pygame.Surface | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> pygame.Surface:
        resolved_key = cache_key
        resolved_target: tuple[int, int] | None = None
        if target_size is not None:
            resolved_target = (
                max(1, int(target_size[0])),
                max(1, int(target_size[1])),
            )
            resolved_key = (
                "scaled_goal_composite",
                *cache_key,
                *resolved_target,
            )
        cached = self.poc_goal_composite_cache.get(resolved_key)
        if cached is not None:
            self.poc_goal_composite_cache.move_to_end(resolved_key)
            return cached
        if base is None:
            composite = pygame.Surface(
                canvas_size,
                pygame.SRCALPHA,
            )
        else:
            if base.get_size() != canvas_size:
                raise RuntimeError(
                    "approved POC goal layer size mismatch"
                )
            composite = base.copy()
        destination = pygame.Rect(*patch_rect)
        material = patch
        if material.get_size() != destination.size:
            material = self.cached_smoothscale(
                material,
                destination.size,
            )
        composite.blit(material, destination)
        if resolved_target is not None:
            # Dynamic composites must not enter SurfaceCache: its strong
            # source keys would retain evicted native-size net surfaces.
            composite = pygame.transform.smoothscale(
                composite,
                resolved_target,
            ).convert_alpha()
        self.poc_goal_composite_cache[resolved_key] = composite
        while (
            len(self.poc_goal_composite_cache)
            > CINEMATIC_GOAL_COMPOSITE_CACHE_LIMIT
        ):
            self.poc_goal_composite_cache.popitem(last=False)
        return composite

    @staticmethod
    def cinematic_poc_net_keyframe(
        sequence: PocSequence,
        sample: PocSequenceSample,
    ) -> PocNetKeyframe | None:
        if not sequence.net_keyframes:
            return None
        local = max(
            0.0,
            sample.elapsed - sequence.impact_seconds,
        )
        index = max(
            0,
            min(
                len(sequence.net_keyframes) - 1,
                round(local * PocSequenceBank.NET_KEYFRAME_HZ),
            ),
        )
        # These peak atlas frames collapse the projected mesh into
        # horizontal streaks at game scale. Adjacent authored states keep
        # the same impulse while preserving continuous net cells.
        index = CINEMATIC_POC_NET_SAFE_FRAME_REMAP.get(index, index)
        return sequence.net_keyframes[index]

    def cinematic_poc_ball_shadow_is_visible(
        self,
        sequence: PocSequence,
        sample: PocSequenceSample,
    ) -> bool:
        if not sample.ball_visible:
            return False
        phase = (
            self.poc_sequences.ball_phase_labels[sample.ball_phase]
            if self.poc_sequences is not None
            else ""
        )
        return (
            sequence.outcome != "goal"
            or sample.elapsed < sequence.line_seconds
            or phase == "goal_settled"
        )

    def cinematic_poc_ball_layers(
        self,
        sequence: PocSequence,
        sample: PocSequenceSample,
        elapsed: float,
        team: TeamProfile,
    ) -> tuple[tuple[PocSequenceSample, int], ...]:
        del team
        layers: list[tuple[PocSequenceSample, int]] = []
        phase = (
            self.poc_sequences.ball_phase_labels[sample.ball_phase]
            if self.poc_sequences is not None
            else ""
        )
        if (
            phase == "shot"
            and self.poc_sequences is not None
            and elapsed - sequence.release_seconds >= 0.075
        ):
            for offset_seconds, alpha in (
                (1.0 / 60.0, 24),
                (1.0 / 120.0, 64),
            ):
                trail_sample = self.poc_sequences.sample(
                    sequence,
                    max(
                        sequence.release_seconds,
                        elapsed - offset_seconds,
                    ),
                )
                trail_phase = self.poc_sequences.ball_phase_labels[
                    trail_sample.ball_phase
                ]
                if trail_sample.ball_visible and trail_phase == "shot":
                    layers.append((trail_sample, alpha))
        layers.append((sample, 255))
        return tuple(layers)

    def draw_cinematic_poc_ball(
        self,
        state: dict[str, object],
    ) -> None:
        sequence = state["poc_contract_sequence"]
        sample = state["poc_contract_sample"]
        viewport = state["poc_viewport"]
        elapsed = float(state["poc_elapsed"])
        if (
            not isinstance(sequence, PocSequence)
            or not isinstance(sample, PocSequenceSample)
            or not isinstance(viewport, PocViewport)
            or not sample.ball_visible
        ):
            return
        size = max(
            1,
            round(POC_BALL_CANVAS_SIZE * viewport.scale),
        )
        team = (
            self.home
            if str(state["possession"]) == "home"
            else self.away
        )
        for layer_sample, alpha in self.cinematic_poc_ball_layers(
            sequence,
            sample,
            elapsed,
            team,
        ):
            ball = self.cached_cinematic_ball_material(
                size,
                self.cinematic_poc2_ball_rotation(
                    sequence,
                    layer_sample,
                    team,
                ),
            )
            if alpha < 255:
                ball = ball.copy()
                ball.set_alpha(alpha)
            ball_x = self.cinematic_poc_ball_x(
                sequence,
                layer_sample,
                team,
            )
            ball_y = (
                self.cinematic_poc2_sequence_sample(
                    sequence,
                    layer_sample,
                    team,
                ).ball.scene_center_y
                if layer_sample.actor_source == 0
                else layer_sample.ball_y
            )
            center = viewport.point(
                ball_x,
                ball_y,
            )
            self.screen.blit(
                ball,
                ball.get_rect(
                    center=(
                        round(center[0]),
                        round(center[1]),
                    )
                ),
            )

    def draw_cinematic_poc_keeper(
        self,
        state: dict[str, object],
    ) -> None:
        sequence = state["poc_contract_sequence"]
        sample = state["poc_contract_sample"]
        viewport = state["poc_viewport"]
        if (
            not isinstance(sequence, PocSequence)
            or not isinstance(sample, PocSequenceSample)
            or not isinstance(viewport, PocViewport)
        ):
            return
        keeper_team = (
            self.away
            if str(state["possession"]) == "home"
            else self.home
        )
        use_left_frames = sequence.keeper_direction == "left"
        if use_left_frames:
            frames = self.assets.cinematic_keeper_frames_left[
                keeper_team.code
            ]
        else:
            frames = self.assets.cinematic_keeper_frames[
                keeper_team.code
            ]
        frame_index = max(
            0,
            min(len(frames) - 1, sample.keeper_frame),
        )
        keeper_x = float(state["poc_keeper_x"])
        keeper_y = float(state["poc_keeper_y"])
        keeper_offset_x = keeper_x - sample.keeper_x
        self.draw_cinematic_poc_shadow(
            viewport,
            sample.keeper_shadow_x + keeper_offset_x,
            sample.keeper_shadow_y,
            sample.keeper_shadow_w,
            sample.keeper_shadow_h,
            sample.keeper_shadow_alpha,
        )
        target = viewport.size(340.0, 340.0)
        top_left = viewport.point(
            keeper_x,
            keeper_y,
        )
        self.screen.blit(
            self.cached_smoothscale(frames[frame_index], target),
            (round(top_left[0]), round(top_left[1])),
        )

    def draw_cinematic_poc_goal_layer(
        self,
        state: dict[str, object],
        *,
        front: bool,
    ) -> None:
        sequence = state["poc_contract_sequence"]
        sample = state["poc_contract_sample"]
        goal_rect = state["goal_rect"]
        if (
            not isinstance(sequence, PocSequence)
            or not isinstance(sample, PocSequenceSample)
            or not isinstance(goal_rect, pygame.Rect)
        ):
            return
        if self.poc_sequences is None:
            return

        if front:
            front_asset = self.poc_sequences.goal_front_layers[
                sequence.goal_side
            ]
            front_layer = self.load_cinematic_poc_layer(
                front_asset.file,
                front_asset.sha256,
            )
            contact_frame = (
                self.poc_sequences.nearest_net_contact_frame(
                    sequence,
                    sample.elapsed,
                )
            )
            if (
                contact_frame is not None
                and sample.elapsed >= sequence.impact_seconds
            ):
                contact = self.load_cinematic_poc_atlas_frame(
                    contact_frame.front_contact,
                    contact_frame.front_contact_sha256,
                    contact_frame.front_contact_source_rect,
                )
                contact_layer = (
                    self.cinematic_poc_composite_goal_layer(
                        cache_key=(
                            "front_contact",
                            contact_frame.front_contact,
                            contact_frame.front_contact_source_rect,
                            contact_frame.front_contact_rect,
                        ),
                        canvas_size=front_layer.get_size(),
                        patch=contact,
                        patch_rect=contact_frame.front_contact_rect,
                        target_size=goal_rect.size,
                    )
                )
                self.screen.blit(
                    contact_layer,
                    goal_rect,
                )
            self.screen.blit(
                self.cached_smoothscale(
                    front_layer,
                    goal_rect.size,
                ),
                goal_rect,
            )
            return

        if (
            sequence.outcome != "goal"
            or sample.elapsed < sequence.impact_seconds
        ):
            base_asset = self.poc_sequences.goal_base_layers[
                sequence.goal_side
            ]
            base_layer = self.load_cinematic_poc_layer(
                base_asset.file,
                base_asset.sha256,
            )
            self.screen.blit(
                self.cached_smoothscale(
                    base_layer,
                    goal_rect.size,
                ),
                goal_rect,
            )
            return

        if (
            sequence.net_static_back is None
            or sequence.net_static_back_sha256 is None
        ):
            raise RuntimeError(
                f"missing approved POC net back: {sequence.key}"
            )
        static_back = self.load_cinematic_poc_layer(
            sequence.net_static_back,
            sequence.net_static_back_sha256,
        )
        keyframe = self.cinematic_poc_net_keyframe(
            sequence,
            sample,
        )
        if keyframe is None:
            self.screen.blit(
                self.cached_smoothscale(
                    static_back,
                    goal_rect.size,
                ),
                goal_rect,
            )
            return
        roi = self.load_cinematic_poc_atlas_frame(
            keyframe.back_roi,
            keyframe.back_roi_sha256,
            keyframe.back_roi_source_rect,
        )
        composite_back = self.cinematic_poc_composite_goal_layer(
            cache_key=(
                "back_net",
                sequence.net_static_back,
                keyframe.back_roi,
                keyframe.back_roi_source_rect,
                keyframe.back_roi_rect,
            ),
            canvas_size=static_back.get_size(),
            patch=roi,
            patch_rect=keyframe.back_roi_rect,
            base=static_back,
            target_size=goal_rect.size,
        )
        self.screen.blit(
            composite_back,
            goal_rect,
        )

    def load_cinematic_poc_layer(
        self,
        relative_path: str,
        expected_sha256: str,
    ) -> pygame.Surface:
        cached = self.poc_layer_cache.get(relative_path)
        if cached is not None:
            return cached
        path = self.cinematic_poc_asset_path(relative_path)
        if not path.exists():
            raise RuntimeError(
                f"missing approved POC layer: {relative_path}"
            )
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"approved POC layer drift: {relative_path}"
            )
        surface = pygame.image.load(path).convert_alpha()
        self.poc_layer_cache[relative_path] = surface
        return surface

    @staticmethod
    def cinematic_poc_asset_path(relative_path: str) -> Path:
        path = (ROOT / relative_path).resolve()
        allowed_root = (
            ROOT
            / "assets"
            / "generated"
            / "cinematic"
            / "poc7_net"
        ).resolve()
        if allowed_root not in path.parents:
            raise RuntimeError(
                f"invalid approved POC layer path: {relative_path}"
            )
        return path

    def load_cinematic_poc_atlas_frame(
        self,
        relative_path: str,
        expected_sha256: str,
        source_rect: tuple[int, int, int, int],
    ) -> pygame.Surface:
        key = (relative_path, source_rect)
        cached = self.poc_layer_frame_cache.get(key)
        if cached is not None:
            return cached
        atlas = self.load_cinematic_poc_layer(
            relative_path,
            expected_sha256,
        )
        rect = pygame.Rect(source_rect)
        if not atlas.get_rect().contains(rect):
            raise RuntimeError(
                f"invalid approved POC atlas rect: {relative_path} {source_rect}"
            )
        frame = atlas.subsurface(rect)
        self.poc_layer_frame_cache[key] = frame
        return frame

    def preload_cinematic_poc_sequence(
        self,
        sequence: PocSequence,
    ) -> None:
        if self.poc_sequences is None:
            return
        base = self.poc_sequences.goal_base_layers[
            sequence.goal_side
        ]
        front = self.poc_sequences.goal_front_layers[
            sequence.goal_side
        ]
        self.load_cinematic_poc_layer(base.file, base.sha256)
        self.load_cinematic_poc_layer(front.file, front.sha256)
        if sequence.outcome != "goal":
            return
        if (
            sequence.net_static_back is None
            or sequence.net_static_back_sha256 is None
        ):
            raise RuntimeError(
                f"missing approved POC net preload: {sequence.key}"
            )
        self.load_cinematic_poc_layer(
            sequence.net_static_back,
            sequence.net_static_back_sha256,
        )
        assets = {
            (
                keyframe.back_roi,
                keyframe.back_roi_sha256,
            )
            for keyframe in sequence.net_keyframes
        }
        assets.update(
            (
                contact_frame.front_contact,
                contact_frame.front_contact_sha256,
            )
            for contact_frame in sequence.net_contact_frames
        )
        for relative_path, expected_sha256 in assets:
            self.load_cinematic_poc_layer(
                relative_path,
                expected_sha256,
            )

    def cinematic_poc_preload_assets(
        self,
        sequences: tuple[PocSequence, ...],
    ) -> tuple[tuple[str, str], ...]:
        if self.poc_sequences is None:
            return ()
        assets: dict[str, str] = {}

        def register(
            relative_path: str | None,
            expected_sha256: str | None,
        ) -> None:
            if relative_path is None or expected_sha256 is None:
                return
            previous = assets.get(relative_path)
            if previous is not None and previous != expected_sha256:
                raise RuntimeError(
                    f"conflicting approved POC hash: {relative_path}"
                )
            assets[relative_path] = expected_sha256

        for sequence in sequences:
            base = self.poc_sequences.goal_base_layers[
                sequence.goal_side
            ]
            front = self.poc_sequences.goal_front_layers[
                sequence.goal_side
            ]
            register(base.file, base.sha256)
            register(front.file, front.sha256)
            if sequence.outcome != "goal":
                continue
            register(
                sequence.net_static_back,
                sequence.net_static_back_sha256,
            )
            for keyframe in sequence.net_keyframes:
                register(
                    keyframe.back_roi,
                    keyframe.back_roi_sha256,
                )
            for contact_frame in sequence.net_contact_frames:
                register(
                    contact_frame.front_contact,
                    contact_frame.front_contact_sha256,
                )
        return tuple(sorted(assets.items()))

    def cinematic_poc_sequences_for_match(
        self,
        runtime: MatchRuntimeState,
    ) -> tuple[PocSequence, ...]:
        if self.poc_sequences is None:
            return ()
        events = (
            *(
                CinematicAttackEvent(
                    minute,
                    side,
                    True,
                    "goal",
                )
                for minute, side in runtime.goals
            ),
            *(
                CinematicAttackEvent(
                    minute,
                    side,
                    False,
                    outcome,
                )
                for minute, side, outcome in runtime.chances
            ),
        )
        unique: dict[str, PocSequence] = {}
        for event in events:
            sequence = self.cinematic_poc_sequence_for_event(event)
            unique[sequence.key] = sequence
        return tuple(unique[key] for key in sorted(unique))

    def cancel_cinematic_poc_preload(self) -> None:
        self.poc_preload_generation += 1
        self.poc_preload_cancel_event.set()
        self.poc_preload_threads = [
            thread
            for thread in self.poc_preload_threads
            if thread.is_alive()
        ]
        self.poc_preload_ready = True
        self.poc_preload_pending = 0
        self.poc_preload_completed = 0
        self.poc_preload_error = ""

    def start_cinematic_poc_preload(
        self,
        sequences: tuple[PocSequence, ...],
    ) -> None:
        self.cancel_cinematic_poc_preload()
        assets = self.cinematic_poc_preload_assets(sequences)
        if not assets:
            return
        generation = self.poc_preload_generation
        cancel_event = threading.Event()
        preload_queue: queue.Queue[
            tuple[object, ...]
        ] = queue.Queue(maxsize=2)
        self.poc_preload_cancel_event = cancel_event
        self.poc_preload_queue = preload_queue
        self.poc_preload_pending = len(assets)
        self.poc_preload_completed = 0
        self.poc_preload_ready = False

        def publish(item: tuple[object, ...]) -> bool:
            while not cancel_event.is_set():
                try:
                    preload_queue.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def worker() -> None:
            try:
                for relative_path, expected_sha256 in assets:
                    if cancel_event.is_set():
                        return
                    path = self.cinematic_poc_asset_path(
                        relative_path
                    )
                    data = path.read_bytes()
                    actual_sha256 = hashlib.sha256(data).hexdigest()
                    if actual_sha256 != expected_sha256:
                        raise RuntimeError(
                            "approved POC layer drift: "
                            f"{relative_path}"
                        )
                    surface = pygame.image.load(
                        io.BytesIO(data),
                        path.name,
                    )
                    if not publish(
                        (
                            "asset",
                            generation,
                            relative_path,
                            surface,
                        )
                    ):
                        return
                publish(("done", generation))
            except Exception as exc:
                publish(("error", generation, str(exc)))

        preload_thread = threading.Thread(
            target=worker,
            name="arena-ai-cinematic-preload",
            daemon=True,
        )
        self.poc_preload_thread = preload_thread
        self.poc_preload_threads.append(preload_thread)
        preload_thread.start()

    def shutdown_cinematic_poc_preloads(
        self,
        *,
        timeout: float = 0.5,
    ) -> bool:
        threads = tuple(self.poc_preload_threads)
        self.cancel_cinematic_poc_preload()
        deadline = time.perf_counter() + max(0.0, timeout)
        for thread in threads:
            if not thread.is_alive():
                continue
            thread.join(timeout=max(0.0, deadline - time.perf_counter()))
        self.poc_preload_threads = [
            thread
            for thread in threads
            if thread.is_alive()
        ]
        self.poc_preload_thread = (
            self.poc_preload_threads[-1]
            if self.poc_preload_threads
            else None
        )
        return not self.poc_preload_threads

    def drain_cinematic_poc_preload(
        self,
        *,
        max_assets: int = 1,
    ) -> None:
        converted = 0
        while converted < max_assets:
            try:
                item = self.poc_preload_queue.get_nowait()
            except queue.Empty:
                break
            if len(item) < 2:
                continue
            kind = str(item[0])
            generation = int(item[1])
            if generation != self.poc_preload_generation:
                continue
            if kind == "asset":
                relative_path = str(item[2])
                surface = item[3]
                if not isinstance(surface, pygame.Surface):
                    raise RuntimeError(
                        "invalid decoded POC preload surface"
                    )
                self.poc_layer_cache[relative_path] = (
                    surface.convert_alpha()
                )
                self.poc_preload_completed += 1
                converted += 1
            elif kind == "done":
                self.poc_preload_ready = (
                    self.poc_preload_completed
                    == self.poc_preload_pending
                )
                if not self.poc_preload_ready:
                    raise RuntimeError(
                        "incomplete approved POC preload"
                    )
            elif kind == "error":
                self.poc_preload_error = str(item[2])
                self.poc_preload_ready = False
                raise RuntimeError(self.poc_preload_error)

    def draw_cinematic_scene(self, field: pygame.Rect, pred: Prediction, state: dict[str, object] | None = None) -> None:
        if state is None:
            state = self.cinematic_scene_state(field, pred)
        if state["neutral"]:
            self.draw_cinematic_neutral(field, state)
            return

        possession = str(state["possession"])
        team = self.home if possession == "home" else self.away
        direction = 1 if possession == "home" else -1
        shot_progress = float(state["shot_progress"])
        poc2_sample = state.get("poc2_dribble_sample")
        if state.get("poc2_dribble"):
            if not isinstance(poc2_sample, Poc2DribbleSample):
                raise RuntimeError(
                    "invalid approved POC 2 dribble state"
                )
            if bool(state.get("settled", False)):
                self.draw_cinematic_neutral_player(
                    team,
                    state["actor_pos"],
                    flip=direction < 0,
                    stride_phase=0.0,
                    neutral_progress=1.0,
                )
            else:
                self.draw_cinematic_poc2_runner(
                    team,
                    (
                        poc2_sample.player.scene_center_x,
                        poc2_sample.player.scene_ground_y,
                    ),
                    left=direction < 0,
                    frame_index=poc2_sample.frame_index,
                    flight=poc2_sample.flight,
                    scale=float(state["poc2_scale"]),
                )
        else:
            self.draw_cinematic_runner(
                team,
                state["actor_pos"],
                flip=direction < 0,
                shot_progress=shot_progress,
                stride_phase=float(state.get("stride_phase", 0.0)),
                run_speed=float(state.get("run_speed", 1.0)),
                settled=bool(state.get("settled", False)),
                runner_pose=state.get("runner_pose") if isinstance(state.get("runner_pose"), dict) else None,
            )

        ball_squash = state.get("ball_squash", (1.0, 1.0))
        if not isinstance(ball_squash, tuple):
            ball_squash = (1.0, 1.0)
        self.draw_cinematic_ball(
            state["ball_pos"],
            active_goal=False,
            shot_progress=shot_progress,
            scale=int(state.get("ball_scale", CINEMATIC_BALL_SIZE)),
            direction=direction,
            prev_pos=state.get("ball_prev_pos"),
            ground_pos=state.get("ball_ground_pos"),
            velocity=state.get("ball_velocity_px_s"),
            rotation_degrees=float(state.get("ball_rotation_degrees", 0.0)),
            squash=(float(ball_squash[0]), float(ball_squash[1])),
            phase=str(state.get("ball_phase", "drible")),
        )

    def draw_cinematic_neutral(self, field: pygame.Rect, state: dict[str, object]) -> None:
        neutral_progress = float(state.get("neutral_progress", 1.0))
        neutral_alpha = int(255 * float(state.get("neutral_reveal", 1.0)))
        self.draw_cinematic_neutral_player(
            self.home,
            state["home_pos"],
            flip=False,
            stride_phase=float(state.get("home_stride_phase", 0.0)),
            neutral_progress=neutral_progress,
            alpha=neutral_alpha,
        )
        self.draw_cinematic_neutral_player(
            self.away,
            state["away_pos"],
            flip=True,
            stride_phase=float(state.get("away_stride_phase", 0.0)),
            neutral_progress=neutral_progress,
            alpha=neutral_alpha,
        )
        ball_pos = state["ball_pos"]
        ball_squash = state.get("ball_squash", (1.0, 1.0))
        if not isinstance(ball_squash, tuple):
            ball_squash = (1.0, 1.0)
        self.draw_cinematic_ball(
            ball_pos,
            active_goal=False,
            shot_progress=0.0,
            scale=CINEMATIC_BALL_SIZE,
            direction=0,
            prev_pos=state.get("ball_prev_pos"),
            ground_pos=state.get("ball_ground_pos"),
            velocity=state.get("ball_velocity_px_s"),
            rotation_degrees=float(state.get("ball_rotation_degrees", 0.0)),
            squash=(float(ball_squash[0]), float(ball_squash[1])),
            phase=str(state.get("ball_phase", "neutro")),
            alpha=neutral_alpha,
        )
        if neutral_alpha > 0:
            text = self.text_cache.render(self.f_md, "EMPATE", GOLD)
            text = self.cached_alpha(text, neutral_alpha)
            self.screen.blit(text, text.get_rect(center=(field.centerx, field.y + 238)))

    def shot_phase(self, shot_progress: float) -> str:
        if shot_progress < SHOT_PLANT_AT:
            return SHOT_PHASE_APPROACH
        if shot_progress < SHOT_KICK_AT:
            return SHOT_PHASE_PLANT
        if shot_progress < SHOT_CONTACT_FREEZE_END:
            return SHOT_PHASE_CONTACT_FREEZE
        if shot_progress < SHOT_RELEASE_END:
            return SHOT_PHASE_RELEASE
        if shot_progress < SHOT_NET_AT:
            return SHOT_PHASE_FOLLOW_THROUGH
        if shot_progress < SHOT_RECOVERY_AT:
            return SHOT_PHASE_NET_IMPACT
        return SHOT_PHASE_RECOVERY

    def cinematic_stride_state(self, shot_progress: float, stride_phase: float) -> tuple[bool, int]:
        shot_phase = self.shot_phase(shot_progress)
        if shot_phase == SHOT_PHASE_PLANT:
            return False, 2
        if shot_phase == SHOT_PHASE_CONTACT_FREEZE and shot_progress < SHOT_KICK_POSE_AT:
            return False, 2
        if shot_phase in {SHOT_PHASE_CONTACT_FREEZE, SHOT_PHASE_RELEASE}:
            return True, 3
        if shot_phase in {SHOT_PHASE_FOLLOW_THROUGH, SHOT_PHASE_NET_IMPACT} and shot_progress < SHOT_FOLLOW_THROUGH_HOLD_END:
            return True, 3
        if shot_phase in {SHOT_PHASE_FOLLOW_THROUGH, SHOT_PHASE_NET_IMPACT}:
            return False, 3
        phase = stride_phase % 4.0
        if phase < 1.05:
            return False, 0
        if phase < 2.0:
            return False, 1
        if phase < 3.05:
            return False, 2
        return False, 3

    def cinematic_runner_plant_blend(self, shot_progress: float) -> float:
        plant_in = smoothstep(
            (shot_progress - (SHOT_PLANT_AT - 0.10))
            / max(0.001, SHOT_KICK_AT - (SHOT_PLANT_AT - 0.10))
        )
        recovery = 1.0 - smoothstep(
            (shot_progress - SHOT_FOLLOW_THROUGH_HOLD_END)
            / max(0.001, SHOT_RECOVERY_AT - SHOT_FOLLOW_THROUGH_HOLD_END)
        )
        return plant_in * recovery

    def cinematic_controlled_stride_phase(
        self,
        shot_progress: float,
        stride_phase: float,
        event_seconds: float | None = None,
        event_start_phase: float | None = None,
    ) -> float:
        phase = stride_phase % 4.0
        motion = self.assets.cinematic_runner_motion
        exit_frame = int(motion.get("kick_exit_runner_frame", 0))
        exit_phase = exit_frame / CINEMATIC_RUNNER_FRAME_COUNT * 4.0
        phase_rate = 124.0 / 60.0 * 2.0

        if 0.0 < shot_progress < SHOT_RECOVERY_AT and event_seconds:
            # The event-start pose is anchored during update(). A 124 steps/min
            # clock then reaches the closest compatible entry monotonically.
            start_phase = 1.75 if event_start_phase is None else event_start_phase % 4.0
            natural_advance = phase_rate * event_seconds * SHOT_RUN_TO_PLANT_AT
            candidates = tuple(
                int(frame)
                for frame in motion.get(
                    "kick_entry_runner_frames",
                    (motion.get("kick_entry_runner_frame", 0),),
                )
            )

            def candidate_plan(frame: int) -> tuple[float, int, float]:
                target_phase = frame / CINEMATIC_RUNNER_FRAME_COUNT * 4.0
                forward = (target_phase - start_phase) % 4.0
                cycles = max(0, round((natural_advance - forward) / 4.0))
                advance = forward + cycles * 4.0
                return abs(advance - natural_advance), candidates.index(frame), advance

            _error, _priority, planned_advance = min(candidate_plan(frame) for frame in candidates)
            progress = clamp(shot_progress / SHOT_RUN_TO_PLANT_AT)
            return (start_phase + planned_advance * progress) % 4.0
        if shot_progress <= SHOT_RUN_TO_PLANT_AT:
            return phase
        if shot_progress >= SHOT_RECOVERY_AT:
            if event_seconds:
                recovery_elapsed = (shot_progress - SHOT_RECOVERY_AT) * event_seconds
                return (exit_phase + phase_rate * recovery_elapsed) % 4.0
            return phase
        return phase

    def cinematic_runner_reference_height(
        self,
        uniform_code: str,
        direction: int,
    ) -> float:
        direction_name = "left" if direction < 0 else "right"
        cache_key = (uniform_code, direction_name)
        cached = self.runner_reference_height_cache.get(cache_key)
        if cached is not None:
            return cached
        motion = self.assets.cinematic_runner_motion
        uniforms = motion.get("uniforms", {})
        uniform_payload = uniforms.get(uniform_code, uniforms.get("gold", {}))  # type: ignore[union-attr]
        frames = uniform_payload["directions"][direction_name]["frames"]  # type: ignore[index]
        heights = sorted(
            float(frame["visible_bbox"][3])
            for frame in frames
        )
        middle = len(heights) // 2
        reference = (
            heights[middle]
            if len(heights) % 2
            else (heights[middle - 1] + heights[middle]) * 0.5
        )
        reference = max(1.0, reference)
        self.runner_reference_height_cache[cache_key] = reference
        return reference

    def cinematic_frame_ground_lift(
        self,
        frame: pygame.Surface,
        target_size: tuple[int, int],
        baseline_y: float,
    ) -> float:
        rendered = self.cached_smoothscale(
            frame,
            target_size,
        )
        visible = self.visible_bbox(rendered)
        if visible.w <= 0 or visible.h <= 0:
            return 0.0
        return max(0.0, float(visible.bottom) - baseline_y)

    def cinematic_runner_pose(
        self,
        actor_pos: tuple[float, float],
        direction: int,
        shot_progress: float,
        stride_phase: float,
        run_speed: float,
        event_seconds: float | None = None,
        event_start_phase: float | None = None,
        uniform_code: str = "gold",
    ) -> dict[str, object]:
        motion = self.assets.cinematic_runner_motion
        direction_name = "left" if direction < 0 else "right"
        uniforms = motion.get("uniforms", {})
        uniform_payload = uniforms.get(uniform_code, uniforms.get("gold", {}))  # type: ignore[union-attr]
        direction_payload = uniform_payload["directions"][direction_name]  # type: ignore[index]
        frame_metadata = direction_payload["frames"]  # type: ignore[index]
        controlled_phase = self.cinematic_controlled_stride_phase(
            shot_progress,
            stride_phase,
            event_seconds,
            event_start_phase,
        )
        poc2_run = event_seconds is None and shot_progress <= 0.0
        frame_position = (
            cinematic_poc2_runtime_frame_position(controlled_phase)
            if poc2_run
            else controlled_phase
            / 4.0
            * CINEMATIC_RUNNER_FRAME_COUNT
        )
        frame_index = int(math.floor(frame_position)) % CINEMATIC_RUNNER_FRAME_COUNT
        next_frame_index = (
            frame_index
            if poc2_run
            else (frame_index + 1) % CINEMATIC_RUNNER_FRAME_COUNT
        )
        frame_blend = (
            0.0
            if poc2_run
            else frame_position - math.floor(frame_position)
        )
        render_frame_index = (
            frame_index
            if poc2_run or frame_blend < 0.5
            else next_frame_index
        )
        metadata = frame_metadata[frame_index]
        next_metadata = frame_metadata[next_frame_index]

        scale = (
            CINEMATIC_POSE_SIZE
            * CINEMATIC_PLAYER_SCALE
            / self.cinematic_runner_reference_height(uniform_code, direction)
        )
        frame_size = int(motion["frame_size"])
        target_size = (max(1, round(frame_size * scale)), max(1, round(frame_size * scale)))
        root_anchor = motion["root_anchor"]
        root_x = float(root_anchor[0]) * scale
        baseline_y = float(motion["ground_baseline_y"]) * scale
        ground_lift = (
            self.cinematic_frame_ground_lift(
                self.assets.cinematic_runner_frames_for_uniform(
                    uniform_code,
                    left=True,
                )[render_frame_index],
                target_size,
                baseline_y,
            )
            if direction < 0
            else 0.0
        )

        support_weight = lerp(
            float(metadata["support_weight"]),
            float(next_metadata["support_weight"]),
            frame_blend,
        )
        def local_point(name: str) -> tuple[float, float]:
            previous = frame_metadata[(frame_index - 1) % CINEMATIC_RUNNER_FRAME_COUNT][name]
            current = metadata[name]
            following = next_metadata[name]
            after = frame_metadata[(frame_index + 2) % CINEMATIC_RUNNER_FRAME_COUNT][name]

            def catmull_rom(axis: int) -> float:
                p0 = float(previous[axis])
                p1 = float(current[axis])
                p2 = float(following[axis])
                p3 = float(after[axis])
                t = frame_blend
                return 0.5 * (
                    2.0 * p1
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
                )

            return catmull_rom(0), catmull_rom(1)
        touch_phase_offset = cinematic_dribble_touch_phase_offset(direction)
        first_touch_frame = round(
            touch_phase_offset / 4.0 * CINEMATIC_RUNNER_FRAME_COUNT
        )
        second_touch_frame = first_touch_frame + CINEMATIC_RUNNER_FRAME_COUNT // 2
        first_contact = frame_metadata[first_touch_frame]["dribble_foot"]
        second_contact = frame_metadata[second_touch_frame]["dribble_foot"]
        cycle_from_touch = (controlled_phase - touch_phase_offset) % 4.0
        contact_amount = smoothstep((cycle_from_touch % 2.0) / 2.0)
        if cycle_from_touch < 2.0:
            current_contact, following_contact = first_contact, second_contact
        else:
            current_contact, following_contact = second_contact, first_contact
        touch_slot = 0 if cycle_from_touch < 1.0 or cycle_from_touch >= 3.0 else 1
        dribble_anchor_x = lerp(float(current_contact[0]), float(following_contact[0]), contact_amount)
        dribble_anchor_y = lerp(float(current_contact[1]), float(following_contact[1]), contact_amount)
        current_support = metadata["support"]
        following_support = next_metadata["support"]
        metadata_support_transition = bool(metadata.get("support_transition"))
        next_metadata_support_transition = bool(next_metadata.get("support_transition"))
        if current_support and current_support == following_support:
            current_support_point = metadata["support_foot"]
            following_support_point = next_metadata["support_foot"]
            support_local = (
                lerp(float(current_support_point[0]), float(following_support_point[0]), frame_blend),
                lerp(float(current_support_point[1]), float(following_support_point[1]), frame_blend),
            )
            support_name = current_support
            support_transition = metadata_support_transition or next_metadata_support_transition
        else:
            selected_support = metadata if frame_blend < 0.5 else next_metadata
            selected_support_point = selected_support["support_foot"]
            support_local = (float(selected_support_point[0]), float(selected_support_point[1]))
            support_name = selected_support["support"]
            support_transition = (
                current_support != following_support
                or metadata_support_transition
                or next_metadata_support_transition
            )

        # Foot locking is baked below the pelvis by the asset generator. The
        # actor root stays fixed so the torso and shirt mark never slide.
        lock_offset_x = 0.0
        left = float(actor_pos[0]) - root_x
        top = float(actor_pos[1]) - baseline_y - ground_lift

        def screen_point(name: str) -> tuple[float, float]:
            point_x, point_y = local_point(name)
            return left + point_x * scale, top + point_y * scale

        support_pos = (left + support_local[0] * scale, top + support_local[1] * scale)
        rendered_dribble_pos = screen_point("dribble_foot")
        dribble_control_pos = screen_point("dribble_control")
        dribble_contact_pos = (
            left + dribble_anchor_x * scale,
            top + dribble_anchor_y * scale,
        )
        pelvis_pos = screen_point("pelvis")
        return {
            "controlled_phase": controlled_phase,
            "frame_index": frame_index,
            "next_frame_index": next_frame_index,
            "frame_blend": frame_blend,
            "render_frame_index": render_frame_index,
            "target_size": target_size,
            "rect_topleft": (left, top),
            "root_offset_x": lock_offset_x,
            "pelvis_pos": pelvis_pos,
            "support_foot_pos": support_pos,
            "dribble_foot_pos": rendered_dribble_pos,
            "dribble_control_pos": dribble_control_pos,
            "dribble_contact_pos": dribble_contact_pos,
            "dribble_touch_slot": touch_slot,
            "support": support_name,
            "support_transition": support_transition,
            "support_weight": support_weight,
            "flight": support_weight < 0.21,
            "root_bob": lerp(float(metadata["root_bob"]), float(next_metadata["root_bob"]), frame_blend) * scale,
            "sole_ground_gap": float(actor_pos[1]) - support_pos[1],
            "uniform_code": uniform_code,
        }

    def draw_soft_shadow(self, rect: pygame.Rect, alpha: int) -> None:
        alpha_key = max(0, min(255, int(round(alpha / 4.0) * 4)))
        width = max(2, int(rect.w * 1.28))
        height = max(2, int(rect.h * 1.28))
        cache_key = ("soft_shadow", width, height, alpha_key)
        surface = self.cinematic_overlay_cache.get(cache_key)
        if surface is None:
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            center = surface.get_rect().center
            for layer, factor in enumerate((1.0, 0.86, 0.72, 0.58)):
                shade = int(alpha_key * (0.28 + layer * 0.24))
                shadow = pygame.Rect(0, 0, max(2, int(width * factor)), max(2, int(height * factor)))
                shadow.center = center
                pygame.draw.ellipse(surface, (0, 0, 0, shade), shadow)
            self.cinematic_overlay_cache[cache_key] = surface
        self.screen.blit(surface, surface.get_rect(center=rect.center))

    def cinematic_planted_x(
        self,
        x: float,
        stride: int,
        kick_window: bool,
        direction: int,
        run_speed: float,
    ) -> float:
        if kick_window:
            return x
        foot_offsets = (-3, 1, 3, 0)
        return x + foot_offsets[stride] * direction * min(1.5, max(0.85, run_speed))

    def cinematic_actor_anchor_screen(
        self,
        midbottom: tuple[float, float],
        target_size: tuple[int, int],
        anchor: tuple[float, float],
        flip: bool,
        source_frame: pygame.Surface | None = None,
    ) -> tuple[float, float]:
        width, height = target_size
        anchor_x = 1.0 - anchor[0] if flip else anchor[0]
        if source_frame is not None:
            bbox = self.visible_bbox(source_frame)
            if bbox.w > 0 and bbox.h > 0:
                scale_x = width / max(1, source_frame.get_width())
                scale_y = height / max(1, source_frame.get_height())
                left = midbottom[0] - bbox.x * scale_x - bbox.w * scale_x / 2
                top = midbottom[1] - bbox.bottom * scale_y
                return left + anchor_x * width, top + anchor[1] * height
        left = midbottom[0] - width / 2
        top = midbottom[1] - height
        return left + anchor_x * width, top + anchor[1] * height

    def cinematic_toe_anchor(self, frame: pygame.Surface, direction: int) -> tuple[float, float]:
        cache_key = (id(frame), 1 if direction >= 0 else -1)
        cached = self.surface_toe_anchor_cache.get(cache_key)
        if cached is not None:
            return cached

        bbox = self.visible_bbox(frame)
        if bbox.w <= 0 or bbox.h <= 0:
            return KICK_FOOT_ANCHOR
        mask = pygame.mask.from_surface(frame, 64)
        y_start = max(bbox.top, int(round(bbox.top + bbox.h * 0.45)))
        y_end = min(bbox.bottom, int(round(bbox.top + bbox.h * 0.86)))
        pixels = [
            (x, y)
            for y in range(y_start, y_end)
            for x in range(bbox.left, bbox.right)
            if mask.get_at((x, y))
        ]
        if not pixels:
            return KICK_FOOT_ANCHOR
        extreme_x = max(x for x, _y in pixels) if direction >= 0 else min(x for x, _y in pixels)
        edge_pixels = [(x, y) for x, y in pixels if abs(x - extreme_x) <= 4]
        toe_x = sum(x for x, _y in edge_pixels) / len(edge_pixels)
        toe_y = sum(y for _x, y in edge_pixels) / len(edge_pixels)
        anchor = (toe_x / frame.get_width(), toe_y / frame.get_height())
        self.surface_toe_anchor_cache[cache_key] = anchor
        return anchor

    def cinematic_visible_midbottom_rect(self, frame: pygame.Surface, midbottom: tuple[float, float]) -> pygame.Rect:
        bbox = self.visible_bbox(frame)
        if bbox.w <= 0 or bbox.h <= 0:
            return frame.get_rect(midbottom=(int(round(midbottom[0])), int(round(midbottom[1]))))
        return pygame.Rect(
            int(round(midbottom[0] - bbox.x - bbox.w / 2)),
            int(round(midbottom[1] - bbox.bottom)),
            frame.get_width(),
            frame.get_height(),
        )

    def cinematic_kick_contact_policy(self, uniform_code: str, direction: int) -> tuple[int, float]:
        direction_name = "left" if direction < 0 else "right"
        uniforms = self.assets.cinematic_runner_motion.get("uniforms", {})
        uniform_payload = uniforms.get(uniform_code, uniforms.get("gold", {}))  # type: ignore[union-attr]
        direction_payload = uniform_payload["directions"][direction_name]  # type: ignore[index]
        return (
            int(direction_payload["kick_contact_frame"]),
            float(direction_payload["kick_contact_gap_px"]),
        )

    def cinematic_ball_clearance_directed_x(
        self,
        pose: dict[str, object],
        uniform_code: str,
        direction: int,
        action: str,
    ) -> float:
        direction_name = "left" if direction < 0 else "right"
        motion = self.assets.cinematic_runner_motion
        uniforms = motion.get("uniforms", {})
        uniform_payload = uniforms.get(uniform_code, uniforms.get("gold", {}))  # type: ignore[union-attr]
        direction_payload = uniform_payload["directions"][direction_name]  # type: ignore[index]
        if action in {"kick", "kick_physical"}:
            metadata_frames = direction_payload["kick_frames"]
            frame_index = int(pose["frame_index"])
        elif action in {"run", "run_approach"}:
            metadata_frames = direction_payload["frames"]
            frame_index = int(pose["render_frame_index"])
        else:
            raise ValueError(f"unsupported cinematic clearance action: {action}")
        metadata = metadata_frames[frame_index]
        if "ball_clearance_offset_px" not in metadata:
            raise RuntimeError(
                f"runner motion metadata is missing the ball clearance contract: "
                f"{uniform_code}/{direction_name}/{action}/{frame_index}"
            )
        offset_field = (
            "ball_corridor_offset_px"
            if action == "kick"
            else "ball_approach_corridor_offset_px"
            if action == "run_approach"
            else "ball_clearance_offset_px"
        )
        if offset_field not in metadata:
            raise RuntimeError(
                f"runner motion metadata is missing {offset_field}: "
                f"{uniform_code}/{direction_name}/{action}/{frame_index}"
            )
        frame_size = float(motion["frame_size"])
        target_size = pose["target_size"]
        scale = float(target_size[0]) / frame_size  # type: ignore[index]
        root_x = float(motion["root_anchor"][0])
        local_ball_x = root_x + direction * float(metadata[offset_field])
        left = float(pose["rect_topleft"][0])  # type: ignore[index]
        if action == "kick_physical":
            return direction * (left + local_ball_x * scale)
        if action != "kick":
            return direction * (left + local_ball_x * scale)

        unshifted_left = (
            left
            - float(pose.get("entry_root_offset_x", 0.0))
            - float(pose.get("contact_root_offset_x", 0.0))
            - float(pose.get("dynamic_contact_catchup_x", 0.0))
        )
        corridor_x = direction * (
            unshifted_left + local_ball_x * scale
        )
        physical_local_x = root_x + direction * float(
            metadata["ball_clearance_offset_px"]
        )
        physical_x = direction * (
            left + physical_local_x * scale
        )
        return max(corridor_x, physical_x)

    def cinematic_kick_contact_anchor(
        self,
        pose: dict[str, object],
        uniform_code: str,
        direction: int,
        contact_gap: float,
    ) -> tuple[float, float]:
        direction_name = "left" if direction < 0 else "right"
        motion = self.assets.cinematic_runner_motion
        uniforms = motion.get("uniforms", {})
        uniform_payload = uniforms.get(uniform_code, uniforms.get("gold", {}))  # type: ignore[union-attr]
        direction_payload = uniform_payload["directions"][direction_name]  # type: ignore[index]
        frame_index = int(pose["frame_index"])
        metadata = direction_payload["kick_frames"][frame_index]
        frame_size = float(motion["frame_size"])
        scale = float(pose["target_size"][0]) / frame_size  # type: ignore[index]
        root_x = float(motion["root_anchor"][0])
        local_ball_x = root_x + direction * float(
            metadata["ball_clearance_offset_px"]
        )
        left = float(pose["rect_topleft"][0])  # type: ignore[index]
        directed_ball_x = direction * (left + local_ball_x * scale)
        contact_radius = CINEMATIC_BALL_SIZE * 0.46
        foot_x = direction * (
            directed_ball_x - contact_radius - contact_gap
        )
        foot_y = float(pose["dribble_foot_pos"][1])  # type: ignore[index]
        return foot_x, foot_y

    @staticmethod
    def cinematic_contact_catchup_weight(shot_progress: float) -> float:
        if shot_progress <= SHOT_RUN_TO_PLANT_AT:
            return 0.0
        if shot_progress <= SHOT_KICK_AT:
            return smoothstep(
                (shot_progress - SHOT_RUN_TO_PLANT_AT)
                / max(0.001, SHOT_KICK_AT - SHOT_RUN_TO_PLANT_AT)
            )
        if shot_progress <= SHOT_CONTACT_FREEZE_END:
            return 1.0
        return 1.0 - smoothstep(
            (shot_progress - SHOT_CONTACT_FREEZE_END)
            / max(0.001, SHOT_RECOVERY_AT - SHOT_CONTACT_FREEZE_END)
        )

    @staticmethod
    def cinematic_ball_corridor_cache_key(
        actor_y: float,
        direction: int,
        uniform_code: str,
        screen_anchor_x: float,
        shot_actor_x: float,
        event_seconds: float,
        event_start_phase: float | None,
    ) -> tuple[object, ...]:
        return (
            uniform_code,
            direction,
            round(actor_y, 3),
            round(screen_anchor_x, 3),
            round(shot_actor_x, 3),
            round(event_seconds, 6),
            None
            if event_start_phase is None
            else round(event_start_phase % 4.0, 6),
        )

    def cinematic_attack_contact_catchup(
        self,
        actor_y: float,
        direction: int,
        uniform_code: str,
        screen_anchor_x: float,
        shot_actor_x: float,
        event_seconds: float,
        event_start_phase: float | None,
    ) -> float:
        key = self.cinematic_ball_corridor_cache_key(
            actor_y,
            direction,
            uniform_code,
            screen_anchor_x,
            shot_actor_x,
            event_seconds,
            event_start_phase,
        )
        cached = self.cinematic_ball_corridor_cache.get(key)
        if cached is None:
            self.cinematic_attack_ball_path_x(
                SHOT_KICK_AT,
                actor_y,
                direction,
                uniform_code,
                screen_anchor_x,
                shot_actor_x,
                event_seconds,
                event_start_phase,
            )
            cached = self.cinematic_ball_corridor_cache.get(key)
        if cached is None:
            raise RuntimeError("cinematic ball corridor cache was not populated")
        return float(cached[1])

    def cinematic_attack_ball_path_x(
        self,
        shot_progress: float,
        actor_y: float,
        direction: int,
        uniform_code: str,
        screen_anchor_x: float,
        shot_actor_x: float,
        event_seconds: float,
        event_start_phase: float | None,
    ) -> float:
        start = CINEMATIC_BALL_CORRIDOR_START
        end = SHOT_KICK_AT
        sample_rate = 120.0
        sample_count = max(
            2,
            math.ceil((end - start) * event_seconds * sample_rate) + 1,
        )
        key = self.cinematic_ball_corridor_cache_key(
            actor_y,
            direction,
            uniform_code,
            screen_anchor_x,
            shot_actor_x,
            event_seconds,
            event_start_phase,
        )
        cached = self.cinematic_ball_corridor_cache.get(key)
        if cached is None:
            if len(self.cinematic_ball_corridor_cache) >= 128:
                self.cinematic_ball_corridor_cache.clear()
            entry_pose = self.cinematic_runner_pose(
                (shot_actor_x, actor_y),
                direction,
                SHOT_RUN_TO_PLANT_AT,
                0.0,
                1.0,
                event_seconds=event_seconds,
                event_start_phase=event_start_phase,
                uniform_code=uniform_code,
            )
            required_floors: list[float] = []
            for index in range(sample_count):
                progress = lerp(start, end, index / (sample_count - 1))
                actor_x = lerp(
                    screen_anchor_x,
                    shot_actor_x,
                    smoothstep(progress / 0.58),
                )
                if progress < SHOT_RUN_TO_PLANT_AT:
                    pose = self.cinematic_runner_pose(
                        (actor_x, actor_y),
                        direction,
                        progress,
                        0.0,
                        1.0,
                        event_seconds=event_seconds,
                        event_start_phase=event_start_phase,
                        uniform_code=uniform_code,
                    )
                    physical_floor = self.cinematic_ball_clearance_directed_x(
                        pose,
                        uniform_code,
                        direction,
                        "run",
                    )
                else:
                    pose = self.cinematic_kick_pose(
                        (actor_x, actor_y),
                        direction,
                        progress,
                        entry_pose,
                    )
                    if pose is None:
                        raise RuntimeError(
                            "kick pose is unavailable while building the ball corridor"
                        )
                    physical_floor = self.cinematic_ball_clearance_directed_x(
                        pose,
                        uniform_code,
                        direction,
                        "kick_physical",
                    )
                required_floor = (
                    physical_floor
                    if progress >= CINEMATIC_KICK_APPROACH_START
                    else -math.inf
                )
                if progress <= CINEMATIC_KICK_APPROACH_START:
                    natural_dribble = self.cinematic_dribble_kinematics(
                        (actor_x, actor_y),
                        direction,
                        float(pose["controlled_phase"]),
                        0.0,
                        shot_progress=progress,
                        kick_contact_position=(0.0, 0.0),
                        foot_position=pose["dribble_control_pos"],  # type: ignore[arg-type]
                        visible_foot_position=pose["dribble_foot_pos"],  # type: ignore[arg-type]
                        contact_foot_position=pose["dribble_contact_pos"],  # type: ignore[arg-type]
                        travel_distance=0.0,
                    )
                    required_floor = max(
                        required_floor,
                        direction * float(natural_dribble.ball.position[0]),
                    )
                required_floors.append(required_floor)

            raw_contact_floor = required_floors[-1]
            contact_catchup = 0.0
            for index, required_floor in enumerate(required_floors[:-1]):
                if not math.isfinite(required_floor):
                    continue
                progress = lerp(start, end, index / (sample_count - 1))
                catchup_weight = self.cinematic_contact_catchup_weight(progress)
                if catchup_weight >= 1.0 - 1e-6:
                    continue
                contact_catchup = max(
                    contact_catchup,
                    (required_floor - raw_contact_floor)
                    / (1.0 - catchup_weight),
                )
            contact_catchup = max(0.0, contact_catchup)
            if contact_catchup > CINEMATIC_CONTACT_CATCHUP_MAX:
                raise RuntimeError(
                    "cinematic contact catch-up exceeds the authored motion budget: "
                    f"{uniform_code}/{direction}={contact_catchup:.3f}px"
                )
            if contact_catchup > 0.0:
                required_floors = [
                    (
                        required_floor
                        + contact_catchup
                        * self.cinematic_contact_catchup_weight(
                            lerp(start, end, index / (sample_count - 1))
                        )
                        if math.isfinite(required_floor)
                        else required_floor
                    )
                    for index, required_floor in enumerate(required_floors)
                ]

            contact_floor = required_floors[-1]
            for index in range(sample_count):
                progress = lerp(start, end, index / (sample_count - 1))
                if progress < CINEMATIC_KICK_APPROACH_START:
                    continue
                approach = smoothstep(
                    (progress - CINEMATIC_KICK_APPROACH_START)
                    / max(
                        0.001,
                        SHOT_KICK_AT - CINEMATIC_KICK_APPROACH_START,
                    )
                )
                authored_floor = lerp(
                    contact_floor - CINEMATIC_KICK_APPROACH_DISTANCE,
                    contact_floor,
                    approach,
                )
                required_floors[index] = max(
                    required_floors[index],
                    authored_floor,
                )

            approach_index = next(
                index
                for index in range(sample_count)
                if lerp(start, end, index / (sample_count - 1))
                >= CINEMATIC_KICK_APPROACH_START
            )
            monotonic_floors: list[float] = []
            running_floor = max(required_floors[:approach_index])
            for required_floor in required_floors[approach_index:]:
                running_floor = max(running_floor, required_floor)
                monotonic_floors.append(running_floor)

            sample_dt = (end - start) * event_seconds / (sample_count - 1)
            max_step = 150.0 * sample_dt
            for index in range(len(monotonic_floors) - 2, -1, -1):
                monotonic_floors[index] = max(
                    monotonic_floors[index],
                    monotonic_floors[index + 1] - max_step,
                )
            approach_floor = monotonic_floors[0]
            lead_sample_count = approach_index
            lead_start_floor = required_floors[0]
            lead_start_slope = required_floors[1] - required_floors[0]
            lead_end_slope = (
                monotonic_floors[1] - monotonic_floors[0]
                if len(monotonic_floors) > 1
                else 0.0
            )
            lead_in_floors: list[float] = []
            for index in range(approach_index):
                blend = index / max(1, lead_sample_count)
                blend_sq = blend * blend
                blend_cu = blend_sq * blend
                h00 = 2.0 * blend_cu - 3.0 * blend_sq + 1.0
                h10 = blend_cu - 2.0 * blend_sq + blend
                h01 = -2.0 * blend_cu + 3.0 * blend_sq
                h11 = blend_cu - blend_sq
                hermite_floor = (
                    h00 * lead_start_floor
                    + h10 * lead_start_slope * lead_sample_count
                    + h01 * approach_floor
                    + h11 * lead_end_slope * lead_sample_count
                )
                lead_in_floors.append(hermite_floor)
            floors = tuple(lead_in_floors + monotonic_floors)
            self.cinematic_ball_corridor_cache[key] = (
                floors,
                contact_catchup,
            )
        else:
            floors, _contact_catchup = cached

        normalized = clamp((shot_progress - start) / max(0.001, end - start))
        position = normalized * (sample_count - 1)
        first = min(sample_count - 1, int(math.floor(position)))
        following = min(sample_count - 1, first + 1)
        return lerp(
            floors[first],
            floors[following],
            position - first,
        )

    def cinematic_kick_frame_index(
        self,
        shot_progress: float,
        uniform_code: str | None = None,
        direction: int = 1,
    ) -> int | None:
        blend = self.cinematic_kick_frame_blend(shot_progress, uniform_code, direction)
        if blend is None:
            return None
        first, following, amount = blend
        return first if amount < 0.5 else following

    def cinematic_kick_frame_blend(
        self,
        shot_progress: float,
        uniform_code: str | None = None,
        direction: int = 1,
    ) -> tuple[int, int, float] | None:
        if shot_progress < SHOT_RUN_TO_PLANT_AT or shot_progress >= SHOT_RECOVERY_AT:
            return None
        contact_frame = (
            int(self.assets.cinematic_runner_motion["kick_contact_frame"])
            if uniform_code is None
            else self.cinematic_kick_contact_policy(uniform_code, direction)[0]
        )
        last_frame = CINEMATIC_KICK_FRAME_COUNT - 1
        if shot_progress < SHOT_KICK_AT:
            normalized = smoothstep(
                (shot_progress - SHOT_RUN_TO_PLANT_AT)
                / max(0.001, SHOT_KICK_AT - SHOT_RUN_TO_PLANT_AT)
            )
            position = normalized * contact_frame
        elif shot_progress < SHOT_CONTACT_FREEZE_END:
            return contact_frame, contact_frame, 0.0
        else:
            follow_through_reach_at = max(
                SHOT_CONTACT_FREEZE_END + 0.001,
                SHOT_FOLLOW_THROUGH_HOLD_END - 0.036,
            )
            normalized = smoothstep(
                (shot_progress - SHOT_CONTACT_FREEZE_END)
                / max(0.001, follow_through_reach_at - SHOT_CONTACT_FREEZE_END)
            )
            position = contact_frame + normalized * (last_frame - contact_frame)
            if shot_progress >= follow_through_reach_at:
                return last_frame, last_frame, 0.0
        first = min(last_frame, int(math.floor(position)))
        following = min(last_frame, first + 1)
        return first, following, position - math.floor(position)

    def cinematic_kick_pose(
        self,
        actor_pos: tuple[float, float],
        direction: int,
        shot_progress: float,
        entry_pose: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        motion = self.assets.cinematic_runner_motion
        direction_name = "left" if direction < 0 else "right"
        uniform_code = str(entry_pose.get("uniform_code", "gold")) if entry_pose is not None else "gold"
        frame_blend = self.cinematic_kick_frame_blend(shot_progress, uniform_code, direction)
        if frame_blend is None:
            return None
        first, following, blend = frame_blend
        frame_index = first if blend < 0.5 else following
        uniforms = motion.get("uniforms", {})
        uniform_payload = uniforms.get(uniform_code, uniforms.get("gold", {}))  # type: ignore[union-attr]
        direction_payload = uniform_payload["directions"][direction_name]  # type: ignore[index]
        metadata_frames = direction_payload["kick_frames"]
        metadata = metadata_frames[frame_index]
        frame_size = float(motion["frame_size"])
        scale = (
            CINEMATIC_POSE_SIZE
            * CINEMATIC_PLAYER_SCALE
            / self.cinematic_runner_reference_height(uniform_code, direction)
        )
        baseline_y = float(motion["ground_baseline_y"])
        target_size = (
            max(1, round(frame_size * scale)),
            max(1, round(frame_size * scale)),
        )

        plant_frame = int(motion.get("kick_plant_frame", 5))
        # The approved kick POC keeps the body root on the authored pelvis arc.
        # A generic lowest-pixel detector can swap the planted and striking
        # boots, so it must not translate the whole player during the strike.
        lock_offset_x = 0.0
        lock_offset_y = 0.0
        ground_lift = (
            self.cinematic_frame_ground_lift(
                self.assets.cinematic_kick_frames_for_uniform(
                    uniform_code,
                    left=True,
                )[frame_index],
                target_size,
                baseline_y * scale,
            )
            if direction < 0
            else 0.0
        )

        metadata_pelvis = metadata["pelvis"]
        left = float(actor_pos[0]) - float(metadata_pelvis[0]) * scale + lock_offset_x * scale
        top = (
            float(actor_pos[1])
            - baseline_y * scale
            - ground_lift
            + lock_offset_y * scale
        )
        entry_root_offset_x = 0.0
        if entry_pose is not None and frame_index < plant_frame:
            entry_weight = 1.0 - smoothstep(
                float(frame_index) / max(1.0, float(plant_frame))
            )
            runner_frame_index = int(entry_pose["render_frame_index"])
            runner_frame = self.assets.cinematic_runner_frames_for_uniform(
                uniform_code,
                left=direction < 0,
            )[runner_frame_index]
            kick_frame = self.assets.cinematic_kick_frames_for_uniform(
                uniform_code,
                left=direction < 0,
            )[frame_index]
            runner_centroid_x, _runner_centroid_y = self.visible_alpha_centroid(
                runner_frame
            )
            kick_centroid_x, _kick_centroid_y = self.visible_alpha_centroid(
                kick_frame
            )
            entry_left, _entry_top = entry_pose["rect_topleft"]  # type: ignore[misc]
            entry_scale = float(entry_pose["target_size"][0]) / frame_size  # type: ignore[index]
            entry_center_x = float(entry_left) + runner_centroid_x * entry_scale
            kick_center_x = left + kick_centroid_x * scale
            entry_root_offset_x = (
                entry_center_x - kick_center_x
            ) * entry_weight
            left += entry_root_offset_x
        contact_frame = int(direction_payload["kick_contact_frame"])
        contact_root_offset_x = 0.0
        contact_root_correction = float(
            direction_payload["kick_contact_root_correction_px"]
        )
        frame_position = float(first) + float(blend)
        if frame_position <= contact_frame:
            root_advance_start = max(0, plant_frame - 2)
            root_advance = smoothstep(
                (frame_position - root_advance_start)
                / max(1.0, float(contact_frame - root_advance_start))
            )
            available_correction = max(
                0.0,
                float(metadata["ball_corridor_offset_px"])
                - float(metadata["ball_clearance_offset_px"]),
            )
            correction_native = min(
                available_correction,
                contact_root_correction * root_advance,
            )
        else:
            last_frame = int(motion["kick_frame_count"]) - 1
            correction_native = contact_root_correction * clamp(
                (last_frame - frame_position)
                / max(1.0, float(last_frame - contact_frame))
            )
        if correction_native > 0.0:
            contact_root_offset_x = direction * correction_native * scale
            left += contact_root_offset_x
        dynamic_contact_catchup_x = 0.0
        if entry_pose is not None:
            contact_catchup = float(entry_pose.get("contact_catchup_px", 0.0))
            if contact_catchup > 0.0:
                dynamic_contact_catchup_x = (
                    direction
                    * contact_catchup
                    * self.cinematic_contact_catchup_weight(shot_progress)
                )
                left += dynamic_contact_catchup_x
        def screen_point(name: str) -> tuple[float, float]:
            point = metadata[name]
            return left + float(point[0]) * scale, top + float(point[1]) * scale

        return {
            "frame_index": frame_index,
            "first_frame_index": first,
            "following_frame_index": following,
            "frame_blend": blend,
            "target_size": target_size,
            "rect_topleft": (left, top),
            "root_lock_offset": (lock_offset_x * scale, lock_offset_y * scale),
            "entry_root_offset_x": entry_root_offset_x,
            "contact_root_offset_x": contact_root_offset_x,
            "dynamic_contact_catchup_x": dynamic_contact_catchup_x,
            "support": metadata.get("support"),
            "support_weight": float(metadata.get("support_weight", 0.0)),
            "support_foot_pos": screen_point("support_foot"),
            "dribble_foot_pos": screen_point("dribble_foot"),
            "pelvis_pos": screen_point("pelvis"),
            "uniform_code": uniform_code,
        }

    def draw_cinematic_poc2_runner(
        self,
        team: TeamProfile,
        pos: tuple[float, float],
        *,
        left: bool,
        frame_index: int,
        flight: bool,
        scale: float,
        alpha: int = 255,
        shadow: bool = True,
    ) -> None:
        frames = (
            self.assets.cinematic_poc2_runners_left[team.code]
            if left
            else self.assets.cinematic_poc2_runners[team.code]
        )
        frame_index = max(0, min(len(frames) - 1, frame_index))
        source = frames[frame_index]
        canvas_size = int(
            self.assets.cinematic_poc2_motion["canvas_size"]
        )
        canvas_ground_y = float(
            self.assets.cinematic_poc2_motion["canvas_ground_y"]
        )
        target = (
            max(1, round(canvas_size * scale)),
            max(1, round(canvas_size * scale)),
        )
        x, ground_y = pos
        rect = pygame.Rect(
            round(x - canvas_size * 0.5 * scale),
            round(ground_y - canvas_ground_y * scale),
            *target,
        )
        if shadow:
            shadow_rect = pygame.Rect(
                0,
                0,
                max(1, round((105 if flight else 122) * scale)),
                max(1, round(15 * scale)),
            )
            shadow_rect.center = (
                round(x),
                round(ground_y - scale),
            )
            self.draw_soft_shadow(
                shadow_rect,
                round(
                    (92 if flight else 124)
                    * clamp(alpha / 255.0)
                ),
            )
        rendered = self.cached_smoothscale(source, target)
        if alpha < 255:
            rendered = self.cached_alpha(rendered, alpha)
        self.screen.blit(rendered, rect)

    def draw_cinematic_runner(
        self,
        team: TeamProfile,
        pos: object,
        flip: bool,
        shot_progress: float,
        stride_phase: float = 0.0,
        run_speed: float = 1.0,
        settled: bool = False,
        alpha: int = 255,
        runner_pose: dict[str, object] | None = None,
    ) -> None:
        x, ground_y = pos  # type: ignore[misc]
        frames = self.assets.cinematic_runners_left[team.code] if flip else self.assets.cinematic_runners[team.code]
        kick_frames = self.assets.cinematic_kicks_left[team.code] if flip else self.assets.cinematic_kicks[team.code]
        stop_frames = self.assets.cinematic_stops_left[team.code] if flip else self.assets.cinematic_stops[team.code]
        direction = -1 if flip else 1
        uniform_code = self.assets.cinematic_source_code(team)
        if runner_pose is None:
            runner_pose = self.cinematic_runner_pose(
                (float(x), float(ground_y)),
                direction,
                shot_progress,
                stride_phase,
                run_speed,
                uniform_code=uniform_code,
            )

        layers: list[tuple[pygame.Surface, tuple[int, int], pygame.Rect, float]] = []

        def add_runner_layers(
            weight: float,
            *,
            standing: bool = False,
            stop_position: float | None = None,
        ) -> None:
            if weight <= 0.001:
                return
            target = runner_pose["target_size"]
            left, top = runner_pose["rect_topleft"]
            rect = pygame.Rect(int(round(float(left))), int(round(float(top))), *target)
            if standing:
                position = (
                    float(len(stop_frames) - 1)
                    if stop_position is None
                    else clamp(stop_position, 0.0, float(len(stop_frames) - 1))
                )
                frame_index = int(round(position))
                source = stop_frames[frame_index]
                bbox = self.visible_bbox(source)
                if bbox.w > 0 and bbox.h > 0:
                    base_scale = float(target[0]) / max(1, source.get_width())
                    visible_center_x = rect.left + bbox.centerx * base_scale
                    visible_bottom = rect.top + bbox.bottom * base_scale
                    pose_scale = (
                        CINEMATIC_POSE_SIZE
                        * CINEMATIC_PLAYER_SCALE
                        / self.cinematic_runner_reference_height(
                            uniform_code,
                            direction,
                        )
                    )
                    target = (
                        max(1, round(source.get_width() * pose_scale)),
                        max(1, round(source.get_height() * pose_scale)),
                    )
                    rect = pygame.Rect(
                        int(round(visible_center_x - bbox.centerx * pose_scale)),
                        int(round(visible_bottom - bbox.bottom * pose_scale)),
                        *target,
                    )
                layers.append((source, target, rect, weight))
                return
            render_frame = int(runner_pose["render_frame_index"])
            layers.append((frames[render_frame], target, rect, weight))

        def add_kick_layer(
            weight: float,
            *,
            pose_progress: float | None = None,
        ) -> None:
            if weight <= 0.001:
                return
            kick_pose = self.cinematic_kick_pose(
                (float(x), float(ground_y)),
                direction,
                shot_progress if pose_progress is None else pose_progress,
                runner_pose,
            )
            if kick_pose is None:
                return
            frame_index = int(kick_pose["frame_index"])
            source = kick_frames[frame_index]
            target = kick_pose["target_size"]
            left, top = kick_pose["rect_topleft"]
            rect = pygame.Rect(int(round(float(left))), int(round(float(top))), *target)
            layers.append((source, target, rect, weight))

        uses_run_shadow = False
        if settled:
            add_runner_layers(1.0, standing=True)
        elif shot_progress < SHOT_RUN_TO_PLANT_AT:
            add_runner_layers(1.0)
            uses_run_shadow = True
        elif shot_progress < SHOT_RECOVERY_AT:
            add_kick_layer(1.0)
        elif shot_progress < SHOT_STOP_BLEND_END:
            recovery = clamp(
                (shot_progress - SHOT_RECOVERY_AT)
                / max(0.001, SHOT_STOP_BLEND_END - SHOT_RECOVERY_AT)
            )
            add_runner_layers(
                1.0,
                standing=True,
                stop_position=smoothstep(recovery) * (len(stop_frames) - 1),
            )
        else:
            add_runner_layers(1.0, standing=True)

        visible_widths = []
        for source, target, _rect, weight in layers:
            if weight <= 0.001:
                continue
            bbox = self.visible_bbox(source)
            visible_widths.append(bbox.w * target[0] / max(1, source.get_width()))
        visible_w = max(visible_widths, default=96.0)
        if uses_run_shadow:
            support_weight = float(runner_pose["support_weight"])
            flight = bool(runner_pose["flight"])
            shadow_center_x = float(x) + (
                10.0 if direction < 0 else 0.0
            )
            shadow_w = max(70, int(visible_w * (0.72 + 0.14 * support_weight)))
            shadow_h = max(
                8,
                int((10.0 + 4.0 * support_weight - (2.0 if flight else 0.0)) * CINEMATIC_PLAYER_SCALE),
            )
            shadow_alpha = int(
                (92 if flight else 124)
                * clamp(alpha / 255.0)
            )
        else:
            shadow_center_x = float(x)
            shadow_w = max(74, int(visible_w * 0.84))
            shadow_h = int(14.0 * CINEMATIC_PLAYER_SCALE)
            shadow_alpha = int(160 * clamp(alpha / 255.0))
        shadow = pygame.Rect(0, 0, shadow_w, shadow_h)
        shadow.center = (
            int(round(shadow_center_x)),
            int(round(float(ground_y) - 1)),
        )
        self.draw_soft_shadow(shadow, shadow_alpha)

        for source, target, rect, weight in layers:
            layer_alpha = max(0, min(255, int(round(alpha * weight))))
            if layer_alpha <= 0:
                continue
            rendered = self.cached_smoothscale(source, target)
            if layer_alpha < 255:
                original_alpha = rendered.get_alpha()
                rendered.set_alpha(layer_alpha)
                self.screen.blit(rendered, rect)
                rendered.set_alpha(original_alpha)
            else:
                self.screen.blit(rendered, rect)

    def neutral_frame_for_phase(
        self,
        team: TeamProfile,
        stride_phase: float,
        neutral_progress: float,
        flip: bool,
    ) -> pygame.Surface:
        runner_frames = self.assets.cinematic_runners_left[team.code] if flip else self.assets.cinematic_runners[team.code]
        stop_frames = self.assets.cinematic_stops_left[team.code] if flip else self.assets.cinematic_stops[team.code]
        if neutral_progress < 0.72:
            frame_index = (
                int(
                    (stride_phase % 4.0)
                    / 4.0
                    * len(runner_frames)
                )
                % len(runner_frames)
            )
            return runner_frames[frame_index]
        stop_progress = smoothstep((neutral_progress - 0.72) / 0.28)
        frame_index = min(len(stop_frames) - 1, int(round(stop_progress * (len(stop_frames) - 1))))
        return stop_frames[frame_index]

    def draw_cinematic_neutral_player(
        self,
        team: TeamProfile,
        pos: object,
        flip: bool,
        stride_phase: float,
        neutral_progress: float,
        alpha: int = 255,
    ) -> None:
        if neutral_progress < 0.72:
            x, ground_y = pos  # type: ignore[misc]
            uniform_code = self.assets.cinematic_source_code(team)
            scale = (
                CINEMATIC_POSE_SIZE
                * CINEMATIC_NEUTRAL_PLAYER_SCALE
                / POC_APPROVED_REFERENCE_VISIBLE_HEIGHT
            )
            elapsed = self.t + (
                self.poc2_dribble.metadata.cycle_seconds * 0.5
                if flip
                else 0.0
            )
            sample = self.poc2_dribble.sample(
                uniform_code,
                flip,
                max(0.0, elapsed),
                float(x),
                float(ground_y),
                scale,
            )
            self.draw_cinematic_poc2_runner(
                team,
                (
                    sample.player.scene_center_x,
                    sample.player.scene_ground_y,
                ),
                left=flip,
                frame_index=sample.frame_index,
                flight=sample.flight,
                scale=scale,
                alpha=alpha,
            )
            return
        x, ground_y = pos  # type: ignore[misc]
        frame = self.neutral_frame_for_phase(team, stride_phase, neutral_progress, flip)
        target = self.cinematic_actor_target_size(frame, CINEMATIC_NEUTRAL_PLAYER_SCALE)
        bbox = self.visible_bbox(frame)
        visible_w = bbox.w * target[0] / max(1, frame.get_width())
        frame = self.cached_smoothscale(frame, target)
        rect = self.cinematic_visible_midbottom_rect(frame, (x, ground_y))
        shadow = pygame.Rect(0, 0, max(72, int(visible_w * 0.76)), int(13 * CINEMATIC_NEUTRAL_PLAYER_SCALE))
        shadow.center = (int(x), int(ground_y - 1))
        self.draw_soft_shadow(shadow, int(150 * clamp(alpha / 255.0)))
        if alpha < 255:
            frame = self.cached_alpha(frame, alpha)
        self.screen.blit(frame, rect)

    def cinematic_actor_target_size(self, frame: pygame.Surface, render_scale: float) -> tuple[int, int]:
        bbox = self.visible_bbox(frame)
        visible_h = max(1, bbox.h)
        target_visible_h = int(CINEMATIC_POSE_SIZE * render_scale)
        scale = target_visible_h / visible_h
        return max(1, int(frame.get_width() * scale)), max(1, int(frame.get_height() * scale))

    def draw_cinematic_kick_impact(self, pos: object, direction: int, shot_progress: float) -> None:
        impact_end = SHOT_KICK_AT + 0.09
        if not SHOT_KICK_AT <= shot_progress <= impact_end:
            return
        x, y = pos  # type: ignore[misc]
        strength = smoothstep((shot_progress - SHOT_KICK_AT) / 0.012) * (
            1.0 - smoothstep((shot_progress - (SHOT_KICK_AT + 0.018)) / 0.072)
        )
        if strength <= 0.02:
            return
        center = (int(x + direction * 5), int(y + 3))
        effect = self.cached_kick_impact_effect(direction, strength, self.t)
        self.screen.blit(effect, effect.get_rect(center=center))

    def cached_kick_impact_effect(self, direction: int, strength: float, time_value: float) -> pygame.Surface:
        strength_key = int(round(clamp(strength) * 24))
        phase_key = int(time_value * 12) % 6
        cache_key = ("kick_impact", direction, strength_key, phase_key)
        cached = self.cinematic_overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        effect = pygame.Surface((76, 52), pygame.SRCALPHA)
        local = (38, 27)
        strength = strength_key / 24.0
        pygame.draw.arc(
            effect,
            (244, 247, 232, int(48 * strength)),
            pygame.Rect(local[0] - 10, local[1] - 10, 20, 20),
            -1.15 if direction > 0 else 2.0,
            0.55 if direction > 0 else 3.70,
            1,
        )
        for index in range(6):
            px = local[0] - direction * int(5 + index * 5 + strength * 4)
            py = local[1] + 9 + int(math.sin(index * 1.7 + phase_key) * 3)
            radius = 1 if index < 4 else 2
            pygame.draw.circle(effect, (183, 218, 157, int((54 - index * 5) * strength)), (px, py), radius)
        self.cinematic_overlay_cache[cache_key] = effect
        return effect

    def cinematic_keeper_animation_state(
        self,
        active_goal: bool,
        shot_progress: float,
        frame_count: int,
        flip: bool,
        keeper_action: str = "",
    ) -> tuple[int, float, int]:
        first, following, blend = self.cinematic_keeper_frame_blend(
            active_goal,
            shot_progress,
            frame_count,
            keeper_action,
        )
        index = first if blend < 0.5 else following
        # The authored GPT Image sequence already carries the complete body arc.
        # Extra rotation or pulsing scale bends the goalkeeper and the shirt mark.
        return index, CINEMATIC_KEEPER_SCALE, 0

    def cinematic_keeper_frame_blend(
        self,
        active_goal: bool,
        shot_progress: float,
        frame_count: int,
        keeper_action: str = "",
    ) -> tuple[int, int, float]:
        if frame_count <= 1 or (not active_goal and not keeper_action):
            return 0, 0, 0.0

        def authored_pair(position: float) -> tuple[int, int, float]:
            position = clamp(position, 0.0, float(frame_count - 1))
            first = int(math.floor(position))
            following = min(frame_count - 1, first + 1)
            return first, following, position - first

        last = float(frame_count - 1)
        keeper_motion_frames = self.assets.cinematic_keeper_motion[
            "runtime_frame_order"
        ]
        keeper_motion_last = max(1.0, float(self.assets.cinematic_keeper_motion["frame_count"]) - 1.0)

        def authored_frame(source_index: int) -> float:
            runtime_frame = float(keeper_motion_frames[source_index])  # type: ignore[index]
            return min(last, runtime_frame / keeper_motion_last * last)

        read_frame = authored_frame(3)
        extended_frame = authored_frame(8)
        landed_frame = authored_frame(12)
        absorb_frame = authored_frame(13)
        rising_frame = authored_frame(14)
        read = smoothstep(
            (shot_progress - SHOT_KEEPER_READ_AT)
            / max(0.001, SHOT_KEEPER_DIVE_AT - SHOT_KEEPER_READ_AT)
        )
        if keeper_action == "stand_save":
            stand_frame = authored_frame(2)
            if shot_progress <= SHOT_KEEPER_DIVE_AT:
                return authored_pair(stand_frame * read)
            hold_end = CHANCE_CONTACT_VISUAL_AT + 0.055
            if shot_progress <= hold_end:
                return authored_pair(stand_frame)
            settle = smoothstep((shot_progress - hold_end) / 0.09)
            return authored_pair(stand_frame * (1.0 - settle))
        if shot_progress <= SHOT_KEEPER_DIVE_AT:
            return authored_pair(read_frame * read)

        contact_at = (
            CHANCE_CONTACT_VISUAL_AT
            if keeper_action
            else SHOT_NET_VISUAL_CONTACT_AT
        )
        recovery_at = contact_at + (0.055 if keeper_action else 0.020)
        fully_extended_at = contact_at - (0.035 if keeper_action else 0.060)
        if shot_progress <= recovery_at:
            extension = smoothstep(
                (shot_progress - SHOT_KEEPER_DIVE_AT)
                / max(0.001, fully_extended_at - SHOT_KEEPER_DIVE_AT)
            )
            return authored_pair(lerp(read_frame, extended_frame, extension))
        landing_end = recovery_at + 0.10
        if shot_progress <= landing_end:
            landing = clamp((shot_progress - recovery_at) / max(0.001, landing_end - recovery_at))
            return authored_pair(lerp(extended_frame, landed_frame, landing))
        if active_goal and not keeper_action:
            return authored_pair(absorb_frame)
        rise_end = landing_end + 0.18
        if shot_progress <= rise_end:
            rise = clamp((shot_progress - landing_end) / max(0.001, rise_end - landing_end))
            return authored_pair(lerp(landed_frame, rising_frame, rise))
        set_end = rise_end + 0.16
        reset = smoothstep((shot_progress - rise_end) / max(0.001, set_end - rise_end))
        return authored_pair(lerp(rising_frame, last, reset))

    def draw_cinematic_keeper(
        self,
        team: TeamProfile,
        pos: object,
        flip: bool,
        active_goal: bool,
        shot_progress: float,
        alpha: int = 255,
        keeper_action: str = "",
        contrast_kit: bool = False,
        ground_y: float | None = None,
    ) -> None:
        x, y = pos  # type: ignore[misc]
        frames = (
            self.assets.cinematic_keeper_frames_left[team.code]
            if flip
            else self.assets.cinematic_keeper_frames[team.code]
        )
        _index, scale, angle = self.cinematic_keeper_animation_state(active_goal, shot_progress, len(frames), flip, keeper_action)
        first, following, blend = self.cinematic_keeper_frame_blend(
            active_goal,
            shot_progress,
            len(frames),
            keeper_action,
        )
        reveal = clamp(alpha / 255.0)
        keeper_shadow = pygame.Rect(0, 0, 130, 18)
        keeper_shadow.center = (
            int(x),
            int(round(float(ground_y) - 1.0)) if ground_y is not None else int(y + 84),
        )
        self.draw_soft_shadow(keeper_shadow, int(150 * reveal))

        frame = self.cinematic_keeper_material(
            team,
            flip,
            first,
            following,
            blend,
            scale,
            angle,
            contrast_kit,
        )
        frame_rect = self.cinematic_keeper_rect(
            frame,
            (x, y),
            shot_progress,
            active_goal,
            keeper_action,
            ground_y=ground_y,
        )
        if alpha < 255:
            original_alpha = frame.get_alpha()
            frame.set_alpha(alpha)
            self.screen.blit(frame, frame_rect)
            frame.set_alpha(original_alpha)
        else:
            self.screen.blit(frame, frame_rect)

    def cinematic_keeper_rect(
        self,
        frame: pygame.Surface,
        pos: tuple[float, float],
        shot_progress: float,
        active_goal: bool,
        keeper_action: str = "",
        ground_y: float | None = None,
    ) -> pygame.Rect:
        x, y = pos
        mask = pygame.mask.from_surface(frame, 48)
        centroid = mask.centroid()
        if centroid == (0, 0) and mask.count() == 0:
            centered = frame.get_rect(center=(round(x), round(y)))
        else:
            centered = pygame.Rect(
                int(round(x - centroid[0])),
                int(round(y - centroid[1])),
                frame.get_width(),
                frame.get_height(),
            )
        if keeper_action == "stand_save":
            grounding = 1.0
        else:
            recovery_at = (
                CHANCE_CONTACT_VISUAL_AT + 0.055
                if keeper_action
                else SHOT_NET_VISUAL_CONTACT_AT + 0.020
            )
            if (
                not active_goal
                and keeper_action != "dive_save"
                and shot_progress < recovery_at
            ):
                return centered

            landing_end = recovery_at + 0.10
            grounding = smoothstep(
                (shot_progress - recovery_at)
                / max(0.001, landing_end - recovery_at)
            )
        if grounding <= 0.0:
            return centered
        visible = self.visible_bbox(frame)
        visible_ground_y = (
            float(ground_y)
            if ground_y is not None
            else float(self.match_field_rect().bottom - 54)
        )
        grounded = pygame.Rect(
            int(round(x - visible.centerx)),
            int(round(visible_ground_y - visible.bottom)),
            frame.get_width(),
            frame.get_height(),
        )
        field = self.match_field_rect()
        visible_field_left = field.left + 8
        visible_field_right = field.right - 8
        visible_left = grounded.x + visible.left
        visible_right = grounded.x + visible.right
        if visible_left < visible_field_left:
            grounded.x += visible_field_left - visible_left
        elif visible_right > visible_field_right:
            grounded.x -= visible_right - visible_field_right
        return pygame.Rect(
            round(lerp(float(centered.x), float(grounded.x), grounding)),
            round(lerp(float(centered.y), float(grounded.y), grounding)),
            frame.get_width(),
            frame.get_height(),
        )

    def cinematic_keeper_material(
        self,
        team: TeamProfile,
        flip: bool,
        first: int,
        following: int,
        blend: float,
        scale: float,
        angle: float,
        contrast_kit: bool = False,
    ) -> pygame.Surface:
        frames = (
            self.assets.cinematic_keeper_frames_left[team.code]
            if flip
            else self.assets.cinematic_keeper_frames[team.code]
        )
        resolved_index = first if blend < 0.5 else following
        transition_key = (
            "keeper_authored",
            team.code,
            resolved_index,
            contrast_kit,
            flip,
            scale,
            angle,
        )
        frame = self.cinematic_overlay_cache.get(transition_key)
        if frame is None:
            source = frames[resolved_index]
            if contrast_kit:
                source = self.cinematic_contrast_keeper_frame(source)
            frame = self.cached_rotozoom(source, angle, scale)
            self.cinematic_overlay_cache[transition_key] = frame
        return frame

    def cinematic_contrast_keeper_frame(self, frame: pygame.Surface) -> pygame.Surface:
        cache_key = ("keeper_contrast_kit", id(frame))
        cached = self.cinematic_overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        result = frame.copy().convert_alpha()
        rgb_view = pygame.surfarray.pixels3d(result)
        rgb = rgb_view.astype(np.float32)
        alpha = pygame.surfarray.array_alpha(result)
        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]
        uniform = (
            (alpha > 40)
            & (green > red * 1.16)
            & (green > blue * 1.10)
            & (green - np.minimum(red, blue) > 22)
        )
        value = np.maximum.reduce((red, green, blue))
        shade = np.clip(value / 170.0, 0.34, 1.24)
        target = np.asarray((214.0, 62.0, 174.0), dtype=np.float32)
        recolored = target[None, None, :] * shade[:, :, None]
        rgb[uniform] = np.clip(recolored[uniform], 0, 255)
        rgb_view[:, :, :] = rgb.astype(np.uint8)
        del rgb_view
        self.cinematic_overlay_cache[cache_key] = result
        return result

    def draw_cinematic_ball(
        self,
        pos: object,
        active_goal: bool,
        shot_progress: float,
        scale: int = 46,
        direction: int = 1,
        prev_pos: object | None = None,
        ground_pos: object | None = None,
        velocity: object | None = None,
        rotation_degrees: float = 0.0,
        squash: tuple[float, float] = (1.0, 1.0),
        phase: str = "drible",
        alpha: int = 255,
    ) -> None:
        x, y = pos  # type: ignore[misc]
        size = max(22, int(scale))
        frame = self.cached_cinematic_ball_material(size, rotation_degrees)

        if isinstance(ground_pos, tuple) and len(ground_pos) == 2:
            shadow_x, shadow_y = float(ground_pos[0]), float(ground_pos[1])
        else:
            shadow_x, shadow_y = float(x) - 7.0, float(y) + size * 0.44
        altitude = max(0.0, shadow_y - (float(y) + size * 0.43))
        shadow_factor = clamp(1.0 - altitude / 150.0, 0.52, 1.0)
        shadow = pygame.Rect(
            0,
            0,
            max(14, int(size * (0.90 * shadow_factor))),
            max(4, int(size * (0.20 * shadow_factor))),
        )
        shadow.center = (int(round(shadow_x)), int(round(shadow_y)))
        shadow_alpha = int((58 if phase in {"drible", "neutro"} else 42) * shadow_factor * clamp(alpha / 255.0))
        if phase == "rede":
            shadow_alpha = min(shadow_alpha, 18)
        self.draw_soft_shadow(shadow, shadow_alpha)

        velocity_x = velocity_y = 0.0
        if isinstance(velocity, tuple) and len(velocity) == 2:
            velocity_x, velocity_y = float(velocity[0]), float(velocity[1])
        elif isinstance(prev_pos, tuple) and len(prev_pos) == 2:
            velocity_x = (float(x) - float(prev_pos[0])) * FPS
            velocity_y = (float(y) - float(prev_pos[1])) * FPS
        speed = math.hypot(velocity_x, velocity_y)
        blur_strength = smoothstep((speed - 160.0) / 220.0) if phase == "chute" else 0.0
        if blur_strength > 0.02:
            render_dt = clamp(
                float(self.cinematic_render_dt),
                1.0 / FPS,
                MAX_FRAME_DT,
            )
            frame_dx = velocity_x * render_dt
            frame_dy = velocity_y * render_dt
            displacement = math.hypot(frame_dx, frame_dy)
            if displacement > 42.0:
                factor = 42.0 / displacement
                frame_dx *= factor
                frame_dy *= factor
            for amount, ghost_alpha in ((0.82, 10), (0.54, 18), (0.28, 28)):
                ghost = self.cached_alpha(
                    frame,
                    int(ghost_alpha * blur_strength * clamp(alpha / 255.0)),
                    step=8,
                )
                ghost_center = (
                    int(round(float(x) - frame_dx * amount)),
                    int(round(float(y) - frame_dy * amount)),
                )
                self.screen.blit(ghost, ghost.get_rect(center=ghost_center))

        if alpha < 255:
            frame = self.cached_alpha(frame, alpha)
        self.screen.blit(frame, frame.get_rect(center=(int(round(x)), int(round(y)))))

    def cached_cinematic_ball_material(self, size: int, rotation_degrees: float) -> pygame.Surface:
        frame_count = len(self.assets.balls)
        frame_float = (rotation_degrees % 360.0) / 360.0 * frame_count
        frame_index = int(frame_float) % frame_count
        return self.cached_smoothscale(self.assets.balls[frame_index], (size, size))

    def draw_cinematic_goal_overlay(self, field: pygame.Rect, pred: Prediction) -> None:
        event = self.active_goal_event(pred)
        if not event:
            return
        goal_minute, side = event
        shot_progress = (self.match_minute_float() - (goal_minute - GOAL_EVENT_WINDOW_MINUTES)) / GOAL_EVENT_WINDOW_MINUTES
        reveal = smoothstep((shot_progress - (SHOT_NET_VISUAL_CONTACT_AT + 0.012)) / 0.030)
        if reveal <= 0:
            return
        alpha = int(255 * reveal * clamp(1 - (self.match_minute_float() - goal_minute) / 4.0))
        if alpha <= 0:
            return
        center = self.cinematic_goal_overlay_center(field)
        cache_key = ("goal_overlay_panel", self.f_lg.get_height())
        panel = self.cinematic_overlay_cache.get(cache_key)
        if panel is None:
            text = self.text_cache.render(self.f_lg, "GOOOL!", GOLD)
            panel = pygame.Surface((text.get_width() + 54, text.get_height() + 22), pygame.SRCALPHA)
            pygame.draw.rect(panel, (2, 9, 13, 218), panel.get_rect(), border_radius=18)
            panel.blit(text, (27, 11))
            self.cinematic_overlay_cache[cache_key] = panel
        panel = self.cached_alpha(panel, alpha, step=8)
        self.screen.blit(panel, panel.get_rect(center=center))

    def cinematic_goal_overlay_center(self, field: pygame.Rect) -> tuple[int, int]:
        return field.centerx, field.y + 128

    def draw_model_flow(self, field: pygame.Rect, pred: Prediction, algo: str) -> None:
        cache_key = ("model_flow_overlay", field.w, field.h)
        overlay = self.cinematic_overlay_cache.get(cache_key)
        if overlay is None:
            overlay = pygame.Surface(field.size, pygame.SRCALPHA)
            self.cinematic_overlay_cache[cache_key] = overlay
        overlay.fill((0, 0, 0, 0))
        color = ALGORITHM_COLORS.get(algo, GOLD)
        home_prob, draw_prob, away_prob = self.live_probs(pred)
        bias = home_prob - away_prob
        direction = 1 if bias >= 0 else -1
        strength = clamp(abs(bias) * 2.8 + 0.18, 0.18, 0.95)
        alpha = int(42 + 48 * strength)
        pitch = self.playable_pitch_local(field)

        root = (pitch.centerx - direction * 112, pitch.centery)
        split = (pitch.centerx + direction * 16, pitch.centery)
        leaves = [
            (pitch.centerx + direction * 170, pitch.y + 66),
            (pitch.centerx + direction * 190, pitch.centery),
            (pitch.centerx + direction * 170, pitch.bottom - 66),
        ]
        pygame.draw.circle(overlay, (*color, alpha), root, 13)
        pygame.draw.circle(overlay, (*color, alpha), split, 11)
        pygame.draw.line(overlay, (*color, alpha), root, split, 3)
        for index, leaf in enumerate(leaves):
            leaf_alpha = alpha if index == int(self.t * 0.9) % len(leaves) else 48
            pygame.draw.line(overlay, (*color, leaf_alpha), split, leaf, 3)
            pygame.draw.circle(overlay, (*color, leaf_alpha), leaf, 17)
        for radius in (38, 68, 98):
            ring_alpha = int(24 + 28 * (1.0 - draw_prob))
            pygame.draw.circle(overlay, (*PURPLE, ring_alpha), pitch.center, radius, 2)
        self.screen.blit(overlay, field.topleft)

    def playable_pitch_local(self, field: pygame.Rect) -> pygame.Rect:
        return pygame.Rect(74, 82, field.w - 148, field.h - 162)

    def draw_side_panel(self, pred: Prediction, cinematic_focus: bool = False) -> None:
        panel = self.match_side_panel_rect()
        state_key = self.match_hud_state_key(cinematic_focus)
        state_label, state_title, state_hint = self.match_hud_state_copy(cinematic_focus)
        state_color = {"live": GOLD, "focus": CYAN, "closed": GREEN}[state_key]
        analysis = self.match_analysis if self.match_analysis and self.match_analysis.prediction == pred else None
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=18)
        pygame.draw.rect(self.screen, state_color, panel, 2, border_radius=18)
        left = panel.x + 24
        right = panel.right - 24
        content_w = right - left

        self.draw_text("Oráculo em campo", self.f_xs, MUTED, left, panel.y + 22)
        state_font = self.fit_font(state_label, 19, content_w, min_size=14)
        self.draw_text_ellipsis(state_label, state_font, state_color, left, panel.y + 48, content_w)
        title_font = self.fit_font(state_title, 30, content_w, min_size=22)
        self.draw_text_ellipsis(state_title, title_font, WHITE, left, panel.y + 75, content_w)
        model_hint = "A IA lê força, ritmo e placares"
        if self.match_result_revealed() and analysis is not None:
            model_hint = (
                f"xG {self.fmt_num(analysis.home_xg)} x {self.fmt_num(analysis.away_xg)}"
                f" | +2,5 {self.fmt_pct(analysis.over_25)}"
            )
        self.draw_text_ellipsis(model_hint, self.f_tiny, MUTED, left, panel.y + 108, content_w)
        pygame.draw.line(self.screen, (44, 64, 75), (left, panel.y + 132), (right, panel.y + 132), 1)

        def draw_flow_row(index: int, y: int, title: str, body: str, color: tuple[int, int, int], detail: str = "") -> None:
            node_center = (left + 14, y + 19)
            if index < 3:
                pygame.draw.line(self.screen, (45, 62, 72), (node_center[0], y + 33), (node_center[0], y + 64), 1)
            pygame.draw.circle(self.screen, color, node_center, 13)
            self.draw_text_centered(str(index), self.f_tiny, BLACK, node_center)
            text_x = left + 42
            text_w = right - text_x
            row_title_font = self.fit_font(title, 19, text_w, min_size=15)
            self.draw_text_ellipsis(title, row_title_font, color, text_x, y, text_w)
            self.draw_text_ellipsis(body, self.f_tiny, WHITE, text_x, y + 25, text_w)
            if detail:
                self.draw_text_ellipsis(detail, self.f_tiny, MUTED, text_x, y + 43, text_w)

        def draw_top_scores(y: int) -> None:
            scores = pred.top_scores[:MATCH_HUD_TOP_SCORE_COUNT]
            if not scores:
                return
            card = pygame.Rect(left, y, content_w, 116)
            pygame.draw.rect(self.screen, (8, 24, 34), card, border_radius=12)
            pygame.draw.rect(self.screen, (45, 69, 80), card, 1, border_radius=12)
            title = "Placares possíveis"
            self.draw_text_ellipsis(title, self.f_tiny, WHITE, card.x + 10, card.y + 8, card.w - 74)
            self.draw_text_right("Poisson/DC", self.f_tiny, GOLD, card.right - 10, card.y + 8)
            max_probability = max(0.001, max(float(probability) for _home_goals, _away_goals, probability in scores))
            for i, (home_goals, away_goals, probability) in enumerate(scores):
                row_y = card.y + 31 + i * 16
                row_score = f"{i + 1}. {home_goals}x{away_goals}"
                row_color = GOLD if i == 0 else WHITE
                self.draw_text_ellipsis(row_score, self.f_tiny, row_color, card.x + 10, row_y, 58)
                self.draw_hud_bar(
                    pygame.Rect(card.x + 74, row_y + 6, 86, 7),
                    float(probability) / max_probability,
                    GOLD if i == 0 else CYAN,
                    bg=(31, 45, 53),
                )
                self.draw_text_right(self.fmt_pct(float(probability), 1), self.f_tiny, GOLD if i == 0 else MUTED, card.right - 10, row_y)

        def draw_audit_pending(y: int) -> None:
            card = pygame.Rect(left, y, content_w, 96)
            pygame.draw.rect(self.screen, (8, 24, 34), card, border_radius=12)
            pygame.draw.rect(self.screen, (45, 69, 80), card, 1, border_radius=12)
            self.draw_text_ellipsis("Sorteio da Copa", self.f_xs, WHITE, card.x + 10, card.y + 10, card.w - 20)
            self.draw_text_ellipsis("O placar só abre no fim.", self.f_tiny, state_color, card.x + 10, card.y + 43, card.w - 20)
            pulse_w = card.w - 20
            pulse = clamp((math.sin(self.t * 4.2) + 1.0) * 0.5)
            pygame.draw.rect(self.screen, (31, 45, 53), (card.x + 10, card.y + 79, pulse_w, 7), border_radius=5)
            pygame.draw.rect(self.screen, state_color, (card.x + 10, card.y + 79, int(pulse_w * pulse), 7), border_radius=5)

        final_home, final_away = self.final_score_from_prediction(pred)
        revealed = self.match_result_revealed()
        top_score = pred.top_scores[0] if pred.top_scores else (final_home, final_away, pred.score_probability)
        if revealed:
            if analysis is not None:
                classifier_detail = f"{self.home.code} {self.fmt_pct(pred.home)} / E {self.fmt_pct(pred.draw)} / {self.away.code} {self.fmt_pct(pred.away)}"
                poisson_detail = f"Pico {top_score[0]} x {top_score[1]} ({self.fmt_pct(top_score[2], 1)})"
            else:
                classifier_detail = f"{self.home.code} {self.fmt_pct(pred.home)} | EMP {self.fmt_pct(pred.draw)} | {self.away.code} {self.fmt_pct(pred.away)}"
                poisson_detail = f"Mais forte: {top_score[0]} x {top_score[1]} ({self.fmt_pct(top_score[2], 1)})"
            hybrid_detail = f"{final_home} x {final_away} | chance {self.fmt_pct(pred.score_probability, 1)}"
        else:
            classifier_detail = "Forma, camisa e mando"
            poisson_detail = "Placar guardado"
            hybrid_detail = ""
        if cinematic_focus:
            active_goal = self.ball_goal_event(pred)
            attacking_side = active_goal[1] if active_goal else self.cinematic_possession_side(pred)
            attacking_code = self.home.code if attacking_side == "home" else self.away.code
            draw_flow_row(
                1,
                panel.y + 146,
                "XGBoost 1X2",
                f"{attacking_code} acelera",
                CYAN,
                classifier_detail,
            )
            draw_flow_row(
                2,
                panel.y + 211,
                "Poisson/DC",
                "Placar provável",
                GOLD,
                poisson_detail,
            )
            draw_flow_row(
                3,
                panel.y + 276,
                "Sorteio da Copa",
                "Futebol decide",
                GREEN,
                hybrid_detail,
            )
            if revealed:
                draw_top_scores(panel.y + 356)
            else:
                draw_audit_pending(panel.y + 370)
            return
        draw_flow_row(
            1,
            panel.y + 146,
            "XGBoost 1X2",
            "Quem chega melhor",
            CYAN,
            classifier_detail,
        )
        draw_flow_row(
            2,
            panel.y + 211,
            "Poisson/DC",
            "Quais placares vivem",
            GOLD,
            poisson_detail,
        )
        draw_flow_row(
            3,
            panel.y + 276,
            "Sorteio da Copa",
            "Um resultado sai",
            GREEN,
            hybrid_detail,
        )
        if revealed:
            draw_top_scores(panel.y + 356)
        else:
            draw_audit_pending(panel.y + 370)

    def draw_score_panel(
        self,
        predictions: dict[str, Prediction],
        current: str,
        result_pred: Prediction,
        cinematic_focus: bool = False,
    ) -> None:
        pred = predictions[current]
        home_score, away_score = self.score_from_prediction(result_pred)
        panel = self.match_score_panel_rect()
        revealed = self.match_result_revealed()
        state_key = self.match_hud_state_key(cinematic_focus)
        state_label, state_title, state_hint = self.match_hud_state_copy(cinematic_focus)
        state_color = {"live": GOLD, "focus": CYAN, "closed": GREEN}[state_key]
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=18)
        pygame.draw.rect(self.screen, state_color if current == "CONFRONTO" else ALGORITHM_COLORS[current], panel, 2, border_radius=18)
        left = panel.x + 28
        right = panel.right - 28
        score = f"{self.home.code} {home_score} x {away_score} {self.away.code}"
        score_label = "PLACAR FINAL" if revealed else "PLACAR AO VIVO"
        if cinematic_focus:
            score_font = self.fit_font(score, 38, 276, min_size=28)
            self.draw_text_ellipsis(score_label, self.f_xs, state_color, left, panel.y + 24, 260)
            self.draw_text_ellipsis(score, score_font, WHITE, left, panel.y + 48, 276)
            self.draw_text_ellipsis(self.elapsed_label(), self.f_xs, MUTED, left, panel.y + 92, 260)
            center_x = panel.centerx - 20
            center_font = self.fit_font(state_label, 20, 340, min_size=16)
            self.draw_text_centered(state_label, center_font, WHITE, (center_x, panel.y + 42))
            self.draw_text_centered(state_hint, self.f_xs, MUTED, (center_x, panel.y + 74))
            pulse_w = 330
            pulse_x = center_x - pulse_w // 2
            pygame.draw.rect(self.screen, (44, 58, 66), (pulse_x, panel.y + 96, pulse_w, 8), border_radius=5)
            pygame.draw.rect(self.screen, state_color, (pulse_x, panel.y + 96, int(pulse_w * clamp((math.sin(self.t * 4.5) + 1) * 0.5)), 8), border_radius=5)
            score_x = panel.right - 318
            score_w = right - score_x
            if revealed:
                self.draw_text_ellipsis(state_label, self.f_xs, state_color, score_x, panel.y + 24, score_w)
                self.draw_text_ellipsis(state_title, self.f_xs, WHITE, score_x, panel.y + 50, score_w)
                self.draw_text_ellipsis(state_hint, self.f_xs, MUTED, score_x, panel.y + 78, score_w)
                self.draw_text_ellipsis(f"Chance do placar: {self.fmt_pct(result_pred.score_probability, 1)}", self.f_xs, CYAN, score_x, panel.y + 98, score_w)
            else:
                self.draw_text_ellipsis(state_label, self.f_xs, state_color, score_x, panel.y + 24, score_w)
                self.draw_text_ellipsis(state_title, self.f_sm, WHITE, score_x, panel.y + 51, score_w)
                self.draw_text_ellipsis(state_hint, self.f_xs, MUTED, score_x, panel.y + 82, score_w)
            return
        live_w = 244
        self.draw_text_ellipsis(score_label, self.f_sm, state_color, left, panel.y + 22, live_w)
        score_font = self.fit_font(score, 34, live_w, min_size=24)
        self.draw_text_ellipsis(score, score_font, WHITE, left, panel.y + 50, live_w)
        self.draw_text_ellipsis(self.elapsed_label(), self.f_xs, MUTED, left, panel.y + 88, live_w)

        live_home, live_draw, live_away = self.live_probs(pred)
        bars = [(self.home.code, live_home, GREEN), ("EMPATE", live_draw, GOLD), (self.away.code, live_away, RED)]
        leader_value = max(live_home, live_draw, live_away)
        probs_x = panel.x + 306
        col_w = 158
        gap = 38
        for i, (label, value, color) in enumerate(bars):
            x = probs_x + i * (col_w + gap)
            self.draw_text_centered(label, self.f_sm, WHITE, (x + col_w // 2, panel.y + 38))
            self.draw_hud_bar(pygame.Rect(x, panel.y + 60, col_w, 18), value, color)
            if revealed:
                value_label = self.fmt_pct(value, 1)
            elif value == leader_value and value >= 0.42:
                value_label = "puxa o jogo"
            elif label == "EMPATE" and value >= 0.25:
                value_label = "empate quente"
            elif value >= 0.30:
                value_label = "ainda vivo"
            elif value >= 0.12:
                value_label = "precisa virar"
            else:
                value_label = "zebra"
            self.draw_text_centered(value_label, self.f_md if revealed else self.f_sm, color, (x + col_w // 2, panel.y + 102))
        score_x = panel.right - 318
        score_w = right - score_x
        if revealed:
            self.draw_text_ellipsis(state_label, self.f_xs, state_color, score_x, panel.y + 24, score_w)
            self.draw_text_ellipsis(state_title, self.f_xs, WHITE, score_x, panel.y + 50, score_w)
            self.draw_text_ellipsis(state_hint, self.f_xs, MUTED, score_x, panel.y + 78, score_w)
            self.draw_text_ellipsis(f"Chance do placar: {self.fmt_pct(result_pred.score_probability, 1)}", self.f_xs, CYAN, score_x, panel.y + 98, score_w)
        else:
            self.draw_text_ellipsis(state_label, self.f_xs, state_color, score_x, panel.y + 24, score_w)
            self.draw_text_ellipsis(state_title, self.f_sm, WHITE, score_x, panel.y + 50, score_w)
            self.draw_text_ellipsis(state_hint, self.f_xs, MUTED, score_x, panel.y + 82, score_w)

    def emit_match_audio_events(self, result_pred: Prediction, previous_minute: float | None = None) -> None:
        minute = self.match_minute_float()
        for goal_minute, side in self.goal_schedule(result_pred):
            audio_thresholds = self.cinematic_poc_audio_thresholds(
                CinematicAttackEvent(
                    goal_minute,
                    side,
                    True,
                    "goal",
                )
            )
            kick_audio_at = audio_thresholds["kick"]
            whoosh_audio_at = audio_thresholds["whoosh"]
            impact_audio_at = audio_thresholds["net"]
            reverb_audio_at = audio_thresholds["reverb"]
            goal_audio_key = (result_pred.algorithm, goal_minute, side)
            previous_progress = self.shot_progress_cursor.get(goal_audio_key, 0.0)
            in_window = goal_minute - GOAL_EVENT_WINDOW_MINUTES <= minute <= goal_minute + GOAL_PAYOFF_MINUTES
            crossed_window = (
                previous_minute is not None
                and previous_minute <= goal_minute + GOAL_PAYOFF_MINUTES
                and minute >= goal_minute - GOAL_EVENT_WINDOW_MINUTES
            )
            catching_up = (
                0.0 < previous_progress < reverb_audio_at
                and not in_window
            )
            if not (in_window or crossed_window or catching_up):
                continue
            shot_progress = max(
                0.0,
                (
                    minute
                    - (goal_minute - GOAL_EVENT_WINDOW_MINUTES)
                )
                / GOAL_EVENT_WINDOW_MINUTES,
            )
            if (crossed_window and minute > goal_minute + GOAL_PAYOFF_MINUTES) or catching_up:
                shot_progress = max(
                    shot_progress,
                    reverb_audio_at,
                )
            crossed = [
                (name, threshold)
                for name, threshold in (
                    ("kick", kick_audio_at),
                    ("whoosh", whoosh_audio_at),
                    ("net", impact_audio_at),
                    ("bass", audio_thresholds["bass"]),
                    ("cheer", audio_thresholds["cheer"]),
                    ("reverb", reverb_audio_at),
                )
                if previous_progress < threshold <= shot_progress + 1e-9
            ]
            if len(crossed) > 1 or shot_progress - previous_progress > 0.18:
                crossed_names = {name for name, _threshold in crossed}
                synchronized_impact_events = set(GOAL_IMPACT_AUDIO_EVENTS)
                if shot_progress + 1e-9 >= reverb_audio_at:
                    synchronized_impact_events.add("reverb")
                if (
                    shot_progress + 1e-9 >= impact_audio_at
                    and crossed_names & GOAL_IMPACT_AUDIO_EVENTS
                ):
                    crossed = [
                        (name, threshold)
                        for name, threshold in crossed
                        if name in synchronized_impact_events
                    ]
                else:
                    crossed = crossed[:1]
            if (
                previous_progress < kick_audio_at
                and shot_progress >= kick_audio_at - 0.09
            ):
                self.sound.duck_commentary(0.55)
            goal_pan = 0.34 if side == "home" else -0.34
            emitted_threshold = previous_progress
            for name, threshold in crossed:
                event_key = (result_pred.algorithm, goal_minute, side, name)
                if event_key not in self.shot_events:
                    self.queue_match_audio_event(name, goal_pan)
                    self.shot_events.add(event_key)
                    emitted_threshold = max(emitted_threshold, threshold)
            if crossed:
                self.shot_progress_cursor[goal_audio_key] = emitted_threshold
            else:
                self.shot_progress_cursor[goal_audio_key] = max(previous_progress, shot_progress)
            goal_key = (result_pred.algorithm, goal_minute, side)
            if shot_progress >= reverb_audio_at:
                self.goal_events.add(goal_key)
        for chance_minute, side, kind in self.chance_schedule(result_pred):
            chance_event = CinematicAttackEvent(
                chance_minute,
                side,
                False,
                kind,
            )
            audio_thresholds = self.cinematic_poc_audio_thresholds(
                chance_event
            )
            kick_audio_at = audio_thresholds["kick"]
            whoosh_audio_at = audio_thresholds["whoosh"]
            contact_name = (
                "save"
                if kind == "save"
                else "near_miss"
            )
            contact_audio_at = audio_thresholds[contact_name]
            chance_audio_key = (result_pred.algorithm, chance_minute, f"{side}:{kind}")
            previous_progress = self.shot_progress_cursor.get(chance_audio_key, 0.0)
            in_window = chance_minute - CHANCE_EVENT_WINDOW_MINUTES <= minute <= chance_minute + CHANCE_PAYOFF_MINUTES
            catching_up = (
                0.0 < previous_progress < contact_audio_at
                and not in_window
            )
            if not in_window and not catching_up:
                continue
            shot_progress = max(
                0.0,
                (
                    minute
                    - (chance_minute - CHANCE_EVENT_WINDOW_MINUTES)
                )
                / CHANCE_EVENT_WINDOW_MINUTES,
            )
            if catching_up:
                shot_progress = max(
                    shot_progress,
                    contact_audio_at,
                )
            crossed = [
                (name, threshold)
                for name, threshold in (
                    ("kick", kick_audio_at),
                    ("whoosh", whoosh_audio_at),
                    (contact_name, contact_audio_at),
                )
                if previous_progress < threshold <= shot_progress + 1e-9
            ]
            if len(crossed) > 1 or shot_progress - previous_progress > 0.18:
                if (
                    shot_progress + 1e-9 >= contact_audio_at
                    and any(
                        name in {"save", "near_miss"}
                        for name, _threshold in crossed
                    )
                ):
                    crossed = [
                        (name, threshold)
                        for name, threshold in crossed
                        if name in {"save", "near_miss"}
                    ]
                else:
                    crossed = crossed[:1]
            if shot_progress >= kick_audio_at - 0.09:
                self.sound.duck_commentary(0.40)
            chance_pan = 0.26 if side == "home" else -0.26
            emitted_threshold = previous_progress
            for name, threshold in crossed:
                event_key = (result_pred.algorithm, chance_minute, f"{side}:{kind}", name)
                if event_key not in self.shot_events:
                    self.queue_match_audio_event(name, chance_pan)
                    self.shot_events.add(event_key)
                    emitted_threshold = max(emitted_threshold, threshold)
            if crossed:
                self.shot_progress_cursor[chance_audio_key] = emitted_threshold
            else:
                self.shot_progress_cursor[chance_audio_key] = max(previous_progress, shot_progress)

    def shot_cursor_completion_threshold(self, key: tuple[object, ...]) -> float:
        if len(key) < 3:
            return 1.0
        event_minute = int(key[1])
        event_kind = str(key[2])
        if event_kind in {"home", "away"}:
            thresholds = self.cinematic_poc_audio_thresholds(
                CinematicAttackEvent(
                    event_minute,
                    event_kind,
                    True,
                    "goal",
                )
            )
            return thresholds["reverb"]
        if ":" in event_kind:
            side, outcome = event_kind.split(":", 1)
            if (
                side in {"home", "away"}
                and outcome in {"save", "wide"}
            ):
                thresholds = self.cinematic_poc_audio_thresholds(
                    CinematicAttackEvent(
                        event_minute,
                        side,
                        False,
                        outcome,
                    )
                )
                terminal = (
                    "save"
                    if outcome == "save"
                    else "near_miss"
                )
                return thresholds[terminal]
        return 1.0

    def queue_match_audio_event(self, name: str, pan: float) -> None:
        self.arm_queued_match_audio_event(name)
        self.match_audio_frame_queue.append((name, pan))

    def arm_queued_match_audio_event(self, name: str) -> None:
        self.sound.arm_event(name)

    def flush_queued_match_audio(self) -> None:
        if not self.match_audio_frame_queue:
            return
        queued = self.match_audio_frame_queue
        self.match_audio_frame_queue = []
        for name, pan in queued:
            self.sound.play(name, pan=pan, already_armed=True)

    def has_pending_match_audio(self) -> bool:
        if self.match_audio_frame_queue:
            return True
        return any(
            progress < self.shot_cursor_completion_threshold(key) - 1e-6
            for key, progress in self.shot_progress_cursor.items()
        )

    def drain_monte_carlo_queue(self) -> None:
        while True:
            try:
                message = self.mc_queue.get_nowait()
            except queue.Empty:
                return
            kind = str(message[0])
            generation = int(message[1])
            if generation != self.mc_generation:
                continue
            if kind == "progress":
                _kind, _generation, done, total, *_odds = message
                self.mc_progress_done = int(done)
                self.mc_progress_total = int(total)
            elif kind == "result":
                _kind, _generation, odds, representative = message
                self.mc_pending_result = (list(odds), dict(representative) if representative is not None else None)
                self.apply_pending_monte_carlo_result_if_ready()
            elif kind == "error":
                _kind, _generation, error_text = message
                self.mc_error = str(error_text)
                self.mc_running = False

    def apply_pending_monte_carlo_result_if_ready(self) -> None:
        if self.mc_pending_result is None:
            return
        elapsed = self.t - self.mc_started_t
        if elapsed < TOURNAMENT_MIN_LOADING_SECONDS:
            return
        odds, representative = self.mc_pending_result
        self.champion_odds = list(odds)
        if representative is not None:
            self.tournament_result = dict(representative)
            self.tournament_reveal_t = self.t
            self.cup_reveal_audio_pending = True
        self.mc_progress_done = self.mc_progress_total
        self.mc_pending_result = None
        self.mc_running = False

    def update_tournament_audio(self) -> None:
        if self.state != "tournament":
            return
        if self.cup_start_audio_pending:
            self.sound.play("cup_start")
            self.cup_start_audio_pending = False
        total = max(1, self.mc_progress_total)
        done = int(round(total * self.monte_carlo_progress()))
        for marker in CUP_PROGRESS_MARKERS:
            if marker not in self.cup_audio_markers and done * 100 >= total * marker:
                self.sound.play("cup_tick")
                self.cup_audio_markers.add(marker)
        if self.cup_reveal_audio_pending and not self.cup_reveal_audio_played:
            self.sound.play("cup_reveal")
            self.cup_reveal_audio_pending = False
            self.cup_reveal_audio_played = True

    def update(self, dt: float) -> None:
        dt = clamp(dt, 0.0, MAX_FRAME_DT)
        self.drain_cinematic_poc_preload()
        cinematic_preload_waiting = (
            self.state == "simulate"
            and not self.poc_preload_ready
        )
        self.drain_monte_carlo_queue()
        if self.pending_tournament_seed is not None:
            if self.mc_thread is not None and self.mc_thread.is_alive():
                self.mc_cancel_event.set()
            else:
                seed = self.pending_tournament_seed
                self.pending_tournament_seed = None
                self.start_champion_odds_job(seed=seed)
        previous_minute = self.match_minute_float() if self.state == "simulate" else None
        if self.state == "simulate":
            if (
                self.match_intro_audio_pending
                and not cinematic_preload_waiting
            ):
                self.sound.play("ui_chime")
                self.sound.play("whistle")
                self.match_intro_audio_pending = False
            if not cinematic_preload_waiting:
                self.cinematic_render_dt = max(1.0 / FPS, dt)
                remaining = dt
                while remaining > 1e-9 and self.t < SIMULATION_SECONDS:
                    step = min(1.0 / FPS, remaining)
                    self.t = min(self.t + step, SIMULATION_SECONDS)
                    self.update_cinematic_scroll(step)
                    remaining -= step
                segment = int(self.t // self.segment_duration())
                if segment != self.segment_started:
                    self.segment_started = segment
        elif self.state == "tournament":
            self.t += dt
            self.apply_pending_monte_carlo_result_if_ready()
        self.update_tournament_audio()
        self.update_soundscape(dt, previous_minute=previous_minute)
        if self.state == "simulate" and self.t >= SIMULATION_SECONDS and not self.final_whistle_played:
            if not self.has_pending_match_audio():
                self.sound.play("final_whistle")
                self.final_whistle_played = True

    def update_cinematic_scroll(self, dt: float) -> None:
        pred = self.match_prediction
        if pred is None or self.simulation_progress() >= 1.0:
            self.ground_scroll_velocity += (0.0 - self.ground_scroll_velocity) * clamp(dt * 5.0)
            return
        active_attack = self.active_attack_event(pred)
        motion = self.cinematic_motion_state(pred)
        target_velocity = float(motion.get("desired_scroll_velocity", 0.0))
        self.ground_scroll_velocity += (target_velocity - self.ground_scroll_velocity) * clamp(dt * 3.8)
        self.ground_scroll += self.ground_scroll_velocity * dt
        self.ground_travel_distance += (
            abs(self.ground_scroll_velocity) * CINEMATIC_TURF_FOREGROUND_PARALLAX * dt
        )
        if active_attack:
            event_key = (
                active_attack.minute,
                active_attack.side,
                active_attack.is_goal,
                active_attack.kind,
            )
            self.cinematic_attack_phase_anchors.setdefault(
                event_key,
                (self.ground_travel_distance / CINEMATIC_RUNNER_STRIDE_DISTANCE * 4.0) % 4.0,
            )

    def update_soundscape(self, dt: float, previous_minute: float | None = None) -> None:
        self.sound.set_scene(self.state)
        if self.state != "simulate":
            ambience = 0.34 if self.state == "menu" else 0.24 if self.state == "tournament" else 0.18
            self.sound.update_crowd(ambience, False, dt, allow_reactions=False)
            return
        pred = self.match_prediction
        if pred is None:
            self.sound.update_crowd(0.18, False, dt)
            return
        minute = self.match_minute_float()
        active_goal = self.ball_goal_event(pred)
        home, draw, away = self.live_probs(pred)
        drama = 1.0 - abs(home - away)
        intensity = 0.18 + 0.36 * self.simulation_progress() + 0.14 * clamp(draw + drama * 0.5)
        if active_goal:
            goal_minute, _side = active_goal
            shot_progress = clamp((minute - (goal_minute - 5.0)) / 5.0)
            intensity = max(intensity, 0.54 + 0.42 * shot_progress)
            if SHOT_KICK_AT - 0.10 <= shot_progress < SHOT_KICK_AT:
                self.sound.duck_commentary(0.45)
                self.sound.suppress_reactions_until_ms = max(
                    self.sound.suppress_reactions_until_ms,
                    pygame.time.get_ticks() + 420,
                )
        else:
            upcoming = [goal_minute for goal_minute, _side in self.goal_schedule(pred) if goal_minute >= minute]
            if upcoming:
                distance = min(upcoming) - minute
                if 0 <= distance <= 14:
                    intensity += 0.24 * (1 - distance / 14)
        self.emit_match_audio_events(pred, previous_minute=previous_minute)
        allow_reactions = not (self.t >= SIMULATION_SECONDS and not self.final_whistle_played)
        self.sound.update_crowd(clamp(intensity), active_goal is not None, dt, allow_reactions=allow_reactions)

    def draw(self) -> None:
        if self.state == "menu":
            self.draw_menu()
        elif self.state == "select":
            self.draw_select()
        elif self.state == "simulate":
            self.draw_simulate()
        else:
            self.draw_tournament()
        pygame.display.flip()

    def draw_tournament(self) -> None:
        self.screen.fill(BG)
        title = "Monte Carlo da Copa 2026" if not self.tournament_result else "Cenário da Copa 2026"
        hint = "BACKSPACE volta | T nova amostra" if not self.tournament_result else ""
        self.draw_top(title, hint)
        self.draw_tournament_background()
        if not self.tournament_result:
            self.draw_tournament_loading()
            return
        self.draw_tournament_tabs()
        self.draw_tournament_shortcuts()
        self.draw_tournament_result_header()
        if self.tournament_view == "bracket":
            self.draw_knockout_page()
        else:
            self.draw_groups_page()

    def draw_tournament_background(self) -> None:
        background_rect = pygame.Rect(0, 88, WIDTH, 536)
        image_key = "mexico_opening"
        if self.tournament_result:
            image_key = "club_final" if self.tournament_view == "bracket" else "detail"
        external = self.assets.fifa_images.get(image_key)
        if external:
            self.draw_cover_image(external, background_rect)
        elif self.assets.stadium_bg:
            self.screen.blit(self.cached_smoothscale(self.assets.stadium_bg, background_rect.size), background_rect.topleft)
        shade_alpha = 174 if self.tournament_result else 156
        shade_key = ("tournament_shade", WIDTH, HEIGHT - 88, shade_alpha)
        shade = self.cinematic_overlay_cache.get(shade_key)
        if shade is None:
            shade = pygame.Surface((WIDTH, HEIGHT - 88), pygame.SRCALPHA)
            shade.fill((0, 8, 13, shade_alpha))
            self.cinematic_overlay_cache[shade_key] = shade
        self.screen.blit(shade, (0, 88))
        vignette_key = ("tournament_vignette", background_rect.w, background_rect.h)
        vignette = self.cinematic_overlay_cache.get(vignette_key)
        if vignette is None:
            vignette = pygame.Surface(background_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(vignette, (0, 0, 0, 82), vignette.get_rect(), width=36)
            self.cinematic_overlay_cache[vignette_key] = vignette
        self.screen.blit(vignette, background_rect.topleft)
        if not self.tournament_result:
            pygame.draw.rect(self.screen, (0, 0, 0, 92), (0, HEIGHT - 118, WIDTH, 118))

    def draw_tournament_tabs(self) -> None:
        tabs = (
            (self.group_tab_rect, "FASE DE GRUPOS", "groups"),
            (self.bracket_tab_rect, "MATA-MATA", "bracket"),
        )
        for rect, label, view in tabs:
            active = self.tournament_view == view
            fill = (24, 68, 82) if active else (14, 36, 48)
            border = GOLD if active else (72, 111, 127)
            pygame.draw.rect(self.screen, fill, rect, border_radius=12)
            pygame.draw.rect(self.screen, border, rect, 2, border_radius=12)
            self.draw_text_centered(label, self.f_xs, WHITE if active else MUTED, rect.center)

    def draw_tournament_shortcuts(self) -> None:
        hint = "BACKSPACE volta | T/R nova | G grupos | M chave"
        hint_x = 244
        hint_width = self.group_tab_rect.x - hint_x - 24
        hint_font = self.fit_font(hint, 16, hint_width, min_size=13)
        self.draw_text_ellipsis(hint, hint_font, MUTED, hint_x, 68, hint_width)

    def draw_tournament_loading(self) -> None:
        panel = pygame.Rect(128, 142, 1024, 472)
        pygame.draw.rect(self.screen, (5, 17, 25, 230), panel, border_radius=26)
        pygame.draw.rect(self.screen, GOLD, panel, 3, border_radius=26)
        left = panel.x + 50
        copy_w = 560
        progress = clamp(self.monte_carlo_progress())
        revealing = self.tournament_revealing()
        title = "REVELANDO CENÁRIO" if revealing else "SIMULANDO A COPA"
        self.draw_text_ellipsis(title, self.f_lg, GOLD, left, panel.y + 42, copy_w)
        loading_copy = (
            "Fechando ranking, grupos e campanha da amostra."
            if revealing
            else "Calculando amostra completa antes do reveal."
        )
        self.draw_text_ellipsis(loading_copy, self.f_sm, WHITE, left, panel.y + 96, copy_w)
        run_mode = "Copas ao vivo" if not TOURNAMENT_MONTE_CARLO_USE_SCENARIO_BANK else "reamostras do banco turbo"
        run_text = f"{self.champion_odds_runs} {run_mode}, em segundo plano."
        self.draw_text_ellipsis(run_text, self.f_sm, MUTED, left, panel.y + 128, copy_w)

        track = pygame.Rect(left + 8, panel.y + 292, 540, 18)
        self.draw_hud_bar(track, progress, GREEN, bg=(36, 51, 61))
        shine_x = track.x + int(track.w * ((self.t * 0.42) % 1.0))
        pygame.draw.circle(self.screen, (82, 226, 255, 130), (shine_x, track.centery), 9)
        if self.assets.balls:
            ball = self.assets.balls[int(self.t * 12) % len(self.assets.balls)]
            ball = self.cached_smoothscale(ball, (58, 58))
            ball_x = track.x + int(track.w * progress)
            bounce = math.sin(self.t * 8.0) * 6
            self.screen.blit(ball, ball.get_rect(center=(ball_x, track.y - 22 + int(bounce))))
        self.draw_text_centered(self.fmt_pct(progress), self.f_md, WHITE, (track.centerx, track.y + 56))
        status = self.mc_error if self.mc_error else (
            "Escolhendo uma campanha representativa da amostra."
            if revealing
            else "Rodando fase de grupos, mata-mata e finais."
        )
        self.draw_text_centered(self.ellipsize(status, self.f_sm, 660), self.f_sm, RED if self.mc_error else CYAN, (track.centerx, panel.y + 386))
        if self.assets.fifa_images:
            self.draw_tournament_mascot_stage(panel)
            return
        for index, label in enumerate(("1X2/XGBoost estima tendência", "Poisson/DC sorteia placares", "Monte Carlo compara cenários")):
            x = panel.x + 86 + index * 300
            pygame.draw.rect(self.screen, PANEL_2, (x, panel.y + 404, 250, 44), border_radius=12)
            self.draw_text_centered(label, self.f_xs, GOLD if index == 2 else WHITE, (x + 125, panel.y + 426))

    def draw_tournament_mascot_stage(self, panel: pygame.Rect) -> None:
        cards = (
            ("maple", "Grupos em rotação"),
            ("zayu", "Chaves em cálculo"),
            ("clutch", "Final em segredo"),
        )
        progress = clamp(self.monte_carlo_progress())
        revealing = self.tournament_revealing()
        if revealing:
            active_index = 2
            active_label = "Revelando campeão"
        elif progress < 0.34:
            active_index = 0
            active_label = cards[0][1]
        elif progress < 0.72:
            active_index = 1
            active_label = cards[1][1]
        else:
            active_index = 2
            active_label = cards[2][1]
        active_key = cards[active_index][0]
        hero = pygame.Rect(panel.right - 412, panel.y + 58 + int(math.sin(self.t * 1.8) * 4), 344, 224)
        pygame.draw.rect(self.screen, (4, 15, 23), hero.inflate(14, 14), border_radius=22)
        active_image = self.assets.fifa_images.get(active_key)
        if active_image:
            self.draw_cover_image(active_image, hero, alpha=245)
        hero_shade = self.cached_filled_overlay(("tournament_hero_shade",), hero.size, (0, 5, 10, 34))
        self.screen.blit(hero_shade, hero.topleft)
        pygame.draw.rect(self.screen, GOLD, hero, 3, border_radius=18)
        title_band = self.cached_filled_overlay(("tournament_title_band",), (hero.w, 42), (0, 8, 13, 166))
        self.screen.blit(title_band, (hero.x, hero.bottom - 42))
        self.draw_text_centered(active_label, self.f_sm, WHITE, (hero.centerx, hero.bottom - 22))

        for index, (key, label) in enumerate(cards):
            image = self.assets.fifa_images.get(key)
            if not image:
                continue
            card = pygame.Rect(panel.right - 410 + index * 112, panel.y + 330, 100, 72)
            pygame.draw.rect(self.screen, (8, 24, 34), card.inflate(8, 8), border_radius=14)
            inner = card.inflate(-4, -4)
            self.draw_cover_image(image, inner, alpha=245 if index == active_index else 168)
            veil = self.cached_filled_overlay(("tournament_mascot_veil", index == active_index), inner.size, (0, 0, 0, 28 if index == active_index else 92))
            self.screen.blit(veil, inner.topleft)
            pygame.draw.rect(self.screen, GOLD if index == active_index else CYAN, card, 2, border_radius=12)
            self.draw_text_centered(label.split()[0], self.f_tiny, WHITE, (card.centerx, card.bottom + 16))

    def draw_tournament_result_header(self) -> None:
        panel = pygame.Rect(48, 104, 1184, 118)
        pygame.draw.rect(self.screen, (5, 17, 25, 228), panel, border_radius=22)
        pygame.draw.rect(self.screen, GOLD, panel, 2, border_radius=22)
        story_key = self.tournament_story_team() or str(self.tournament_result.get("champion", ""))
        rank, odds = self.champion_rank_and_odds(story_key)
        leader = self.champion_odds[0] if self.champion_odds else (story_key, 0, 0.0)
        final = self.tournament_final()
        card_gap = 14
        ranking_rect = pygame.Rect(panel.x + 16, panel.y + 14, 424, 90)
        favorite_rect = pygame.Rect(ranking_rect.right + card_gap, panel.y + 14, 288, 90)
        story_rect = pygame.Rect(favorite_rect.right + card_gap, panel.y + 14, panel.right - favorite_rect.right - card_gap - 16, 90)
        leader_wins = int(round(float(leader[2]) * self.champion_odds_runs))
        rank_label = f"#{rank}" if rank else "fora do ranking"
        self.draw_tournament_info_card(ranking_rect, "CAMINHOS MAIS FORTES", GOLD)
        top_parts = []
        for team, _wins, pct in self.champion_odds[:5]:
            code = self.team_code(team)
            marker = "*" if str(team) == story_key else ""
            top_parts.append(f"{marker}{code} {self.fmt_pct(float(pct), 1)}")
        top_line = "Caminhos fortes: " + "  ".join(top_parts)
        top_font = self.fit_font(top_line, 15, ranking_rect.w - 28, min_size=11)
        self.draw_text_ellipsis(top_line, top_font, WHITE, ranking_rect.x + 14, ranking_rect.y + 34, ranking_rect.w - 28)
        self.draw_text_ellipsis(
            f"{self.champion_odds_runs} Copas simuladas; faixa mostra incerteza.",
            self.f_tiny,
            MUTED,
            ranking_rect.x + 14,
            ranking_rect.y + 62,
            ranking_rect.w - 28,
        )

        self.draw_tournament_info_card(favorite_rect, "FAVORITO DO ORÁCULO", CYAN)
        self.draw_text_midleft(self.team_code(leader[0]), self.f_lg, WHITE, (favorite_rect.x + 14, favorite_rect.y + 55))
        self.draw_text_ellipsis(
            f"{self.fmt_mc_pct(float(leader[2]))}",
            self.f_sm,
            CYAN,
            favorite_rect.x + 118,
            favorite_rect.y + 34,
            favorite_rect.w - 132,
        )
        self.draw_text_ellipsis(
            f"{leader_wins} Copas vencidas",
            self.f_tiny,
            MUTED,
            favorite_rect.x + 118,
            favorite_rect.y + 62,
            favorite_rect.w - 132,
        )

        self.draw_tournament_info_card(story_rect, "HISTÓRIA DA SIMULAÇÃO", GREEN)
        self.draw_text_midleft(self.team_code(story_key), self.f_lg, WHITE, (story_rect.x + 14, story_rect.y + 55))
        story_name = f"{rank_label} no ranking: {self.team_name(story_key)}"
        story_font = self.fit_font(story_name, 20, story_rect.w - 136, min_size=13)
        self.draw_text_ellipsis(story_name, story_font, WHITE, story_rect.x + 116, story_rect.y + 30, story_rect.w - 136)
        if final:
            final_text = f"Final: {self.team_code(final['home'])} {int(final['home_goals'])} x {int(final['away_goals'])} {self.team_code(final['away'])}"
            self.draw_text_ellipsis(final_text, self.f_sm, GOLD, story_rect.x + 116, story_rect.y + 54, story_rect.w - 136)
        phrase = self.tournament_plausibility_phrase()
        phrase_font = self.fit_font(phrase, 13, story_rect.w - 28, min_size=10)
        self.draw_text_ellipsis(phrase, phrase_font, CYAN, story_rect.x + 14, story_rect.y + 76, story_rect.w - 28)

    def draw_tournament_info_card(self, rect: pygame.Rect, title: str, accent: tuple[int, int, int]) -> None:
        pygame.draw.rect(self.screen, (8, 25, 35, 226), rect, border_radius=16)
        pygame.draw.rect(self.screen, accent, rect, 1, border_radius=16)
        self.draw_text_ellipsis(title, self.f_tiny, accent, rect.x + 14, rect.y + 10, rect.w - 28)

    def team_profile_for_key(self, team_key: object) -> TeamProfile | None:
        key = str(team_key)
        return next((team for team in self.teams if team.key == key or team.code == key), None)

    def draw_team_badge(self, team_key: object, x: int, y: int, active: bool = False) -> None:
        code = self.team_code(team_key)
        flag = self.assets.flags.get(code)
        if flag:
            self.screen.blit(self.cached_smoothscale(flag, (28, 18)), (x, y + 1))
        color = GREEN if active else WHITE
        self.draw_text(code, self.f_xs, color, x + 34, y)

    def draw_groups_page(self) -> None:
        groups: dict[str, list[dict[str, object]]] = {}
        for row in self.tournament_result["standings"]:
            groups.setdefault(str(row["group"]), []).append(row)
        for rows in groups.values():
            rows.sort(key=lambda item: int(item.get("rank", 99)))
        thirds = self.qualified_thirds()
        start_x, start_y = 48, 230
        card_w, card_h = 282, 140
        gap_x, gap_y = 22, 12
        for index, group in enumerate(sorted(groups)):
            col = index % 4
            row = index // 4
            rect = pygame.Rect(start_x + col * (card_w + gap_x), start_y + row * (card_h + gap_y), card_w, card_h)
            self.draw_group_card(rect, group, groups[group], thirds)

    def draw_group_card(self, rect: pygame.Rect, group: str, rows: list[dict[str, object]], thirds: set[str]) -> None:
        pygame.draw.rect(self.screen, (6, 20, 29, 230), rect, border_radius=14)
        pygame.draw.rect(self.screen, (45, 78, 63), rect, 2, border_radius=14)
        pygame.draw.rect(self.screen, (18, 43, 55), (rect.x, rect.y, rect.w, 34), border_top_left_radius=14, border_top_right_radius=14)
        self.draw_text(f"GRUPO {group}", self.f_sm, GOLD, rect.x + 14, rect.y + 7)
        for index, team_row in enumerate(rows[:4]):
            y = rect.y + 44 + index * 23
            team_key = str(team_row["team"])
            rank = int(team_row.get("rank", 99))
            advanced = rank <= 2 or (rank == 3 and team_key in thirds)
            pygame.draw.rect(self.screen, (16, 43, 52) if advanced else (12, 27, 35), (rect.x + 10, y - 2, rect.w - 20, 21), border_radius=7)
            self.draw_text(str(rank), self.f_xs, GOLD if advanced else MUTED, rect.x + 18, y)
            self.draw_team_badge(team_key, rect.x + 42, y, active=advanced)
            self.draw_text_right(f"{int(team_row['pts'])} pts", self.f_xs, GREEN if advanced else MUTED, rect.right - 14, y)

    def round_matches(self, round_name: str) -> list[dict[str, object]]:
        aliases = {
            "Round of 32": {"Round of 32"},
            "Round of 16": {"Round of 16"},
            "Quarter-finals": {"Quarter-finals", "Quarterfinals"},
            "Semi-finals": {"Semi-finals", "Semifinals"},
            "Final": {"Final"},
        }
        names = aliases.get(round_name, {round_name})
        return [dict(row) for row in self.tournament_result["rounds"] if str(row.get("round")) in names]

    def draw_knockout_page(self) -> None:
        champion = self.tournament_story_team() or str(self.tournament_result.get("champion", ""))
        stages = [
            ("Round of 16", "OITAVAS", 430, 258, 170, 8),
            ("Quarter-finals", "QUARTAS", 620, 306, 170, 4),
            ("Semi-finals", "SEMIS", 810, 356, 150, 2),
            ("Final", "FINAL", 984, 368, 238, 1),
        ]
        round32_matches = self.round_matches("Round of 32")
        self.draw_text("FASE DE 32", self.f_sm, GOLD, 44, 228)
        self.draw_text("primeira fase eliminatória", self.f_tiny, MUTED, 164, 232)
        for index, match in enumerate(round32_matches[:16]):
            col = index // 8
            row = index % 8
            x = 44 + col * 186
            y = 254 + row * 30
            self.draw_knockout_match(pygame.Rect(x, y, 172, 24), match, champion)
        for stage, title, x, y, w, expected in stages:
            self.draw_text(title, self.f_sm, GOLD, x, y - 30)
            matches = self.round_matches(stage)
            if stage == "Final":
                self.draw_final_trophy_panel(pygame.Rect(x, y, w, 250), matches[0] if matches else None, champion)
                continue
            spacing = 30 if expected >= 8 else 50
            for index, match in enumerate(matches[:expected]):
                self.draw_knockout_match(pygame.Rect(x, y + index * spacing, w, 24), match, champion)

    def draw_knockout_match(self, rect: pygame.Rect, match: dict[str, object], champion: str) -> None:
        active = str(match.get("winner")) == champion
        pygame.draw.rect(self.screen, (11, 31, 41, 235), rect, border_radius=7)
        pygame.draw.rect(self.screen, GOLD if active else (49, 78, 92), rect, 1, border_radius=7)
        home = self.team_code(match["home"])
        away = self.team_code(match["away"])
        score = f"{home} {int(match['home_goals'])}-{int(match['away_goals'])} {away}"
        self.draw_text_centered(score, self.f_xs, GREEN if active else WHITE, rect.center)

    def draw_final_trophy_panel(self, rect: pygame.Rect, match: dict[str, object] | None, champion: str) -> None:
        pygame.draw.rect(self.screen, (6, 20, 29, 238), rect, border_radius=18)
        pygame.draw.rect(self.screen, GOLD, rect, 3, border_radius=18)
        final_photo = self.assets.fifa_images.get("ecomm")
        if final_photo:
            photo_rect = rect.inflate(-18, -18)
            self.draw_cover_image(final_photo, photo_rect, alpha=112)
            veil = self.cached_filled_overlay(("final_trophy_photo_veil",), photo_rect.size, (0, 10, 16, 140))
            self.screen.blit(veil, photo_rect.topleft)
        if self.trophy_icon:
            self.screen.blit(self.trophy_icon, self.trophy_icon.get_rect(center=(rect.centerx, rect.y + 76)))
        self.draw_text_centered("CAMPEÃO", self.f_sm, GOLD, (rect.centerx, rect.y + 154))
        self.draw_text_centered(self.team_code(champion), self.f_lg, WHITE, (rect.centerx, rect.y + 194))
        if match:
            score = f"{self.team_code(match['home'])} {int(match['home_goals'])} x {int(match['away_goals'])} {self.team_code(match['away'])}"
            self.draw_text_centered(score, self.f_sm, CYAN, (rect.centerx, rect.y + 230))

    def team_code(self, team_key: object) -> str:
        key = str(team_key)
        return next((team.code for team in self.teams if team.key == key), key[:3].upper())

    def team_name(self, team_key: object) -> str:
        key = str(team_key)
        team = next((profile for profile in self.teams if profile.key == key), None)
        if team:
            return TEAM_DISPLAY_NAMES_PT.get(team.code, team.name)
        return TEAM_DISPLAY_NAMES_PT.get(key[:3].upper(), key)

    def tournament_final(self) -> dict[str, object] | None:
        if not self.tournament_result:
            return None
        for row in self.tournament_result["rounds"]:
            if str(row.get("round")) == "Final":
                return row
        return None

    def monte_carlo_progress(self) -> float:
        total = max(1, self.mc_progress_total)
        raw = max(0.0, min(1.0, self.mc_progress_done / total))
        if self.state == "tournament" and not self.tournament_result and not self.mc_error and self.mc_started_t >= 0.0:
            elapsed = max(0.0, self.t - self.mc_started_t)
            paced_cap = min(0.985, (elapsed / TOURNAMENT_MIN_LOADING_SECONDS) * 0.985)
            if self.mc_pending_result is not None or raw >= 0.985:
                return min(raw, paced_cap)
        return raw

    def tournament_revealing(self) -> bool:
        if self.tournament_result or self.mc_error:
            return False
        if self.mc_pending_result is not None:
            return True
        return self.monte_carlo_progress() >= 0.965

    def monte_carlo_favorite_ready(self) -> bool:
        if not self.champion_odds:
            return False
        if self.mc_running or self.mc_error:
            return False
        if self.mc_progress_done < self.mc_progress_total:
            return False
        return bool(self.tournament_result and self.tournament_result.get("representative_for"))

    def tournament_story_team(self) -> str | None:
        if not self.tournament_result:
            return None
        team = self.tournament_result.get("representative_for") or self.tournament_result.get("champion")
        return str(team) if team else None

    def champion_rank_and_odds(self, team_key: str | None) -> tuple[int, float]:
        if not team_key:
            return 0, 0.0
        for rank, (team, _wins, odds) in enumerate(self.champion_odds, start=1):
            if str(team) == team_key:
                return rank, odds
        return 0, 0.0

    def monte_carlo_uncertainty(self, probability: float) -> float:
        runs = max(1, int(self.champion_odds_runs))
        p = clamp(float(probability))
        return 1.96 * math.sqrt(max(0.0, p * (1.0 - p)) / runs)

    def fmt_mc_pct(self, probability: float, digits: int = 1) -> str:
        return f"{self.fmt_pct(probability, digits)} ±{self.fmt_pct(self.monte_carlo_uncertainty(probability), digits)}"

    def tournament_plausibility_phrase(self) -> str:
        if not self.tournament_result:
            return ""
        level = str(self.tournament_result.get("representative_surprise_level", "plausivel"))
        if level == "surpresa_controlada":
            prefix = "Surpresa controlada"
        elif level == "zebra_controlada":
            prefix = "Zebra controlada"
        else:
            prefix = "Plausível"
        runner_rank = int(self.tournament_result.get("representative_runner_up_finalist_rank", 0) or 0)
        runner_note = f"vice #{runner_rank} em finais" if 0 < runner_rank < 999 else "vice veio da amostra"
        final_diff = int(self.tournament_result.get("representative_final_goal_diff", 0) or 0)
        if final_diff <= 1:
            final_note = "final equilibrada"
        elif final_diff <= 2:
            final_note = "final sem goleada"
        else:
            final_note = "final filtrada"
        return f"{prefix}: {runner_note}; {final_note}."

    def qualified_thirds(self) -> set[str]:
        if not self.tournament_result:
            return set()
        top_two = {str(row["team"]) for row in self.tournament_result["standings"] if int(row.get("rank", 0)) <= 2}
        round32_teams: set[str] = set()
        for row in self.tournament_result["rounds"]:
            if str(row.get("round")) == "Round of 32":
                round32_teams.add(str(row["home"]))
                round32_teams.add(str(row["away"]))
        return round32_teams - top_two

    def cancel_champion_odds_job(self) -> None:
        self.mc_generation += 1
        self.mc_cancel_event.set()
        self.pending_tournament_seed = None
        self.mc_pending_result = None
        self.mc_running = False

    def start_champion_odds_job(self, seed: int = 2026) -> None:
        if self.mc_thread is not None and self.mc_thread.is_alive():
            self.mc_cancel_event.set()
            self.pending_tournament_seed = seed
            self.mc_running = True
            return
        self.mc_cancel_event.set()
        self.mc_generation += 1
        generation = self.mc_generation
        cancel_event = threading.Event()
        self.mc_cancel_event = cancel_event
        self.mc_seed = seed
        self.champion_odds = []
        self.mc_progress_done = 0
        self.mc_progress_total = self.champion_odds_runs
        self.mc_running = True
        self.mc_error = ""
        self.mc_pending_result = None
        self.mc_started_t = self.t

        def progress(done: int, total: int, odds: list[tuple[str, int, float]]) -> bool:
            if generation != self.mc_generation or cancel_event.is_set():
                return False
            self.mc_queue.put(("progress", generation, done, total))
            return True

        def worker() -> None:
            try:
                odds, representative = self.model.champion_odds_with_representative(
                    runs=self.champion_odds_runs,
                    seed=seed,
                    workers=self.champion_odds_workers,
                    progress_callback=progress,
                    progress_with_odds=False,
                    use_scenario_bank=TOURNAMENT_MONTE_CARLO_USE_SCENARIO_BANK,
                )
            except Exception as exc:
                if generation == self.mc_generation and not cancel_event.is_set():
                    self.mc_queue.put(("error", generation, str(exc)))
                return
            if generation == self.mc_generation and not cancel_event.is_set():
                self.mc_queue.put(("result", generation, odds, representative))

        self.mc_thread = threading.Thread(target=worker, name="arena-ai-monte-carlo", daemon=True)
        self.mc_thread.start()

    def set_select(self) -> None:
        self.cancel_champion_odds_job()
        self.cancel_cinematic_poc_preload()
        self.sound.stop_one_shots()
        self.sound.reset_scene_queues()
        self.match_audio_frame_queue.clear()
        self.match_prediction = None
        self.match_analysis = None
        self.match_runtime_state_cache.clear()
        self.cinematic_shot_plan_cache.clear()
        self.poc_layer_frame_cache.clear()
        self.poc_layer_cache.clear()
        self.state = "select"

    def set_simulate(self, mode: str) -> None:
        self.cancel_cinematic_poc_preload()
        self.sound.stop_one_shots()
        self.sound.reset_scene_queues()
        self.match_audio_frame_queue.clear()
        self.mode = mode
        self.state = "simulate"
        self.t = 0.0
        self.ground_scroll = 0.0
        self.ground_scroll_velocity = 0.0
        self.ground_travel_distance = 0.0
        self.cinematic_attack_phase_anchors.clear()
        self.segment_started = 0
        self.match_seed = self.rng.randint(1, 999999)
        self.match_runtime_state_cache.clear()
        self.cinematic_shot_plan_cache.clear()
        self.poc_layer_frame_cache.clear()
        self.poc_layer_cache.clear()
        self.match_analysis = self.model.analyze_match(self.home, self.away, seed=self.match_seed)
        self.match_prediction = self.match_analysis.prediction
        runtime = self.match_runtime_state(
            self.match_prediction
        )
        self.start_cinematic_poc_preload(
            self.cinematic_poc_sequences_for_match(runtime)
        )
        self.goal_events.clear()
        self.shot_events.clear()
        self.shot_progress_cursor.clear()
        self.final_whistle_played = False
        self.match_intro_audio_pending = True

    def set_tournament(self) -> None:
        self.mc_cancel_event.set()
        self.mc_generation += 1
        self.cancel_cinematic_poc_preload()
        self.sound.stop_one_shots()
        self.sound.reset_scene_queues()
        self.match_audio_frame_queue.clear()
        self.poc_layer_frame_cache.clear()
        self.poc_layer_cache.clear()
        self.state = "tournament"
        self.t = 0.0
        seed = self.tournament_rng.randint(1, 999999)
        self.tournament_result = None
        self.tournament_view = "groups"
        self.champion_odds = []
        self.mc_progress_done = 0
        self.mc_progress_total = self.champion_odds_runs
        self.mc_error = ""
        self.mc_pending_result = None
        self.mc_started_t = 0.0
        self.pending_tournament_seed = seed
        self.cup_audio_markers.clear()
        self.cup_start_audio_pending = True
        self.cup_reveal_audio_pending = False
        self.cup_reveal_audio_played = False

    def cycle_home(self, delta: int) -> None:
        self.home_idx = (self.home_idx + delta) % len(self.teams)
        if self.home_idx == self.away_idx:
            self.home_idx = (self.home_idx + delta) % len(self.teams)
        self.match_prediction = None
        self.match_analysis = None

    def cycle_away(self, delta: int) -> None:
        self.away_idx = (self.away_idx + delta) % len(self.teams)
        if self.away_idx == self.home_idx:
            self.away_idx = (self.away_idx + delta) % len(self.teams)
        self.match_prediction = None
        self.match_analysis = None

    def handle_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            raise SystemExit
        if self.state == "menu" and key in {pygame.K_RETURN, pygame.K_SPACE}:
            self.set_select()
        elif self.state == "select":
            if key == pygame.K_LEFT:
                self.cycle_home(-1)
            elif key == pygame.K_RIGHT:
                self.cycle_home(1)
            elif key == pygame.K_a:
                self.cycle_away(-1)
            elif key == pygame.K_d:
                self.cycle_away(1)
            elif key == pygame.K_SPACE:
                self.set_simulate("match")
            elif key == pygame.K_RETURN:
                self.set_simulate("match")
            elif key == pygame.K_t:
                self.set_tournament()
        elif self.state == "simulate":
            if key == pygame.K_BACKSPACE:
                self.set_select()
            elif key == pygame.K_SPACE or key == pygame.K_r:
                self.set_simulate(self.mode)
            elif key == pygame.K_t:
                self.set_tournament()
        elif self.state == "tournament":
            if key == pygame.K_BACKSPACE:
                self.set_select()
            elif key in {pygame.K_t, pygame.K_SPACE, pygame.K_r}:
                self.set_tournament()
            elif key == pygame.K_g and self.tournament_result:
                self.tournament_view = "groups"
            elif key == pygame.K_m and self.tournament_result:
                self.tournament_view = "bracket"

    def handle_click(self, pos: tuple[int, int]) -> None:
        if self.state == "menu" and self.start_button.rect.collidepoint(pos):
            self.set_select()
        elif self.state == "select":
            if self.back_button.rect.collidepoint(pos):
                self.sound.stop_one_shots()
                self.sound.reset_scene_queues()
                self.state = "menu"
            elif self.team_arrow_rects(pygame.Rect(56, 118, 420, 460))[0].collidepoint(pos):
                self.cycle_home(-1)
            elif self.team_arrow_rects(pygame.Rect(56, 118, 420, 460))[1].collidepoint(pos):
                self.cycle_home(1)
            elif self.team_arrow_rects(pygame.Rect(804, 118, 420, 460))[0].collidepoint(pos):
                self.cycle_away(-1)
            elif self.team_arrow_rects(pygame.Rect(804, 118, 420, 460))[1].collidepoint(pos):
                self.cycle_away(1)
            elif self.single_button.rect.collidepoint(pos):
                self.set_simulate("match")
            elif self.cup_button.rect.collidepoint(pos):
                self.set_tournament()
        elif self.state == "simulate" and self.back_button.rect.collidepoint(pos):
            self.set_select()
        elif self.state == "tournament" and self.back_button.rect.collidepoint(pos):
            self.set_select()
        elif self.state == "tournament" and self.tournament_result:
            if self.group_tab_rect.collidepoint(pos):
                self.tournament_view = "groups"
            elif self.bracket_tab_rect.collidepoint(pos):
                self.tournament_view = "bracket"

    def run(self) -> None:
        running = True
        while running:
            dt = min(self.clock.tick(FPS) / 1000, MAX_FRAME_DT)
            self.mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    try:
                        self.handle_key(event.key)
                    except SystemExit:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
            self.update(dt)
            self.draw()
            self.flush_queued_match_audio()
        preload_stopped = self.shutdown_cinematic_poc_preloads(
            timeout=5.0,
        )
        self.cancel_champion_odds_job()
        if preload_stopped:
            pygame.quit()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()

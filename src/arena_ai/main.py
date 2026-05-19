from __future__ import annotations

import math
import os
import queue
import random
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import pygame

from arena_ai.audio import AudioEngine, pre_init_mixer
from arena_ai.audio_manifest import CUP_PROGRESS_MARKERS
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
CINEMATIC_PLAYER_SCALE = 1.06
CINEMATIC_NEUTRAL_PLAYER_SCALE = 1.00
CINEMATIC_KEEPER_SCALE = 0.84
CINEMATIC_BALL_SIZE = 50
CINEMATIC_SHOT_BALL_SIZE = 48
CINEMATIC_TURF_SPEED = 116.0
SHOT_KICK_AT = 0.56
SHOT_WHOOSH_AT = 0.62
SHOT_NET_AT = 0.96
SHOT_NET_VISUAL_CONTACT_AT = 0.982
SHOT_BASS_AT = SHOT_NET_VISUAL_CONTACT_AT
SHOT_CHEER_AT = SHOT_NET_VISUAL_CONTACT_AT
SHOT_REVERB_AT = 0.994
SHOT_KICK_AUDIO_AT = SHOT_KICK_AT - 0.006
SHOT_WHOOSH_AUDIO_AT = SHOT_WHOOSH_AT - 0.006
SHOT_NET_AUDIO_AT = SHOT_NET_VISUAL_CONTACT_AT - 0.008
SHOT_BASS_AUDIO_AT = SHOT_NET_AUDIO_AT
SHOT_CHEER_AUDIO_AT = SHOT_NET_AUDIO_AT
CHANCE_CONTACT_VISUAL_AT = SHOT_NET_AT
CHANCE_CONTACT_AUDIO_AT = CHANCE_CONTACT_VISUAL_AT - 0.008
SHOT_PLANT_AT = 0.46
SHOT_CONTACT_FREEZE_END = 0.60
SHOT_RELEASE_END = 0.70
SHOT_RECOVERY_AT = 0.98
SHOT_FOLLOW_THROUGH_HOLD_END = 0.74
SHOT_GOAL_REVEAL_AT = 0.30
SHOT_GOAL_FULL_AT = 0.54
SHOT_KEEPER_REVEAL_AT = 0.34
SHOT_KEEPER_FULL_AT = 0.64
SHOT_KEEPER_READ_AT = 0.38
SHOT_KEEPER_DIVE_AT = 0.58
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
GOAL_EVENT_WINDOW_MINUTES = 5.0
GOAL_PAYOFF_MINUTES = 3.25
GOAL_MIN_SPACING_MINUTES = 11
CHANCE_EVENT_WINDOW_MINUTES = 4.6
CHANCE_PAYOFF_MINUTES = 2.2
CHANCE_MIN_SPACING_MINUTES = 13
RUNNER_FOOT_ANCHORS = (
    (0.76, 0.78),
    (0.72, 0.82),
    (0.80, 0.78),
    (0.74, 0.82),
)
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


class AssetFactory:
    def __init__(self, profiles: list[TeamProfile]):
        self.profiles = profiles
        self._cinematic_pose_cache: dict[tuple[str, bool], list[pygame.Surface]] = {}
        self._cinematic_runner_cache: dict[tuple[str, bool], list[pygame.Surface]] = {}
        self._cinematic_keeper_cache: list[pygame.Surface] | None = None
        self.cinematic_players: dict[str, list[pygame.Surface]] = {}
        self.cinematic_players_left: dict[str, list[pygame.Surface]] = {}
        self.cinematic_runners: dict[str, list[pygame.Surface]] = {}
        self.cinematic_runners_left: dict[str, list[pygame.Surface]] = {}
        self.cinematic_keepers: dict[str, pygame.Surface] = {}
        self.cinematic_keeper_frames: dict[str, list[pygame.Surface]] = {}
        self.goal_net_frames: list[pygame.Surface] = []
        self.goal_front_frames: list[pygame.Surface] = []
        self.goal_impact_frames: list[pygame.Surface] = []
        self.flags: dict[str, pygame.Surface] = {}
        self.balls: list[pygame.Surface] = []
        self.field: pygame.Surface | None = None
        self.stadium_bg: pygame.Surface | None = None
        self.turf_mid_strip: pygame.Surface | None = None
        self.turf_near_strip: pygame.Surface | None = None
        self.fifa_images: dict[str, pygame.Surface] = {}
        self.generate_all()

    def generate_all(self) -> None:
        flags_dir = ASSETS / "generated" / "flags"
        for index, profile in enumerate(self.profiles):
            flag_path = flags_dir / f"{profile.code.lower()}.png"
            if not flag_path.exists():
                raise RuntimeError(f"missing generated image_gen flag sprite: {flag_path}")
            self.flags[profile.code] = pygame.image.load(flag_path).convert_alpha()
            cinematic_frames = self.load_cinematic_frames(profile)
            cinematic_left_frames = self.load_cinematic_frames(profile, left=True)
            runner_frames = self.load_cinematic_runner_frames(profile)
            runner_left_frames = self.load_cinematic_runner_frames(profile, left=True)
            keeper_frames = self.load_cinematic_keeper_frames(profile)
            if not cinematic_frames:
                raise RuntimeError(f"missing generated cinematic pose sprites for {profile.code}")
            if not cinematic_left_frames:
                raise RuntimeError(f"missing generated left-facing cinematic pose sprites for {profile.code}")
            if not runner_frames:
                raise RuntimeError(f"missing generated cinematic runner sprites for {profile.code}")
            if not runner_left_frames:
                raise RuntimeError(f"missing generated left-facing cinematic runner sprites for {profile.code}")
            if not keeper_frames:
                raise RuntimeError(f"missing generated cinematic goalkeeper animation for {profile.code}")
            self.cinematic_players[profile.code] = cinematic_frames
            self.cinematic_players_left[profile.code] = cinematic_left_frames
            self.cinematic_runners[profile.code] = runner_frames
            self.cinematic_runners_left[profile.code] = runner_left_frames
            self.cinematic_keeper_frames[profile.code] = keeper_frames
            self.cinematic_keepers[profile.code] = keeper_frames[0]
        self.balls = self.load_ball_frames()
        if not self.balls:
            raise RuntimeError("missing generated ball sprites")
        self.goal_net_frames = self.load_goal_net_frames()
        if len(self.goal_net_frames) < 4:
            raise RuntimeError("missing generated goal/net animation sprites")
        self.goal_front_frames = self.load_goal_front_frames()
        if len(self.goal_front_frames) < 4:
            raise RuntimeError("missing generated goal front-post sprites")
        self.goal_impact_frames = self.load_goal_impact_frames()
        if len(self.goal_impact_frames) < 4:
            raise RuntimeError("missing generated goal impact net sprites")
        self.stadium_bg = self.load_stadium_background((1520, 472))
        if not self.stadium_bg:
            raise RuntimeError("missing generated stadium parallax background")
        self.turf_mid_strip = self.load_turf_strip("turf_mid_strip.png")
        self.turf_near_strip = self.load_turf_strip("turf_near_strip.png")
        if not self.turf_mid_strip or not self.turf_near_strip:
            raise RuntimeError("missing generated parallax turf strips")
        self.field = self.load_3d_field((910, 490))
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

    def load_3d_field(self, size: tuple[int, int]) -> pygame.Surface | None:
        path = ASSETS / "generated" / "field" / "match_field_3d.png"
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

    def load_cinematic_frames(self, profile: TeamProfile, left: bool = False) -> list[pygame.Surface] | None:
        code = self.cinematic_source_code(profile)
        cache_key = (code, left)
        if cache_key in self._cinematic_pose_cache:
            return self._cinematic_pose_cache[cache_key]
        poses = ["idle", "run1", "dribble", "kick"]
        base = ASSETS / "generated" / "cinematic"
        prefix = "left_" if left else ""
        paths = [base / f"{prefix}{code}_{pose}.png" for pose in poses]
        if all(path.exists() for path in paths):
            frames = [pygame.image.load(path).convert_alpha() for path in paths]
            self._cinematic_pose_cache[cache_key] = frames
            return frames
        return None

    def load_cinematic_runner_frames(self, profile: TeamProfile, left: bool = False) -> list[pygame.Surface] | None:
        code = self.cinematic_source_code(profile)
        cache_key = (code, left)
        if cache_key in self._cinematic_runner_cache:
            return self._cinematic_runner_cache[cache_key]
        base = ASSETS / "generated" / "cinematic"
        stem = "runner_left" if left else "runner"
        paths = [base / f"{stem}_{code}_{index}.png" for index in range(4)]
        if all(path.exists() for path in paths):
            frames = [pygame.image.load(path).convert_alpha() for path in paths]
            self._cinematic_runner_cache[cache_key] = frames
            return frames
        return None

    def load_cinematic_keeper_frames(self, profile: TeamProfile) -> list[pygame.Surface] | None:
        if self._cinematic_keeper_cache is not None:
            return self._cinematic_keeper_cache
        base = ASSETS / "generated" / "cinematic"
        paths = [base / f"keeper_anim_{index}.png" for index in range(4)]
        if all(path.exists() for path in paths):
            self._cinematic_keeper_cache = [pygame.image.load(path).convert_alpha() for path in paths]
            return self._cinematic_keeper_cache
        return None

    def load_goal_net_frames(self) -> list[pygame.Surface]:
        base = ASSETS / "generated" / "cinematic"
        paths = [base / f"goal_net_{index}.png" for index in range(4)]
        if all(path.exists() for path in paths):
            return [pygame.image.load(path).convert_alpha() for path in paths]
        return []

    def load_goal_front_frames(self) -> list[pygame.Surface]:
        base = ASSETS / "generated" / "cinematic"
        paths = [base / f"goal_front_{index}.png" for index in range(4)]
        if all(path.exists() for path in paths):
            return [pygame.image.load(path).convert_alpha() for path in paths]
        return []

    def load_goal_impact_frames(self) -> list[pygame.Surface]:
        base = ASSETS / "generated" / "cinematic"
        paths = [base / f"goal_impact_{index}.png" for index in range(4)]
        if all(path.exists() for path in paths):
            return [pygame.image.load(path).convert_alpha() for path in paths]
        return []

    def load_turf_strip(self, filename: str) -> pygame.Surface | None:
        path = ASSETS / "generated" / "parallax" / filename
        if path.exists():
            return pygame.image.load(path).convert_alpha()
        return None

    def load_ball_frames(self) -> list[pygame.Surface] | None:
        base = ASSETS / "generated" / "balls3d"
        paths = [base / f"ball_{i}.png" for i in range(8)]
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
        self.turf_tile_cache: dict[tuple[int, int, int, int], pygame.Surface] = {}
        self.gradient_mask_cache: dict[tuple[int, int, int, int], pygame.Surface] = {}
        self.gradient_tile_cache: dict[tuple[int, int, int, int, int], pygame.Surface] = {}
        self.goal_orientation_cache: dict[tuple[int, str], pygame.Surface] = {}
        self.surface_bbox_cache: dict[int, pygame.Rect] = {}
        self.scaled_surface_cache = self.surface_cache.scaled
        self.flipped_surface_cache = self.surface_cache.flipped
        self.roto_surface_cache = self.surface_cache.roto
        self.cinematic_overlay_cache: dict[tuple[object, ...], pygame.Surface] = {}
        self.prepare_turf_tile_cache()
        self.prepare_goal_orientation_cache()
        self.title_bg = self.load_image("generated/title_stadium_ai.png", (WIDTH, HEIGHT))
        self.state = "menu"
        self.home_idx = self.team_index("BRA", 0)
        self.away_idx = self.team_index("FRA", 1)
        self.mode = "single"
        self.t = 0.0
        self.ground_scroll = 0.0
        self.ground_scroll_velocity = 0.0
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
        cache_key = id(image)
        cached = self.surface_bbox_cache.get(cache_key)
        if cached is None:
            cached = image.get_bounding_rect()
            self.surface_bbox_cache[cache_key] = cached
        return cached.copy()

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

    def prepare_goal_orientation_cache(self) -> None:
        for frame in (*self.assets.goal_net_frames, *self.assets.goal_front_frames, *self.assets.goal_impact_frames):
            self.orient_cinematic_goal_frame(frame, "left")
            self.orient_cinematic_goal_frame(frame, "right")

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
            visual_goal_minute = goal_minute - GOAL_EVENT_WINDOW_MINUTES + SHOT_NET_VISUAL_CONTACT_AT * GOAL_EVENT_WINDOW_MINUTES
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
            visual_goal_minute = goal_minute - GOAL_EVENT_WINDOW_MINUTES + SHOT_NET_VISUAL_CONTACT_AT * GOAL_EVENT_WINDOW_MINUTES
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
        shot_progress = clamp((self.match_minute_float() - (goal_minute - 5.0)) / 5.0)
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
        self.draw_cinematic_background(rect, pred)
        if not state.get("active_attack") and self.simulation_progress() < 0.88:
            self.draw_model_flow(rect, pred, algo)
        self.draw_cinematic_goal(rect, pred, state)
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
        if side == "left":
            return pygame.Rect(field.x + 76, field.y + 228, 180, 188)
        return pygame.Rect(field.right - 256, field.y + 228, 180, 188)

    def cinematic_camera_progress(self, pred: Prediction) -> float:
        active_attack = self.active_attack_event(pred)
        possession = active_attack.side if active_attack else self.cinematic_possession_side(pred)
        direction = 1 if possession != "away" else -1
        minute = self.match_minute_float()
        if active_attack:
            event_window = GOAL_EVENT_WINDOW_MINUTES if active_attack.is_goal else CHANCE_EVENT_WINDOW_MINUTES
            event_minute = active_attack.minute
            attack_progress = smoothstep(clamp((minute - (event_minute - event_window)) / event_window))
            cruise_progress = self.cinematic_cruise_camera_progress(pred, minute)
            settle_after = GOAL_PAYOFF_MINUTES + 0.45 if active_attack.is_goal else CHANCE_PAYOFF_MINUTES
            settle = smoothstep((minute - (event_minute + settle_after)) / 1.55)
            progress = lerp(attack_progress, cruise_progress, settle)
        else:
            progress = self.cinematic_cruise_camera_progress(pred, minute)
        return progress if direction > 0 else 1.0 - progress

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
        run_speed = 0.42 + approach_speed * 0.58 + shot_speed * 0.52
        camera = self.cinematic_camera_progress(pred)
        stride_phase = (self.t * (6.1 + run_speed * 1.8)) % 4.0
        desired_scroll_velocity = direction * CINEMATIC_TURF_SPEED * (0.64 + run_speed * 0.36)
        return {
            "direction": direction,
            "camera": camera,
            "run_speed": run_speed,
            "stride_phase": stride_phase,
            "ground_scroll": self.ground_scroll,
            "desired_scroll_velocity": desired_scroll_velocity,
        }

    def cinematic_shot_profile(self, goal_rect: pygame.Rect, direction: int, goal_minute: int) -> ShotProfile:
        seed = int(self.match_seed) * 1_000_003 + int(goal_minute) * 9_176 + (0 if direction > 0 else 4_699)
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
        target_x = goal_rect.right - depth + target_jitter_x if direction > 0 else goal_rect.left + depth - target_jitter_x
        target_x = clamp(
            target_x,
            goal_rect.left + CINEMATIC_SHOT_BALL_SIZE * 0.66,
            goal_rect.right - CINEMATIC_SHOT_BALL_SIZE * 0.66,
        )
        mouth_x = goal_rect.left - CINEMATIC_SHOT_BALL_SIZE * 0.62 if direction > 0 else goal_rect.right + CINEMATIC_SHOT_BALL_SIZE * 0.62
        mouth_y = target_y - rng.uniform(0.0, 5.0)
        entry = (mouth_x, mouth_y)
        mouth = (mouth_x, mouth_y)
        bend = direction * (bend_base * 9.0 + rng.uniform(-1.8, 1.8))
        return ShotProfile(
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

    def cinematic_shot_target(self, goal_rect: pygame.Rect, direction: int, goal_minute: int) -> tuple[float, float]:
        return self.cinematic_shot_profile(goal_rect, direction, goal_minute).target

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
    ) -> tuple[tuple[float, float], str, int, tuple[float, float], float]:
        release = (foot[0] + direction * 12, foot[1] - 2)
        if shot_profile is None:
            shot_profile = ShotProfile(
                zone="legado",
                target=target,
                entry=(target[0] - direction * 146, target[1]),
                mouth=(target[0] - direction * 126, target[1]),
                bend=direction * math.sin(self.match_seed * 0.013 + goal_minute) * 20.0,
                loft=1.0,
                dip=13.0,
                speed=1.0,
                spin=34.0,
            )
        if shot_progress <= SHOT_KICK_AT:
            settle = smoothstep((shot_progress - 0.40) / max(0.001, SHOT_KICK_AT - 0.40))
            dribble = 1.0 - settle
            roll_phase = time_value * 12.0 + goal_minute * 0.31
            roll = math.sin(roll_phase)
            contact = abs(roll)
            lead = 13.5 + 4.5 * contact + 2.5 * dribble
            dribble_pos = (
                foot[0] + direction * lead,
                foot[1] + 1.8 - 2.6 * contact + math.sin(roll_phase * 0.5) * 0.8 * dribble,
            )
            return (
                (
                    lerp(dribble_pos[0], release[0], settle),
                    lerp(dribble_pos[1], release[1], settle),
                ),
                "drible",
                CINEMATIC_BALL_SIZE,
                (1.0, 1.0),
                8.0 + 9.0 * contact * dribble,
            )

        flight = clamp((shot_progress - SHOT_KICK_AT) / (SHOT_NET_AT - SHOT_KICK_AT))
        if flight < 1.0:
            if is_goal:
                entry = shot_profile.entry
                drive = 1.0 - (1.0 - flight) ** shot_profile.speed
                lift = math.sin(math.pi * flight)
                arc_height = clamp(abs(entry[0] - release[0]) * 0.12 * shot_profile.loft, 18.0, 40.0)
                late_dip = smoothstep((flight - 0.70) / 0.30) * shot_profile.dip
                curve = shot_profile.bend * lift * (0.40 + 0.28 * flight)
                x = lerp(release[0], entry[0], drive) + curve
                y = lerp(release[1], entry[1], smoothstep(flight)) - arc_height * lift + late_dip
            else:
                curve_seed = math.sin(self.match_seed * 0.013 + goal_minute)
                x_ease = 1.0 - (1.0 - flight) ** 1.65
                y_ease = smoothstep(flight)
                arc_height = clamp(abs(target[0] - release[0]) * 0.28, 52.0, 82.0)
                curve = direction * curve_seed * 16.0 * math.sin(math.pi * flight)
                x = lerp(release[0], target[0], x_ease) + curve
                y = (
                    lerp(release[1], target[1], y_ease)
                    - arc_height * (math.sin(math.pi * flight) ** 0.9)
                    + curve_seed * 6.0 * math.sin(math.tau * flight)
                )
            x = min(x, target[0]) if direction > 0 else max(x, target[0])
            ball_size = int(lerp(CINEMATIC_BALL_SIZE, CINEMATIC_BALL_SIZE - 2, smoothstep(max(0.0, flight - 0.74) / 0.26)))
            return (
                (x, y),
                "chute",
                ball_size,
                (1.0, 1.0),
                shot_profile.spin - 7.0 * smoothstep(flight),
            )

        if not is_goal:
            settle_t = clamp((shot_progress - SHOT_NET_AT) / 0.18)
            damp = 1.0 - smoothstep(settle_t)
            bounce = abs(math.sin(settle_t * math.tau * 1.15)) * 7.0 * damp
            return (
                (
                    target[0] - direction * 10.0 * damp,
                    target[1] + bounce,
                ),
                "chute" if settle_t < 0.75 else "neutro",
                int(lerp(CINEMATIC_BALL_SIZE - 6, CINEMATIC_BALL_SIZE - 9, smoothstep(settle_t))),
                (1.0, 1.0),
                20.0 * damp,
            )

        entry = shot_profile.entry
        contact_t = clamp((shot_progress - SHOT_NET_AT) / 0.064)
        after_contact = clamp((shot_progress - SHOT_NET_VISUAL_CONTACT_AT) / 0.12)
        settle = smoothstep(contact_t)
        net_settle = smoothstep(after_contact)
        depth_push = direction * math.sin(after_contact * math.pi) * 2.4 * (1.0 - net_settle)
        ball_size = int(lerp(CINEMATIC_BALL_SIZE - 2, CINEMATIC_BALL_SIZE - 4, net_settle))
        impact_point = (
            lerp(entry[0], target[0], settle),
            lerp(entry[1], target[1], settle),
        )
        return (
            (
                lerp(impact_point[0], target[0], net_settle) + depth_push,
                lerp(impact_point[1], target[1], net_settle) + math.sin(after_contact * math.pi) * 1.0 * (1.0 - net_settle),
            ),
            "rede",
            ball_size,
            (1.0, 1.0),
            12.0 * (1.0 - net_settle),
        )

    def cinematic_save_variant(self, chance_minute: int, side: str) -> str:
        seed = int(self.match_seed) + int(chance_minute) * 31 + (17 if side == "away" else 0)
        return "stand" if seed % 2 == 0 else "dive"

    def cinematic_chance_target(
        self,
        goal_rect: pygame.Rect,
        direction: int,
        chance_minute: int,
        kind: str,
        save_variant: str = "",
    ) -> tuple[float, float]:
        base = self.cinematic_shot_target(goal_rect, direction, chance_minute)
        if kind == "wide":
            return base[0] + direction * 110, base[1] + (22 if base[1] <= goal_rect.centery else -22)
        if kind == "save" and save_variant == "stand":
            return goal_rect.centerx + direction * 28, goal_rect.centery + 18
        if kind == "save":
            return goal_rect.centerx + direction * 24, goal_rect.centery + 2
        return goal_rect.centerx + direction * 34, goal_rect.centery + (28 if base[1] >= goal_rect.centery else -30)

    def cinematic_scene_state(self, field: pygame.Rect, pred: Prediction) -> dict[str, object]:
        possession = self.cinematic_possession_side(pred)
        neutral = possession == "neutral"
        minute = self.match_minute_float()
        active_attack = self.active_attack_event(pred)
        active_goal = (active_attack.minute, active_attack.side) if active_attack and active_attack.is_goal else None
        goal_minute = active_attack.minute if active_attack else 0
        scoring_side = active_attack.side if active_attack else possession
        goal_side = self.cinematic_goal_side(scoring_side)
        goal_rect = self.cinematic_goal_rect(field, goal_side)
        direction = 1 if scoring_side != "away" else -1
        shot_progress = 0.0
        raw_shot_progress = 0.0
        if active_attack:
            event_window = GOAL_EVENT_WINDOW_MINUTES if active_attack.is_goal else CHANCE_EVENT_WINDOW_MINUTES
            raw_shot_progress = (minute - (goal_minute - event_window)) / event_window
            shot_progress = clamp(raw_shot_progress)

        if neutral:
            neutral_progress = smoothstep((self.simulation_progress() - DRAW_NEUTRAL_START_PROGRESS) / DRAW_NEUTRAL_RAMP)
            neutral_reveal = max(
                smoothstep((neutral_progress - 0.04) / 0.38),
                smoothstep(neutral_progress / 0.16) * 0.74,
            )
            ground_y = field.bottom - 54
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
            ball_pos = (
                lerp(field.centerx - 18, field.centerx, ball_roll) + math.sin(self.t * 2.2) * 0.8 * approach,
                ground_y - CINEMATIC_BALL_SIZE * 0.43 + abs(math.sin(self.t * 6.0)) * 0.9 * approach,
            )
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
                "stride_phase": (self.t * (2.2 + approach * 1.2)) % 4.0,
                "home_stride_phase": (self.t * (2.4 + approach * 1.4) + 0.25) % 4.0,
                "away_stride_phase": (self.t * (2.4 + approach * 1.4) + 2.15) % 4.0,
                "run_speed": 0.34 + approach * 0.30,
                "neutral_progress": neutral_progress,
                "neutral_reveal": neutral_reveal,
                "ball_prev_pos": (ball_pos[0] - 1, ball_pos[1]),
                "ball_phase": "neutro",
                "ball_scale": CINEMATIC_BALL_SIZE,
                "ball_squash": (1.0, 1.0),
                "ball_spin_rate": 10.0 * approach if neutral_progress < 0.95 else 0.0,
                "keeper_phase": 0.0,
                "net_progress": 0.0,
                "goal_impact_pos": ball_pos,
                "active_goal": active_goal,
                "active_attack": active_attack,
                "attack_kind": active_attack.kind if active_attack else "",
            }

        motion = self.cinematic_motion_state(pred)
        lane = math.sin(self.t * 0.7 + (0.0 if scoring_side == "home" else 2.1)) * 8
        start_x = field.x + 304 if direction > 0 else field.right - 304
        schedule = self.goal_schedule(pred)
        if schedule:
            first_attack_start = max(6.0, float(schedule[0][0]) - GOAL_EVENT_WINDOW_MINUTES)
        else:
            chance_schedule = self.chance_schedule(pred)
            first_attack_start = max(6.0, float(chance_schedule[0][0]) - CHANCE_EVENT_WINDOW_MINUTES) if chance_schedule else 60.0
        run_progress = smoothstep(clamp(minute / first_attack_start))
        screen_anchor = field.centerx - direction * 108
        cruise_x = lerp(start_x, screen_anchor, run_progress)
        if run_progress > 0.98:
            cruise_x += math.sin(self.t * 1.15) * 5 * direction
        shot_contact_x = field.centerx + direction * 46
        shot_actor_x = shot_contact_x - direction * 68
        run_x = cruise_x
        run_y = field.bottom - 54
        if active_attack:
            approach = smoothstep(clamp(shot_progress / 0.54))
            run_x = lerp(screen_anchor, shot_actor_x, approach)
            run_y = field.bottom - 54
        settled = self.simulation_progress() >= 0.985 and not active_attack
        if settled:
            run_x = screen_anchor
            run_y = field.bottom - 54
        actor_pos = (run_x, run_y)

        keeper_base_x = goal_rect.centerx
        keeper_base_y = goal_rect.centery + 24 + math.sin(self.t * 1.8) * 2
        keeper_x, keeper_y = keeper_base_x, keeper_base_y
        keeper_phase = 0.0
        shot_profile = self.cinematic_shot_profile(goal_rect, direction, goal_minute)
        target = shot_profile.target
        save_variant = ""
        if active_attack and not active_attack.is_goal:
            if active_attack.kind == "save":
                save_variant = self.cinematic_save_variant(goal_minute, active_attack.side)
            target = self.cinematic_chance_target(goal_rect, direction, goal_minute, active_attack.kind, save_variant)
        if active_attack:
            read = smoothstep((shot_progress - SHOT_KEEPER_READ_AT) / max(0.001, SHOT_KEEPER_DIVE_AT - SHOT_KEEPER_READ_AT))
            dive = smoothstep((shot_progress - SHOT_KEEPER_DIVE_AT) / 0.28)
            target_y_bias = clamp((target[1] - goal_rect.centery) / 80.0, -1.0, 1.0)
            target_dx = target[0] - goal_rect.centerx
            target_side = 1 if target_dx >= 0 else -1
            lateral_intensity = clamp(abs(target_dx) / 54.0)
            if active_attack.kind == "save" and save_variant == "stand":
                catch = smoothstep((shot_progress - SHOT_KEEPER_DIVE_AT) / 0.20)
                keeper_target_x = goal_rect.centerx + target_side * clamp(abs(target_dx) * 0.44 + 8.0, 8.0, 24.0)
                keeper_x = lerp(keeper_base_x, keeper_target_x, clamp(0.58 * read + 0.42 * catch))
                keeper_y = keeper_base_y + target_y_bias * 14 * read - 8 * math.sin(catch * math.pi)
                keeper_phase = max(read * 0.62, catch * 0.48)
            elif active_attack.kind == "save" and save_variant == "dive":
                keeper_target_x = goal_rect.centerx - target_side * 46.0
                keeper_target_y = target[1] + 7.0
                keeper_x = lerp(keeper_base_x, keeper_target_x, clamp(0.22 * read + 0.78 * dive))
                keeper_y = lerp(keeper_base_y, keeper_target_y, clamp(0.30 * read + 0.70 * dive))
                keeper_y -= 10 * math.sin(dive * math.pi)
                keeper_phase = max(read * 0.5, dive * 0.82)
            else:
                leap = math.sin(dive * math.pi)
                keeper_reach_x = clamp(abs(target_dx) * 0.72 + 8.0, 10.0, 58.0)
                keeper_target_x = goal_rect.centerx + target_side * keeper_reach_x
                keeper_x = lerp(keeper_base_x, keeper_target_x, clamp(0.22 * read + 0.78 * dive))
                leap_height = lerp(16.0, 52.0, lateral_intensity)
                keeper_y = keeper_base_y + target_y_bias * 28 * read - leap_height * leap + 18 * smoothstep((shot_progress - 0.82) / 0.16)
                keeper_phase = max(read * 0.5, dive * (0.55 + 0.45 * lateral_intensity))
        keeper_margin = 146
        keeper_x = clamp(keeper_x, field.x + keeper_margin, field.right - keeper_margin)
        keeper_y = clamp(keeper_y, field.y + keeper_margin, field.bottom - keeper_margin)
        keeper_pos = (keeper_x, keeper_y)

        scoring_team = self.home if scoring_side == "home" else self.away
        flip_actor = direction < 0
        kick_window, stride = self.cinematic_stride_state(shot_progress, float(motion["stride_phase"]))
        pose_frames_for_side = self.assets.cinematic_players_left[scoring_team.code] if flip_actor else self.assets.cinematic_players[scoring_team.code]
        runner_frames_for_side = self.assets.cinematic_runners_left[scoring_team.code] if flip_actor else self.assets.cinematic_runners[scoring_team.code]
        anchor_frame = pose_frames_for_side[3] if kick_window else runner_frames_for_side[stride]
        anchor_target = self.cinematic_actor_target_size(anchor_frame, CINEMATIC_PLAYER_SCALE)
        kick_frame = pose_frames_for_side[3]
        kick_target = self.cinematic_actor_target_size(kick_frame, CINEMATIC_PLAYER_SCALE)
        kick_contact_foot = self.cinematic_actor_anchor_screen(
            (shot_actor_x, field.bottom - 54),
            kick_target,
            KICK_FOOT_ANCHOR,
            flip_actor,
            kick_frame,
        )
        effective_run_speed = 0.0 if settled else float(motion["run_speed"])
        planted_x = self.cinematic_planted_x(actor_pos[0], stride, kick_window, direction, effective_run_speed)
        visual_foot = self.cinematic_actor_anchor_screen(
            (planted_x, actor_pos[1]),
            anchor_target,
            KICK_FOOT_ANCHOR if kick_window else RUNNER_FOOT_ANCHORS[stride],
            flip_actor,
            anchor_frame,
        )
        foot = visual_foot
        if active_attack and shot_progress <= SHOT_KICK_AT:
            settle_to_kick = smoothstep((shot_progress - SHOT_PLANT_AT) / max(0.001, SHOT_KICK_AT - SHOT_PLANT_AT))
            roll_cycle = math.sin(shot_progress * 38.0 + self.match_seed * 0.01)
            dribble_foot = (
                actor_pos[0] + direction * (50 + roll_cycle * 4),
                actor_pos[1] - 42 + math.cos(shot_progress * 32.0) * 2,
            )
            foot = (
                lerp(dribble_foot[0], kick_contact_foot[0], settle_to_kick),
                lerp(dribble_foot[1], kick_contact_foot[1], settle_to_kick),
            )
        elif active_attack:
            foot = kick_contact_foot
        if active_attack:
            ball_pos, ball_phase, ball_scale, ball_squash, ball_spin_rate = self.cinematic_ball_for_progress(
                foot,
                target,
                direction,
                shot_progress,
                self.t,
                goal_minute,
                is_goal=active_attack.is_goal,
                shot_profile=shot_profile if active_attack.is_goal else None,
            )
            previous_progress = max(0.0, shot_progress - 0.035)
            ball_prev_pos = self.cinematic_ball_for_progress(
                foot,
                target,
                direction,
                previous_progress,
                max(0.0, self.t - 0.035),
                goal_minute,
                is_goal=active_attack.is_goal,
                shot_profile=shot_profile if active_attack.is_goal else None,
            )[0]
        else:
            ball_pos = (
                foot[0] + math.sin(self.t * 7.5) * 7,
                foot[1] + math.cos(self.t * 8.0) * 5,
            )
            ball_prev_pos = (ball_pos[0] - direction * 6, ball_pos[1])
            ball_phase = "drible"
            ball_scale = CINEMATIC_BALL_SIZE
            ball_squash = (1.0, 1.0)
            ball_spin_rate = 11.0
            if settled:
                ball_pos = (foot[0] + direction * 14, foot[1] + 2)
                ball_prev_pos = ball_pos
                ball_phase = "neutro"
                ball_spin_rate = 0.0
        net_progress = 0.0
        net_decay = 0.0
        net_ripple_start = SHOT_NET_VISUAL_CONTACT_AT - 0.004
        if active_goal and raw_shot_progress >= net_ripple_start:
            contact_elapsed = max(0.0, raw_shot_progress - net_ripple_start)
            impact = smoothstep(contact_elapsed / 0.016)
            net_decay = 1.0 - 0.74 * smoothstep((contact_elapsed - 0.07) / 0.48)
            elastic = 1.0 + 0.07 * math.sin(contact_elapsed * 24.0) * (1.0 - smoothstep((contact_elapsed - 0.03) / 0.32))
            net_progress = clamp((0.18 + 0.82 * impact) * net_decay * elastic)

        keeper_glove_pos = None
        chance_near_post_pos = None
        if active_attack and active_attack.kind == "save" and save_variant:
            glove_side = 1 if target[0] >= goal_rect.centerx else -1
            if save_variant == "stand":
                keeper_glove_pos = (keeper_x, keeper_y + 5.0)
            else:
                keeper_glove_pos = (keeper_x + glove_side * 60.0, keeper_y - 2.0)
        elif active_attack and active_attack.kind == "wide":
            post_x = goal_rect.right if direction > 0 else goal_rect.left
            chance_near_post_pos = (float(post_x), float(target[1]))

        return {
            "neutral": False,
            "possession": possession,
            "goal_side": goal_side,
            "goal_rect": goal_rect,
            "shot_phase": self.shot_phase(shot_progress),
            "ball_pos": ball_pos,
            "kick_pos": foot,
            "actor_pos": actor_pos,
            "keeper_pos": keeper_pos,
            "shot_progress": shot_progress,
            "raw_shot_progress": raw_shot_progress,
            "stride_phase": 0.0 if settled else motion["stride_phase"],
            "run_speed": 0.0 if settled else motion["run_speed"],
            "ball_prev_pos": ball_prev_pos,
            "ball_phase": ball_phase,
            "ball_scale": ball_scale,
            "ball_squash": ball_squash,
            "ball_spin_rate": ball_spin_rate,
            "keeper_phase": keeper_phase,
            "net_progress": net_progress,
            "net_decay": net_decay,
            "net_ripple_decay": net_decay,
            "goal_impact_pos": ball_pos if active_goal and ball_phase == "rede" else target,
            "shot_target": target,
            "shot_profile": shot_profile if active_attack and active_attack.is_goal else None,
            "keeper_glove_pos": keeper_glove_pos,
            "chance_near_post_pos": chance_near_post_pos,
            "active_goal": active_goal,
            "active_attack": active_attack,
            "attack_kind": active_attack.kind if active_attack else "",
            "save_variant": save_variant,
            "keeper_action": f"{save_variant}_save" if save_variant else "",
            "settled": settled,
        }

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
                scroll * 0.78,
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

    def draw_cinematic_goal(self, field: pygame.Rect, pred: Prediction, state: dict[str, object] | None = None) -> None:
        if state is None:
            state = self.cinematic_scene_state(field, pred)
        shot_progress = float(state["shot_progress"])
        active_attack = state.get("active_attack")
        chance_scene = bool(active_attack) and not state.get("active_goal")
        if not (state.get("active_goal") or chance_scene) or shot_progress < SHOT_GOAL_REVEAL_AT:
            return
        goal_alpha = self.cinematic_goal_alpha(shot_progress)
        if goal_alpha <= 0:
            return
        for side in (str(state["goal_side"]),):
            goal = self.cinematic_goal_rect(field, side)
            ripple = float(state.get("net_progress", 0.0)) if state.get("active_goal") else 0.0
            self.draw_cinematic_goal_3d(goal, side, ripple, state.get("goal_impact_pos"), front_only=False, alpha=goal_alpha)

    def cinematic_goal_alpha(self, shot_progress: float) -> int:
        reveal = smoothstep((shot_progress - SHOT_GOAL_REVEAL_AT) / max(0.001, SHOT_GOAL_FULL_AT - SHOT_GOAL_REVEAL_AT))
        return int(round(255 * reveal))

    def cinematic_keeper_alpha(self, shot_progress: float) -> int:
        reveal = smoothstep((shot_progress - SHOT_KEEPER_REVEAL_AT) / max(0.001, SHOT_KICK_AT - SHOT_KEEPER_REVEAL_AT))
        alpha = int(round(255 * reveal))
        if shot_progress >= SHOT_PLANT_AT:
            alpha = max(alpha, 236)
        if shot_progress >= SHOT_KICK_AT:
            alpha = 255
        return alpha

    def draw_cinematic_goal_3d(
        self,
        goal: pygame.Rect,
        side: str,
        ripple: float,
        impact_pos: object | None = None,
        front_only: bool = False,
        alpha: int = 255,
    ) -> None:
        if front_only:
            self.draw_cinematic_goal_front(goal, side, ripple, impact_pos, alpha=alpha)
            return

        self.draw_cinematic_goal_layer(goal, side, ripple, self.assets.goal_net_frames, as_front=False, alpha=alpha)

    def draw_cinematic_goal_layer(
        self,
        goal: pygame.Rect,
        side: str,
        ripple: float,
        frames: list[pygame.Surface],
        as_front: bool,
        alpha: int = 255,
    ) -> None:
        if not frames:
            return
        frame_float = clamp(ripple) * (len(frames) - 1)
        frame_index = int(math.floor(frame_float))
        next_index = min(len(frames) - 1, frame_index + 1)
        frame_blend = frame_float - frame_index
        frame = self.orient_cinematic_goal_frame(frames[frame_index], side)
        next_frame = self.orient_cinematic_goal_frame(frames[next_index], side)
        aspect = frame.get_width() / max(1, frame.get_height())
        direction = -1 if side == "left" else 1
        wave = math.sin(clamp(ripple) * math.pi)
        scale_pulse = wave * (0.005 if not as_front else 0.0)
        target_h = int(round((goal.h * (1.22 + scale_pulse)) / 2) * 2)
        target_w = int(target_h * aspect)
        target = pygame.Rect(0, 0, target_w, target_h)
        target.midbottom = goal.midbottom
        target.move_ip(-18 if side == "right" else 18, -4)
        if ripple > 0.02 and not as_front:
            target.move_ip(int(direction * wave * 1.5), 0)

        if not as_front:
            shadow = pygame.Rect(0, 0, int(target.w * 0.82), 18)
            shadow.center = (target.centerx, target.bottom - 6)
            pygame.draw.ellipse(self.screen, (0, 0, 0, 62), shadow)
        base_alpha = int((245 if as_front else 168) * clamp(alpha / 255.0))
        layer = self.cached_smoothscale(frame, target.size)
        if frame_blend > 0.01 and next_index != frame_index:
            layer = self.cached_alpha(layer, int(base_alpha * (1.0 - frame_blend)))
        elif base_alpha < 255:
            layer = self.cached_alpha(layer, base_alpha)
        self.screen.blit(layer, target)
        if frame_blend > 0.01 and next_index != frame_index:
            next_layer = self.cached_alpha(self.cached_smoothscale(next_frame, target.size), int(base_alpha * frame_blend))
            self.screen.blit(next_layer, target)

    def draw_cinematic_goal_front(
        self,
        goal: pygame.Rect,
        side: str,
        ripple: float,
        impact_pos: object | None = None,
        alpha: int = 255,
    ) -> None:
        self.draw_cinematic_goal_front_posts(goal, side, ripple, alpha)
        self.draw_cinematic_goal_impact(goal, side, ripple, impact_pos, alpha)

    def draw_cinematic_goal_front_posts(
        self,
        goal: pygame.Rect,
        side: str,
        ripple: float,
        alpha: int = 255,
    ) -> None:
        if self.assets.goal_front_frames:
            self.draw_cinematic_goal_layer(goal, side, ripple, self.assets.goal_front_frames, as_front=True, alpha=alpha)

    def draw_cinematic_goal_ball_occlusion(
        self,
        goal: pygame.Rect,
        side: str,
        ripple: float,
        ball_pos: object | None,
        alpha: int = 255,
    ) -> None:
        if not self.assets.goal_front_frames or ripple <= 0.04 or not isinstance(ball_pos, tuple):
            return
        ball_x, ball_y = ball_pos  # type: ignore[misc]
        clip = pygame.Rect(0, 0, 96, 82)
        clip.center = (int(ball_x), int(ball_y))
        clip = clip.clip(goal.inflate(104, 88))
        if clip.w <= 0 or clip.h <= 0:
            return
        old_clip = self.screen.get_clip()
        self.screen.set_clip(clip)
        self.draw_cinematic_goal_front_posts(goal, side, ripple, alpha=int(alpha * 0.64))
        self.screen.set_clip(old_clip)

    def draw_cinematic_goal_impact(
        self,
        goal: pygame.Rect,
        side: str,
        ripple: float,
        impact_pos: object | None,
        alpha: int = 255,
    ) -> None:
        if not self.assets.goal_impact_frames or ripple <= 0.02 or not isinstance(impact_pos, tuple):
            return
        frame_float = clamp(ripple) * (len(self.assets.goal_impact_frames) - 1)
        frame_index = min(len(self.assets.goal_impact_frames) - 1, int(math.floor(frame_float)))
        frame = self.orient_cinematic_goal_frame(self.assets.goal_impact_frames[frame_index], side)
        pulse = math.sin(clamp(ripple) * math.pi)
        target_w = int(goal.w * (0.22 + 0.012 * pulse))
        target_h = int(target_w * frame.get_height() / max(1, frame.get_width()))
        impact_x, impact_y = impact_pos  # type: ignore[misc]
        rect = pygame.Rect(0, 0, target_w, target_h)
        rect.center = (
            int(clamp(float(impact_x), goal.x + 32, goal.right - 32)),
            int(clamp(float(impact_y), goal.y + 28, goal.bottom - 34)),
        )
        layer = self.cached_smoothscale(frame, rect.size)
        reveal = smoothstep((ripple - 0.03) / 0.22)
        layer = self.cached_alpha(layer, int(30 * reveal * (0.72 + 0.28 * pulse) * clamp(alpha / 255.0)))
        self.screen.blit(layer, rect)
        burst = self.cached_goal_impact_burst(side, ripple, int(255 * clamp(alpha / 255.0)))
        self.screen.blit(burst, burst.get_rect(center=rect.center))

    def cached_goal_impact_burst(self, side: str, ripple: float, alpha: int) -> pygame.Surface:
        ripple_key = int(round(clamp(ripple) * 28))
        alpha_key = int(round(clamp(alpha / 255.0) * 16))
        cache_key = ("goal_impact_burst", side, ripple_key, alpha_key)
        burst = self.cinematic_overlay_cache.get(cache_key)
        if burst is not None:
            return burst

        burst = pygame.Surface((188, 132), pygame.SRCALPHA)
        local = (94, 66)
        ripple_value = ripple_key / 28.0
        pulse = math.sin(clamp(ripple_value) * math.pi)
        reveal = smoothstep((ripple_value - 0.03) / 0.22)
        burst_strength = reveal * (0.08 + 0.24 * pulse) * (alpha_key / 16.0)
        direction = -1 if side == "left" else 1
        if burst_strength > 0.03:
            impact_alpha = int((118 + 28 * reveal) * (alpha_key / 16.0))
            for index, radius in enumerate((24, 42)):
                line_alpha = int((88 - index * 18) * burst_strength)
                offset = int(direction * pulse * (index + 1) * 2)
                pygame.draw.ellipse(
                    burst,
                    (232, 247, 255, line_alpha),
                    pygame.Rect(local[0] - radius + offset, local[1] - int(radius * 0.54), radius * 2, int(radius * 1.08)),
                    2,
                )
            for index in range(5):
                y = local[1] - 24 + index * 12
                elastic = math.sin(ripple_value * math.pi * 1.45 + index * 0.72) * 4 * burst_strength
                start = (local[0] - 56, int(y))
                end = (local[0] + 56 + int(direction * elastic), int(y + elastic * 0.16))
                pygame.draw.line(burst, (240, 250, 255, int(80 * burst_strength)), start, end, 1)
            for index in range(5):
                x = local[0] - 44 + index * 22
                elastic = math.sin(ripple_value * math.pi * 1.6 + index * 0.55) * 3 * burst_strength
                start = (int(x + elastic), local[1] - 28)
                end = (int(x - elastic * 0.35), local[1] + 32)
                pygame.draw.line(burst, (232, 248, 252, int(68 * burst_strength)), start, end, 1)
            for offset in (-12, 12):
                pygame.draw.line(
                    burst,
                    (255, 255, 245, int(impact_alpha * 0.24)),
                    (local[0] - 26, local[1] + offset),
                    (local[0] + 26 + int(direction * pulse * 3), local[1] + offset + int(pulse * 2)),
                    1,
                )
            flash_radius = int(8 + 9 * burst_strength)
            pygame.draw.circle(burst, (255, 255, 235, int(impact_alpha * 0.22)), local, max(3, flash_radius // 4), 0)
            pygame.draw.circle(burst, (255, 255, 255, int(impact_alpha * 0.30)), local, flash_radius, 1)
        self.cinematic_overlay_cache[cache_key] = burst
        return burst

    def orient_cinematic_goal_frame(self, frame: pygame.Surface, side: str) -> pygame.Surface:
        # The generated goal sprite faces right by default. A right-side goal must
        # face the attacker coming from midfield, so it needs the horizontal flip.
        cache_key = (id(frame), side)
        oriented = self.goal_orientation_cache.get(cache_key)
        if oriented is None:
            oriented = pygame.transform.flip(frame, True, False).convert_alpha() if side == "right" else frame
            self.goal_orientation_cache[cache_key] = oriented
        return oriented

    def draw_cinematic_scene(self, field: pygame.Rect, pred: Prediction, state: dict[str, object] | None = None) -> None:
        if state is None:
            state = self.cinematic_scene_state(field, pred)
        if state["neutral"]:
            self.draw_cinematic_neutral(field, state)
            return

        possession = str(state["possession"])
        team = self.home if possession == "home" else self.away
        keeper_team = self.away if possession == "home" else self.home
        direction = 1 if possession == "home" else -1
        shot_progress = float(state["shot_progress"])
        active_attack = bool(state.get("active_attack"))
        self.draw_cinematic_runner(
            team,
            state["actor_pos"],
            flip=direction < 0,
            shot_progress=shot_progress,
            stride_phase=float(state.get("stride_phase", 0.0)),
            run_speed=float(state.get("run_speed", 1.0)),
            settled=bool(state.get("settled", False)),
        )
        if active_attack:
            self.draw_cinematic_kick_impact(state["kick_pos"], direction, shot_progress)
        scoring_attack = bool(state.get("active_goal"))
        goal = state["goal_rect"]
        goal_side = str(state["goal_side"])
        goal_alpha = self.cinematic_goal_alpha(shot_progress) if active_attack and isinstance(goal, pygame.Rect) else 0
        show_goal_front = active_attack and isinstance(goal, pygame.Rect) and goal_alpha > 0
        if show_goal_front:
            self.draw_cinematic_goal_front_posts(
                goal,
                goal_side,
                float(state.get("net_progress", 0.0)),
                alpha=goal_alpha,
            )
        if active_attack and shot_progress >= SHOT_GOAL_REVEAL_AT:
            keeper_alpha = self.cinematic_keeper_alpha(shot_progress)
            if keeper_alpha > 0:
                keeper_action = str(state.get("keeper_action", ""))
                keeper_dive = scoring_attack or keeper_action == "dive_save"
                self.draw_cinematic_keeper(
                    keeper_team,
                    state["keeper_pos"],
                    flip=direction < 0,
                    active_goal=keeper_dive,
                    shot_progress=shot_progress,
                    alpha=keeper_alpha,
                    keeper_action=keeper_action,
                )
        ball_squash = state.get("ball_squash", (1.0, 1.0))
        if not isinstance(ball_squash, tuple):
            ball_squash = (1.0, 1.0)
        self.draw_cinematic_ball(
            state["ball_pos"],
            active_goal=scoring_attack,
            shot_progress=shot_progress,
            scale=int(state.get("ball_scale", CINEMATIC_SHOT_BALL_SIZE if scoring_attack and shot_progress > 0.54 else CINEMATIC_BALL_SIZE)),
            direction=direction,
            prev_pos=state.get("ball_prev_pos"),
            squash=(float(ball_squash[0]), float(ball_squash[1])),
            phase=str(state.get("ball_phase", "drible")),
            spin_rate=float(state.get("ball_spin_rate", 14.0)),
        )
        if show_goal_front:
            if str(state.get("ball_phase", "")) == "rede":
                self.draw_cinematic_goal_ball_occlusion(
                    goal,
                    goal_side,
                    float(state.get("net_progress", 0.0)),
                    state.get("ball_pos"),
                    goal_alpha,
                )
            self.draw_cinematic_goal_impact(goal, goal_side, float(state.get("net_progress", 0.0)), state.get("goal_impact_pos"), goal_alpha)
        if active_attack and shot_progress >= SHOT_GOAL_REVEAL_AT:
            if not scoring_attack:
                self.draw_cinematic_chance_payoff(state, direction)

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
            squash=(float(ball_squash[0]), float(ball_squash[1])),
            phase=str(state.get("ball_phase", "neutro")),
            spin_rate=float(state.get("ball_spin_rate", 5.0)),
            alpha=255,
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
        if shot_phase in {SHOT_PHASE_PLANT, SHOT_PHASE_CONTACT_FREEZE, SHOT_PHASE_RELEASE}:
            return True, 3
        if shot_phase == SHOT_PHASE_FOLLOW_THROUGH and shot_progress < SHOT_FOLLOW_THROUGH_HOLD_END:
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

    def draw_soft_shadow(self, rect: pygame.Rect, alpha: int) -> None:
        for layer, factor in enumerate((1.28, 1.10, 0.94, 0.78)):
            shade = int(alpha * (0.18 + layer * 0.19))
            shadow = pygame.Rect(0, 0, max(2, int(rect.w * factor)), max(2, int(rect.h * factor)))
            shadow.center = rect.center
            pygame.draw.ellipse(self.screen, (0, 0, 0, shade), shadow)

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
    ) -> None:
        x, ground_y = pos  # type: ignore[misc]
        frames = self.assets.cinematic_runners_left[team.code] if flip else self.assets.cinematic_runners[team.code]
        direction = -1 if flip else 1
        kick_window, stride = self.cinematic_stride_state(shot_progress, stride_phase)
        if settled:
            pose_frames = self.assets.cinematic_players_left[team.code] if flip else self.assets.cinematic_players[team.code]
            frame = pose_frames[0]
            render_scale = CINEMATIC_PLAYER_SCALE
            kick_window = False
            stride = 0
        elif kick_window:
            pose_frames = self.assets.cinematic_players_left[team.code] if flip else self.assets.cinematic_players[team.code]
            frame = pose_frames[3]
            render_scale = CINEMATIC_PLAYER_SCALE
        else:
            stride = min(len(frames) - 1, stride)
            frame = frames[stride]
            render_scale = CINEMATIC_PLAYER_SCALE
        target = self.cinematic_actor_target_size(frame, render_scale)
        bbox = self.visible_bbox(frame)
        visible_w = bbox.w * target[0] / max(1, frame.get_width())
        frame = self.cached_smoothscale(frame, target)
        planted_x = self.cinematic_planted_x(x, stride, kick_window, direction, run_speed)
        rect = self.cinematic_visible_midbottom_rect(frame, (planted_x, ground_y))
        shadow_w = max(74, int(visible_w * 0.84))
        shadow_h = int((13 + (1 if stride in (0, 2) else -1)) * CINEMATIC_PLAYER_SCALE)
        shadow = pygame.Rect(0, 0, shadow_w, shadow_h)
        shadow.center = (int(planted_x), int(ground_y - 1))
        self.draw_soft_shadow(shadow, int(64 * clamp(alpha / 255.0)))
        if alpha < 255:
            frame = self.cached_alpha(frame, alpha)
        self.screen.blit(frame, rect)

    def neutral_frame_for_phase(
        self,
        team: TeamProfile,
        stride_phase: float,
        neutral_progress: float,
        flip: bool,
    ) -> pygame.Surface:
        runner_frames = self.assets.cinematic_runners_left[team.code] if flip else self.assets.cinematic_runners[team.code]
        pose_frames = self.assets.cinematic_players_left[team.code] if flip else self.assets.cinematic_players[team.code]
        if neutral_progress < 0.48:
            return runner_frames[int(stride_phase) % len(runner_frames)]
        if neutral_progress < 0.76:
            return pose_frames[1]
        return pose_frames[0]

    def draw_cinematic_neutral_player(
        self,
        team: TeamProfile,
        pos: object,
        flip: bool,
        stride_phase: float,
        neutral_progress: float,
        alpha: int = 255,
    ) -> None:
        if neutral_progress < 0.48:
            approach = 1.0 - smoothstep(neutral_progress / 0.48)
            self.draw_cinematic_runner(
                team,
                pos,
                flip=flip,
                shot_progress=0.0,
                stride_phase=stride_phase,
                run_speed=0.34 + approach * 0.24,
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
        self.draw_soft_shadow(shadow, int(68 * clamp(alpha / 255.0)))
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
        if not SHOT_KICK_AT <= shot_progress <= 0.68:
            return
        x, y = pos  # type: ignore[misc]
        strength = smoothstep((shot_progress - SHOT_KICK_AT) / 0.04) * (1 - smoothstep((shot_progress - 0.63) / 0.05))
        if strength <= 0.02:
            return
        center = (int(x + direction * 8), int(y - 4))
        effect = self.cached_kick_impact_effect(direction, strength, self.t)
        self.screen.blit(effect, effect.get_rect(center=center))

    def cached_kick_impact_effect(self, direction: int, strength: float, time_value: float) -> pygame.Surface:
        strength_key = int(round(clamp(strength) * 24))
        phase_key = int(time_value * 10) % 8
        cache_key = ("kick_impact", direction, strength_key, phase_key)
        cached = self.cinematic_overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        effect = pygame.Surface((96, 76), pygame.SRCALPHA)
        local = (48, 38)
        strength = strength_key / 24.0
        for index in range(7):
            angle = -0.72 + index * 0.24
            start = (
                local[0] - direction * int(8 + index * 2),
                local[1] + int(math.sin(angle) * 6),
            )
            end = (
                local[0] - direction * int(28 + 18 * strength + index * 4),
                local[1] + int(math.sin(angle) * (10 + 6 * strength)),
            )
            color = (255, 244, 192, int((88 - index * 5) * strength))
            pygame.draw.line(effect, color, start, end, max(1, int(2 * strength)))
        for index in range(5):
            px = local[0] - direction * int(14 + index * 7)
            py = local[1] + 12 + int(math.sin(index + phase_key) * 3)
            pygame.draw.circle(effect, (186, 226, 116, int(72 * strength)), (px, py), max(1, int(2 * strength)))
        self.cinematic_overlay_cache[cache_key] = effect
        return effect

    def draw_cinematic_player(
        self,
        team: TeamProfile,
        pos: object,
        pose_index: int,
        flip: bool,
        scale: float,
    ) -> None:
        x, y = pos  # type: ignore[misc]
        frames = self.assets.cinematic_players_left[team.code] if flip else self.assets.cinematic_players[team.code]
        frame = frames[pose_index % len(frames)]
        target = (int(frame.get_width() * scale), int(frame.get_height() * scale))
        frame = self.cached_smoothscale(frame, target)
        if pose_index == 1:
            frame = self.cached_rotozoom(frame, math.sin(self.t * 8.5) * 1.8, 1.0 + math.sin(self.t * 17.0) * 0.012)
        shadow = pygame.Rect(0, 0, int(116 * scale), int(18 * scale))
        shadow.center = (int(x), int(y + 82 * scale))
        self.draw_soft_shadow(shadow, 92)
        self.screen.blit(frame, frame.get_rect(center=(x, y)))

    def cinematic_keeper_animation_state(
        self,
        active_goal: bool,
        shot_progress: float,
        frame_count: int,
        flip: bool,
        keeper_action: str = "",
    ) -> tuple[int, float, int]:
        index = 0
        if keeper_action == "dive_save":
            if shot_progress > SHOT_KEEPER_DIVE_AT and frame_count > 2:
                index = 2
            elif shot_progress > SHOT_KEEPER_READ_AT and frame_count > 1:
                index = 1
        elif keeper_action == "stand_save":
            if shot_progress > SHOT_KEEPER_READ_AT and frame_count > 1:
                index = 1
        elif active_goal and shot_progress > 0.80 and frame_count > 3:
            index = 3
        elif active_goal and shot_progress > SHOT_KEEPER_DIVE_AT and frame_count > 2:
            index = 2
        elif active_goal and shot_progress > SHOT_KEEPER_READ_AT and frame_count > 1:
            index = 1

        scale = CINEMATIC_KEEPER_SCALE
        angle = 0.0
        if keeper_action == "stand_save" and shot_progress > SHOT_KEEPER_READ_AT:
            read = smoothstep((shot_progress - SHOT_KEEPER_READ_AT) / max(0.001, SHOT_KEEPER_DIVE_AT - SHOT_KEEPER_READ_AT))
            catch = smoothstep((shot_progress - SHOT_KEEPER_DIVE_AT) / 0.20)
            if shot_progress < SHOT_NET_VISUAL_CONTACT_AT - 0.04 and frame_count > 1:
                index = 1
            else:
                index = 0
            scale += 0.010 * read + 0.006 * math.sin(catch * math.pi)
            angle = (-1 if flip else 1) * 0.8 * read * (1.0 - catch)
        elif keeper_action == "dive_save" and shot_progress > SHOT_KEEPER_READ_AT:
            read = smoothstep((shot_progress - SHOT_KEEPER_READ_AT) / max(0.001, SHOT_KEEPER_DIVE_AT - SHOT_KEEPER_READ_AT))
            dive = smoothstep((shot_progress - SHOT_KEEPER_DIVE_AT) / 0.28)
            leap = math.sin(dive * math.pi)
            scale += 0.010 * read + 0.018 * leap
            angle = ((-1 if flip else 1) * 1.4 * read) + ((-3 if flip else 3) * leap)
        elif active_goal and shot_progress > SHOT_KEEPER_READ_AT:
            read = smoothstep((shot_progress - SHOT_KEEPER_READ_AT) / max(0.001, SHOT_KEEPER_DIVE_AT - SHOT_KEEPER_READ_AT))
            dive = smoothstep((shot_progress - SHOT_KEEPER_DIVE_AT) / 0.28)
            leap = math.sin(dive * math.pi)
            scale += 0.010 * read + 0.030 * leap
            angle = ((-2 if flip else 2) * read) + ((-6 if flip else 6) * leap)
        return index, round(scale / 0.01) * 0.01, round(angle)

    def draw_cinematic_keeper(
        self,
        team: TeamProfile,
        pos: object,
        flip: bool,
        active_goal: bool,
        shot_progress: float,
        alpha: int = 255,
        keeper_action: str = "",
    ) -> None:
        x, y = pos  # type: ignore[misc]
        frames = self.assets.cinematic_keeper_frames[team.code]
        index, scale, angle = self.cinematic_keeper_animation_state(active_goal, shot_progress, len(frames), flip, keeper_action)
        frame = frames[index]
        if flip:
            frame = self.cached_flip(frame)
        frame = self.cached_rotozoom(frame, angle, scale)
        glow_key = ("keeper_glow", 220, 190)
        glow = self.cinematic_overlay_cache.get(glow_key)
        if glow is None:
            glow = pygame.Surface((220, 190), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 255, 255, 22), (110, 92), 58)
            self.cinematic_overlay_cache[glow_key] = glow
        reveal = clamp(alpha / 255.0)
        glow_layer = self.cached_alpha(glow, alpha)
        self.screen.blit(glow_layer, glow_layer.get_rect(center=(x, y + 12)))
        keeper_shadow = pygame.Rect(0, 0, 130, 18)
        keeper_shadow.center = (int(x), int(y + 84))
        self.draw_soft_shadow(keeper_shadow, int(78 * reveal))
        if alpha < 255:
            frame = self.cached_alpha(frame, alpha)
        self.screen.blit(frame, frame.get_rect(center=(x, y)))

    def draw_cinematic_chance_payoff(self, state: dict[str, object], direction: int) -> None:
        shot_progress = float(state.get("shot_progress", 0.0))
        chance_impact_start = CHANCE_CONTACT_VISUAL_AT
        strength = smoothstep((shot_progress - chance_impact_start) / 0.10) * (1.0 - smoothstep((shot_progress - 1.04) / 0.18))
        if strength <= 0.02:
            return
        ball_x, ball_y = state["ball_pos"]  # type: ignore[misc]
        kind = str(state.get("attack_kind", "save"))
        effect = self.cached_chance_payoff_effect(kind, direction, strength, self.t)
        self.screen.blit(effect, effect.get_rect(center=(int(ball_x), int(ball_y))))

    def cached_chance_payoff_effect(self, kind: str, direction: int, strength: float, time_value: float) -> pygame.Surface:
        strength_key = int(round(clamp(strength) * 24))
        phase_key = int(time_value * 8) % 12
        cache_key = ("chance_payoff", kind, direction, strength_key, phase_key)
        cached = self.cinematic_overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        effect = pygame.Surface((176, 136), pygame.SRCALPHA)
        local = (88, 68)
        strength = strength_key / 24.0
        if kind == "wide":
            pygame.draw.ellipse(effect, (2, 8, 10, int(135 * strength)), (local[0] - 58, local[1] + 18, 116, 22))
            pygame.draw.line(
                effect,
                (3, 10, 12, int(170 * strength)),
                (local[0] - direction * 8, local[1] + 30),
                (local[0] - direction * 94, local[1] + 50),
                max(3, int(8 * strength)),
            )
            pygame.draw.circle(effect, (255, 247, 198, int(150 * strength)), local, int(16 + 8 * strength))
            pygame.draw.circle(effect, (250, 195, 67, int(170 * strength)), local, int(30 + 10 * strength), 3)
            for index in range(5):
                offset = index * 16
                alpha = int((150 - index * 18) * strength)
                start = (local[0] - direction * (20 + offset), local[1] + index * 4 - 8)
                end = (local[0] - direction * (70 + offset), local[1] + index * 11 - 22)
                pygame.draw.line(effect, (255, 244, 192, alpha), start, end, max(2, 4 - index // 2))
            for index in range(7):
                angle = -0.95 + index * 0.28
                spark = (
                    local[0] - direction * int(math.cos(angle) * 58 * strength),
                    local[1] + int(math.sin(angle) * 38 * strength),
                )
                pygame.draw.circle(effect, (255, 255, 230, int(128 * strength)), spark, max(2, int(4 * strength)))
            pygame.draw.arc(effect, (250, 195, 67, int(178 * strength)), (local[0] - 40, local[1] - 36, 80, 72), -0.8, 1.1, 4)
        else:
            core_alpha = int(255 * clamp(strength * 1.8))
            ring_alpha = int(230 * clamp(strength * 1.45))
            pygame.draw.circle(effect, (255, 255, 244, core_alpha), local, int(18 + 7 * strength))
            pygame.draw.circle(effect, (255, 255, 255, ring_alpha), local, int(30 + 9 * strength), 4)
            pygame.draw.circle(effect, (82, 226, 255, int(220 * clamp(strength * 1.35))), local, int(42 + 10 * strength), 4)
            for radius in (22, 36, 52):
                pygame.draw.circle(effect, (82, 226, 255, int((220 - radius) * clamp(strength * 1.25))), local, radius, 3)
            for index in range(12):
                angle = index * math.tau / 12.0 + phase_key * 0.18
                end = (local[0] + int(math.cos(angle) * 76 * strength), local[1] + int(math.sin(angle) * 54 * strength))
                pygame.draw.line(effect, (255, 255, 245, ring_alpha), local, end, 4)
        self.cinematic_overlay_cache[cache_key] = effect
        return effect

    def draw_cinematic_ball(
        self,
        pos: object,
        active_goal: bool,
        shot_progress: float,
        scale: int = 46,
        direction: int = 1,
        prev_pos: object | None = None,
        squash: tuple[float, float] = (1.0, 1.0),
        phase: str = "drible",
        spin_rate: float = 14.0,
        alpha: int = 255,
    ) -> None:
        x, y = pos  # type: ignore[misc]
        if active_goal and phase == "chute" and shot_progress < SHOT_NET_AT and prev_pos is not None:
            px, py = prev_pos  # type: ignore[misc]
            vx = x - px
            vy = y - py
            distance = max(0.001, math.hypot(vx, vy))
            ux, uy = vx / distance, vy / distance
            trail = self.cached_ball_trail(ux, uy, int(alpha * 0.20))
            cx, cy = 110, 70
            self.screen.blit(trail, (int(x - cx), int(y - cy)))
        if phase == "chute":
            flight = clamp((shot_progress - SHOT_KICK_AT) / max(0.001, SHOT_NET_AT - SHOT_KICK_AT))
            lift = math.sin(flight * math.pi)
            shadow_w = max(18, int(scale * (0.92 - 0.34 * lift)))
            shadow_h = max(5, int(scale * (0.18 - 0.06 * lift)))
            shadow = pygame.Rect(0, 0, shadow_w, shadow_h)
            shadow.center = (int(x - direction * (8 + 4 * lift)), int(y + scale * 0.64 + 42 * lift))
            pygame.draw.ellipse(self.screen, (0, 0, 0, int((42 - 20 * lift) * clamp(alpha / 255.0))), shadow)
        elif phase != "rede":
            shadow = pygame.Rect(0, 0, max(26, int(scale * 0.94)), max(8, int(scale * 0.22)))
            shadow_y_offset = scale * 0.46
            shadow.center = (int(x), int(y + shadow_y_offset))
            pygame.draw.ellipse(self.screen, (0, 0, 0, int(66 * clamp(alpha / 255.0))), shadow)
        if phase in {"drible", "neutro"}:
            spin_phase = abs(self.ground_scroll) * 0.058 + self.t * max(0.0, spin_rate) * 0.18
        else:
            spin_phase = self.t * spin_rate
        if prev_pos is not None:
            px, py = prev_pos  # type: ignore[misc]
            distance = math.hypot(x - px, y - py)
            spin_phase += distance * (0.18 if phase == "rede" else 0.42)
        frame = self.assets.balls[int(spin_phase) % len(self.assets.balls)]
        size = int(scale)
        # The ball itself stays circular. Impact energy is expressed by trail, shadow,
        # glint and net deformation; anisotropic scaling made the rotated sprite read
        # as an oval ball.
        size_x = size_y = max(22, size)
        frame = self.cached_smoothscale(frame, (size_x, size_y))
        if phase in {"drible", "neutro", "chute"}:
            spin_direction = -1 if direction < 0 else 1
            rotation_gain = 18.0 if phase in {"drible", "neutro"} else 12.0
            angle = (spin_direction * spin_phase * rotation_gain) % 360.0
            angle = round(angle / 3.0) * 3.0
            frame = self.cached_rotozoom(frame, angle, 1.0)
        if alpha < 255:
            frame = self.cached_alpha(frame, alpha)
        ball_rect = frame.get_rect(center=(int(round(x)), int(round(y))))
        if phase == "rede":
            contrast_shadow = ball_rect.inflate(2, 2)
            pygame.draw.ellipse(self.screen, (0, 0, 0, int(14 * clamp(alpha / 255.0))), contrast_shadow)
        self.screen.blit(frame, ball_rect)
        if phase in {"chute", "rede"} and alpha > 18:
            glint_alpha = int((164 if phase == "chute" else 96) * clamp(alpha / 255.0))
            glint_radius = max(6, int(min(size_x, size_y) * (0.18 if phase == "chute" else 0.17)))
            glint_center = (
                int(ball_rect.centerx - size_x * 0.16),
                int(ball_rect.centery - size_y * 0.18),
            )
            if phase == "rede":
                pygame.draw.circle(self.screen, (255, 255, 235, glint_alpha), glint_center, glint_radius, 2)
                pygame.draw.circle(self.screen, (255, 255, 235, int(48 * clamp(alpha / 255.0))), glint_center, max(3, glint_radius // 3))
            else:
                pygame.draw.circle(self.screen, (255, 255, 235, glint_alpha), glint_center, glint_radius)

    def cached_ball_trail(self, ux: float, uy: float, alpha: int) -> pygame.Surface:
        angle_key = int(round(math.atan2(uy, ux) / math.tau * 24)) % 24
        alpha_key = int(round(clamp(alpha / 255.0) * 16))
        cache_key = ("ball_trail", angle_key, alpha_key)
        cached = self.cinematic_overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        angle = angle_key / 24.0 * math.tau
        ux = math.cos(angle)
        uy = math.sin(angle)
        trail = pygame.Surface((220, 140), pygame.SRCALPHA)
        cx, cy = 110, 70
        alpha_scale = alpha_key / 16.0
        for index, (length, width, segment_alpha, color) in enumerate(
            (
                (36, 2, 6, (255, 255, 255)),
                (24, 1, 4, (116, 226, 255)),
                (14, 1, 3, (250, 195, 67)),
            )
        ):
            start = (int(cx - ux * (16 + index * 7)), int(cy - uy * (16 + index * 7)))
            end = (int(cx - ux * length), int(cy - uy * length))
            pygame.draw.line(trail, (*color, int(segment_alpha * alpha_scale)), start, end, width)
        self.cinematic_overlay_cache[cache_key] = trail
        return trail

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
        panel = self.cached_alpha(panel, alpha)
        self.screen.blit(panel, panel.get_rect(center=center))

    def cinematic_goal_overlay_center(self, field: pygame.Rect) -> tuple[int, int]:
        return field.centerx, field.y + 146

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
            goal_audio_key = (result_pred.algorithm, goal_minute, side)
            previous_progress = self.shot_progress_cursor.get(goal_audio_key, 0.0)
            in_window = goal_minute - GOAL_EVENT_WINDOW_MINUTES <= minute <= goal_minute + GOAL_PAYOFF_MINUTES
            crossed_window = (
                previous_minute is not None
                and previous_minute <= goal_minute + GOAL_PAYOFF_MINUTES
                and minute >= goal_minute - GOAL_EVENT_WINDOW_MINUTES
            )
            catching_up = 0.0 < previous_progress < SHOT_REVERB_AT and not in_window
            if not (in_window or crossed_window or catching_up):
                continue
            shot_progress = clamp((minute - (goal_minute - GOAL_EVENT_WINDOW_MINUTES)) / GOAL_EVENT_WINDOW_MINUTES)
            if (crossed_window and minute > goal_minute + GOAL_PAYOFF_MINUTES) or catching_up:
                shot_progress = 1.0
            crossed = [
                (name, threshold)
                for name, threshold in (
                    ("kick", SHOT_KICK_AUDIO_AT),
                    ("whoosh", SHOT_WHOOSH_AUDIO_AT),
                    ("net", SHOT_NET_AUDIO_AT),
                    ("bass", SHOT_BASS_AUDIO_AT),
                    ("cheer", SHOT_CHEER_AUDIO_AT),
                    ("reverb", SHOT_REVERB_AT),
                )
                if previous_progress < threshold <= shot_progress + 1e-9
            ]
            if len(crossed) > 1 or shot_progress - previous_progress > 0.18:
                crossed_names = {name for name, _threshold in crossed}
                if shot_progress + 1e-9 >= SHOT_NET_AUDIO_AT and crossed_names & GOAL_IMPACT_AUDIO_EVENTS:
                    crossed = [
                        (name, threshold)
                        for name, threshold in crossed
                        if name in GOAL_IMPACT_AUDIO_EVENTS
                    ]
                else:
                    crossed = crossed[:1]
            if previous_progress < SHOT_KICK_AT and shot_progress >= SHOT_KICK_AT - 0.09:
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
            if shot_progress >= SHOT_REVERB_AT:
                self.goal_events.add(goal_key)
        for chance_minute, side, kind in self.chance_schedule(result_pred):
            chance_audio_key = (result_pred.algorithm, chance_minute, f"{side}:{kind}")
            previous_progress = self.shot_progress_cursor.get(chance_audio_key, 0.0)
            in_window = chance_minute - CHANCE_EVENT_WINDOW_MINUTES <= minute <= chance_minute + CHANCE_PAYOFF_MINUTES
            catching_up = 0.0 < previous_progress < CHANCE_CONTACT_AUDIO_AT and not in_window
            if not in_window and not catching_up:
                continue
            shot_progress = clamp((minute - (chance_minute - CHANCE_EVENT_WINDOW_MINUTES)) / CHANCE_EVENT_WINDOW_MINUTES)
            if catching_up:
                shot_progress = max(shot_progress, CHANCE_CONTACT_AUDIO_AT)
            crossed = [
                (name, threshold)
                for name, threshold in (
                    ("kick", SHOT_KICK_AUDIO_AT),
                    ("whoosh", SHOT_WHOOSH_AUDIO_AT),
                    ("save" if kind == "save" else "near_miss", CHANCE_CONTACT_AUDIO_AT),
                )
                if previous_progress < threshold <= shot_progress + 1e-9
            ]
            if len(crossed) > 1 or shot_progress - previous_progress > 0.18:
                if shot_progress + 1e-9 >= CHANCE_CONTACT_AUDIO_AT and any(name in {"save", "near_miss"} for name, _threshold in crossed):
                    crossed = [
                        (name, threshold)
                        for name, threshold in crossed
                        if name in {"save", "near_miss"}
                    ]
                else:
                    crossed = crossed[:1]
            if shot_progress >= SHOT_KICK_AT - 0.09:
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
        if len(key) >= 3 and str(key[2]) in {"home", "away"}:
            return SHOT_REVERB_AT
        return CHANCE_CONTACT_AUDIO_AT

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
            self.sound.play(name, pan=pan)

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
            if self.match_intro_audio_pending:
                self.sound.play("ui_chime")
                self.sound.play("whistle")
                self.match_intro_audio_pending = False
            self.t += dt
            self.t = min(self.t, SIMULATION_SECONDS)
            self.update_cinematic_scroll(dt)
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
        motion = self.cinematic_motion_state(pred)
        target_velocity = float(motion.get("desired_scroll_velocity", 0.0))
        self.ground_scroll_velocity += (target_velocity - self.ground_scroll_velocity) * clamp(dt * 3.8)
        self.ground_scroll += self.ground_scroll_velocity * dt

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
        self.sound.stop_one_shots()
        self.sound.reset_scene_queues()
        self.match_audio_frame_queue.clear()
        self.match_prediction = None
        self.match_analysis = None
        self.match_runtime_state_cache.clear()
        self.state = "select"

    def set_simulate(self, mode: str) -> None:
        self.sound.stop_one_shots()
        self.sound.reset_scene_queues()
        self.match_audio_frame_queue.clear()
        self.mode = mode
        self.state = "simulate"
        self.t = 0.0
        self.ground_scroll = 0.0
        self.ground_scroll_velocity = 0.0
        self.segment_started = 0
        self.match_seed = self.rng.randint(1, 999999)
        self.match_runtime_state_cache.clear()
        self.match_analysis = self.model.analyze_match(self.home, self.away, seed=self.match_seed)
        self.match_prediction = self.match_analysis.prediction
        self.goal_events.clear()
        self.shot_events.clear()
        self.shot_progress_cursor.clear()
        self.final_whistle_played = False
        self.match_intro_audio_pending = True

    def set_tournament(self) -> None:
        self.mc_cancel_event.set()
        self.mc_generation += 1
        self.sound.stop_one_shots()
        self.sound.reset_scene_queues()
        self.match_audio_frame_queue.clear()
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
        self.cancel_champion_odds_job()
        pygame.quit()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()

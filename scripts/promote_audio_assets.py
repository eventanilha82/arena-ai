from __future__ import annotations

import os
import shutil
import wave
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame


ROOT = Path(__file__).resolve().parents[1]
SOUND_DIR = ROOT / "assets" / "sounds"
RUNTIME_DIR = SOUND_DIR / "runtime_assets"
REJECTED_DIR = SOUND_DIR / "rejected_assets"
PACK_DIR = SOUND_DIR / "candidates" / "freesound" / "sandermotions_soccer_match_stadium_sounds_pack_27815"
PROMOTED_DIR = "candidates/promoted_sources"


COPY_ASSETS = (
    (f"{PROMOTED_DIR}/stadium_crowd_loop.mp3", "stadium_base_loop.mp3"),
    (f"{PROMOTED_DIR}/light_crowd_mixkit.mp3", "crowd_light_loop.mp3"),
    (f"{PROMOTED_DIR}/crowd_tension_mixkit.mp3", "crowd_tension_loop.mp3"),
    (f"{PROMOTED_DIR}/stadium_chant_mixkit.mp3", "crowd_chant_loop.mp3"),
    (f"{PROMOTED_DIR}/crowd_goal_roar.mp3", "goal_roar_main.mp3"),
    (f"{PROMOTED_DIR}/stadium_reverb_tail_mixkit.mp3", "stadium_reverb_tail.mp3"),
    (f"{PROMOTED_DIR}/analysis_chime.wav", "ui_chime_01.wav"),
    ("candidates/pixabay/bombinsound-football-football-soccer-game-music-15-second-490555.mp3", "opening_theme.mp3"),
    (f"candidates/freesound/{PACK_DIR.name}/494350__sandermotions__soccer-stadium-02.wav", "stadium_air_loop.wav"),
    (f"candidates/freesound/{PACK_DIR.name}/494362__sandermotions__soccer-stadium-oehh.wav", "crowd_attack_rise.wav"),
    (f"candidates/freesound/{PACK_DIR.name}/494361__sandermotions__soccer-stadium-oehh-2.wav", "crowd_attack_short.wav"),
    (f"candidates/freesound/{PACK_DIR.name}/494352__sandermotions__goal.wav", "goal_crowd_cc0.wav"),
)

TRIM_ASSETS = (
    ("candidates/freesound/555042__bittermelonheart__soccer-ball-kick.wav", "kick_grass_01.wav", 0.055, 0.014, 0.56, 0.96),
    ("candidates/mixkit/soccer_ball_kick_2108.mp3", "kick_grass_02.wav", 0.070, 0.012, 0.54, 0.82),
    ("candidates/mixkit/hitting_soccer_ball_2112.mp3", "kick_grass_03.wav", 0.060, 0.012, 0.55, 0.78),
    ("candidates/mixkit/sports_ball_hit_2082.mp3", "kick_grass_04.wav", 0.070, 0.010, 0.50, 0.68),
    ("candidates/mixkit/fast_sweep_transition_174.mp3", "ball_whoosh_01.wav", 0.060, 0.010, 0.72, 0.72),
    ("candidates/mixkit/cinematic_trailer_riser_790.mp3", "ball_whoosh_02.wav", 0.050, 0.000, 0.70, 0.38),
    ("candidates/pixabay/forza1903-a-football-hits-the-net-goal-313216.mp3", "net_ripple_01.wav", 0.055, 0.014, 1.18, 0.70),
    ("candidates/mixkit/basketball_ball_hitting_net_2084.mp3", "net_ripple_02.wav", 0.060, 0.010, 0.70, 0.62),
    (f"{PROMOTED_DIR}/goal_explosion_mixkit_a.mp3", "goal_explosion_01.wav", 0.050, 0.010, 4.20, 0.58),
    ("candidates/mixkit/crowd_at_stadium_2111.mp3", "goal_explosion_02.wav", 0.050, 0.010, 4.20, 0.52),
    ("candidates/mixkit/stadium_joy_shouting_crowd_3022.mp3", "goal_explosion_03.wav", 0.050, 0.010, 4.20, 0.48),
    (f"{PROMOTED_DIR}/bass_hit_mixkit.mp3", "bass_hit_01.wav", 0.080, 0.012, 1.05, 0.96),
    ("candidates/mixkit/movie_trailer_epic_impact_2908.mp3", "cup_reveal_stinger.wav", 0.050, 0.014, 4.80, 0.62),
    ("candidates/pixabay/mrmark81-stadium-roar-concert-471943.mp3", "goal_roar_pixabay_01.wav", 0.040, 0.030, 8.20, 0.54),
    ("candidates/pixabay/vishiv-crowd-cheering-in-stadium-435357.mp3", "goal_roar_pixabay_02.wav", 0.050, 0.025, 8.20, 0.36),
    ("candidates/mixkit/crowd_yelling_at_stadium_2097.mp3", "crowd_reaction_01.wav", 0.050, 0.020, 3.60, 0.50),
    (f"candidates/freesound/{PACK_DIR.name}/494357__sandermotions__soccer-stadium-09.wav", "crowd_reaction_02.wav", 0.050, 0.020, 3.60, 0.42),
    (f"candidates/freesound/{PACK_DIR.name}/494359__sandermotions__soccer-stadium-booohh.wav", "crowd_near_miss_01.wav", 0.050, 0.020, 3.40, 0.72),
    ("candidates/freesound/538422__rosa-orenes256__referee-whistle-sound.wav", "whistle_start_01.wav", 0.055, 0.016, 0.56, 0.78),
    ("candidates/freesound/218318__splicesound__referee-whistle-blow-gymnasium.wav", "whistle_final_01.wav", 0.055, 0.016, 1.45, 0.78),
    (f"{PROMOTED_DIR}/analysis_chime.wav", "cup_progress_tick_01.wav", 0.050, 0.000, 0.38, 0.42),
)


def ensure_mixer() -> int:
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    return int(pygame.mixer.get_init()[0])


def decoded_array(path: Path) -> tuple[np.ndarray, int]:
    frequency = ensure_mixer()
    sound = pygame.mixer.Sound(path)
    data = pygame.sndarray.array(sound).astype(np.float32)
    if data.ndim == 1:
        data = np.column_stack([data, data])
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2], frequency


def trim_to_transient(src: Path, dst: Path, threshold: float, preroll: float, max_seconds: float, gain: float) -> None:
    data, frequency = decoded_array(src)
    mono = np.max(np.abs(data), axis=1)
    peak = float(np.max(mono))
    if peak <= 0:
        raise RuntimeError(f"{src} is silent")
    above = np.flatnonzero(mono >= peak * threshold)
    start = int(above[0]) if above.size else 0
    start = max(0, start - int(preroll * frequency))
    length = max(1, int(max_seconds * frequency))
    trimmed = data[start : start + length].copy()
    if trimmed.shape[0] < length:
        pad = np.zeros((length - trimmed.shape[0], 2), dtype=np.float32)
        trimmed = np.vstack([trimmed, pad])

    fade_samples = min(int(0.025 * frequency), max(1, trimmed.shape[0] // 8))
    if fade_samples:
        fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        trimmed[-fade_samples:] *= fade[:, None]
    trimmed *= gain
    trimmed = np.clip(trimmed, -32768, 32767).astype(np.int16)
    write_wav(dst, trimmed, frequency)


def write_wav(path: Path, samples: np.ndarray, frequency: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(frequency)
        handle.writeframes(samples.tobytes())


def main() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    (REJECTED_DIR / ".gitkeep").touch()
    for src_rel, dst_name in COPY_ASSETS:
        src = SOUND_DIR / src_rel
        dst = RUNTIME_DIR / dst_name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
    for src_rel, dst_name, threshold, preroll, max_seconds, gain in TRIM_ASSETS:
        src = SOUND_DIR / src_rel
        dst = RUNTIME_DIR / dst_name
        if not src.exists():
            raise FileNotFoundError(src)
        trim_to_transient(src, dst, threshold, preroll, max_seconds, gain)
    for stale_name in ("goal_explosion_01.mp3", "goal_explosion_02.mp3", "bass_hit_01.mp3", "cup_reveal_stinger.mp3"):
        stale = RUNTIME_DIR / stale_name
        if stale.exists():
            stale.unlink()
    print(f"promoted {len(COPY_ASSETS) + len(TRIM_ASSETS)} runtime audio assets to {RUNTIME_DIR}")


if __name__ == "__main__":
    main()

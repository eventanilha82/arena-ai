from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioAsset:
    filename: str
    bus: str
    role: str
    min_duration: float
    max_duration: float
    transient_start_limit: float | None = None


AUDIO_ASSETS: tuple[AudioAsset, ...] = (
    AudioAsset("stadium_base_loop.mp3", "ambience", "stadium_base", 20.0, 60.0),
    AudioAsset("stadium_air_loop.wav", "ambience", "stadium_air", 45.0, 80.0),
    AudioAsset("crowd_light_loop.mp3", "crowd", "light_crowd", 5.0, 20.0),
    AudioAsset("crowd_tension_loop.mp3", "crowd", "crowd_tension", 20.0, 100.0),
    AudioAsset("crowd_attack_rise.wav", "crowd", "crowd_attack", 8.0, 12.0),
    AudioAsset("crowd_attack_short.wav", "crowd", "crowd_attack_short", 3.0, 5.0),
    AudioAsset("crowd_chant_loop.mp3", "crowd", "chant", 8.0, 24.0),
    AudioAsset("goal_roar_main.mp3", "goal", "goal_roar", 8.0, 24.0),
    AudioAsset("goal_roar_pixabay_01.wav", "goal", "goal_roar", 6.0, 8.8, 0.090),
    AudioAsset("goal_roar_pixabay_02.wav", "goal", "goal_roar", 6.0, 8.8, 0.090),
    AudioAsset("goal_crowd_cc0.wav", "goal", "goal_crowd_tail", 10.0, 18.0),
    AudioAsset("goal_explosion_01.wav", "goal", "goal_explosion", 2.5, 4.5, 0.080),
    AudioAsset("goal_explosion_02.wav", "goal", "goal_explosion", 2.5, 4.5, 0.080),
    AudioAsset("goal_explosion_03.wav", "goal", "goal_explosion_alt", 2.5, 4.5, 0.080),
    AudioAsset("bass_hit_01.wav", "goal", "bass_hit", 0.4, 2.5, 0.090),
    AudioAsset("stadium_reverb_tail.mp3", "goal", "reverb_tail", 3.0, 12.0),
    AudioAsset("kick_grass_01.wav", "ball", "kick", 0.35, 0.70, 0.060),
    AudioAsset("kick_grass_02.wav", "ball", "kick_alt", 0.35, 0.70, 0.060),
    AudioAsset("kick_grass_03.wav", "ball", "kick_alt", 0.35, 0.70, 0.060),
    AudioAsset("kick_grass_04.wav", "ball", "kick_alt", 0.35, 0.70, 0.060),
    AudioAsset("ball_whoosh_01.wav", "ball", "whoosh", 0.45, 0.90, 0.080),
    AudioAsset("ball_whoosh_02.wav", "ball", "whoosh_alt", 0.45, 0.90, 0.080),
    AudioAsset("net_ripple_01.wav", "ball", "net", 0.70, 1.35, 0.060),
    AudioAsset("net_ripple_02.wav", "ball", "net_alt", 0.45, 0.85, 0.060),
    AudioAsset("crowd_reaction_01.wav", "crowd", "crowd_reaction", 2.8, 4.0, 0.080),
    AudioAsset("crowd_reaction_02.wav", "crowd", "crowd_reaction", 2.8, 4.0, 0.080),
    AudioAsset("crowd_near_miss_01.wav", "crowd", "near_miss_reaction", 2.8, 3.8, 0.080),
    AudioAsset("ui_chime_01.wav", "ui", "ui_chime", 1.0, 4.0),
    AudioAsset("cup_progress_tick_01.wav", "ui", "cup_tick", 0.20, 0.50, 0.050),
    AudioAsset("cup_reveal_stinger.wav", "ui", "cup_reveal", 2.5, 6.0, 0.090),
    AudioAsset("opening_theme.mp3", "music", "opening_theme", 10.0, 20.0),
    AudioAsset("whistle_start_01.wav", "ui", "whistle_start", 0.20, 0.75, 0.060),
    AudioAsset("whistle_final_01.wav", "ui", "whistle_final", 0.9, 1.8, 0.080),
)

AUDIO_ASSET_BY_NAME = {asset.filename: asset for asset in AUDIO_ASSETS}
AUDIO_RUNTIME_FILES = tuple(asset.filename for asset in AUDIO_ASSETS)
AUDIO_DURATION_LIMITS = {asset.filename: (asset.min_duration, asset.max_duration) for asset in AUDIO_ASSETS}
AUDIO_TRANSIENT_START_LIMITS = {
    asset.filename: asset.transient_start_limit
    for asset in AUDIO_ASSETS
    if asset.transient_start_limit is not None
}
AUDIO_BUSES = frozenset(asset.bus for asset in AUDIO_ASSETS)
REQUIRED_AUDIO_BUSES = frozenset({"ambience", "crowd", "ball", "goal", "ui", "music"})
GOAL_AUDIO_SEQUENCE = ("kick", "whoosh", "net", "bass", "cheer", "reverb")
KICK_SOUND_BAG = ("kick_grass_01.wav", "kick_grass_02.wav", "kick_grass_03.wav", "kick_grass_04.wav")
WHOOSH_SOUND_BAG = ("ball_whoosh_01.wav", "ball_whoosh_02.wav")
NET_SOUND_BAG = ("net_ripple_01.wav", "net_ripple_02.wav")
GOAL_ROAR_SOUND_BAG = ("goal_roar_main.mp3", "goal_roar_pixabay_01.wav", "goal_roar_pixabay_02.wav")
GOAL_EXPLOSION_SOUND_BAG = ("goal_explosion_01.wav", "goal_explosion_02.wav", "goal_explosion_03.wav")
CROWD_REACTION_SOUND_BAG = ("crowd_attack_short.wav", "crowd_reaction_01.wav", "crowd_reaction_02.wav")
NEAR_MISS_REACTION_SOUND_BAG = ("crowd_near_miss_01.wav", "crowd_attack_short.wav", "crowd_reaction_01.wav")
CUP_REVEAL_CROWD_SOUND_BAG = ("crowd_attack_short.wav", "crowd_reaction_01.wav", "crowd_reaction_02.wav")
CUP_PROGRESS_MARKERS = (25, 50, 75, 100)


def audio_asset(filename: str) -> AudioAsset:
    try:
        return AUDIO_ASSET_BY_NAME[filename]
    except KeyError as exc:
        raise KeyError(f"audio asset is not registered in runtime manifest: {filename}") from exc

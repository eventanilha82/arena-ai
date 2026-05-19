from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pygame

from .audio_manifest import (
    CROWD_REACTION_SOUND_BAG,
    CUP_REVEAL_CROWD_SOUND_BAG,
    GOAL_EXPLOSION_SOUND_BAG,
    GOAL_ROAR_SOUND_BAG,
    KICK_SOUND_BAG,
    NEAR_MISS_REACTION_SOUND_BAG,
    NET_SOUND_BAG,
    WHOOSH_SOUND_BAG,
    audio_asset,
)


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
AUDIO_DIR = ROOT / "assets" / "sounds" / "runtime_assets"
MIXER_FREQUENCY = 44100
MIXER_SIZE = -16
MIXER_CHANNELS = 2
MIXER_BUFFER = 512
CUP_TICK_QUEUE_LIMIT = 6
CUP_TICK_SPACING_MS = 190
ONE_SHOT_CHANNELS = (
    "ball",
    "motion",
    "net",
    "explosion",
    "bass",
    "reverb",
    "ui",
    "react",
    "roar",
    "whistle",
    "tick",
    "goal_alt",
    "ball_alt",
    "ui_alt",
)
AUDIO_SOUND_VOLUMES = {
    "stadium_base_loop.mp3": 0.26,
    "stadium_air_loop.wav": 0.10,
    "crowd_light_loop.mp3": 0.08,
    "crowd_tension_loop.mp3": 0.07,
    "crowd_attack_rise.wav": 0.14,
    "crowd_attack_short.wav": 0.12,
    "crowd_chant_loop.mp3": 0.06,
    "opening_theme.mp3": 0.18,
    "stadium_reverb_tail.mp3": 0.22,
    "bass_hit_01.wav": 0.42,
    "goal_roar_main.mp3": 0.44,
    "goal_roar_pixabay_01.wav": 0.40,
    "goal_roar_pixabay_02.wav": 0.34,
    "goal_crowd_cc0.wav": 0.24,
    "goal_explosion_01.wav": 0.36,
    "goal_explosion_02.wav": 0.32,
    "goal_explosion_03.wav": 0.30,
    "crowd_reaction_01.wav": 0.13,
    "crowd_reaction_02.wav": 0.12,
    "crowd_near_miss_01.wav": 0.12,
    "ui_chime_01.wav": 0.24,
    "cup_progress_tick_01.wav": 0.42,
    "cup_reveal_stinger.wav": 0.28,
    "kick_grass_01.wav": 0.52,
    "kick_grass_02.wav": 0.38,
    "kick_grass_03.wav": 0.40,
    "kick_grass_04.wav": 0.34,
    "ball_whoosh_01.wav": 0.15,
    "ball_whoosh_02.wav": 0.11,
    "net_ripple_01.wav": 0.34,
    "net_ripple_02.wav": 0.25,
    "whistle_start_01.wav": 0.20,
    "whistle_final_01.wav": 0.24,
}


@dataclass(frozen=True)
class AudioCuePolicy:
    suppress_ms: int = 0
    next_reaction_ms: int = 0
    duck_seconds: float = 0.0
    goal_boost_ms: int = 0
    impact_focus_ms: int = 0


@dataclass(frozen=True)
class AudioCueMix:
    volume: float
    maxtime: float
    pan_scale: float = 1.0


AUDIO_CUE_POLICIES: dict[str, AudioCuePolicy] = {
    "analysis": AudioCuePolicy(duck_seconds=0.55),
    "ui_chime": AudioCuePolicy(duck_seconds=0.55),
    "cup_start": AudioCuePolicy(duck_seconds=0.45, suppress_ms=600),
    "cup_reveal": AudioCuePolicy(duck_seconds=0.45, suppress_ms=1600, next_reaction_ms=1800, goal_boost_ms=3200),
    "whistle": AudioCuePolicy(suppress_ms=1100, next_reaction_ms=1400),
    "final_whistle": AudioCuePolicy(suppress_ms=1400, next_reaction_ms=1600, duck_seconds=0.45),
    "kick": AudioCuePolicy(suppress_ms=760),
    "whoosh": AudioCuePolicy(suppress_ms=620),
    "net": AudioCuePolicy(suppress_ms=620, impact_focus_ms=320),
    "save": AudioCuePolicy(suppress_ms=1250, next_reaction_ms=1500, duck_seconds=0.28),
    "near_miss": AudioCuePolicy(suppress_ms=1100, next_reaction_ms=1350, duck_seconds=0.28),
    "bass": AudioCuePolicy(suppress_ms=1200, impact_focus_ms=420),
    "cheer": AudioCuePolicy(suppress_ms=2600, next_reaction_ms=3000, goal_boost_ms=6200, impact_focus_ms=520),
    "reverb": AudioCuePolicy(suppress_ms=900),
}

MATCH_CUE_MIX: dict[str, AudioCueMix] = {
    "kick": AudioCueMix(0.52, 0.560),
    "whoosh": AudioCueMix(0.14, 0.720),
    "net": AudioCueMix(0.36, 0.980),
    "bass": AudioCueMix(0.46, 1.100, 0.35),
    "goal_roar": AudioCueMix(0.58, 6.800, 0.20),
    "goal_explosion": AudioCueMix(0.46, 4.200, 0.45),
    "goal_cc0_tail": AudioCueMix(0.20, 4.200, 0.18),
    "goal_attack_swell": AudioCueMix(0.23, 4.200, 0.16),
    "reverb": AudioCueMix(0.26, 2.800, 0.22),
}
GOAL_ATTACK_SWELL_FILENAME = "crowd_attack_rise.wav"
GOAL_CC0_TAIL_FILENAME = "goal_crowd_cc0.wav"


def pre_init_mixer() -> None:
    pygame.mixer.pre_init(MIXER_FREQUENCY, MIXER_SIZE, MIXER_CHANNELS, MIXER_BUFFER)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class AudioBus:
    def __init__(self, name: str, channels: dict[str, pygame.mixer.Channel], channel_names: tuple[str, ...], master: float = 1.0):
        self.name = name
        self.channels = channels
        self.channel_names = channel_names
        self.master = master
        self._cursor = 0
        self._cooldowns: dict[str, int] = {}

    def play(
        self,
        sound: pygame.mixer.Sound | None,
        *,
        volume: float,
        key: str,
        preferred: str | None = None,
        cooldown_ms: int = 0,
        loops: int = 0,
        maxtime: int = 0,
        fade_ms: int = 0,
        pan: float = 0.0,
    ) -> pygame.mixer.Channel | None:
        if sound is None:
            return None
        now = pygame.time.get_ticks()
        if cooldown_ms and now < self._cooldowns.get(key, 0):
            return None
        channel = self._select_channel(preferred)
        if channel is None:
            return None
        effective = clamp(volume * self.master)
        pan = clamp(pan, -1.0, 1.0)
        left = effective * (1.0 - max(0.0, pan))
        right = effective * (1.0 + min(0.0, pan))
        channel.set_volume(left, right)
        channel.play(sound, loops, maxtime, fade_ms)
        if cooldown_ms:
            self._cooldowns[key] = now + cooldown_ms
        return channel

    def _select_channel(self, preferred: str | None) -> pygame.mixer.Channel | None:
        if preferred and preferred in self.channels:
            return self.channels[preferred]
        if not self.channel_names:
            return None
        for offset in range(len(self.channel_names)):
            name = self.channel_names[(self._cursor + offset) % len(self.channel_names)]
            channel = self.channels.get(name)
            if channel and not channel.get_busy():
                self._cursor = (self._cursor + offset + 1) % len(self.channel_names)
                return channel
        return self.channels.get(self.channel_names[self._cursor % len(self.channel_names)])


class AudioEngine:
    def __init__(self, seed: int | None = None):
        self.disabled_reason = ""
        self.rng = random.Random(seed)
        self.duck_until_ms = 0
        self.goal_boost_until_ms = 0
        self.impact_focus_until_ms = 0
        self.next_reaction_ms = 0
        self.suppress_reactions_until_ms = 0
        self.pending_cup_ticks = 0
        self.pending_cup_reveal = False
        self.next_cup_tick_ms = 0
        self.scene = "menu"
        self._last_bag_choice: dict[str, int] = {}
        self.layer_volumes = {
            "base": 0.0,
            "air": 0.0,
            "light": 0.0,
            "tension": 0.0,
            "chant": 0.0,
            "music": 0.0,
        }
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=MIXER_FREQUENCY,
                    size=MIXER_SIZE,
                    channels=MIXER_CHANNELS,
                    buffer=MIXER_BUFFER,
                )
            pygame.mixer.set_num_channels(20)
            self.channels = {
                "ball": pygame.mixer.Channel(0),
                "motion": pygame.mixer.Channel(1),
                "net": pygame.mixer.Channel(2),
                "explosion": pygame.mixer.Channel(3),
                "bass": pygame.mixer.Channel(4),
                "reverb": pygame.mixer.Channel(5),
                "ui": pygame.mixer.Channel(6),
                "base": pygame.mixer.Channel(7),
                "light": pygame.mixer.Channel(8),
                "tension": pygame.mixer.Channel(9),
                "chant": pygame.mixer.Channel(10),
                "react": pygame.mixer.Channel(11),
                "roar": pygame.mixer.Channel(12),
                "air": pygame.mixer.Channel(13),
                "whistle": pygame.mixer.Channel(14),
                "music": pygame.mixer.Channel(15),
                "tick": pygame.mixer.Channel(16),
                "goal_alt": pygame.mixer.Channel(17),
                "ball_alt": pygame.mixer.Channel(18),
                "ui_alt": pygame.mixer.Channel(19),
            }
        except (pygame.error, OSError) as exc:
            self.disabled_reason = str(exc)
            self.channels = {}
            self.buses = {}
            return

        self.buses = {
            "ambience": AudioBus("ambience", self.channels, ("base", "air"), 1.0),
            "crowd": AudioBus("crowd", self.channels, ("light", "tension", "chant", "react", "roar"), 1.0),
            "ball": AudioBus("ball", self.channels, ("ball", "motion", "net", "ball_alt"), 1.0),
            "goal": AudioBus("goal", self.channels, ("explosion", "bass", "reverb", "roar", "goal_alt"), 1.0),
            "ui": AudioBus("ui", self.channels, ("ui", "ui_alt", "tick", "whistle"), 1.0),
            "music": AudioBus("music", self.channels, ("music",), 1.0),
        }

        self.opening_crowd = self.load_sound("stadium_base_loop.mp3")
        self.stadium_base = self.opening_crowd
        self.stadium_air = self.load_sound("stadium_air_loop.wav")
        self.light_crowd = self.load_sound("crowd_light_loop.mp3")
        self.crowd_tension = self.load_sound("crowd_tension_loop.mp3")
        self.crowd_attack = self.load_sound(GOAL_ATTACK_SWELL_FILENAME)
        self.crowd_attack_short = self.load_sound("crowd_attack_short.wav")
        self.chant = self.load_sound("crowd_chant_loop.mp3")
        self.opening_theme = self.load_sound("opening_theme.mp3")
        self.reverb_tail = self.load_sound("stadium_reverb_tail.mp3")
        self.bass_hit = self.load_sound("bass_hit_01.wav")
        self.goal_roars = [sound for filename in GOAL_ROAR_SOUND_BAG if (sound := self.load_sound(filename)) is not None]
        self.goal_roar = self.goal_roars[0] if self.goal_roars else None
        self.goal_crowd_cc0 = self.load_sound(GOAL_CC0_TAIL_FILENAME)
        self.goal_explosions = [sound for filename in GOAL_EXPLOSION_SOUND_BAG if (sound := self.load_sound(filename)) is not None]
        self.crowd_reaction = self.load_sound("crowd_reaction_01.wav")
        self.crowd_reaction_alt = self.load_sound("crowd_reaction_02.wav")
        self.crowd_near_miss = self.load_sound("crowd_near_miss_01.wav")
        crowd_sound_by_name = {
            "crowd_attack_short.wav": self.crowd_attack_short,
            "crowd_reaction_01.wav": self.crowd_reaction,
            "crowd_reaction_02.wav": self.crowd_reaction_alt,
            "crowd_near_miss_01.wav": self.crowd_near_miss,
        }
        self.crowd_reactions = [sound for filename in CROWD_REACTION_SOUND_BAG if (sound := crowd_sound_by_name[filename]) is not None]
        self.near_miss_reactions = [sound for filename in NEAR_MISS_REACTION_SOUND_BAG if (sound := crowd_sound_by_name[filename]) is not None]
        self.cup_reveal_reactions = [sound for filename in CUP_REVEAL_CROWD_SOUND_BAG if (sound := crowd_sound_by_name[filename]) is not None]
        self.analysis = self.load_sound("ui_chime_01.wav")
        self.ui_chime = self.analysis
        self.cup_tick = self.load_sound("cup_progress_tick_01.wav")
        self.cup_reveal = self.load_sound("cup_reveal_stinger.wav")
        self.kick = self.load_sound(KICK_SOUND_BAG[0])
        self.kick_alt = self.load_sound(KICK_SOUND_BAG[1])
        self.kick_more = self.load_sound(KICK_SOUND_BAG[2])
        self.kick_soft = self.load_sound(KICK_SOUND_BAG[3])
        self.whoosh = self.load_sound(WHOOSH_SOUND_BAG[0])
        self.whoosh_alt = self.load_sound(WHOOSH_SOUND_BAG[1])
        self.net = self.load_sound(NET_SOUND_BAG[0])
        self.net_alt = self.load_sound(NET_SOUND_BAG[1])
        self.kick_bag = [sound for sound in (self.kick, self.kick_alt, self.kick_more, self.kick_soft) if sound is not None]
        self.whoosh_bag = [sound for sound in (self.whoosh, self.whoosh_alt) if sound is not None]
        self.net_bag = [sound for sound in (self.net, self.net_alt) if sound is not None]
        self.whistle = self.load_sound("whistle_start_01.wav")
        self.final_whistle = self.load_sound("whistle_final_01.wav")

        self.start_loop("base", self.stadium_base, 0.22, 1800)
        self.start_loop("air", self.stadium_air, 0.07, 2200)
        self.start_loop("light", self.light_crowd, 0.05, 2200)
        self.start_loop("tension", self.crowd_tension, 0.0, 2500)
        self.start_loop("chant", self.chant, 0.0, 3000)
        self.start_loop("music", self.opening_theme, 0.0, 1800)

    def load_sound(self, filename: str) -> pygame.mixer.Sound | None:
        try:
            sound = pygame.mixer.Sound(AUDIO_DIR / audio_asset(filename).filename)
            sound.set_volume(AUDIO_SOUND_VOLUMES[filename])
            return sound
        except (pygame.error, OSError) as exc:
            self.disabled_reason = str(exc)
            return None

    def choose_bag(self, key: str, sounds: list[pygame.mixer.Sound]) -> pygame.mixer.Sound | None:
        if not sounds:
            return None
        if len(sounds) == 1:
            return sounds[0]
        last = self._last_bag_choice.get(key, -1)
        choices = [index for index in range(len(sounds)) if index != last]
        index = self.rng.choice(choices)
        self._last_bag_choice[key] = index
        return sounds[index]

    def set_scene(self, scene: str) -> None:
        self.scene = scene

    def reset_scene_queues(self) -> None:
        self.pending_cup_ticks = 0
        self.pending_cup_reveal = False
        self.next_cup_tick_ms = 0

    def start_loop(self, channel_name: str, sound: pygame.mixer.Sound | None, volume: float, fade_ms: int) -> None:
        channel = self.channels.get(channel_name)
        if not channel or not sound:
            return
        channel.set_volume(volume)
        channel.play(sound, -1, 0, fade_ms)
        self.layer_volumes[channel_name] = volume

    def stop_one_shots(self, fade_ms: int = 180) -> None:
        if not self.channels:
            return
        for name in ONE_SHOT_CHANNELS:
            channel = self.channels.get(name)
            if channel and channel.get_busy():
                if fade_ms <= 0:
                    channel.stop()
                else:
                    channel.fadeout(fade_ms)
        now = pygame.time.get_ticks()
        self.duck_until_ms = min(self.duck_until_ms, now + fade_ms)
        self.goal_boost_until_ms = min(self.goal_boost_until_ms, now + fade_ms)
        self.impact_focus_until_ms = min(self.impact_focus_until_ms, now + fade_ms)
        self.suppress_reactions_until_ms = min(self.suppress_reactions_until_ms, now + fade_ms)
        self.reset_scene_queues()

    def duck_commentary(self, seconds: float = 1.25) -> None:
        self.duck_until_ms = max(self.duck_until_ms, pygame.time.get_ticks() + int(seconds * 1000))

    def arm_event(self, name: str) -> None:
        policy = AUDIO_CUE_POLICIES.get(name)
        if policy is None:
            return
        now = pygame.time.get_ticks()
        if policy.duck_seconds > 0:
            self.duck_commentary(policy.duck_seconds)
        if policy.suppress_ms > 0:
            self.suppress_reactions_until_ms = max(self.suppress_reactions_until_ms, now + policy.suppress_ms)
        if policy.next_reaction_ms > 0:
            self.next_reaction_ms = max(self.next_reaction_ms, now + policy.next_reaction_ms)
        if policy.goal_boost_ms > 0:
            self.goal_boost_until_ms = max(self.goal_boost_until_ms, now + policy.goal_boost_ms)
        if policy.impact_focus_ms > 0:
            self.impact_focus_until_ms = max(self.impact_focus_until_ms, now + policy.impact_focus_ms)

    def set_layer_volume(self, name: str, target: float, dt: float) -> None:
        channel = self.channels.get(name)
        if not channel:
            return
        current = self.layer_volumes.get(name, 0.0)
        speed = clamp(dt * 4.8)
        volume = current + (target - current) * speed
        channel.set_volume(volume)
        self.layer_volumes[name] = volume

    def update_crowd(self, intensity: float, goal_active: bool, dt: float, allow_reactions: bool = True) -> None:
        if not self.channels:
            return
        if self.scene == "tournament":
            self.flush_cup_audio_queue()
        now = pygame.time.get_ticks()
        intensity = clamp(intensity)
        duck = now < self.duck_until_ms
        impact_focus = now < self.impact_focus_until_ms
        boost = clamp((self.goal_boost_until_ms - now) / 3600)
        bed_duck = 0.86 if duck else 1.0
        crowd_duck = 0.66 if duck else 1.0
        music_duck = 0.34 if duck else 1.0
        focus_scale = 0.76 if impact_focus and not goal_active else 1.0
        scene_boost = 0.08 if self.scene == "menu" else 0.04 if self.scene == "tournament" else 0.0
        music_target = {
            "menu": 0.18,
            "select": 0.038,
            "tournament": 0.042,
            "simulate": 0.0,
        }.get(self.scene, 0.0)
        self.set_layer_volume("music", music_target * music_duck, dt)
        self.set_layer_volume("base", (0.20 + scene_boost + 0.08 * intensity + 0.05 * boost) * bed_duck, dt)
        self.set_layer_volume("air", (0.055 + 0.045 * intensity + 0.025 * boost) * bed_duck, dt)
        self.set_layer_volume("light", (0.045 + 0.045 * intensity + 0.03 * boost) * crowd_duck, dt)
        self.set_layer_volume("tension", (0.015 + 0.22 * intensity + (0.09 if goal_active else 0.0)) * crowd_duck * focus_scale, dt)
        self.set_layer_volume("chant", (0.02 + 0.11 * intensity + 0.28 * boost) * crowd_duck, dt)
        if not allow_reactions or now < self.suppress_reactions_until_ms or now < self.next_reaction_ms or not self.crowd_reactions:
            return
        reaction = self.choose_bag("ambient_crowd_reaction", self.crowd_reactions)
        channel = self.channels.get("react")
        if channel:
            channel.set_volume((0.045 + 0.08 * intensity) * crowd_duck)
            channel.play(reaction, 0, self.rng.randint(1200, 2600), 180)
        self.next_reaction_ms = now + self.rng.randint(5200, 9800)

    def flush_cup_audio_queue(self) -> None:
        if not self.channels:
            return
        now = pygame.time.get_ticks()
        tick_channel = self.channels.get("tick")
        if self.pending_cup_ticks > 0 and tick_channel and not tick_channel.get_busy() and now >= self.next_cup_tick_ms:
            played = self.buses["ui"].play(self.cup_tick, volume=0.36, key="cup_tick_queue", preferred="tick", maxtime=380)
            if played is not None:
                self.pending_cup_ticks -= 1
                self.next_cup_tick_ms = now + CUP_TICK_SPACING_MS
        if self.pending_cup_ticks == 0 and self.pending_cup_reveal and (not tick_channel or not tick_channel.get_busy()):
            self.pending_cup_reveal = False
            self._play_cup_reveal_now()

    def _queue_cup_tick(self) -> None:
        self.pending_cup_ticks = min(CUP_TICK_QUEUE_LIMIT, self.pending_cup_ticks + 1)
        self.flush_cup_audio_queue()

    def _play_cup_reveal_now(self) -> None:
        self.arm_event("cup_reveal")
        self.buses["ui"].play(self.cup_reveal, volume=0.30, key="cup_reveal", preferred="ui_alt", cooldown_ms=1000, maxtime=2600, fade_ms=80)
        self.buses["crowd"].play(
            self.choose_bag("cup_reveal_crowd", self.cup_reveal_reactions),
            volume=0.13,
            key="cup_reveal_crowd",
            preferred="react",
            cooldown_ms=1000,
            maxtime=2200,
            fade_ms=120,
        )

    def play(self, name: str, pan: float = 0.0) -> None:
        if not self.channels:
            return
        pan = clamp(pan, -0.75, 0.75)
        if name in {"analysis", "ui_chime"}:
            self.arm_event(name)
            self.buses["ui"].play(self.ui_chime, volume=0.20, key="ui_chime", preferred="ui", cooldown_ms=180, maxtime=900, fade_ms=12)
            return
        if name == "cup_start":
            self.arm_event(name)
            self.buses["ui"].play(self.ui_chime, volume=0.17, key="cup_start", preferred="ui", cooldown_ms=250, maxtime=700, fade_ms=12)
            return
        if name == "cup_tick":
            self._queue_cup_tick()
            return
        if name == "cup_reveal":
            tick_channel = self.channels.get("tick")
            if self.pending_cup_ticks > 0 or (tick_channel is not None and tick_channel.get_busy()):
                self.pending_cup_reveal = True
                self.arm_event("cup_reveal")
                self.flush_cup_audio_queue()
            else:
                self._play_cup_reveal_now()
            return
        if name == "whistle":
            self.arm_event(name)
            self.buses["ui"].play(self.whistle, volume=0.20, key="whistle", preferred="whistle", cooldown_ms=450)
            return
        if name == "final_whistle":
            self.arm_event(name)
            self.buses["ui"].play(self.final_whistle, volume=0.24, key="final_whistle", preferred="whistle", cooldown_ms=1200, maxtime=1400)
            return
        if name == "kick":
            self.arm_event(name)
            sound = self.choose_bag("kick", self.kick_bag)
            cue = MATCH_CUE_MIX["kick"]
            self.buses["ball"].play(sound, volume=cue.volume, key="kick", preferred="ball", cooldown_ms=120, maxtime=int(cue.maxtime * 1000), pan=pan)
            return
        if name == "whoosh":
            self.arm_event(name)
            whoosh_pan = clamp(pan + self.rng.uniform(-0.12, 0.12), -0.75, 0.75)
            cue = MATCH_CUE_MIX["whoosh"]
            self.buses["ball"].play(self.choose_bag("whoosh", self.whoosh_bag), volume=cue.volume, key="whoosh", preferred="motion", cooldown_ms=120, maxtime=int(cue.maxtime * 1000), pan=whoosh_pan)
            return
        if name == "net":
            self.arm_event(name)
            cue = MATCH_CUE_MIX["net"]
            self.buses["ball"].play(self.choose_bag("net", self.net_bag), volume=cue.volume, key="net", preferred="net", cooldown_ms=180, maxtime=int(cue.maxtime * 1000), pan=pan)
            return
        if name == "save":
            self.arm_event(name)
            self.buses["ball"].play(self.choose_bag("save_net", self.net_bag), volume=0.20, key="save_ball", preferred="net", cooldown_ms=240, maxtime=620, fade_ms=20, pan=pan)
            self.buses["crowd"].play(self.choose_bag("save_reaction", self.crowd_reactions), volume=0.12, key="save_reaction", preferred="react", cooldown_ms=650, maxtime=1700, fade_ms=80, pan=pan * 0.25)
            return
        if name == "near_miss":
            self.arm_event(name)
            self.buses["crowd"].play(self.choose_bag("near_miss_reaction", self.near_miss_reactions), volume=0.10, key="near_miss_reaction", preferred="react", cooldown_ms=650, maxtime=1800, fade_ms=90, pan=pan * 0.20)
            self.buses["goal"].play(self.reverb_tail, volume=0.08, key="near_miss_tail", preferred="reverb", cooldown_ms=700, maxtime=1450, fade_ms=180, pan=pan * 0.18)
            return
        if name == "bass":
            self.arm_event(name)
            cue = MATCH_CUE_MIX["bass"]
            self.buses["goal"].play(self.bass_hit, volume=cue.volume, key="bass", preferred="bass", cooldown_ms=300, maxtime=int(cue.maxtime * 1000), pan=pan * cue.pan_scale)
            return
        if name == "cheer":
            self.arm_event(name)
            roar = MATCH_CUE_MIX["goal_roar"]
            self.buses["goal"].play(self.choose_bag("goal_roar", self.goal_roars), volume=roar.volume, key="goal_roar", preferred="roar", cooldown_ms=700, maxtime=int(roar.maxtime * 1000), fade_ms=55, pan=pan * roar.pan_scale)
            explosion = self.choose_bag("goal_explosion", self.goal_explosions) or self.goal_crowd_cc0
            explosion_cue = MATCH_CUE_MIX["goal_explosion"]
            self.buses["goal"].play(explosion, volume=explosion_cue.volume, key="goal_explosion", preferred="explosion", cooldown_ms=600, maxtime=int(explosion_cue.maxtime * 1000), fade_ms=60, pan=pan * explosion_cue.pan_scale)
            cc0 = MATCH_CUE_MIX["goal_cc0_tail"]
            self.buses["goal"].play(self.goal_crowd_cc0, volume=cc0.volume, key="goal_cc0_tail", preferred="goal_alt", cooldown_ms=900, maxtime=int(cc0.maxtime * 1000), fade_ms=120, pan=pan * cc0.pan_scale)
            swell = MATCH_CUE_MIX["goal_attack_swell"]
            self.buses["crowd"].play(self.crowd_attack, volume=swell.volume, key="goal_attack_swell", preferred="react", cooldown_ms=900, maxtime=int(swell.maxtime * 1000), fade_ms=70, pan=pan * swell.pan_scale)
            return
        if name == "reverb":
            self.arm_event(name)
            cue = MATCH_CUE_MIX["reverb"]
            self.buses["goal"].play(self.reverb_tail, volume=cue.volume, key="reverb", preferred="reverb", cooldown_ms=360, maxtime=int(cue.maxtime * 1000), fade_ms=260, pan=pan * cue.pan_scale)

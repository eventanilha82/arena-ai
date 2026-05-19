from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from arena_ai.main import (
    App,
    CHANCE_CONTACT_AUDIO_AT,
    CHANCE_CONTACT_VISUAL_AT,
    CHANCE_EVENT_WINDOW_MINUTES,
    GOAL_EVENT_WINDOW_MINUTES,
    SHOT_BASS_AUDIO_AT,
    SHOT_CHEER_AUDIO_AT,
    SHOT_KICK_AUDIO_AT,
    SHOT_KICK_AT,
    SHOT_NET_AUDIO_AT,
    SHOT_NET_AT,
    SHOT_NET_VISUAL_CONTACT_AT,
    SHOT_REVERB_AT,
    SHOT_WHOOSH_AUDIO_AT,
    SHOT_WHOOSH_AT,
    SIMULATION_SECONDS,
    TOURNAMENT_MIN_LOADING_SECONDS,
    GOAL_IMPACT_AUDIO_EVENTS,
)
from arena_ai.audio import (
    AUDIO_CUE_POLICIES,
    AUDIO_SOUND_VOLUMES,
    GOAL_ATTACK_SWELL_FILENAME,
    GOAL_CC0_TAIL_FILENAME,
    MATCH_CUE_MIX,
    ONE_SHOT_CHANNELS,
)
from arena_ai.audio_manifest import (
    AUDIO_ASSETS,
    AUDIO_BUSES,
    AUDIO_DURATION_LIMITS,
    CROWD_REACTION_SOUND_BAG,
    CUP_REVEAL_CROWD_SOUND_BAG,
    GOAL_EXPLOSION_SOUND_BAG,
    GOAL_ROAR_SOUND_BAG,
    AUDIO_RUNTIME_FILES,
    AUDIO_TRANSIENT_START_LIMITS,
    CUP_PROGRESS_MARKERS,
    GOAL_AUDIO_SEQUENCE,
    KICK_SOUND_BAG,
    NEAR_MISS_REACTION_SOUND_BAG,
    NET_SOUND_BAG,
    REQUIRED_AUDIO_BUSES,
    WHOOSH_SOUND_BAG,
)


RUNTIME_DIR = ROOT / "assets" / "sounds" / "runtime_assets"
AUDIO_MANIFEST_JSON = ROOT / "assets" / "sounds" / "audio_manifest.json"
DOWNLOADED_MANIFEST = ROOT / "assets" / "sounds" / "downloaded_audio_manifest.csv"
MAIN_PY = SRC / "arena_ai" / "main.py"
UNRESOLVED_LICENSE_TOKENS = ("verificar", "pendente", "confirmar", "todo")
APPROVED_LICENSE_TOKENS = (
    "Creative Commons 0",
    "Mixkit Free License",
    "Pixabay Content License confirmado",
    "Acervo do projeto",
)
MIX_PEAK_LIMIT = 0.55
MIX_NEAR_CLIP_AMPLITUDE = 0.98
MIX_NEAR_CLIP_RATIO_LIMIT = 0.0005
MIX_RMS_DBFS_MIN = -34.0
MIX_RMS_DBFS_MAX = -26.8
MIX_RMS_DBFS_SPREAD_LIMIT = 4.6
SHORT_MIX_PEAK_LIMIT = 0.72
SHORT_MIX_NEAR_CLIP_RATIO_LIMIT = 0.0001
SHORT_MIX_RMS_DBFS_MAX = -22.0
GOAL_MIX_CROWD_INTENSITY = 0.92
GOAL_MIX_BOOST = 1.0
GOAL_MIX_LOOP_START_RATIO = 0.33
LOOP_SEAM_FILES = (
    "stadium_base_loop.mp3",
    "stadium_air_loop.wav",
    "crowd_light_loop.mp3",
    "crowd_tension_loop.mp3",
    "crowd_chant_loop.mp3",
    "opening_theme.mp3",
)
LOOP_SEAM_WINDOW_SECONDS = 0.025
LOOP_SEAM_ENDPOINT_JUMP_LIMIT = 0.08
LOOP_SEAM_DIFF_ABSOLUTE_LIMIT = 0.12
LOOP_SEAM_DIFF_RATIO_LIMIT = 3.0
PROJECT_ARCHIVE_STATUS = "Acervo do projeto"
PROJECT_ARCHIVE_RECEIPT_REQUIRED_FIELDS = (
    "runtime_asset",
    "source_asset",
    "author",
    "origin",
    "captured_on",
    "license",
    "source_sha256",
    "runtime_sha256",
    "transform",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RELATIVE_SOURCE_RE = re.compile(r"assets/sounds/[A-Za-z0-9_./-]+\.(?:mp3|wav|txt)")
SOUND_BAG_FINGERPRINT_DISTANCE_MIN = 0.18
SOUND_BAG_CORRELATION_MAX = 0.88
SOUND_ARRAY_CACHE: dict[str, np.ndarray] = {}


def init_pygame() -> None:
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)


def sound_array(sound: pygame.mixer.Sound) -> np.ndarray:
    data = pygame.sndarray.array(sound).astype(np.float32)
    if data.ndim > 1:
        return np.max(np.abs(data), axis=1)
    return np.abs(data)


def transient_start(sound: pygame.mixer.Sound) -> float:
    frequency = int(pygame.mixer.get_init()[0])
    mono = sound_array(sound)
    peak = float(np.max(mono))
    if peak <= 0:
        return 999.0
    above = np.flatnonzero(mono >= peak * 0.08)
    return float(above[0] / frequency) if above.size else 999.0


def clipping_ratio(sound: pygame.mixer.Sound) -> float:
    mono = sound_array(sound)
    return float(np.mean(mono >= 32760.0))


def parse_runtime_source_rows() -> dict[str, dict[str, str]]:
    if not AUDIO_MANIFEST_JSON.exists():
        raise AssertionError(f"audio manifest is missing: {AUDIO_MANIFEST_JSON.relative_to(ROOT)}")
    manifest = json.loads(AUDIO_MANIFEST_JSON.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise AssertionError("audio_manifest.json must declare schema_version=1")
    if manifest.get("runtime_directory") != "assets/sounds/runtime_assets":
        raise AssertionError("audio_manifest.json runtime_directory drifted")
    if manifest.get("operational_contract") != "src/arena_ai/audio_manifest.py":
        raise AssertionError("audio_manifest.json must point at the Python operational contract")

    rows: dict[str, dict[str, str]] = {}
    runtime_assets = manifest.get("runtime_assets")
    if not isinstance(runtime_assets, list) or not runtime_assets:
        raise AssertionError("audio_manifest.json must contain a non-empty runtime_assets list")
    for item in runtime_assets:
        if not isinstance(item, dict):
            raise AssertionError("audio_manifest.json runtime_assets entries must be objects")
        filename = str(item.get("filename", ""))
        if not filename:
            raise AssertionError("audio_manifest.json runtime asset entry is missing filename")
        if filename in rows:
            raise AssertionError(f"duplicate runtime audio manifest row: {filename}")
        rows[filename] = {
            "origin": str(item.get("origin", "")),
            "license": str(item.get("license", "")),
            "usage": str(item.get("usage", "")),
            "runtime_path": str(item.get("runtime_path", "")),
            "bus": str(item.get("bus", "")),
            "role": str(item.get("role", "")),
            "runtime_sha256": str(item.get("runtime_sha256", "")),
            "duration_seconds": item.get("duration_seconds"),
            "transient_start_limit_seconds": item.get("transient_start_limit_seconds"),
            "source_paths": item.get("source_paths"),
            "project_archive_receipt": item.get("project_archive_receipt"),
        }
    if set(manifest.get("required_buses", [])) != REQUIRED_AUDIO_BUSES:
        raise AssertionError("audio_manifest.json required_buses drifted from REQUIRED_AUDIO_BUSES")
    if tuple(manifest.get("goal_audio_sequence", [])) != GOAL_AUDIO_SEQUENCE:
        raise AssertionError("audio_manifest.json goal_audio_sequence drifted from GOAL_AUDIO_SEQUENCE")
    if tuple(manifest.get("cup_progress_markers", [])) != CUP_PROGRESS_MARKERS:
        raise AssertionError("audio_manifest.json cup_progress_markers drifted from CUP_PROGRESS_MARKERS")
    return rows


def load_downloaded_manifest_rows() -> list[dict[str, str]]:
    with DOWNLOADED_MANIFEST.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def validate_runtime_source_docs(source_rows: dict[str, dict[str, str]]) -> None:
    expected = set(AUDIO_RUNTIME_FILES)
    actual = set(source_rows)
    if actual != expected:
        raise AssertionError(f"audio manifest mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    contracts = {asset.filename: asset for asset in AUDIO_ASSETS}
    for filename in sorted(expected):
        row = source_rows[filename]
        origin = row["origin"]
        license_status = row["license"]
        usage = row["usage"]
        if not origin or not license_status or not usage:
            raise AssertionError(f"{filename} has incomplete audio manifest provenance")
        if row["runtime_path"] != f"assets/sounds/runtime_assets/{filename}":
            raise AssertionError(f"{filename} has incorrect runtime_path in audio_manifest.json")
        contract = contracts[filename]
        if row["bus"] != contract.bus or row["role"] != contract.role:
            raise AssertionError(f"{filename} bus/role drifted between audio_manifest.json and audio_manifest.py")
        duration = row.get("duration_seconds")
        if not isinstance(duration, dict):
            raise AssertionError(f"{filename} must keep min/max duration_seconds in audio_manifest.json")
        if abs(float(duration.get("min", -1.0)) - contract.min_duration) > 1e-6:
            raise AssertionError(f"{filename} min_duration drifted in audio_manifest.json")
        if abs(float(duration.get("max", -1.0)) - contract.max_duration) > 1e-6:
            raise AssertionError(f"{filename} max_duration drifted in audio_manifest.json")
        manifest_transient = row.get("transient_start_limit_seconds")
        if manifest_transient is None:
            if contract.transient_start_limit is not None:
                raise AssertionError(f"{filename} lost transient_start_limit_seconds in audio_manifest.json")
        elif contract.transient_start_limit is None or abs(float(manifest_transient) - contract.transient_start_limit) > 1e-6:
            raise AssertionError(f"{filename} transient_start_limit_seconds drifted in audio_manifest.json")
        if not SHA256_RE.match(row["runtime_sha256"]):
            raise AssertionError(f"{filename} runtime_sha256 is missing or malformed in audio_manifest.json")
        if row["runtime_sha256"] != file_sha256(RUNTIME_DIR / filename):
            raise AssertionError(f"{filename} runtime_sha256 is stale in audio_manifest.json")
        combined = f"{origin} {license_status}".lower()
        if any(token in combined for token in UNRESOLVED_LICENSE_TOKENS):
            raise AssertionError(f"{filename} still has unresolved license/source wording: {license_status}")
        if not any(token in license_status for token in APPROVED_LICENSE_TOKENS):
            raise AssertionError(f"{filename} uses an unrecognized approved license/status: {license_status}")
        if "pixabay" in combined and ("https://pixabay.com/" not in origin or "confirmado" not in license_status.lower()):
            raise AssertionError(f"{filename} must keep a confirmed Pixabay runtime URL and status")
        if "freesound" in combined and "creative commons 0" not in license_status.lower():
            raise AssertionError(f"{filename} Freesound runtime asset must be CC0")
        if "mixkit" in combined and "mixkit free license" not in license_status.lower():
            raise AssertionError(f"{filename} Mixkit runtime asset must cite Mixkit Free License")


def validate_runtime_license_receipts(source_rows: dict[str, dict[str, str]]) -> None:
    manifest_rows = load_downloaded_manifest_rows()
    for runtime_filename, source_row in source_rows.items():
        origin = source_row["origin"]
        for receipt in manifest_rows:
            receipt_filename = receipt.get("filename", "")
            if not receipt_filename or receipt_filename not in origin:
                continue
            license_status = receipt.get("license", "")
            if any(token in license_status.lower() for token in UNRESOLVED_LICENSE_TOKENS):
                raise AssertionError(f"{runtime_filename} promoted source receipt still needs license confirmation: {receipt_filename}")
            if not receipt.get("url", "").startswith("https://"):
                raise AssertionError(f"{runtime_filename} promoted source receipt is missing source URL: {receipt_filename}")


def runtime_origin_paths(origin: str) -> tuple[Path, ...]:
    paths: list[Path] = []
    for match in RELATIVE_SOURCE_RE.finditer(origin):
        candidate = ROOT / match.group(0)
        if candidate not in paths:
            paths.append(candidate)
    return tuple(paths)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_promoted_source_integrity(source_rows: dict[str, dict[str, str]]) -> None:
    runtime_hashes: dict[str, str] = {}
    for runtime_filename, source_row in sorted(source_rows.items()):
        runtime_asset = RUNTIME_DIR / runtime_filename
        runtime_hash = file_sha256(runtime_asset)
        if runtime_hash in runtime_hashes:
            raise AssertionError(f"{runtime_filename} duplicates runtime audio hash from {runtime_hashes[runtime_hash]}")
        runtime_hashes[runtime_hash] = runtime_filename

        local_sources = runtime_origin_paths(source_row["origin"])
        if source_row["origin"].startswith("assets/sounds/") and not local_sources:
            raise AssertionError(f"{runtime_filename} cites a local source but no auditable relative asset path was parsed")
        for source_asset in local_sources:
            if not source_asset.exists():
                raise AssertionError(f"{runtime_filename} runtime source path is missing: {source_asset.relative_to(ROOT)}")
            if source_asset.stat().st_size <= 0:
                raise AssertionError(f"{runtime_filename} runtime source path is empty: {source_asset.relative_to(ROOT)}")
            source_hash = file_sha256(source_asset)
            if not SHA256_RE.match(source_hash):
                raise AssertionError(f"{runtime_filename} source hash is malformed for {source_asset.relative_to(ROOT)}")


def validate_project_archive_receipts(source_rows: dict[str, dict[str, str]]) -> None:
    for runtime_filename, source_row in sorted(source_rows.items()):
        if PROJECT_ARCHIVE_STATUS not in source_row["license"]:
            continue
        receipt = source_row.get("project_archive_receipt")
        if not isinstance(receipt, dict):
            raise AssertionError(f"{runtime_filename} project archive receipt is missing in audio_manifest.json")
        missing = [field for field in PROJECT_ARCHIVE_RECEIPT_REQUIRED_FIELDS if not receipt.get(field)]
        if missing:
            raise AssertionError(f"{runtime_filename} project archive receipt in audio_manifest.json is incomplete: missing={missing}")
        if receipt["runtime_asset"] != runtime_filename:
            raise AssertionError(f"{runtime_filename} receipt points at a different runtime asset: {receipt['runtime_asset']}")
        if PROJECT_ARCHIVE_STATUS not in receipt["license"]:
            raise AssertionError(f"{runtime_filename} receipt license must keep '{PROJECT_ARCHIVE_STATUS}'")
        for field in ("source_sha256", "runtime_sha256"):
            if not SHA256_RE.match(receipt[field]):
                raise AssertionError(f"{runtime_filename} receipt {field} must be a lowercase SHA-256 digest")
        if not DATE_RE.match(receipt["captured_on"]):
            raise AssertionError(f"{runtime_filename} receipt captured_on must use YYYY-MM-DD: {receipt['captured_on']}")
        if receipt["source_asset"].startswith("/") or ".." in receipt["source_asset"].split("/"):
            raise AssertionError(f"{runtime_filename} receipt source_asset must stay repo-relative")
        source_asset = ROOT / receipt["source_asset"]
        runtime_asset = RUNTIME_DIR / runtime_filename
        if not source_asset.exists():
            raise AssertionError(f"{runtime_filename} receipt source asset does not exist: {receipt['source_asset']}")
        if receipt["source_sha256"] != file_sha256(source_asset):
            raise AssertionError(f"{runtime_filename} receipt source hash is stale")
        if receipt["runtime_sha256"] != file_sha256(runtime_asset):
            raise AssertionError(f"{runtime_filename} receipt runtime hash is stale")
        if receipt["source_asset"] not in source_row["origin"]:
            raise AssertionError(f"{runtime_filename} receipt source does not match audio_manifest.json origin")
        if len(receipt["transform"]) < 12:
            raise AssertionError(f"{runtime_filename} receipt transform must describe how the runtime asset was produced")


def stereo_sound_array(filename: str) -> np.ndarray:
    cached = SOUND_ARRAY_CACHE.get(filename)
    if cached is not None:
        return cached
    data = pygame.sndarray.array(pygame.mixer.Sound(RUNTIME_DIR / filename)).astype(np.float32) / 32768.0
    if data.ndim == 1:
        data = np.column_stack((data, data))
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    SOUND_ARRAY_CACHE[filename] = data[:, :2]
    return SOUND_ARRAY_CACHE[filename]


def mono_sound_array(filename: str, max_seconds: float = 2.0) -> np.ndarray:
    frequency = int(pygame.mixer.get_init()[0])
    data = stereo_sound_array(filename)
    limit = max(1, int(frequency * max_seconds))
    mono = np.mean(data[:limit], axis=1)
    return mono.astype(np.float32, copy=False)


def audible_fingerprint(filename: str) -> np.ndarray:
    frequency = int(pygame.mixer.get_init()[0])
    mono = mono_sound_array(filename)
    rms = float(np.sqrt(np.mean(np.square(mono))))
    peak = float(np.max(np.abs(mono)))
    crest = peak / max(rms, 1e-9)
    zero_crossing_rate = float(np.mean(np.abs(np.diff(np.signbit(mono)))))
    fft_window = mono[: min(mono.size, frequency)]
    windowed = fft_window * np.hanning(fft_window.size)
    magnitude = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(windowed.size, 1.0 / frequency)
    magnitude_sum = float(np.sum(magnitude))
    if magnitude_sum <= 0.0:
        centroid = 0.0
        rolloff = 0.0
    else:
        centroid = float(np.sum(frequencies * magnitude) / magnitude_sum) / (frequency / 2.0)
        rolloff_index = int(np.searchsorted(np.cumsum(magnitude), magnitude_sum * 0.85))
        rolloff = float(frequencies[min(rolloff_index, frequencies.size - 1)]) / (frequency / 2.0)
    return np.array(
        [
            np.log10(max(rms, 1e-8)),
            np.log10(max(peak, 1e-8)),
            np.log10(max(crest, 1e-8)),
            zero_crossing_rate,
            centroid,
            rolloff,
        ],
        dtype=np.float32,
    )


def audible_correlation(left_filename: str, right_filename: str) -> float:
    left = mono_sound_array(left_filename, max_seconds=1.0)
    right = mono_sound_array(right_filename, max_seconds=1.0)
    sample_count = min(left.size, right.size)
    if sample_count <= 0:
        return 1.0
    left = left[:sample_count] - float(np.mean(left[:sample_count]))
    right = right[:sample_count] - float(np.mean(right[:sample_count]))
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-9))


def validate_sound_bag_audible_fingerprints() -> None:
    bags = {
        "kick": KICK_SOUND_BAG,
        "whoosh": WHOOSH_SOUND_BAG,
        "net": NET_SOUND_BAG,
        "goal_roar": GOAL_ROAR_SOUND_BAG,
        "goal_explosion": GOAL_EXPLOSION_SOUND_BAG,
        "crowd_reaction": CROWD_REACTION_SOUND_BAG,
        "near_miss_reaction": NEAR_MISS_REACTION_SOUND_BAG,
        "cup_reveal_crowd": CUP_REVEAL_CROWD_SOUND_BAG,
    }
    for bag_name, filenames in bags.items():
        hashes = {filename: file_sha256(RUNTIME_DIR / filename) for filename in filenames}
        if len(set(hashes.values())) != len(hashes):
            raise AssertionError(f"{bag_name} sound-bag contains duplicate runtime audio hashes: {filenames}")
        fingerprints = {filename: audible_fingerprint(filename) for filename in filenames}
        for index, left in enumerate(filenames):
            for right in filenames[index + 1 :]:
                distance = float(np.linalg.norm(fingerprints[left] - fingerprints[right]))
                correlation = abs(audible_correlation(left, right))
                if distance < SOUND_BAG_FINGERPRINT_DISTANCE_MIN and correlation > SOUND_BAG_CORRELATION_MAX:
                    raise AssertionError(
                        f"{bag_name} sound-bag has near-duplicate audible takes: {left} vs {right} "
                        f"fingerprint_distance={distance:.3f} correlation={correlation:.3f}"
                    )


def pan_gains(volume: float, pan: float) -> np.ndarray:
    pan = max(-1.0, min(1.0, pan))
    return np.array([volume * (1.0 - max(0.0, pan)), volume * (1.0 + min(0.0, pan))], dtype=np.float32)


def runtime_pan_gains(filename: str, channel_volume: float, pan: float) -> np.ndarray:
    return pan_gains(AUDIO_SOUND_VOLUMES[filename] * channel_volume, pan)


def loop_window(filename: str, length: int) -> np.ndarray:
    data = stereo_sound_array(filename)
    if length <= 0:
        return data[:0]
    start = int(max(0, data.shape[0] - length) * GOAL_MIX_LOOP_START_RATIO)
    if start + length <= data.shape[0]:
        return data[start : start + length]
    indexes = (np.arange(length) + start) % data.shape[0]
    return data[indexes]


def active_goal_loop_beds() -> tuple[tuple[str, float], ...]:
    intensity = GOAL_MIX_CROWD_INTENSITY
    boost = GOAL_MIX_BOOST
    return (
        ("stadium_base_loop.mp3", 0.20 + 0.08 * intensity + 0.05 * boost),
        ("stadium_air_loop.wav", 0.055 + 0.045 * intensity + 0.025 * boost),
        ("crowd_light_loop.mp3", 0.045 + 0.045 * intensity + 0.03 * boost),
        ("crowd_tension_loop.mp3", 0.015 + 0.22 * intensity + 0.09),
        ("crowd_chant_loop.mp3", 0.02 + 0.11 * intensity + 0.28 * boost),
    )


def render_goal_mix(kick_filename: str, whoosh_filename: str, net_filename: str, roar_filename: str, explosion_filename: str) -> np.ndarray:
    frequency = int(pygame.mixer.get_init()[0])
    window_seconds = GOAL_EVENT_WINDOW_MINUTES / 90.0 * SIMULATION_SECONDS
    base_threshold = SHOT_KICK_AUDIO_AT
    pan = 0.34

    def offset(threshold: float) -> float:
        return (threshold - base_threshold) * window_seconds

    cues = (
        (kick_filename, offset(SHOT_KICK_AUDIO_AT), MATCH_CUE_MIX["kick"], 1.0),
        (whoosh_filename, offset(SHOT_WHOOSH_AUDIO_AT), MATCH_CUE_MIX["whoosh"], 1.0),
        (net_filename, offset(SHOT_NET_AUDIO_AT), MATCH_CUE_MIX["net"], 1.0),
        ("bass_hit_01.wav", offset(SHOT_BASS_AUDIO_AT), MATCH_CUE_MIX["bass"], MATCH_CUE_MIX["bass"].pan_scale),
        (roar_filename, offset(SHOT_CHEER_AUDIO_AT), MATCH_CUE_MIX["goal_roar"], MATCH_CUE_MIX["goal_roar"].pan_scale),
        (explosion_filename, offset(SHOT_CHEER_AUDIO_AT), MATCH_CUE_MIX["goal_explosion"], MATCH_CUE_MIX["goal_explosion"].pan_scale),
        (GOAL_CC0_TAIL_FILENAME, offset(SHOT_CHEER_AUDIO_AT), MATCH_CUE_MIX["goal_cc0_tail"], MATCH_CUE_MIX["goal_cc0_tail"].pan_scale),
        (GOAL_ATTACK_SWELL_FILENAME, offset(SHOT_CHEER_AUDIO_AT), MATCH_CUE_MIX["goal_attack_swell"], MATCH_CUE_MIX["goal_attack_swell"].pan_scale),
        ("stadium_reverb_tail.mp3", offset(SHOT_REVERB_AT), MATCH_CUE_MIX["reverb"], MATCH_CUE_MIX["reverb"].pan_scale),
    )
    mix_length = max(int((start + cue.maxtime) * frequency) + 1 for _filename, start, cue, _pan_scale in cues)
    mix = np.zeros((mix_length, 2), dtype=np.float32)
    for filename, start_seconds, cue, pan_scale in cues:
        samples = stereo_sound_array(filename)[: int(cue.maxtime * frequency)]
        start = int(start_seconds * frequency)
        mix[start : start + samples.shape[0]] += samples * runtime_pan_gains(filename, cue.volume, pan * pan_scale)
    for filename, channel_volume in active_goal_loop_beds():
        mix += loop_window(filename, mix.shape[0]) * runtime_pan_gains(filename, channel_volume, 0.0)
    return mix


def add_cue(mix: np.ndarray, filename: str, start_seconds: float, channel_volume: float, maxtime_seconds: float, pan: float = 0.0) -> None:
    frequency = int(pygame.mixer.get_init()[0])
    start = max(0, int(start_seconds * frequency))
    samples = stereo_sound_array(filename)[: int(maxtime_seconds * frequency)]
    end = min(mix.shape[0], start + samples.shape[0])
    if end <= start:
        return
    mix[start:end] += samples[: end - start] * runtime_pan_gains(filename, channel_volume, pan)


def render_short_mix(cues: tuple[tuple[str, float, float, float, float], ...], length_seconds: float = 3.2) -> np.ndarray:
    frequency = int(pygame.mixer.get_init()[0])
    mix = np.zeros((int(length_seconds * frequency), 2), dtype=np.float32)
    bed_level = {
        "stadium_base_loop.mp3": 0.24,
        "stadium_air_loop.wav": 0.08,
        "crowd_light_loop.mp3": 0.08,
        "crowd_tension_loop.mp3": 0.10,
    }
    for filename, channel_volume in bed_level.items():
        mix += loop_window(filename, mix.shape[0]) * runtime_pan_gains(filename, channel_volume, 0.0)
    for filename, start_seconds, channel_volume, maxtime_seconds, pan in cues:
        add_cue(mix, filename, start_seconds, channel_volume, maxtime_seconds, pan)
    return mix


def assert_short_mix_headroom(label: str, mix: np.ndarray) -> None:
    peak = float(np.max(np.abs(mix)))
    overloaded = float(np.mean(np.abs(mix) > 1.0))
    near_clip = float(np.mean(np.abs(mix) >= MIX_NEAR_CLIP_AMPLITUDE))
    rms = float(np.sqrt(np.mean(np.square(mix))))
    rms_dbfs = 20.0 * np.log10(max(rms, 1e-9))
    if overloaded > 0.0 or near_clip > SHORT_MIX_NEAR_CLIP_RATIO_LIMIT or peak > SHORT_MIX_PEAK_LIMIT or rms_dbfs > SHORT_MIX_RMS_DBFS_MAX:
        raise AssertionError(
            f"short audio mix lacks headroom for {label}: "
            f"peak={peak:.3f} overloaded={overloaded:.4%} near_clip={near_clip:.4%} rms_dbfs={rms_dbfs:.2f}"
        )


def validate_short_event_mixes_headroom() -> None:
    scenarios: list[tuple[str, tuple[tuple[str, float, float, float, float], ...]]] = []
    for net_filename, reaction_filename in product(NET_SOUND_BAG, CROWD_REACTION_SOUND_BAG):
        scenarios.append(
            (
                f"save:{net_filename}+{reaction_filename}",
                (
                    (net_filename, 0.00, 0.20, 0.62, 0.22),
                    (reaction_filename, 0.05, 0.12, 1.70, 0.06),
                ),
            )
        )
    for reaction_filename in NEAR_MISS_REACTION_SOUND_BAG:
        scenarios.append(
            (
                f"near_miss:{reaction_filename}",
                (
                    (reaction_filename, 0.00, 0.10, 1.80, -0.06),
                    ("stadium_reverb_tail.mp3", 0.10, 0.08, 1.45, -0.05),
                ),
            )
        )
    for reaction_filename in CUP_REVEAL_CROWD_SOUND_BAG:
        scenarios.append(
            (
                f"cup_reveal:{reaction_filename}",
                (
                    ("cup_reveal_stinger.wav", 0.00, 0.30, 2.60, 0.0),
                    (reaction_filename, 0.12, 0.13, 2.20, 0.0),
                ),
            )
        )
    scenarios.append(("final_whistle", (("whistle_final_01.wav", 0.00, 0.24, 1.40, 0.0),)))
    scenarios.append(
        (
            "headroom_stress",
            (
                ("whistle_final_01.wav", 0.00, 0.24, 1.40, 0.0),
                ("cup_reveal_stinger.wav", 0.04, 0.30, 2.60, 0.0),
                ("crowd_attack_short.wav", 0.12, 0.13, 2.20, 0.0),
            ),
        )
    )
    for label, cues in scenarios:
        assert_short_mix_headroom(label, render_short_mix(cues))


def validate_goal_mix_headroom() -> None:
    rms_levels: list[float] = []
    for kick_filename, whoosh_filename, net_filename, roar_filename, explosion_filename in product(
        KICK_SOUND_BAG,
        WHOOSH_SOUND_BAG,
        NET_SOUND_BAG,
        GOAL_ROAR_SOUND_BAG,
        GOAL_EXPLOSION_SOUND_BAG,
    ):
        mix = render_goal_mix(kick_filename, whoosh_filename, net_filename, roar_filename, explosion_filename)
        peak = float(np.max(np.abs(mix)))
        overloaded = float(np.mean(np.abs(mix) > 1.0))
        near_clip = float(np.mean(np.abs(mix) >= MIX_NEAR_CLIP_AMPLITUDE))
        if overloaded > 0.0 or near_clip > MIX_NEAR_CLIP_RATIO_LIMIT or peak > MIX_PEAK_LIMIT:
            raise AssertionError(
                f"goal mix clips or lacks headroom with kick={kick_filename} whoosh={whoosh_filename} "
                f"net={net_filename} roar={roar_filename} explosion={explosion_filename}: "
                f"peak={peak:.3f} overloaded={overloaded:.4%} near_clip={near_clip:.4%}"
            )
        rms = float(np.sqrt(np.mean(np.square(mix))))
        rms_dbfs = 20.0 * np.log10(max(rms, 1e-9))
        rms_levels.append(rms_dbfs)
        if not MIX_RMS_DBFS_MIN <= rms_dbfs <= MIX_RMS_DBFS_MAX:
            raise AssertionError(
                f"goal mix loudness outside QA range with kick={kick_filename} whoosh={whoosh_filename} "
                f"net={net_filename} roar={roar_filename} explosion={explosion_filename}: rms_dbfs={rms_dbfs:.2f}"
            )
    rms_spread = max(rms_levels) - min(rms_levels)
    if rms_spread > MIX_RMS_DBFS_SPREAD_LIMIT:
        raise AssertionError(f"goal mix sound-bag loudness spread is too wide: {rms_spread:.2f} dB")


def validate_goal_mix_matrix_contract() -> None:
    expected = len(KICK_SOUND_BAG) * len(WHOOSH_SOUND_BAG) * len(NET_SOUND_BAG) * len(GOAL_ROAR_SOUND_BAG) * len(GOAL_EXPLOSION_SOUND_BAG)
    if expected < 144:
        raise AssertionError(f"goal mix matrix lost sound-bag coverage: {expected} combinations")


def validate_long_loop_seams() -> None:
    frequency = int(pygame.mixer.get_init()[0])
    window = max(1, int(LOOP_SEAM_WINDOW_SECONDS * frequency))
    for filename in LOOP_SEAM_FILES:
        data = stereo_sound_array(filename)
        if data.shape[0] <= window * 2:
            raise AssertionError(f"{filename} is too short for loop seam QA")
        endpoint_jump = float(np.max(np.abs(data[0] - data[-1])))
        seam = np.vstack((data[-window:], data[:window]))
        seam_step_peak = float(np.max(np.abs(np.diff(seam, axis=0))))
        body_step = np.max(np.abs(np.diff(data, axis=0)), axis=1)
        body_step_p999 = float(np.percentile(body_step, 99.9))
        seam_step_limit = max(LOOP_SEAM_DIFF_ABSOLUTE_LIMIT, body_step_p999 * LOOP_SEAM_DIFF_RATIO_LIMIT)
        if endpoint_jump > LOOP_SEAM_ENDPOINT_JUMP_LIMIT or seam_step_peak > seam_step_limit:
            raise AssertionError(
                f"{filename} has a likely loop seam click: endpoint_jump={endpoint_jump:.4f} "
                f"limit={LOOP_SEAM_ENDPOINT_JUMP_LIMIT:.4f} seam_step_peak={seam_step_peak:.4f} "
                f"limit={seam_step_limit:.4f}"
            )


def validate_runtime_assets() -> None:
    source_rows = parse_runtime_source_rows()
    actual = {path.name for path in RUNTIME_DIR.iterdir() if path.is_file()}
    expected = set(AUDIO_RUNTIME_FILES)
    if actual != expected:
        raise AssertionError(f"runtime_assets mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    if set(AUDIO_SOUND_VOLUMES) != expected:
        raise AssertionError(
            f"runtime sound volume map mismatch: missing={sorted(expected - set(AUDIO_SOUND_VOLUMES))} "
            f"extra={sorted(set(AUDIO_SOUND_VOLUMES) - expected)}"
        )
    validate_runtime_source_docs(source_rows)
    validate_runtime_license_receipts(source_rows)
    validate_runtime_promoted_source_integrity(source_rows)
    validate_project_archive_receipts(source_rows)
    if len({asset.role for asset in AUDIO_ASSETS}) < 10:
        raise AssertionError("audio manifest lost functional role coverage")
    if AUDIO_BUSES != REQUIRED_AUDIO_BUSES:
        raise AssertionError(f"audio buses drifted from the required contract: {sorted(AUDIO_BUSES)}")
    if "crowd_near_miss_01.wav" in CROWD_REACTION_SOUND_BAG:
        raise AssertionError("near-miss crowd asset leaked into the ambient crowd reaction bag")
    if "crowd_near_miss_01.wav" not in NEAR_MISS_REACTION_SOUND_BAG:
        raise AssertionError("near-miss event bag lost its dedicated near-miss crowd asset")
    for filename, (low, high) in AUDIO_DURATION_LIMITS.items():
        sound = pygame.mixer.Sound(RUNTIME_DIR / filename)
        duration = sound.get_length()
        if not low <= duration <= high:
            raise AssertionError(f"{filename} duration {duration:.2f}s outside {low:.2f}-{high:.2f}s")
        if filename in AUDIO_TRANSIENT_START_LIMITS:
            start = transient_start(sound)
            limit = AUDIO_TRANSIENT_START_LIMITS[filename]
            if start > limit:
                raise AssertionError(f"{filename} transient starts too late: {start:.3f}s > {limit:.3f}s")
        clipped = clipping_ratio(sound)
        if clipped > 0.002:
            raise AssertionError(f"{filename} has too many near-clipped samples: {clipped:.4%}")


def validate_no_candidate_runtime_imports() -> None:
    forbidden = ("assets/sounds/candidates", "sounds/candidates", "candidates/")
    for source in (SRC / "arena_ai").rglob("*.py"):
        text = source.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in text:
                raise AssertionError(f"runtime source imports candidate audio directly: {source} token={token}")


def validate_draw_purity() -> None:
    in_draw = False
    method_name = ""
    for lineno, line in enumerate(MAIN_PY.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("def "):
            method_name = stripped.split("(", 1)[0].replace("def ", "")
            in_draw = method_name.startswith("draw")
        elif stripped.startswith("class "):
            in_draw = False
        if in_draw and any(token in line for token in ("self.sound.", "pygame.mixer", "AudioEngine(")):
            raise AssertionError(f"draw method {method_name} touches audio at {MAIN_PY}:{lineno}")


def validate_audio_update_ownership() -> None:
    allowed = {"emit_match_audio_events", "flush_queued_match_audio", "update", "update_tournament_audio"}
    method_name = ""
    for lineno, line in enumerate(MAIN_PY.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("def "):
            method_name = stripped.split("(", 1)[0].replace("def ", "")
        if "self.sound.play(" in line and method_name not in allowed:
            raise AssertionError(f"audio event outside update-owned timeline: {method_name} at {MAIN_PY}:{lineno}")


def validate_runtime_mixer_pre_init_order() -> None:
    text = MAIN_PY.read_text(encoding="utf-8")
    pre_init_index = text.find("        pre_init_mixer()")
    init_index = text.find("        pygame.init()")
    if pre_init_index < 0 or init_index < 0 or pre_init_index > init_index:
        raise AssertionError("runtime must call pre_init_mixer() before pygame.init()")


def validate_engine_contract() -> None:
    app = App(seed=2026)
    mixer = pygame.mixer.get_init()
    if mixer is None or mixer[0] < 44100:
        raise AssertionError(f"mixer must run at low-latency 44.1kHz or higher: {mixer}")
    if pygame.mixer.get_num_channels() < 20:
        raise AssertionError("AudioEngine must reserve at least 20 channels for buses/layers")
    if set(app.sound.buses) != REQUIRED_AUDIO_BUSES:
        raise AssertionError(f"unexpected audio buses: {sorted(app.sound.buses)}")
    if app.sound.channels["base"].get_sound() is not app.sound.stadium_base:
        raise AssertionError("base ambience loop is not running on the base channel")
    if len(app.sound.kick_bag) < 4 or len(app.sound.whoosh_bag) < 2 or len(app.sound.net_bag) < 2:
        raise AssertionError("ball sound-bags lost take diversity")
    if len(app.sound.goal_roars) < 3 or len(app.sound.goal_explosions) < 3 or len(app.sound.crowd_reactions) < 3 or len(app.sound.near_miss_reactions) < 3:
        raise AssertionError("crowd/goal sound-bags lost take diversity")
    chosen_kicks = [app.sound.choose_bag("qa_kick", app.sound.kick_bag) for _index in range(8)]
    if any(chosen_kicks[index] is chosen_kicks[index - 1] for index in range(1, len(chosen_kicks))):
        raise AssertionError("sound-bag should avoid immediate repeated kick takes")
    chosen_goal_explosions = [app.sound.choose_bag("qa_goal_explosion", app.sound.goal_explosions) for _index in range(8)]
    if any(chosen_goal_explosions[index] is chosen_goal_explosions[index - 1] for index in range(1, len(chosen_goal_explosions))):
        raise AssertionError("goal explosion sound-bag should avoid immediate repeated takes")
    chosen_goal_roars = [app.sound.choose_bag("qa_goal_roar", app.sound.goal_roars) for _index in range(8)]
    if any(chosen_goal_roars[index] is chosen_goal_roars[index - 1] for index in range(1, len(chosen_goal_roars))):
        raise AssertionError("goal roar sound-bag should avoid immediate repeated takes")
    chosen_crowd_reactions = [app.sound.choose_bag("qa_crowd_reaction", app.sound.crowd_reactions) for _index in range(8)]
    if any(chosen_crowd_reactions[index] is chosen_crowd_reactions[index - 1] for index in range(1, len(chosen_crowd_reactions))):
        raise AssertionError("crowd reaction sound-bag should avoid immediate repeated takes")
    chosen_cup_reactions = [app.sound.choose_bag("qa_cup_reveal_crowd", app.sound.cup_reveal_reactions) for _index in range(8)]
    if any(chosen_cup_reactions[index] is chosen_cup_reactions[index - 1] for index in range(1, len(chosen_cup_reactions))):
        raise AssertionError("cup reveal crowd sound-bag should avoid immediate repeated takes")
    app.sound.play("kick")
    if app.sound.suppress_reactions_until_ms <= pygame.time.get_ticks():
        raise AssertionError("kick must suppress random crowd reactions briefly")
    app.sound.next_reaction_ms = 0
    app.sound.channels["react"].stop()
    app.sound.update_crowd(0.95, True, 0.25, allow_reactions=True)
    if app.sound.channels["react"].get_busy():
        raise AssertionError("random crowd reaction played on top of kick cooldown")
    app.sound.play("save")
    if not app.sound.channels["react"].get_busy():
        raise AssertionError("save event should trigger a controlled crowd reaction")
    app.sound.stop_one_shots(fade_ms=0)
    app.sound.play("near_miss")
    if not app.sound.channels["react"].get_busy():
        raise AssertionError("near-miss event should trigger a controlled crowd reaction")
    app.sound.stop_one_shots(fade_ms=0)
    app.sound.play("final_whistle")
    now = pygame.time.get_ticks()
    if app.sound.suppress_reactions_until_ms <= now or app.sound.next_reaction_ms <= now:
        raise AssertionError("final whistle must suppress late random reactions")
    if not app.sound.channels["whistle"].get_busy():
        raise AssertionError("final whistle should occupy the whistle channel")
    app.sound.stop_one_shots(fade_ms=0)
    app.sound.play("cheer")
    if not app.sound.channels["roar"].get_busy():
        raise AssertionError("goal roar should occupy its dedicated one-shot channel")
    app.sound.stop_one_shots(fade_ms=0)
    busy_one_shots = [name for name in ONE_SHOT_CHANNELS if app.sound.channels[name].get_busy()]
    if busy_one_shots:
        raise AssertionError(f"one-shot reset left channels busy: {busy_one_shots}")
    if not app.sound.channels["base"].get_busy():
        raise AssertionError("one-shot reset must not stop ambience loops")


def validate_audio_cue_policy_contract() -> None:
    required = set(GOAL_AUDIO_SEQUENCE) | {"save", "near_miss", "whistle", "final_whistle", "cup_reveal"}
    missing = required - set(AUDIO_CUE_POLICIES)
    if missing:
        raise AssertionError(f"audio cue policy missing runtime events: {sorted(missing)}")
    if AUDIO_CUE_POLICIES["cheer"].goal_boost_ms < 5000 or AUDIO_CUE_POLICIES["cheer"].impact_focus_ms < 350:
        raise AssertionError("goal cheer policy must create a real post-goal boost and impact focus")
    ducked_impact = [name for name in GOAL_IMPACT_AUDIO_EVENTS if AUDIO_CUE_POLICIES[name].duck_seconds > 0.0]
    if ducked_impact:
        raise AssertionError(f"goal impact cues should not hard-duck the stadium bed: {ducked_impact}")
    source = MAIN_PY.read_text(encoding="utf-8")
    if "elif name == \"bass\"" in source or "elif name == \"cheer\"" in source:
        raise AssertionError("App-level match cue policy drifted back into main.py")


def validate_goal_impact_layer_contract() -> None:
    app = App(seed=2026)
    app.sound.play("cheer", pan=0.34)
    app.sound.update_crowd(0.92, True, 0.30, allow_reactions=False)
    volumes = app.sound.layer_volumes
    if volumes["base"] < 0.25 or volumes["air"] < 0.09:
        raise AssertionError(f"goal impact should preserve the stadium bed, got {volumes}")
    if volumes["chant"] < 0.34:
        raise AssertionError(f"post-goal chant boost is too weak, got {volumes}")
    app.sound.duck_commentary(0.30)
    app.sound.update_crowd(0.92, True, 0.30, allow_reactions=False)
    volumes = app.sound.layer_volumes
    if volumes["base"] < 0.20 or volumes["air"] < 0.08:
        raise AssertionError(f"commentary duck over-muted the stadium bed during goal impact: {volumes}")


def validate_initial_whistle_suppresses_reactions() -> None:
    app = App(seed=2026)
    app.sound.play("whistle")
    now = pygame.time.get_ticks()
    if app.sound.suppress_reactions_until_ms <= now or app.sound.next_reaction_ms <= now:
        raise AssertionError("initial whistle must arm a reaction cooldown before crowd updates")
    app.sound.channels["react"].stop()
    app.sound.update_crowd(0.98, False, 0.20, allow_reactions=True)
    if app.sound.channels["react"].get_busy():
        raise AssertionError("random crowd reaction leaked during initial whistle cooldown")


def validate_match_events_suppress_before_crowd() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = pred
    app.match_intro_audio_pending = False
    goal_minute = app.goal_schedule(pred)[0][0]
    app.t = (goal_minute - 5.0 + (SHOT_KICK_AUDIO_AT + 0.002) * 5.0) / 90.0 * 45.0
    app.sound.next_reaction_ms = 0
    app.sound.suppress_reactions_until_ms = 0
    suppressions_seen: list[bool] = []
    original_update_crowd = app.sound.update_crowd

    def spy_update_crowd(*args: object, **kwargs: object) -> None:
        suppressions_seen.append(app.sound.suppress_reactions_until_ms > pygame.time.get_ticks())
        original_update_crowd(*args, **kwargs)

    app.sound.update_crowd = spy_update_crowd  # type: ignore[method-assign]
    app.update_soundscape(1 / 60, previous_minute=goal_minute - 5.0)
    if not suppressions_seen or not suppressions_seen[0]:
        raise AssertionError("match kick/net/goal events must suppress reactions before update_crowd")


def validate_match_event_order() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = pred
    goal_minute = app.goal_schedule(pred)[0][0]
    played: list[str] = []
    pans: list[float] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        played.append(name)
        if name in GOAL_AUDIO_SEQUENCE:
            pans.append(float(_kwargs.get("pan", 0.0)))

    app.sound.play = spy  # type: ignore[method-assign]
    for progress in (
        SHOT_KICK_AUDIO_AT,
        SHOT_WHOOSH_AT + 0.01,
        SHOT_NET_VISUAL_CONTACT_AT,
        SHOT_REVERB_AT + 0.01,
        SHOT_REVERB_AT + 0.01,
        SHOT_REVERB_AT + 0.01,
    ):
        app.t = (goal_minute - 5.0 + progress * 5.0) / 90.0 * 45.0
        app.update_soundscape(1 / 60)
        app.flush_queued_match_audio()
    expected = list(GOAL_AUDIO_SEQUENCE)
    if played != expected:
        raise AssertionError(f"match audio event order changed: {played}")
    if not pans or max(abs(pan) for pan in pans) < 0.25:
        raise AssertionError(f"goal audio should carry side-based pan, got {pans}")


def validate_match_audio_visual_sync_thresholds() -> None:
    kick_lead = SHOT_KICK_AT - SHOT_KICK_AUDIO_AT
    if not 0.004 <= kick_lead <= 0.014:
        raise AssertionError(
            f"kick audio must lead the visual foot contact by one transient-safe slice: "
            f"lead={kick_lead:.3f}, audio={SHOT_KICK_AUDIO_AT:.3f}, visual={SHOT_KICK_AT:.3f}"
        )
    impact_thresholds = {
        "net": SHOT_NET_AUDIO_AT,
        "bass": SHOT_BASS_AUDIO_AT,
        "cheer": SHOT_CHEER_AUDIO_AT,
    }
    drifted = {
        name: threshold
        for name, threshold in impact_thresholds.items()
        if abs(threshold - SHOT_NET_AUDIO_AT) > 1e-9
    }
    if drifted:
        raise AssertionError(f"goal impact cues must stay bundled in one cinematic frame: {drifted}")
    late_or_early = {
        name: SHOT_NET_VISUAL_CONTACT_AT - threshold
        for name, threshold in impact_thresholds.items()
        if not 0.004 <= SHOT_NET_VISUAL_CONTACT_AT - threshold <= 0.014
    }
    if late_or_early:
        raise AssertionError(
            f"goal impact audio must be pre-rolled for transient sync with visual net contact={SHOT_NET_VISUAL_CONTACT_AT:.3f}: {late_or_early}"
        )

    app = App(seed=2026)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = pred
    goal_minute, side = app.goal_schedule(pred)[0]
    played: list[tuple[str, float]] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        minute = app.match_minute_float()
        progress = (minute - (goal_minute - GOAL_EVENT_WINDOW_MINUTES)) / GOAL_EVENT_WINDOW_MINUTES
        played.append((name, progress))

    app.sound.play = spy  # type: ignore[method-assign]
    samples = (
        SHOT_KICK_AUDIO_AT - 0.002,
        SHOT_KICK_AUDIO_AT,
        SHOT_NET_AUDIO_AT - 0.002,
        SHOT_NET_AUDIO_AT,
        SHOT_NET_VISUAL_CONTACT_AT,
        SHOT_REVERB_AT + 0.002,
    )
    for progress in samples:
        before = len(played)
        app.t = (goal_minute - GOAL_EVENT_WINDOW_MINUTES + progress * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS
        app.update_soundscape(1 / 60)
        if len(played) != before:
            raise AssertionError("match audio played before the post-frame queue was flushed")
        app.flush_queued_match_audio()

    kick_times = [progress for name, progress in played if name == "kick"]
    if not kick_times or abs(kick_times[0] - SHOT_KICK_AUDIO_AT) > 1e-9:
        raise AssertionError(f"kick audio did not land on the visual contact follow-through: {played}")
    impact_times = {name: progress for name, progress in played if name in GOAL_IMPACT_AUDIO_EVENTS}
    if set(impact_times) != GOAL_IMPACT_AUDIO_EVENTS:
        raise AssertionError(f"goal impact bundle missing synchronized cues: {played}")
    for name, progress in impact_times.items():
        expected = impact_thresholds[name]
        if abs(progress - expected) > 1e-9:
            raise AssertionError(f"{name} audio drifted from its pre-roll threshold={expected:.3f}: progress={progress:.3f}, events={played}")


def validate_whoosh_reverb_chance_timing_contract() -> None:
    whoosh_lead = SHOT_WHOOSH_AT - SHOT_WHOOSH_AUDIO_AT
    if not 0.006 <= whoosh_lead <= 0.010:
        raise AssertionError(
            f"whoosh audio must pre-roll the visual release by 6-10ms: "
            f"lead={whoosh_lead:.3f}, audio={SHOT_WHOOSH_AUDIO_AT:.3f}, visual={SHOT_WHOOSH_AT:.3f}"
        )
    reverb_lag = SHOT_REVERB_AT - SHOT_NET_VISUAL_CONTACT_AT
    if not 0.006 <= reverb_lag <= 0.020:
        raise AssertionError(
            f"reverb tail must trail the visual net impact without drifting late: "
            f"lag={reverb_lag:.3f}, reverb={SHOT_REVERB_AT:.3f}, visual={SHOT_NET_VISUAL_CONTACT_AT:.3f}"
        )
    chance_lead = CHANCE_CONTACT_VISUAL_AT - CHANCE_CONTACT_AUDIO_AT
    if not 0.006 <= chance_lead <= 0.010:
        raise AssertionError(
            f"save/near-miss audio must pre-roll the chance visual contact by 6-10ms: "
            f"lead={chance_lead:.3f}, audio={CHANCE_CONTACT_AUDIO_AT:.3f}, visual={CHANCE_CONTACT_VISUAL_AT:.3f}"
        )
    if CHANCE_CONTACT_VISUAL_AT != SHOT_NET_AT:
        raise AssertionError(
            f"chance visual contact must stay explicit at the no-goal payoff frame: "
            f"chance={CHANCE_CONTACT_VISUAL_AT:.3f}, shot_net={SHOT_NET_AT:.3f}"
        )
    if CHANCE_CONTACT_AUDIO_AT == SHOT_NET_AUDIO_AT or CHANCE_CONTACT_VISUAL_AT == SHOT_NET_VISUAL_CONTACT_AT:
        raise AssertionError("chance contact markers drifted back to the goal net-impact markers")


def validate_match_audio_quantized_frame_sync() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = pred
    goal_minute, _side = app.goal_schedule(pred)[0]
    played: list[str] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        played.append(name)

    app.sound.play = spy  # type: ignore[method-assign]
    previous_progress = SHOT_KICK_AUDIO_AT - 0.003
    current_progress = SHOT_KICK_AT + 0.010
    previous_minute = goal_minute - GOAL_EVENT_WINDOW_MINUTES + previous_progress * GOAL_EVENT_WINDOW_MINUTES
    app.t = (goal_minute - GOAL_EVENT_WINDOW_MINUTES + current_progress * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS
    app.update_soundscape(1 / 20, previous_minute=previous_minute)
    if played:
        raise AssertionError(f"frame-quantized kick escaped before post-frame flush: {played}")
    if [name for name, _pan in app.match_audio_frame_queue] != ["kick"]:
        raise AssertionError(f"frame-quantized kick was not queued cleanly: {app.match_audio_frame_queue}")
    app.flush_queued_match_audio()
    if played != ["kick"]:
        raise AssertionError(f"frame-quantized kick did not flush on the contact frame: {played}")


def validate_match_impact_quantized_frame_sync() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = pred
    goal_minute, side = app.goal_schedule(pred)[0]
    goal_audio_key = (pred.algorithm, goal_minute, side)
    played: list[str] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        played.append(name)

    app.sound.play = spy  # type: ignore[method-assign]
    previous_progress = SHOT_NET_AUDIO_AT - 0.003
    app.shot_progress_cursor[goal_audio_key] = previous_progress
    previous_minute = goal_minute - GOAL_EVENT_WINDOW_MINUTES + previous_progress * GOAL_EVENT_WINDOW_MINUTES
    app.t = (
        goal_minute - GOAL_EVENT_WINDOW_MINUTES + (SHOT_NET_VISUAL_CONTACT_AT + 0.004) * GOAL_EVENT_WINDOW_MINUTES
    ) / 90.0 * SIMULATION_SECONDS
    app.update_soundscape(1 / 20, previous_minute=previous_minute)
    if played:
        raise AssertionError(f"frame-quantized goal impact escaped before post-frame flush: {played}")
    volumes = app.sound.layer_volumes
    if volumes["base"] < 0.10 or volumes["air"] < 0.055:
        raise AssertionError(f"frame-quantized impact over-muted the stadium bed: {volumes}")
    if volumes["tension"] < 0.025 or volumes["chant"] < 0.020:
        raise AssertionError(f"frame-quantized impact lost crowd tension/chant layers: {volumes}")
    queued = [name for name, _pan in app.match_audio_frame_queue]
    if queued != ["net", "bass", "cheer"]:
        raise AssertionError(f"frame-quantized goal impact bundle drifted: {queued}")
    app.flush_queued_match_audio()
    if played != ["net", "bass", "cheer"]:
        raise AssertionError(f"frame-quantized goal impact did not flush as one bundle: {played}")


def validate_low_fps_goal_timeline_does_not_collapse() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = pred
    goal_minute, side = app.goal_schedule(pred)[0]
    goal_audio_key = (pred.algorithm, goal_minute, side)
    previous_progress = SHOT_NET_AT - 0.02
    app.shot_progress_cursor[goal_audio_key] = previous_progress
    played: list[str] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        played.append(name)

    app.sound.play = spy  # type: ignore[method-assign]
    previous_minute = goal_minute - GOAL_EVENT_WINDOW_MINUTES + previous_progress * GOAL_EVENT_WINDOW_MINUTES
    app.t = (goal_minute - GOAL_EVENT_WINDOW_MINUTES + (SHOT_REVERB_AT + 0.01) * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * 45.0
    app.update_soundscape(0.33, previous_minute=previous_minute)
    app.flush_queued_match_audio()
    if played != ["net", "bass", "cheer"]:
        raise AssertionError(f"low-FPS goal impact should keep the impact bundle synchronized: {played}")


def validate_match_audio_uses_post_frame_queue() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = pred
    goal_minute, _side = app.goal_schedule(pred)[0]
    played: list[str] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        played.append(name)

    app.sound.play = spy  # type: ignore[method-assign]
    app.t = (goal_minute - GOAL_EVENT_WINDOW_MINUTES + SHOT_KICK_AUDIO_AT * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS
    app.update_soundscape(1 / 60)
    if played:
        raise AssertionError(f"match audio escaped before the visual frame was presented: {played}")
    if [name for name, _pan in app.match_audio_frame_queue] != ["kick"]:
        raise AssertionError(f"match audio was not queued for post-frame playback: {app.match_audio_frame_queue}")
    app.flush_queued_match_audio()
    if played != ["kick"]:
        raise AssertionError(f"post-frame match audio queue did not flush correctly: {played}")


def validate_chance_audio_events() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = pred
    app.goal_schedule = lambda _pred: []  # type: ignore[method-assign]
    played: list[str] = []
    pans: list[float] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        played.append(name)
        pans.append(float(_kwargs.get("pan", 0.0)))

    app.sound.play = spy  # type: ignore[method-assign]

    for kind, expected, side in (("save", ["kick", "whoosh", "save"], "home"), ("wide", ["kick", "whoosh", "near_miss"], "away")):
        app.shot_events.clear()
        app.shot_progress_cursor.clear()
        played.clear()
        pans.clear()
        chance_minute = 36 if kind == "save" else 54
        app.chance_schedule = lambda _pred, chance_minute=chance_minute, side=side, kind=kind: [(chance_minute, side, kind)]  # type: ignore[method-assign]
        for progress in (SHOT_KICK_AUDIO_AT, SHOT_WHOOSH_AUDIO_AT, CHANCE_CONTACT_AUDIO_AT):
            app.t = (chance_minute - CHANCE_EVENT_WINDOW_MINUTES + progress * CHANCE_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS
            app.emit_match_audio_events(pred)
            app.flush_queued_match_audio()
        if played != expected:
            raise AssertionError(f"chance {kind} audio event order changed: {played}")
        if not pans or max(abs(pan) for pan in pans) < 0.20:
            raise AssertionError(f"chance {kind} audio should carry side-based pan, got {pans}")


def validate_save_near_miss_contact_pre_roll() -> None:
    for kind, terminal, side in (("save", "save", "home"), ("wide", "near_miss", "away")):
        app = App(seed=2026)
        app.set_simulate("match")
        pred = app.model.predict_matchup(app.home, app.away, seed=2026)
        app.match_prediction = pred
        app.goal_schedule = lambda _pred: []  # type: ignore[method-assign]
        chance_minute = 38 if kind == "save" else 52
        app.chance_schedule = lambda _pred, chance_minute=chance_minute, side=side, kind=kind: [(chance_minute, side, kind)]  # type: ignore[method-assign]
        played: list[tuple[str, float]] = []

        def spy(name: str, *_args: object, **_kwargs: object) -> None:
            minute = app.match_minute_float()
            progress = (minute - (chance_minute - CHANCE_EVENT_WINDOW_MINUTES)) / CHANCE_EVENT_WINDOW_MINUTES
            played.append((name, progress))

        app.sound.play = spy  # type: ignore[method-assign]
        for progress in (CHANCE_CONTACT_AUDIO_AT - 0.002, CHANCE_CONTACT_AUDIO_AT, CHANCE_CONTACT_VISUAL_AT):
            app.t = (chance_minute - CHANCE_EVENT_WINDOW_MINUTES + progress * CHANCE_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS
            before = len(played)
            app.emit_match_audio_events(pred)
            if len(played) != before:
                raise AssertionError(f"chance {kind} audio escaped before post-frame flush: {played}")
            app.flush_queued_match_audio()
        terminal_times = [progress for name, progress in played if name == terminal]
        if not terminal_times:
            raise AssertionError(f"chance {kind} never played terminal contact cue: {played}")
        if abs(terminal_times[0] - CHANCE_CONTACT_AUDIO_AT) > 1e-9:
            raise AssertionError(
                f"chance {kind} terminal cue drifted from pre-roll marker: "
                f"played={terminal_times[0]:.3f}, expected={CHANCE_CONTACT_AUDIO_AT:.3f}, events={played}"
            )


def validate_chance_cursor_does_not_block_final_whistle() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = pred
    app.match_intro_audio_pending = False
    app.t = SIMULATION_SECONDS
    chance_key = (pred.algorithm, 74, "home:save")
    app.shot_progress_cursor[chance_key] = CHANCE_CONTACT_AUDIO_AT - 0.002
    played: list[str] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        played.append(name)

    app.sound.play = spy  # type: ignore[method-assign]
    app.update(0.0)
    if "final_whistle" in played:
        raise AssertionError(f"chance cursor played final whistle before chance payoff: {played}")
    app.shot_progress_cursor[chance_key] = CHANCE_CONTACT_AUDIO_AT - 0.001
    app.update(0.0)
    if "final_whistle" in played:
        raise AssertionError(f"chance cursor played final whistle before audio contact threshold: {played}")
    app.shot_progress_cursor[chance_key] = CHANCE_CONTACT_AUDIO_AT
    app.update(0.0)
    if "final_whistle" not in played:
        raise AssertionError(f"chance cursor blocked the final whistle: {played}")


def validate_final_whistle_blocks_same_frame_reaction() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = replace(pred, score_home=0, score_away=0, outcome_class=1)
    app.match_intro_audio_pending = False
    app.final_whistle_played = False
    app.t = SIMULATION_SECONDS - 0.01
    app.sound.next_reaction_ms = 0
    app.sound.suppress_reactions_until_ms = 0
    if "react" in app.sound.channels:
        app.sound.channels["react"].stop()
    allow_reactions_seen: list[bool] = []
    original_update_crowd = app.sound.update_crowd

    def spy_update_crowd(intensity: float, goal_active: bool, dt: float, allow_reactions: bool = True) -> None:
        allow_reactions_seen.append(allow_reactions)
        original_update_crowd(intensity, goal_active, dt, allow_reactions=allow_reactions)

    app.sound.update_crowd = spy_update_crowd  # type: ignore[method-assign]
    app.update(0.02)
    if not app.final_whistle_played:
        raise AssertionError("final whistle did not play when the match reached full time")
    if allow_reactions_seen[-1:] != [False]:
        raise AssertionError(f"final whistle frame allowed random crowd reactions: {allow_reactions_seen}")
    if app.sound.channels.get("react") and app.sound.channels["react"].get_busy():
        raise AssertionError("random crowd reaction played in the same frame as final whistle")


def validate_cup_audio_contract() -> None:
    app = App(seed=2026)
    app.state = "tournament"
    app.mc_progress_total = 100
    played: list[str] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        played.append(name)

    app.sound.play = spy  # type: ignore[method-assign]
    for done in (CUP_PROGRESS_MARKERS[0] - 1, *CUP_PROGRESS_MARKERS):
        app.mc_progress_done = done
        app.update_tournament_audio()
    app.cup_reveal_audio_pending = True
    app.t = TOURNAMENT_MIN_LOADING_SECONDS + 0.02
    app.tournament_result = {"representative_for": "Brazil"}
    app.update_tournament_audio()
    if played != ["cup_tick", "cup_tick", "cup_tick", "cup_tick", "cup_reveal"]:
        raise AssertionError(f"Monte Carlo audio should use ticks plus reveal only: {played}")
    forbidden_goal = set(GOAL_AUDIO_SEQUENCE)
    if forbidden_goal.intersection(played):
        raise AssertionError(f"Monte Carlo audio used match/goal cues: {played}")


def validate_cup_tick_queue_contract() -> None:
    app = App(seed=2026)
    app.sound.set_scene("tournament")
    for _index in range(4):
        app.sound.play("cup_tick")
    tick_channel = app.sound.channels["tick"]
    if not tick_channel.get_busy():
        raise AssertionError("first Monte Carlo tick did not reach the tick channel")
    if app.sound.pending_cup_ticks != 3:
        raise AssertionError(f"rapid Monte Carlo ticks were not queued: pending={app.sound.pending_cup_ticks}")
    app.sound.play("cup_reveal")
    if not app.sound.pending_cup_reveal:
        raise AssertionError("cup reveal should wait behind queued progress ticks")
    for _index in range(3):
        tick_channel.stop()
        app.sound.next_cup_tick_ms = 0
        app.sound.update_crowd(0.4, False, 0.20, allow_reactions=False)
    if app.sound.pending_cup_ticks != 0:
        raise AssertionError(f"queued Monte Carlo ticks did not drain: pending={app.sound.pending_cup_ticks}")
    tick_channel.stop()
    app.sound.flush_cup_audio_queue()
    if app.sound.pending_cup_reveal:
        raise AssertionError("cup reveal stayed queued after progress ticks drained")
    if not app.sound.channels["ui_alt"].get_busy():
        raise AssertionError("cup reveal did not play after queued ticks drained")


def validate_cup_audio_queue_scene_reset() -> None:
    app = App(seed=2026)
    app.sound.set_scene("tournament")
    for _index in range(3):
        app.sound.play("cup_tick")
    app.sound.play("cup_reveal")
    if app.sound.pending_cup_ticks <= 0 or not app.sound.pending_cup_reveal:
        raise AssertionError("test setup failed to queue Monte Carlo audio")
    app.set_select()
    if app.sound.pending_cup_ticks or app.sound.pending_cup_reveal:
        raise AssertionError("scene transition did not clear pending Monte Carlo audio queue")
    app.sound.channels["tick"].stop()
    leaked: list[str] = []

    def spy_reveal() -> None:
        leaked.append("cup_reveal")

    app.sound._play_cup_reveal_now = spy_reveal  # type: ignore[method-assign]
    app.sound.set_scene("select")
    app.sound.update_crowd(0.3, False, 0.25, allow_reactions=False)
    if leaked:
        raise AssertionError("queued Copa reveal leaked after leaving tournament scene")


def validate_external_update_dt_clamp_prevents_parallel_goal_stack() -> None:
    app = App(seed=2026)
    app.set_simulate("match")
    pred = app.model.predict_matchup(app.home, app.away, seed=2026)
    app.match_prediction = pred
    app.match_intro_audio_pending = False
    app.goal_schedule = lambda _pred: [(23, "home"), (40, "home"), (68, "away")]  # type: ignore[method-assign]
    app.t = 0.0
    played: list[str] = []

    def spy(name: str, *_args: object, **_kwargs: object) -> None:
        played.append(name)

    app.sound.play = spy  # type: ignore[method-assign]
    app.update(45.0)
    if app.t > (1.0 / 20.0) + 1e-6:
        raise AssertionError(f"external update dt was not clamped before timeline advance: t={app.t:.4f}")
    if played or app.match_audio_frame_queue:
        raise AssertionError(f"giant external update stacked match audio cues in one frame: played={played} queued={app.match_audio_frame_queue}")


SMOKE_STEPS = (
    validate_runtime_assets,
    validate_sound_bag_audible_fingerprints,
    validate_no_candidate_runtime_imports,
    validate_audio_update_ownership,
    validate_runtime_mixer_pre_init_order,
    validate_engine_contract,
    validate_audio_cue_policy_contract,
)

FULL_STEPS = (
    validate_runtime_assets,
    validate_sound_bag_audible_fingerprints,
    validate_goal_mix_matrix_contract,
    validate_goal_mix_headroom,
    validate_short_event_mixes_headroom,
    validate_long_loop_seams,
    validate_no_candidate_runtime_imports,
    validate_draw_purity,
    validate_audio_update_ownership,
    validate_runtime_mixer_pre_init_order,
    validate_engine_contract,
    validate_audio_cue_policy_contract,
    validate_goal_impact_layer_contract,
    validate_initial_whistle_suppresses_reactions,
    validate_match_events_suppress_before_crowd,
    validate_match_event_order,
    validate_match_audio_visual_sync_thresholds,
    validate_whoosh_reverb_chance_timing_contract,
    validate_match_audio_quantized_frame_sync,
    validate_match_impact_quantized_frame_sync,
    validate_low_fps_goal_timeline_does_not_collapse,
    validate_match_audio_uses_post_frame_queue,
    validate_chance_audio_events,
    validate_save_near_miss_contact_pre_roll,
    validate_chance_cursor_does_not_block_final_whistle,
    validate_final_whistle_blocks_same_frame_reaction,
    validate_cup_audio_contract,
    validate_cup_tick_queue_contract,
    validate_cup_audio_queue_scene_reset,
    validate_external_update_dt_clamp_prevents_parallel_goal_stack,
)

SUITES = {
    "smoke": SMOKE_STEPS,
    "full": FULL_STEPS,
}


def step_label(step: object) -> str:
    name = getattr(step, "__name__", str(step))
    return name.removeprefix("validate_").replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Arena AI audio assets and runtime cue contracts.")
    parser.add_argument(
        "--suite",
        choices=tuple(SUITES),
        default="full",
        help="smoke checks runtime audio wiring; full also checks mixes, seams and timelines.",
    )
    args = parser.parse_args()

    init_pygame()
    suite_started = time.perf_counter()
    try:
        steps = SUITES[args.suite]
        for index, step in enumerate(steps, start=1):
            label = step_label(step)
            started = time.perf_counter()
            print(f"[audio-qa] step {index}/{len(steps)} {label}", flush=True)
            step()
            print(f"[audio-qa] ok {label} ({time.perf_counter() - started:.2f}s)", flush=True)
    finally:
        pygame.quit()
    elapsed = time.perf_counter() - suite_started
    if args.suite == "smoke":
        print(f"audio smoke passed in {elapsed:.2f}s: runtime assets, governance, mixer setup, buses and cue policy")
    else:
        print(f"audio qa passed in {elapsed:.2f}s: audio_manifest.json governance, project archive receipts, transients, 144-combo active-bed mix headroom/loudness, short event mix headroom, long-loop seams, buses, cue policy, impact-bed preservation, frame sync, goal/chance/final-whistle events, Monte Carlo tick queue/reset, and draw purity")


if __name__ == "__main__":
    main()

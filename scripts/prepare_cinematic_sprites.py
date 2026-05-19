from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pygame

from arena_ai.cinematic_uniforms import CINEMATIC_UNIFORMS, UNIFORM_CODES


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
SPRITES = ROOT / "assets" / "generated" / "cinematic"
SOURCES = ROOT / "assets" / "generated" / "cinematic_sources"
BALLS = ROOT / "assets" / "generated" / "balls3d"
BALL_SOURCE = ROOT / "assets" / "generated" / "ball_sources" / "plain_ball_sheet_8frames.png"

POSES = ("idle", "run1", "dribble", "kick")
POSE_SOURCE_COLUMNS = ("idle", "run1", "dribble", "kick", "keeper")
POSE_FRAME_SIZE = (256, 256)
RUNNER_FRAME_SIZE = (256, 256)
KEEPER_FRAME_SIZE = (288, 288)
GOAL_FRAME_SIZE = (360, 240)
GOAL_IMPACT_FRAME_SIZE = (240, 180)
BALL_FRAME_SIZE = (128, 128)
SHORTS_BY_CODE = {uniform.code: uniform.shorts for uniform in CINEMATIC_UNIFORMS}

POSE_SOURCES = (
    ("imagen_oracle_pose_sheet_8rows.png", ("blue", "sky", "red", "white", "green", "gold", "orange", "black")),
    ("imagen_oracle_burgundy_pose_row.png", ("burgundy",)),
)
RUNNER_SOURCES = (
    ("imagen_oracle_runner_sheet_8rows.png", ("blue", "sky", "red", "white", "green", "gold", "orange", "black")),
    ("imagen_oracle_burgundy_runner_row.png", ("burgundy",)),
)
POSE_LEFT_SOURCES = (
    ("imagen_oracle_pose_left_sheet_8rows.png", ("blue", "sky", "red", "white", "green", "gold", "orange", "black")),
    ("imagen_oracle_burgundy_pose_left_row.png", ("burgundy",)),
)
RUNNER_LEFT_SOURCES = (
    ("imagen_oracle_runner_left_sheet_8rows.png", ("blue", "sky", "red", "white", "green", "gold", "orange", "black")),
    ("imagen_oracle_burgundy_runner_left_row.png", ("burgundy",)),
)
KEEPER_SHEET = SOURCES / "plain_keeper_sheet_green.png"
GOAL_SHEET = SOURCES / "plain_goal_net_sheet.png"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def expected_runtime_files() -> set[str]:
    expected: set[str] = set()
    for code in UNIFORM_CODES:
        expected.update(f"{code}_{pose}.png" for pose in POSES)
        expected.update(f"left_{code}_{pose}.png" for pose in POSES)
        expected.update(f"runner_{code}_{index}.png" for index in range(4))
        expected.update(f"runner_left_{code}_{index}.png" for index in range(4))
    expected.update(f"keeper_anim_{index}.png" for index in range(4))
    expected.update(f"goal_net_{index}.png" for index in range(4))
    expected.update(f"goal_front_{index}.png" for index in range(4))
    expected.update(f"goal_impact_{index}.png" for index in range(4))
    return expected


def clean_runtime_dir() -> None:
    SPRITES.mkdir(parents=True, exist_ok=True)
    expected = expected_runtime_files()
    for path in SPRITES.glob("*.png"):
        if path.name not in expected:
            path.unlink()


def clean_ball_dir() -> None:
    BALLS.mkdir(parents=True, exist_ok=True)
    expected = expected_ball_files()
    for path in BALLS.glob("*.png"):
        if path.name not in expected:
            path.unlink()


def expected_ball_files() -> set[str]:
    return {f"ball_{index}.png" for index in range(8)}


def required_source_files() -> set[Path]:
    files = {SOURCES / filename for filename, _rows in POSE_SOURCES}
    files.update(SOURCES / filename for filename, _rows in RUNNER_SOURCES)
    files.update(SOURCES / filename for filename, _rows in POSE_LEFT_SOURCES)
    files.update(SOURCES / filename for filename, _rows in RUNNER_LEFT_SOURCES)
    files.update({KEEPER_SHEET, GOAL_SHEET, BALL_SOURCE})
    return files


def stale_runtime_files() -> list[Path]:
    expected = expected_runtime_files()
    return sorted(path for path in SPRITES.glob("*.png") if path.name not in expected)


def stale_ball_files() -> list[Path]:
    expected = expected_ball_files()
    return sorted(path for path in BALLS.glob("*.png") if path.name not in expected)


def missing_runtime_files() -> list[Path]:
    return sorted(SPRITES / filename for filename in expected_runtime_files() if not (SPRITES / filename).exists())


def missing_ball_files() -> list[Path]:
    return sorted(BALLS / filename for filename in expected_ball_files() if not (BALLS / filename).exists())


def missing_source_files() -> list[Path]:
    return sorted(path for path in required_source_files() if not path.exists())


def check_prepared_assets() -> None:
    missing_sources = missing_source_files()
    if missing_sources:
        raise RuntimeError("missing cinematic source files:\n" + "\n".join(f"  - {rel(path)}" for path in missing_sources))
    missing = missing_runtime_files() + missing_ball_files()
    if missing:
        raise RuntimeError("missing prepared runtime sprites:\n" + "\n".join(f"  - {rel(path)}" for path in missing))
    stale = stale_runtime_files() + stale_ball_files()
    if stale:
        raise RuntimeError("stale runtime sprites would be removed by prepare:\n" + "\n".join(f"  - {rel(path)}" for path in stale))


def dry_run() -> None:
    missing_sources = missing_source_files()
    if missing_sources:
        raise RuntimeError("missing cinematic source files:\n" + "\n".join(f"  - {rel(path)}" for path in missing_sources))
    missing = missing_runtime_files() + missing_ball_files()
    stale = stale_runtime_files() + stale_ball_files()
    print(f"[prepare-cinematic-sprites] would prepare {len(expected_runtime_files()) + len(expected_ball_files())} runtime sprite files")
    if stale:
        print("[prepare-cinematic-sprites] would remove stale files:")
        for path in stale:
            print(f"  - {rel(path)}")
    if missing:
        print("[prepare-cinematic-sprites] currently missing generated outputs:")
        for path in missing:
            print(f"  - {rel(path)}")
    if not missing and not stale:
        print("[prepare-cinematic-sprites] current output inventory already matches expected filenames")


def surface_from_arrays(rgb: np.ndarray, alpha: np.ndarray) -> pygame.Surface:
    width, height = alpha.shape
    result = pygame.Surface((width, height), pygame.SRCALPHA)
    target_rgb = pygame.surfarray.pixels3d(result)
    target_rgb[:, :, :] = np.clip(rgb, 0, 255).astype(np.uint8)
    del target_rgb
    target_alpha = pygame.surfarray.pixels_alpha(result)
    target_alpha[:, :] = np.clip(alpha, 0, 255).astype(np.uint8)
    del target_alpha
    return result


def smoothstep_array(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def detected_key_mode(rgb: np.ndarray) -> str:
    width, height = rgb.shape[:2]
    sample_w = max(6, width // 12)
    sample_h = max(6, height // 12)
    samples = np.concatenate(
        (
            rgb[:sample_w, :sample_h].reshape(-1, 3),
            rgb[width - sample_w :, :sample_h].reshape(-1, 3),
            rgb[:sample_w, height - sample_h :].reshape(-1, 3),
            rgb[width - sample_w :, height - sample_h :].reshape(-1, 3),
        ),
        axis=0,
    ).astype(np.float32)
    red = float(np.median(samples[:, 0]))
    green = float(np.median(samples[:, 1]))
    blue = float(np.median(samples[:, 2]))
    if green > max(red, blue) + 44 and green > 116:
        return "green"
    if min(red, blue) > green + 34 and min(red, blue) > 86:
        return "magenta"
    return "none"


def source_has_real_alpha(alpha: np.ndarray) -> bool:
    return bool(np.mean(alpha < 250) > 0.01 or np.percentile(alpha, 1) < 245)


def estimate_background_rgb(rgb: np.ndarray) -> np.ndarray:
    width, height = rgb.shape[:2]
    border = max(4, min(width, height) // 20)
    samples = np.concatenate(
        (
            rgb[:border, :, :].reshape(-1, 3),
            rgb[width - border :, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, height - border :, :].reshape(-1, 3),
        ),
        axis=0,
    ).astype(np.float32)
    return np.median(samples, axis=0)


def border_connected(mask: np.ndarray) -> np.ndarray:
    width, height = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(width):
        if mask[x, 0]:
            stack.append((x, 0))
        if mask[x, height - 1]:
            stack.append((x, height - 1))
    for y in range(height):
        if mask[0, y]:
            stack.append((0, y))
        if mask[width - 1, y]:
            stack.append((width - 1, y))
    while stack:
        x, y = stack.pop()
        if visited[x, y] or not mask[x, y]:
            continue
        visited[x, y] = True
        for nx in (x - 1, x, x + 1):
            for ny in (y - 1, y, y + 1):
                if nx == x and ny == y:
                    continue
                if 0 <= nx < width and 0 <= ny < height and not visited[nx, ny] and mask[nx, ny]:
                    stack.append((nx, ny))
    return visited


def keyed_surface(surface: pygame.Surface) -> pygame.Surface:
    converted = surface.convert_alpha()
    rgb_view = pygame.surfarray.pixels3d(converted)
    alpha_view = pygame.surfarray.pixels_alpha(converted)
    rgb = rgb_view.astype(np.float32)
    alpha = alpha_view.astype(np.float32)
    del rgb_view
    del alpha_view

    if source_has_real_alpha(alpha):
        transparent = alpha <= 1.0
        rgb[transparent] = 0
        alpha[transparent] = 0
        return surface_from_arrays(rgb, alpha)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    background = estimate_background_rgb(rgb)
    key_mode = detected_key_mode(rgb)
    green_score = np.zeros_like(alpha)
    magenta_score = np.zeros_like(alpha)
    if key_mode == "green":
        brightest_non_green = np.maximum(red, blue)
        green_dominance = green - brightest_non_green
        green_score = smoothstep_array((green_dominance - 28.0) / 88.0) * smoothstep_array((green - 108.0) / 92.0)
        distance = np.sqrt(np.sum((rgb - background[None, None, :]) ** 2, axis=2))
        candidate = (distance < 126.0) | (green_score > 0.08)
        connected = border_connected(candidate)
        key_score = np.where(connected, np.maximum(green_score, smoothstep_array((134.0 - distance) / 86.0)), 0.0)
    elif key_mode == "magenta":
        magenta_core = np.minimum(red, blue)
        magenta_dominance = magenta_core - green
        magenta_score = smoothstep_array((magenta_dominance - 16.0) / 78.0) * smoothstep_array((magenta_core - 78.0) / 92.0)
        distance = np.sqrt(np.sum((rgb - background[None, None, :]) ** 2, axis=2))
        loose_magenta = (magenta_core > 112.0) & (magenta_dominance > 12.0) & (green < 184.0)
        candidate = (distance < 176.0) | loose_magenta | (magenta_score > 0.04)
        connected = border_connected(candidate)
        key_score = np.where(connected, np.maximum(magenta_score, smoothstep_array((184.0 - distance) / 138.0)), 0.0)
    else:
        key_score = np.zeros_like(alpha)

    alpha *= 1.0 - key_score
    alpha[key_score > 0.78] = 0.0
    visible = alpha > 1.0

    green_spill = visible & (green_score > 0.05)
    if np.any(green_spill):
        neutral_green = (red + blue) * 0.5 + 10.0
        rgb[:, :, 1][green_spill] = np.minimum(green[green_spill], neutral_green[green_spill])

    magenta_spill = visible & (magenta_score > 0.05) & (alpha < 245.0)
    if np.any(magenta_spill):
        neutral_rb = green + 18.0
        rgb[:, :, 0][magenta_spill] = np.minimum(red[magenta_spill], neutral_rb[magenta_spill])
        rgb[:, :, 2][magenta_spill] = np.minimum(blue[magenta_spill], neutral_rb[magenta_spill])

    transparent = alpha <= 1.0
    rgb[transparent] = 0
    alpha[transparent] = 0
    return surface_from_arrays(rgb, alpha)


def suppress_chroma_fringe(surface: pygame.Surface) -> pygame.Surface:
    result = surface.convert_alpha()
    rgb = pygame.surfarray.pixels3d(result)
    alpha = pygame.surfarray.pixels_alpha(result)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    magenta_core = np.minimum(red, blue)
    magenta_dominance = magenta_core - green
    strong_fringe = (
        (alpha > 0)
        & (alpha < 216)
        & (magenta_core > 142)
        & (magenta_dominance > 46)
        & (np.abs(red - blue) < 92)
    )
    alpha[strong_fringe] = 0
    spill = (
        (alpha > 0)
        & (magenta_core > 118)
        & (magenta_dominance > 24)
        & (np.abs(red - blue) < 112)
    )
    if np.any(spill):
        neutral = green + 14
        rgb[:, :, 0][spill] = np.minimum(red[spill], neutral[spill])
        rgb[:, :, 2][spill] = np.minimum(blue[spill], neutral[spill])
    transparent = alpha == 0
    rgb[:, :, 0][transparent] = 0
    rgb[:, :, 1][transparent] = 0
    rgb[:, :, 2][transparent] = 0
    del rgb
    del alpha
    return result


def remove_chroma_artifact_blobs(surface: pygame.Surface, min_pixels: int = 8) -> pygame.Surface:
    result = surface.convert_alpha()
    rgb = pygame.surfarray.pixels3d(result)
    alpha = pygame.surfarray.pixels_alpha(result)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    magenta_core = np.minimum(red, blue)
    magenta_dominance = magenta_core - green
    artifact = (
        (alpha > 10)
        & (red > 168)
        & (blue > 158)
        & (green < 98)
        & (magenta_dominance > 88)
        & (np.abs(red - blue) < 72)
    )
    width, height = artifact.shape
    visited = np.zeros_like(artifact, dtype=bool)
    for start_x in range(width):
        for start_y in range(height):
            if visited[start_x, start_y] or not artifact[start_x, start_y]:
                continue
            stack = [(start_x, start_y)]
            visited[start_x, start_y] = True
            component: list[tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                component.append((x, y))
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if nx < 0 or nx >= width or ny < 0 or ny >= height:
                            continue
                        if visited[nx, ny] or not artifact[nx, ny]:
                            continue
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            if len(component) >= min_pixels:
                for x, y in component:
                    alpha[x, y] = 0
                    rgb[x, y] = (0, 0, 0)
    del rgb
    del alpha
    return result


def premultiplied_smoothscale(surface: pygame.Surface, size: tuple[int, int]) -> pygame.Surface:
    if surface.get_size() == size:
        return surface.copy()
    width, height = surface.get_size()
    rgb_view = pygame.surfarray.pixels3d(surface)
    alpha_view = pygame.surfarray.pixels_alpha(surface)
    alpha = alpha_view.astype(np.float32) / 255.0
    premultiplied = rgb_view.astype(np.float32) * alpha[:, :, None]
    premultiplied_surface = surface_from_arrays(premultiplied, alpha * 255.0)
    del rgb_view
    del alpha_view

    scaled = pygame.transform.smoothscale(premultiplied_surface, size).convert_alpha()
    scaled_rgb_view = pygame.surfarray.pixels3d(scaled)
    scaled_alpha_view = pygame.surfarray.pixels_alpha(scaled)
    scaled_alpha = scaled_alpha_view.astype(np.float32) / 255.0
    scaled_rgb = scaled_rgb_view.astype(np.float32)
    mask = scaled_alpha > 0.001
    scaled_rgb[mask] = scaled_rgb[mask] / scaled_alpha[:, :, None][mask]
    scaled_rgb[~mask] = 0
    result = surface_from_arrays(scaled_rgb, scaled_alpha * 255.0)
    del scaled_rgb_view
    del scaled_alpha_view
    return result


def sharpen_visible(surface: pygame.Surface, amount: float = 0.42) -> pygame.Surface:
    width, height = surface.get_size()
    if width < 8 or height < 8:
        return surface
    small = pygame.transform.smoothscale(surface, (max(1, width // 2), max(1, height // 2)))
    blurred = pygame.transform.smoothscale(small, (width, height)).convert_alpha()
    rgb_view = pygame.surfarray.pixels3d(surface)
    alpha_view = pygame.surfarray.pixels_alpha(surface)
    blur_rgb_view = pygame.surfarray.pixels3d(blurred)
    rgb = rgb_view.astype(np.float32)
    blur_rgb = blur_rgb_view.astype(np.float32)
    alpha = alpha_view.astype(np.float32)
    mask = alpha > 48
    rgb[mask] = np.clip(rgb[mask] + (rgb[mask] - blur_rgb[mask]) * amount, 0, 255)
    result = surface_from_arrays(rgb, alpha)
    del rgb_view
    del alpha_view
    del blur_rgb_view
    return result


def repair_uniform_artifacts(surface: pygame.Surface, uniform_code: str) -> pygame.Surface:
    shorts = SHORTS_BY_CODE.get(uniform_code)
    if shorts is None or sum(shorts) < 520:
        return surface
    result = surface.convert_alpha()
    rgb = pygame.surfarray.pixels3d(result)
    alpha = pygame.surfarray.pixels_alpha(result)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    distance = (
        np.abs(red - shorts[0])
        + np.abs(green - shorts[1])
        + np.abs(blue - shorts[2])
    )
    shorts_pixels = (alpha > 90) & (distance < 112)
    if int(shorts_pixels.sum()) < 120:
        del rgb
        del alpha
        return result

    neighbor_count = np.zeros(shorts_pixels.shape, dtype=np.int16)
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx == 0 and dy == 0:
                continue
            neighbor_count += np.roll(np.roll(shorts_pixels, dx, axis=0), dy, axis=1)
    brightness = red + green + blue
    skin_like = (red > 135) & (green > 70) & (blue < 125) & (red > green + 18)
    lower_body = np.zeros(shorts_pixels.shape, dtype=bool)
    lower_body[:, int(shorts_pixels.shape[1] * 0.50) :] = True
    internal_holes = (alpha > 20) & (alpha < 150) & (brightness < 220) & (neighbor_count >= 9) & lower_body & ~skin_like
    repair_pixels = internal_holes
    rgb[:, :, 0][repair_pixels] = shorts[0]
    rgb[:, :, 1][repair_pixels] = shorts[1]
    rgb[:, :, 2][repair_pixels] = shorts[2]
    alpha[repair_pixels] = np.maximum(alpha[repair_pixels], 218)
    del rgb
    del alpha
    return result


def remove_small_alpha_components(surface: pygame.Surface, min_pixels: int = 600) -> pygame.Surface:
    result = surface.copy()
    alpha = pygame.surfarray.pixels_alpha(result)
    width, height = result.get_size()
    visited = np.zeros((width, height), dtype=bool)
    for start_x in range(width):
        for start_y in range(height):
            if visited[start_x, start_y] or alpha[start_x, start_y] <= 25:
                continue
            stack = [(start_x, start_y)]
            visited[start_x, start_y] = True
            component = []
            while stack:
                x, y = stack.pop()
                component.append((x, y))
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if nx < 0 or nx >= width or ny < 0 or ny >= height:
                            continue
                        if visited[nx, ny] or alpha[nx, ny] <= 25:
                            continue
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            if len(component) < min_pixels:
                for x, y in component:
                    alpha[x, y] = 0
    del alpha
    return result


def keep_largest_alpha_component(surface: pygame.Surface) -> pygame.Surface:
    result = surface.copy()
    alpha = pygame.surfarray.pixels_alpha(result)
    width, height = result.get_size()
    visited = np.zeros((width, height), dtype=bool)
    largest: list[tuple[int, int]] = []
    for start_x in range(width):
        for start_y in range(height):
            if visited[start_x, start_y] or alpha[start_x, start_y] <= 25:
                continue
            stack = [(start_x, start_y)]
            visited[start_x, start_y] = True
            component = []
            while stack:
                x, y = stack.pop()
                component.append((x, y))
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if nx < 0 or nx >= width or ny < 0 or ny >= height:
                            continue
                        if visited[nx, ny] or alpha[nx, ny] <= 25:
                            continue
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            if len(component) > len(largest):
                largest = component
    keep = np.zeros((width, height), dtype=bool)
    for x, y in largest:
        keep[x, y] = True
    alpha[~keep] = 0
    del alpha
    return suppress_chroma_fringe(result)


def normalize_goal_net_frame(surface: pygame.Surface) -> pygame.Surface:
    result = surface.convert_alpha()
    rgb = pygame.surfarray.pixels3d(result)
    alpha = pygame.surfarray.pixels_alpha(result)
    visible = alpha > 4
    strength = np.maximum.reduce((rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]))
    net = visible & (strength > 80)
    alpha[net] = np.maximum(alpha[net], 56)
    shade = np.clip(strength.astype(np.int16) + 16, 178, 248).astype(np.uint8)
    depth = np.linspace(0, 18, result.get_height(), dtype=np.uint8)[None, :]
    shade_i = shade.astype(np.int16)
    depth_i = depth.repeat(result.get_width(), axis=0).astype(np.int16)
    rgb[:, :, 0][net] = np.clip(shade_i[net] - 4, 0, 255).astype(np.uint8)
    rgb[:, :, 1][net] = np.clip(shade_i[net] + 2, 0, 255).astype(np.uint8)
    rgb[:, :, 2][net] = np.clip(shade_i[net] + depth_i[net], 0, 255).astype(np.uint8)
    transparent = alpha <= 4
    rgb[:, :, 0][transparent] = 0
    rgb[:, :, 1][transparent] = 0
    rgb[:, :, 2][transparent] = 0
    alpha[transparent] = 0
    del rgb
    del alpha
    return result


def extract_goal_front_frame(surface: pygame.Surface) -> pygame.Surface:
    result = surface.convert_alpha()
    rgb = pygame.surfarray.pixels3d(result)
    alpha = pygame.surfarray.pixels_alpha(result)
    strength = np.maximum.reduce((rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2])).astype(np.int16)
    visible = (alpha > 20) & (strength > 146)
    padded = np.pad(visible.astype(np.int16), 4, mode="constant")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    density = integral[8:, 8:] - integral[:-8, 8:] - integral[8:, :-8] + integral[:-8, :-8]
    columns = np.where(visible.any(axis=1))[0]
    rows = np.where(visible.any(axis=0))[0]
    if len(columns) == 0 or len(rows) == 0:
        alpha[:, :] = 0
        del rgb
        del alpha
        return result
    left, right = int(columns[0]), int(columns[-1])
    top, bottom = int(rows[0]), int(rows[-1])
    width = max(1, right - left + 1)
    height = max(1, bottom - top + 1)
    xs = np.arange(alpha.shape[0])[:, None]
    ys = np.arange(alpha.shape[1])[None, :]
    post_shell = (
        (xs <= left + max(22, int(width * 0.15)))
        | (xs >= right - max(22, int(width * 0.15)))
        | (ys <= top + max(22, int(height * 0.18)))
        | (ys >= bottom - max(12, int(height * 0.08)))
    )
    inner_mesh = (
        (xs > left + int(width * 0.21))
        & (xs < right - int(width * 0.21))
        & (ys > top + int(height * 0.24))
        & (ys < bottom - int(height * 0.16))
    )
    keep = visible & (density >= 34) & post_shell & ~inner_mesh
    alpha[~keep] = 0
    alpha[keep] = np.maximum(alpha[keep], 210)
    rgb[:, :, 0][keep] = np.maximum(rgb[:, :, 0][keep], 236)
    rgb[:, :, 1][keep] = np.maximum(rgb[:, :, 1][keep], 238)
    rgb[:, :, 2][keep] = np.maximum(rgb[:, :, 2][keep], 234)
    transparent = alpha == 0
    rgb[:, :, 0][transparent] = 0
    rgb[:, :, 1][transparent] = 0
    rgb[:, :, 2][transparent] = 0
    del rgb
    del alpha
    return result


def fit_into_frame(
    surface: pygame.Surface,
    frame_size: tuple[int, int],
    max_size: tuple[int, int],
    bottom_pad: int,
) -> pygame.Surface:
    rect = surface.get_bounding_rect()
    if rect.w <= 0 or rect.h <= 0:
        raise RuntimeError("empty sprite cell after chroma key")
    subject = surface.subsurface(rect).copy()
    scale = min(max_size[0] / subject.get_width(), max_size[1] / subject.get_height())
    scaled_size = (max(1, int(subject.get_width() * scale)), max(1, int(subject.get_height() * scale)))
    scaled = premultiplied_smoothscale(subject, scaled_size)
    scaled = sharpen_visible(remove_chroma_artifact_blobs(suppress_chroma_fringe(scaled)))
    frame = pygame.Surface(frame_size, pygame.SRCALPHA)
    x = (frame_size[0] - scaled.get_width()) // 2
    y = frame_size[1] - scaled.get_height() - bottom_pad
    frame.blit(scaled, (x, y))
    return remove_chroma_artifact_blobs(suppress_chroma_fringe(frame))


def fit_centered_frame(
    surface: pygame.Surface,
    frame_size: tuple[int, int],
    max_size: tuple[int, int],
) -> pygame.Surface:
    rect = surface.get_bounding_rect()
    if rect.w <= 0 or rect.h <= 0:
        raise RuntimeError("empty sprite cell after chroma key")
    subject = surface.subsurface(rect).copy()
    scale = min(max_size[0] / subject.get_width(), max_size[1] / subject.get_height())
    scaled_size = (max(1, int(subject.get_width() * scale)), max(1, int(subject.get_height() * scale)))
    scaled = premultiplied_smoothscale(subject, scaled_size)
    scaled = sharpen_visible(remove_chroma_artifact_blobs(suppress_chroma_fringe(scaled)))
    frame = pygame.Surface(frame_size, pygame.SRCALPHA)
    x = (frame_size[0] - scaled.get_width()) // 2
    y = (frame_size[1] - scaled.get_height()) // 2
    frame.blit(scaled, (x, y))
    return remove_chroma_artifact_blobs(suppress_chroma_fringe(frame))


def extract_grid(
    sheet_path: Path,
    row_names: tuple[str, ...],
    column_names: tuple[str, ...],
    frame_size: tuple[int, int],
    max_size: tuple[int, int],
    bottom_pad: int,
    output_pattern: str,
    min_component: int,
    keep_largest: bool = False,
    saved_columns: set[str] | None = None,
    crop_pad_y: int = 0,
) -> list[Path]:
    if not sheet_path.exists():
        raise FileNotFoundError(f"missing image_gen cinematic source sheet: {sheet_path}")
    sheet = pygame.image.load(sheet_path).convert_alpha()
    width, height = sheet.get_size()
    outputs = []
    for row, row_name in enumerate(row_names):
        y1 = round(row * height / len(row_names))
        y2 = round((row + 1) * height / len(row_names))
        for col, col_name in enumerate(column_names):
            x1 = round(col * width / len(column_names))
            x2 = round((col + 1) * width / len(column_names))
            crop_y1 = max(0, y1 - crop_pad_y)
            crop_y2 = min(height, y2 + crop_pad_y)
            cell = sheet.subsurface(pygame.Rect(x1, crop_y1, x2 - x1, crop_y2 - crop_y1)).copy()
            cleaned = remove_small_alpha_components(keyed_surface(cell), min_pixels=min_component)
            if keep_largest:
                cleaned = keep_largest_alpha_component(cleaned)
            frame = fit_into_frame(cleaned, frame_size, max_size, bottom_pad)
            frame = repair_uniform_artifacts(frame, row_name)
            if saved_columns is not None and col_name not in saved_columns:
                continue
            path = SPRITES / output_pattern.format(row=row_name, col=col_name)
            pygame.image.save(frame, path)
            outputs.append(path)
    return outputs


def extract_row_sheet(
    sheet_path: Path,
    frame_size: tuple[int, int],
    max_size: tuple[int, int],
    bottom_pad: int,
    columns: int = 4,
    min_component: int = 700,
    keep_largest: bool = False,
) -> list[pygame.Surface]:
    if not sheet_path.exists():
        raise FileNotFoundError(f"missing cinematic source sheet: {sheet_path}")
    sheet = pygame.image.load(sheet_path).convert_alpha()
    width, height = sheet.get_size()
    frames = []
    for col in range(columns):
        x1 = round(col * width / columns)
        x2 = round((col + 1) * width / columns)
        cell = sheet.subsurface(pygame.Rect(x1, 0, x2 - x1, height)).copy()
        cleaned = remove_small_alpha_components(keyed_surface(cell), min_pixels=min_component)
        if keep_largest:
            cleaned = keep_largest_alpha_component(cleaned)
        frames.append(fit_into_frame(cleaned, frame_size, max_size, bottom_pad))
    return frames


def extract_sheet_row(
    sheet_path: Path,
    row: int,
    rows: int,
    frame_size: tuple[int, int],
    max_size: tuple[int, int],
    bottom_pad: int,
    columns: int = 4,
    min_component: int = 700,
) -> list[pygame.Surface]:
    if not sheet_path.exists():
        raise FileNotFoundError(f"missing cinematic source sheet: {sheet_path}")
    sheet = pygame.image.load(sheet_path).convert_alpha()
    width, height = sheet.get_size()
    y1 = round(row * height / rows)
    y2 = round((row + 1) * height / rows)
    frames = []
    for col in range(columns):
        x1 = round(col * width / columns)
        x2 = round((col + 1) * width / columns)
        cell = sheet.subsurface(pygame.Rect(x1, y1, x2 - x1, y2 - y1)).copy()
        cleaned = remove_small_alpha_components(keyed_surface(cell), min_pixels=min_component)
        frames.append(fit_into_frame(cleaned, frame_size, max_size, bottom_pad))
    return frames


def create_pose_variants() -> list[Path]:
    outputs = []
    for sources, pattern in (
        (POSE_SOURCES, "{row}_{col}.png"),
        (POSE_LEFT_SOURCES, "left_{row}_{col}.png"),
    ):
        for filename, rows in sources:
            outputs.extend(
                extract_grid(
                    SOURCES / filename,
                    rows,
                    POSE_SOURCE_COLUMNS,
                    POSE_FRAME_SIZE,
                    (218, 218),
                    16,
                    pattern,
                    700,
                    True,
                    set(POSES),
                    36,
                )
            )
    return outputs


def create_runner_variants() -> list[Path]:
    outputs = []
    for sources, pattern in (
        (RUNNER_SOURCES, "runner_{row}_{col}.png"),
        (RUNNER_LEFT_SOURCES, "runner_left_{row}_{col}.png"),
    ):
        for filename, rows in sources:
            outputs.extend(
                extract_grid(
                    SOURCES / filename,
                    rows,
                    ("0", "1", "2", "3"),
                    RUNNER_FRAME_SIZE,
                    (232, 238),
                    8,
                    pattern,
                    2200,
                    True,
                    None,
                    28,
                )
            )
    return outputs


def create_keeper_animation() -> list[Path]:
    frames = extract_row_sheet(KEEPER_SHEET, KEEPER_FRAME_SIZE, (264, 246), 12, min_component=700, keep_largest=True)
    outputs = []
    for index, image in enumerate(frames):
        path = SPRITES / f"keeper_anim_{index}.png"
        pygame.image.save(image, path)
        outputs.append(path)
    return outputs


def create_goal_net_animation() -> list[Path]:
    frames = extract_sheet_row(GOAL_SHEET, 0, 2, GOAL_FRAME_SIZE, (348, 210), 10, min_component=1)
    outputs = []
    for index, image in enumerate(frames):
        image = normalize_goal_net_frame(image)
        path = SPRITES / f"goal_net_{index}.png"
        pygame.image.save(image, path)
        outputs.append(path)
        front = extract_goal_front_frame(image)
        path = SPRITES / f"goal_front_{index}.png"
        pygame.image.save(front, path)
        outputs.append(path)
    impact_frames = extract_sheet_row(GOAL_SHEET, 1, 2, GOAL_IMPACT_FRAME_SIZE, (224, 156), 12, min_component=1)
    for index, image in enumerate(impact_frames):
        image = normalize_goal_net_frame(image)
        path = SPRITES / f"goal_impact_{index}.png"
        pygame.image.save(image, path)
        outputs.append(path)
    return outputs


def create_ball_animation() -> list[Path]:
    raw_frames = extract_row_sheet(BALL_SOURCE, BALL_FRAME_SIZE, (116, 116), 0, columns=8, min_component=250, keep_largest=True)
    outputs = []
    for index, image in enumerate(raw_frames):
        image = fit_centered_frame(image, BALL_FRAME_SIZE, (110, 110))
        path = BALLS / f"ball_{index}.png"
        pygame.image.save(image, path)
        outputs.append(path)
    return outputs


def create_variants() -> list[Path]:
    missing_sources = missing_source_files()
    if missing_sources:
        raise RuntimeError("missing cinematic source files:\n" + "\n".join(f"  - {rel(path)}" for path in missing_sources))
    pygame.init()
    pygame.display.set_mode((1, 1))
    clean_runtime_dir()
    clean_ball_dir()
    outputs = []
    outputs.extend(create_pose_variants())
    outputs.extend(create_runner_variants())
    outputs.extend(create_keeper_animation())
    outputs.extend(create_goal_net_animation())
    outputs.extend(create_ball_animation())
    missing = []
    for code in UNIFORM_CODES:
        for pose in POSES:
            if not (SPRITES / f"{code}_{pose}.png").exists():
                missing.append(f"{code}_{pose}.png")
            if not (SPRITES / f"left_{code}_{pose}.png").exists():
                missing.append(f"left_{code}_{pose}.png")
        for index in range(4):
            if not (SPRITES / f"runner_{code}_{index}.png").exists():
                missing.append(f"runner_{code}_{index}.png")
            if not (SPRITES / f"runner_left_{code}_{index}.png").exists():
                missing.append(f"runner_left_{code}_{index}.png")
    if missing:
        raise RuntimeError(f"missing generated cinematic sprites: {missing}")
    pygame.quit()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare generated cinematic runtime sprites from source sheets.")
    parser.add_argument("--dry-run", action="store_true", help="Report the files that would be generated or removed without writing.")
    parser.add_argument("--check", action="store_true", help="Validate that prepared outputs already match the expected inventory without writing.")
    args = parser.parse_args()
    if args.dry_run and args.check:
        parser.error("--dry-run and --check are mutually exclusive")
    if args.dry_run:
        dry_run()
        return
    if args.check:
        check_prepared_assets()
        print("[prepare-cinematic-sprites] prepared sprite inventory OK")
        return
    outputs = create_variants()
    print(f"prepared {len(outputs)} image_gen cinematic sprite files")


if __name__ == "__main__":
    main()

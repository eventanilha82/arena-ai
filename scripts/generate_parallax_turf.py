from __future__ import annotations

import os
from pathlib import Path

import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "assets" / "generated" / "parallax_sources"
OUTPUT_DIR = ROOT / "assets" / "generated" / "parallax"
SOURCES = {
    "turf_near_strip.png": "imagen_turf_near_source.png",
    "turf_mid_strip.png": "imagen_turf_mid_source.png",
}
TARGET_SIZE = (1440, 232)
SEAM_BLEND = 180
EDGE_BLEND = 96


def prepare_strip(source: pygame.Surface, layer_name: str) -> pygame.Surface:
    near_layer = "near" in layer_name
    scale_bias = 1.18 if near_layer else 1.04
    scale = max(TARGET_SIZE[0] / source.get_width(), TARGET_SIZE[1] / source.get_height()) * scale_bias
    scaled = pygame.transform.smoothscale(
        source,
        (
            max(TARGET_SIZE[0], int(source.get_width() * scale)),
            max(TARGET_SIZE[1], int(source.get_height() * scale)),
        ),
    )
    crop = pygame.Rect(0, 0, TARGET_SIZE[0], TARGET_SIZE[1])
    crop.center = scaled.get_rect().center
    if near_layer:
        crop.y = min(max(0, scaled.get_height() - TARGET_SIZE[1]), crop.y + int(TARGET_SIZE[1] * 0.16))
    else:
        crop.y = max(0, crop.y - int(TARGET_SIZE[1] * 0.10))
    result = pygame.Surface(TARGET_SIZE, pygame.SRCALPHA)
    result.blit(scaled, (0, 0), crop)
    result = grade_layer(result, near_layer)
    return make_horizontal_seamless(result)


def grade_layer(surface: pygame.Surface, near_layer: bool) -> pygame.Surface:
    result = surface.copy()
    width, height = result.get_size()
    for y in range(height):
        depth = y / max(1, height - 1)
        for x in range(width):
            color = result.get_at((x, y))
            stripe = 1.0 + (0.035 if ((x // (56 if near_layer else 92)) % 2 == 0) else -0.025)
            if near_layer:
                contrast = 1.08 + depth * 0.07
                tint = (8, 22, 5)
                alpha = 242
            else:
                contrast = 0.88
                tint = (0, 10, 16)
                alpha = 215
            r = clamp_channel((color.r + tint[0]) * contrast * stripe)
            g = clamp_channel((color.g + tint[1]) * contrast * stripe)
            b = clamp_channel((color.b + tint[2]) * contrast * stripe)
            result.set_at((x, y), (r, g, b, min(color.a, alpha)))
    if not near_layer:
        result = pygame.transform.smoothscale(result, (TARGET_SIZE[0] // 2, TARGET_SIZE[1] // 2))
        result = pygame.transform.smoothscale(result, TARGET_SIZE)
    return result


def clamp_channel(value: float) -> int:
    return max(0, min(255, int(value)))


def make_horizontal_seamless(surface: pygame.Surface) -> pygame.Surface:
    width, height = surface.get_size()
    half = width // 2
    wrapped = pygame.Surface((width, height), pygame.SRCALPHA)
    wrapped.blit(surface, (-half, 0))
    wrapped.blit(surface, (width - half, 0))

    blend_width = min(SEAM_BLEND, width // 4)
    seam_x = half
    blended = wrapped.copy()
    for offset in range(-blend_width, blend_width):
        x = seam_x + offset
        if x < 0 or x >= width:
            continue
        t = (offset + blend_width) / max(1, blend_width * 2)
        sample_left = max(0, min(width - 1, x - blend_width))
        sample_right = max(0, min(width - 1, x + blend_width))
        for y in range(height):
            left = wrapped.get_at((sample_left, y))
            right = wrapped.get_at((sample_right, y))
            color = (
                int(left.r * (1 - t) + right.r * t),
                int(left.g * (1 - t) + right.g * t),
                int(left.b * (1 - t) + right.b * t),
                int(left.a * (1 - t) + right.a * t),
            )
            blended.set_at((x, y), color)
    return match_horizontal_edges(blended)


def match_horizontal_edges(surface: pygame.Surface) -> pygame.Surface:
    result = surface.copy()
    width, height = result.get_size()
    edge_width = min(EDGE_BLEND, width // 5)
    for x in range(edge_width):
        strength = 1.0 - x / max(1, edge_width - 1)
        left_x = x
        right_x = width - 1 - x
        for y in range(height):
            left = result.get_at((left_x, y))
            right = result.get_at((right_x, y))
            avg = (
                int((left.r + right.r) / 2),
                int((left.g + right.g) / 2),
                int((left.b + right.b) / 2),
                int((left.a + right.a) / 2),
            )
            left_mix = (
                int(left.r * (1 - strength) + avg[0] * strength),
                int(left.g * (1 - strength) + avg[1] * strength),
                int(left.b * (1 - strength) + avg[2] * strength),
                int(left.a * (1 - strength) + avg[3] * strength),
            )
            right_mix = (
                int(right.r * (1 - strength) + avg[0] * strength),
                int(right.g * (1 - strength) + avg[1] * strength),
                int(right.b * (1 - strength) + avg[2] * strength),
                int(right.a * (1 - strength) + avg[3] * strength),
            )
            result.set_at((left_x, y), left_mix)
            result.set_at((right_x, y), right_mix)
    return result


def generate_strips() -> list[Path]:
    pygame.init()
    pygame.display.set_mode((1, 1))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for output_name, source_name in SOURCES.items():
        source_path = SOURCE_DIR / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"missing image_gen parallax source: {source_path}")
        image = pygame.image.load(source_path).convert_alpha()
        output_path = OUTPUT_DIR / output_name
        pygame.image.save(prepare_strip(image, output_name), output_path)
        paths.append(output_path)
    pygame.quit()
    return paths


def main() -> None:
    paths = generate_strips()
    print("prepared image_gen parallax strips:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

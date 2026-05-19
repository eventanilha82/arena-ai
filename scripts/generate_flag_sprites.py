from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
SHEET_DIR = ROOT / "assets" / "generated" / "flag_sheets"
OUTPUT = ROOT / "assets" / "generated" / "flags"
TARGET_SIZE = (172, 108)
MAX_SUBJECT = (164, 100)

SHEETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("imagen_flags_01.png", ("mex", "rsa", "kor", "cze", "can", "bih")),
    ("imagen_flags_02.png", ("qat", "sui", "bra", "mar", "hai", "sco")),
    ("imagen_flags_03.png", ("usa", "par", "aus", "tur", "ger", "cur")),
    ("imagen_flags_04.png", ("civ", "ecu", "ned", "jpn", "swe", "tun")),
    ("imagen_flags_05.png", ("bel", "egy", "irn", "nzl", "esp", "cpv")),
    ("imagen_flags_06.png", ("ksa", "uru", "fra", "sen", "irq", "nor")),
    ("imagen_flags_07.png", ("arg", "alg", "aut", "jor", "por", "cod")),
    ("imagen_flags_08.png", ("uzb", "col", "eng", "cro", "gha", "pan")),
)


def chroma_key_magenta(surface: pygame.Surface) -> pygame.Surface:
    image = surface.convert_alpha()
    rgb = pygame.surfarray.pixels3d(image)
    alpha = pygame.surfarray.pixels_alpha(image)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)

    key = (red > 168) & (blue > 168) & (green < 132) & (np.abs(red - blue) < 118)
    fringe = (alpha > 0) & (red > 136) & (blue > 136) & (green < 156) & (np.abs(red - blue) < 140)
    purple_shadow = (alpha > 0) & (red > 60) & (blue > 60) & (green < 88) & (np.abs(red - blue) < 96)
    candidates = key | fringe | purple_shadow
    width, height = image.get_size()
    remove = np.zeros((width, height), dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(width):
        if candidates[x, 0]:
            stack.append((x, 0))
        if candidates[x, height - 1]:
            stack.append((x, height - 1))
    for y in range(height):
        if candidates[0, y]:
            stack.append((0, y))
        if candidates[width - 1, y]:
            stack.append((width - 1, y))
    while stack:
        x, y = stack.pop()
        if remove[x, y] or not candidates[x, y]:
            continue
        remove[x, y] = True
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and not remove[nx, ny] and candidates[nx, ny]:
                stack.append((nx, ny))
    alpha[remove] = 0
    del rgb
    del alpha
    return image


def remove_small_components(surface: pygame.Surface, min_pixels: int = 120) -> pygame.Surface:
    result = surface.copy()
    alpha = pygame.surfarray.pixels_alpha(result)
    width, height = result.get_size()
    visited = np.zeros((width, height), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for start_x in range(width):
        for start_y in range(height):
            if visited[start_x, start_y] or alpha[start_x, start_y] <= 18:
                continue
            stack = [(start_x, start_y)]
            visited[start_x, start_y] = True
            pixels = []
            while stack:
                x, y = stack.pop()
                pixels.append((x, y))
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if nx < 0 or nx >= width or ny < 0 or ny >= height:
                            continue
                        if visited[nx, ny] or alpha[nx, ny] <= 18:
                            continue
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            components.append(pixels)

    if not components:
        del alpha
        return result
    largest = max(len(component) for component in components)
    for component in components:
        if len(component) < min_pixels or len(component) < largest * 0.015:
            for x, y in component:
                alpha[x, y] = 0
    del alpha
    return result


def fit_sprite(subject: pygame.Surface) -> pygame.Surface:
    bbox = subject.get_bounding_rect()
    if bbox.w <= 0 or bbox.h <= 0:
        raise RuntimeError("empty generated flag cell after chroma key")
    cropped = subject.subsurface(bbox).copy()
    scale = min(MAX_SUBJECT[0] / cropped.get_width(), MAX_SUBJECT[1] / cropped.get_height())
    scaled_size = (max(1, int(cropped.get_width() * scale)), max(1, int(cropped.get_height() * scale)))
    scaled = pygame.transform.smoothscale(cropped, scaled_size)
    canvas = pygame.Surface(TARGET_SIZE, pygame.SRCALPHA)
    canvas.blit(scaled, scaled.get_rect(center=(TARGET_SIZE[0] // 2, TARGET_SIZE[1] // 2)))
    return canvas


def extract_sheet(sheet_path: Path, codes: tuple[str, ...]) -> list[Path]:
    sheet = pygame.image.load(sheet_path).convert_alpha()
    width, height = sheet.get_size()
    cell_w = width / 3
    cell_h = height / 2
    paths = []
    for index, code in enumerate(codes):
        col = index % 3
        row = index // 3
        cell = pygame.Rect(
            round(col * cell_w),
            round(row * cell_h),
            round(cell_w),
            round(cell_h),
        ).clip(sheet.get_rect())
        prepared = remove_small_components(chroma_key_magenta(sheet.subsurface(cell).copy()))
        sprite = fit_sprite(prepared)
        path = OUTPUT / f"{code}.png"
        pygame.image.save(sprite, path)
        paths.append(path)
    return paths


def generate_flags() -> list[Path]:
    pygame.init()
    pygame.display.set_mode((1, 1))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated = []
    for filename, codes in SHEETS:
        sheet_path = SHEET_DIR / filename
        if not sheet_path.exists():
            raise FileNotFoundError(f"missing image_gen flag sheet: {sheet_path}")
        generated.extend(extract_sheet(sheet_path, codes))
    pygame.quit()
    return generated


def main() -> None:
    paths = generate_flags()
    print(f"recut {len(paths)} image_gen flag sprites into {OUTPUT}")


if __name__ == "__main__":
    main()

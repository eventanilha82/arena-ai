from __future__ import annotations

from collections import OrderedDict

import pygame


class SurfaceCache:
    """Bounded cache for expensive pygame surface transforms."""

    def __init__(
        self,
        max_scaled: int = 900,
        max_flipped: int = 400,
        max_roto: int = 1200,
        max_cover: int = 240,
        max_alpha: int = 720,
    ):
        self.max_scaled = max_scaled
        self.max_flipped = max_flipped
        self.max_roto = max_roto
        self.max_cover = max_cover
        self.max_alpha = max_alpha
        self.scaled: OrderedDict[tuple[pygame.Surface, int, int], pygame.Surface] = OrderedDict()
        self.flipped: OrderedDict[pygame.Surface, pygame.Surface] = OrderedDict()
        self.roto: OrderedDict[tuple[pygame.Surface, int, int], pygame.Surface] = OrderedDict()
        self.covered: OrderedDict[tuple[pygame.Surface, int, int, int], pygame.Surface] = OrderedDict()
        self.alpha: OrderedDict[tuple[pygame.Surface, int], pygame.Surface] = OrderedDict()

    @staticmethod
    def _trim(cache: OrderedDict[object, pygame.Surface], limit: int) -> None:
        while len(cache) > limit:
            cache.popitem(last=False)

    @staticmethod
    def _hit(cache: OrderedDict[object, pygame.Surface], key: object) -> pygame.Surface | None:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
        return cached

    def smoothscale(self, image: pygame.Surface, size: tuple[int, int]) -> pygame.Surface:
        width = max(1, int(size[0]))
        height = max(1, int(size[1]))
        key = (image, width, height)
        cached = self._hit(self.scaled, key)
        if cached is None:
            cached = pygame.transform.smoothscale(image, (width, height)).convert_alpha()
            self.scaled[key] = cached
            self._trim(self.scaled, self.max_scaled)
        return cached

    def flip(self, image: pygame.Surface) -> pygame.Surface:
        key = image
        cached = self._hit(self.flipped, key)
        if cached is None:
            cached = pygame.transform.flip(image, True, False).convert_alpha()
            self.flipped[key] = cached
            self._trim(self.flipped, self.max_flipped)
        return cached

    def rotozoom(self, image: pygame.Surface, angle: float, scale: float) -> pygame.Surface:
        angle_key = int(round(angle * 10))
        scale_key = int(round(scale * 1000))
        key = (image, angle_key, scale_key)
        cached = self._hit(self.roto, key)
        if cached is None:
            cached = pygame.transform.rotozoom(image, angle_key / 10.0, scale_key / 1000.0).convert_alpha()
            self.roto[key] = cached
            self._trim(self.roto, self.max_roto)
        return cached

    def cover(self, image: pygame.Surface, size: tuple[int, int], alpha: int = 255) -> pygame.Surface:
        width = max(1, int(size[0]))
        height = max(1, int(size[1]))
        alpha = max(0, min(255, int(alpha)))
        key = (image, width, height, alpha)
        cached = self._hit(self.covered, key)
        if cached is not None:
            return cached

        scale = max(width / image.get_width(), height / image.get_height())
        scaled_size = (max(1, int(image.get_width() * scale)), max(1, int(image.get_height() * scale)))
        scaled = self.smoothscale(image, scaled_size)
        source = pygame.Rect(
            max(0, (scaled.get_width() - width) // 2),
            max(0, (scaled.get_height() - height) // 2),
            width,
            height,
        )
        covered = pygame.Surface((width, height), pygame.SRCALPHA)
        covered.blit(scaled, (0, 0), source)
        if alpha < 255:
            covered.set_alpha(alpha)
        self.covered[key] = covered
        self._trim(self.covered, self.max_cover)
        return covered

    def with_alpha(self, image: pygame.Surface, alpha: int, step: int = 4) -> pygame.Surface:
        alpha = max(0, min(255, int(alpha)))
        if alpha >= 255:
            return image
        alpha_key = int(round(alpha / max(1, step)) * max(1, step))
        alpha_key = max(0, min(255, alpha_key))
        key = (image, alpha_key)
        cached = self._hit(self.alpha, key)
        if cached is None:
            cached = image.copy()
            cached.set_alpha(alpha_key)
            self.alpha[key] = cached
            self._trim(self.alpha, self.max_alpha)
        return cached

    def stats(self) -> dict[str, int]:
        return {
            "scaled": len(self.scaled),
            "flipped": len(self.flipped),
            "roto": len(self.roto),
            "covered": len(self.covered),
            "alpha": len(self.alpha),
        }


class TextCache:
    """Small cache for repeated HUD text surfaces."""

    def __init__(self, max_entries: int = 1800):
        self.max_entries = max_entries
        self.surfaces: dict[tuple[int, str, tuple[int, int, int]], pygame.Surface] = {}

    def render(self, text_font: pygame.font.Font, text: str, color: tuple[int, int, int]) -> pygame.Surface:
        key = (id(text_font), text, color)
        rendered = self.surfaces.get(key)
        if rendered is None:
            rendered = text_font.render(text, True, color)
            self.surfaces[key] = rendered
            while len(self.surfaces) > self.max_entries:
                self.surfaces.pop(next(iter(self.surfaces)))
        return rendered

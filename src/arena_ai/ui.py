from __future__ import annotations

import pygame


WHITE = (242, 247, 250)
BUTTON_FILL = (18, 50, 64)
BUTTON_FILL_HOVER = (28, 76, 94)
BUTTON_BORDER = (77, 119, 136)


class Button:
    def __init__(self, rect: pygame.Rect, label: str, accent: tuple[int, int, int]):
        self.rect = rect
        self.label = label
        self.accent = accent
        self._label_cache: dict[tuple[int, str, tuple[int, int], tuple[int, int, int]], pygame.Surface] = {}

    def fitted_label_surface(self, text_font: pygame.font.Font, color: tuple[int, int, int] = WHITE) -> pygame.Surface:
        max_size = (max(8, self.rect.w - 28), max(8, self.rect.h - 16))
        cache_key = (id(text_font), self.label, max_size, color)
        cached = self._label_cache.get(cache_key)
        if cached is not None:
            return cached

        rendered = text_font.render(self.label, True, color)
        width, height = rendered.get_size()
        max_width, max_height = max_size
        if width > max_width or height > max_height:
            scale = min(max_width / max(1, width), max_height / max(1, height))
            fitted_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            rendered = pygame.transform.smoothscale(rendered, fitted_size)

        if len(self._label_cache) > 16:
            self._label_cache.clear()
        self._label_cache[cache_key] = rendered
        return rendered

    def draw(self, surface: pygame.Surface, text_font: pygame.font.Font, mouse: tuple[int, int]) -> None:
        hover = self.rect.collidepoint(mouse)
        pygame.draw.rect(surface, BUTTON_FILL_HOVER if hover else BUTTON_FILL, self.rect, border_radius=14)
        pygame.draw.rect(surface, self.accent if hover else BUTTON_BORDER, self.rect, 2, border_radius=14)
        rendered = self.fitted_label_surface(text_font)
        surface.blit(rendered, rendered.get_rect(center=self.rect.center))

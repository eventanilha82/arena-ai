from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CinematicUniform:
    code: str
    label_pt: str
    primary: tuple[int, int, int]
    shorts: tuple[int, int, int]
    diversity_note: str


CINEMATIC_UNIFORMS: tuple[CinematicUniform, ...] = (
    CinematicUniform("blue", "azul", (42, 105, 224), (236, 238, 242), "Africa"),
    CinematicUniform("sky", "azul claro/celeste", (112, 205, 242), (34, 62, 95), "Asia"),
    CinematicUniform("red", "vermelho", (218, 48, 42), (246, 246, 240), "South America"),
    CinematicUniform("burgundy", "vinho/bordo", (116, 28, 45), (246, 246, 240), "Middle East/North Africa"),
    CinematicUniform("white", "branco", (238, 239, 235), (28, 32, 38), "Europe"),
    CinematicUniform("green", "verde", (25, 154, 73), (245, 245, 238), "Asia/Oceania"),
    CinematicUniform("gold", "amarelo/dourado", (238, 198, 35), (34, 86, 176), "South America/Africa"),
    CinematicUniform("orange", "laranja", (232, 111, 24), (30, 32, 34), "Oceania"),
    CinematicUniform("black", "preto", (24, 25, 28), (238, 238, 232), "North America/Caribbean"),
)

UNIFORM_CODES: tuple[str, ...] = tuple(uniform.code for uniform in CINEMATIC_UNIFORMS)
UNIFORM_COLORS: dict[str, tuple[int, int, int]] = {uniform.code: uniform.primary for uniform in CINEMATIC_UNIFORMS}

TEAM_UNIFORM_OVERRIDES: dict[str, str] = {
    "ARG": "sky",
    "BRA": "gold",
    "COL": "gold",
    "ECU": "gold",
    "FRA": "blue",
    "GER": "white",
    "DEU": "white",
    "ENG": "white",
    "ESP": "red",
    "POR": "red",
    "PRT": "red",
    "URU": "sky",
    "URY": "sky",
    "NED": "orange",
    "MEX": "green",
    "KSA": "green",
    "QAT": "burgundy",
    "NZL": "black",
}

from __future__ import annotations

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena_ai.main import (  # noqa: E402
    App,
    BG,
    CHANCE_EVENT_WINDOW_MINUTES,
    GOAL_EVENT_WINDOW_MINUTES,
    GOLD,
    HEIGHT,
    SIMULATION_SECONDS,
    SHOT_NET_VISUAL_CONTACT_AT,
    WIDTH,
    font,
)
from scripts.validate_visuals import away_win_prediction, home_win_prediction, neutral_prediction, seek_match_time  # noqa: E402


OUTPUT_DIR = ROOT / "artifacts" / "visual_qa" / "current"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_frame(app: App, seconds: float, label: str, filename: str) -> pygame.Surface:
    app.screen.fill(BG)
    pred = app.match_prediction
    if pred is None:
        raise RuntimeError("visual QA requires app.match_prediction")
    seek_match_time(app, pred, seconds)
    app.screen.fill(BG)
    app.draw_top(label, "QA visual")
    cinematic_focus = app.match_cinematic_focus(pred)
    app.draw_field(pred, pred, "CONFRONTO")
    app.draw_side_panel(pred, cinematic_focus=cinematic_focus)
    app.draw_score_panel({"CONFRONTO": pred}, "CONFRONTO", pred, cinematic_focus=cinematic_focus)
    frame = app.screen.copy()
    pygame.image.save(frame, OUTPUT_DIR / filename)
    return frame


def append_sample(
    samples: list[tuple[str, str, pygame.Surface]],
    label: str,
    filename: str,
    frame: pygame.Surface,
) -> None:
    samples.append((label, filename, frame))


def set_matchup(app: App, home_code: str, away_code: str) -> None:
    codes = [team.code for team in app.teams]
    app.home_idx = codes.index(home_code)
    app.away_idx = codes.index(away_code)


def append_contact_cell(
    sheet: pygame.Surface,
    frame: pygame.Surface,
    label: str,
    index: int,
    label_font: pygame.font.Font,
    columns: int,
) -> None:
    cell_w = WIDTH // columns
    cell_h = HEIGHT // 2
    x = (index % columns) * cell_w
    y = (index // columns) * cell_h
    thumbnail = pygame.transform.smoothscale(frame, (cell_w, cell_h))
    sheet.blit(thumbnail, (x, y))
    panel = pygame.Surface((cell_w, 34), pygame.SRCALPHA)
    panel.fill((0, 7, 10, 190))
    rendered = label_font.render(label, True, GOLD)
    panel.blit(rendered, (16, 8))
    sheet.blit(panel, (x, y))


def main() -> None:
    pygame.init()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_frame in OUTPUT_DIR.glob("*.png"):
        old_frame.unlink()
    for old_metadata in OUTPUT_DIR.glob("*.json"):
        old_metadata.unlink()
    app = App(seed=2026)
    label_font = font(16)
    samples: list[tuple[str, str, pygame.Surface]] = []

    app.draw_menu()
    menu_frame = app.screen.copy()
    menu_filename = "00_menu.png"
    pygame.image.save(menu_frame, OUTPUT_DIR / menu_filename)
    append_sample(samples, "tela inicial - ícone", menu_filename, menu_frame)

    app.state = "select"
    app.screen.fill(BG)
    app.draw_select()
    select_frame = app.screen.copy()
    select_filename = "00b_selecao.png"
    pygame.image.save(select_frame, OUTPUT_DIR / select_filename)
    append_sample(samples, "seleção - confronto", select_filename, select_frame)

    app.set_simulate("match")

    def goal_progress_seconds(goal_minute: int, progress: float) -> float:
        return (goal_minute - GOAL_EVENT_WINDOW_MINUTES + progress * GOAL_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS

    def chance_progress_seconds(chance_minute: int, progress: float) -> float:
        return (chance_minute - CHANCE_EVENT_WINDOW_MINUTES + progress * CHANCE_EVENT_WINDOW_MINUTES) / 90.0 * SIMULATION_SECONDS

    def first_chance_event(pred, kind: str) -> tuple[int, str, str]:
        for chance in app.chance_schedule(pred):
            if chance[2] == kind:
                return chance
        raise RuntimeError(f"visual QA requires a {kind!r} chance event")

    home_pred = home_win_prediction()
    app.match_prediction = home_pred
    first_home_goal = app.goal_schedule(home_pred)[0][0]
    append_sample(samples, "00s - posse e parallax", "01_posse_inicial.png", save_frame(app, 0.0, "Confronto | posse inicial", "01_posse_inicial.png"))
    append_sample(
        samples,
        "aproximação - corrida",
        "02_aproximacao.png",
        save_frame(app, goal_progress_seconds(first_home_goal, 0.44), "Confronto | aproximação", "02_aproximacao.png"),
    )
    append_sample(
        samples,
        "pé na bola",
        "03_pe_na_bola.png",
        save_frame(app, goal_progress_seconds(first_home_goal, 0.555), "Confronto | pé na bola", "03_pe_na_bola.png"),
    )
    append_sample(
        samples,
        "chute - bola em voo",
        "04_bola_em_voo.png",
        save_frame(app, goal_progress_seconds(first_home_goal, 0.66), "Confronto | bola em voo", "04_bola_em_voo.png"),
    )
    append_sample(
        samples,
        "rede - impacto",
        "05_impacto_rede.png",
        save_frame(
            app,
            goal_progress_seconds(first_home_goal, SHOT_NET_VISUAL_CONTACT_AT),
            "Confronto | impacto na rede",
            "05_impacto_rede.png",
        ),
    )
    append_sample(
        samples,
        "GOOOL - overlay pós-impacto",
        "05b_gol_overlay.png",
        save_frame(
            app,
            goal_progress_seconds(first_home_goal, min(1.0, SHOT_NET_VISUAL_CONTACT_AT + 0.012)),
            "Confronto | GOOOL",
            "05b_gol_overlay.png",
        ),
    )

    away_pred = away_win_prediction()
    app.match_prediction = away_pred
    first_away_goal = app.goal_schedule(away_pred)[0][0]
    append_sample(
        samples,
        "lado invertido",
        "06_gol_visitante.png",
        save_frame(
            app,
            goal_progress_seconds(first_away_goal, SHOT_NET_VISUAL_CONTACT_AT),
            "Confronto | gol visitante",
            "06_gol_visitante.png",
        ),
    )

    draw_pred = neutral_prediction()
    app.match_prediction = draw_pred
    append_sample(samples, "empate - jogo vivo", "07_empate_vivo.png", save_frame(app, 10.0, "Confronto | empate vivo", "07_empate_vivo.png"))
    append_sample(samples, "empate final", "08_empate_final.png", save_frame(app, SIMULATION_SECONDS, "Confronto | empate final", "08_empate_final.png"))

    set_matchup(app, "PAR", "ALG")
    app.match_prediction = app.model.predict_matchup(app.home, app.away, seed=2026)
    append_sample(samples, "regressão PAR x ALG", "08b_par_alg_regressao.png", save_frame(app, 43.0, "Confronto | PAR x ALG", "08b_par_alg_regressao.png"))
    app.match_prediction = away_win_prediction()
    alg_goal = app.goal_schedule(app.match_prediction)[0][0]
    append_sample(
        samples,
        "visitante verde - chute",
        "08c_visitante_verde_chute.png",
        save_frame(
            app,
            goal_progress_seconds(alg_goal, 0.54),
            "Confronto | visitante verde",
            "08c_visitante_verde_chute.png",
        ),
    )
    append_sample(
        samples,
        "visitante verde - gol",
        "08d_visitante_verde_gol.png",
        save_frame(
            app,
            goal_progress_seconds(alg_goal, SHOT_NET_VISUAL_CONTACT_AT),
            "Confronto | visitante verde gol",
            "08d_visitante_verde_gol.png",
        ),
    )

    home_pred = home_win_prediction()
    app.match_prediction = home_pred
    save_chance_minute, _save_side, _save_kind = first_chance_event(home_pred, "save")
    append_sample(
        samples,
        "quase gol - defesa",
        "08e_quase_gol_defesa.png",
        save_frame(
            app,
            chance_progress_seconds(save_chance_minute, SHOT_NET_VISUAL_CONTACT_AT),
            "Confronto | quase gol - defesa",
            "08e_quase_gol_defesa.png",
        ),
    )
    wide_chance_minute, _wide_side, _wide_kind = first_chance_event(home_pred, "wide")
    append_sample(
        samples,
        "trave raspando",
        "08f_trave_raspando.png",
        save_frame(
            app,
            chance_progress_seconds(wide_chance_minute, SHOT_NET_VISUAL_CONTACT_AT),
            "Confronto | trave raspando",
            "08f_trave_raspando.png",
        ),
    )
    append_sample(samples, "fim - placar", "09_placar_final.png", save_frame(app, SIMULATION_SECONDS, "Confronto | placar final", "09_placar_final.png"))

    for code, color_name, filename in (
        ("MEX", "verde", "13_uniforme_verde.png"),
        ("NED", "laranja", "14_uniforme_laranja.png"),
        ("NZL", "preto", "15_uniforme_preto.png"),
    ):
        set_matchup(app, code, "FRA")
        app.match_prediction = app.model.predict_matchup(app.home, app.away, seed=2026)
        home_goal_minutes = [goal_minute for goal_minute, side in app.goal_schedule(app.match_prediction) if side == "home"]
        uniform_goal_minute = home_goal_minutes[0] if home_goal_minutes else app.goal_schedule(app.match_prediction)[0][0]
        append_sample(
            samples,
            f"uniforme {color_name}",
            filename,
            save_frame(
                app,
                goal_progress_seconds(uniform_goal_minute, 0.54),
                f"Confronto | uniforme {color_name}",
                filename,
            ),
        )

    app.state = "tournament"
    app.t = 1.7
    app.tournament_result = None
    app.champion_odds = []
    app.mc_running = True
    app.mc_progress_done = 420
    app.mc_progress_total = app.champion_odds_runs
    app.screen.fill(BG)
    app.draw_tournament()
    loading_frame = app.screen.copy()
    loading_filename = "10_copa_calculando.png"
    pygame.image.save(loading_frame, OUTPUT_DIR / loading_filename)
    append_sample(samples, "copa - calculando", loading_filename, loading_frame)

    odds, representative = app.model.champion_odds_with_representative(
        runs=120,
        seed=2026,
        workers=app.champion_odds_workers,
        progress_with_odds=False,
    )
    app.mc_running = False
    app.mc_progress_done = app.champion_odds_runs
    app.mc_progress_total = app.champion_odds_runs
    app.champion_odds = odds
    app.tournament_result = representative
    for view, label, filename in (
        ("groups", "copa - grupos", "11_copa_grupos.png"),
        ("bracket", "copa - mata-mata", "12_copa_mata_mata.png"),
    ):
        app.tournament_view = view
        app.screen.fill(BG)
        app.draw_tournament()
        frame = app.screen.copy()
        pygame.image.save(frame, OUTPUT_DIR / filename)
        append_sample(samples, label, filename, frame)

    columns = 3
    rows = ceil(len(samples) / columns)
    sheet = pygame.Surface((WIDTH, rows * (HEIGHT // 2)), pygame.SRCALPHA)
    sheet.fill(BG)
    for index, (label, _filename, frame) in enumerate(samples):
        append_contact_cell(sheet, frame, label, index, label_font, columns)
    contact_sheet = OUTPUT_DIR / "contact_sheet.png"
    pygame.image.save(sheet, contact_sheet)
    metadata = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "seed": 2026,
        "viewport": {"width": WIDTH, "height": HEIGHT},
        "sample_count": len(samples),
        "contact_sheet": contact_sheet.name,
        "frames": [
            {
                "file": filename,
                "label": label,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for label, filename, _frame in samples
            for path in (OUTPUT_DIR / filename,)
        ],
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pygame.quit()
    print(f"visual QA frames saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

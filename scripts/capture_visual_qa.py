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
    SHOT_KICK_AT,
    SHOT_NET_AT,
    SHOT_NET_SETTLE_PROGRESS,
    SHOT_NET_VISUAL_CONTACT_AT,
    SHOT_RELEASE_END,
    WIDTH,
    font,
)
from arena_ai.worldcup_model import Prediction  # noqa: E402
from scripts.validate_visuals import (  # noqa: E402
    away_win_prediction,
    ball_contract_snapshot,
    home_win_prediction,
    neutral_prediction,
    seek_match_time,
)


OUTPUT_DIR = ROOT / "artifacts" / "visual_qa" / "current"
SEQUENCE_DIR = OUTPUT_DIR / "cinematic_sequence"
VARIANT_DIR = OUTPUT_DIR / "cinematic_variants"
FIELD_RECT = pygame.Rect(32, 110, 910, 490)
CONTACT_LABEL_HEIGHT = 34
CONTACT_COLUMNS = 3
SEQUENCE_COLUMNS = 4


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
    target = OUTPUT_DIR / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(frame, target)
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
    frame_h = round(cell_w * HEIGHT / WIDTH)
    cell_h = frame_h + CONTACT_LABEL_HEIGHT
    x = (index % columns) * cell_w
    y = (index // columns) * cell_h
    panel = pygame.Surface((cell_w, CONTACT_LABEL_HEIGHT), pygame.SRCALPHA)
    panel.fill((0, 7, 10, 190))
    rendered = label_font.render(label, True, GOLD)
    panel.blit(rendered, (16, 8))
    sheet.blit(panel, (x, y))
    thumbnail = pygame.transform.smoothscale(frame, (cell_w, frame_h))
    sheet.blit(thumbnail, (x, y + CONTACT_LABEL_HEIGHT))


def contact_sheet_size(sample_count: int, columns: int) -> tuple[int, int, int, int]:
    cell_w = WIDTH // columns
    frame_h = round(cell_w * HEIGHT / WIDTH)
    cell_h = frame_h + CONTACT_LABEL_HEIGHT
    return WIDTH, ceil(sample_count / columns) * cell_h, cell_w, frame_h


def capture_cinematic_sequence(
    app: App,
    pred: Prediction,
    goal_minute: int,
    label_font: pygame.font.Font,
) -> dict[str, object]:
    progress_samples = tuple(
        sorted(
            (
                0.50,
                0.66,
                SHOT_KICK_AT - 0.01,
                SHOT_KICK_AT,
                SHOT_KICK_AT + 0.02,
                0.78,
                max(SHOT_RELEASE_END + 0.05, 0.86),
                SHOT_NET_AT,
                SHOT_NET_VISUAL_CONTACT_AT,
                1.04,
                SHOT_NET_VISUAL_CONTACT_AT + SHOT_NET_SETTLE_PROGRESS * 0.58,
                SHOT_NET_VISUAL_CONTACT_AT + SHOT_NET_SETTLE_PROGRESS + 0.02,
            )
        )
    )
    sequence_samples: list[tuple[str, str, pygame.Surface]] = []
    sequence_metadata: list[dict[str, object]] = []
    app.match_prediction = pred
    for index, progress in enumerate(progress_samples):
        seconds = (
            goal_minute - GOAL_EVENT_WINDOW_MINUTES + progress * GOAL_EVENT_WINDOW_MINUTES
        ) / 90.0 * SIMULATION_SECONDS
        label = f"{index + 1:02d} | p={progress:.3f}"
        filename = f"cinematic_sequence/shot_{index:02d}_p{progress:.3f}.png"
        frame = save_frame(app, seconds, f"Confronto | sequência {label}", filename)
        state = app.cinematic_scene_state(FIELD_RECT, pred)
        contract = ball_contract_snapshot(state, f"capture sequence progress={progress:.3f}")
        path = OUTPUT_DIR / filename
        sequence_samples.append((label, filename, frame))
        sequence_metadata.append(
            {
                "index": index,
                "file": filename,
                "label": label,
                "requested_progress": progress,
                "seconds": seconds,
                "post_impact": progress > 1.0,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "ball_state": contract,
            }
        )

    strip_width, strip_height, cell_w, frame_h = contact_sheet_size(
        len(sequence_samples), SEQUENCE_COLUMNS
    )
    strip = pygame.Surface((strip_width, strip_height), pygame.SRCALPHA)
    strip.fill(BG)
    for index, (label, _filename, frame) in enumerate(sequence_samples):
        append_contact_cell(strip, frame, label, index, label_font, SEQUENCE_COLUMNS)
    strip_path = OUTPUT_DIR / "cinematic_shot_sequence.png"
    pygame.image.save(strip, strip_path)
    return {
        "event": "home_goal",
        "goal_minute": goal_minute,
        "frame_count": len(sequence_samples),
        "contains_post_impact_progress": any(progress > 1.0 for progress in progress_samples),
        "post_impact_frame_count": sum(progress > 1.0 for progress in progress_samples),
        "strip": strip_path.name,
        "strip_sha256": file_sha256(strip_path),
        "layout": {
            "columns": SEQUENCE_COLUMNS,
            "cell_width": cell_w,
            "frame_height": frame_h,
            "label_height": CONTACT_LABEL_HEIGHT,
            "aspect_ratio_preserved": True,
        },
        "ffmpeg_required": False,
        "frames": sequence_metadata,
    }


def capture_cinematic_variants(app: App, label_font: pygame.font.Font) -> dict[str, object]:
    zones = (
        "alto firme",
        "baixo cruzado",
        "meia altura",
        "angulo seco",
        "rasteiro forte",
        "central forte",
    )
    samples: list[tuple[str, str, pygame.Surface]] = []
    metadata: list[dict[str, object]] = []

    def capture_variant(
        label: str,
        filename: str,
        pred: Prediction,
        event_minute: int,
        event_window: float,
        progress: float,
        extra: dict[str, object],
    ) -> None:
        seconds = (
            event_minute - event_window + progress * event_window
        ) / 90.0 * SIMULATION_SECONDS
        frame = save_frame(app, seconds, f"Confronto | {label}", filename)
        state = app.cinematic_scene_state(FIELD_RECT, pred)
        contract = ball_contract_snapshot(state, f"variant {label}")
        path = OUTPUT_DIR / filename
        samples.append((label, filename, frame))
        metadata.append(
            {
                "file": filename,
                "label": label,
                "progress": progress,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "ball_state": contract,
                **extra,
            }
        )

    for zone_index, zone in enumerate(zones):
        scoring_side = "home" if zone_index % 2 == 0 else "away"
        pred = home_win_prediction() if scoring_side == "home" else away_win_prediction()
        example: tuple[int, int] | None = None
        for seed in range(2026, 2400):
            app.match_seed = seed
            app.match_prediction = pred
            for goal_minute, side in app.goal_schedule(pred):
                if side != scoring_side:
                    continue
                direction = 1 if side == "home" else -1
                goal = app.cinematic_goal_rect(FIELD_RECT, "right" if direction > 0 else "left")
                if app.cinematic_shot_profile(goal, direction, goal_minute).zone == zone:
                    example = (seed, goal_minute)
                    break
            if example is not None:
                break
        if example is None:
            raise RuntimeError(f"visual QA could not find profile {zone!r} for {scoring_side}")

        seed, goal_minute = example
        app.match_seed = seed
        app.match_prediction = pred
        slug = zone.replace(" ", "_")
        for phase, progress in (
            ("voo", max(SHOT_RELEASE_END + 0.04, 0.85)),
            ("contato", SHOT_NET_VISUAL_CONTACT_AT),
            ("rede", SHOT_NET_VISUAL_CONTACT_AT + 0.14),
        ):
            filename = f"cinematic_variants/{zone_index:02d}_{slug}_{phase}.png"
            capture_variant(
                f"{zone} | {scoring_side} | {phase}",
                filename,
                pred,
                goal_minute,
                GOAL_EVENT_WINDOW_MINUTES,
                progress,
                {"kind": "goal", "zone": zone, "side": scoring_side, "seed": seed},
            )

    app.match_seed = 2026
    chance_pred = home_win_prediction()
    app.match_prediction = chance_pred
    chance_by_kind = {
        kind: (minute, side)
        for minute, side, kind in app.chance_schedule(chance_pred)
    }
    for kind, progresses in (
        ("save", (SHOT_NET_AT - 0.03, SHOT_NET_AT, SHOT_NET_AT + 0.10)),
        ("wide", (SHOT_NET_AT - 0.03, SHOT_NET_AT - 0.01, SHOT_NET_AT + 0.02)),
    ):
        if kind not in chance_by_kind:
            raise RuntimeError(f"visual QA requires a {kind!r} chance event")
        chance_minute, side = chance_by_kind[kind]
        for phase, progress in zip(("pre", "contato", "pos"), progresses):
            filename = f"cinematic_variants/{kind}_{phase}.png"
            capture_variant(
                f"{kind} | {phase}",
                filename,
                chance_pred,
                chance_minute,
                CHANCE_EVENT_WINDOW_MINUTES,
                progress,
                {"kind": kind, "side": side, "seed": app.match_seed},
            )

    sheet_width, sheet_height, cell_width, frame_height = contact_sheet_size(len(samples), 3)
    sheet = pygame.Surface((sheet_width, sheet_height), pygame.SRCALPHA)
    sheet.fill(BG)
    for index, (label, _filename, frame) in enumerate(samples):
        append_contact_cell(sheet, frame, label, index, label_font, 3)
    sheet_path = OUTPUT_DIR / "cinematic_variants_contact_sheet.png"
    pygame.image.save(sheet, sheet_path)
    return {
        "frame_count": len(samples),
        "goal_profiles": list(zones),
        "sides": ["home", "away"],
        "chance_kinds": ["save", "wide"],
        "sheet": sheet_path.name,
        "sheet_sha256": file_sha256(sheet_path),
        "layout": {
            "columns": 3,
            "cell_width": cell_width,
            "frame_height": frame_height,
            "label_height": CONTACT_LABEL_HEIGHT,
            "aspect_ratio_preserved": True,
        },
        "frames": metadata,
    }


def main() -> None:
    pygame.init()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_frame in OUTPUT_DIR.glob("*.png"):
        old_frame.unlink()
    for old_metadata in OUTPUT_DIR.glob("*.json"):
        old_metadata.unlink()
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    for old_sequence_frame in SEQUENCE_DIR.glob("*.png"):
        old_sequence_frame.unlink()
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    for old_variant_frame in VARIANT_DIR.glob("*.png"):
        old_variant_frame.unlink()
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
    cinematic_sequence = capture_cinematic_sequence(app, home_pred, first_home_goal, label_font)
    cinematic_variants = capture_cinematic_variants(app, label_font)
    app.match_seed = 2026
    app.match_prediction = home_pred
    first_home_goal = app.goal_schedule(home_pred)[0][0]
    flight_progress = min(SHOT_NET_AT - 0.02, max(SHOT_RELEASE_END + 0.05, 0.86))
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
        save_frame(
            app,
            goal_progress_seconds(first_home_goal, SHOT_KICK_AT),
            "Confronto | pé na bola",
            "03_pe_na_bola.png",
        ),
    )
    append_sample(
        samples,
        "chute - bola em voo",
        "04_bola_em_voo.png",
        save_frame(
            app,
            goal_progress_seconds(first_home_goal, flight_progress),
            "Confronto | bola em voo",
            "04_bola_em_voo.png",
        ),
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
            chance_progress_seconds(save_chance_minute, SHOT_NET_AT + 0.10),
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
            chance_progress_seconds(wide_chance_minute, SHOT_NET_AT - 0.01),
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

    sheet_width, sheet_height, contact_cell_width, contact_frame_height = contact_sheet_size(
        len(samples), CONTACT_COLUMNS
    )
    sheet = pygame.Surface((sheet_width, sheet_height), pygame.SRCALPHA)
    sheet.fill(BG)
    for index, (label, _filename, frame) in enumerate(samples):
        append_contact_cell(sheet, frame, label, index, label_font, CONTACT_COLUMNS)
    contact_sheet = OUTPUT_DIR / "contact_sheet.png"
    pygame.image.save(sheet, contact_sheet)
    metadata = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "seed": 2026,
        "viewport": {"width": WIDTH, "height": HEIGHT},
        "sample_count": len(samples),
        "contact_sheet": contact_sheet.name,
        "contact_sheet_layout": {
            "columns": CONTACT_COLUMNS,
            "cell_width": contact_cell_width,
            "frame_height": contact_frame_height,
            "label_height": CONTACT_LABEL_HEIGHT,
            "aspect_ratio_preserved": True,
        },
        "cinematic_sequence": cinematic_sequence,
        "cinematic_variants": cinematic_variants,
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
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pygame.quit()
    print(f"visual QA frames saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

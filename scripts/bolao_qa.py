from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arena_ai import bolao  # noqa: E402
from arena_ai.worldcup_model import WorldCupModel, sota  # noqa: E402


EXPECTED_GROUPS = tuple("ABCDEFGHIJKL")
EXPECTED_GROUP_MATCHES = 72
EXPECTED_TEAMS = 48
EXPECTED_QUALIFIERS = 32
EXPECTED_ANNEX_C_MATRIX_SHA256 = "e26273d1b736d39d5b0876f36c8b818d8e687ba12db5991cb642b0b0ce17201e"
OFFICIAL_KNOCKOUT_SLOTS = {
    73: ("2A", "2B"),
    74: ("1E", "3ABCDF"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", "3CDFGH"),
    78: ("2E", "2I"),
    79: ("1A", "3CEFHI"),
    80: ("1L", "3EHIJK"),
    81: ("1D", "3BEFIJ"),
    82: ("1G", "3AEHIJ"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", "3EFGIJ"),
    86: ("1J", "2H"),
    87: ("1K", "3DEIJL"),
    88: ("2D", "2G"),
    89: ("W74", "W77"),
    90: ("W73", "W75"),
    91: ("W76", "W78"),
    92: ("W79", "W80"),
    93: ("W83", "W84"),
    94: ("W81", "W82"),
    95: ("W86", "W88"),
    96: ("W85", "W87"),
    97: ("W89", "W90"),
    98: ("W93", "W94"),
    99: ("W91", "W92"),
    100: ("W95", "W96"),
    101: ("W97", "W98"),
    102: ("W99", "W100"),
    103: ("RU101", "RU102"),
    104: ("W101", "W102"),
}
OFFICIAL_KNOCKOUT_CITIES = {
    73: "Los Angeles",
    74: "Boston",
    75: "Monterrey",
    76: "Houston",
    77: "New York/New Jersey",
    78: "Dallas",
    79: "Mexico City",
    80: "Atlanta",
    81: "San Francisco Bay Area",
    82: "Seattle",
    83: "Toronto",
    84: "Los Angeles",
    85: "Vancouver",
    86: "Miami",
    87: "Kansas City",
    88: "Dallas",
    89: "Philadelphia",
    90: "Houston",
    91: "New York/New Jersey",
    92: "Mexico City",
    93: "Dallas",
    94: "Seattle",
    95: "Atlanta",
    96: "Vancouver",
    97: "Boston",
    98: "Los Angeles",
    99: "Miami",
    100: "Kansas City",
    101: "Dallas",
    102: "Atlanta",
    103: "Miami",
    104: "New York/New Jersey",
}
EXPECTED_ROUND32_RESULTS = {
    73: ("South Africa", "Canada", 0, 1, 0, 0, "Canada", "90min", None, None),
    74: ("Germany", "Paraguay", 1, 1, 0, 0, "Paraguay", "penalties", 3, 4),
    75: ("Netherlands", "Morocco", 1, 1, 0, 0, "Morocco", "penalties", 2, 3),
    76: ("Brazil", "Japan", 2, 1, 0, 0, "Brazil", "90min", None, None),
    77: ("France", "Sweden", 3, 0, 0, 0, "France", "90min", None, None),
    78: ("Côte d'Ivoire", "Norway", 1, 2, 0, 0, "Norway", "90min", None, None),
    79: ("Mexico", "Ecuador", 2, 0, 0, 0, "Mexico", "90min", None, None),
    80: ("England", "Congo DR", 2, 1, 0, 0, "England", "90min", None, None),
    81: ("USA", "Bosnia and Herzegovina", 2, 0, 0, 0, "USA", "90min", None, None),
    82: ("Belgium", "Senegal", 2, 2, 1, 0, "Belgium", "extra_time", None, None),
    83: ("Portugal", "Croatia", 2, 1, 0, 0, "Portugal", "90min", None, None),
    84: ("Spain", "Austria", 3, 0, 0, 0, "Spain", "90min", None, None),
    85: ("Switzerland", "Algeria", 2, 0, 0, 0, "Switzerland", "90min", None, None),
    86: ("Argentina", "Cape Verde Islands", 1, 1, 2, 1, "Argentina", "extra_time", None, None),
    87: ("Colombia", "Ghana", 1, 0, 0, 0, "Colombia", "90min", None, None),
    88: ("Australia", "Egypt", 1, 1, 0, 0, "Egypt", "penalties", 2, 4),
}
EXPECTED_ROUND16_RESULTS = {
    89: ("Paraguay", "France", 0, 1, 0, 0, "France", "90min", None, None),
    90: ("Canada", "Morocco", 0, 3, 0, 0, "Morocco", "90min", None, None),
    91: ("Brazil", "Norway", 1, 2, 0, 0, "Norway", "90min", None, None),
    92: ("Mexico", "England", 2, 3, 0, 0, "England", "90min", None, None),
    93: ("Portugal", "Spain", 0, 1, 0, 0, "Spain", "90min", None, None),
    94: ("USA", "Belgium", 1, 4, 0, 0, "Belgium", "90min", None, None),
    95: ("Argentina", "Egypt", 3, 2, 0, 0, "Argentina", "90min", None, None),
    96: ("Switzerland", "Colombia", 0, 0, 0, 0, "Switzerland", "penalties", 4, 3),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def csv_rows() -> list[dict[str, str]]:
    path = bolao.OBSERVED_GROUP_RESULTS_PATH
    require(path.is_file(), f"CSV interno do bolao ausente: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        required = {"match_number", "group", "home_team", "away_team", "home_goals", "away_goals", "source"}
        require(reader.fieldnames is not None, "CSV interno do bolao esta sem cabecalho")
        missing = sorted(required.difference(reader.fieldnames))
        require(not missing, f"CSV interno do bolao sem colunas obrigatorias: {', '.join(missing)}")

    require(len(rows) == EXPECTED_GROUP_MATCHES, f"CSV precisa ter {EXPECTED_GROUP_MATCHES} jogos, encontrou {len(rows)}")
    numbers: list[int] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            match_number = int(str(row["match_number"]).strip())
            home_goals = int(str(row["home_goals"]).strip())
            away_goals = int(str(row["away_goals"]).strip())
        except (TypeError, ValueError) as error:
            raise AssertionError(f"CSV invalido na linha {line_number}: placar ou numero de jogo invalido") from error
        require(match_number > 0, f"CSV invalido na linha {line_number}: match_number deve ser positivo")
        require(home_goals >= 0 and away_goals >= 0, f"CSV invalido na linha {line_number}: gols nao podem ser negativos")
        require(str(row["group"]).strip().upper() in EXPECTED_GROUPS, f"CSV invalido na linha {line_number}: grupo invalido")
        require(bool(str(row["source"]).strip()), f"CSV invalido na linha {line_number}: source obrigatorio")
        numbers.append(match_number)
    require(len(numbers) == len(set(numbers)), "CSV interno do bolao tem match_number duplicado")
    require(
        {str(row["source"]).strip() for row in rows} == {"user_reported_group_stage"},
        "CSV precisa identificar a proveniência manual do snapshot",
    )
    return rows


def validate_group_stage(model: WorldCupModel) -> bolao.GroupStageBoard:
    snapshot = bolao.load_observed_snapshot_metadata()
    observed = bolao.load_observed_group_results(model, snapshot=snapshot)
    fixtures = model.fixtures[model.fixtures["stage_id"] == sota.GROUP_STAGE_ID].sort_values("match_number")
    expected_numbers = {int(row.match_number) for row in fixtures.itertuples(index=False)}

    require(len(fixtures) == EXPECTED_GROUP_MATCHES, f"fixture esperado com {EXPECTED_GROUP_MATCHES} jogos, encontrou {len(fixtures)}")
    require(set(observed) == expected_numbers, "CSV nao cobre exatamente todos os jogos oficiais da fase de grupos")
    require(snapshot.snapshot_kind == "manual_local", "metadata precisa identificar snapshot manual local")
    require(snapshot.official_source is False, "snapshot manual não pode se apresentar como fonte oficial")
    require("fifa" not in f"{snapshot.source_label} {snapshot.row_source}".casefold(), "snapshot manual se apresentou como FIFA")
    latest_kickoff = pd.Timestamp(fixtures["kickoff_at"].max())
    require(pd.Timestamp(snapshot.as_of) > latest_kickoff, "as_of precisa ser posterior ao último kickoff observado")

    board = bolao.build_group_stage_board(model)
    require(board.snapshot == snapshot, "quadro de grupos não preservou a metadata do snapshot")
    matches = [match for group_matches in board.matches_by_group.values() for match in group_matches]
    require(len(matches) == EXPECTED_GROUP_MATCHES, f"fase de grupos montou {len(matches)} jogos, esperado {EXPECTED_GROUP_MATCHES}")
    require(all(match.is_observed for match in matches), "fase de grupos ainda contem placar projetado")
    require(Counter(match.group for match in matches) == Counter({group: 6 for group in EXPECTED_GROUPS}), "cada grupo deve ter seis jogos")

    for match in matches:
        result = observed[match.match_number]
        require(
            (match.home, match.away, match.home_goals, match.away_goals)
            == (result.home, result.away, result.home_goals, result.away_goals),
            f"placar do jogo {match.match_number} diverge entre CSV e quadro de grupos",
        )

    standings = board.standings
    require(len(standings) == EXPECTED_TEAMS, f"classificacao tem {len(standings)} equipes, esperado {EXPECTED_TEAMS}")
    require(set(standings["group"]) == set(EXPECTED_GROUPS), "classificacao nao contem todos os grupos")
    require(Counter(str(group) for group in standings["group"]) == Counter({group: 4 for group in EXPECTED_GROUPS}), "cada grupo deve ter quatro equipes")
    require((standings["played"] == 3).all(), "todas as equipes devem encerrar a fase de grupos com tres partidas")
    require(len(board.best_third_teams) == 8, "devem existir oito melhores terceiros")
    require(len(board.qualified_teams) == EXPECTED_QUALIFIERS, f"devem existir {EXPECTED_QUALIFIERS} classificados")
    require(len(board.qualifiers) == EXPECTED_QUALIFIERS, "chave de classificados incompleta")
    require(len(board.third_order) == 8, "ordem dos terceiros classificados incompleta")
    require("fifa_rank" in standings.columns, "classificação precisa carregar o ranking FIFA embalado")
    require(standings["fifa_rank"].notna().all(), "classificação ficou sem ranking FIFA embalado")
    return board


def validate_official_knockout_bracket(model: WorldCupModel) -> None:
    knockout = model.fixtures[model.fixtures["stage_id"] > sota.GROUP_STAGE_ID]
    by_number = {int(row.match_number): row for row in knockout.itertuples(index=False)}
    require(set(by_number) == set(OFFICIAL_KNOCKOUT_SLOTS), "fixture não contém os 32 jogos oficiais do mata-mata")
    for match_number, expected_slots in OFFICIAL_KNOCKOUT_SLOTS.items():
        fixture = by_number[match_number]
        require(
            sota.parse_match_label(str(fixture.match_label)) == expected_slots,
            f"slots oficiais divergentes no jogo {match_number}",
        )
        require(
            str(fixture.city_city_name) == OFFICIAL_KNOCKOUT_CITIES[match_number],
            f"cidade oficial divergente no jogo {match_number}",
        )
        for slot in expected_slots:
            reference = slot[2:] if slot.startswith("RU") else slot[1:] if slot.startswith("W") else ""
            if reference.isdigit():
                require(
                    int(reference) < match_number,
                    f"chave contém referência circular ou futura: jogo {match_number} -> {slot}",
                )


def validate_annex_c_third_place_matrix(model: WorldCupModel, board: bolao.GroupStageBoard) -> None:
    """Lock the official Annex C table and prove the live third-place assignment."""
    sota.run_fifa_2026_rule_self_tests()
    matrix_payload = json.dumps(
        sota.FIFA_2026_ANNEX_C_THIRD_PLACE_MATRIX,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual_hash = hashlib.sha256(matrix_payload).hexdigest()
    require(actual_hash == EXPECTED_ANNEX_C_MATRIX_SHA256, "matriz do Anexo C divergiu da transcrição FIFA validada")

    slots = list(sota.FIFA_2026_ANNEX_C_THIRD_PLACE_SLOT_ORDER)
    for groups, allocation in sota.FIFA_2026_ANNEX_C_THIRD_PLACE_MATRIX.items():
        assignment = sota.assign_third_slots(slots, list(reversed(groups)))
        require(
            tuple(assignment[slot] for slot in slots) == tuple(f"3{group}" for group in allocation),
            f"Anexo C não preservou a opção para grupos {groups}",
        )
        require(
            set(value[1] for value in assignment.values()) == set(groups),
            f"Anexo C não usa exatamente os oito terceiros para grupos {groups}",
        )
        require(
            all(value[1] in slot[1:] for slot, value in assignment.items()),
            f"Anexo C alocou terceiro em slot inelegível para grupos {groups}",
        )

    current_assignment = sota.assign_third_slots(slots, board.third_order)
    require("".join(sorted(board.third_order)) == "BDEFIJKL", "foto atual deveria selecionar a combinação FIFA BDEFIJKL")
    require(current_assignment["3ABCDF"] == "3D", "Anexo C atual precisa colocar 3D no M74")
    require(current_assignment["3CDFGH"] == "3F", "Anexo C atual precisa colocar 3F no M77")
    require(
        (board.qualifiers["1E"], board.qualifiers[current_assignment["3ABCDF"]]) == ("Germany", "Paraguay"),
        "M74 atual não confere com Alemanha x Paraguai",
    )
    require(
        (board.qualifiers["1I"], board.qualifiers[current_assignment["3CDFGH"]]) == ("France", "Sweden"),
        "M77 atual não confere com França x Suécia",
    )


def validate_snapshot_guard(model: WorldCupModel, observed: dict[int, bolao.ObservedResult]) -> None:
    snapshot = bolao.load_observed_snapshot_metadata()
    fixtures = model.fixtures[model.fixtures["stage_id"] == sota.GROUP_STAGE_ID].sort_values(
        ["kickoff_at", "match_number"]
    )
    first_match_number = int(next(fixtures.itertuples(index=False)).match_number)
    chronological_numbers = [int(row.match_number) for row in fixtures.itertuples(index=False)]
    prefix = {match_number: observed[match_number] for match_number in chronological_numbers[:48]}
    bolao.validate_observed_result_snapshot(model, prefix, snapshot=snapshot)
    complete_fair_play = {str(row.team_key): 0.0 for row in model.squad.itertuples(index=False)}
    fair_play_snapshot = replace(snapshot, fair_play_scores=complete_fair_play)
    require(
        bolao.validate_observed_fair_play_scores(model, fair_play_snapshot) == complete_fair_play,
        "metadata não preservou o mapa completo de fair play",
    )
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="bolao_fair_play_metadata_", dir=artifacts) as temporary_dir:
        metadata_path = Path(temporary_dir) / "snapshot.json"
        payload = json.loads(bolao.OBSERVED_SNAPSHOT_METADATA_PATH.read_text(encoding="utf-8"))
        payload["fair_play_scores"] = complete_fair_play
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        with patch.object(bolao, "OBSERVED_SNAPSHOT_METADATA_PATH", metadata_path):
            parsed_snapshot = bolao.load_observed_snapshot_metadata()
    require(
        parsed_snapshot.fair_play_scores == complete_fair_play,
        "parser de metadata não carregou fair_play_scores do JSON",
    )
    try:
        bolao.validate_observed_fair_play_scores(model, replace(snapshot, fair_play_scores={"Brazil": 0.0}))
    except ValueError as error:
        require("cobrir todas" in str(error), f"fair play parcial falhou de forma inesperada: {error}")
    else:
        raise AssertionError("fair play parcial não pode decidir desempate de grupo")
    stale_snapshot = replace(snapshot, as_of=snapshot.as_of - timedelta(days=30))
    try:
        bolao.validate_observed_result_snapshot(model, {first_match_number: observed[first_match_number]}, snapshot=stale_snapshot)
    except ValueError as error:
        require("duração mínima" in str(error), f"guard de as_of falhou com erro inesperado: {error}")
    else:
        raise AssertionError("guard de as_of aceitou resultado antes da duração mínima")
    latest_kickoff = pd.Timestamp(fixtures["kickoff_at"].max())
    incomplete_snapshot = replace(snapshot, as_of=latest_kickoff + timedelta(minutes=90))
    try:
        bolao.validate_observed_result_snapshot(model, observed, snapshot=incomplete_snapshot)
    except ValueError as error:
        require("duração mínima" in str(error), f"guard de conclusão falhou com erro inesperado: {error}")
    else:
        raise AssertionError("guard de as_of aceitou a última partida antes de sua duração mínima")
    partial = dict(observed)
    partial.pop(first_match_number)
    try:
        bolao.validate_observed_result_snapshot(model, partial, snapshot=snapshot)
    except ValueError:
        return
    raise AssertionError("guard temporal aceitou uma foto com jogo inicial ausente")


def validate_observed_xg_pre_match_base(model: WorldCupModel, board: bolao.GroupStageBoard) -> None:
    matches = {
        match.match_number: match
        for group_matches in board.matches_by_group.values()
        for match in group_matches
    }
    team_context: dict[str, dict[str, object]] = {}
    fixtures = model.fixtures[model.fixtures["stage_id"] == sota.GROUP_STAGE_ID].sort_values("kickoff_at")
    for game in fixtures.itertuples(index=False):
        home = sota.canonical_team(game.home_team)
        away = sota.canonical_team(game.away_team)
        context = sota.fixture_context(game, team_context, home, away)
        pre_match = sota.predict_match(model.package, home, away, context=context)
        match = matches[int(game.match_number)]
        require(match.is_observed, f"jogo CSV {match.match_number} não foi marcado como observado")
        require(
            abs(match.base_home_xg - float(pre_match["home_xg"])) < 1e-12
            and abs(match.base_away_xg - float(pre_match["away_xg"])) < 1e-12,
            f"jogo CSV {match.match_number} não preservou o xG pré-jogo/base",
        )
        xg_text, xg_reference = bolao.group_match_xg_display(match)
        require(xg_reference == "pré-jogo/base", f"jogo CSV {match.match_number} exibiu referência de xG incorreta")
        require(
            xg_text == f"{float(pre_match['home_xg']):.2f} x {float(pre_match['away_xg']):.2f}",
            f"jogo CSV {match.match_number} exibiu xG recalculado em vez do pré-jogo/base",
        )
        sota.update_team_context(team_context, home, away, game)


def validate_form_calibration(model: WorldCupModel, board: bolao.GroupStageBoard) -> None:
    form = board.form
    require(form.validation_matches == 24, "calibração temporal deveria reservar 24 jogos para holdout")
    if form.is_enabled:
        require(form.calibration_status == "enabled_validation", "forma ativa sem status de validação")
        require(form.teams, "forma ativa sem estatísticas por seleção")
        require(
            form.validation_log_likelihood > form.historical_validation_log_likelihood,
            "forma ativa sem ganho no holdout",
        )
    else:
        require(not form.teams, "fallback histórico não pode deixar multiplicadores ativos")
        require(
            form.calibration_status == "fallback_history_validation",
            f"status de calibração inesperado: {form.calibration_status}",
        )
        require(
            form.validation_log_likelihood < form.historical_validation_log_likelihood,
            "fallback histórico precisa ser sustentado pelo log-score do holdout",
        )

    group_form = bolao.build_tournament_form(model, form.observed_results)
    require(form.knockout_observed_matches == 24, "forma não registrou os 24 jogos observados do mata-mata")
    require(form.knockout_form_matches == 24, "forma não incorporou os 24 placares regulamentares")
    require(
        form.knockout_form_policy == bolao.KNOCKOUT_FORM_POLICY,
        f"política de atualização eliminatória inesperada: {form.knockout_form_policy}",
    )
    require(
        form.prior_goal_equivalents == group_form.prior_goal_equivalents,
        "mata-mata retunou o prior escolhido antes dos 16 avos",
    )
    match_counts: Counter[str] = Counter()
    goals_for: Counter[str] = Counter()
    goals_against: Counter[str] = Counter()
    for observed in board.knockout_results.values():
        match_counts.update((observed.home, observed.away))
        goals_for[observed.home] += observed.home_goals_90
        goals_against[observed.home] += observed.away_goals_90
        goals_for[observed.away] += observed.away_goals_90
        goals_against[observed.away] += observed.home_goals_90
    for team, observed_matches in match_counts.items():
        before = group_form.teams[team]
        after = form.teams[team]
        require(
            after.observed_matches == before.observed_matches + observed_matches,
            f"forma não adicionou {observed_matches} observação(ões) para {team}",
        )
        require(
            after.goals_for - before.goals_for == goals_for[team]
            and after.goals_against - before.goals_against == goals_against[team],
            f"forma de {team} misturou prorrogação ou pênaltis na taxa de 90 minutos",
        )

    home, away = sorted(board.qualified_teams)[:2]
    synthetic_form = bolao.TournamentForm(
        observed_results=form.observed_results,
        teams={
            home: bolao.TeamForm(3, 5, 2, 3.0, 3.0, 0.5, 0.5, 1.25, 0.90),
            away: bolao.TeamForm(3, 2, 4, 3.0, 3.0, 0.5, 0.5, 0.90, 1.20),
        },
        prior_goal_equivalents=3.0,
        median_current_weight=0.5,
        is_enabled=True,
        calibration_status="enabled_validation",
        validation_matches=24,
        validation_log_likelihood=0.0,
        historical_validation_log_likelihood=-1.0,
    )
    baseline = bolao.form_aware_match(model, bolao.historical_tournament_form(
        form.observed_results,
        status="fallback_history_validation",
        validation_matches=24,
        validation_log_likelihood=-1.0,
        historical_validation_log_likelihood=0.0,
    ), home, away, knockout=True)
    adjusted = bolao.form_aware_match(model, synthetic_form, home, away, knockout=True)
    require(adjusted.form_weight > 0.0, "ramo de forma ativa não aplicou peso")
    require(
        (adjusted.prediction["home_xg"], adjusted.prediction["away_xg"])
        != (baseline.prediction["home_xg"], baseline.prediction["away_xg"]),
        "ramo de forma ativa não alterou os lambdas Poisson/Dixon-Coles",
    )


def validate_fifa_tiebreaks() -> None:
    group = pd.DataFrame(
        [
            {"team": "A", "pts": 6, "gd": 4, "gf": 5, "wins": 2, "model_rating": 0.0, "fifa_rank": 20},
            {"team": "B", "pts": 6, "gd": 1, "gf": 2, "wins": 2, "model_rating": 0.0, "fifa_rank": 10},
            {"team": "C", "pts": 3, "gd": -2, "gf": 2, "wins": 1, "model_rating": 0.0, "fifa_rank": 30},
            {"team": "D", "pts": 0, "gd": -3, "gf": 1, "wins": 0, "model_rating": 0.0, "fifa_rank": 40},
        ]
    )
    ranked = sota.rank_group(
        group,
        [{"home": "A", "away": "B", "home_goals": 0, "away_goals": 1}],
    )
    require(
        list(ranked["team"][:2]) == ["B", "A"],
        "desempate FIFA deve aplicar confronto direto antes do saldo global",
    )

    thirds = pd.DataFrame(
        [
            {"team": "A", "group": "A", "pts": 3, "gd": 0, "gf": 3, "fifa_rank": 20},
            {"team": "B", "group": "B", "pts": 3, "gd": 0, "gf": 3, "fifa_rank": 10},
        ]
    )
    ranked_thirds = sota.select_best_thirds(thirds)
    order = list(ranked_thirds["team"])
    require(order.index("B") < order.index("A"), "ranking FIFA deve resolver empate terminal por ranking FIFA")

    fair_play_thirds = pd.DataFrame(
        [
            {"team": "C", "group": "C", "pts": 3, "gd": 0, "gf": 3, "fair_play_score": -1, "fifa_rank": 99},
            {"team": "D", "group": "D", "pts": 3, "gd": 0, "gf": 3, "fair_play_score": -3, "fifa_rank": 1},
        ]
    )
    require(
        list(sota.select_best_thirds(fair_play_thirds)["team"]) == ["C", "D"],
        "fair play deve anteceder ranking FIFA quando disponível",
    )

    group_missing_fair_play = pd.DataFrame(
        [
            {"team": "Ecuador", "pts": 1, "gd": 0, "gf": 1, "wins": 0, "model_rating": 0.0, "fifa_rank": 25},
            {"team": "Ghana", "pts": 1, "gd": 0, "gf": 1, "wins": 0, "model_rating": 0.0, "fifa_rank": 45},
        ]
    )
    try:
        sota.rank_group(
            group_missing_fair_play,
            [{"home": "Ecuador", "away": "Ghana", "home_goals": 1, "away_goals": 1}],
        )
    except ValueError as error:
        require("fair-play" in str(error), f"empate de grupo sem fair play falhou de forma inesperada: {error}")
    else:
        raise AssertionError("empate terminal de grupo não pode cair no ranking FIFA sem fair play")


def validate_simulated_fair_play(model: WorldCupModel) -> None:
    standings, _groups, _qualifiers, _third_order, _context = sota.simulate_group_stage(
        model.package,
        model.fixtures,
        seed=2026,
    )
    require("fair_play_score" in standings.columns, "simulação não gerou score de fair play")
    require(standings["fair_play_score"].notna().all(), "simulação deixou fair play ausente")
    require((standings["fair_play_score"] <= 0).all(), "score de fair play simulado não segue a escala de deduções FIFA")


def validate_observed_knockout_results(model: WorldCupModel, board: bolao.GroupStageBoard) -> None:
    require(bolao.stage_name("Round of 32") == "16 avos", "rótulo em português dos 16 avos está incorreto")
    observed = board.knockout_results
    expected_results = {**EXPECTED_ROUND32_RESULTS, **EXPECTED_ROUND16_RESULTS}
    require(set(observed) == set(expected_results), "snapshot não cobre os 24 jogos eliminatórios encerrados")
    for match_number, expected in expected_results.items():
        result = observed[match_number]
        actual = (
            result.home,
            result.away,
            result.home_goals_90,
            result.away_goals_90,
            result.extra_time_home_goals,
            result.extra_time_away_goals,
            result.winner,
            result.resolution,
            result.shootout_home,
            result.shootout_away,
        )
        require(actual == expected, f"resultado observado do jogo {match_number} não confere: {actual!r}")
    for eliminated_team in ("Germany", "Brazil", "Colombia"):
        try:
            bolao.build_conditioned_knockout(model, board, eliminated_team)
        except ValueError as error:
            require("eliminado" in str(error), f"eliminação observada falhou de forma inesperada: {error}")
        else:
            raise AssertionError(f"trilha condicionada ressuscitou {eliminated_team} após resultado observado")
    bracket = bolao.build_conditioned_knockout(model, board, "France")
    observed_row = bracket.loc[bracket["match_number"] == 74].iloc[0]
    require(
        (str(observed_row.winner), str(observed_row.resolution), int(observed_row.home_goals), int(observed_row.away_goals))
        == ("Paraguay", "penalties", 1, 1),
        "chave condicionada não preservou o resultado observado",
    )
    expected_round16 = {
        89: ("Paraguay", "France"),
        90: ("Canada", "Morocco"),
        91: ("Brazil", "Norway"),
        92: ("Mexico", "England"),
        93: ("Portugal", "Spain"),
        94: ("USA", "Belgium"),
        95: ("Argentina", "Egypt"),
        96: ("Switzerland", "Colombia"),
    }
    for match_number, expected in expected_round16.items():
        row = bracket.loc[bracket["match_number"] == match_number].iloc[0]
        require((str(row.home), str(row.away)) == expected, f"oitavas incorretas no jogo {match_number}")
    expected_quarterfinals = {
        97: ("France", "Morocco"),
        98: ("Spain", "Belgium"),
        99: ("Norway", "England"),
        100: ("Argentina", "Switzerland"),
    }
    for match_number, expected in expected_quarterfinals.items():
        row = bracket.loc[bracket["match_number"] == match_number].iloc[0]
        require((str(row.home), str(row.away)) == expected, f"quartas incorretas no jogo {match_number}")


def validate_neutral_order_invariance(model: WorldCupModel, board: bolao.GroupStageBoard) -> None:
    teams = sorted(board.qualified_teams)
    pairings = list(zip(teams[::2], teams[1::2]))
    require(pairings, "não há pares classificados para auditar ordem neutra")
    for home, away in pairings:
        forward = bolao.form_aware_match(model, board.form, home, away, knockout=True).prediction
        reverse = bolao.form_aware_match(model, board.form, away, home, knockout=True).prediction
        require(bool(forward.get("neutral_order_symmetrized")), f"{home} x {away} não marcou simetrização neutra")
        require(bool(reverse.get("neutral_order_symmetrized")), f"{away} x {home} não marcou simetrização neutra")
        require(
            abs(float(forward["p_home_win_90"]) - float(reverse["p_away_win_90"])) <= 1e-10
            and abs(float(forward["p_draw_90"]) - float(reverse["p_draw_90"])) <= 1e-10
            and abs(float(forward["p_away_win_90"]) - float(reverse["p_home_win_90"])) <= 1e-10,
            f"1X2 neutro mudou ao inverter {home} x {away}",
        )
        require(
            abs(float(forward["p_home_advances"]) - float(reverse["p_away_advances"])) <= 1e-10
            and abs(float(forward["p_home_advances_if_draw"]) + float(reverse["p_home_advances_if_draw"]) - 1.0) <= 1e-10,
            f"avanço de mata-mata mudou ao inverter {home} x {away}",
        )
        require(
            abs(float(forward["home_xg"]) - float(reverse["away_xg"])) <= 1e-10
            and abs(float(forward["away_xg"]) - float(reverse["home_xg"])) <= 1e-10,
            f"xG neutro mudou ao inverter {home} x {away}",
        )


def validate_knockout_policy(model: WorldCupModel, board: bolao.GroupStageBoard) -> None:
    rho = sota.dixon_coles_rho_from_package(model.package)
    teams = sorted(board.qualified_teams)
    eliminated = {
        result.away if result.winner == result.home else result.home
        for result in board.knockout_results.values()
    }
    surviving_teams = [team for team in teams if team not in eliminated]
    require(surviving_teams, "não há sobrevivente observado para testar a chave condicionada")
    for home in teams:
        for away in teams:
            if home == away:
                continue
            distribution = bolao.form_aware_match(model, board.form, home, away, knockout=True)
            resolution = bolao.knockout_resolution_policy(distribution.prediction, rho=rho)
            require(
                abs(resolution.home_advances_if_draw - float(distribution.prediction["p_home_advances_if_draw"]))
                < 1e-10,
                f"prorrogação/pênaltis alteraram P(avança | empate) para {home} x {away}",
            )
            base_prediction = sota.predict_match(model.package, home, away, knockout=True)
            base_resolution = sota.knockout_resolution_policy(base_prediction, rho=rho)
            require(
                abs(base_resolution.home_advances_if_draw - float(base_prediction["p_home_advances_if_draw"]))
                < 1e-10,
                f"pipeline alterou P(avança | empate) para {home} x {away}",
            )

    rng = random.Random(2026)
    distribution = bolao.form_aware_match(model, board.form, teams[0], teams[1], knockout=True)
    resolutions: set[str] = set()
    for _ in range(250):
        winner, home_goals, away_goals, resolution = bolao.sample_form_aware_knockout_result(
            distribution,
            teams[0],
            teams[1],
            rng,
            rho=rho,
        )
        require(winner in {teams[0], teams[1]}, "mata-mata sorteou vencedor inexistente")
        require(resolution in {"90min", "extra_time", "penalties"}, "resolução de mata-mata inválida")
        require(not (resolution == "90min" and home_goals == away_goals), "empate em 90min não foi resolvido")
        resolutions.add(resolution)
    require("90min" in resolutions, "amostra de mata-mata não gerou nenhuma vitória em 90min")

    champion = surviving_teams[0]
    bracket = bolao.build_conditioned_knockout(model, board, champion)
    expected_matches = len(model.fixtures[model.fixtures["stage_id"] > sota.GROUP_STAGE_ID])
    require(len(bracket) == expected_matches, "chave condicionada não cobriu todos os jogos oficiais")
    final = bracket[bracket["round"] == "Final"]
    require(len(final) == 1, "chave condicionada deveria ter uma final")
    require(str(final.iloc[0]["winner"]) == champion, "chave condicionada não terminou no campeão escolhido")
    require(set(bracket["resolution"]).issubset({"90min", "extra_time", "penalties"}), "chave usou resolução inválida")
    require(
        not ((bracket["resolution"] == "90min") & (bracket["home_goals"] == bracket["away_goals"])).any(),
        "chave condicionada manteve empate sem resolução em 90min",
    )


def validate_penalty_scores_include_extra_time() -> None:
    regular_matrix = np.array([[0.05, 0.01], [0.01, 0.93]], dtype=float)
    regular_blend = np.array([0.02, 0.96, 0.02], dtype=float)
    extra_time_matrix = np.array([[0.01, 0.001], [0.001, 0.988]], dtype=float)
    resolution_policy = SimpleNamespace(
        extra_time_matrix=extra_time_matrix,
        extra_time_outcomes=np.array([0.001, 0.998, 0.001], dtype=float),
        home_penalty_probability=0.80,
    )
    selected = bolao.best_advancing_score(
        regular_matrix,
        regular_blend,
        {},
        "HOME",
        "HOME",
        "AWAY",
        resolution_policy=resolution_policy,
    )
    require(selected.resolution == "penalties", "cenário dirigido não selecionou pênaltis")
    require(
        (selected.home_goals, selected.away_goals) == (2, 2),
        "caminho condicionado de pênaltis não acumulou o placar da prorrogação",
    )

    distribution = bolao.MatchDistribution(
        prediction={},
        matrix=regular_matrix,
        blend=np.array([0.0, 1.0, 0.0], dtype=float),
        poisson=regular_blend,
        form_weight=0.0,
    )
    sample_policy = SimpleNamespace(
        extra_time_matrix=extra_time_matrix,
        extra_time_outcomes=np.array([0.001, 0.998, 0.001], dtype=float),
        home_penalty_probability=1.0,
    )
    with (
        patch.object(bolao, "sample_score_for_outcome", return_value=(1, 1)),
        patch.object(bolao, "sample_score_from_matrix", return_value=(1, 1)),
        patch.object(bolao, "knockout_resolution_policy", return_value=sample_policy),
    ):
        winner, home_goals, away_goals, resolution = bolao.sample_form_aware_knockout_result(
            distribution,
            "HOME",
            "AWAY",
            random.Random(1),
            rho=0.0,
        )
    require(winner == "HOME" and resolution == "penalties", "amostra dirigida não resolveu pênaltis para o mandante")
    require(
        (home_goals, away_goals) == (2, 2),
        "amostra de pênaltis não acumulou o placar de 90min mais prorrogação",
    )


def ranking_signature(ranking: list[bolao.ChampionOption]) -> tuple[tuple[int, str, int], ...]:
    return tuple((option.rank, option.team, option.wins) for option in ranking)


def validate_monte_carlo(model: WorldCupModel, board: bolao.GroupStageBoard, runs: int, seed: int) -> None:
    top_n = len(board.qualified_teams)
    first = bolao.build_champion_ranking(
        model,
        board,
        runs=runs,
        seed=seed,
        top_n=top_n,
        workers=1,
    )
    second = bolao.build_champion_ranking(
        model,
        board,
        runs=runs,
        seed=seed,
        top_n=top_n,
        workers=1,
    )
    require(first, "Monte Carlo nao retornou nenhum campeao")
    require(ranking_signature(first) == ranking_signature(second), "Monte Carlo nao foi reproduzivel para a mesma seed")
    require(sum(option.wins for option in first) == runs, "ranking Monte Carlo nao contabiliza todas as simulacoes")
    require(
        all(option.team in board.qualified_teams for option in first),
        "Monte Carlo escolheu campeao fora dos classificados da fase fixa",
    )
    observed_losers = {
        result.away if result.winner == result.home else result.home
        for result in board.knockout_results.values()
    }
    require(
        not observed_losers.intersection(option.team for option in first),
        "Monte Carlo manteve campeão já eliminado em resultado observado",
    )
    require(
        [option.rank for option in first] == list(range(1, len(first) + 1)),
        "ranking Monte Carlo contem posicoes invalidas",
    )


def validate_knockout_calibration_audit() -> None:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="bolao_knockout_qa_", dir=artifacts) as temporary_dir:
        base = Path(temporary_dir)
        output = base / "calibration.json"
        audit_csv = base / "round32.csv"
        round16_csv = base / "round16.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bolao_knockout_calibration.py"),
                "--out",
                str(output),
                "--audit-csv",
                str(audit_csv),
                "--round16-csv",
                str(round16_csv),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        terminal_output = result.stdout + result.stderr
        require(result.returncode == 0, f"auditoria dos 16 avos falhou:\n{terminal_output}")
        require(output.is_file() and audit_csv.is_file() and round16_csv.is_file(), "auditoria não gerou os artefatos")
        report = json.loads(output.read_text(encoding="utf-8"))
        audit_rows = pd.read_csv(audit_csv)
        round16_rows = pd.read_csv(round16_csv)
    require(report["approved"] is True, "gate de calibração dos 16 avos não passou")
    require(len(audit_rows) == 16, "auditoria não cobriu os 16 jogos observados")
    require(len(round16_rows) == 8, "auditoria não resolveu os oito confrontos das oitavas")
    require(
        report["group_validated_form"]["log_loss_1x2"]
        <= report["historical_baseline"]["log_loss_1x2"],
        "forma dos grupos piorou o log-loss externo dos 16 avos",
    )
    require(
        report["decision"]["retune_xgboost_or_global_hybrid_weights"] is False,
        "amostra curta retunou pesos globais do híbrido",
    )
    require(
        report["current_form_update"]["extra_time_goals_appended"] == 0
        and report["current_form_update"]["shootout_kicks_appended"] == 0,
        "auditoria incluiu prorrogação ou cobranças na taxa regulamentar",
    )
    require(
        report["current_form_update"]["max_abs_round16_advance_probability_delta"] <= 0.05,
        "uma única rodada deslocou excessivamente a chance de avanço nas oitavas",
    )


def validate_round16_calibration_audit() -> None:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="bolao_round16_qa_", dir=artifacts) as temporary_dir:
        base = Path(temporary_dir)
        output = base / "calibration.json"
        audit_csv = base / "round16.csv"
        quarterfinal_csv = base / "quarterfinals.csv"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bolao_round16_calibration.py"),
                "--out",
                str(output),
                "--audit-csv",
                str(audit_csv),
                "--quarterfinal-csv",
                str(quarterfinal_csv),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        terminal_output = result.stdout + result.stderr
        require(result.returncode == 0, f"auditoria das oitavas falhou:\n{terminal_output}")
        require(
            output.is_file() and audit_csv.is_file() and quarterfinal_csv.is_file(),
            "auditoria das oitavas não gerou os artefatos",
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        audit_rows = pd.read_csv(audit_csv)
        quarterfinal_rows = pd.read_csv(quarterfinal_csv)
    require(report["approved"] is True, "gate de calibração das oitavas não passou")
    require(len(audit_rows) == 8, "auditoria não cobriu os oito jogos das oitavas")
    require(len(quarterfinal_rows) == 4, "auditoria não resolveu os quatro confrontos das quartas")
    require(
        report["pre_round16_form"]["log_loss_1x2"] > report["historical_baseline"]["log_loss_1x2"],
        "relatório não registrou a leve piora externa da forma nas oitavas",
    )
    require(
        report["operational_margin"]["all_metrics_within_margin"] is True,
        "forma ultrapassou a margem operacional nas oitavas",
    )
    require(
        report["decision"]["retune_xgboost_or_global_hybrid_weights"] is False,
        "amostra de oito jogos retunou pesos globais do híbrido",
    )
    require(
        report["current_form_update"]["extra_time_goals_appended"] == 0
        and report["current_form_update"]["shootout_kicks_appended"] == 0,
        "auditoria das oitavas incluiu prorrogação ou cobranças na taxa regulamentar",
    )
    require(
        report["current_form_update"]["max_abs_quarterfinal_advance_probability_delta"] <= 0.05,
        "uma única rodada deslocou excessivamente a chance de avanço nas quartas",
    )
    require(
        report["hard_gates"]["switzerland_colombia_correction_verified"] is True,
        "auditoria não preservou a vitória suíça por 4-3 nos pênaltis",
    )


def validate_mc_stability_audit(board: bolao.GroupStageBoard) -> None:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="bolao_mc_qa_", dir=artifacts) as temporary_dir:
        output = Path(temporary_dir) / "bolao_mc_stability.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bolao_mc_stability.py"),
                "--runs",
                "16,32",
                "--top",
                "5",
                "--max-probability-delta",
                "1.0",
                "--min-top-overlap",
                "0.0",
                "--independent-seeds",
                "2026,2027",
                "--independent-runs",
                "32",
                "--max-independent-probability-delta",
                "1.0",
                "--min-independent-top-overlap",
                "0.0",
                "--max-independent-sampling-z",
                "999",
                "--out",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        terminal_output = result.stdout + result.stderr
        require(result.returncode == 0, f"auditoria MC curta falhou:\n{terminal_output}")
        require(output.is_file(), "auditoria MC curta não gerou JSON")
        report = json.loads(output.read_text(encoding="utf-8"))
    require(report["simulation_scope"]["group_stage"] == "fixed_board", "auditoria MC não fixou a fase de grupos")
    require(report["simulation_scope"]["current_tournament_form"] == "included", "auditoria MC não incluiu a forma")
    require(
        report["simulation_scope"]["knockout"] == "observed_results_locked_then_form_aware_hybrid_sampled",
        "auditoria MC não travou resultados observados antes do mata-mata simulado",
    )
    require(
        report["simulation_scope"]["sampling"] == "nested_prefixes_one_seed_plus_independent_seeds",
        "auditoria MC não incluiu amostras de seeds independentes",
    )
    require(report["fixed_group_stage"]["is_fixed"] is True, "relatório MC não marcou grupos fixos")
    require(
        report["fixed_group_stage"]["form"]["knockout_form_policy"] == bolao.KNOCKOUT_FORM_POLICY
        and report["fixed_group_stage"]["form"]["knockout_form_matches"] == 24,
        "relatório MC não registrou a atualização regulamentar do mata-mata",
    )
    expected_locked_results = [
        {
            "match_number": int(result.match_number),
            "home": result.home,
            "away": result.away,
            "winner": result.winner,
            "resolution": result.resolution,
        }
        for result in sorted(board.knockout_results.values(), key=lambda item: item.match_number)
    ]
    require(
        report["fixed_group_stage"]["locked_knockout_results"] == expected_locked_results,
        "auditoria MC não registrou os 24 resultados observados da chave",
    )
    require(
        report["uncertainty"]["scope"] == "monte_carlo_sampling_error_only",
        "relatório MC apresentou IC como incerteza total do modelo",
    )
    require(
        report["uncertainty"]["interval"] == "wilson_score_95_percent",
        "relatório MC não usa intervalo Wilson limitado",
    )
    fingerprints = report.get("runtime_fingerprints", {})
    for name in (
        "bolao",
        "worldcup_model",
        "sota_pipeline",
        "model_package",
        "runtime_cache",
        "observed_results",
        "observed_snapshot",
        "observed_knockout_results",
        "stability_audit",
    ):
        require(name in fingerprints and fingerprints[name].get("sha256"), f"relatório MC sem fingerprint de {name}")
    lower, upper = bolao.mc_sampling_interval_95(0.002, 1000)
    require(0.0 <= lower <= upper <= 1.0, "intervalo MC Wilson saiu dos limites de probabilidade")
    require("model_misspecification" in report["uncertainty"]["does_not_measure"], "relatório MC omitiu limite do IC")
    require(report["gate"]["passed"] is True, "gate da auditoria MC curta não passou")
    seed_audit = report["independent_seed_audit"]
    require(seed_audit["runs_per_seed"] == 32, "auditoria MC curta não respeitou o volume por seed")
    require(seed_audit["seeds"] == [2026, 2027], "auditoria MC curta não registrou as seeds independentes")
    require(seed_audit["gate"]["passed"] is True, "gate de seeds independentes não passou")
    require("max_two_sample_z" in seed_audit["gate"]["checks"], "gate independente não mediu compatibilidade amostral")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA semântico do utilitário de console do bolão.")
    parser.add_argument("--runs", type=int, default=100, help="Quantidade curta de Copas para o smoke Monte Carlo.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed usada para verificar repetibilidade do Monte Carlo.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(args.runs > 0, "--runs deve ser positivo")
    rows = csv_rows()
    model = WorldCupModel()
    validate_official_knockout_bracket(model)
    board = validate_group_stage(model)
    validate_annex_c_third_place_matrix(model, board)
    validate_snapshot_guard(model, board.form.observed_results)
    validate_observed_xg_pre_match_base(model, board)
    validate_form_calibration(model, board)
    validate_fifa_tiebreaks()
    validate_simulated_fair_play(model)
    validate_observed_knockout_results(model, board)
    validate_knockout_calibration_audit()
    validate_round16_calibration_audit()
    validate_neutral_order_invariance(model, board)
    validate_knockout_policy(model, board)
    validate_penalty_scores_include_extra_time()
    validate_monte_carlo(model, board, int(args.runs), int(args.seed))
    validate_mc_stability_audit(board)
    print(
        "[bolao-qa] CSV OK: "
        f"{len(rows)}/{EXPECTED_GROUP_MATCHES} jogos | grupos OK: {len(board.qualified_teams)} classificados | "
        f"mata-mata observado: {len(board.knockout_results)} jogos | "
        f"forma temporal OK: {board.form.calibration_status} | "
        f"desempates/mata-mata/Monte Carlo OK: {args.runs} runs reproduziveis"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

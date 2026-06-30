from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table

from arena_ai.worldcup_model import SOTA_ROOT, WorldCupModel, sota


GROUPS = tuple("ABCDEFGHIJKL")
OBSERVED_GROUP_RESULTS_PATH = SOTA_ROOT / "data" / "observed" / "worldcup_2026_group_stage_results.csv"
OBSERVED_SNAPSHOT_METADATA_PATH = SOTA_ROOT / "data" / "observed" / "worldcup_2026_group_stage_snapshot.json"
OBSERVED_KNOCKOUT_RESULTS_PATH = SOTA_ROOT / "data" / "observed" / "worldcup_2026_knockout_results.csv"
OBSERVED_SNAPSHOT_SCHEMA_VERSION = 1
OBSERVED_SNAPSHOT_KIND = "manual_local"
OBSERVED_GROUP_MATCH_MINIMUM_DURATION = pd.Timedelta(hours=2)
KnockoutResolutionPolicy = sota.KnockoutResolutionPolicy
CURRENT_FORM_PRIOR_GOAL_GRID = tuple(np.arange(0.5, 8.01, 0.25))
CURRENT_FORM_FACTOR_FLOOR = 0.55
CURRENT_FORM_FACTOR_CEILING = 1.85
CURRENT_FORM_MIN_XG = 0.15
CURRENT_FORM_MAX_XG = 5.5
CURRENT_FORM_MIN_OBSERVED_MATCHES = 72
CURRENT_FORM_VALIDATION_FRACTION = 1.0 / 3.0
CURRENT_FORM_REQUIRED_LOG_SCORE_GAIN_PER_MATCH = 0.01
HISTORICAL_FORM_PRIOR_GOALS = 1_000_000.0

TEAM_DISPLAY_NAMES_PT = {
    "ALG": "Argélia",
    "ARG": "Argentina",
    "AUS": "Austrália",
    "AUT": "Áustria",
    "BEL": "Bélgica",
    "BIH": "Bósnia e Herzegovina",
    "BRA": "Brasil",
    "CAN": "Canadá",
    "CIV": "Costa do Marfim",
    "COD": "RD Congo",
    "COL": "Colômbia",
    "CPV": "Cabo Verde",
    "CRO": "Croácia",
    "CUR": "Curaçao",
    "CZE": "Tchéquia",
    "ECU": "Equador",
    "EGY": "Egito",
    "ENG": "Inglaterra",
    "ESP": "Espanha",
    "FRA": "França",
    "GER": "Alemanha",
    "GHA": "Gana",
    "HAI": "Haiti",
    "IRN": "Irã",
    "IRQ": "Iraque",
    "JOR": "Jordânia",
    "JPN": "Japão",
    "KOR": "Coreia do Sul",
    "KSA": "Arábia Saudita",
    "MAR": "Marrocos",
    "MEX": "México",
    "NED": "Holanda",
    "NOR": "Noruega",
    "NZL": "Nova Zelândia",
    "PAN": "Panamá",
    "PAR": "Paraguai",
    "POR": "Portugal",
    "QAT": "Catar",
    "RSA": "África do Sul",
    "SCO": "Escócia",
    "SEN": "Senegal",
    "SUI": "Suíça",
    "SWE": "Suécia",
    "TUN": "Tunísia",
    "TUR": "Turquia",
    "URU": "Uruguai",
    "USA": "Estados Unidos",
    "UZB": "Uzbequistão",
}


@dataclass(frozen=True)
class GroupMatch:
    match_number: int
    group: str
    home: str
    away: str
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float
    base_home_xg: float
    base_away_xg: float
    home_probability: float
    draw_probability: float
    away_probability: float
    outcome_probability: float
    score_probability: float
    is_observed: bool
    form_weight: float


@dataclass(frozen=True)
class GroupStageBoard:
    matches_by_group: dict[str, list[GroupMatch]]
    standings: pd.DataFrame
    qualified_teams: set[str]
    best_third_teams: set[str]
    qualifiers: dict[str, str]
    third_order: list[str]
    team_context: dict[str, dict[str, object]]
    policy: dict[str, object]
    form: TournamentForm
    snapshot: ObservedSnapshotMetadata
    knockout_results: dict[int, ObservedKnockoutResult]


@dataclass(frozen=True)
class ChampionOption:
    rank: int
    team: str
    wins: int
    probability: float


@dataclass(frozen=True)
class ScoreCandidate:
    home_goals: int
    away_goals: int
    outcome: int
    resolution: str
    score_probability: float
    path_probability: float


@dataclass(frozen=True)
class ObservedResult:
    match_number: int
    group: str
    home: str
    away: str
    home_goals: int
    away_goals: int
    source: str


@dataclass(frozen=True)
class ObservedKnockoutResult:
    match_number: int
    round_name: str
    home: str
    away: str
    home_goals_90: int
    away_goals_90: int
    extra_time_home_goals: int
    extra_time_away_goals: int
    winner: str
    resolution: str
    shootout_home: int | None
    shootout_away: int | None
    source: str

    @property
    def home_goals(self) -> int:
        return self.home_goals_90 + self.extra_time_home_goals

    @property
    def away_goals(self) -> int:
        return self.away_goals_90 + self.extra_time_away_goals


@dataclass(frozen=True)
class ObservedSnapshotMetadata:
    schema_version: int
    snapshot_kind: str
    as_of: datetime
    source_label: str
    row_source: str
    results_file: str
    result_count: int
    results_sha256: str
    official_source: bool
    fair_play_scores: dict[str, float]

    @property
    def as_of_utc_text(self) -> str:
        return self.as_of.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TeamForm:
    observed_matches: int
    goals_for: int
    goals_against: int
    expected_goals_for: float
    expected_goals_against: float
    attack_weight: float
    defense_weight: float
    attack_multiplier: float
    defense_multiplier: float


@dataclass(frozen=True)
class TournamentForm:
    observed_results: dict[int, ObservedResult]
    teams: dict[str, TeamForm]
    prior_goal_equivalents: float
    median_current_weight: float
    is_enabled: bool
    calibration_status: str
    validation_matches: int
    validation_log_likelihood: float
    historical_validation_log_likelihood: float


@dataclass(frozen=True)
class MatchDistribution:
    prediction: dict[str, float]
    matrix: np.ndarray
    blend: np.ndarray
    poisson: np.ndarray
    form_weight: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_observed_snapshot_metadata() -> ObservedSnapshotMetadata:
    if not OBSERVED_SNAPSHOT_METADATA_PATH.is_file():
        raise FileNotFoundError(f"metadata do snapshot manual ausente: {OBSERVED_SNAPSHOT_METADATA_PATH}")
    try:
        payload = json.loads(OBSERVED_SNAPSHOT_METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("metadata do snapshot manual não é JSON válido") from error
    if not isinstance(payload, dict):
        raise ValueError("metadata do snapshot manual precisa ser um objeto JSON")

    required = {
        "schema_version",
        "snapshot_kind",
        "as_of",
        "source_label",
        "row_source",
        "results_file",
        "result_count",
        "results_sha256",
        "official_source",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"metadata do snapshot manual sem campos obrigatórios: {', '.join(missing)}")
    if int(payload["schema_version"]) != OBSERVED_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"versão de metadata de snapshot não suportada: {payload['schema_version']!r}")
    if str(payload["snapshot_kind"]).strip() != OBSERVED_SNAPSHOT_KIND:
        raise ValueError("metadata precisa identificar o snapshot como manual_local")
    if payload["official_source"] is not False:
        raise ValueError("metadata manual precisa declarar official_source=false")

    source_label = str(payload["source_label"]).strip()
    row_source = str(payload["row_source"]).strip()
    if not source_label or not row_source:
        raise ValueError("metadata manual sem proveniência local explícita")
    if "fifa" in f"{source_label} {row_source}".casefold():
        raise ValueError("snapshot manual não pode se apresentar como fonte FIFA")
    if str(payload["results_file"]).strip() != OBSERVED_GROUP_RESULTS_PATH.name:
        raise ValueError("metadata aponta para um CSV de resultados diferente do carregado pelo bolão")
    if isinstance(payload["result_count"], bool) or int(payload["result_count"]) < 0:
        raise ValueError("metadata manual contém result_count inválido")

    as_of_text = str(payload["as_of"]).strip()
    try:
        as_of = datetime.fromisoformat(as_of_text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("metadata manual contém as_of inválido") from error
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("metadata manual precisa informar as_of com timezone")

    results_sha256 = str(payload["results_sha256"]).strip().lower()
    if len(results_sha256) != 64 or any(character not in "0123456789abcdef" for character in results_sha256):
        raise ValueError("metadata manual contém results_sha256 inválido")
    fair_play_payload = payload.get("fair_play_scores", {})
    if not isinstance(fair_play_payload, dict):
        raise ValueError("metadata manual contém fair_play_scores inválido")
    fair_play_scores: dict[str, float] = {}
    for raw_team, raw_score in fair_play_payload.items():
        team = sota.canonical_team(str(raw_team))
        if not team or team in fair_play_scores:
            raise ValueError("metadata manual contém equipe duplicada em fair_play_scores")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise ValueError(f"fair_play_scores inválido para {raw_team!r}") from error
        if not np.isfinite(score) or score > 0 or not score.is_integer():
            raise ValueError(f"fair_play_scores inválido para {raw_team!r}")
        fair_play_scores[team] = score
    return ObservedSnapshotMetadata(
        schema_version=int(payload["schema_version"]),
        snapshot_kind=OBSERVED_SNAPSHOT_KIND,
        as_of=as_of.astimezone(timezone.utc),
        source_label=source_label,
        row_source=row_source,
        results_file=OBSERVED_GROUP_RESULTS_PATH.name,
        result_count=int(payload["result_count"]),
        results_sha256=results_sha256,
        official_source=False,
        fair_play_scores=fair_play_scores,
    )


def validate_observed_fair_play_scores(model: WorldCupModel, snapshot: ObservedSnapshotMetadata) -> dict[str, float]:
    """Validate a full manual fair-play map when the snapshot provides one."""
    if not snapshot.fair_play_scores:
        return {}
    expected_teams = {str(row.team_key) for row in model.squad.itertuples(index=False)}
    observed_teams = set(snapshot.fair_play_scores)
    unknown = sorted(observed_teams.difference(expected_teams))
    missing = sorted(expected_teams.difference(observed_teams))
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"equipes desconhecidas: {', '.join(unknown)}")
        if missing:
            details.append(f"equipes sem fair play: {', '.join(missing)}")
        raise ValueError("fair_play_scores precisa cobrir todas as seleções da fase de grupos; " + "; ".join(details))
    return dict(snapshot.fair_play_scores)


def load_observed_group_results(
    model: WorldCupModel,
    *,
    snapshot: ObservedSnapshotMetadata | None = None,
) -> dict[int, ObservedResult]:
    snapshot = snapshot or load_observed_snapshot_metadata()
    if not OBSERVED_GROUP_RESULTS_PATH.is_file():
        raise FileNotFoundError(f"CSV do snapshot manual ausente: {OBSERVED_GROUP_RESULTS_PATH}")
    actual_sha256 = sha256_file(OBSERVED_GROUP_RESULTS_PATH)
    if actual_sha256 != snapshot.results_sha256:
        raise ValueError(
            "CSV de resultados não confere com o hash da metadata do snapshot; "
            "atualize a metadata junto com a foto manual"
        )

    try:
        frame = pd.read_csv(OBSERVED_GROUP_RESULTS_PATH)
    except pd.errors.EmptyDataError as error:
        raise ValueError("CSV de resultados está vazio") from error
    required_columns = {"match_number", "group", "home_team", "away_team", "home_goals", "away_goals", "source"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"CSV de resultados sem colunas obrigatórias: {', '.join(missing_columns)}")

    fixtures = model.fixtures[model.fixtures["stage_id"] == sota.GROUP_STAGE_ID]
    fixtures_by_number = {int(row.match_number): row for row in fixtures.itertuples(index=False)}
    observed: dict[int, ObservedResult] = {}
    for row in frame.itertuples(index=False):
        match_number = int(row.match_number)
        if match_number in observed:
            raise ValueError(f"CSV de resultados contém jogo duplicado: {match_number}")
        fixture = fixtures_by_number.get(match_number)
        if fixture is None:
            raise ValueError(f"CSV de resultados referencia jogo de grupos inexistente: {match_number}")
        home = sota.canonical_team(fixture.home_team)
        away = sota.canonical_team(fixture.away_team)
        if sota.canonical_team(str(row.home_team)) != home or sota.canonical_team(str(row.away_team)) != away:
            raise ValueError(f"CSV de resultados não confere com o confronto oficial do jogo {match_number}")
        group = str(row.group).strip().upper()
        if group != str(fixture.group):
            raise ValueError(f"CSV de resultados informa grupo incorreto no jogo {match_number}: {group}")
        home_goals = parse_observed_goal(row.home_goals, match_number)
        away_goals = parse_observed_goal(row.away_goals, match_number)
        source = str(row.source).strip()
        if not source or source.lower() == "nan":
            raise ValueError(f"CSV de resultados sem proveniência no jogo {match_number}")
        if source != snapshot.row_source:
            raise ValueError(
                f"CSV de resultados usa proveniência diferente da metadata no jogo {match_number}: {source!r}"
            )
        observed[match_number] = ObservedResult(
            match_number=match_number,
            group=group,
            home=home,
            away=away,
            home_goals=home_goals,
            away_goals=away_goals,
            source=source,
        )
    if len(observed) != snapshot.result_count:
        raise ValueError(
            "CSV de resultados não confere com result_count da metadata do snapshot: "
            f"{len(observed)} != {snapshot.result_count}"
        )
    validate_observed_result_snapshot(model, observed, snapshot=snapshot)
    return observed


def validate_observed_result_snapshot(
    model: WorldCupModel,
    observed_results: dict[int, ObservedResult],
    *,
    snapshot: ObservedSnapshotMetadata | None = None,
) -> None:
    """Reject a partial CSV that skips an earlier group-stage result.

    A current-Cup calibration can only consume information available at one
    point in time. Requiring a chronological prefix prevents future scores
    from leaking into a snapshot that still has earlier matches missing.
    """
    snapshot = snapshot or load_observed_snapshot_metadata()
    if not observed_results:
        return
    group_games = model.fixtures[model.fixtures["stage_id"] == sota.GROUP_STAGE_ID].sort_values(
        ["kickoff_at", "match_number"]
    )
    ordered_numbers = [int(game.match_number) for game in group_games.itertuples(index=False)]
    fixtures_by_number = {int(game.match_number): game for game in group_games.itertuples(index=False)}
    as_of = pd.Timestamp(snapshot.as_of)
    for match_number in observed_results:
        kickoff_at = pd.Timestamp(fixtures_by_number[match_number].kickoff_at)
        if kickoff_at.tzinfo is None:
            kickoff_at = kickoff_at.tz_localize("UTC")
        completed_at = kickoff_at + OBSERVED_GROUP_MATCH_MINIMUM_DURATION
        if as_of < completed_at:
            raise ValueError(
                "snapshot manual contém resultado antes da duração mínima de uma partida de grupos: "
                f"jogo {match_number} ({completed_at.isoformat()} > {snapshot.as_of_utc_text})"
            )
    index_by_number = {match_number: index for index, match_number in enumerate(ordered_numbers)}
    last_observed_index = max(index_by_number[match_number] for match_number in observed_results)
    missing_before_latest = [
        match_number
        for match_number in ordered_numbers[: last_observed_index + 1]
        if match_number not in observed_results
    ]
    if missing_before_latest:
        preview = ", ".join(str(match_number) for match_number in missing_before_latest[:8])
        raise ValueError(
            "CSV de resultados não representa uma foto cronológica completa; "
            f"faltam jogos anteriores ao último resultado informado: {preview}"
        )


def parse_observed_goal(value: object, match_number: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Placar inválido no jogo {match_number}") from error
    if numeric < 0 or not numeric.is_integer():
        raise ValueError(f"Placar inválido no jogo {match_number}")
    return int(numeric)


def parse_optional_observed_goal(value: object, match_number: int, field: str) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} inválido no jogo {match_number}") from error
    if numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{field} inválido no jogo {match_number}")
    return int(numeric)


def load_observed_knockout_results(
    model: WorldCupModel,
    board: GroupStageBoard,
) -> dict[int, ObservedKnockoutResult]:
    """Load only completed knockout matches and require their real bracket inputs."""
    if not OBSERVED_KNOCKOUT_RESULTS_PATH.is_file():
        return {}
    try:
        frame = pd.read_csv(OBSERVED_KNOCKOUT_RESULTS_PATH)
    except pd.errors.EmptyDataError as error:
        raise ValueError("CSV de resultados do mata-mata está vazio") from error
    required_columns = {
        "match_number",
        "round",
        "home_team",
        "away_team",
        "home_goals_90",
        "away_goals_90",
        "extra_time_home_goals",
        "extra_time_away_goals",
        "winner",
        "resolution",
        "shootout_home",
        "shootout_away",
        "source",
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"CSV de resultados do mata-mata sem colunas obrigatórias: {', '.join(missing_columns)}")

    fixtures = model.fixtures[model.fixtures["stage_id"] > sota.GROUP_STAGE_ID].sort_values("match_number")
    fixture_by_number = {int(row.match_number): row for row in fixtures.itertuples(index=False)}
    loaded: dict[int, ObservedKnockoutResult] = {}
    for row in frame.itertuples(index=False):
        match_number = int(row.match_number)
        if match_number in loaded:
            raise ValueError(f"CSV de resultados do mata-mata contém jogo duplicado: {match_number}")
        fixture = fixture_by_number.get(match_number)
        if fixture is None:
            raise ValueError(f"CSV de resultados do mata-mata referencia jogo inexistente: {match_number}")
        round_name = str(row.round).strip()
        if round_name != str(fixture.stage):
            raise ValueError(f"CSV do mata-mata informa fase incorreta no jogo {match_number}: {round_name}")
        home = sota.canonical_team(str(row.home_team))
        away = sota.canonical_team(str(row.away_team))
        winner = sota.canonical_team(str(row.winner))
        if not home or not away or home == away or winner not in {home, away}:
            raise ValueError(f"CSV do mata-mata contém equipes inválidas no jogo {match_number}")
        resolution = str(row.resolution).strip()
        if resolution not in {"90min", "extra_time", "penalties"}:
            raise ValueError(f"CSV do mata-mata contém resolução inválida no jogo {match_number}")
        source = str(row.source).strip()
        if not source or source.lower() == "nan":
            raise ValueError(f"CSV do mata-mata sem proveniência no jogo {match_number}")
        home_goals_90 = parse_observed_goal(row.home_goals_90, match_number)
        away_goals_90 = parse_observed_goal(row.away_goals_90, match_number)
        extra_time_home_goals = parse_observed_goal(row.extra_time_home_goals, match_number)
        extra_time_away_goals = parse_observed_goal(row.extra_time_away_goals, match_number)
        shootout_home = parse_optional_observed_goal(row.shootout_home, match_number, "shootout_home")
        shootout_away = parse_optional_observed_goal(row.shootout_away, match_number, "shootout_away")
        full_home_goals = home_goals_90 + extra_time_home_goals
        full_away_goals = away_goals_90 + extra_time_away_goals
        if resolution == "90min":
            if home_goals_90 == away_goals_90 or extra_time_home_goals or extra_time_away_goals:
                raise ValueError(f"resultado em 90min inválido no jogo {match_number}")
            if shootout_home is not None or shootout_away is not None:
                raise ValueError(f"resultado em 90min não pode conter disputa de pênaltis no jogo {match_number}")
            expected_winner = home if home_goals_90 > away_goals_90 else away
        elif resolution == "extra_time":
            if home_goals_90 != away_goals_90 or full_home_goals == full_away_goals:
                raise ValueError(f"resultado na prorrogação inválido no jogo {match_number}")
            if shootout_home is not None or shootout_away is not None:
                raise ValueError(f"resultado na prorrogação não pode conter disputa de pênaltis no jogo {match_number}")
            expected_winner = home if full_home_goals > full_away_goals else away
        else:
            if home_goals_90 != away_goals_90 or full_home_goals != full_away_goals:
                raise ValueError(f"resultado nos pênaltis inválido no jogo {match_number}")
            if shootout_home is None or shootout_away is None or shootout_home == shootout_away:
                raise ValueError(f"disputa de pênaltis inválida no jogo {match_number}")
            expected_winner = home if shootout_home > shootout_away else away
        if winner != expected_winner:
            raise ValueError(f"vencedor do mata-mata não confere com o placar no jogo {match_number}")
        loaded[match_number] = ObservedKnockoutResult(
            match_number=match_number,
            round_name=round_name,
            home=home,
            away=away,
            home_goals_90=home_goals_90,
            away_goals_90=away_goals_90,
            extra_time_home_goals=extra_time_home_goals,
            extra_time_away_goals=extra_time_away_goals,
            winner=winner,
            resolution=resolution,
            shootout_home=shootout_home,
            shootout_away=shootout_away,
            source=source,
        )

    knockout_games = list(fixtures.itertuples(index=False))
    round32_slots = [
        slot
        for game in knockout_games
        if int(game.stage_id) == 2
        for slot in sota.parse_match_label(game.match_label)
        if slot.startswith("3")
    ]
    third_slot_assignment = sota.assign_third_slots(round32_slots, board.third_order)
    winners: dict[int, str] = {}
    runners_up: dict[int, str] = {}
    for match_number in sorted(loaded):
        fixture = fixture_by_number[match_number]
        left_slot, right_slot = sota.parse_match_label(fixture.match_label)
        try:
            expected_home = sota.resolve_bracket_slot(left_slot, board.qualifiers, winners, runners_up, third_slot_assignment)
            expected_away = sota.resolve_bracket_slot(right_slot, board.qualifiers, winners, runners_up, third_slot_assignment)
        except (KeyError, ValueError) as error:
            raise ValueError(
                "CSV do mata-mata não representa uma foto de chave resolvível; "
                f"faltam resultados anteriores ao jogo {match_number}"
            ) from error
        observed = loaded[match_number]
        if (observed.home, observed.away) != (expected_home, expected_away):
            raise ValueError(
                f"CSV do mata-mata não confere com a chave no jogo {match_number}: "
                f"esperado {expected_home} x {expected_away}"
            )
        loser = observed.away if observed.winner == observed.home else observed.home
        winners[match_number] = observed.winner
        runners_up[match_number] = loser
    return loaded


def build_tournament_form(model: WorldCupModel, observed_results: dict[int, ObservedResult]) -> TournamentForm:
    rho = sota.dixon_coles_rho_from_package(model.package)
    classifier_weight = float(sota.simulation_policy_from_package(model.package)["classifier_weight"])
    team_context: dict[str, dict[str, object]] = {}
    observed_rows: list[tuple[str, str, dict[str, float], ObservedResult]] = []
    group_games = model.fixtures[model.fixtures["stage_id"] == sota.GROUP_STAGE_ID].sort_values("kickoff_at")

    for game in group_games.itertuples(index=False):
        home = sota.canonical_team(game.home_team)
        away = sota.canonical_team(game.away_team)
        context = sota.fixture_context(game, team_context, home, away)
        prediction = sota.predict_match(model.package, home, away, context=context)
        observed = observed_results.get(int(game.match_number))
        if observed is not None:
            observed_rows.append((home, away, prediction, observed))
        # Travel and rest are schedule facts, independent of whether the score is already reported.
        sota.update_team_context(team_context, home, away, game)

    if not observed_rows:
        return historical_tournament_form(
            observed_results,
            status="no_observed_results",
            validation_matches=0,
            validation_log_likelihood=0.0,
            historical_validation_log_likelihood=0.0,
        )

    if len(observed_rows) < CURRENT_FORM_MIN_OBSERVED_MATCHES:
        return historical_tournament_form(
            observed_results,
            status="insufficient_observed_matches",
            validation_matches=0,
            validation_log_likelihood=0.0,
            historical_validation_log_likelihood=0.0,
        )

    validation_count = max(1, int(round(len(observed_rows) * CURRENT_FORM_VALIDATION_FRACTION)))
    training_count = len(observed_rows) - validation_count
    if training_count < 1:
        return historical_tournament_form(
            observed_results,
            status="insufficient_training_matches",
            validation_matches=0,
            validation_log_likelihood=0.0,
            historical_validation_log_likelihood=0.0,
        )

    training_rows = observed_rows[:training_count]
    validation_rows = observed_rows[training_count:]
    candidate_rows: list[tuple[float, float]] = []
    for prior_goals in CURRENT_FORM_PRIOR_GOAL_GRID:
        log_likelihood, _stats = prequential_form_log_likelihood(
            training_rows,
            float(prior_goals),
            rho,
            classifier_weight=classifier_weight,
        )
        candidate_rows.append((float(prior_goals), log_likelihood))

    prior_goals, _training_log_likelihood = max(candidate_rows, key=lambda row: (row[1], -row[0]))
    training_stats = accumulate_form_stats(training_rows)
    validation_log_likelihood, _validation_stats = prequential_form_log_likelihood(
        validation_rows,
        prior_goals,
        rho,
        classifier_weight=classifier_weight,
        initial_stats=training_stats,
    )
    historical_validation_log_likelihood, _historical_validation_stats = prequential_form_log_likelihood(
        validation_rows,
        HISTORICAL_FORM_PRIOR_GOALS,
        rho,
        classifier_weight=classifier_weight,
        initial_stats=training_stats,
    )
    required_gain = CURRENT_FORM_REQUIRED_LOG_SCORE_GAIN_PER_MATCH * len(validation_rows)
    if validation_log_likelihood <= historical_validation_log_likelihood + required_gain:
        return historical_tournament_form(
            observed_results,
            status="fallback_history_validation",
            validation_matches=len(validation_rows),
            validation_log_likelihood=validation_log_likelihood,
            historical_validation_log_likelihood=historical_validation_log_likelihood,
        )

    stats = accumulate_form_stats(observed_rows)
    median_weight = form_weight_median(stats, prior_goals)
    teams = build_team_forms(stats, prior_goals)
    return TournamentForm(
        observed_results=observed_results,
        teams=teams,
        prior_goal_equivalents=prior_goals,
        median_current_weight=median_weight,
        is_enabled=True,
        calibration_status="enabled_validation",
        validation_matches=len(validation_rows),
        validation_log_likelihood=validation_log_likelihood,
        historical_validation_log_likelihood=historical_validation_log_likelihood,
    )


def historical_tournament_form(
    observed_results: dict[int, ObservedResult],
    *,
    status: str,
    validation_matches: int,
    validation_log_likelihood: float,
    historical_validation_log_likelihood: float,
) -> TournamentForm:
    return TournamentForm(
        observed_results=observed_results,
        teams={},
        prior_goal_equivalents=0.0,
        median_current_weight=0.0,
        is_enabled=False,
        calibration_status=status,
        validation_matches=validation_matches,
        validation_log_likelihood=validation_log_likelihood,
        historical_validation_log_likelihood=historical_validation_log_likelihood,
    )


def empty_form_stats() -> defaultdict[str, dict[str, float]]:
    return defaultdict(lambda: {"matches": 0.0, "gf": 0.0, "ga": 0.0, "xgf": 0.0, "xga": 0.0})


def accumulate_form_stats(
    observed_rows: list[tuple[str, str, dict[str, float], ObservedResult]],
) -> defaultdict[str, dict[str, float]]:
    stats = empty_form_stats()
    for home, away, prediction, observed in observed_rows:
        update_form_stats(
            stats,
            home,
            away,
            observed.home_goals,
            observed.away_goals,
            float(prediction["home_xg"]),
            float(prediction["away_xg"]),
        )
    return stats


def update_form_stats(
    stats: defaultdict[str, dict[str, float]],
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    home_xg: float,
    away_xg: float,
) -> None:
    for team, goals_for, goals_against, expected_for, expected_against in [
        (home, home_goals, away_goals, home_xg, away_xg),
        (away, away_goals, home_goals, away_xg, home_xg),
    ]:
        stat = stats[team]
        stat["matches"] += 1.0
        stat["gf"] += float(goals_for)
        stat["ga"] += float(goals_against)
        stat["xgf"] += float(expected_for)
        stat["xga"] += float(expected_against)


def rolling_form_log_likelihood(
    observed_rows: list[tuple[str, str, dict[str, float], ObservedResult]],
    prior_goals: float,
    rho: float,
    *,
    classifier_weight: float = sota.MATCH_CLASSIFIER_WEIGHT,
) -> float:
    value, _stats = prequential_form_log_likelihood(
        observed_rows,
        prior_goals,
        rho,
        classifier_weight=classifier_weight,
    )
    return value


def copy_form_stats(
    stats: defaultdict[str, dict[str, float]],
) -> defaultdict[str, dict[str, float]]:
    copied = empty_form_stats()
    for team, stat in stats.items():
        copied[team].update({field: float(stat[field]) for field in ("matches", "gf", "ga", "xgf", "xga")})
    return copied


def prequential_form_log_likelihood(
    observed_rows: list[tuple[str, str, dict[str, float], ObservedResult]],
    prior_goals: float,
    rho: float,
    *,
    classifier_weight: float,
    initial_stats: defaultdict[str, dict[str, float]] | None = None,
) -> tuple[float, defaultdict[str, dict[str, float]]]:
    stats = copy_form_stats(initial_stats) if initial_stats is not None else empty_form_stats()
    value = 0.0
    for home, away, prediction, observed in observed_rows:
        home_xg, away_xg = form_adjusted_xg_from_stats(prediction, stats, home, away, prior_goals)
        max_goals = max(7, observed.home_goals, observed.away_goals)
        matrix = sota.score_matrix(home_xg, away_xg, max_goals=max_goals, rho=rho)
        probability = hybrid_score_probability(
            prediction,
            matrix,
            observed.home_goals,
            observed.away_goals,
            classifier_weight=classifier_weight,
        )
        value += log(max(1e-12, probability))
        update_form_stats(
            stats,
            home,
            away,
            observed.home_goals,
            observed.away_goals,
            float(prediction["home_xg"]),
            float(prediction["away_xg"]),
        )
    return value, stats


def hybrid_score_probability(
    prediction: dict[str, float],
    matrix: np.ndarray,
    home_goals: int,
    away_goals: int,
    *,
    classifier_weight: float,
) -> float:
    outcome = 0 if home_goals > away_goals else 2 if away_goals > home_goals else 1
    blend, poisson = sota.hybrid_outcome_probs(
        sota.classifier_probs_from_prediction(prediction),
        matrix,
        classifier_weight,
    )
    conditional_score_probability = float(matrix[home_goals, away_goals]) / max(1e-12, float(poisson[outcome]))
    return float(blend[outcome] * conditional_score_probability)


def form_weight_median(stats: defaultdict[str, dict[str, float]], prior_goals: float) -> float:
    weights: list[float] = []
    for stat in stats.values():
        weights.extend(
            [
                current_form_weight(float(stat["xgf"]), prior_goals),
                current_form_weight(float(stat["xga"]), prior_goals),
            ]
        )
    return float(np.median(weights)) if weights else 0.0


def current_form_weight(expected_goals: float, prior_goals: float) -> float:
    if expected_goals <= 0 or prior_goals <= 0:
        return 0.0
    return float(expected_goals / (expected_goals + prior_goals))


def posterior_form_multiplier(goals: float, expected_goals: float, prior_goals: float) -> float:
    if expected_goals <= 0 or prior_goals <= 0:
        return 1.0
    raw_multiplier = (prior_goals + goals) / (prior_goals + expected_goals)
    return float(np.clip(raw_multiplier, CURRENT_FORM_FACTOR_FLOOR, CURRENT_FORM_FACTOR_CEILING))


def build_team_forms(stats: defaultdict[str, dict[str, float]], prior_goals: float) -> dict[str, TeamForm]:
    forms: dict[str, TeamForm] = {}
    for team, stat in stats.items():
        forms[team] = TeamForm(
            observed_matches=int(stat["matches"]),
            goals_for=int(stat["gf"]),
            goals_against=int(stat["ga"]),
            expected_goals_for=float(stat["xgf"]),
            expected_goals_against=float(stat["xga"]),
            attack_weight=current_form_weight(float(stat["xgf"]), prior_goals),
            defense_weight=current_form_weight(float(stat["xga"]), prior_goals),
            attack_multiplier=posterior_form_multiplier(float(stat["gf"]), float(stat["xgf"]), prior_goals),
            defense_multiplier=posterior_form_multiplier(float(stat["ga"]), float(stat["xga"]), prior_goals),
        )
    return forms


def neutral_team_form() -> TeamForm:
    return TeamForm(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)


def form_adjusted_xg_from_stats(
    prediction: dict[str, float],
    stats: defaultdict[str, dict[str, float]],
    home: str,
    away: str,
    prior_goals: float,
) -> tuple[float, float]:
    forms = build_team_forms(stats, prior_goals)
    return form_adjusted_xg(prediction, forms, home, away)


def form_adjusted_xg(
    prediction: dict[str, float],
    forms: dict[str, TeamForm],
    home: str,
    away: str,
) -> tuple[float, float]:
    home_form = forms.get(home, neutral_team_form())
    away_form = forms.get(away, neutral_team_form())
    home_multiplier = sqrt(home_form.attack_multiplier * away_form.defense_multiplier)
    away_multiplier = sqrt(away_form.attack_multiplier * home_form.defense_multiplier)
    home_xg = float(np.clip(float(prediction["home_xg"]) * home_multiplier, CURRENT_FORM_MIN_XG, CURRENT_FORM_MAX_XG))
    away_xg = float(np.clip(float(prediction["away_xg"]) * away_multiplier, CURRENT_FORM_MIN_XG, CURRENT_FORM_MAX_XG))
    return home_xg, away_xg


def form_match_weight(form: TournamentForm, home: str, away: str) -> float:
    home_form = form.teams.get(home, neutral_team_form())
    away_form = form.teams.get(away, neutral_team_form())
    return float(
        np.mean(
            [
                home_form.attack_weight,
                home_form.defense_weight,
                away_form.attack_weight,
                away_form.defense_weight,
            ]
        )
    )


def form_aware_match(
    model: WorldCupModel,
    form: TournamentForm,
    home: str,
    away: str,
    *,
    knockout: bool = False,
    context: dict[str, float | str] | None = None,
) -> MatchDistribution:
    prediction = sota.predict_match(model.package, home, away, knockout=knockout, context=context)
    rho = sota.dixon_coles_rho_from_package(model.package)
    policy = sota.simulation_policy_from_package(model.package)
    home_xg, away_xg = form_adjusted_xg(prediction, form.teams, home, away)
    matrix = sota.score_matrix(home_xg, away_xg, rho=rho)
    classifier_probs = sota.classifier_probs_from_prediction(prediction)
    weight = form_match_weight(form, home, away)
    # The current-Cup evidence updates Poisson/Dixon-Coles lambdas. The trained XGBoost
    # share remains intact because the temporal check does not support reweighting it after two games.
    blend, poisson = sota.hybrid_outcome_probs(
        classifier_probs,
        matrix,
        float(policy["classifier_weight"]),
    )

    resolution_policy = knockout_resolution_policy(
        {"home_xg": home_xg, "away_xg": away_xg},
        rho=rho,
    )
    home_advances_if_draw = float(resolution_policy.home_advances_if_draw)
    adjusted = dict(prediction)
    adjusted.update(
        {
            "home_xg": home_xg,
            "away_xg": away_xg,
            "p_home_win_90": float(blend[0]),
            "p_draw_90": float(blend[1]),
            "p_away_win_90": float(blend[2]),
            "p_home_advances_if_draw": home_advances_if_draw,
            "p_home_advances": float(blend[0] + blend[1] * home_advances_if_draw),
            "p_away_advances": float(1.0 - (blend[0] + blend[1] * home_advances_if_draw)),
            "form_weight": weight,
            "form_home_attack_multiplier": form.teams.get(home, neutral_team_form()).attack_multiplier,
            "form_home_defense_multiplier": form.teams.get(home, neutral_team_form()).defense_multiplier,
            "form_away_attack_multiplier": form.teams.get(away, neutral_team_form()).attack_multiplier,
            "form_away_defense_multiplier": form.teams.get(away, neutral_team_form()).defense_multiplier,
        }
    )
    return MatchDistribution(
        prediction=adjusted,
        matrix=matrix,
        blend=blend,
        poisson=poisson,
        form_weight=weight,
    )


def knockout_resolution_policy(
    prediction: dict[str, float],
    *,
    rho: float,
) -> KnockoutResolutionPolicy:
    return sota.knockout_resolution_policy(prediction, rho=rho)


def build_group_stage_board(model: WorldCupModel) -> GroupStageBoard:
    policy = sota.simulation_policy_from_package(model.package)
    snapshot = load_observed_snapshot_metadata()
    observed_results = load_observed_group_results(model, snapshot=snapshot)
    observed_fair_play = validate_observed_fair_play_scores(model, snapshot)
    form = build_tournament_form(model, observed_results)
    states = sota.ensure_states(model.package)
    fifa_ranks = {profile.key: profile.fifa_rank for profile in model.profiles()}
    team_context: dict[str, dict[str, object]] = {}
    match_results: dict[str, list[dict[str, object]]] = {group: [] for group in GROUPS}
    matches_by_group: dict[str, list[GroupMatch]] = {group: [] for group in GROUPS}
    table = {
        row.team_key: {
            "team": row.team_key,
            "group": row.group_letter,
            "pts": 0,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "model_rating": float(states[str(row.team_key)].elo),
            "fifa_rank": int(fifa_ranks[str(row.team_key)]),
            "fair_play_score": observed_fair_play.get(str(row.team_key), float("nan")),
        }
        for row in model.squad.itertuples(index=False)
    }

    group_games = model.fixtures[model.fixtures["stage_id"] == sota.GROUP_STAGE_ID].sort_values("kickoff_at")
    for game in group_games.itertuples(index=False):
        home = sota.canonical_team(game.home_team)
        away = sota.canonical_team(game.away_team)
        group = str(game.group)
        context = sota.fixture_context(game, team_context, home, away)
        base_prediction = sota.predict_match(model.package, home, away, context=context)
        distribution = form_aware_match(model, form, home, away, context=context)
        prediction = distribution.prediction
        observed = observed_results.get(int(game.match_number))
        if observed is not None:
            home_goals = observed.home_goals
            away_goals = observed.away_goals
            outcome = 0 if home_goals > away_goals else 2 if away_goals > home_goals else 1
            score_probability = float(
                distribution.matrix[
                    min(home_goals, distribution.matrix.shape[0] - 1),
                    min(away_goals, distribution.matrix.shape[1] - 1),
                ]
            )
            meta = {
                "sim_outcome": float(outcome),
                "sim_outcome_probability": float(distribution.blend[outcome]),
                "sim_score_probability": score_probability,
            }
        else:
            home_goals, away_goals, meta = choose_group_score(distribution)
        sota.update_team_context(team_context, home, away, game)

        match_results[group].append(
            {
                "home": home,
                "away": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                **meta,
            }
        )
        matches_by_group[group].append(
            GroupMatch(
                match_number=int(game.match_number),
                group=group,
                home=home,
                away=away,
                home_goals=home_goals,
                away_goals=away_goals,
                home_xg=float(prediction["home_xg"]),
                away_xg=float(prediction["away_xg"]),
                base_home_xg=float(base_prediction["home_xg"]),
                base_away_xg=float(base_prediction["away_xg"]),
                home_probability=float(prediction["p_home_win_90"]),
                draw_probability=float(prediction["p_draw_90"]),
                away_probability=float(prediction["p_away_win_90"]),
                outcome_probability=float(meta["sim_outcome_probability"]),
                score_probability=float(meta["sim_score_probability"]),
                is_observed=observed is not None,
                form_weight=distribution.form_weight,
            )
        )
        update_table(table, home, away, home_goals, away_goals)

    standings = pd.DataFrame(table.values())
    ranked_groups: dict[str, pd.DataFrame] = {}
    thirds: list[dict[str, object]] = []
    for group, group_df in standings.groupby("group", sort=True):
        ranked = sota.rank_group(group_df, match_results[str(group)]).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked_groups[str(group)] = ranked
        thirds.append(ranked.iloc[2].to_dict())

    best_thirds = sota.select_best_thirds(pd.DataFrame(thirds)).reset_index(drop=True)
    best_third_teams = {str(row.team) for row in best_thirds.itertuples(index=False)}
    standings = pd.concat(ranked_groups.values(), ignore_index=True)
    qualified_teams = set(standings[standings["rank"] <= 2]["team"]).union(best_third_teams)
    qualifiers: dict[str, str] = {}
    for group, ranked in ranked_groups.items():
        qualifiers[f"1{group}"] = str(ranked.iloc[0]["team"])
        qualifiers[f"2{group}"] = str(ranked.iloc[1]["team"])
    third_order = []
    for row in best_thirds.itertuples(index=False):
        qualifiers[f"3{row.group}"] = str(row.team)
        third_order.append(str(row.group))

    for matches in matches_by_group.values():
        matches.sort(key=lambda match: match.match_number)

    board = GroupStageBoard(
        matches_by_group=matches_by_group,
        standings=standings,
        qualified_teams=qualified_teams,
        best_third_teams=best_third_teams,
        qualifiers=qualifiers,
        third_order=third_order,
        team_context=team_context,
        policy=policy,
        form=form,
        snapshot=snapshot,
        knockout_results={},
    )
    return replace(board, knockout_results=load_observed_knockout_results(model, board))


def choose_group_score(
    distribution: MatchDistribution,
) -> tuple[int, int, dict[str, float]]:
    outcome = int(np.argmax(distribution.blend))
    selected = candidate_for_outcome(
        distribution.matrix,
        distribution.blend,
        outcome,
        "90min",
        advance_multiplier=1.0,
    )
    return selected.home_goals, selected.away_goals, {
        "sim_outcome": float(outcome),
        "sim_outcome_probability": float(distribution.blend[outcome]),
        "sim_score_probability": float(selected.score_probability),
    }


def update_table(
    table: dict[str, dict[str, object]],
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
) -> None:
    home_row = table[home]
    away_row = table[away]
    home_row["played"] = int(home_row["played"]) + 1
    away_row["played"] = int(away_row["played"]) + 1
    home_row["gf"] = int(home_row["gf"]) + home_goals
    home_row["ga"] = int(home_row["ga"]) + away_goals
    away_row["gf"] = int(away_row["gf"]) + away_goals
    away_row["ga"] = int(away_row["ga"]) + home_goals
    home_row["gd"] = int(home_row["gf"]) - int(home_row["ga"])
    away_row["gd"] = int(away_row["gf"]) - int(away_row["ga"])
    if home_goals > away_goals:
        home_row["pts"] = int(home_row["pts"]) + 3
        home_row["wins"] = int(home_row["wins"]) + 1
        away_row["losses"] = int(away_row["losses"]) + 1
    elif away_goals > home_goals:
        away_row["pts"] = int(away_row["pts"]) + 3
        away_row["wins"] = int(away_row["wins"]) + 1
        home_row["losses"] = int(home_row["losses"]) + 1
    else:
        home_row["pts"] = int(home_row["pts"]) + 1
        away_row["pts"] = int(away_row["pts"]) + 1
        home_row["draws"] = int(home_row["draws"]) + 1
        away_row["draws"] = int(away_row["draws"]) + 1


def build_champion_ranking(
    model: WorldCupModel,
    board: GroupStageBoard,
    *,
    runs: int,
    seed: int,
    top_n: int,
    workers: int | None,
    progress_callback: object | None = None,
) -> list[ChampionOption]:
    # The calibrated simulation is stateful because each generated bracket changes travel context.
    # Keep it sequential so the seed remains reproducible and the shared prediction cache stays effective.
    _ = workers
    total = max(1, int(runs))
    rng = random.Random(int(seed))
    counts: Counter[str] = Counter()
    knockout_games = list(
        model.fixtures[model.fixtures["stage_id"] > sota.GROUP_STAGE_ID].sort_values("match_number").itertuples(index=False)
    )
    round32_slots: list[str] = []
    for game in knockout_games:
        if int(game.stage_id) == 2:
            round32_slots.extend(slot for slot in sota.parse_match_label(game.match_label) if slot.startswith("3"))
    third_slot_assignment = sota.assign_third_slots(round32_slots, board.third_order)
    snapshot_interval = max(1, total // 20)
    for index in range(total):
        champion = simulate_form_aware_knockout(
            model,
            board,
            rng,
            knockout_games=knockout_games,
            third_slot_assignment=third_slot_assignment,
        )
        counts[champion] += 1
        done = index + 1
        if progress_callback is not None and (done == total or done % snapshot_interval == 0):
            snapshot = [(team, wins, wins / done) for team, wins in counts.most_common()]
            progress_callback(done, total, snapshot)

    return [
        ChampionOption(rank=rank, team=team, wins=wins, probability=wins / total)
        for rank, (team, wins) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n], start=1)
    ]


def simulate_form_aware_knockout(
    model: WorldCupModel,
    board: GroupStageBoard,
    rng: random.Random,
    *,
    knockout_games: list[object],
    third_slot_assignment: dict[str, str],
) -> str:
    team_context = {team: dict(context) for team, context in board.team_context.items()}
    winners: dict[int, str] = {}
    runners_up: dict[int, str] = {}
    final_winner = ""
    rho = sota.dixon_coles_rho_from_package(model.package)
    for game in knockout_games:
        left_slot, right_slot = sota.parse_match_label(game.match_label)
        home = sota.resolve_bracket_slot(left_slot, board.qualifiers, winners, runners_up, third_slot_assignment)
        away = sota.resolve_bracket_slot(right_slot, board.qualifiers, winners, runners_up, third_slot_assignment)
        context = sota.fixture_context(game, team_context, home, away)
        observed = observed_knockout_for_game(board, game, home, away)
        if observed is None:
            distribution = form_aware_match(model, board.form, home, away, knockout=True, context=context)
            winner, _home_goals, _away_goals, _resolution = sample_form_aware_knockout_result(
                distribution,
                home,
                away,
                rng,
                rho=rho,
            )
        else:
            winner = observed.winner
        loser = away if winner == home else home
        winners[int(game.match_number)] = winner
        runners_up[int(game.match_number)] = loser
        sota.update_team_context(team_context, home, away, game)
        if str(game.stage) == "Final":
            final_winner = winner
    if not final_winner:
        raise RuntimeError("Chave eliminatória sem final")
    return final_winner


def observed_knockout_for_game(
    board: GroupStageBoard,
    game: object,
    home: str,
    away: str,
) -> ObservedKnockoutResult | None:
    observed = board.knockout_results.get(int(game.match_number))
    if observed is None:
        return None
    if (observed.home, observed.away) != (home, away):
        raise RuntimeError(f"resultado observado não confere com a chave no jogo {int(game.match_number)}")
    return observed


def sample_form_aware_knockout_result(
    distribution: MatchDistribution,
    home: str,
    away: str,
    rng: random.Random,
    *,
    rho: float,
) -> tuple[str, int, int, str]:
    """Sample the canonical knockout sequence: 90min, extra time, then penalties."""
    outcome = int(rng.choices([0, 1, 2], weights=distribution.blend)[0])
    home_goals, away_goals = sample_score_for_outcome(distribution.matrix, outcome, rng)
    if home_goals > away_goals:
        return home, home_goals, away_goals, "90min"
    if away_goals > home_goals:
        return away, home_goals, away_goals, "90min"

    resolution_policy = knockout_resolution_policy(distribution.prediction, rho=rho)
    extra_home_goals, extra_away_goals = sample_score_from_matrix(resolution_policy.extra_time_matrix, rng)
    if extra_home_goals > extra_away_goals:
        return home, home_goals + extra_home_goals, away_goals + extra_away_goals, "extra_time"
    if extra_away_goals > extra_home_goals:
        return away, home_goals + extra_home_goals, away_goals + extra_away_goals, "extra_time"

    winner = home if rng.random() < resolution_policy.home_penalty_probability else away
    return winner, home_goals + extra_home_goals, away_goals + extra_away_goals, "penalties"


def build_conditioned_knockout(model: WorldCupModel, board: GroupStageBoard, champion: str) -> pd.DataFrame:
    if champion not in board.qualified_teams:
        raise ValueError(f"{champion} não se classificou na fase de grupos fixa.")

    team_context = {team: dict(context) for team, context in board.team_context.items()}
    winners: dict[int, str] = {}
    runners_up: dict[int, str] = {}
    rows: list[dict[str, object]] = []
    knockout_games = model.fixtures[model.fixtures["stage_id"] > sota.GROUP_STAGE_ID].sort_values("match_number")
    round32_slots: list[str] = []
    for game in knockout_games[knockout_games["stage_id"] == 2].itertuples(index=False):
        round32_slots.extend(slot for slot in sota.parse_match_label(game.match_label) if slot.startswith("3"))
    third_slot_assignment = sota.assign_third_slots(round32_slots, board.third_order)

    for game in knockout_games.itertuples(index=False):
        left_slot, right_slot = sota.parse_match_label(game.match_label)
        home = sota.resolve_bracket_slot(left_slot, board.qualifiers, winners, runners_up, third_slot_assignment)
        away = sota.resolve_bracket_slot(right_slot, board.qualifiers, winners, runners_up, third_slot_assignment)
        context = sota.fixture_context(game, team_context, home, away)
        observed = observed_knockout_for_game(board, game, home, away)
        if observed is None:
            winner, home_goals, away_goals, resolution, meta = conditioned_knockout_result(
                model,
                board.form,
                home,
                away,
                champion,
                context=context,
            )
            meta["is_observed"] = False
        else:
            if champion in {home, away} and champion != observed.winner:
                raise ValueError(f"{champion} já foi eliminado no resultado observado do jogo {int(game.match_number)}.")
            winner = observed.winner
            home_goals = observed.home_goals
            away_goals = observed.away_goals
            resolution = observed.resolution
            meta = {
                "conditioned_for_champion": "",
                "is_observed": True,
                "observed_source": observed.source,
                "observed_home_goals_90": float(observed.home_goals_90),
                "observed_away_goals_90": float(observed.away_goals_90),
                "observed_shootout_home": float(observed.shootout_home) if observed.shootout_home is not None else -1.0,
                "observed_shootout_away": float(observed.shootout_away) if observed.shootout_away is not None else -1.0,
            }
        sota.update_team_context(team_context, home, away, game)
        loser = away if winner == home else home
        winners[int(game.match_number)] = winner
        runners_up[int(game.match_number)] = loser
        rows.append(
            {
                "match_number": int(game.match_number),
                "round": game.stage,
                "slot_home": left_slot,
                "slot_away": right_slot,
                "home": home,
                "away": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "winner": winner,
                "runner_up": loser,
                "resolution": resolution,
                **meta,
            }
        )
    return pd.DataFrame(rows)


def conditioned_knockout_result(
    model: WorldCupModel,
    form: TournamentForm,
    home: str,
    away: str,
    champion: str,
    *,
    context: dict[str, float | str] | None = None,
) -> tuple[str, int, int, str, dict[str, float | str]]:
    distribution = form_aware_match(model, form, home, away, knockout=True, context=context)
    prediction = distribution.prediction
    matrix = distribution.matrix
    blend = distribution.blend
    poisson = distribution.poisson
    policy = sota.simulation_policy_from_package(model.package)
    resolution_policy = knockout_resolution_policy(
        prediction,
        rho=sota.dixon_coles_rho_from_package(model.package),
    )

    if champion == home:
        desired_winner = home
        forced = True
    elif champion == away:
        desired_winner = away
        forced = True
    else:
        desired_winner = home if float(prediction["p_home_advances"]) >= float(prediction["p_away_advances"]) else away
        forced = False

    selected = best_advancing_score(
        matrix,
        blend,
        prediction,
        desired_winner,
        home,
        away,
        resolution_policy=resolution_policy,
    )
    meta = {
        "conditioned_for_champion": champion if forced else "",
        "sim_outcome": float(selected.outcome),
        "sim_outcome_probability": float(blend[selected.outcome]),
        "sim_score_probability": float(selected.score_probability),
        "sim_path_probability": float(selected.path_probability),
        "sim_classifier_weight": float(policy["classifier_weight"]),
        "sim_blend_home": float(blend[0]),
        "sim_blend_draw": float(blend[1]),
        "sim_blend_away": float(blend[2]),
        "sim_poisson_home": float(poisson[0]),
        "sim_poisson_draw": float(poisson[1]),
        "sim_poisson_away": float(poisson[2]),
        "sim_home_advances_if_draw": float(resolution_policy.home_advances_if_draw),
        "sim_home_penalties_if_extra_time_draw": float(resolution_policy.home_penalty_probability),
        "sim_form_weight": float(distribution.form_weight),
        "sim_form_prior_goals": float(form.prior_goal_equivalents),
        "sim_form_home_attack_multiplier": float(prediction["form_home_attack_multiplier"]),
        "sim_form_home_defense_multiplier": float(prediction["form_home_defense_multiplier"]),
        "sim_form_away_attack_multiplier": float(prediction["form_away_attack_multiplier"]),
        "sim_form_away_defense_multiplier": float(prediction["form_away_defense_multiplier"]),
    }
    return desired_winner, selected.home_goals, selected.away_goals, selected.resolution, meta


def best_advancing_score(
    matrix: np.ndarray,
    blend: np.ndarray,
    prediction: dict[str, float],
    desired_winner: str,
    home: str,
    away: str,
    *,
    resolution_policy: KnockoutResolutionPolicy | None = None,
    rho: float | None = None,
) -> ScoreCandidate:
    win_outcome = 0 if desired_winner == home else 2
    if resolution_policy is None:
        resolved_rho = float(rho if rho is not None else sota.DEFAULT_DIXON_COLES_RHO)
        resolution_policy = knockout_resolution_policy(prediction, rho=resolved_rho)

    win_candidate = candidate_for_outcome(matrix, blend, win_outcome, "90min", advance_multiplier=1.0)
    draw_90_candidate = candidate_for_outcome(matrix, blend, 1, "draw_90", advance_multiplier=1.0)
    extra_time_win = candidate_for_outcome(
        resolution_policy.extra_time_matrix,
        resolution_policy.extra_time_outcomes,
        win_outcome,
        "extra_time",
        advance_multiplier=1.0,
    )
    extra_time_candidate = ScoreCandidate(
        home_goals=draw_90_candidate.home_goals + extra_time_win.home_goals,
        away_goals=draw_90_candidate.away_goals + extra_time_win.away_goals,
        outcome=1,
        resolution="extra_time",
        score_probability=float(draw_90_candidate.score_probability * extra_time_win.score_probability),
        path_probability=float(draw_90_candidate.path_probability * extra_time_win.path_probability),
    )

    extra_time_draw = candidate_for_outcome(
        resolution_policy.extra_time_matrix,
        resolution_policy.extra_time_outcomes,
        1,
        "extra_time_draw",
        advance_multiplier=1.0,
    )
    penalties_home = resolution_policy.home_penalty_probability
    penalty_probability = penalties_home if desired_winner == home else 1.0 - penalties_home
    penalties_candidate = ScoreCandidate(
        home_goals=draw_90_candidate.home_goals + extra_time_draw.home_goals,
        away_goals=draw_90_candidate.away_goals + extra_time_draw.away_goals,
        outcome=1,
        resolution="penalties",
        score_probability=float(draw_90_candidate.score_probability * extra_time_draw.score_probability),
        path_probability=float(
            draw_90_candidate.path_probability * extra_time_draw.path_probability * penalty_probability
        ),
    )
    resolution_rank = {"90min": 2, "extra_time": 1, "penalties": 0}
    return max(
        (win_candidate, extra_time_candidate, penalties_candidate),
        key=lambda candidate: (
            candidate.path_probability,
            candidate.score_probability,
            resolution_rank[candidate.resolution],
            -candidate.home_goals - candidate.away_goals,
        ),
    )


def candidate_for_outcome(
    matrix: np.ndarray,
    blend: np.ndarray,
    outcome: int,
    resolution: str,
    *,
    advance_multiplier: float,
) -> ScoreCandidate:
    rows = score_rows_for_outcome(matrix, outcome)
    total = max(0.001, sum(probability for _home, _away, probability in rows))
    home_goals, away_goals, raw_probability = max(rows, key=lambda item: (item[2], -item[0] - item[1]))
    score_probability = float(raw_probability / total)
    path_probability = float(blend[outcome] * score_probability * max(0.001, advance_multiplier))
    return ScoreCandidate(
        home_goals=int(home_goals),
        away_goals=int(away_goals),
        outcome=int(outcome),
        resolution=resolution,
        score_probability=score_probability,
        path_probability=path_probability,
    )


def sample_score_for_outcome(
    matrix: np.ndarray,
    outcome: int,
    rng: random.Random,
) -> tuple[int, int]:
    rows = score_rows_for_outcome(matrix, outcome)
    selected = rng.choices(range(len(rows)), weights=[row[2] for row in rows])[0]
    home_goals, away_goals, _probability = rows[selected]
    return int(home_goals), int(away_goals)


def sample_score_from_matrix(matrix: np.ndarray, rng: random.Random) -> tuple[int, int]:
    flat_index = rng.choices(range(matrix.size), weights=matrix.ravel())[0]
    return int(flat_index // matrix.shape[1]), int(flat_index % matrix.shape[1])


def score_rows_for_outcome(matrix: np.ndarray, outcome: int) -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            if outcome == 0 and home_goals <= away_goals:
                continue
            if outcome == 1 and home_goals != away_goals:
                continue
            if outcome == 2 and home_goals >= away_goals:
                continue
            rows.append((home_goals, away_goals, float(matrix[home_goals, away_goals])))
    if not rows:
        raise ValueError(f"sem placares compatíveis com outcome {outcome}")
    return rows


def display_name_map(model: WorldCupModel) -> dict[str, str]:
    names = {}
    for profile in model.profiles():
        names[profile.key] = TEAM_DISPLAY_NAMES_PT.get(profile.code, profile.name)
    return names


def render_intro(console: Console, board: GroupStageBoard) -> None:
    total_group_matches = sum(len(matches) for matches in board.matches_by_group.values())
    observed_matches = len(board.form.observed_results)
    locked_knockout_matches = len(board.knockout_results)
    snapshot = board.snapshot
    form = board.form
    if form.is_enabled:
        form_line = (
            f"[yellow]Foto atual validada:[/yellow] peso mediano {pct(form.median_current_weight)} "
            f"na forma da Copa (prior histórico: {form.prior_goal_equivalents:.2f} gols equivalentes; "
            f"holdout: {form.validation_log_likelihood - form.historical_validation_log_likelihood:+.3f} log-score)"
        )
    elif form.calibration_status == "fallback_history_validation":
        form_line = (
            "[yellow]Foto atual:[/yellow] preservado o histórico; a forma da Copa não superou o baseline "
            f"no holdout temporal ({form.validation_log_likelihood - form.historical_validation_log_likelihood:+.3f} log-score)"
        )
    elif form.calibration_status == "insufficient_observed_matches":
        form_line = (
            "[yellow]Foto atual:[/yellow] histórico preservado até haver jogos suficientes para o holdout temporal"
        )
    else:
        form_line = "[yellow]Foto atual:[/yellow] histórico preservado; não há resultados observados para calibrar"
    console.print(
        Panel(
            "[bold white]Copa atualizada[/bold white]\n"
            f"[green]Resultados registrados:[/green] {observed_matches}/{total_group_matches} no CSV interno "
            f"({snapshot.source_label}; as-of {snapshot.as_of_utc_text}) | "
            f"[cyan]projeções pendentes:[/cyan] {total_group_matches - observed_matches}\n"
            f"[green]Mata-mata confirmado:[/green] {locked_knockout_matches} jogo(s) travado(s) no CSV interno\n"
            "[yellow]Proveniência:[/yellow] snapshot manual local; sem fonte FIFA ou validação independente.\n"
            "[cyan]Sem seed:[/cyan] a mesma base de resultados e forma preserva o mesmo placar projetado.\n"
            f"[cyan]Política:[/cyan] {float(board.policy['classifier_weight']):.0%} classificador 1X2 + "
            f"{float(board.policy['poisson_weight']):.0%} Poisson/Dixon-Coles | prorrogação DC + pênaltis 50/50\n"
            f"{form_line}\n"
            "[green]Avançam:[/green] 1º e 2º de cada grupo + 8 melhores terceiros",
            title="[bold cyan]Bolão Arena AI[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def render_board(console: Console, board: GroupStageBoard, names: dict[str, str], groups: tuple[str, ...]) -> None:
    render_intro(console, board)
    for group in groups:
        render_group(console, group, board, names)


def render_group(console: Console, group: str, board: GroupStageBoard, names: dict[str, str]) -> None:
    console.print(Panel.fit(f"[bold white]Grupo {group}[/bold white]", border_style="green", box=box.ROUNDED))
    console.print(match_table(group, board, names))
    console.print(standings_table(group, board, names))


def render_monte_carlo_ranking(
    console: Console,
    model: WorldCupModel,
    names: dict[str, str],
    board: GroupStageBoard,
    *,
    runs: int,
    seed: int,
    top_n: int,
    workers: int | None,
) -> list[ChampionOption]:
    _ = workers
    form_line = (
        "aplica a forma observada da Copa validada no holdout temporal."
        if board.form.is_enabled
        else "preserva o baseline histórico porque a forma atual não venceu o holdout temporal."
    )
    console.print(
        Panel(
            f"[bold white]{runs} Copas Monte Carlo[/bold white]\n"
            f"[cyan]Top:[/cyan] {top_n} campeões mais frequentes | [cyan]Seed MC:[/cyan] {seed}\n"
            f"A simulação parte da fase de grupos fixa, preserva {len(board.knockout_results)} resultado(s) já encerrado(s) e {form_line} "
            "A tabela mostra somente o IC 95% do erro de amostragem MC, não a incerteza total do modelo.",
            title="[bold cyan]Ranking de Campeões[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=38),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Rodando Monte Carlo", total=max(1, int(runs)))

        def on_progress(done: int, total: int, _snapshot: object) -> bool:
            progress.update(task_id, completed=done, total=total)
            return True

        ranking = build_champion_ranking(
            model,
            board,
            runs=runs,
            seed=seed,
            top_n=top_n,
            workers=workers,
            progress_callback=on_progress,
        )
    console.print(champion_ranking_table(ranking, names, board, runs))
    return ranking


def champion_ranking_table(
    ranking: list[ChampionOption],
    names: dict[str, str],
    board: GroupStageBoard,
    runs: int,
) -> Table:
    table = Table(
        title="Top campeões para explorar",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", style="yellow", no_wrap=True)
    table.add_column("Campeão", style="bold", max_width=18, overflow="ellipsis", no_wrap=True)
    table.add_column("Títulos", justify="right", no_wrap=True)
    table.add_column("Prob.", justify="right", style="green", no_wrap=True)
    table.add_column("IC MC 95% (amostral)", justify="right", style="dim", no_wrap=True)
    table.add_column("Fase fixa", style="cyan", max_width=18, overflow="ellipsis", no_wrap=True)
    for option in ranking:
        status = "classificado" if option.team in board.qualified_teams else "fora da chave fixa"
        interval_lower, interval_upper = mc_sampling_interval_95(option.probability, runs)
        table.add_row(
            str(option.rank),
            team_name(option.team, names),
            f"{option.wins}/{runs}",
            pct(option.probability),
            f"{pct(interval_lower)}–{pct(interval_upper)}",
            status,
        )
    return table


def render_selected_champion(
    console: Console,
    model: WorldCupModel,
    board: GroupStageBoard,
    names: dict[str, str],
    option: ChampionOption,
) -> None:
    if option.team not in board.qualified_teams:
        console.print(f"[red]{team_name(option.team, names)} não se classificou na fase fixa.[/red]")
        return
    group = str(board.standings.loc[board.standings["team"] == option.team, "group"].iloc[0])
    bracket = build_conditioned_knockout(model, board, option.team)
    final = bracket[bracket["round"] == "Final"].iloc[0]
    console.rule(f"[bold gold1]{team_name(option.team, names)} campeão a partir da fase fixa[/bold gold1]")
    console.print(
        Panel(
            f"[bold green]Escolha Monte Carlo:[/bold green] #{option.rank} "
            f"({option.wins} títulos, {pct(option.probability)})\n"
            f"[bold cyan]Base:[/bold cyan] grupo {group} fixo, sem seed | "
            f"[bold cyan]Final condicionada:[/bold cyan] {team_name(str(final.home), names)} "
            f"{int(final.home_goals)} x {int(final.away_goals)} {team_name(str(final.away), names)}\n"
            "Esta é uma trilha modal condicionada ao campeão, não uma amostra de P(chave | campeão).",
            border_style="gold1",
            box=box.ROUNDED,
        )
    )
    render_group(console, group, board, names)
    console.print(knockout_table(bracket, names, champion_team=option.team))


def match_table(group: str, board: GroupStageBoard, names: dict[str, str]) -> Table:
    table = Table(
        title="Jogos e placares",
        box=box.ROUNDED,
        border_style="blue",
        header_style="bold blue",
    )
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Mandante", style="bold", max_width=18, overflow="ellipsis", no_wrap=True)
    table.add_column("Placar", justify="center", style="bold cyan", no_wrap=True)
    table.add_column("Visitante", style="bold", max_width=18, overflow="ellipsis", no_wrap=True)
    table.add_column("xG exibido", justify="center", no_wrap=True)
    table.add_column("Registro", justify="center", no_wrap=True)
    for match in board.matches_by_group[group]:
        xg_text, xg_reference = group_match_xg_display(match)
        table.add_row(
            str(match.match_number),
            team_name(match.home, names),
            f"{match.home_goals} x {match.away_goals}",
            team_name(match.away, names),
            xg_text,
            f"[green]CSV manual[/green] ({xg_reference})"
            if match.is_observed
            else f"[cyan]Modelo[/cyan] ({xg_reference})",
        )
    return table


def group_match_xg_display(match: GroupMatch) -> tuple[str, str]:
    if match.is_observed:
        return f"{match.base_home_xg:.2f} x {match.base_away_xg:.2f}", "pré-jogo/base"
    return f"{match.home_xg:.2f} x {match.away_xg:.2f}", "projeção com forma"


def standings_table(group: str, board: GroupStageBoard, names: dict[str, str]) -> Table:
    table = Table(
        title="Classificação",
        caption=(
            "Critérios FIFA: pontos; confronto direto (pontos, saldo, gols); "
            "saldo e gols gerais; fair play; ranking FIFA embalado. "
            "Empate terminal sem fair play informado é recusado."
        ),
        box=box.ROUNDED,
        border_style="green",
        header_style="bold green",
    )
    table.add_column("Rank", justify="right", style="yellow", no_wrap=True)
    table.add_column("Equipe", style="bold", max_width=18, overflow="ellipsis", no_wrap=True)
    table.add_column("Pts", justify="right", style="green", no_wrap=True)
    table.add_column("PJ", justify="right", no_wrap=True)
    table.add_column("VIT", justify="right", no_wrap=True)
    table.add_column("E", justify="right", no_wrap=True)
    table.add_column("DER", justify="right", no_wrap=True)
    table.add_column("GM", justify="right", no_wrap=True)
    table.add_column("GC", justify="right", no_wrap=True)
    table.add_column("SG", justify="right", no_wrap=True)
    table.add_column("Status", style="cyan", max_width=18, overflow="ellipsis", no_wrap=True)

    standings = board.standings[board.standings["group"] == group].sort_values("rank")
    for row in standings.itertuples(index=False):
        team = str(row.team)
        rank = int(row.rank)
        status, row_style = qualification_status(team, rank, board)
        table.add_row(
            str(rank),
            team_name(team, names),
            str(int(row.pts)),
            str(int(row.played)),
            str(int(row.wins)),
            str(int(row.draws)),
            str(int(row.losses)),
            str(int(row.gf)),
            str(int(row.ga)),
            signed_int(int(row.gd)),
            status,
            style=row_style,
        )
    return table


def knockout_table(bracket: pd.DataFrame, names: dict[str, str], *, champion_team: str) -> Table:
    table = Table(
        title="Mata-mata condicionado",
        box=box.ROUNDED,
        border_style="gold1",
        header_style="bold gold1",
    )
    table.add_column("Fase", style="bold", max_width=10, overflow="ellipsis", no_wrap=True)
    table.add_column("Mandante", max_width=16, overflow="ellipsis", no_wrap=True)
    table.add_column("Placar", justify="center", style="bold cyan", min_width=5, no_wrap=True)
    table.add_column("Visitante", max_width=16, overflow="ellipsis", no_wrap=True)
    table.add_column("Venceu", style="green", max_width=16, overflow="ellipsis", no_wrap=True)
    table.add_column("Como", style="cyan", max_width=10, overflow="ellipsis", no_wrap=True)
    for row in bracket.itertuples(index=False):
        home = str(row.home)
        away = str(row.away)
        winner = str(row.winner)
        style = "bold green" if winner == champion_team or home == champion_team or away == champion_team else None
        table.add_row(
            stage_name(row.round),
            team_name(home, names),
            f"{int(row.home_goals)} x {int(row.away_goals)}",
            team_name(away, names),
            team_name(winner, names),
            str(row.resolution),
            style=style,
        )
    return table


def qualification_status(team: str, rank: int, board: GroupStageBoard) -> tuple[str, str | None]:
    if rank <= 2:
        return "[bold green]Eliminatórias[/bold green]", None
    if team in board.best_third_teams:
        return "[bold yellow]Eliminatórias[/bold yellow]", None
    return "[dim]Fora[/dim]", "dim"


def resolve_choice(choice: str, ranking: list[ChampionOption], names: dict[str, str]) -> ChampionOption | None:
    value = choice.strip()
    if not value:
        return None
    if value.isdigit():
        rank = int(value)
        for option in ranking:
            if option.rank == rank:
                return option
        return None
    normalized = normalize_text(value)
    for option in ranking:
        candidates = {
            normalize_text(option.team),
            normalize_text(team_name(option.team, names)),
        }
        if normalized in candidates:
            return option
    canonical = sota.canonical_team(value)
    for option in ranking:
        if canonical == option.team:
            return option
    return None


def choose_interactively(console: Console, ranking: list[ChampionOption], names: dict[str, str]) -> ChampionOption | None:
    while True:
        choice = Prompt.ask("Escolha campeão por número/nome ou 0 para sair", default="0")
        if choice.strip() == "0":
            return None
        option = resolve_choice(choice, ranking, names)
        if option is not None:
            return option
        console.print("[red]Escolha inválida. Use o número do ranking ou o nome do campeão.[/red]")


def normalize_text(value: str) -> str:
    return value.strip().casefold()


def team_name(team: str, names: dict[str, str]) -> str:
    return names.get(team, team)


def pct(value: float) -> str:
    return f"{float(value):.1%}"


def mc_sampling_interval_95(probability: float, runs: int) -> tuple[float, float]:
    """Wilson interval for a champion's Monte Carlo sampling proportion."""
    total = max(1, int(runs))
    value = float(np.clip(probability, 0.0, 1.0))
    z = 1.96
    denominator = 1.0 + (z * z / total)
    center = (value + (z * z / (2.0 * total))) / denominator
    radius = (z / denominator) * sqrt((value * (1.0 - value) / total) + (z * z / (4.0 * total * total)))
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def signed_int(value: int) -> str:
    if value > 0:
        return f"[green]+{value}[/green]"
    if value < 0:
        return f"[red]{value}[/red]"
    return "0"


def stage_name(stage: object) -> str:
    return {
        "Round of 32": "32 avos",
        "Round of 16": "Oitavas",
        "Quarterfinals": "Quartas",
        "Semifinals": "Semis",
        "Third Place Playoff": "3º lugar",
        "Final": "Final",
    }.get(str(stage), str(stage))


def parse_group(value: str) -> str:
    group = value.strip().upper()
    if group not in GROUPS:
        allowed = ", ".join(GROUPS)
        raise argparse.ArgumentTypeError(f"grupo inválido: {value}. Use um destes: {allowed}")
    return group


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Console Rich do bolão da Copa 2026.")
    parser.add_argument("--runs", type=int, default=1000, help="Número de Copas Monte Carlo para montar o top de campeões.")
    parser.add_argument("--top", type=int, default=10, help="Quantidade de campeões exibidos no ranking Monte Carlo.")
    parser.add_argument("--mc-seed", type=int, default=2026, help="Seed do Monte Carlo; não afeta a fase de grupos fixa.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Mantido por compatibilidade; o Monte Carlo calibrado roda sequencialmente para preservar o estado da chave.",
    )
    parser.add_argument("--campeao", help="Escolhe direto um campeão por rank ou nome, sem prompt interativo.")
    parser.add_argument("--somente-grupos", action="store_true", help="Mostra apenas a fase de grupos fixa.")
    parser.add_argument(
        "--grupo",
        action="append",
        type=parse_group,
        help="Filtra grupos exibidos na fase fixa. Pode repetir: --grupo C --grupo G.",
    )
    parser.add_argument("--no-color", action="store_true", help="Desativa cores ANSI na saída.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    console = Console(no_color=bool(args.no_color))
    model = WorldCupModel()
    names = display_name_map(model)
    groups = tuple(args.grupo) if args.grupo else GROUPS
    board = build_group_stage_board(model)
    render_board(console, board, names, groups)
    if args.somente_grupos:
        return

    ranking = render_monte_carlo_ranking(
        console,
        model,
        names,
        board,
        runs=max(1, int(args.runs)),
        seed=int(args.mc_seed),
        top_n=max(1, int(args.top)),
        workers=args.workers,
    )

    if args.campeao:
        option = resolve_choice(str(args.campeao), ranking, names)
        if option is None:
            raise SystemExit(f"Campeão inválido: {args.campeao}")
        render_selected_champion(console, model, board, names, option)
        return

    while True:
        option = choose_interactively(console, ranking, names)
        if option is None:
            return
        render_selected_champion(console, model, board, names, option)
        again = Prompt.ask("Escolher outro campeão?", choices=["s", "n"], default="s")
        if again == "n":
            return


if __name__ == "__main__":
    main()

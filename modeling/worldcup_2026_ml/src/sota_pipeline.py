from __future__ import annotations

import json
import math
import pickle
import random
import re
import argparse
import itertools
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CANDIDATES = RAW / "candidates"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
RANDOM_SEED = 2026
GROUP_STAGE_ID = 1
MATCH_CLASSIFIER_WEIGHT = 0.88
MATCH_DRAW_FLOOR = 0.04
MATCH_DRAW_CEILING = 0.46
DEFAULT_DIXON_COLES_RHO = -0.08
DEFAULT_MANUAL_BLEND_WEIGHTS = {"xgb": 0.40, "competitive": 0.18, "logistic": 0.08, "elo": 0.14, "poisson": 0.10, "count_poisson": 0.10}
BLEND_COMPONENT_NAMES = ("xgb", "competitive", "logistic", "elo", "poisson", "count_poisson")
MAX_MONTE_CARLO_WORKERS = 8
REPRESENTATIVE_TOP_N = 5
REPRESENTATIVE_FINALIST_TOP_N = 10
REPRESENTATIVE_STORY_POOL_SIZE = 12
_PREDICTION_CACHE_LOCK_GUARD = threading.Lock()

HOST_TEAMS = {"Canada", "Mexico", "USA"}
MAJOR_TOURNAMENT_HINTS = (
    "World Cup",
    "FIFA World Cup",
    "UEFA Euro",
    "Copa America",
    "African Cup",
    "AFC Asian Cup",
    "CONCACAF Championship",
    "CONCACAF Gold Cup",
)

COUNTRY_ALIASES = {
    "USA": "USA",
    "United States": "USA",
    "Iran": "IR Iran",
    "IR Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "Korea Republic": "Korea Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Czechia": "Czechia",
    "Czech Republic": "Czechia",
    "Türkiye": "Türkiye",
    "Turkey": "Türkiye",
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Côte d'Ivoire": "Côte d'Ivoire",
    "DR Congo": "Congo DR",
    "Congo DR": "Congo DR",
    "Congo, DR": "Congo DR",
    "Democratic Republic of Congo": "Congo DR",
    "Cape Verde": "Cape Verde Islands",
    "Cabo Verde": "Cape Verde Islands",
    "Cape Verde Islands": "Cape Verde Islands",
    "Korea, South": "Korea Republic",
    "South Korea": "Korea Republic",
    "Korea, North": "Korea DPR",
}

BASE_FEATURES = [
    "elo_diff",
    "attack_diff",
    "defense_diff",
    "win_rate_diff",
    "form_diff",
    "experience_diff",
    "neutral",
    "home_advantage",
    "major_tournament",
    "tournament_weight",
    "elo_abs_diff",
    "draw_pressure",
    "fifa_rank_diff",
    "fifa_points_diff",
    "fifa_rank_change_diff",
    "external_elo_diff",
    "external_elo_abs_diff",
    "same_confederation",
    "home_conf_strength",
    "away_conf_strength",
    "conf_strength_diff",
]

SQUAD_FEATURES = [
    "squad_top26_diff",
    "attack_strength_diff",
    "midfield_strength_diff",
    "defense_strength_diff",
    "gk_strength_diff",
    "market_value_log_diff",
    "age_diff",
    "tm_market_value_log_diff",
    "tm_caps_diff",
    "tm_recent_injury_days_diff",
    "tm_injury_value_log_diff",
    "tm_coverage_diff",
]


@dataclass(frozen=True)
class RepresentativeCandidate:
    seed: int
    champion: str
    runner_up: str
    final_home: str
    final_away: str
    home_goals: int
    away_goals: int
    resolution: str

    @property
    def goal_diff(self) -> int:
        return abs(int(self.home_goals) - int(self.away_goals))

    @property
    def total_goals(self) -> int:
        return int(self.home_goals) + int(self.away_goals)

LCHIKRY_FEATURES = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_avg_overall",
    "home_max_overall",
    "home_avg_attack",
    "home_avg_defense",
    "home_avg_pace",
    "home_avg_shooting",
    "home_avg_passing",
    "away_avg_overall",
    "away_max_overall",
    "away_avg_attack",
    "away_avg_defense",
    "away_avg_pace",
    "away_avg_shooting",
    "away_avg_passing",
    "overall_diff",
    "attack_diff",
    "defense_diff",
    "home_form_scored",
    "home_form_conceded",
    "home_form_win_rate",
    "away_form_scored",
    "away_form_conceded",
    "away_form_win_rate",
    "is_neutral",
    "is_world_cup",
    "is_continental",
]

SOTA_V1_BASELINE_METRICS = {
    "baseline_elo_1x2": {"accuracy": 0.5893},
    "baseline_fifa_rank_1x2": {"accuracy": 0.5629},
    "baseline_elo_winner_no_draw": {"accuracy": 0.7858},
    "baseline_fifa_rank_winner_no_draw": {"accuracy": 0.7567},
    "logistic_1x2": {"accuracy": 0.5680, "top2_accuracy": 0.8411, "log_loss": 0.8842, "draw_recall": 0.2298},
    "xgb_1x2": {"accuracy": 0.6010, "top2_accuracy": 0.8361, "log_loss": 0.8650, "draw_recall": 0.0035},
    "winner_xgb_no_draw": {"accuracy": 0.7891, "log_loss": 0.4409, "brier": 0.1439},
    "competitive_xgb_1x2": {"accuracy": 0.6100, "top2_accuracy": 0.8430, "log_loss": 0.8528},
}

AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277),
    "BOS": (42.3656, -71.0096),
    "DAL": (32.8998, -97.0403),
    "IAH": (29.9902, -95.3368),
    "MCI": (39.2976, -94.7139),
    "LAX": (33.9416, -118.4085),
    "MIA": (25.7959, -80.2870),
    "JFK": (40.6413, -73.7781),
    "PHL": (39.8744, -75.2424),
    "SFO": (37.6213, -122.3790),
    "SEA": (47.4502, -122.3088),
    "YYZ": (43.6777, -79.6248),
    "YVR": (49.1967, -123.1815),
    "GDL": (20.5218, -103.3112),
    "MEX": (19.4361, -99.0719),
    "MTY": (25.7785, -100.1070),
}

HOST_HOME_COUNTRY = {
    "USA": "USA",
    "United States": "USA",
    "Mexico": "Mexico",
    "Canada": "Canada",
}


@dataclass
class RunningTeam:
    elo: float = 1500.0
    matches: int = 0
    goals_for: float = 0.0
    goals_against: float = 0.0
    wins: int = 0
    form: deque[float] | None = None

    def __post_init__(self) -> None:
        if self.form is None:
            self.form = deque(maxlen=12)

    @property
    def avg_for(self) -> float:
        return self.goals_for / self.matches if self.matches else 1.25

    @property
    def avg_against(self) -> float:
        return self.goals_against / self.matches if self.matches else 1.25

    @property
    def win_rate(self) -> float:
        return self.wins / self.matches if self.matches else 0.34

    @property
    def form_score(self) -> float:
        return sum(self.form) / len(self.form) if self.form else 0.40

    def update(self, goals_for: int, goals_against: int) -> None:
        self.matches += 1
        self.goals_for += goals_for
        self.goals_against += goals_against
        if goals_for > goals_against:
            self.wins += 1
            self.form.append(1.0)
        elif goals_for == goals_against:
            self.form.append(0.5)
        else:
            self.form.append(0.0)


def canonical_team(name: str) -> str:
    normalized = " ".join(str(name).replace("\u00a0", " ").split())
    return COUNTRY_ALIASES.get(normalized, normalized)


def parse_mixed_match_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str).str.strip(), format="mixed", errors="coerce")


def load_rankings() -> pd.DataFrame:
    path = RAW / "fifa_rankings_1992_2024.csv"
    if not path.exists():
        return pd.DataFrame(columns=["team", "rank_date", "rank", "total_points", "rank_change", "confederation"])
    ranks = pd.read_csv(path, parse_dates=["rank_date"])
    if "country_full" in ranks.columns:
        ranks = ranks.rename(columns={"country_full": "team"})
    ranks["team"] = ranks["team"].map(canonical_team)
    for col in ["rank", "total_points", "previous_points", "rank_change"]:
        if col in ranks.columns:
            ranks[col] = pd.to_numeric(ranks[col], errors="coerce")
    ranks = ranks.sort_values(["team", "rank_date"])
    return ranks[["team", "rank_date", "rank", "total_points", "rank_change", "confederation"]].dropna(subset=["team", "rank_date"])


def load_external_elo() -> pd.DataFrame:
    path = CANDIDATES / "saifalnimri_eloratings.csv"
    if not path.exists():
        return pd.DataFrame(columns=["team", "elo_date", "external_elo", "external_elo_change"])
    elo = pd.read_csv(path)
    raw_rows = int(len(elo))
    elo["elo_date"] = parse_mixed_match_dates(elo["date"])
    elo["team"] = elo["team"].map(canonical_team)
    elo["external_elo"] = pd.to_numeric(elo["rating"], errors="coerce")
    elo["external_elo_change"] = pd.to_numeric(elo.get("change", 0), errors="coerce").fillna(0.0)
    parsed_rows = int(elo["elo_date"].notna().sum())
    if raw_rows and parsed_rows / raw_rows < 0.95:
        raise ValueError(f"external ELO date coverage too low: {parsed_rows}/{raw_rows}")
    elo = elo.dropna(subset=["team", "elo_date", "external_elo"]).sort_values(["team", "elo_date"])
    latest = elo.groupby("team").tail(1) if not elo.empty else elo
    max_date = elo["elo_date"].max() if not elo.empty else pd.NaT
    if pd.isna(max_date) or max_date.year < 2024 or latest["team"].nunique() < 100:
        raise ValueError("external ELO coverage is not current enough for runtime features")
    elo.attrs["quality"] = {
        "raw_rows": raw_rows,
        "parsed_rows": parsed_rows,
        "parsed_ratio": round(parsed_rows / max(1, raw_rows), 6),
        "team_count": int(elo["team"].nunique()),
        "latest_team_count": int(latest["team"].nunique()),
        "max_date": str(max_date.date()),
    }
    return elo[["team", "elo_date", "external_elo", "external_elo_change"]]


def external_elo_quality(elo: pd.DataFrame, fixtures: pd.DataFrame) -> dict[str, object]:
    quality = dict(elo.attrs.get("quality", {}))
    if elo.empty or fixtures.empty:
        quality.update({"qualified_team_count": 0, "qualified_team_coverage": 0, "missing_qualified_teams": []})
        return quality
    latest_teams = set(elo.sort_values("elo_date").groupby("team").tail(1)["team"])
    qualified_teams = {
        team
        for team in set(fixtures["home_team"].map(canonical_team)) | set(fixtures["away_team"].map(canonical_team))
        if team and team.lower() != "nan"
    }
    missing = sorted(qualified_teams - latest_teams)
    quality.update(
        {
            "qualified_team_count": int(len(qualified_teams)),
            "qualified_team_coverage": int(len(qualified_teams & latest_teams)),
            "qualified_team_coverage_ratio": round(len(qualified_teams & latest_teams) / max(1, len(qualified_teams)), 6),
            "missing_qualified_teams": missing,
        }
    )
    if quality["qualified_team_coverage_ratio"] < 0.95:
        raise ValueError(f"external ELO qualified-team coverage too low: {quality}")
    return quality


def add_external_elo_features(frame: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    if elo.empty:
        frame["home_external_elo"] = 1500.0
        frame["away_external_elo"] = 1500.0
    else:
        frame = frame.sort_values("date").copy()
        ratings = elo.sort_values("elo_date")
        home_elo = pd.merge_asof(
            frame[["date", "home_team"]].rename(columns={"home_team": "team"}),
            ratings,
            left_on="date",
            right_on="elo_date",
            by="team",
            direction="backward",
        ).rename(columns={"external_elo": "home_external_elo"})
        away_elo = pd.merge_asof(
            frame[["date", "away_team"]].rename(columns={"away_team": "team"}),
            ratings,
            left_on="date",
            right_on="elo_date",
            by="team",
            direction="backward",
        ).rename(columns={"external_elo": "away_external_elo"})
        frame["home_external_elo"] = home_elo["home_external_elo"].to_numpy()
        frame["away_external_elo"] = away_elo["away_external_elo"].to_numpy()
    frame["home_external_elo"] = pd.to_numeric(frame["home_external_elo"], errors="coerce").fillna(1500.0)
    frame["away_external_elo"] = pd.to_numeric(frame["away_external_elo"], errors="coerce").fillna(1500.0)
    frame["external_elo_diff"] = frame["home_external_elo"] - frame["away_external_elo"]
    frame["external_elo_abs_diff"] = frame["external_elo_diff"].abs()
    return frame


def add_ranking_features(frame: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        frame["home_fifa_rank"] = 120.0
        frame["away_fifa_rank"] = 120.0
        frame["home_fifa_points"] = 1000.0
        frame["away_fifa_points"] = 1000.0
        frame["home_fifa_rank_change"] = 0.0
        frame["away_fifa_rank_change"] = 0.0
        frame["home_confederation"] = "UNKNOWN"
        frame["away_confederation"] = "UNKNOWN"
    else:
        frame = frame.sort_values("date").copy()
        ranks = rankings.sort_values("rank_date")
        home_ranks = pd.merge_asof(
            frame[["date", "home_team"]].rename(columns={"home_team": "team"}),
            ranks,
            left_on="date",
            right_on="rank_date",
            by="team",
            direction="backward",
        ).rename(
            columns={
                "rank": "home_fifa_rank",
                "total_points": "home_fifa_points",
                "rank_change": "home_fifa_rank_change",
                "confederation": "home_confederation",
            }
        )
        away_ranks = pd.merge_asof(
            frame[["date", "away_team"]].rename(columns={"away_team": "team"}),
            ranks,
            left_on="date",
            right_on="rank_date",
            by="team",
            direction="backward",
        ).rename(
            columns={
                "rank": "away_fifa_rank",
                "total_points": "away_fifa_points",
                "rank_change": "away_fifa_rank_change",
                "confederation": "away_confederation",
            }
        )
        frame["home_fifa_rank"] = home_ranks["home_fifa_rank"].to_numpy()
        frame["away_fifa_rank"] = away_ranks["away_fifa_rank"].to_numpy()
        frame["home_fifa_points"] = home_ranks["home_fifa_points"].to_numpy()
        frame["away_fifa_points"] = away_ranks["away_fifa_points"].to_numpy()
        frame["home_fifa_rank_change"] = home_ranks["home_fifa_rank_change"].to_numpy()
        frame["away_fifa_rank_change"] = away_ranks["away_fifa_rank_change"].to_numpy()
        frame["home_confederation"] = home_ranks["home_confederation"].fillna("UNKNOWN").to_numpy()
        frame["away_confederation"] = away_ranks["away_confederation"].fillna("UNKNOWN").to_numpy()
    for col, default in {
        "home_fifa_rank": 120.0,
        "away_fifa_rank": 120.0,
        "home_fifa_points": 1000.0,
        "away_fifa_points": 1000.0,
        "home_fifa_rank_change": 0.0,
        "away_fifa_rank_change": 0.0,
    }.items():
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(default)
    frame["fifa_rank_diff"] = frame["away_fifa_rank"] - frame["home_fifa_rank"]
    frame["fifa_points_diff"] = frame["home_fifa_points"] - frame["away_fifa_points"]
    frame["fifa_rank_change_diff"] = frame["away_fifa_rank_change"] - frame["home_fifa_rank_change"]
    conf_strength = {"UEFA": 1.00, "CONMEBOL": 0.95, "CONCACAF": 0.78, "CAF": 0.76, "AFC": 0.72, "OFC": 0.58, "UNKNOWN": 0.70}
    frame["home_confederation"] = frame.get("home_confederation", "UNKNOWN")
    frame["away_confederation"] = frame.get("away_confederation", "UNKNOWN")
    frame["same_confederation"] = (frame["home_confederation"] == frame["away_confederation"]).astype(float)
    frame["home_conf_strength"] = frame["home_confederation"].map(conf_strength).fillna(conf_strength["UNKNOWN"])
    frame["away_conf_strength"] = frame["away_confederation"].map(conf_strength).fillna(conf_strength["UNKNOWN"])
    frame["conf_strength_diff"] = frame["home_conf_strength"] - frame["away_conf_strength"]
    return frame


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def haversine_km(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(value)))


def tournament_weight(name: str) -> float:
    text = str(name)
    if "World Cup" in text:
        return 1.35
    if any(hint in text for hint in MAJOR_TOURNAMENT_HINTS):
        return 1.18
    lower = text.lower()
    if "qualification" in lower or "qualifier" in lower:
        return 1.08
    if "friendly" in lower:
        return 0.72
    return 1.0


def label_1x2(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def label_winner(home_score: int, away_score: int) -> int | None:
    if home_score > away_score:
        return 0
    if away_score > home_score:
        return 1
    return None


def base_features(states: dict[str, RunningTeam], home: str, away: str, neutral: bool, tournament: str) -> dict[str, float]:
    h = states[home]
    a = states[away]
    elo_diff = h.elo - a.elo
    draw_pressure = max(0.0, 1.0 - min(abs(elo_diff), 420.0) / 420.0)
    return {
        "elo_diff": elo_diff,
        "attack_diff": h.avg_for - a.avg_for,
        "defense_diff": a.avg_against - h.avg_against,
        "win_rate_diff": h.win_rate - a.win_rate,
        "form_diff": h.form_score - a.form_score,
        "experience_diff": math.log1p(h.matches) - math.log1p(a.matches),
        "neutral": 1.0 if neutral else 0.0,
        "home_advantage": 0.0 if neutral else 1.0,
        "major_tournament": 1.0 if any(hint in str(tournament) for hint in MAJOR_TOURNAMENT_HINTS) else 0.0,
        "tournament_weight": tournament_weight(tournament),
        "elo_abs_diff": abs(elo_diff),
        "draw_pressure": draw_pressure,
    }


def update_elo(states: dict[str, RunningTeam], home: str, away: str, label: int, margin: int, weight: float, neutral: bool) -> None:
    h = states[home]
    a = states[away]
    home_elo = h.elo + (0 if neutral else 35)
    expected_home = expected_score(home_elo, a.elo)
    expected_away = expected_score(a.elo, home_elo)
    actual_home, actual_away = ((1.0, 0.0), (0.5, 0.5), (0.0, 1.0))[label]
    margin_mult = math.log(abs(margin) + 1.0) + 1.0
    k = 20 * weight * margin_mult
    h.elo += k * (actual_home - expected_home)
    a.elo += k * (actual_away - expected_away)


def state_to_record(state: RunningTeam) -> dict[str, object]:
    return {
        "elo": state.elo,
        "matches": state.matches,
        "goals_for": state.goals_for,
        "goals_against": state.goals_against,
        "wins": state.wins,
        "form": list(state.form or []),
    }


def record_to_state(record: dict[str, object]) -> RunningTeam:
    state = RunningTeam(
        elo=float(record.get("elo", 1500.0)),
        matches=int(record.get("matches", 0)),
        goals_for=float(record.get("goals_for", 0.0)),
        goals_against=float(record.get("goals_against", 0.0)),
        wins=int(record.get("wins", 0)),
    )
    state.form.clear()
    for value in record.get("form", []):
        state.form.append(float(value))
    return state


def load_results_source() -> tuple[pd.DataFrame, str]:
    candidate = CANDIDATES / "pataterie_all_matches.csv"
    if candidate.exists():
        results = pd.read_csv(candidate, parse_dates=["date"])
        results["city"] = ""
        return results, "patateriedata/all-international-football-results"
    raise FileNotFoundError(f"Missing required training history: {candidate}")


def ensure_states(package: dict[str, object]) -> dict[str, RunningTeam]:
    states = package["states"]
    if states and isinstance(next(iter(states.values())), dict):
        package["states"] = {name: record_to_state(record) for name, record in states.items()}
    return package["states"]


def build_training_frame() -> tuple[pd.DataFrame, dict[str, RunningTeam]]:
    results, source_name = load_results_source()
    results = results.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    results["home_team"] = results["home_team"].map(canonical_team)
    results["away_team"] = results["away_team"].map(canonical_team)
    results = results.sort_values("date")

    states: dict[str, RunningTeam] = defaultdict(RunningTeam)
    rows = []
    for row in results.itertuples(index=False):
        home = canonical_team(row.home_team)
        away = canonical_team(row.away_team)
        hs = int(row.home_score)
        AS = int(row.away_score)
        neutral = bool(row.neutral)
        tournament = str(row.tournament)
        feats = base_features(states, home, away, neutral, tournament)
        label = label_1x2(hs, AS)
        winner = label_winner(hs, AS)
        if row.date.year >= 1990:
            rows.append(
                {
                    "date": row.date,
                    "home_team": home,
                    "away_team": away,
                    "home_score": hs,
                    "away_score": AS,
                    "tournament": tournament,
                    "target_1x2": label,
                    "target_winner": winner,
                    "source_name": source_name,
                    **feats,
                }
            )
        update_elo(states, home, away, label, hs - AS, tournament_weight(tournament), neutral)
        states[home].update(hs, AS)
        states[away].update(AS, hs)
    frame = pd.DataFrame(rows)
    frame = add_ranking_features(frame, load_rankings())
    frame = add_external_elo_features(frame, load_external_elo())
    return frame, states


def build_squad_strength() -> pd.DataFrame:
    teams = pd.read_csv(RAW / "teams.csv")
    players = pd.read_csv(RAW / "fc26_players.csv", low_memory=False)
    players["nationality_name"] = players["nationality_name"].map(canonical_team)
    players["player_positions"] = players["player_positions"].fillna("")
    for col in ["overall", "value_eur", "age"]:
        players[col] = pd.to_numeric(players[col], errors="coerce")
    players["overall"] = players["overall"].fillna(players["overall"].median())
    players["value_eur"] = players["value_eur"].fillna(0)
    players["age"] = players["age"].fillna(players["age"].median())
    global_rating = float(players["overall"].mean())
    tm_strength = build_transfermarkt_strength()
    rows = []

    for team in teams.itertuples(index=False):
        team_key = canonical_team(team.team_name)
        subset = players[players["nationality_name"] == team_key].sort_values("overall", ascending=False)
        if subset.empty:
            top11 = top26 = pd.DataFrame({"overall": [global_rating], "age": [27.0], "value_eur": [0.0], "player_positions": [""]})
        else:
            top11 = subset.head(11)
            top26 = subset.head(26)

        def pos_mean(tokens: tuple[str, ...]) -> float:
            if subset.empty:
                return global_rating
            mask = subset["player_positions"].apply(lambda text: any(tok in str(text).split(", ") for tok in tokens))
            return float(subset[mask].head(8)["overall"].mean()) if mask.any() else float(top11["overall"].mean())

        tm = tm_strength.get(team_key, {})
        rows.append(
            {
                "team_id": int(team.id),
                "team_name": team.team_name,
                "team_key": team_key,
                "fifa_code": team.fifa_code,
                "group_letter": team.group_letter,
                "is_placeholder": bool(team.is_placeholder),
                "players_found": int(len(subset)),
                "squad_top11": float(top11["overall"].mean()),
                "squad_top26": float(top26["overall"].mean()),
                "attack_strength": pos_mean(("ST", "CF", "LW", "RW", "LF", "RF")),
                "midfield_strength": pos_mean(("CM", "CAM", "CDM", "LM", "RM")),
                "defense_strength": pos_mean(("CB", "LB", "RB", "LWB", "RWB")),
                "gk_strength": pos_mean(("GK",)),
                "avg_age_top26": float(top26["age"].mean()),
                "market_value_top26": float(top26["value_eur"].sum()),
                "tm_players_found": int(tm.get("players_found", 0)),
                "tm_market_value_top25": float(tm.get("market_value_top25", 0.0)),
                "tm_caps_top25": float(tm.get("caps_top25", 0.0)),
                "tm_recent_injury_days_top25": float(tm.get("recent_injury_days_top25", 0.0)),
                "tm_injury_value_risk_top25": float(tm.get("injury_value_risk_top25", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_transfermarkt_strength() -> dict[str, dict[str, float]]:
    profiles_path = CANDIDATES / "transfermarkt_player_profiles.csv"
    values_path = CANDIDATES / "transfermarkt_player_market_value.csv"
    injuries_path = CANDIDATES / "transfermarkt_player_injuries.csv"
    national_path = CANDIDATES / "transfermarkt_player_national_performances.csv"
    if not (profiles_path.exists() and values_path.exists()):
        return {}

    profiles = pd.read_csv(profiles_path, low_memory=False)
    values = pd.read_csv(values_path, low_memory=False)
    profiles["team_key"] = profiles["citizenship"].fillna("").map(
        lambda value: [canonical_team(part.strip()) for part in re.split(r"\s{2,}", str(value)) if part.strip()]
    )
    profiles = profiles.explode("team_key")
    values["value_date"] = pd.to_datetime(values["date_unix"], errors="coerce")
    values["value"] = pd.to_numeric(values["value"], errors="coerce").fillna(0.0)
    latest_values = values.sort_values("value_date").groupby("player_id").tail(1)[["player_id", "value"]]
    merged = profiles[["player_id", "team_key"]].merge(latest_values, on="player_id", how="left")
    merged["value"] = merged["value"].fillna(0.0)

    if national_path.exists():
        national = pd.read_csv(national_path, low_memory=False)
        caps = national.groupby("player_id", as_index=False)["matches"].sum().rename(columns={"matches": "caps"})
        merged = merged.merge(caps, on="player_id", how="left")
    else:
        merged["caps"] = 0.0
    merged["caps"] = pd.to_numeric(merged["caps"], errors="coerce").fillna(0.0)

    if injuries_path.exists():
        injuries = pd.read_csv(injuries_path, low_memory=False)
        injuries["from_date"] = pd.to_datetime(injuries["from_date"], errors="coerce")
        injuries["days_missed"] = pd.to_numeric(injuries["days_missed"], errors="coerce").fillna(0.0)
        recent_cutoff = pd.Timestamp("2024-06-01")
        recent = injuries[injuries["from_date"] >= recent_cutoff].groupby("player_id", as_index=False)["days_missed"].sum()
        recent = recent.rename(columns={"days_missed": "recent_injury_days"})
        merged = merged.merge(recent, on="player_id", how="left")
    else:
        merged["recent_injury_days"] = 0.0
    merged["recent_injury_days"] = pd.to_numeric(merged["recent_injury_days"], errors="coerce").fillna(0.0)
    merged["injury_value_risk"] = merged["value"] * np.minimum(1.0, merged["recent_injury_days"] / 120.0)

    result: dict[str, dict[str, float]] = {}
    for team, group in merged.groupby("team_key"):
        if not team:
            continue
        top = group.sort_values("value", ascending=False).head(25)
        result[team] = {
            "players_found": float(len(group)),
            "market_value_top25": float(top["value"].sum()),
            "caps_top25": float(top["caps"].sum()),
            "recent_injury_days_top25": float(top["recent_injury_days"].sum()),
            "injury_value_risk_top25": float(top["injury_value_risk"].sum()),
        }
    return result


def squad_diffs(squad: pd.DataFrame, home: str, away: str) -> dict[str, float]:
    if "team_key" not in squad.columns:
        return {key: 0.0 for key in SQUAD_FEATURES}
    indexed = squad.set_index("team_key")
    if home not in indexed.index or away not in indexed.index:
        return {key: 0.0 for key in SQUAD_FEATURES}
    h = indexed.loc[home]
    a = indexed.loc[away]
    return {
        "squad_top26_diff": float(h["squad_top26"] - a["squad_top26"]),
        "attack_strength_diff": float(h["attack_strength"] - a["attack_strength"]),
        "midfield_strength_diff": float(h["midfield_strength"] - a["midfield_strength"]),
        "defense_strength_diff": float(h["defense_strength"] - a["defense_strength"]),
        "gk_strength_diff": float(h["gk_strength"] - a["gk_strength"]),
        "market_value_log_diff": float(math.log1p(h["market_value_top26"]) - math.log1p(a["market_value_top26"])),
        "age_diff": float(h["avg_age_top26"] - a["avg_age_top26"]),
        "tm_market_value_log_diff": float(math.log1p(h.get("tm_market_value_top25", 0.0)) - math.log1p(a.get("tm_market_value_top25", 0.0))),
        "tm_caps_diff": float(math.log1p(h.get("tm_caps_top25", 0.0)) - math.log1p(a.get("tm_caps_top25", 0.0))),
        "tm_recent_injury_days_diff": float(math.log1p(h.get("tm_recent_injury_days_top25", 0.0)) - math.log1p(a.get("tm_recent_injury_days_top25", 0.0))),
        "tm_injury_value_log_diff": float(math.log1p(h.get("tm_injury_value_risk_top25", 0.0)) - math.log1p(a.get("tm_injury_value_risk_top25", 0.0))),
        "tm_coverage_diff": float(math.log1p(h.get("tm_players_found", 0.0)) - math.log1p(a.get("tm_players_found", 0.0))),
    }


def lchikry_match_features(package: dict[str, object], home: str, away: str, neutral: bool = True) -> pd.DataFrame:
    states = ensure_states(package)
    h = canonical_team(home)
    a = canonical_team(away)
    states.setdefault(h, RunningTeam())
    states.setdefault(a, RunningTeam())
    squad = package["squad_strength"]
    indexed = squad.set_index("team_key") if "team_key" in squad.columns else pd.DataFrame()

    def squad_row(team: str) -> dict[str, float]:
        if not indexed.empty and team in indexed.index:
            row = indexed.loc[team]
            return {
                "overall": float(row.get("squad_top26", 70.0)),
                "max_overall": float(row.get("squad_top11", 70.0)),
                "attack": float(row.get("attack_strength", 70.0)),
                "defense": float(row.get("defense_strength", 70.0)),
                "pace": float(row.get("squad_top26", 70.0)),
                "shooting": float(row.get("attack_strength", 70.0)),
                "passing": float(row.get("midfield_strength", 70.0)),
            }
        return {key: 70.0 for key in ["overall", "max_overall", "attack", "defense", "pace", "shooting", "passing"]}

    hs = squad_row(h)
    AS = squad_row(a)
    row = {
        "home_elo": states[h].elo,
        "away_elo": states[a].elo,
        "elo_diff": states[h].elo - states[a].elo,
        "home_avg_overall": hs["overall"],
        "home_max_overall": hs["max_overall"],
        "home_avg_attack": hs["attack"],
        "home_avg_defense": hs["defense"],
        "home_avg_pace": hs["pace"],
        "home_avg_shooting": hs["shooting"],
        "home_avg_passing": hs["passing"],
        "away_avg_overall": AS["overall"],
        "away_max_overall": AS["max_overall"],
        "away_avg_attack": AS["attack"],
        "away_avg_defense": AS["defense"],
        "away_avg_pace": AS["pace"],
        "away_avg_shooting": AS["shooting"],
        "away_avg_passing": AS["passing"],
        "overall_diff": hs["overall"] - AS["overall"],
        "attack_diff": hs["attack"] - AS["attack"],
        "defense_diff": hs["defense"] - AS["defense"],
        "home_form_scored": states[h].avg_for,
        "home_form_conceded": states[h].avg_against,
        "home_form_win_rate": states[h].win_rate,
        "away_form_scored": states[a].avg_for,
        "away_form_conceded": states[a].avg_against,
        "away_form_win_rate": states[a].win_rate,
        "is_neutral": 1.0 if neutral else 0.0,
        "is_world_cup": 1.0,
        "is_continental": 0.0,
    }
    return pd.DataFrame([row], columns=LCHIKRY_FEATURES)


def latest_rank_lookup(rankings: pd.DataFrame) -> dict[str, dict[str, float]]:
    if rankings.empty:
        return {}
    latest = rankings.sort_values("rank_date").groupby("team").tail(1)
    result = {}
    for row in latest.itertuples(index=False):
        result[row.team] = {
            "rank": float(row.rank) if pd.notna(row.rank) else 120.0,
            "points": float(row.total_points) if pd.notna(row.total_points) else 1000.0,
            "rank_change": float(row.rank_change) if pd.notna(row.rank_change) else 0.0,
        }
    return result


def latest_external_elo_lookup(elo: pd.DataFrame) -> dict[str, float]:
    if elo.empty:
        return {}
    latest = elo.sort_values("elo_date").groupby("team").tail(1)
    return {row.team: float(row.external_elo) for row in latest.itertuples(index=False)}


def latest_confederation_lookup(rankings: pd.DataFrame) -> dict[str, str]:
    if rankings.empty:
        return {}
    latest = rankings.sort_values("rank_date").groupby("team").tail(1)
    return {row.team: str(row.confederation) if pd.notna(row.confederation) else "UNKNOWN" for row in latest.itertuples(index=False)}


def load_fixtures() -> pd.DataFrame:
    teams = pd.read_csv(RAW / "teams.csv")
    matches = pd.read_csv(RAW / "matches.csv")
    stages = pd.read_csv(RAW / "tournament_stages.csv")
    cities = pd.read_csv(RAW / "host_cities.csv")
    fixtures = (
        matches.merge(teams.add_prefix("home_"), left_on="home_team_id", right_on="home_id", how="left")
        .merge(teams.add_prefix("away_"), left_on="away_team_id", right_on="away_id", how="left")
        .merge(stages.add_prefix("stage_"), left_on="stage_id", right_on="stage_id")
        .merge(cities.add_prefix("city_"), left_on="city_id", right_on="city_id", how="left")
    )
    fixtures = fixtures.rename(
        columns={
            "stage_stage_name": "stage",
            "home_team_name": "home_team",
            "away_team_name": "away_team",
            "home_group_letter": "group",
            "city_airport_code": "airport_code",
            "city_country": "host_country",
        }
    )
    fixtures["home_team"] = fixtures["home_team"].fillna("")
    fixtures["away_team"] = fixtures["away_team"].fillna("")
    fixtures["kickoff_at"] = pd.to_datetime(fixtures["kickoff_at"], errors="coerce", utc=True)
    return fixtures


def fit_stack_base_models(train: pd.DataFrame) -> dict[str, object]:
    x_train = train[BASE_FEATURES]
    y_train = train["target_1x2"].astype(int)
    weight = temporal_sample_weight(train)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=800, class_weight="balanced", random_state=RANDOM_SEED))
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=180,
        max_depth=4,
        learning_rate=0.045,
        subsample=0.88,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )
    competitive = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=160,
        max_depth=4,
        learning_rate=0.045,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )
    home_poisson = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.002, max_iter=500))
    away_poisson = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.002, max_iter=500))
    home_count = XGBRegressor(
        objective="count:poisson",
        n_estimators=160,
        max_depth=3,
        learning_rate=0.045,
        subsample=0.88,
        colsample_bytree=0.9,
        eval_metric="poisson-nloglik",
        random_state=RANDOM_SEED,
    )
    away_count = XGBRegressor(
        objective="count:poisson",
        n_estimators=160,
        max_depth=3,
        learning_rate=0.045,
        subsample=0.88,
        colsample_bytree=0.9,
        eval_metric="poisson-nloglik",
        random_state=RANDOM_SEED,
    )
    fit_pipeline_with_optional_weight(logistic, x_train, y_train, weight)
    xgb.fit(x_train, y_train, sample_weight=weight)
    comp_train = train[train["tournament_weight"] >= 1.0].copy()
    if len(comp_train) < 100:
        comp_train = train
    competitive.fit(
        comp_train[BASE_FEATURES],
        comp_train["target_1x2"].astype(int),
        sample_weight=temporal_sample_weight(comp_train, train["date"].max()),
    )
    fit_pipeline_with_optional_weight(home_poisson, x_train, train["home_score"], weight)
    fit_pipeline_with_optional_weight(away_poisson, x_train, train["away_score"], weight)
    home_count.fit(x_train, train["home_score"], sample_weight=weight)
    away_count.fit(x_train, train["away_score"], sample_weight=weight)
    return {
        "logistic_1x2": logistic,
        "xgb_1x2": xgb,
        "competitive_xgb_1x2": competitive,
        "home_goals_poisson": home_poisson,
        "away_goals_poisson": away_poisson,
        "home_goals_xgb_count": home_count,
        "away_goals_xgb_count": away_count,
    }


def build_stack_matrix(models: dict[str, object], x: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    p_log = models["logistic_1x2"].predict_proba(x)
    p_xgb = models["xgb_1x2"].predict_proba(x)
    p_comp = models["competitive_xgb_1x2"].predict_proba(x)
    elo_home = 1 / (1 + 10 ** (-frame["elo_diff"].to_numpy() / 400))
    draw = np.clip(0.27 - np.abs(elo_home - 0.5) * 0.20, 0.15, 0.30)
    p_elo = np.column_stack([(1 - draw) * elo_home, draw, (1 - draw) * (1 - elo_home)])
    p_poisson = lambdas_to_prob_array(
        np.clip(models["home_goals_poisson"].predict(x), 0.05, 5.5),
        np.clip(models["away_goals_poisson"].predict(x), 0.05, 5.5),
        rho=DEFAULT_DIXON_COLES_RHO,
    )
    p_count = lambdas_to_prob_array(
        np.clip(models["home_goals_xgb_count"].predict(x), 0.05, 5.5),
        np.clip(models["away_goals_xgb_count"].predict(x), 0.05, 5.5),
        rho=DEFAULT_DIXON_COLES_RHO,
    )
    return np.hstack([p_xgb, p_comp, p_log, p_elo, p_poisson, p_count])


def temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(0.2, float(temperature))
    logits = np.log(np.clip(probs, 1e-8, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def learn_temperature(probs: np.ndarray, y: pd.Series) -> float:
    y_true = y.astype(int).to_numpy()
    best_temp = 1.0
    best_loss = float(log_loss(y_true, probs, labels=[0, 1, 2]))
    for temp in np.linspace(0.70, 1.80, 23):
        scaled = temperature_scale_probs(probs, float(temp))
        value = float(log_loss(y_true, scaled, labels=[0, 1, 2]))
        if value < best_loss:
            best_loss = value
            best_temp = float(temp)
    return best_temp


def temporal_sample_weight(frame: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> np.ndarray:
    if frame.empty:
        return np.array([])
    ref = reference_date or frame["date"].max()
    age_years = (ref - frame["date"]).dt.days.clip(lower=0) / 365.25
    recency = np.exp(-age_years / 9.0)
    importance = frame["tournament_weight"].clip(lower=0.65, upper=1.45)
    return np.asarray(0.45 + 0.75 * recency * importance, dtype=float)


def fit_pipeline_with_optional_weight(model: object, x: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> None:
    if sample_weight is None:
        model.fit(x, y)
        return
    model.fit(x, y, **{f"{model.steps[-1][0]}__sample_weight": sample_weight})


def train_models(training: pd.DataFrame) -> tuple[dict[str, object], dict[str, object], float, float]:
    train = training[training["date"] < "2024-01-01"].copy()
    test = training[training["date"] >= "2024-01-01"].copy()
    x_train = train[BASE_FEATURES]
    x_test = test[BASE_FEATURES]
    y_train = train["target_1x2"]
    y_test = test["target_1x2"]
    train_weight = temporal_sample_weight(train)
    winner_test_baseline = test.dropna(subset=["target_winner"]).copy()

    logistic_1x2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_SEED))
    xgb_1x2 = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=360,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.88,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )
    fit_pipeline_with_optional_weight(logistic_1x2, x_train, y_train, train_weight)
    xgb_1x2.fit(x_train, y_train, sample_weight=train_weight)

    winner_train = train.dropna(subset=["target_winner"]).copy()
    winner_test = test.dropna(subset=["target_winner"]).copy()
    winner_model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    winner_model.fit(
        winner_train[BASE_FEATURES],
        winner_train["target_winner"].astype(int),
        sample_weight=temporal_sample_weight(winner_train, train["date"].max()),
    )

    competitive_train = train[train["tournament_weight"] >= 1.0].copy()
    competitive_test = test[test["tournament_weight"] >= 1.0].copy()
    competitive_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=320,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )
    competitive_model.fit(
        competitive_train[BASE_FEATURES],
        competitive_train["target_1x2"],
        sample_weight=temporal_sample_weight(competitive_train, train["date"].max()),
    )

    home_goals_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.002, max_iter=800))
    away_goals_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.002, max_iter=800))
    fit_pipeline_with_optional_weight(home_goals_model, x_train, train["home_score"], train_weight)
    fit_pipeline_with_optional_weight(away_goals_model, x_train, train["away_score"], train_weight)
    home_goals_count_model = XGBRegressor(
        objective="count:poisson",
        n_estimators=260,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.88,
        colsample_bytree=0.9,
        eval_metric="poisson-nloglik",
        random_state=RANDOM_SEED,
    )
    away_goals_count_model = XGBRegressor(
        objective="count:poisson",
        n_estimators=260,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.88,
        colsample_bytree=0.9,
        eval_metric="poisson-nloglik",
        random_state=RANDOM_SEED,
    )
    home_goals_count_model.fit(x_train, train["home_score"], sample_weight=train_weight)
    away_goals_count_model.fit(x_train, train["away_score"], sample_weight=train_weight)
    dc_rho = estimate_dixon_coles_rho(
        train[train["date"] >= "2022-01-01"],
        np.clip(home_goals_model.predict(train[train["date"] >= "2022-01-01"][BASE_FEATURES]), 0.05, 5.5),
        np.clip(away_goals_model.predict(train[train["date"] >= "2022-01-01"][BASE_FEATURES]), 0.05, 5.5),
    )

    stacking_meta = None
    xgb_temperature = 1.0
    stack_train = train[train["date"] < "2022-01-01"].copy()
    stack_val = train[(train["date"] >= "2022-01-01") & (train["date"] < "2024-01-01")].copy()
    if len(stack_train) > 1000 and len(stack_val) > 100:
        stack_models = fit_stack_base_models(stack_train)
        stack_x = build_stack_matrix(stack_models, stack_val[BASE_FEATURES], stack_val)
        stacking_meta = LogisticRegression(max_iter=800, class_weight="balanced", random_state=RANDOM_SEED)
        stacking_meta.fit(stack_x, stack_val["target_1x2"].astype(int))
        xgb_temperature = learn_temperature(stack_models["xgb_1x2"].predict_proba(stack_val[BASE_FEATURES]), stack_val["target_1x2"])

    lchikry_model = None
    lchikry_metrics = None
    lchikry_path = CANDIDATES / "teams_match_features.csv"
    if lchikry_path.exists():
        lchikry = pd.read_csv(lchikry_path, parse_dates=["_date"])
        lchikry = lchikry.dropna(subset=["home_goals", "away_goals", "_date"])
        lchikry = lchikry[lchikry["_date"].dt.year >= 1990].copy()
        lchikry["target_1x2"] = [label_1x2(int(h), int(a)) for h, a in zip(lchikry["home_goals"], lchikry["away_goals"])]
        for col in LCHIKRY_FEATURES:
            lchikry[col] = pd.to_numeric(lchikry[col], errors="coerce")
        lchikry = lchikry.dropna(subset=LCHIKRY_FEATURES + ["target_1x2"])
        l_train = lchikry[lchikry["_date"] < "2024-01-01"].copy()
        l_test = lchikry[lchikry["_date"] >= "2024-01-01"].copy()
        if len(l_train) > 1000 and len(l_test) > 100:
            lchikry_model = XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=420,
                max_depth=4,
                learning_rate=0.035,
                subsample=0.88,
                colsample_bytree=0.9,
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
            )
            lchikry_model.fit(l_train[LCHIKRY_FEATURES], l_train["target_1x2"])
            l_probs = lchikry_model.predict_proba(l_test[LCHIKRY_FEATURES])
            l_pred = l_probs.argmax(axis=1)
            l_y = l_test["target_1x2"]
            lchikry_metrics = {
                "accuracy": round(float(accuracy_score(l_y, l_pred)), 4),
                "top2_accuracy": round(float(np.mean([y in np.argsort(p)[-2:] for y, p in zip(l_y, l_probs)])), 4),
                "log_loss": round(float(log_loss(l_y, l_probs, labels=[0, 1, 2])), 4),
                "draw_recall": round(float(np.mean(l_pred[l_y.to_numpy() == 1] == 1)), 4),
                "test_rows": int(len(l_test)),
            }

    metrics = {}
    elo_pred = np.where(test["elo_diff"].abs() < 35, 1, np.where(test["elo_diff"] > 0, 0, 2))
    fifa_pred = np.where(test["fifa_rank_diff"].abs() < 8, 1, np.where(test["fifa_rank_diff"] > 0, 0, 2))
    metrics["baseline_elo_1x2"] = {
        "accuracy": round(float(accuracy_score(y_test, elo_pred)), 4),
        "test_rows": int(len(test)),
    }
    metrics["baseline_fifa_rank_1x2"] = {
        "accuracy": round(float(accuracy_score(y_test, fifa_pred)), 4),
        "test_rows": int(len(test)),
    }
    if not winner_test_baseline.empty:
        y_winner_base = winner_test_baseline["target_winner"].astype(int).to_numpy()
        elo_winner_pred = np.where(winner_test_baseline["elo_diff"].to_numpy() >= 0, 0, 1)
        fifa_winner_pred = np.where(winner_test_baseline["fifa_rank_diff"].to_numpy() >= 0, 0, 1)
        metrics["baseline_elo_winner_no_draw"] = {
            "accuracy": round(float(accuracy_score(y_winner_base, elo_winner_pred)), 4),
            "test_rows": int(len(winner_test_baseline)),
        }
        metrics["baseline_fifa_rank_winner_no_draw"] = {
            "accuracy": round(float(accuracy_score(y_winner_base, fifa_winner_pred)), 4),
            "test_rows": int(len(winner_test_baseline)),
        }
    for name, model in {"logistic_1x2": logistic_1x2, "xgb_1x2": xgb_1x2}.items():
        probs = model.predict_proba(x_test)
        pred = probs.argmax(axis=1)
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "top2_accuracy": round(float(np.mean([y in np.argsort(p)[-2:] for y, p in zip(y_test, probs)])), 4),
            "log_loss": round(float(log_loss(y_test, probs, labels=[0, 1, 2])), 4),
            "draw_recall": round(float(np.mean(pred[y_test.to_numpy() == 1] == 1)), 4),
            "test_rows": int(len(test)),
        }
    xgb_calibrated_probs = temperature_scale_probs(xgb_1x2.predict_proba(x_test), xgb_temperature)
    xgb_calibrated_pred = xgb_calibrated_probs.argmax(axis=1)
    metrics["xgb_temperature_calibrated_1x2"] = {
        "accuracy": round(float(accuracy_score(y_test, xgb_calibrated_pred)), 4),
        "top2_accuracy": round(float(np.mean([y in np.argsort(p)[-2:] for y, p in zip(y_test, xgb_calibrated_probs)])), 4),
        "log_loss": round(float(log_loss(y_test, xgb_calibrated_probs, labels=[0, 1, 2])), 4),
        "draw_recall": round(float(np.mean(xgb_calibrated_pred[y_test.to_numpy() == 1] == 1)), 4),
        "temperature": round(float(xgb_temperature), 4),
        "test_rows": int(len(test)),
    }
    winner_probs = winner_model.predict_proba(winner_test[BASE_FEATURES])[:, 1]
    winner_pred = (winner_probs >= 0.5).astype(int)
    y_winner = winner_test["target_winner"].astype(int).to_numpy()
    metrics["winner_xgb_no_draw"] = {
        "accuracy": round(float(accuracy_score(y_winner, winner_pred)), 4),
        "log_loss": round(float(log_loss(y_winner, np.column_stack([1 - winner_probs, winner_probs]), labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y_winner, winner_probs)), 4),
        "test_rows": int(len(winner_test)),
    }
    comp_probs = competitive_model.predict_proba(competitive_test[BASE_FEATURES])
    comp_pred = comp_probs.argmax(axis=1)
    metrics["competitive_xgb_1x2"] = {
        "accuracy": round(float(accuracy_score(competitive_test["target_1x2"], comp_pred)), 4),
        "top2_accuracy": round(float(np.mean([y in np.argsort(p)[-2:] for y, p in zip(competitive_test["target_1x2"], comp_probs)])), 4),
        "log_loss": round(float(log_loss(competitive_test["target_1x2"], comp_probs, labels=[0, 1, 2])), 4),
        "test_rows": int(len(competitive_test)),
    }

    home_lam = np.clip(home_goals_model.predict(x_test), 0.05, 5.5)
    away_lam = np.clip(away_goals_model.predict(x_test), 0.05, 5.5)
    poisson_probs = []
    for h_lam, a_lam in zip(home_lam, away_lam):
        poisson_probs.append(score_probs_from_lambdas(float(h_lam), float(a_lam), rho=dc_rho))
    poisson_probs = np.array(poisson_probs)
    poisson_pred = poisson_probs.argmax(axis=1)
    metrics["poisson_goal_model_1x2"] = {
        "accuracy": round(float(accuracy_score(y_test, poisson_pred)), 4),
        "top2_accuracy": round(float(np.mean([y in np.argsort(p)[-2:] for y, p in zip(y_test, poisson_probs)])), 4),
        "log_loss": round(float(log_loss(y_test, poisson_probs, labels=[0, 1, 2])), 4),
        "draw_recall": round(float(np.mean(poisson_pred[y_test.to_numpy() == 1] == 1)), 4),
        "home_goal_mae": round(float(mean_absolute_error(test["home_score"], home_lam)), 4),
        "away_goal_mae": round(float(mean_absolute_error(test["away_score"], away_lam)), 4),
        "test_rows": int(len(test)),
    }
    count_home_lam = np.clip(home_goals_count_model.predict(x_test), 0.05, 5.5)
    count_away_lam = np.clip(away_goals_count_model.predict(x_test), 0.05, 5.5)
    count_probs = lambdas_to_prob_array(count_home_lam, count_away_lam, rho=dc_rho)
    count_pred = count_probs.argmax(axis=1)
    metrics["xgb_count_poisson_1x2"] = {
        "accuracy": round(float(accuracy_score(y_test, count_pred)), 4),
        "top2_accuracy": round(float(np.mean([y in np.argsort(p)[-2:] for y, p in zip(y_test, count_probs)])), 4),
        "log_loss": round(float(log_loss(y_test, count_probs, labels=[0, 1, 2])), 4),
        "draw_recall": round(float(np.mean(count_pred[y_test.to_numpy() == 1] == 1)), 4),
        "home_goal_mae": round(float(mean_absolute_error(test["home_score"], count_home_lam)), 4),
        "away_goal_mae": round(float(mean_absolute_error(test["away_score"], count_away_lam)), 4),
        "test_rows": int(len(test)),
    }
    if stacking_meta is not None:
        stack_test = build_stack_matrix(
            {
                "logistic_1x2": logistic_1x2,
                "xgb_1x2": xgb_1x2,
                "competitive_xgb_1x2": competitive_model,
                "home_goals_poisson": home_goals_model,
                "away_goals_poisson": away_goals_model,
                "home_goals_xgb_count": home_goals_count_model,
                "away_goals_xgb_count": away_goals_count_model,
            },
            x_test,
            test,
        )
        stack_probs = stacking_meta.predict_proba(stack_test)
        stack_pred = stack_probs.argmax(axis=1)
        metrics["stacking_meta_1x2"] = {
            "accuracy": round(float(accuracy_score(y_test, stack_pred)), 4),
            "top2_accuracy": round(float(np.mean([y in np.argsort(p)[-2:] for y, p in zip(y_test, stack_probs)])), 4),
            "log_loss": round(float(log_loss(y_test, stack_probs, labels=[0, 1, 2])), 4),
            "draw_recall": round(float(np.mean(stack_pred[y_test.to_numpy() == 1] == 1)), 4),
            "test_rows": int(len(test)),
        }
    if lchikry_metrics is not None:
        metrics["lchikry_external_xgb_1x2"] = lchikry_metrics
    return {
        "logistic_1x2": logistic_1x2,
        "xgb_1x2": xgb_1x2,
        "winner_xgb": winner_model,
        "competitive_xgb_1x2": competitive_model,
        "home_goals_poisson": home_goals_model,
        "away_goals_poisson": away_goals_model,
        "home_goals_xgb_count": home_goals_count_model,
        "away_goals_xgb_count": away_goals_count_model,
        "stacking_meta_1x2": stacking_meta,
        "lchikry_xgb_1x2": lchikry_model,
    }, metrics, dc_rho, xgb_temperature


def brier_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    encoded = np.zeros_like(probs)
    encoded[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - encoded) ** 2, axis=1)))


def rps_1x2(y_true: np.ndarray, probs: np.ndarray) -> float:
    encoded = np.zeros_like(probs)
    encoded[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((np.cumsum(probs, axis=1) - np.cumsum(encoded, axis=1)) ** 2, axis=1) / 2.0))


def ece_multiclass(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    total = len(y_true)
    ece = 0.0
    for low in np.linspace(0, 1, bins, endpoint=False):
        high = low + 1 / bins
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            ece += float(mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean()))
    return ece


def evaluate_1x2_metrics(y_true: pd.Series, probs: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    pred = probs.argmax(axis=1)
    draw_mask = y == 1
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "top2_accuracy": round(float(np.mean([target in np.argsort(prob)[-2:] for target, prob in zip(y, probs)])), 4),
        "log_loss": round(float(log_loss(y, probs, labels=[0, 1, 2])), 4),
        "brier": round(brier_multiclass(y, probs), 4),
        "rps": round(rps_1x2(y, probs), 4),
        "ece": round(ece_multiclass(y, probs), 4),
        "draw_recall": round(float(np.mean(pred[draw_mask] == 1)), 4) if draw_mask.any() else 0.0,
    }


def simulation_policy_grid() -> tuple[list[float], list[float], list[float]]:
    classifier_weights = [round(float(value), 4) for value in np.linspace(0.50, 0.94, 23)]
    draw_floors = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
    draw_ceilings = [0.30, 0.34, 0.38, 0.42, 0.46]
    return classifier_weights, draw_floors, draw_ceilings


def policy_classifier_probs(pre_draw_probs: np.ndarray, draw_floor: float, draw_ceiling: float) -> np.ndarray:
    base = np.asarray(pre_draw_probs, dtype=float)
    if base.ndim == 1:
        base = base.reshape(1, -1)
    p_draw = np.clip(base[:, 1], float(draw_floor), float(draw_ceiling))
    non_draw = np.clip(base[:, 0] + base[:, 2], 0.01, None)
    home_non_draw_share = base[:, 0] / non_draw
    probs = np.column_stack(
        [
            (1.0 - p_draw) * home_non_draw_share,
            p_draw,
            (1.0 - p_draw) * (1.0 - home_non_draw_share),
        ]
    )
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 0.001, None)


def score_simulation_policy_candidate(
    y: np.ndarray,
    pre_draw_probs: np.ndarray,
    poisson_probs: np.ndarray,
    classifier_weight: float,
    draw_floor: float,
    draw_ceiling: float,
) -> dict[str, float]:
    actual_draw_rate = float(np.mean(y == 1))
    draw_mask = y == 1
    classifier_probs = policy_classifier_probs(pre_draw_probs, draw_floor, draw_ceiling)
    probs = classifier_weight * classifier_probs + (1.0 - classifier_weight) * poisson_probs
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 0.001, None)
    pred_labels = probs.argmax(axis=1)
    draw_recall = float(np.mean(pred_labels[draw_mask] == 1)) if draw_mask.any() else 0.0
    logloss = float(log_loss(y, probs, labels=[0, 1, 2]))
    rps = rps_1x2(y, probs)
    ece = ece_multiclass(y, probs)
    brier = brier_multiclass(y, probs)
    draw_expected_rate = float(probs[:, 1].mean())
    draw_gap = abs(draw_expected_rate - actual_draw_rate)
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1).mean() / math.log(3))
    poisson_weight = 1.0 - float(classifier_weight)
    randomness_penalty = max(0.0, 0.12 - poisson_weight)
    objective = (
        logloss
        + 0.30 * rps
        + 0.08 * ece
        + 0.05 * brier
        + 0.42 * draw_gap
        + 0.05 * max(0.0, 0.68 - entropy)
        + 0.25 * randomness_penalty
    )
    return {
        "classifier_weight": round(float(classifier_weight), 4),
        "poisson_weight": round(float(poisson_weight), 4),
        "draw_floor": round(float(draw_floor), 4),
        "draw_ceiling": round(float(draw_ceiling), 4),
        "objective": round(float(objective), 6),
        "log_loss": round(logloss, 6),
        "rps": round(rps, 6),
        "brier": round(brier, 6),
        "ece": round(ece, 6),
        "draw_recall": round(draw_recall, 6),
        "draw_expected_rate": round(draw_expected_rate, 6),
        "draw_actual_rate": round(actual_draw_rate, 6),
        "draw_gap": round(float(draw_gap), 6),
        "entropy": round(entropy, 6),
        "randomness_penalty": round(float(randomness_penalty), 6),
    }


def normalize_manual_blend_weights(weights: dict[str, float]) -> dict[str, float]:
    values = {name: max(0.0, float(weights.get(name, 0.0))) for name in BLEND_COMPONENT_NAMES}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("manual blend weights must contain at least one active component")
    return {name: round(value / total, 6) for name, value in values.items()}


def manual_blend_candidate_grid() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[tuple[float, ...]] = set()
    base = dict(DEFAULT_MANUAL_BLEND_WEIGHTS)
    for size in range(1, len(BLEND_COMPONENT_NAMES) + 1):
        for subset in itertools.combinations(BLEND_COMPONENT_NAMES, size):
            raw = {name: (base[name] if name in subset else 0.0) for name in BLEND_COMPONENT_NAMES}
            weights = normalize_manual_blend_weights(raw)
            signature = tuple(weights[name] for name in BLEND_COMPONENT_NAMES)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(
                {
                    "name": "subset__" + "+".join(subset),
                    "active_components": list(subset),
                    "manual_blend_weights": weights,
                }
            )
    return candidates


def blend_component_probability_arrays(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    normalized = normalize_manual_blend_weights(weights)
    blend: np.ndarray | None = None
    for name, weight in normalized.items():
        if weight <= 0 or name not in components:
            continue
        part = np.asarray(components[name], dtype=float) * float(weight)
        blend = part if blend is None else blend + part
    if blend is None:
        raise ValueError("manual blend does not reference available components")
    blend = np.clip(blend, 0.001, 0.998)
    return blend / np.clip(blend.sum(axis=1, keepdims=True), 0.001, None)


def select_simulation_policy_from_arrays(y: np.ndarray, pre_draw_probs: np.ndarray, poisson_probs: np.ndarray) -> tuple[dict[str, float], list[dict[str, float]]]:
    classifier_weights, draw_floors, draw_ceilings = simulation_policy_grid()
    candidates: list[dict[str, float]] = []
    for draw_floor in draw_floors:
        for draw_ceiling in draw_ceilings:
            if draw_floor >= draw_ceiling:
                continue
            for weight in classifier_weights:
                candidates.append(score_simulation_policy_candidate(y, pre_draw_probs, poisson_probs, weight, draw_floor, draw_ceiling))
    candidates_sorted = sorted(candidates, key=lambda item: item["objective"])
    return candidates_sorted[0], candidates_sorted


def select_weighted_policy_from_nested_rows(rows: list[dict[str, object]]) -> dict[str, float] | None:
    if not rows:
        return None
    buckets: dict[tuple[float, float, float], dict[str, float]] = {}
    for row in rows:
        key = (
            float(row["selected_classifier_weight"]),
            float(row["selected_draw_floor"]),
            float(row["selected_draw_ceiling"]),
        )
        bucket = buckets.setdefault(key, {"outer_rows": 0.0, "inner_objective": 0.0, "folds": 0.0})
        outer_rows = float(row.get("outer_rows", 1) or 1)
        bucket["outer_rows"] += outer_rows
        bucket["inner_objective"] += float(row.get("inner_objective", 0.0)) * outer_rows
        bucket["folds"] += 1
    selected_key, selected_bucket = sorted(
        buckets.items(),
        key=lambda item: (
            -item[1]["outer_rows"],
            item[1]["inner_objective"] / max(1.0, item[1]["outer_rows"]),
            -item[0][0],
            item[0][1],
            item[0][2],
        ),
    )[0]
    classifier_weight, draw_floor, draw_ceiling = selected_key
    return {
        "classifier_weight": round(float(classifier_weight), 4),
        "poisson_weight": round(float(1.0 - classifier_weight), 4),
        "draw_floor": round(float(draw_floor), 4),
        "draw_ceiling": round(float(draw_ceiling), 4),
        "selected_outer_rows": int(selected_bucket["outer_rows"]),
        "selected_folds": int(selected_bucket["folds"]),
        "selected_avg_inner_objective": round(float(selected_bucket["inner_objective"] / max(1.0, selected_bucket["outer_rows"])), 6),
    }


def select_policy_from_nested_candidate_grid(candidate_rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not candidate_rows:
        return None
    buckets: dict[tuple[float, float, float], dict[str, float]] = {}
    for row in candidate_rows:
        key = (
            float(row["classifier_weight"]),
            float(row["draw_floor"]),
            float(row["draw_ceiling"]),
        )
        outer_rows = float(row.get("outer_rows", 1.0) or 1.0)
        bucket = buckets.setdefault(key, {"outer_rows": 0.0, "inner_objective": 0.0, "inner_entropy": 0.0, "folds": 0.0})
        bucket["outer_rows"] += outer_rows
        bucket["inner_objective"] += float(row.get("objective", 0.0)) * outer_rows
        bucket["inner_entropy"] += float(row.get("entropy", 0.0)) * outer_rows
        bucket["folds"] += 1.0
    selected_key, selected_bucket = sorted(
        buckets.items(),
        key=lambda item: (
            item[1]["inner_objective"] / max(1.0, item[1]["outer_rows"]),
            -(item[1]["inner_entropy"] / max(1.0, item[1]["outer_rows"])),
            abs((1.0 - item[0][0]) - 0.12),
            item[0][1],
            item[0][2],
        ),
    )[0]
    classifier_weight, draw_floor, draw_ceiling = selected_key
    return {
        "classifier_weight": round(float(classifier_weight), 4),
        "poisson_weight": round(float(1.0 - classifier_weight), 4),
        "draw_floor": round(float(draw_floor), 4),
        "draw_ceiling": round(float(draw_ceiling), 4),
        "selected_outer_rows": int(selected_bucket["outer_rows"]),
        "selected_folds": int(selected_bucket["folds"]),
        "selected_avg_inner_objective": round(float(selected_bucket["inner_objective"] / max(1.0, selected_bucket["outer_rows"])), 6),
        "selected_avg_inner_entropy": round(float(selected_bucket["inner_entropy"] / max(1.0, selected_bucket["outer_rows"])), 6),
    }


def select_component_policy_from_nested_candidate_grid(candidate_rows: list[dict[str, object]]) -> dict[str, object] | None:
    if not candidate_rows:
        return None
    buckets: dict[tuple[str, float, float, float], dict[str, object]] = {}
    for row in candidate_rows:
        key = (
            str(row["component_candidate"]),
            float(row["classifier_weight"]),
            float(row["draw_floor"]),
            float(row["draw_ceiling"]),
        )
        outer_rows = float(row.get("outer_rows", 1.0) or 1.0)
        bucket = buckets.setdefault(
            key,
            {
                "outer_rows": 0.0,
                "inner_objective": 0.0,
                "inner_entropy": 0.0,
                "folds": 0.0,
                "manual_blend_weights": row.get("manual_blend_weights", {}),
                "active_components": row.get("active_components", []),
            },
        )
        bucket["outer_rows"] = float(bucket["outer_rows"]) + outer_rows
        bucket["inner_objective"] = float(bucket["inner_objective"]) + float(row.get("objective", 0.0)) * outer_rows
        bucket["inner_entropy"] = float(bucket["inner_entropy"]) + float(row.get("entropy", 0.0)) * outer_rows
        bucket["folds"] = float(bucket["folds"]) + 1.0
    selected_key, selected_bucket = sorted(
        buckets.items(),
        key=lambda item: (
            float(item[1]["inner_objective"]) / max(1.0, float(item[1]["outer_rows"])),
            -(float(item[1]["inner_entropy"]) / max(1.0, float(item[1]["outer_rows"]))),
            abs((1.0 - item[0][1]) - 0.12),
            len(item[1].get("active_components", [])),
            item[0][0],
        ),
    )[0]
    component_name, classifier_weight, draw_floor, draw_ceiling = selected_key
    weights = normalize_manual_blend_weights(dict(selected_bucket.get("manual_blend_weights", {})))
    return {
        "component_candidate": component_name,
        "active_components": list(selected_bucket.get("active_components", [])),
        "manual_blend_weights": weights,
        "classifier_weight": round(float(classifier_weight), 4),
        "poisson_weight": round(float(1.0 - classifier_weight), 4),
        "draw_floor": round(float(draw_floor), 4),
        "draw_ceiling": round(float(draw_ceiling), 4),
        "selected_outer_rows": int(float(selected_bucket["outer_rows"])),
        "selected_folds": int(float(selected_bucket["folds"])),
        "selected_avg_inner_objective": round(float(selected_bucket["inner_objective"]) / max(1.0, float(selected_bucket["outer_rows"])), 6),
        "selected_avg_inner_entropy": round(float(selected_bucket["inner_entropy"]) / max(1.0, float(selected_bucket["outer_rows"])), 6),
    }


def simulation_policy_arrays_from_package(package: dict[str, object], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    pre_draw_rows: list[np.ndarray] = []
    poisson_rows: list[np.ndarray] = []
    rho = dixon_coles_rho_from_package(package)
    for row in frame.itertuples(index=False):
        pred = predict_match(package, str(row.home_team), str(row.away_team), neutral=bool(row.neutral))
        matrix = score_matrix(float(pred["home_xg"]), float(pred["away_xg"]), rho=rho)
        pre_draw_rows.append(
            np.array(
                [
                    float(pred["p_pre_draw_home_win_90"]),
                    float(pred["p_pre_draw_draw_90"]),
                    float(pred["p_pre_draw_away_win_90"]),
                ]
            )
        )
        poisson_rows.append(outcome_probs_from_matrix(matrix))
        y_true.append(int(row.target_1x2))
    return np.asarray(y_true, dtype=int), np.vstack(pre_draw_rows), np.vstack(poisson_rows)


def run_nested_temporal_policy_validation(package: dict[str, object], training: pd.DataFrame) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, float]] = []
    outer_payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    component_candidate_rows: list[dict[str, object]] = []
    component_outer_payloads: list[tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]] = []
    component_fold_rows: list[dict[str, object]] = []
    component_candidates = manual_blend_candidate_grid()
    years = [int(year) for year in sorted(training["date"].dt.year.unique()) if 2018 <= int(year) <= 2025]
    use_stacking = bool(package.get("use_stacking_ensemble", False))
    for year in years:
        outer_test = training[training["date"].dt.year == year].copy()
        outer_train = training[training["date"] < f"{year}-01-01"].copy()
        if len(outer_test) < 100 or len(outer_train) < 2000:
            continue
        inner_cutoff = outer_train["date"].max() - pd.Timedelta(days=730)
        inner_train = outer_train[outer_train["date"] < inner_cutoff].copy()
        inner_val = outer_train[outer_train["date"] >= inner_cutoff].copy()
        if len(inner_train) < 2000 or len(inner_val) < 120:
            continue
        inner_models = fit_backtest_models(inner_train)
        inner_rho = estimate_backtest_models_rho(inner_models, inner_train)
        y_inner, inner_components, poi_inner = backtest_component_arrays(inner_models, inner_val, rho=inner_rho)
        pre_inner = (
            inner_models["stacking_meta_1x2"].predict_proba(build_stack_matrix(inner_models, inner_val[BASE_FEATURES], inner_val))
            if use_stacking and inner_models.get("stacking_meta_1x2") is not None
            else blend_component_probability_arrays(inner_components, DEFAULT_MANUAL_BLEND_WEIGHTS)
        )
        selected, inner_candidates = select_simulation_policy_from_arrays(y_inner, pre_inner, poi_inner)
        for candidate in inner_candidates:
            candidate_rows.append(
                {
                    "classifier_weight": float(candidate["classifier_weight"]),
                    "draw_floor": float(candidate["draw_floor"]),
                    "draw_ceiling": float(candidate["draw_ceiling"]),
                    "objective": float(candidate["objective"]),
                    "entropy": float(candidate.get("entropy", 0.0)),
                    "outer_rows": float(len(outer_test)),
                }
            )
        outer_models = fit_backtest_models(outer_train)
        outer_rho = estimate_backtest_models_rho(outer_models, outer_train)
        y_outer, outer_components, poi_outer = backtest_component_arrays(outer_models, outer_test, rho=outer_rho)
        pre_outer = (
            outer_models["stacking_meta_1x2"].predict_proba(build_stack_matrix(outer_models, outer_test[BASE_FEATURES], outer_test))
            if use_stacking and outer_models.get("stacking_meta_1x2") is not None
            else blend_component_probability_arrays(outer_components, DEFAULT_MANUAL_BLEND_WEIGHTS)
        )
        outer_payloads.append((y_outer, pre_outer, poi_outer))
        component_outer_payloads.append((y_outer, outer_components, poi_outer))
        outer_metrics = score_simulation_policy_candidate(
            y_outer,
            pre_outer,
            poi_outer,
            float(selected["classifier_weight"]),
            float(selected["draw_floor"]),
            float(selected["draw_ceiling"]),
        )
        rows.append(
            {
                "outer_year": year,
                "inner_train_start": str(inner_train["date"].min().date()),
                "inner_train_end": str(inner_train["date"].max().date()),
                "inner_train_rows": int(len(inner_train)),
                "inner_start": str(inner_val["date"].min().date()),
                "inner_end": str(inner_val["date"].max().date()),
                "inner_rows": int(len(inner_val)),
                "outer_rows": int(len(outer_test)),
                "inner_rho": round(float(inner_rho), 4),
                "outer_rho": round(float(outer_rho), 4),
                "selected_classifier_weight": selected["classifier_weight"],
                "selected_poisson_weight": selected["poisson_weight"],
                "selected_draw_floor": selected["draw_floor"],
                "selected_draw_ceiling": selected["draw_ceiling"],
                "inner_objective": selected["objective"],
                **{f"outer_{key}": value for key, value in outer_metrics.items() if key not in {"classifier_weight", "poisson_weight", "draw_floor", "draw_ceiling"}},
            }
        )
        fold_best_component: dict[str, object] | None = None
        for component_candidate in component_candidates:
            component_name = str(component_candidate["name"])
            component_weights = dict(component_candidate["manual_blend_weights"])
            component_pre_inner = blend_component_probability_arrays(inner_components, component_weights)
            component_selected, component_inner_candidates = select_simulation_policy_from_arrays(y_inner, component_pre_inner, poi_inner)
            for component_inner_candidate in component_inner_candidates:
                component_candidate_rows.append(
                    {
                        "outer_year": year,
                        "component_candidate": component_name,
                        "active_components": list(component_candidate["active_components"]),
                        "manual_blend_weights": component_weights,
                        "classifier_weight": float(component_inner_candidate["classifier_weight"]),
                        "draw_floor": float(component_inner_candidate["draw_floor"]),
                        "draw_ceiling": float(component_inner_candidate["draw_ceiling"]),
                        "objective": float(component_inner_candidate["objective"]),
                        "entropy": float(component_inner_candidate.get("entropy", 0.0)),
                        "outer_rows": float(len(outer_test)),
                    }
                )
            component_pre_outer = blend_component_probability_arrays(outer_components, component_weights)
            component_outer_metrics = score_simulation_policy_candidate(
                y_outer,
                component_pre_outer,
                poi_outer,
                float(component_selected["classifier_weight"]),
                float(component_selected["draw_floor"]),
                float(component_selected["draw_ceiling"]),
            )
            fold_component_row: dict[str, object] = {
                "outer_year": year,
                "component_candidate": component_name,
                "active_components": list(component_candidate["active_components"]),
                "manual_blend_weights": component_weights,
                "outer_rows": int(len(outer_test)),
                "selected_classifier_weight": component_selected["classifier_weight"],
                "selected_poisson_weight": component_selected["poisson_weight"],
                "selected_draw_floor": component_selected["draw_floor"],
                "selected_draw_ceiling": component_selected["draw_ceiling"],
                "inner_objective": component_selected["objective"],
                **{
                    f"outer_{key}": value
                    for key, value in component_outer_metrics.items()
                    if key not in {"classifier_weight", "poisson_weight", "draw_floor", "draw_ceiling"}
                },
            }
            if fold_best_component is None or float(fold_component_row["inner_objective"]) < float(fold_best_component["inner_objective"]):
                fold_best_component = fold_component_row
        if fold_best_component:
            component_fold_rows.append(fold_best_component)
    aggregate: dict[str, float] = {}
    if rows:
        total = sum(int(row["outer_rows"]) for row in rows)
        metric_keys = [key for key in rows[0] if key.startswith("outer_") and isinstance(rows[0][key], (int, float))]
        aggregate = {
            key: round(float(sum(float(row[key]) * int(row["outer_rows"]) for row in rows) / total), 6)
            for key in metric_keys
        }
        aggregate["folds"] = len(rows)
        aggregate["outer_rows"] = int(total)
    fold_winner_policy = select_weighted_policy_from_nested_rows(rows)
    selected_policy = select_policy_from_nested_candidate_grid(candidate_rows)
    if selected_policy and outer_payloads:
        y_all = np.concatenate([payload[0] for payload in outer_payloads])
        pre_all = np.vstack([payload[1] for payload in outer_payloads])
        poi_all = np.vstack([payload[2] for payload in outer_payloads])
        selected_outer_metrics = score_simulation_policy_candidate(
            y_all,
            pre_all,
            poi_all,
            float(selected_policy["classifier_weight"]),
            float(selected_policy["draw_floor"]),
            float(selected_policy["draw_ceiling"]),
        )
        selected_policy.update(
            {
                f"outer_{key}": value
                for key, value in selected_outer_metrics.items()
                if key not in {"classifier_weight", "poisson_weight", "draw_floor", "draw_ceiling"}
            }
        )
    selected_component_policy = select_component_policy_from_nested_candidate_grid(component_candidate_rows)
    component_aggregate_rows: list[dict[str, object]] = []
    if component_candidate_rows:
        best_by_component_fold: dict[tuple[str, int], dict[str, object]] = {}
        for row in component_candidate_rows:
            fold_key = (str(row["component_candidate"]), int(row["outer_year"]))
            current = best_by_component_fold.get(fold_key)
            if current is None or float(row.get("objective", 0.0)) < float(current.get("objective", 0.0)):
                best_by_component_fold[fold_key] = row
        buckets: dict[str, dict[str, object]] = {}
        for row in best_by_component_fold.values():
            component_name = str(row["component_candidate"])
            outer_rows = float(row.get("outer_rows", 1.0) or 1.0)
            bucket = buckets.setdefault(
                component_name,
                {
                    "component_candidate": component_name,
                    "active_components": row.get("active_components", []),
                    "manual_blend_weights": row.get("manual_blend_weights", {}),
                    "outer_rows": 0.0,
                    "inner_objective": 0.0,
                    "inner_entropy": 0.0,
                    "folds": 0.0,
                },
            )
            bucket["outer_rows"] = float(bucket["outer_rows"]) + outer_rows
            bucket["inner_objective"] = float(bucket["inner_objective"]) + float(row.get("objective", 0.0)) * outer_rows
            bucket["inner_entropy"] = float(bucket["inner_entropy"]) + float(row.get("entropy", 0.0)) * outer_rows
            bucket["folds"] = float(bucket["folds"]) + 1.0
        for bucket in buckets.values():
            rows_weight = max(1.0, float(bucket["outer_rows"]))
            component_aggregate_rows.append(
                {
                    "component_candidate": bucket["component_candidate"],
                    "active_components": bucket["active_components"],
                    "manual_blend_weights": bucket["manual_blend_weights"],
                    "selected_outer_rows": int(float(bucket["outer_rows"])),
                    "selected_folds": int(float(bucket["folds"])),
                    "avg_inner_objective": round(float(bucket["inner_objective"]) / rows_weight, 6),
                    "avg_inner_entropy": round(float(bucket["inner_entropy"]) / rows_weight, 6),
                }
            )
        component_aggregate_rows.sort(key=lambda item: (float(item["avg_inner_objective"]), -float(item["avg_inner_entropy"])))
    if selected_component_policy and component_outer_payloads:
        selected_weights = dict(selected_component_policy["manual_blend_weights"])
        y_all = np.concatenate([payload[0] for payload in component_outer_payloads])
        pre_all = np.vstack([blend_component_probability_arrays(payload[1], selected_weights) for payload in component_outer_payloads])
        poi_all = np.vstack([payload[2] for payload in component_outer_payloads])
        selected_component_outer_metrics = score_simulation_policy_candidate(
            y_all,
            pre_all,
            poi_all,
            float(selected_component_policy["classifier_weight"]),
            float(selected_component_policy["draw_floor"]),
            float(selected_component_policy["draw_ceiling"]),
        )
        selected_component_policy.update(
            {
                f"outer_{key}": value
                for key, value in selected_component_outer_metrics.items()
                if key not in {"classifier_weight", "poisson_weight", "draw_floor", "draw_ceiling"}
            }
        )
        selected_policy = selected_component_policy
    return {
        "version": "nested_temporal_policy_v3_component_ablation_no_leakage_no_draw_xgb",
        "description": "Each outer year trains inner models only before the internal validation window, selects blend components plus classifier/Poisson/draw policy on that later internal window, then refits on all prior data and evaluates the outer year without retuning.",
        "aggregate": aggregate,
        "rows": rows,
        "fold_winner_policy": fold_winner_policy,
        "selected_policy": selected_policy,
        "component_ablation": {
            "version": "nested_component_subset_ablation_v1",
            "candidate_count": len(component_candidates),
            "policy_candidate_count_per_component": len(simulation_policy_grid()[0]) * len(simulation_policy_grid()[1]) * len(simulation_policy_grid()[2]),
            "selection": "joint_nested_temporal_component_and_policy_grid",
            "selected_policy": selected_component_policy,
            "top_component_candidates": component_aggregate_rows[:20],
            "fold_best_rows": component_fold_rows,
        },
    }


def calibrate_simulation_policy(package: dict[str, object], training: pd.DataFrame) -> dict[str, object]:
    holdout = training[training["date"] >= "2024-01-01"].copy()
    if len(holdout) < 120:
        return simulation_policy_from_package(package)

    nested_validation = run_nested_temporal_policy_validation(package, training)
    nested_selected = nested_validation.get("selected_policy") or {}
    manual_blend_weights = normalize_manual_blend_weights(
        dict(nested_selected.get("manual_blend_weights", DEFAULT_MANUAL_BLEND_WEIGHTS))
    )
    y, pre_draw_probs, poisson_probs = backtest_policy_arrays(
        package["models"],
        holdout,
        rho=dixon_coles_rho_from_package(package),
        use_stacking=bool(package.get("use_stacking_ensemble", False)),
        manual_weights=manual_blend_weights,
    )
    holdout_best, candidates_sorted = select_simulation_policy_from_arrays(y, pre_draw_probs, poisson_probs)
    best = (
        score_simulation_policy_candidate(
            y,
            pre_draw_probs,
            poisson_probs,
            float(nested_selected["classifier_weight"]),
            float(nested_selected["draw_floor"]),
            float(nested_selected["draw_ceiling"]),
        )
        if nested_selected
        else holdout_best
    )
    reference_metrics: dict[str, dict[str, float]] = {
        "reference_policy_0_62": score_simulation_policy_candidate(y, pre_draw_probs, poisson_probs, 0.62, 0.08, 0.38),
        "reference_policy_previous_0_80": score_simulation_policy_candidate(y, pre_draw_probs, poisson_probs, 0.80, 0.08, 0.38),
    }
    best_by_classifier_weight: list[dict[str, float]] = []
    classifier_weights, draw_floors, draw_ceilings = simulation_policy_grid()
    for weight in classifier_weights:
        weight_candidates = [candidate for candidate in candidates_sorted if candidate["classifier_weight"] == weight]
        if weight_candidates:
            best_by_classifier_weight.append(min(weight_candidates, key=lambda item: item["objective"]))
    return {
        "name": "hybrid_classifier_poisson",
        "classifier_weight": float(best["classifier_weight"]),
        "poisson_weight": float(best["poisson_weight"]),
        "draw_floor": float(best["draw_floor"]),
        "draw_ceiling": float(best["draw_ceiling"]),
        "score_engine": "Poisson/Dixon-Coles",
        "selected_by": "strict_nested_temporal_component_ablation_no_leakage_no_draw_xgb" if nested_selected else "holdout_grid_no_draw_xgb",
        "manual_blend_weights": manual_blend_weights,
        "backtest_rows": int(len(holdout)),
        "candidate_count": len(candidates_sorted),
        "backtest_metrics": best,
        "holdout_best_metrics": holdout_best,
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidates_sorted[:80],
        "best_by_classifier_weight": best_by_classifier_weight,
        "nested_temporal_validation": nested_validation,
        "search_space": {
            "classifier_weights": classifier_weights,
            "draw_floors": draw_floors,
            "draw_ceilings": draw_ceilings,
            "candidate_count": len(candidates_sorted),
            "minimum_poisson_weight_preferred": 0.12,
            "entropy_floor_preferred": 0.68,
        },
        "description": "Leakage-free nested-temporal selected classifier/Poisson policy with draw floor/ceiling guardrails; draw_xgb was removed because zero-weight inactive models are not SOTA/KISS.",
    }


def fit_backtest_models(train: pd.DataFrame) -> dict[str, object]:
    x_train = train[BASE_FEATURES]
    y_train = train["target_1x2"].astype(int)
    weight = temporal_sample_weight(train)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED))
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=180,
        max_depth=4,
        learning_rate=0.045,
        subsample=0.88,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )
    competitive = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )
    home_goals = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.002, max_iter=600))
    away_goals = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.002, max_iter=600))
    home_count = XGBRegressor(
        objective="count:poisson",
        n_estimators=140,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.88,
        colsample_bytree=0.9,
        eval_metric="poisson-nloglik",
        random_state=RANDOM_SEED,
    )
    away_count = XGBRegressor(
        objective="count:poisson",
        n_estimators=140,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.88,
        colsample_bytree=0.9,
        eval_metric="poisson-nloglik",
        random_state=RANDOM_SEED,
    )
    fit_pipeline_with_optional_weight(logistic, x_train, y_train, weight)
    xgb.fit(x_train, y_train, sample_weight=weight)
    comp_train = train[train["tournament_weight"] >= 1.0].copy()
    if len(comp_train) < 100:
        comp_train = train
    competitive.fit(
        comp_train[BASE_FEATURES],
        comp_train["target_1x2"].astype(int),
        sample_weight=temporal_sample_weight(comp_train, train["date"].max()),
    )
    fit_pipeline_with_optional_weight(home_goals, x_train, train["home_score"], weight)
    fit_pipeline_with_optional_weight(away_goals, x_train, train["away_score"], weight)
    home_count.fit(x_train, train["home_score"], sample_weight=weight)
    away_count.fit(x_train, train["away_score"], sample_weight=weight)
    models = {
        "logistic_1x2": logistic,
        "xgb_1x2": xgb,
        "competitive_xgb_1x2": competitive,
        "home_goals_poisson": home_goals,
        "away_goals_poisson": away_goals,
        "home_goals_xgb_count": home_count,
        "away_goals_xgb_count": away_count,
    }
    stack_train = train[train["date"] < train["date"].max() - pd.Timedelta(days=730)].copy()
    stack_val = train[train["date"] >= train["date"].max() - pd.Timedelta(days=730)].copy()
    if len(stack_train) > 1000 and len(stack_val) > 100:
        stack_base = fit_stack_base_models(stack_train)
        stack_x = build_stack_matrix(stack_base, stack_val[BASE_FEATURES], stack_val)
        stacker = LogisticRegression(max_iter=600, class_weight="balanced", random_state=RANDOM_SEED)
        stacker.fit(stack_x, stack_val["target_1x2"].astype(int))
        models["stacking_meta_1x2"] = stacker
    return models


def elo_policy_probs_from_frame(frame: pd.DataFrame) -> np.ndarray:
    elo_home = 1.0 / (1.0 + np.power(10.0, -frame["elo_diff"].to_numpy(dtype=float) / 400.0))
    draw = np.clip(0.27 - np.abs(elo_home - 0.5) * 0.20, 0.15, 0.30)
    return np.column_stack([(1.0 - draw) * elo_home, draw, (1.0 - draw) * (1.0 - elo_home)])


def backtest_component_arrays(
    models: dict[str, object],
    frame: pd.DataFrame,
    rho: float = DEFAULT_DIXON_COLES_RHO,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    x = frame[BASE_FEATURES]
    y = frame["target_1x2"].astype(int).to_numpy()
    home_lam = np.clip(models["home_goals_poisson"].predict(x), 0.05, 5.5)
    away_lam = np.clip(models["away_goals_poisson"].predict(x), 0.05, 5.5)
    poisson_probs = lambdas_to_prob_array(home_lam, away_lam, rho=rho)
    count_home_lam = np.clip(models["home_goals_xgb_count"].predict(x), 0.05, 5.5)
    count_away_lam = np.clip(models["away_goals_xgb_count"].predict(x), 0.05, 5.5)
    components = {
        "xgb": models["xgb_1x2"].predict_proba(x),
        "competitive": models["competitive_xgb_1x2"].predict_proba(x),
        "logistic": models["logistic_1x2"].predict_proba(x),
        "elo": elo_policy_probs_from_frame(frame),
        "poisson": poisson_probs,
        "count_poisson": lambdas_to_prob_array(count_home_lam, count_away_lam, rho=rho),
    }
    return y, components, poisson_probs


def backtest_policy_arrays(
    models: dict[str, object],
    frame: pd.DataFrame,
    rho: float = DEFAULT_DIXON_COLES_RHO,
    use_stacking: bool = False,
    manual_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = frame[BASE_FEATURES]
    y, stack, poisson_probs = backtest_component_arrays(models, frame, rho=rho)
    if use_stacking and models.get("stacking_meta_1x2") is not None:
        classifier_probs = models["stacking_meta_1x2"].predict_proba(build_stack_matrix(models, x, frame))
    else:
        classifier_probs = blend_component_probability_arrays(stack, manual_weights or DEFAULT_MANUAL_BLEND_WEIGHTS)
    classifier_probs = np.clip(classifier_probs, 0.001, 0.998)
    classifier_probs = classifier_probs / np.clip(classifier_probs.sum(axis=1, keepdims=True), 0.001, None)
    return y, classifier_probs, poisson_probs


def run_world_cup_backtest(training: pd.DataFrame) -> dict[str, object]:
    cup_years = [1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
    folds = []
    rows = []
    for year in cup_years:
        test = training[(training["date"].dt.year == year) & (training["tournament"].astype(str) == "World Cup")].copy()
        if test.empty:
            continue
        train = training[training["date"] < test["date"].min()].copy()
        if len(train) < 1000:
            continue
        models = fit_backtest_models(train)
        fold_rho = estimate_backtest_models_rho(models, train)
        fold_metrics: dict[str, dict[str, float]] = {}

        x_test = test[BASE_FEATURES]
        y_test = test["target_1x2"].astype(int)
        elo_pred = np.where(test["elo_diff"].abs() < 35, 1, np.where(test["elo_diff"] > 0, 0, 2))
        elo_probs = np.full((len(test), 3), 0.12)
        for idx, pred in enumerate(elo_pred):
            elo_probs[idx, pred] = 0.76
        fold_metrics["baseline_elo_1x2"] = evaluate_1x2_metrics(y_test, elo_probs)

        for name in ["logistic_1x2", "xgb_1x2"]:
            probs = models[name].predict_proba(x_test)
            fold_metrics[name] = evaluate_1x2_metrics(y_test, probs)
        calibration_val = train[train["date"] >= train["date"].max() - pd.Timedelta(days=730)].copy()
        if len(calibration_val) > 100:
            temp = learn_temperature(models["xgb_1x2"].predict_proba(calibration_val[BASE_FEATURES]), calibration_val["target_1x2"])
            fold_metrics["xgb_temperature_calibrated_1x2"] = evaluate_1x2_metrics(
                y_test,
                temperature_scale_probs(models["xgb_1x2"].predict_proba(x_test), temp),
            )
            fold_metrics["xgb_temperature_calibrated_1x2"]["temperature"] = round(float(temp), 4)

        home_lam = np.clip(models["home_goals_poisson"].predict(x_test), 0.05, 5.5)
        away_lam = np.clip(models["away_goals_poisson"].predict(x_test), 0.05, 5.5)
        poisson_probs = []
        for h_lam, a_lam in zip(home_lam, away_lam):
            poisson_probs.append(score_probs_from_lambdas(float(h_lam), float(a_lam), rho=fold_rho))
        poisson_probs = np.array(poisson_probs)
        fold_metrics["poisson_goal_model_1x2"] = evaluate_1x2_metrics(y_test, poisson_probs)
        fold_metrics["poisson_goal_model_1x2"]["home_goal_mae"] = round(float(mean_absolute_error(test["home_score"], home_lam)), 4)
        fold_metrics["poisson_goal_model_1x2"]["away_goal_mae"] = round(float(mean_absolute_error(test["away_score"], away_lam)), 4)

        count_home_lam = np.clip(models["home_goals_xgb_count"].predict(x_test), 0.05, 5.5)
        count_away_lam = np.clip(models["away_goals_xgb_count"].predict(x_test), 0.05, 5.5)
        count_probs = lambdas_to_prob_array(count_home_lam, count_away_lam, rho=fold_rho)
        fold_metrics["xgb_count_poisson_1x2"] = evaluate_1x2_metrics(y_test, count_probs)
        fold_metrics["xgb_count_poisson_1x2"]["home_goal_mae"] = round(float(mean_absolute_error(test["home_score"], count_home_lam)), 4)
        fold_metrics["xgb_count_poisson_1x2"]["away_goal_mae"] = round(float(mean_absolute_error(test["away_score"], count_away_lam)), 4)

        if "stacking_meta_1x2" in models:
            stack_probs = models["stacking_meta_1x2"].predict_proba(build_stack_matrix(models, x_test, test))
            fold_metrics["stacking_meta_1x2"] = evaluate_1x2_metrics(y_test, stack_probs)

        inner_val = train[train["date"] >= train["date"].max() - pd.Timedelta(days=730)].copy()
        inner_train = train[train["date"] < train["date"].max() - pd.Timedelta(days=730)].copy()
        if len(inner_train) > 1000 and len(inner_val) > 100:
            inner_models = fit_backtest_models(inner_train)
            inner_rho = estimate_backtest_models_rho(inner_models, inner_train)
            y_inner, pre_inner, poi_inner = backtest_policy_arrays(inner_models, inner_val, rho=inner_rho)
            selected_policy, _policy_candidates = select_simulation_policy_from_arrays(y_inner, pre_inner, poi_inner)
            y_outer, pre_outer, poi_outer = backtest_policy_arrays(models, test, rho=fold_rho)
            policy_metrics = score_simulation_policy_candidate(
                y_outer,
                pre_outer,
                poi_outer,
                float(selected_policy["classifier_weight"]),
                float(selected_policy["draw_floor"]),
                float(selected_policy["draw_ceiling"]),
            )
            fold_metrics["hybrid_nested_policy_1x2"] = {
                **policy_metrics,
                "inner_train_rows": int(len(inner_train)),
                "inner_rows": int(len(inner_val)),
                "inner_rho": round(float(inner_rho), 4),
                "outer_rho": round(float(fold_rho), 4),
            }

        fold = {
            "cup_year": year,
            "train_start": str(train["date"].min().date()),
            "train_end": str(train["date"].max().date()),
            "test_rows": int(len(test)),
            "metrics": fold_metrics,
        }
        folds.append(fold)
        for model_name, metric in fold_metrics.items():
            rows.append({"cup_year": year, "model": model_name, "test_rows": int(len(test)), **metric})

    fold_df = pd.DataFrame(rows)
    aggregate: dict[str, dict[str, float]] = {}
    if not fold_df.empty:
        metric_cols = [
            col
            for col in fold_df.columns
            if col not in {"cup_year", "model", "test_rows"} and pd.api.types.is_numeric_dtype(fold_df[col])
        ]
        for model_name, model_df in fold_df.groupby("model"):
            total = model_df["test_rows"].sum()
            aggregate[model_name] = {
                col: round(float((model_df[col] * model_df["test_rows"]).sum() / total), 4)
                for col in metric_cols
                if model_df[col].notna().any()
            }
    return {"version": "worldcup_2026_sota_v2_backtest", "folds": folds, "aggregate_weighted_by_matches": aggregate, "rows": rows}


def prepare_match_features(package: dict[str, object], home: str, away: str, neutral: bool = True, tournament: str = "FIFA World Cup") -> pd.DataFrame:
    states = ensure_states(package)
    h = canonical_team(home)
    a = canonical_team(away)
    states.setdefault(h, RunningTeam())
    states.setdefault(a, RunningTeam())
    row = base_features(states, h, a, neutral, tournament)
    ranks = package.get("latest_rankings", {})
    h_rank = ranks.get(h, {"rank": 120.0, "points": 1000.0, "rank_change": 0.0})
    a_rank = ranks.get(a, {"rank": 120.0, "points": 1000.0, "rank_change": 0.0})
    row["fifa_rank_diff"] = a_rank["rank"] - h_rank["rank"]
    row["fifa_points_diff"] = h_rank["points"] - a_rank["points"]
    row["fifa_rank_change_diff"] = a_rank["rank_change"] - h_rank["rank_change"]
    external_elo = package.get("latest_external_elo", {})
    h_ext = float(external_elo.get(h, states[h].elo))
    a_ext = float(external_elo.get(a, states[a].elo))
    row["external_elo_diff"] = h_ext - a_ext
    row["external_elo_abs_diff"] = abs(h_ext - a_ext)
    conf_strength = {"UEFA": 1.00, "CONMEBOL": 0.95, "CONCACAF": 0.78, "CAF": 0.76, "AFC": 0.72, "OFC": 0.58, "UNKNOWN": 0.70}
    confs = package.get("latest_confederations", {})
    h_conf = str(confs.get(h, "UNKNOWN"))
    a_conf = str(confs.get(a, "UNKNOWN"))
    row["same_confederation"] = 1.0 if h_conf == a_conf else 0.0
    row["home_conf_strength"] = conf_strength.get(h_conf, conf_strength["UNKNOWN"])
    row["away_conf_strength"] = conf_strength.get(a_conf, conf_strength["UNKNOWN"])
    row["conf_strength_diff"] = row["home_conf_strength"] - row["away_conf_strength"]
    return pd.DataFrame([row], columns=BASE_FEATURES)


def fixture_context(game: object, team_context: dict[str, dict[str, object]] | None = None, home: str = "", away: str = "") -> dict[str, float | str]:
    airport = str(getattr(game, "airport_code", "") or "")
    coords = AIRPORT_COORDS.get(airport)
    kickoff = getattr(game, "kickoff_at", pd.NaT)
    host_country = str(getattr(game, "host_country", "") or "")
    context: dict[str, float | str] = {
        "airport_code": airport,
        "host_country": host_country,
        "home_rest_days": 5.0,
        "away_rest_days": 5.0,
        "home_travel_km": 0.0,
        "away_travel_km": 0.0,
    }
    if team_context is None:
        return context
    for side, team in [("home", canonical_team(home)), ("away", canonical_team(away))]:
        previous = team_context.get(team)
        if previous:
            previous_kickoff = previous.get("kickoff_at")
            previous_airport = str(previous.get("airport_code", "") or "")
            if pd.notna(kickoff) and previous_kickoff is not None and pd.notna(previous_kickoff):
                context[f"{side}_rest_days"] = max(1.0, float((kickoff - previous_kickoff).total_seconds() / 86400.0))
            context[f"{side}_travel_km"] = haversine_km(AIRPORT_COORDS.get(previous_airport), coords)
    return context


def update_team_context(team_context: dict[str, dict[str, object]], home: str, away: str, game: object) -> None:
    airport = str(getattr(game, "airport_code", "") or "")
    kickoff = getattr(game, "kickoff_at", pd.NaT)
    for team in [canonical_team(home), canonical_team(away)]:
        team_context[team] = {"airport_code": airport, "kickoff_at": kickoff}


def logistic_context_adjustment(home: str, away: str, context: dict[str, float | str] | None) -> float:
    if not context:
        return 0.0
    home_rest = float(context.get("home_rest_days", 5.0))
    away_rest = float(context.get("away_rest_days", 5.0))
    home_travel = float(context.get("home_travel_km", 0.0))
    away_travel = float(context.get("away_travel_km", 0.0))
    host_country = str(context.get("host_country", "") or "")
    rest_edge = max(-2.0, min(2.0, home_rest - away_rest))
    travel_edge = max(-3500.0, min(3500.0, away_travel - home_travel))
    host_edge = 0.0
    if HOST_HOME_COUNTRY.get(canonical_team(home)) == host_country:
        host_edge += 0.10
    if HOST_HOME_COUNTRY.get(canonical_team(away)) == host_country:
        host_edge -= 0.10
    return rest_edge * 0.020 + travel_edge / 3500.0 * 0.045 + host_edge


def apply_logit_shift(probs: np.ndarray, shift: float) -> np.ndarray:
    adjusted = probs.astype(float).copy()
    adjusted[0] *= math.exp(shift)
    adjusted[2] *= math.exp(-shift)
    adjusted = np.clip(adjusted, 0.005, None)
    return adjusted / adjusted.sum()


def apply_draw_policy_to_probs(
    base_probs: np.ndarray,
    draw_floor: float,
    draw_ceiling: float,
) -> np.ndarray:
    return policy_classifier_probs(np.asarray(base_probs, dtype=float), draw_floor, draw_ceiling)[0]


def simulation_policy_from_package(package: dict[str, object] | None = None) -> dict[str, object]:
    policy = dict(package.get("simulation_policy", {}) if package else {})
    classifier_weight = float(policy.get("classifier_weight", MATCH_CLASSIFIER_WEIGHT))
    draw_floor = float(policy.get("draw_floor", MATCH_DRAW_FLOOR))
    draw_ceiling = float(policy.get("draw_ceiling", MATCH_DRAW_CEILING))
    if draw_floor > draw_ceiling:
        draw_floor, draw_ceiling = draw_ceiling, draw_floor
    return {
        "name": str(policy.get("name", "hybrid_classifier_poisson")),
        "classifier_weight": float(np.clip(classifier_weight, 0.05, 0.95)),
        "poisson_weight": float(np.clip(1.0 - classifier_weight, 0.05, 0.95)),
        "draw_floor": float(np.clip(draw_floor, 0.0, 0.48)),
        "draw_ceiling": float(np.clip(draw_ceiling, 0.05, 0.60)),
        "score_engine": str(policy.get("score_engine", "Poisson/Dixon-Coles")),
        "selected_by": str(policy.get("selected_by", "default_policy")),
        "backtest_metrics": policy.get("backtest_metrics", {}),
        "description": str(
            policy.get(
                "description",
                "Sample 90min outcome from classifier+Poisson blend, then sample a compatible score from the Dixon-Coles matrix.",
            )
        ),
    }


def base_probability_stack(package: dict[str, object], x: pd.DataFrame, h: str, a: str, neutral: bool, p_elo: np.ndarray) -> dict[str, np.ndarray]:
    p_xgb = package["models"]["xgb_1x2"].predict_proba(x)[0]
    if package.get("use_temperature_calibration"):
        p_xgb = temperature_scale_probs(np.array([p_xgb]), float(package.get("xgb_temperature", 1.0)))[0]
    p_log = package["models"]["logistic_1x2"].predict_proba(x)[0]
    p_comp = package["models"]["competitive_xgb_1x2"].predict_proba(x)[0]
    models = package["models"]
    rho = dixon_coles_rho_from_package(package)
    home_lam = float(np.clip(models["home_goals_poisson"].predict(x)[0], 0.05, 5.5))
    away_lam = float(np.clip(models["away_goals_poisson"].predict(x)[0], 0.05, 5.5))
    p_poisson = score_probs_from_lambdas(home_lam, away_lam, rho)
    if "home_goals_xgb_count" in models and "away_goals_xgb_count" in models:
        count_home_lam = float(np.clip(models["home_goals_xgb_count"].predict(x)[0], 0.05, 5.5))
        count_away_lam = float(np.clip(models["away_goals_xgb_count"].predict(x)[0], 0.05, 5.5))
        p_count = score_probs_from_lambdas(count_home_lam, count_away_lam, rho)
    else:
        p_count = p_poisson
    stack = {
        "xgb": p_xgb,
        "logistic": p_log,
        "competitive": p_comp,
        "elo": p_elo,
        "poisson": p_poisson,
        "count_poisson": p_count,
    }
    lchikry_model = models.get("lchikry_xgb_1x2")
    if package.get("use_lchikry_ensemble") and lchikry_model is not None:
        stack["lchikry"] = lchikry_model.predict_proba(lchikry_match_features(package, h, a, neutral))[0]
    return stack


def stack_feature_vector(prob_stack: dict[str, np.ndarray]) -> np.ndarray:
    ordered = ["xgb", "competitive", "logistic", "elo", "poisson", "count_poisson"]
    return np.concatenate([prob_stack[name] for name in ordered if name in prob_stack])


def prediction_cache_lock(package: dict[str, object]) -> threading.RLock:
    lock = package.get("_prediction_cache_lock")
    if lock is None:
        with _PREDICTION_CACHE_LOCK_GUARD:
            lock = package.get("_prediction_cache_lock")
            if lock is None:
                lock = threading.RLock()
                package["_prediction_cache_lock"] = lock
    return lock


def predict_match(
    package: dict[str, object],
    home: str,
    away: str,
    neutral: bool = True,
    knockout: bool = False,
    context: dict[str, float | str] | None = None,
) -> dict[str, float]:
    cache = package.setdefault("prediction_cache", {})
    cache_lock = prediction_cache_lock(package)
    context_key = tuple(sorted((context or {}).items()))
    key = (canonical_team(home), canonical_team(away), bool(neutral), bool(knockout), context_key)
    with cache_lock:
        cached = cache.get(key)
    if cached is not None:
        return cached

    h = canonical_team(home)
    a = canonical_team(away)
    base_cache = package.setdefault("prediction_base_cache", {})
    base_key = (h, a, bool(neutral), bool(knockout))
    with cache_lock:
        base = base_cache.get(base_key)
        if base is None:
            x = prepare_match_features(package, h, a, neutral)
            states = ensure_states(package)
            elo_home = expected_score(states[h].elo, states[a].elo)
            draw = max(0.15, min(0.30, 0.27 - abs(elo_home - 0.5) * 0.20))
            p_elo = np.array([(1 - draw) * elo_home, draw, (1 - draw) * (1 - elo_home)])

            prob_stack = base_probability_stack(package, x, h, a, neutral, p_elo)
            stacking_meta = package["models"].get("stacking_meta_1x2")
            if package.get("use_stacking_ensemble") and stacking_meta is not None:
                probs_90 = stacking_meta.predict_proba([stack_feature_vector(prob_stack)])[0]
            else:
                weights = package.get("manual_blend_weights", DEFAULT_MANUAL_BLEND_WEIGHTS)
                probs_90 = sum(float(weights.get(name, 0.0)) * probs for name, probs in prob_stack.items() if name in weights)
            diffs = squad_diffs(package["squad_strength"], h, a)
            squad_adjust = np.array([diffs["squad_top26_diff"] * 0.007, -abs(diffs["squad_top26_diff"]) * 0.0015, -diffs["squad_top26_diff"] * 0.007])
            transfermarkt_adjust = np.array(
                [
                    diffs["tm_market_value_log_diff"] * 0.010 + diffs["tm_caps_diff"] * 0.004 - diffs["tm_recent_injury_days_diff"] * 0.003 - diffs["tm_injury_value_log_diff"] * 0.003,
                    -abs(diffs["tm_market_value_log_diff"]) * 0.001,
                    -diffs["tm_market_value_log_diff"] * 0.010 - diffs["tm_caps_diff"] * 0.004 + diffs["tm_recent_injury_days_diff"] * 0.003 + diffs["tm_injury_value_log_diff"] * 0.003,
                ]
            )
            base_probs_90 = np.clip(probs_90 + squad_adjust + transfermarkt_adjust, 0.01, 0.98)
            base_probs_90 = base_probs_90 / base_probs_90.sum()
            winner_x = package["models"]["winner_xgb"].predict_proba(x)[0]
            static_strength_term = (
                diffs["squad_top26_diff"] * 0.018
                + diffs["attack_strength_diff"] * 0.010
                + diffs["midfield_strength_diff"] * 0.005
                - diffs["defense_strength_diff"] * 0.006
                + diffs["tm_market_value_log_diff"] * 0.020
                + diffs["tm_caps_diff"] * 0.010
                - diffs["tm_recent_injury_days_diff"] * 0.010
                - diffs["tm_injury_value_log_diff"] * 0.008
            )
            model_home_xg = None
            model_away_xg = None
            if "home_goals_poisson" in package["models"] and "away_goals_poisson" in package["models"]:
                model_home_xg = float(np.clip(package["models"]["home_goals_poisson"].predict(x)[0], 0.15, 4.5))
                model_away_xg = float(np.clip(package["models"]["away_goals_poisson"].predict(x)[0], 0.15, 4.5))
            base = {
                "probs_90": base_probs_90,
                "prob_stack": prob_stack,
                "diffs": diffs,
                "elo_home": float(elo_home),
                "p_home_winner_model": float(winner_x[0]),
                "home_avg_for": float(states[h].avg_for),
                "home_avg_against": float(states[h].avg_against),
                "away_avg_for": float(states[a].avg_for),
                "away_avg_against": float(states[a].avg_against),
                "static_strength_term": float(static_strength_term),
                "model_home_xg": model_home_xg,
                "model_away_xg": model_away_xg,
            }
            base_cache[base_key] = base

    prob_stack = base["prob_stack"]
    diffs = base["diffs"]
    elo_home = float(base["elo_home"])
    context_shift = logistic_context_adjustment(h, a, context)
    probs_90 = np.array(base["probs_90"], dtype=float)
    probs_90 = apply_logit_shift(probs_90, context_shift)
    pre_draw_probs_90 = probs_90.copy()
    sim_policy = simulation_policy_from_package(package)
    probs_90 = apply_draw_policy_to_probs(
        pre_draw_probs_90,
        float(sim_policy["draw_floor"]),
        float(sim_policy["draw_ceiling"]),
    )

    p_home_winner_model = float(base["p_home_winner_model"])
    no_draw_total = probs_90[0] + probs_90[2]
    p_home_no_draw = float(probs_90[0] / no_draw_total) if no_draw_total else 0.5
    p_home_advances_if_draw = 0.48 * p_home_winner_model + 0.32 * p_home_no_draw + 0.20 * elo_home
    p_home_advances = float(probs_90[0] + probs_90[1] * p_home_advances_if_draw)
    p_away_advances = 1.0 - p_home_advances

    strength_term = float(base["static_strength_term"]) + context_shift * 0.18
    # xG must stay independent from the simulation policy being calibrated.
    # The calibrated policy controls 1X2 sampling; xG remains a base match signal.
    home_xg = max(
        0.15,
        1.12 + float(base["home_avg_for"]) * 0.32 - float(base["away_avg_against"]) * 0.12 + (pre_draw_probs_90[0] - pre_draw_probs_90[2]) * 0.92 + strength_term,
    )
    away_xg = max(
        0.15,
        1.08 + float(base["away_avg_for"]) * 0.32 - float(base["home_avg_against"]) * 0.12 + (pre_draw_probs_90[2] - pre_draw_probs_90[0]) * 0.92 - strength_term,
    )
    model_home_xg = base.get("model_home_xg")
    model_away_xg = base.get("model_away_xg")
    if model_home_xg is not None and model_away_xg is not None:
        home_xg = 0.55 * home_xg + 0.45 * model_home_xg
        away_xg = 0.55 * away_xg + 0.45 * model_away_xg

    result = {
        "p_home_win_90": float(probs_90[0]),
        "p_draw_90": float(probs_90[1]),
        "p_away_win_90": float(probs_90[2]),
        "p_xgb_home_win_90": float(prob_stack["xgb"][0]),
        "p_xgb_draw_90": float(prob_stack["xgb"][1]),
        "p_xgb_away_win_90": float(prob_stack["xgb"][2]),
        "p_pre_draw_home_win_90": float(pre_draw_probs_90[0]),
        "p_pre_draw_draw_90": float(pre_draw_probs_90[1]),
        "p_pre_draw_away_win_90": float(pre_draw_probs_90[2]),
        "p_poisson_home_win_90": float(prob_stack["poisson"][0]),
        "p_poisson_draw_90": float(prob_stack["poisson"][1]),
        "p_poisson_away_win_90": float(prob_stack["poisson"][2]),
        "p_home_advances": p_home_advances,
        "p_away_advances": p_away_advances,
        "home_xg": float(home_xg),
        "away_xg": float(away_xg),
        "context_shift": float(context_shift),
        "home_rest_days": float((context or {}).get("home_rest_days", 5.0)),
        "away_rest_days": float((context or {}).get("away_rest_days", 5.0)),
        "home_travel_km": float((context or {}).get("home_travel_km", 0.0)),
        "away_travel_km": float((context or {}).get("away_travel_km", 0.0)),
        **diffs,
    }
    with cache_lock:
        cache[key] = result
    return result


def poisson_sample(lam: float, rng: random.Random) -> int:
    weights = [math.exp(-lam) * lam**k / math.factorial(k) for k in range(8)]
    return int(rng.choices(range(8), weights=weights)[0])


def dixon_coles_tau(home_goals: int, away_goals: int, home_xg: float, away_xg: float, rho: float = DEFAULT_DIXON_COLES_RHO) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1 - home_xg * away_xg * rho
    if home_goals == 0 and away_goals == 1:
        return 1 + home_xg * rho
    if home_goals == 1 and away_goals == 0:
        return 1 + away_xg * rho
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def score_matrix(home_xg: float, away_xg: float, max_goals: int = 7, rho: float = DEFAULT_DIXON_COLES_RHO) -> np.ndarray:
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        hp = math.exp(-home_xg) * home_xg**h / math.factorial(h)
        for a in range(max_goals + 1):
            ap = math.exp(-away_xg) * away_xg**a / math.factorial(a)
            matrix[h, a] = hp * ap * dixon_coles_tau(h, a, home_xg, away_xg, rho=rho)
    matrix = np.clip(matrix, 0.0, None)
    return matrix / matrix.sum()


def score_probs_from_lambdas(home_xg: float, away_xg: float, rho: float = DEFAULT_DIXON_COLES_RHO) -> np.ndarray:
    matrix = score_matrix(home_xg, away_xg, rho=rho)
    return np.array([float(np.tril(matrix, -1).sum()), float(np.trace(matrix)), float(np.triu(matrix, 1).sum())])


def lambdas_to_prob_array(home_lam: np.ndarray, away_lam: np.ndarray, rho: float = DEFAULT_DIXON_COLES_RHO) -> np.ndarray:
    return np.array([score_probs_from_lambdas(float(h), float(a), rho=rho) for h, a in zip(home_lam, away_lam)])


def dixon_coles_rho_from_package(package: dict[str, object] | None) -> float:
    if not package:
        return DEFAULT_DIXON_COLES_RHO
    return float(package.get("dixon_coles_rho", DEFAULT_DIXON_COLES_RHO))


def estimate_dixon_coles_rho(frame: pd.DataFrame, home_lam: np.ndarray, away_lam: np.ndarray) -> float:
    if frame.empty:
        return DEFAULT_DIXON_COLES_RHO
    y = frame["target_1x2"].astype(int).to_numpy()
    best_rho = DEFAULT_DIXON_COLES_RHO
    best_loss = float("inf")
    for rho in np.linspace(-0.18, 0.08, 14):
        probs = lambdas_to_prob_array(home_lam, away_lam, rho=float(rho))
        value = float(log_loss(y, probs, labels=[0, 1, 2]))
        if value < best_loss:
            best_loss = value
            best_rho = float(rho)
    return best_rho


def estimate_backtest_models_rho(models: dict[str, object], frame: pd.DataFrame) -> float:
    if frame.empty:
        return DEFAULT_DIXON_COLES_RHO
    x = frame[BASE_FEATURES]
    home_lam = np.clip(models["home_goals_poisson"].predict(x), 0.05, 5.5)
    away_lam = np.clip(models["away_goals_poisson"].predict(x), 0.05, 5.5)
    return estimate_dixon_coles_rho(frame, home_lam, away_lam)


def sample_score(home_xg: float, away_xg: float, rng: random.Random, rho: float = DEFAULT_DIXON_COLES_RHO) -> tuple[int, int]:
    matrix = score_matrix(home_xg, away_xg, rho=rho)
    flat_index = rng.choices(range(matrix.size), weights=matrix.ravel())[0]
    return int(flat_index // matrix.shape[1]), int(flat_index % matrix.shape[1])


def outcome_probs_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(np.tril(matrix, -1).sum()),
            float(np.trace(matrix)),
            float(np.triu(matrix, 1).sum()),
        ]
    )


def hybrid_outcome_probs(classifier_probs: np.ndarray, matrix: np.ndarray, classifier_weight: float = MATCH_CLASSIFIER_WEIGHT) -> tuple[np.ndarray, np.ndarray]:
    classifier = np.clip(np.asarray(classifier_probs, dtype=float), 0.001, 0.998)
    classifier = classifier / classifier.sum()
    poisson = outcome_probs_from_matrix(matrix)
    poisson = poisson / max(0.001, float(poisson.sum()))
    blend = classifier_weight * classifier + (1.0 - classifier_weight) * poisson
    return blend / max(0.001, float(blend.sum())), poisson


def classifier_probs_from_prediction(pred: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            float(pred["p_home_win_90"]),
            float(pred["p_draw_90"]),
            float(pred["p_away_win_90"]),
        ]
    )


def raw_xgb_probs_from_prediction(pred: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            float(pred.get("p_xgb_home_win_90", pred["p_home_win_90"])),
            float(pred.get("p_xgb_draw_90", pred["p_draw_90"])),
            float(pred.get("p_xgb_away_win_90", pred["p_away_win_90"])),
        ]
    )


def hybrid_score_choice(
    matrix: np.ndarray,
    classifier_probs: np.ndarray,
    rng: random.Random | None = None,
    classifier_weight: float = MATCH_CLASSIFIER_WEIGHT,
) -> tuple[int, int, dict[str, float]]:
    blend, poisson = hybrid_outcome_probs(classifier_probs, matrix, classifier_weight)
    if rng is None:
        outcome = int(np.argmax(blend))
    else:
        outcome = int(rng.choices([0, 1, 2], weights=blend)[0])
    candidates: list[tuple[int, int, float]] = []
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            if outcome == 0 and home_goals <= away_goals:
                continue
            if outcome == 1 and home_goals != away_goals:
                continue
            if outcome == 2 and home_goals >= away_goals:
                continue
            candidates.append((home_goals, away_goals, float(matrix[home_goals, away_goals])))
    if not candidates:
        if rng is None:
            flat_index = int(np.argmax(matrix))
        else:
            flat_index = rng.choices(range(matrix.size), weights=matrix.ravel())[0]
        home_goals = int(flat_index // matrix.shape[1])
        away_goals = int(flat_index % matrix.shape[1])
        outcome = label_1x2(home_goals, away_goals)
        score_probability = float(matrix[home_goals, away_goals])
    else:
        weights = [score_probability for _home, _away, score_probability in candidates]
        if rng is None:
            home_goals, away_goals, raw_probability = max(candidates, key=lambda item: item[2])
        else:
            selected = rng.choices(range(len(candidates)), weights=weights)[0]
            home_goals, away_goals, raw_probability = candidates[selected]
        score_probability = float(raw_probability / max(0.001, sum(weights)))
    meta = {
        "sim_outcome": float(outcome),
        "sim_outcome_probability": float(blend[outcome]),
        "sim_score_probability": score_probability,
        "sim_classifier_home": float(classifier_probs[0]),
        "sim_classifier_draw": float(classifier_probs[1]),
        "sim_classifier_away": float(classifier_probs[2]),
        "sim_blend_home": float(blend[0]),
        "sim_blend_draw": float(blend[1]),
        "sim_blend_away": float(blend[2]),
        "sim_poisson_home": float(poisson[0]),
        "sim_poisson_draw": float(poisson[1]),
        "sim_poisson_away": float(poisson[2]),
        "sim_classifier_weight": float(classifier_weight),
    }
    return int(home_goals), int(away_goals), meta


def sample_hybrid_score(
    pred: dict[str, float],
    rng: random.Random,
    rho: float = DEFAULT_DIXON_COLES_RHO,
    classifier_weight: float | None = None,
    policy: dict[str, object] | None = None,
) -> tuple[int, int, dict[str, float]]:
    matrix = score_matrix(float(pred["home_xg"]), float(pred["away_xg"]), rho=rho)
    resolved_policy = dict(policy or {})
    resolved_weight = float(resolved_policy.get("classifier_weight", classifier_weight if classifier_weight is not None else MATCH_CLASSIFIER_WEIGHT))
    return hybrid_score_choice(matrix, classifier_probs_from_prediction(pred), rng, resolved_weight)


def group_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["pts", "gd", "gf", "model_rating"], ascending=[False, False, False, False])


def rank_group(group_df: pd.DataFrame, match_results: list[dict[str, object]]) -> pd.DataFrame:
    ranked = group_df.copy()
    ranked["h2h_pts"] = 0
    ranked["h2h_gd"] = 0
    ranked["h2h_gf"] = 0
    tied_cols = ["pts", "gd", "gf"]
    for _key, tied in ranked.groupby(tied_cols, sort=False):
        if len(tied) <= 1:
            continue
        tied_teams = set(tied["team"])
        h2h = {team: {"pts": 0, "gd": 0, "gf": 0} for team in tied_teams}
        for result in match_results:
            home = result["home"]
            away = result["away"]
            if home not in tied_teams or away not in tied_teams:
                continue
            hg = int(result["home_goals"])
            ag = int(result["away_goals"])
            h2h[home]["gf"] += hg
            h2h[home]["gd"] += hg - ag
            h2h[away]["gf"] += ag
            h2h[away]["gd"] += ag - hg
            if hg > ag:
                h2h[home]["pts"] += 3
            elif ag > hg:
                h2h[away]["pts"] += 3
            else:
                h2h[home]["pts"] += 1
                h2h[away]["pts"] += 1
        for team, vals in h2h.items():
            mask = ranked["team"] == team
            ranked.loc[mask, "h2h_pts"] = vals["pts"]
            ranked.loc[mask, "h2h_gd"] = vals["gd"]
            ranked.loc[mask, "h2h_gf"] = vals["gf"]
    return ranked.sort_values(
        ["pts", "gd", "gf", "wins", "h2h_pts", "h2h_gd", "h2h_gf", "model_rating", "team"],
        ascending=[False, False, False, False, False, False, False, False, True],
    )


def select_best_thirds(thirds: pd.DataFrame) -> pd.DataFrame:
    return thirds.sort_values(["pts", "gd", "gf", "wins", "model_rating", "team"], ascending=[False, False, False, False, False, True]).head(8)


def simulate_group_stage(package: dict[str, object], fixtures: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, str], list[str], dict[str, dict[str, object]]]:
    rng = random.Random(seed)
    teams = package["squad_strength"]
    table = {
        row.team_key: {"team": row.team_key, "group": row.group_letter, "pts": 0, "wins": 0, "gf": 0, "ga": 0, "gd": 0, "model_rating": 0.0}
        for row in teams.itertuples(index=False)
    }
    group_games = fixtures[fixtures["stage_id"] == GROUP_STAGE_ID]
    states = ensure_states(package)
    match_results: dict[str, list[dict[str, object]]] = defaultdict(list)
    team_context: dict[str, dict[str, object]] = {}
    for game in group_games.sort_values("kickoff_at").itertuples(index=False):
        h = canonical_team(game.home_team)
        a = canonical_team(game.away_team)
        context = fixture_context(game, team_context, h, a)
        pred = predict_match(package, h, a, context=context)
        hg, ag, sim_meta = sample_hybrid_score(pred, rng, rho=dixon_coles_rho_from_package(package), policy=simulation_policy_from_package(package))
        update_team_context(team_context, h, a, game)
        match_results[str(game.group)].append({"home": h, "away": a, "home_goals": hg, "away_goals": ag, **sim_meta})
        table[h]["gf"] += hg
        table[h]["ga"] += ag
        table[a]["gf"] += ag
        table[a]["ga"] += hg
        table[h]["model_rating"] = states[h].elo
        table[a]["model_rating"] = states[a].elo
        if hg > ag:
            table[h]["pts"] += 3
            table[h]["wins"] += 1
        elif ag > hg:
            table[a]["pts"] += 3
            table[a]["wins"] += 1
        else:
            table[h]["pts"] += 1
            table[a]["pts"] += 1
    for row in table.values():
        row["gd"] = row["gf"] - row["ga"]
    standings = pd.DataFrame(table.values())
    group_rankings = {}
    thirds = []
    for _group, group_df in standings.groupby("group", sort=True):
        ranked = rank_group(group_df, match_results[str(_group)]).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        group_rankings[str(_group)] = ranked
        thirds.append(ranked.iloc[2].to_dict())
    third_df = select_best_thirds(pd.DataFrame(thirds)).reset_index(drop=True)
    qualifiers: dict[str, str] = {}
    for group, ranked in group_rankings.items():
        qualifiers[f"1{group}"] = str(ranked.iloc[0]["team"])
        qualifiers[f"2{group}"] = str(ranked.iloc[1]["team"])
    third_order = []
    for row in third_df.itertuples(index=False):
        qualifiers[f"3{row.group}"] = str(row.team)
        third_order.append(str(row.group))
    standings = pd.concat(group_rankings.values(), ignore_index=True)
    return standings, group_rankings, qualifiers, third_order, team_context


def knockout_winner(
    package: dict[str, object],
    home: str,
    away: str,
    rng: random.Random,
    context: dict[str, float | str] | None = None,
) -> tuple[str, int, int, str, dict[str, float]]:
    pred = predict_match(package, home, away, knockout=True, context=context)
    rho = dixon_coles_rho_from_package(package)
    hg, ag, sim_meta = sample_hybrid_score(pred, rng, rho=rho, policy=simulation_policy_from_package(package))
    if hg > ag:
        return home, hg, ag, "90min", sim_meta
    if ag > hg:
        return away, hg, ag, "90min", sim_meta
    et_hg, et_ag = sample_score(
        max(0.05, pred["home_xg"] * 0.28),
        max(0.05, pred["away_xg"] * 0.28),
        rng,
        rho=rho,
    )
    if et_hg > et_ag:
        return home, hg + et_hg, ag + et_ag, "extra_time", sim_meta
    if et_ag > et_hg:
        return away, hg + et_hg, ag + et_ag, "extra_time", sim_meta
    penalties_home = 0.50 + (pred["p_home_advances"] - 0.50) * 0.45
    winner = home if rng.random() < penalties_home else away
    return winner, hg, ag, "penalties", sim_meta


def assign_third_slots(slots: list[str], third_order: list[str]) -> dict[str, str]:
    unique_slots = list(dict.fromkeys(slots))
    rank = {group: idx for idx, group in enumerate(third_order)}
    assignment: dict[str, str] = {}
    used: set[str] = set()
    ordered_slots = sorted(unique_slots, key=lambda slot: len(slot[1:]))

    def search(index: int) -> bool:
        if index >= len(ordered_slots):
            return True
        slot = ordered_slots[index]
        candidates = sorted((group for group in slot[1:] if group in third_order and group not in used), key=lambda group: rank[group])
        for group in candidates:
            used.add(group)
            assignment[slot] = f"3{group}"
            if search(index + 1):
                return True
            used.remove(group)
            assignment.pop(slot, None)
        return False

    if not search(0):
        for slot in unique_slots:
            if slot in assignment:
                continue
            for group in third_order:
                key = f"3{group}"
                if key not in assignment.values():
                    assignment[slot] = key
                    break
    return assignment


def resolve_bracket_slot(
    slot: str,
    qualifiers: dict[str, str],
    winners: dict[int, str],
    runners_up: dict[int, str],
    third_slot_assignment: dict[str, str],
) -> str:
    slot = str(slot).strip()
    if not slot:
        raise ValueError("empty bracket slot")
    if slot.startswith("W") and slot[1:].isdigit():
        match_id = int(slot[1:])
        if match_id == 100 and match_id not in winners and 96 in winners:
            match_id = 96
        return winners[match_id]
    if slot.startswith("RU") and slot[2:].isdigit():
        return runners_up[int(slot[2:])]
    if len(slot) == 2 and slot[0] in {"1", "2"}:
        return qualifiers[slot]
    if slot.startswith("3"):
        assigned = third_slot_assignment.get(slot)
        if assigned and assigned in qualifiers:
            return qualifiers[assigned]
    raise KeyError(f"Could not resolve bracket slot {slot!r}")


def parse_match_label(label: str) -> tuple[str, str]:
    parts = [part.strip() for part in str(label).split(" vs ")]
    if len(parts) != 2:
        raise ValueError(f"Unsupported match label: {label!r}")
    return parts[0], parts[1]


def simulate_tournament(package: dict[str, object], seed: int = RANDOM_SEED) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    fixtures = package["fixtures"]
    standings, _group_rankings, qualifiers, third_order, team_context = simulate_group_stage(package, fixtures, seed)
    winners: dict[int, str] = {}
    runners_up: dict[int, str] = {}
    rows = []
    knockout_games = fixtures[fixtures["stage_id"] > GROUP_STAGE_ID].sort_values("match_number")
    round32_slots = []
    for game in knockout_games[knockout_games["stage_id"] == 2].itertuples(index=False):
        round32_slots.extend(slot for slot in parse_match_label(game.match_label) if slot.startswith("3"))
    third_slot_assignment = assign_third_slots(round32_slots, third_order)
    for game in knockout_games.itertuples(index=False):
        left_slot, right_slot = parse_match_label(game.match_label)
        home = resolve_bracket_slot(left_slot, qualifiers, winners, runners_up, third_slot_assignment)
        away = resolve_bracket_slot(right_slot, qualifiers, winners, runners_up, third_slot_assignment)
        context = fixture_context(game, team_context, home, away)
        winner, hg, ag, resolution, sim_meta = knockout_winner(package, home, away, rng, context=context)
        update_team_context(team_context, home, away, game)
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
                "home_goals": hg,
                "away_goals": ag,
                "winner": winner,
                "runner_up": loser,
                "resolution": resolution,
                **sim_meta,
            }
        )
    return winners[104], pd.DataFrame(rows), standings


def rank_group_rows(rows: list[dict[str, object]], match_results: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = [dict(row, h2h_pts=0, h2h_gd=0, h2h_gf=0) for row in rows]
    tied_groups: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in ranked:
        tied_groups[(int(row["pts"]), int(row["gd"]), int(row["gf"]))].append(row)
    for tied in tied_groups.values():
        if len(tied) <= 1:
            continue
        tied_teams = {str(row["team"]) for row in tied}
        h2h = {team: {"pts": 0, "gd": 0, "gf": 0} for team in tied_teams}
        for result in match_results:
            home = str(result["home"])
            away = str(result["away"])
            if home not in tied_teams or away not in tied_teams:
                continue
            hg = int(result["home_goals"])
            ag = int(result["away_goals"])
            h2h[home]["gf"] += hg
            h2h[home]["gd"] += hg - ag
            h2h[away]["gf"] += ag
            h2h[away]["gd"] += ag - hg
            if hg > ag:
                h2h[home]["pts"] += 3
            elif ag > hg:
                h2h[away]["pts"] += 3
            else:
                h2h[home]["pts"] += 1
                h2h[away]["pts"] += 1
        for row in tied:
            row.update({f"h2h_{key}": value for key, value in h2h[str(row["team"])].items()})
    return sorted(
        ranked,
        key=lambda row: (
            -int(row["pts"]),
            -int(row["gd"]),
            -int(row["gf"]),
            -int(row["wins"]),
            -int(row["h2h_pts"]),
            -int(row["h2h_gd"]),
            -int(row["h2h_gf"]),
            -float(row["model_rating"]),
            str(row["team"]),
        ),
    )


def simulate_group_stage_fast(package: dict[str, object], fixtures: pd.DataFrame, seed: int) -> tuple[dict[str, str], list[str], dict[str, dict[str, object]]]:
    rng = random.Random(seed)
    teams = package["squad_strength"]
    table = {
        row.team_key: {"team": row.team_key, "group": row.group_letter, "pts": 0, "wins": 0, "gf": 0, "ga": 0, "gd": 0, "model_rating": 0.0}
        for row in teams.itertuples(index=False)
    }
    group_games = fixtures[fixtures["stage_id"] == GROUP_STAGE_ID]
    states = ensure_states(package)
    match_results: dict[str, list[dict[str, object]]] = defaultdict(list)
    team_context: dict[str, dict[str, object]] = {}
    for game in group_games.sort_values("kickoff_at").itertuples(index=False):
        h = canonical_team(game.home_team)
        a = canonical_team(game.away_team)
        context = fixture_context(game, team_context, h, a)
        pred = predict_match(package, h, a, context=context)
        hg, ag, _sim_meta = sample_hybrid_score(pred, rng, rho=dixon_coles_rho_from_package(package), policy=simulation_policy_from_package(package))
        update_team_context(team_context, h, a, game)
        match_results[str(game.group)].append({"home": h, "away": a, "home_goals": hg, "away_goals": ag})
        table[h]["gf"] += hg
        table[h]["ga"] += ag
        table[a]["gf"] += ag
        table[a]["ga"] += hg
        table[h]["model_rating"] = states[h].elo
        table[a]["model_rating"] = states[a].elo
        if hg > ag:
            table[h]["pts"] += 3
            table[h]["wins"] += 1
        elif ag > hg:
            table[a]["pts"] += 3
            table[a]["wins"] += 1
        else:
            table[h]["pts"] += 1
            table[a]["pts"] += 1
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in table.values():
        row["gd"] = int(row["gf"]) - int(row["ga"])
        groups[str(row["group"])].append(row)

    qualifiers: dict[str, str] = {}
    thirds: list[dict[str, object]] = []
    for group, rows in groups.items():
        ranked = rank_group_rows(rows, match_results[str(group)])
        qualifiers[f"1{group}"] = str(ranked[0]["team"])
        qualifiers[f"2{group}"] = str(ranked[1]["team"])
        thirds.append(ranked[2])
    best_thirds = sorted(
        thirds,
        key=lambda row: (
            -int(row["pts"]),
            -int(row["gd"]),
            -int(row["gf"]),
            -int(row["wins"]),
            -float(row["model_rating"]),
            str(row["team"]),
        ),
    )[:8]
    third_order = []
    for row in best_thirds:
        qualifiers[f"3{row['group']}"] = str(row["team"])
        third_order.append(str(row["group"]))
    return qualifiers, third_order, team_context


def simulate_tournament_champion(package: dict[str, object], seed: int = RANDOM_SEED) -> str:
    rng = random.Random(seed)
    fixtures = package["fixtures"]
    qualifiers, third_order, team_context = simulate_group_stage_fast(package, fixtures, seed)
    winners: dict[int, str] = {}
    runners_up: dict[int, str] = {}
    knockout_games = fixtures[fixtures["stage_id"] > GROUP_STAGE_ID].sort_values("match_number")
    round32_slots = []
    for game in knockout_games[knockout_games["stage_id"] == 2].itertuples(index=False):
        round32_slots.extend(slot for slot in parse_match_label(game.match_label) if slot.startswith("3"))
    third_slot_assignment = assign_third_slots(round32_slots, third_order)
    for game in knockout_games.itertuples(index=False):
        left_slot, right_slot = parse_match_label(game.match_label)
        home = resolve_bracket_slot(left_slot, qualifiers, winners, runners_up, third_slot_assignment)
        away = resolve_bracket_slot(right_slot, qualifiers, winners, runners_up, third_slot_assignment)
        context = fixture_context(game, team_context, home, away)
        winner, _hg, _ag, _resolution, _sim_meta = knockout_winner(package, home, away, rng, context=context)
        update_team_context(team_context, home, away, game)
        loser = away if winner == home else home
        winners[int(game.match_number)] = winner
        runners_up[int(game.match_number)] = loser
    return winners[104]


def simulate_tournament_champion_story(package: dict[str, object], seed: int = RANDOM_SEED) -> RepresentativeCandidate:
    rng = random.Random(seed)
    fixtures = package["fixtures"]
    qualifiers, third_order, team_context = simulate_group_stage_fast(package, fixtures, seed)
    winners: dict[int, str] = {}
    runners_up: dict[int, str] = {}
    final_candidate: RepresentativeCandidate | None = None
    knockout_games = fixtures[fixtures["stage_id"] > GROUP_STAGE_ID].sort_values("match_number")
    round32_slots = []
    for game in knockout_games[knockout_games["stage_id"] == 2].itertuples(index=False):
        round32_slots.extend(slot for slot in parse_match_label(game.match_label) if slot.startswith("3"))
    third_slot_assignment = assign_third_slots(round32_slots, third_order)
    for game in knockout_games.itertuples(index=False):
        left_slot, right_slot = parse_match_label(game.match_label)
        home = resolve_bracket_slot(left_slot, qualifiers, winners, runners_up, third_slot_assignment)
        away = resolve_bracket_slot(right_slot, qualifiers, winners, runners_up, third_slot_assignment)
        context = fixture_context(game, team_context, home, away)
        winner, hg, ag, resolution, _sim_meta = knockout_winner(package, home, away, rng, context=context)
        update_team_context(team_context, home, away, game)
        loser = away if winner == home else home
        winners[int(game.match_number)] = winner
        runners_up[int(game.match_number)] = loser
        if int(game.match_number) == 104:
            final_candidate = RepresentativeCandidate(
                seed=int(seed),
                champion=str(winner),
                runner_up=str(loser),
                final_home=str(home),
                final_away=str(away),
                home_goals=int(hg),
                away_goals=int(ag),
                resolution=str(resolution),
            )
    if final_candidate is None:
        champion = winners[104]
        runner_up = runners_up[104]
        return RepresentativeCandidate(
            seed=int(seed),
            champion=str(champion),
            runner_up=str(runner_up),
            final_home=str(champion),
            final_away=str(runner_up),
            home_goals=0,
            away_goals=0,
            resolution="unknown",
        )
    return final_candidate


def display_stage_name(stage: str) -> str:
    stage_map = {
        "Quarterfinals": "Quarter-finals",
        "Semifinals": "Semi-finals",
    }
    return stage_map.get(str(stage), str(stage))


def champion_odds_from_counts(counts: dict[str, dict[str, int]], completed_runs: int) -> pd.DataFrame:
    denominator = max(1, completed_runs)
    odds_rows = []
    for team, stats in counts.items():
        odds_rows.append({"team": team, "wins": stats["Champion"], "champion_probability": stats["Champion"] / denominator})
    return pd.DataFrame(odds_rows).sort_values("champion_probability", ascending=False)


def champion_odds_rows(champion_odds: pd.DataFrame | list[tuple[str, int, float]]) -> list[tuple[str, int, float]]:
    if isinstance(champion_odds, list):
        return [(str(team), int(wins), float(probability)) for team, wins, probability in champion_odds]
    return [
        (str(row.team), int(row.wins), float(row.champion_probability))
        for row in champion_odds.itertuples(index=False)
    ]


def monte_carlo_worker_count(workers: int | None = None) -> int:
    if workers is not None:
        return min(MAX_MONTE_CARLO_WORKERS, max(1, int(workers)))
    return MAX_MONTE_CARLO_WORKERS


def accumulate_tournament_counts(
    counts: dict[str, dict[str, int]],
    package: dict[str, object],
    champion: str,
    bracket: pd.DataFrame,
) -> None:
    for team in package["squad_strength"]["team_key"]:
        counts[team]["Group Stage"] += 1
    for row in bracket.itertuples(index=False):
        stage = display_stage_name(row.round)
        counts[row.home][stage] += 1
        counts[row.away][stage] += 1
        counts[row.winner]["advanced_events"] += 1
    counts[champion]["Champion"] += 1


def representative_story_jitter(champion: str, seed: int) -> float:
    champion_key = sum((index + 1) * ord(char) for index, char in enumerate(champion))
    return random.Random(seed * 1009 + champion_key * 37).random()


def record_representative_candidate(
    representative_candidates: dict[str, list[RepresentativeCandidate]] | None,
    candidate: RepresentativeCandidate,
) -> None:
    if representative_candidates is None:
        return
    representative_candidates.setdefault(str(candidate.champion), []).append(candidate)


def finalist_profile(
    representative_candidates: dict[str, list[RepresentativeCandidate]],
) -> tuple[dict[str, int], dict[str, int], dict[str, float], int]:
    finalist_counts: dict[str, int] = defaultdict(int)
    total = 0
    for candidates in representative_candidates.values():
        for candidate in candidates:
            finalist_counts[str(candidate.final_home)] += 1
            finalist_counts[str(candidate.final_away)] += 1
            total += 1
    ranked = sorted(finalist_counts.items(), key=lambda item: (-item[1], item[0]))
    finalist_ranks = {team: rank for rank, (team, _count) in enumerate(ranked, start=1)}
    denominator = max(1, total)
    finalist_probabilities = {team: count / denominator for team, count in finalist_counts.items()}
    return finalist_counts, finalist_ranks, finalist_probabilities, total


def finalist_rank_score(rank: int) -> float:
    if rank <= 0:
        return 0.12
    if rank <= 5:
        return 1.0
    if rank <= REPRESENTATIVE_FINALIST_TOP_N:
        return 0.86
    if rank <= 16:
        return 0.66
    if rank <= 24:
        return 0.42
    return 0.16


def final_score_plausibility(candidate: RepresentativeCandidate) -> float:
    diff_scores = {0: 0.92, 1: 1.0, 2: 0.82, 3: 0.52, 4: 0.24}
    diff_score = diff_scores.get(candidate.goal_diff, 0.08)
    if candidate.total_goals <= 4:
        total_score = 1.0
    elif candidate.total_goals == 5:
        total_score = 0.74
    elif candidate.total_goals == 6:
        total_score = 0.44
    else:
        total_score = 0.18
    return (diff_score * 0.66) + (total_score * 0.34)


def representative_candidate_score(
    candidate: RepresentativeCandidate,
    finalist_ranks: dict[str, int],
    finalist_probabilities: dict[str, float],
) -> float:
    runner_rank = finalist_ranks.get(str(candidate.runner_up), 999)
    runner_final_score = finalist_rank_score(runner_rank)
    leader_final_probability = max(finalist_probabilities.values(), default=1.0)
    runner_probability = finalist_probabilities.get(str(candidate.runner_up), 0.0)
    runner_rate_score = min(1.0, runner_probability / max(0.001, leader_final_probability))
    score_score = final_score_plausibility(candidate)
    seed_jitter = representative_story_jitter(candidate.champion, candidate.seed)
    return (
        runner_final_score * 0.44
        + runner_rate_score * 0.24
        + score_score * 0.24
        + (1.0 - seed_jitter) * 0.08
    )


def surprise_label(finalist_rank: int, goal_diff: int) -> str:
    if finalist_rank <= REPRESENTATIVE_FINALIST_TOP_N and goal_diff <= 2:
        return "plausivel"
    if finalist_rank <= 16 and goal_diff <= 3:
        return "surpresa_controlada"
    return "zebra_controlada"


def select_plausible_representative_story(
    champion_odds: pd.DataFrame | list[tuple[str, int, float]],
    representative_candidates: dict[str, list[RepresentativeCandidate]],
    seed: int,
    pool_size: int = REPRESENTATIVE_TOP_N,
) -> dict[str, object] | None:
    odds = champion_odds_rows(champion_odds)
    champion_candidates = [
        (rank, team, wins, probability)
        for rank, (team, wins, probability) in enumerate(odds[:pool_size], start=1)
        if representative_candidates.get(team) and wins > 0 and probability > 0
    ]
    if not champion_candidates:
        return None

    champion_entropy = seed * 1009 + sum((rank * 37) + wins for rank, _team, wins, _probability in champion_candidates)
    champion_rng = random.Random(champion_entropy)
    rank, team, wins, probability = champion_rng.choices(
        champion_candidates,
        weights=[max(0.001, probability) for _rank, _team, _wins, probability in champion_candidates],
        k=1,
    )[0]

    _finalist_counts, finalist_ranks, finalist_probabilities, total_runs = finalist_profile(representative_candidates)
    scored: list[tuple[float, RepresentativeCandidate]] = []
    for candidate in representative_candidates[str(team)]:
        scored.append((representative_candidate_score(candidate, finalist_ranks, finalist_probabilities), candidate))
    scored.sort(key=lambda item: (-item[0], item[1].goal_diff, item[1].total_goals, item[1].seed))
    story_pool = scored[: max(1, min(REPRESENTATIVE_STORY_POOL_SIZE, len(scored)))]

    story_entropy = champion_entropy + sum(int(score * 10000) + candidate.seed for score, candidate in story_pool)
    story_rng = random.Random(story_entropy)
    story_score, story_candidate = story_rng.choices(
        story_pool,
        weights=[max(0.01, score) ** 2 for score, _candidate in story_pool],
        k=1,
    )[0]
    runner_up = str(story_candidate.runner_up)
    runner_rank = int(finalist_ranks.get(runner_up, 999))
    runner_probability = float(finalist_probabilities.get(runner_up, 0.0))
    return {
        "team": team,
        "rank": rank,
        "wins": wins,
        "probability": probability,
        "leader": odds[0][0] if odds else team,
        "leader_probability": odds[0][2] if odds else probability,
        "seed": int(story_candidate.seed),
        "runner_up": runner_up,
        "runner_up_finalist_rank": runner_rank,
        "runner_up_finalist_probability": runner_probability,
        "final_home": str(story_candidate.final_home),
        "final_away": str(story_candidate.final_away),
        "final_home_goals": int(story_candidate.home_goals),
        "final_away_goals": int(story_candidate.away_goals),
        "final_goal_diff": int(story_candidate.goal_diff),
        "final_total_goals": int(story_candidate.total_goals),
        "final_resolution": str(story_candidate.resolution),
        "plausibility_score": float(story_score),
        "candidate_count": len(representative_candidates[str(team)]),
        "story_pool_size": len(story_pool),
        "sample_runs": int(total_runs),
        "pool": [candidate_team for _rank, candidate_team, _wins, _probability in champion_candidates],
        "policy": f"top{pool_size}_weighted_plausible_story",
        "surprise_level": surprise_label(runner_rank, story_candidate.goal_diff),
    }


def simulate_tournament_seeded(package: dict[str, object], seed: int) -> tuple[int, str, pd.DataFrame, pd.DataFrame]:
    champion, bracket, standings = simulate_tournament(package, seed)
    return seed, champion, bracket, standings


def simulate_tournament_seeded_chunk(package: dict[str, object], start_seed: int, run_count: int) -> list[tuple[int, str, pd.DataFrame, pd.DataFrame]]:
    return [simulate_tournament_seeded(package, start_seed + offset) for offset in range(run_count)]


def simulate_tournament_seeded_champion(package: dict[str, object], seed: int) -> tuple[int, str]:
    return seed, simulate_tournament_champion(package, seed)


def simulate_tournament_seeded_champion_chunk(package: dict[str, object], start_seed: int, run_count: int) -> list[tuple[int, str]]:
    return [simulate_tournament_seeded_champion(package, start_seed + offset) for offset in range(run_count)]


def simulate_tournament_seeded_champion_story(package: dict[str, object], seed: int) -> RepresentativeCandidate:
    return simulate_tournament_champion_story(package, seed)


def simulate_tournament_seeded_champion_story_chunk(package: dict[str, object], start_seed: int, run_count: int) -> list[RepresentativeCandidate]:
    return [simulate_tournament_seeded_champion_story(package, start_seed + offset) for offset in range(run_count)]


def monte_carlo(
    package: dict[str, object],
    runs: int = 1000,
    seed: int = RANDOM_SEED,
    progress_callback: object | None = None,
    workers: int | None = None,
    representative_candidates: dict[str, list[RepresentativeCandidate]] | None = None,
    fast_champion_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = max(1, int(runs))
    worker_count = min(monte_carlo_worker_count(workers), runs)
    fast_champion_only = bool(fast_champion_only)
    stage_order = {"Group Stage": 0, "Round of 32": 1, "Round of 16": 2, "Quarter-finals": 3, "Semi-finals": 4, "Final": 5, "Champion": 6}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    completed_runs = 0
    snapshot_interval = max(1, runs // 100)

    if worker_count <= 1:
        for i in range(runs):
            if fast_champion_only:
                if representative_candidates is not None:
                    candidate = simulate_tournament_seeded_champion_story(package, seed + i)
                    champion = candidate.champion
                    record_representative_candidate(representative_candidates, candidate)
                else:
                    _sim_seed, champion = simulate_tournament_seeded_champion(package, seed + i)
                counts[champion]["Champion"] += 1
            else:
                _sim_seed, champion, bracket, standings = simulate_tournament_seeded(package, seed + i)
                accumulate_tournament_counts(counts, package, champion, bracket)
            completed_runs += 1
            if progress_callback and (completed_runs == runs or completed_runs % snapshot_interval == 0):
                snapshot = champion_odds_from_counts(counts, completed_runs)
                if progress_callback(completed_runs, runs, snapshot) is False:
                    break
    else:
        next_run = 0
        chunk_size = max(1, runs // max(1, worker_count * 8))
        max_pending = max(worker_count, worker_count * 2)
        cancelled = False
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="arena-ai-mc")
        if fast_champion_only and representative_candidates is not None:
            chunk_fn = simulate_tournament_seeded_champion_story_chunk
        else:
            chunk_fn = simulate_tournament_seeded_champion_chunk if fast_champion_only else simulate_tournament_seeded_chunk

        def submit_pending(pending: set[object]) -> None:
            nonlocal next_run
            while next_run < runs and len(pending) < max_pending:
                current_count = min(chunk_size, runs - next_run)
                pending.add(executor.submit(chunk_fn, package, seed + next_run, current_count))
                next_run += current_count

        pending: set[object] = set()
        try:
            submit_pending(pending)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    for result in future.result():
                        if fast_champion_only:
                            if representative_candidates is not None:
                                candidate = result
                                champion = candidate.champion
                                record_representative_candidate(representative_candidates, candidate)
                            else:
                                _sim_seed, champion = result
                            counts[champion]["Champion"] += 1
                        else:
                            _sim_seed, champion, bracket, standings = result
                            accumulate_tournament_counts(counts, package, champion, bracket)
                        completed_runs += 1
                        if progress_callback and (completed_runs == runs or completed_runs % snapshot_interval == 0):
                            snapshot = champion_odds_from_counts(counts, completed_runs)
                            if progress_callback(completed_runs, runs, snapshot) is False:
                                cancelled = True
                                break
                    if cancelled:
                        break
                if cancelled:
                    break
                submit_pending(pending)
        finally:
            if cancelled:
                for future in pending:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

    champion_odds = champion_odds_from_counts(counts, completed_runs)

    stage_rows = []
    for team, stats in counts.items():
        for stage in stage_order:
            if stage == "Champion":
                value = stats["Champion"]
            else:
                value = stats[stage]
            stage_rows.append({"team": team, "stage": stage, "probability": value / max(1, completed_runs)})
    stage_odds = pd.DataFrame(stage_rows)
    return champion_odds, stage_odds


def build_sota_package(runs: int = 10000) -> dict[str, object]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    previous_metrics = {}
    previous_report_path = REPORTS / "sota_model_report.json"
    if previous_report_path.exists():
        try:
            previous_metrics = json.loads(previous_report_path.read_text(encoding="utf-8")).get("metrics", {})
        except json.JSONDecodeError:
            previous_metrics = {}

    training, states = build_training_frame()
    squad = build_squad_strength()
    fixtures = load_fixtures()
    rankings = load_rankings()
    external_elo = load_external_elo()
    external_elo_audit = external_elo_quality(external_elo, fixtures)
    models, metrics, dc_rho, xgb_temperature = train_models(training)
    use_lchikry = (
        "lchikry_external_xgb_1x2" in metrics
        and metrics["lchikry_external_xgb_1x2"]["log_loss"] <= metrics["xgb_1x2"]["log_loss"] + 0.01
    )
    use_stacking = (
        "stacking_meta_1x2" in metrics
        and metrics["stacking_meta_1x2"]["log_loss"] <= metrics["xgb_1x2"]["log_loss"]
        and metrics["stacking_meta_1x2"]["log_loss"] <= metrics["competitive_xgb_1x2"]["log_loss"] + 0.015
    )
    use_temperature = (
        "xgb_temperature_calibrated_1x2" in metrics
        and metrics["xgb_temperature_calibrated_1x2"]["log_loss"] <= metrics["xgb_1x2"]["log_loss"]
    )
    package = {
        "version": "worldcup_2026_sota_v3",
        "base_features": BASE_FEATURES,
        "squad_features": SQUAD_FEATURES,
        "lchikry_features": LCHIKRY_FEATURES,
        "label_map_1x2": {0: "home_win_90", 1: "draw_90", 2: "away_win_90"},
        "label_map_winner": {0: "home_winner", 1: "away_winner"},
        "models": models,
        "states": dict(states),
        "squad_strength": squad,
        "fixtures": fixtures,
        "latest_rankings": latest_rank_lookup(rankings),
        "latest_external_elo": latest_external_elo_lookup(external_elo),
        "external_elo_audit": external_elo_audit,
        "latest_confederations": latest_confederation_lookup(rankings),
        "use_lchikry_ensemble": use_lchikry,
        "use_stacking_ensemble": use_stacking,
        "dixon_coles_rho": dc_rho,
        "xgb_temperature": xgb_temperature,
        "use_temperature_calibration": use_temperature,
        "manual_blend_weights": dict(DEFAULT_MANUAL_BLEND_WEIGHTS),
        "simulation_policy": {
            "name": "hybrid_classifier_poisson",
            "classifier_weight": MATCH_CLASSIFIER_WEIGHT,
            "draw_floor": MATCH_DRAW_FLOOR,
            "draw_ceiling": MATCH_DRAW_CEILING,
            "score_engine": "Poisson/Dixon-Coles",
            "description": "Sample 90min outcome from classifier+Poisson blend, then sample a compatible score from the Dixon-Coles matrix.",
        },
        "prediction_cache": {},
        "notes": [
            "SOTA v3 separates 90-minute 1X2 prediction from mandatory knockout winner prediction.",
            "SOTA v3 adds learned stacking validation, XGB count:poisson goal regressors, Dixon-Coles rho tuning, fixture rest/travel context, and injury value risk.",
            "Group simulation keeps draws; knockout simulation always returns a winner.",
            "Monte Carlo samples each 90-minute score with a hybrid classifier-plus-Poisson policy: XGBoost chooses tendency, Dixon-Coles keeps football variance.",
            "Monte Carlo probabilities are the primary tournament-level output.",
        ],
    }
    package["simulation_policy"] = calibrate_simulation_policy(package, training)
    package["manual_blend_weights"] = normalize_manual_blend_weights(
        dict(package["simulation_policy"].get("manual_blend_weights", package["manual_blend_weights"]))
    )
    package["prediction_cache"] = {}
    nested_policy_rows = package["simulation_policy"].get("nested_temporal_validation", {}).get("rows", [])
    pd.DataFrame(nested_policy_rows).to_csv(REPORTS / "sota_policy_nested_temporal_validation.csv", index=False)

    probs = []
    preview_context: dict[str, dict[str, object]] = {}
    for game in fixtures[fixtures["stage_id"] == GROUP_STAGE_ID].itertuples(index=False):
        context = fixture_context(game, preview_context, game.home_team, game.away_team)
        pred = predict_match(package, game.home_team, game.away_team, context=context)
        update_team_context(preview_context, game.home_team, game.away_team, game)
        probs.append({"match_number": game.match_number, "group": game.group, "home_team": game.home_team, "away_team": game.away_team, **pred})
    match_probs = pd.DataFrame(probs)

    champion, bracket, standings = simulate_tournament(package, RANDOM_SEED)
    champion_odds, stage_odds = monte_carlo(package, runs=runs)
    backtest = run_world_cup_backtest(training)
    metric_gains = {}
    for name, current in metrics.items():
        baseline = previous_metrics.get(name, {})
        deltas = {}
        for key, value in current.items():
            if key in baseline and isinstance(value, (int, float)) and isinstance(baseline[key], (int, float)):
                deltas[key] = round(float(value) - float(baseline[key]), 4)
        if deltas:
            metric_gains[name] = deltas
    metric_gains_vs_v1 = {}
    for name, current in metrics.items():
        baseline = SOTA_V1_BASELINE_METRICS.get(name, {})
        deltas = {}
        for key, value in current.items():
            if key in baseline and isinstance(value, (int, float)):
                deltas[key] = round(float(value) - float(baseline[key]), 4)
        if deltas:
            metric_gains_vs_v1[name] = deltas

    training.to_csv(PROCESSED / "sota_training_matches.csv", index=False)
    squad.to_csv(PROCESSED / "sota_team_strength_2026.csv", index=False)
    fixtures.to_csv(PROCESSED / "sota_fixtures_2026.csv", index=False)
    match_probs.to_csv(REPORTS / "sota_match_probabilities.csv", index=False)
    bracket.to_csv(REPORTS / "sota_tournament_simulation.csv", index=False)
    standings.to_csv(REPORTS / "sota_group_standings_seed.csv", index=False)
    champion_odds.to_csv(REPORTS / "sota_champion_odds.csv", index=False)
    stage_odds.to_csv(REPORTS / "sota_stage_odds.csv", index=False)
    pd.DataFrame(backtest["rows"]).to_csv(REPORTS / "sota_world_cup_backtest_folds.csv", index=False)

    pickle_package = dict(package)
    pickle_package["states"] = {name: state_to_record(state) for name, state in package["states"].items()}
    pickle_package["prediction_cache"] = {}
    pickle_package.pop("_prediction_cache_lock", None)
    with (MODELS / "model_sota.pkl").open("wb") as file:
        pickle.dump(pickle_package, file)

    report = {
        "version": package["version"],
        "training_rows": int(len(training)),
        "training_source": str(training["source_name"].iloc[0]) if "source_name" in training.columns and len(training) else "unknown",
        "train_start": str(training["date"].min().date()),
        "train_end": str(training["date"].max().date()),
        "metrics": metrics,
        "metric_gains_vs_previous_report": metric_gains,
        "metric_gains_vs_sota_v1": metric_gains_vs_v1,
        "accepted_experiments": {
            "pataterie_history": True,
            "external_elo_features": True,
            "transfermarkt_2026_squad_layer": True,
            "poisson_goal_models": True,
            "xgb_count_poisson_goal_models": True,
            "dixon_coles_rho_grid": True,
            "xgb_temperature_calibration": bool(use_temperature),
            "stacking_meta_ensemble": bool(use_stacking),
            "lchikry_ensemble": bool(use_lchikry),
            "official_2026_bracket_slots": True,
            "world_cup_walk_forward_backtest": True,
            "extra_time_penalty_layer": True,
            "fixture_rest_travel_home_context": True,
            "transfermarkt_injury_value_risk": True,
            "confederation_features": True,
            "temporal_importance_sample_weight": True,
        },
        "external_elo_audit": external_elo_audit,
        "world_cup_backtest_aggregate": backtest["aggregate_weighted_by_matches"],
        "dixon_coles_rho": round(float(dc_rho), 4),
        "xgb_temperature": round(float(xgb_temperature), 4),
        "use_temperature_calibration": bool(use_temperature),
        "use_stacking_ensemble": bool(use_stacking),
        "simulation_policy": package["simulation_policy"],
        "monte_carlo_runs": runs,
        "sample_champion_seed_2026": champion,
        "top_10_champion_odds": champion_odds.head(10).to_dict(orient="records"),
        "artifacts": {
            "pickle": str(MODELS / "model_sota.pkl"),
            "report": str(REPORTS / "sota_model_report.json"),
            "champion_odds": str(REPORTS / "sota_champion_odds.csv"),
            "stage_odds": str(REPORTS / "sota_stage_odds.csv"),
            "match_probabilities": str(REPORTS / "sota_match_probabilities.csv"),
            "simulation": str(REPORTS / "sota_tournament_simulation.csv"),
            "world_cup_backtest": str(REPORTS / "sota_world_cup_backtest.json"),
            "policy_nested_temporal_validation": str(REPORTS / "sota_policy_nested_temporal_validation.csv"),
        },
    }
    (REPORTS / "sota_model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    backtest_report = {key: value for key, value in backtest.items() if key != "rows"}
    (REPORTS / "sota_world_cup_backtest.json").write_text(json.dumps(backtest_report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and package the World Cup 2026 SOTA model.")
    parser.add_argument("--runs", type=int, default=10000, help="Monte Carlo tournament simulations used for stage and champion odds.")
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    report = build_sota_package(runs=args.runs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

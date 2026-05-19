from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import pickle
import random
import sys
import threading
from pathlib import Path
from collections.abc import Callable

import numpy as np


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
SOTA_ROOT = ROOT / "modeling" / "worldcup_2026_ml"
SOTA_SRC = SOTA_ROOT / "src"
SOTA_PIPELINE_PATH = SOTA_SRC / "sota_pipeline.py"
MODEL_PATH = SOTA_ROOT / "models" / "model_sota.pkl"
RUNTIME_PREDICTION_CACHE_PATH = SOTA_ROOT / "models" / "runtime_prediction_cache.pkl"
REPORT_PATH = SOTA_ROOT / "reports" / "sota_model_report.json"

if "sota_pipeline" in sys.modules:
    sota = sys.modules["sota_pipeline"]
else:
    spec = importlib.util.spec_from_file_location("sota_pipeline", SOTA_PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar {SOTA_PIPELINE_PATH}")
    sota = importlib.util.module_from_spec(spec)
    sys.modules["sota_pipeline"] = sota
    spec.loader.exec_module(sota)


def effective_monte_carlo_workers(workers: int | None = None) -> int:
    return int(sota.monte_carlo_worker_count(workers))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


BASE_FLAGS = {
    "BRA": ((33, 156, 76), (249, 221, 50), (43, 88, 181)),
    "ARG": ((105, 181, 232), (255, 255, 255), (105, 181, 232)),
    "FRA": ((35, 72, 166), (255, 255, 255), (224, 48, 49)),
    "GER": ((20, 20, 20), (215, 40, 40), (245, 203, 66)),
    "DEU": ((20, 20, 20), (215, 40, 40), (245, 203, 66)),
    "ESP": ((196, 33, 44), (246, 198, 70), (196, 33, 44)),
    "ENG": ((255, 255, 255), (213, 43, 55), (255, 255, 255)),
    "POR": ((20, 145, 82), (220, 39, 49), (245, 210, 73)),
    "PRT": ((20, 145, 82), (220, 39, 49), (245, 210, 73)),
    "ITA": ((0, 146, 70), (245, 245, 245), (206, 43, 55)),
    "BEL": ((20, 20, 20), (250, 214, 61), (216, 43, 55)),
    "MEX": ((0, 104, 71), (245, 245, 245), (206, 17, 38)),
    "USA": ((178, 34, 52), (245, 245, 245), (60, 59, 110)),
    "JPN": ((245, 245, 245), (188, 0, 45), (245, 245, 245)),
    "KOR": ((245, 245, 245), (196, 32, 54), (0, 71, 160)),
    "DEN": ((198, 12, 48), (245, 245, 245), (198, 12, 48)),
    "DNK": ((198, 12, 48), (245, 245, 245), (198, 12, 48)),
    "SWE": ((0, 106, 167), (254, 204, 0), (0, 106, 167)),
    "NED": ((174, 28, 40), (245, 245, 245), (33, 70, 139)),
    "URU": ((245, 245, 245), (0, 56, 168), (255, 205, 0)),
    "URY": ((245, 245, 245), (0, 56, 168), (255, 205, 0)),
    "SEN": ((0, 133, 63), (253, 239, 66), (227, 27, 35)),
    "MAR": ((193, 39, 45), (0, 98, 51), (193, 39, 45)),
    "RSA": ((0, 119, 73), (255, 184, 28), (224, 60, 49)),
    "ZAF": ((0, 119, 73), (255, 184, 28), (224, 60, 49)),
    "SCO": ((0, 94, 184), (245, 245, 245), (0, 94, 184)),
    "CZE": ((245, 245, 245), (213, 43, 30), (17, 69, 126)),
    "TUR": ((227, 10, 23), (245, 245, 245), (227, 10, 23)),
    "AUS": ((0, 42, 92), (245, 245, 245), (226, 55, 62)),
    "COL": ((252, 209, 22), (0, 56, 147), (206, 17, 38)),
    "ECU": ((255, 221, 0), (0, 48, 135), (237, 28, 36)),
    "CRO": ((245, 245, 245), (23, 73, 143), (213, 43, 30)),
    "POL": ((245, 245, 245), (220, 20, 60), (245, 245, 245)),
    "CAN": ((213, 43, 30), (245, 245, 245), (213, 43, 30)),
    "SUI": ((213, 43, 30), (245, 245, 245), (213, 43, 30)),
}

BASE_KITS = {
    "BRA": (245, 222, 51),
    "ARG": (130, 199, 245),
    "FRA": (42, 75, 184),
    "GER": (238, 238, 228),
    "DEU": (238, 238, 228),
    "ESP": (196, 33, 44),
    "ENG": (240, 244, 250),
    "POR": (197, 33, 47),
    "PRT": (197, 33, 47),
    "NED": (238, 104, 44),
    "ITA": (45, 86, 170),
    "BEL": (225, 38, 48),
    "MEX": (32, 126, 85),
    "USA": (245, 245, 245),
    "JPN": (45, 88, 185),
    "KOR": (225, 45, 55),
    "DEN": (205, 42, 62),
    "SWE": (254, 204, 0),
    "URU": (115, 190, 240),
    "MAR": (190, 39, 49),
    "CAN": (220, 42, 52),
}

PALETTE = [
    ((26, 78, 158), (245, 245, 245), (202, 38, 49)),
    ((20, 136, 78), (246, 206, 65), (28, 75, 154)),
    ((187, 37, 48), (245, 245, 245), (28, 77, 148)),
    ((245, 245, 245), (41, 87, 168), (226, 55, 62)),
    ((34, 34, 34), (235, 196, 55), (216, 58, 65)),
    ((32, 126, 85), (245, 245, 245), (210, 45, 55)),
]

@dataclass(frozen=True)
class TeamProfile:
    key: str
    name: str
    code: str
    group: str
    fifa_rank: int
    squad_rating: float
    flag: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    kit: tuple[int, int, int]
    elo: float
    goals_for: float
    goals_against: float
    win_rate: float
    form: float
    matches: int


@dataclass(frozen=True)
class Prediction:
    algorithm: str
    home: float
    draw: float
    away: float
    home_goals: float
    away_goals: float
    confidence: float
    reason: str
    home_advances: float = 0.5
    away_advances: float = 0.5
    top_scores: tuple[tuple[int, int, float], ...] = ()
    over_25: float = 0.0
    btts: float = 0.0
    score_home: int | None = None
    score_away: int | None = None
    outcome_class: int = 1
    outcome_probability: float = 0.0
    score_probability: float = 0.0
    blend_probs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    poisson_outcome_probs: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MatchDriver:
    key: str
    label: str
    value: float
    description: str


@dataclass(frozen=True)
class MatchDrivers:
    squad_top26_diff: float
    attack_strength_diff: float
    midfield_strength_diff: float
    defense_strength_diff: float
    gk_strength_diff: float
    tm_market_value_log_diff: float
    tm_caps_diff: float
    tm_recent_injury_days_diff: float
    context_shift: float

    def rows(self) -> tuple[MatchDriver, ...]:
        return (
            MatchDriver("squad_top26_diff", "Elenco top26", self.squad_top26_diff, "Força relativa do elenco"),
            MatchDriver("attack_strength_diff", "Ataque", self.attack_strength_diff, "Diferença de qualidade ofensiva"),
            MatchDriver("midfield_strength_diff", "Meio", self.midfield_strength_diff, "Diferença de meio-campo"),
            MatchDriver("defense_strength_diff", "Defesa", self.defense_strength_diff, "Diferença defensiva"),
            MatchDriver("gk_strength_diff", "Goleiro", self.gk_strength_diff, "Diferença de goleiro"),
            MatchDriver("tm_market_value_log_diff", "Valor TM", self.tm_market_value_log_diff, "Diferença log de valor de mercado"),
            MatchDriver("tm_caps_diff", "Caps TM", self.tm_caps_diff, "Diferença de experiência internacional"),
            MatchDriver(
                "tm_recent_injury_days_diff",
                "Lesões recentes",
                self.tm_recent_injury_days_diff,
                "Diferença de dias recentes lesionado",
            ),
            MatchDriver("context_shift", "Contexto jogo", self.context_shift, "Descanso, viagem e vantagem de sede"),
        )


@dataclass(frozen=True)
class MatchAnalysis:
    home: TeamProfile
    away: TeamProfile
    prediction: Prediction
    base_classifier_probs: tuple[float, float, float]
    final_classifier_probs: tuple[float, float, float]
    poisson_outcome_probs: tuple[float, float, float]
    blend_probs: tuple[float, float, float]
    top_scores: tuple[tuple[int, int, float], ...]
    over_25: float
    btts: float
    rho: float
    draw_floor: float
    draw_ceiling: float
    classifier_weight: float
    poisson_weight: float
    sampled_home: int
    sampled_away: int
    outcome_class: int
    outcome_probability: float
    score_probability: float
    home_xg: float
    away_xg: float
    home_advances: float
    away_advances: float
    drivers: MatchDrivers

    @property
    def under_25(self) -> float:
        return 1.0 - self.over_25

    @property
    def draw_policy_text(self) -> str:
        return f"faixa calibrada {self.draw_floor:.0%}-{self.draw_ceiling:.0%}; sem modelo dedicado"


class WorldCupModel:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo SOTA não encontrado: {MODEL_PATH}")
        with MODEL_PATH.open("rb") as file:
            self.package = pickle.load(file)
        self._load_runtime_prediction_cache()
        self.states = sota.ensure_states(self.package)
        self.squad = self.package["squad_strength"].copy()
        self.fixtures = self.package["fixtures"].copy()
        self.report = self._load_report()
        self.training_rows = int(self.report.get("training_rows", 0))
        self.examples = range(self.training_rows)
        self.targets = ()
        self._champion_odds_cache: dict[tuple[int, int], list[tuple[str, int, float]]] = {}
        self._champion_odds_cache_lock = threading.Lock()
        self._profiles_cache: list[TeamProfile] | None = None
        self._scenario_bank: list[object] = list(self.package.get("_runtime_scenario_bank", []))

    def _load_runtime_prediction_cache(self) -> None:
        if not RUNTIME_PREDICTION_CACHE_PATH.exists():
            return
        try:
            with RUNTIME_PREDICTION_CACHE_PATH.open("rb") as file:
                payload = pickle.load(file)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        if payload.get("model_sha256") != file_sha256(MODEL_PATH):
            return
        if payload.get("sota_pipeline_sha256") != file_sha256(SOTA_PIPELINE_PATH):
            return
        prediction_cache = payload.get("prediction_cache")
        if isinstance(prediction_cache, dict):
            self.package["prediction_cache"] = prediction_cache
        prediction_base_cache = payload.get("prediction_base_cache")
        if isinstance(prediction_base_cache, dict):
            self.package["prediction_base_cache"] = prediction_base_cache
        scenario_bank = payload.get("scenario_bank")
        if isinstance(scenario_bank, list):
            self.package["_runtime_scenario_bank"] = scenario_bank

    def _scenario_bank_champion_odds_with_representative(
        self,
        runs: int,
        seed: int,
        progress_callback: Callable[[int, int, list[tuple[str, int, float]]], bool | None] | None,
        progress_with_odds: bool,
    ) -> tuple[list[tuple[str, int, float]], dict[str, object] | None] | None:
        if not self._scenario_bank:
            return None
        total = max(1, int(runs))
        rng = random.Random(seed * 1009 + total * 37 + len(self._scenario_bank))
        representative_candidates: dict[str, list[object]] = {}
        counts: dict[str, int] = {}
        snapshot_interval = max(1, total // 20)
        for index in range(total):
            candidate = self._scenario_bank[rng.randrange(len(self._scenario_bank))]
            champion = str(getattr(candidate, "champion"))
            counts[champion] = counts.get(champion, 0) + 1
            representative_candidates.setdefault(champion, []).append(candidate)
            done = index + 1
            if progress_callback is not None and (done == total or done % snapshot_interval == 0):
                rows = sorted(
                    ((team, wins, wins / done) for team, wins in counts.items()),
                    key=lambda item: (-item[2], item[0]),
                )
                if progress_callback(done, total, rows if progress_with_odds else []) is False:
                    break
        denominator = max(1, sum(counts.values()))
        rows = sorted(
            ((team, wins, wins / denominator) for team, wins in counts.items()),
            key=lambda item: (-item[2], item[0]),
        )
        representative = self._plausible_representative_result(representative_candidates, rows, seed) if rows else None
        if representative is not None:
            representative["runtime_monte_carlo_source"] = "scenario_bank_bootstrap"
            representative["runtime_scenario_bank_size"] = len(self._scenario_bank)
        return rows, representative

    def _load_report(self) -> dict[str, object]:
        if not REPORT_PATH.exists():
            return {}
        import json

        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    def profiles(self) -> list[TeamProfile]:
        if self._profiles_cache is not None:
            return list(self._profiles_cache)
        result = []
        ranks = self.package.get("latest_rankings", {})
        for index, row in enumerate(self.squad.itertuples(index=False)):
            key = str(row.team_key)
            code = str(row.fifa_code)
            state = self.states.get(key, sota.RunningTeam())
            colors = BASE_FLAGS.get(code, PALETTE[index % len(PALETTE)])
            rank = int(float(ranks.get(key, {}).get("rank", 120)))
            result.append(
                TeamProfile(
                    key=key,
                    name=str(row.team_name),
                    code=code,
                    group=str(row.group_letter),
                    fifa_rank=rank,
                    squad_rating=float(row.squad_top26),
                    flag=colors,
                    kit=BASE_KITS.get(code, colors[0]),
                    elo=float(state.elo),
                    goals_for=float(state.avg_for),
                    goals_against=float(state.avg_against),
                    win_rate=float(state.win_rate),
                    form=float(state.form_score),
                    matches=int(state.matches),
                )
            )
        self._profiles_cache = result
        return list(result)

    def profile_for(self, team: TeamProfile | str) -> TeamProfile:
        if isinstance(team, TeamProfile):
            return team
        value = str(team).strip()
        canonical = sota.canonical_team(value)
        for profile in self.profiles():
            if value in {profile.key, profile.name, profile.code} or canonical in {profile.key, profile.name}:
                return profile
        raise KeyError(f"Seleção não encontrada no fixture 2026: {team}")

    def fixture_teams(self) -> list[tuple[str, str]]:
        return sorted(
            ((profile.key, profile.group) for profile in self.profiles()),
            key=lambda item: (item[1], item[0]),
        )

    def team_names(self) -> list[str]:
        return [team for team, _group in self.fixture_teams()]

    def simulation_policy(self) -> dict[str, float | str]:
        return sota.simulation_policy_from_package(self.package)

    def team_strength_rows(self, limit: int | None = None) -> list[dict[str, object]]:
        frame = self.squad.sort_values("squad_top26", ascending=False)
        if limit is not None:
            frame = frame.head(max(0, int(limit)))
        return frame.to_dict(orient="records")

    def feature_vector(self, home: TeamProfile, away: TeamProfile) -> list[float]:
        features = sota.prepare_match_features(self.package, home.key, away.key)
        return [float(features.iloc[0][name]) for name in self.package["base_features"]]

    def predict_matchup(self, home: TeamProfile, away: TeamProfile, seed: int | None = None) -> Prediction:
        return self.analyze_match(home, away, seed=seed).prediction

    def analyze_match(self, home: TeamProfile | str, away: TeamProfile | str, seed: int | None = None) -> MatchAnalysis:
        home_profile = self.profile_for(home)
        away_profile = self.profile_for(away)
        raw = sota.predict_match(self.package, home_profile.key, away_profile.key, neutral=True, knockout=True)
        classifier_probs = sota.classifier_probs_from_prediction(raw)
        base_classifier_probs = self._prob_tuple(raw, "p_xgb", fallback=classifier_probs)
        home_xg = float(raw["home_xg"])
        away_xg = float(raw["away_xg"])
        rho = sota.dixon_coles_rho_from_package(self.package)
        matrix = sota.score_matrix(home_xg, away_xg, rho=rho)
        top_scores = self._top_scores(matrix)
        over_25 = float(sum(matrix[h, a] for h in range(matrix.shape[0]) for a in range(matrix.shape[1]) if h + a >= 3))
        btts = float(sum(matrix[h, a] for h in range(1, matrix.shape[0]) for a in range(1, matrix.shape[1])))
        home_advances = float(raw["p_home_advances"])
        away_advances = float(raw["p_away_advances"])
        (
            score_home,
            score_away,
            score_prob,
            outcome_class,
            outcome_prob,
            blend_probs,
            poisson_outcome_probs,
            sim_meta,
        ) = self._score_for_hybrid(
            matrix,
            classifier_probs,
            seed,
        )
        reason = "Tendência 1X2/XGBoost calibrada; Poisson/DC mantém a variação dos placares."
        prediction = self._prediction_from_probs(
            "CONFRONTO",
            classifier_probs,
            home_xg,
            away_xg,
            reason,
            home_advances=home_advances,
            away_advances=away_advances,
            top_scores=top_scores,
            over_25=over_25,
            btts=btts,
            score_home=score_home,
            score_away=score_away,
            outcome_class=outcome_class,
            outcome_probability=outcome_prob,
            score_probability=score_prob,
            blend_probs=blend_probs,
            poisson_outcome_probs=poisson_outcome_probs,
        )
        policy = self.simulation_policy()
        return MatchAnalysis(
            home=home_profile,
            away=away_profile,
            prediction=prediction,
            base_classifier_probs=base_classifier_probs,
            final_classifier_probs=tuple(float(value) for value in classifier_probs),
            poisson_outcome_probs=tuple(float(value) for value in poisson_outcome_probs),
            blend_probs=tuple(float(value) for value in blend_probs),
            top_scores=top_scores,
            over_25=over_25,
            btts=btts,
            rho=float(rho),
            draw_floor=float(policy["draw_floor"]),
            draw_ceiling=float(policy["draw_ceiling"]),
            classifier_weight=float(sim_meta["sim_classifier_weight"]),
            poisson_weight=float(1.0 - sim_meta["sim_classifier_weight"]),
            sampled_home=int(score_home),
            sampled_away=int(score_away),
            outcome_class=int(outcome_class),
            outcome_probability=float(outcome_prob),
            score_probability=float(score_prob),
            home_xg=home_xg,
            away_xg=away_xg,
            home_advances=home_advances,
            away_advances=away_advances,
            drivers=self._driver_values(raw),
        )

    def simulate_tournament(self, seed: int = 2026) -> dict[str, object]:
        champion, bracket, standings = sota.simulate_tournament(self.package, seed=seed)
        return self._tournament_result(champion, bracket, standings, seed=seed)

    def _tournament_result(
        self,
        champion: str,
        bracket: object,
        standings: object,
        seed: int | None = None,
        representative_for: str | None = None,
    ) -> dict[str, object]:
        result = {
            "champion": champion,
            "rounds": bracket.to_dict(orient="records"),
            "standings": standings.to_dict(orient="records"),
        }
        if seed is not None:
            result["seed"] = seed
        if representative_for is not None:
            result["representative_for"] = representative_for
        return result

    def _plausible_representative_result(
        self,
        representative_candidates: dict[str, list[object]],
        champion_odds: object,
        seed: int,
    ) -> dict[str, object] | None:
        story = sota.select_plausible_representative_story(champion_odds, representative_candidates, seed)
        if story is None:
            return None
        selected_team = str(story["team"])
        story_seed = int(story["seed"])
        champion, bracket, standings = sota.simulate_tournament(self.package, seed=story_seed)
        if str(champion) != selected_team:
            selected_team = str(champion)
        result = self._tournament_result(
            selected_team,
            bracket,
            standings,
            seed=story_seed,
            representative_for=selected_team,
        )
        result["representative_rank"] = int(story["rank"])
        result["representative_probability"] = float(story["probability"])
        result["representative_pool"] = list(story["pool"])
        result["representative_policy"] = str(story["policy"])
        result["representative_runner_up"] = str(story["runner_up"])
        result["representative_runner_up_finalist_rank"] = int(story["runner_up_finalist_rank"])
        result["representative_runner_up_finalist_probability"] = float(story["runner_up_finalist_probability"])
        result["representative_plausibility_score"] = float(story["plausibility_score"])
        result["representative_candidate_count"] = int(story["candidate_count"])
        result["representative_story_pool_size"] = int(story["story_pool_size"])
        result["representative_sample_runs"] = int(story["sample_runs"])
        result["representative_surprise_level"] = str(story["surprise_level"])
        result["representative_final_goal_diff"] = int(story["final_goal_diff"])
        result["representative_final_total_goals"] = int(story["final_total_goals"])
        result["odds_leader"] = str(story["leader"])
        result["odds_leader_probability"] = float(story["leader_probability"])
        return result

    def _champion_odds_rows(self, champion_odds: object) -> list[tuple[str, int, float]]:
        return sota.champion_odds_rows(champion_odds)

    def champion_odds(
        self,
        runs: int = 1000,
        seed: int = 2026,
        workers: int | None = None,
        progress_callback: Callable[[int, int, list[tuple[str, int, float]]], bool | None] | None = None,
    ) -> list[tuple[str, int, float]]:
        cache_key = (max(1, runs), seed)
        if progress_callback is None:
            with self._champion_odds_cache_lock:
                cached = self._champion_odds_cache.get(cache_key)
            if cached is not None:
                return cached

        def progress(done: int, total: int, snapshot: object) -> bool | None:
            if progress_callback is None:
                return None
            return progress_callback(done, total, self._champion_odds_rows(snapshot))

        champion_odds, _stage_odds = sota.monte_carlo(
            self.package,
            runs=cache_key[0],
            seed=cache_key[1],
            progress_callback=progress if progress_callback else None,
            workers=workers,
            fast_champion_only=True,
        )
        result = self._champion_odds_rows(champion_odds)
        if progress_callback is None:
            with self._champion_odds_cache_lock:
                self._champion_odds_cache[cache_key] = result
        return result

    def champion_odds_with_representative(
        self,
        runs: int = 1000,
        seed: int = 2026,
        workers: int | None = None,
        progress_callback: Callable[[int, int, list[tuple[str, int, float]]], bool | None] | None = None,
        progress_with_odds: bool = True,
        use_scenario_bank: bool = False,
    ) -> tuple[list[tuple[str, int, float]], dict[str, object] | None]:
        if use_scenario_bank:
            bank_result = self._scenario_bank_champion_odds_with_representative(runs, seed, progress_callback, progress_with_odds)
            if bank_result is not None:
                return bank_result

        representative_candidates: dict[str, list[object]] = {}

        def progress(done: int, total: int, snapshot: object) -> bool | None:
            if progress_callback is None:
                return None
            odds = self._champion_odds_rows(snapshot) if progress_with_odds else []
            return progress_callback(done, total, odds)

        champion_odds, _stage_odds = sota.monte_carlo(
            self.package,
            runs=max(1, runs),
            seed=seed,
            progress_callback=progress if progress_callback else None,
            workers=workers,
            representative_candidates=representative_candidates,
            fast_champion_only=True,
        )
        result = self._champion_odds_rows(champion_odds)
        representative = self._plausible_representative_result(representative_candidates, champion_odds, seed) if result else None
        return result, representative

    def _prob_tuple(
        self,
        raw: dict[str, object],
        prefix: str,
        fallback: np.ndarray | tuple[float, float, float],
    ) -> tuple[float, float, float]:
        keys = (
            f"{prefix}_home_win_90",
            f"{prefix}_draw_90",
            f"{prefix}_away_win_90",
        )
        if all(key in raw for key in keys):
            values = tuple(float(raw[key]) for key in keys)
        else:
            values = tuple(float(value) for value in fallback)
        total = max(0.001, sum(values))
        return tuple(float(value / total) for value in values)  # type: ignore[return-value]

    def _driver_values(self, raw: dict[str, object]) -> MatchDrivers:
        missing = [
            key
            for key in MatchDrivers.__dataclass_fields__
            if key not in raw
        ]
        if missing:
            raise KeyError(f"Predição sem drivers obrigatórios: {', '.join(missing)}")
        return MatchDrivers(
            squad_top26_diff=float(raw["squad_top26_diff"]),
            attack_strength_diff=float(raw["attack_strength_diff"]),
            midfield_strength_diff=float(raw["midfield_strength_diff"]),
            defense_strength_diff=float(raw["defense_strength_diff"]),
            gk_strength_diff=float(raw["gk_strength_diff"]),
            tm_market_value_log_diff=float(raw["tm_market_value_log_diff"]),
            tm_caps_diff=float(raw["tm_caps_diff"]),
            tm_recent_injury_days_diff=float(raw["tm_recent_injury_days_diff"]),
            context_shift=float(raw["context_shift"]),
        )

    def _prediction_from_probs(
        self,
        algorithm: str,
        probs: np.ndarray,
        home_xg: float,
        away_xg: float,
        reason: str,
        home_advances: float | None = None,
        away_advances: float | None = None,
        top_scores: tuple[tuple[int, int, float], ...] = (),
        over_25: float = 0.0,
        btts: float = 0.0,
        score_home: int | None = None,
        score_away: int | None = None,
        outcome_class: int = 1,
        outcome_probability: float = 0.0,
        score_probability: float = 0.0,
        blend_probs: tuple[float, float, float] | None = None,
        poisson_outcome_probs: tuple[float, float, float] | None = None,
    ) -> Prediction:
        probs = np.clip(np.asarray(probs, dtype=float), 0.001, 0.998)
        probs = probs / probs.sum()
        if blend_probs is None:
            blend_probs = (float(probs[0]), float(probs[1]), float(probs[2]))
        if poisson_outcome_probs is None:
            poisson_outcome_probs = (0.0, 0.0, 0.0)
        if home_advances is None or away_advances is None:
            non_draw = max(0.001, probs[0] + probs[2])
            home_advances = float(probs[0] / non_draw)
            away_advances = 1.0 - home_advances
        if outcome_probability <= 0:
            outcome_probability = float(blend_probs[outcome_class])
        return Prediction(
            algorithm=algorithm,
            home=float(probs[0]),
            draw=float(probs[1]),
            away=float(probs[2]),
            home_goals=float(max(0.05, home_xg)),
            away_goals=float(max(0.05, away_xg)),
            confidence=float(probs.max()),
            reason=reason,
            home_advances=float(home_advances),
            away_advances=float(away_advances),
            top_scores=top_scores,
            over_25=float(over_25),
            btts=float(btts),
            score_home=score_home,
            score_away=score_away,
            outcome_class=int(outcome_class),
            outcome_probability=float(outcome_probability),
            score_probability=float(score_probability),
            blend_probs=blend_probs,
            poisson_outcome_probs=poisson_outcome_probs,
        )

    def _top_scores(self, matrix: np.ndarray, limit: int = 5) -> tuple[tuple[int, int, float], ...]:
        scores = [
            (h, a, float(matrix[h, a]))
            for h in range(matrix.shape[0])
            for a in range(matrix.shape[1])
        ]
        scores.sort(key=lambda item: item[2], reverse=True)
        return tuple(scores[:limit])

    def _score_for_hybrid(
        self,
        matrix: np.ndarray,
        probs: np.ndarray,
        seed: int | None = None,
    ) -> tuple[int, int, float, int, float, tuple[float, float, float], tuple[float, float, float], dict[str, float]]:
        policy = sota.simulation_policy_from_package(self.package)
        rng = random.Random(seed) if seed is not None else None
        home_goals, away_goals, meta = sota.hybrid_score_choice(
            matrix,
            np.asarray(probs, dtype=float),
            rng=rng,
            classifier_weight=float(policy["classifier_weight"]),
        )
        blend_tuple = (float(meta["sim_blend_home"]), float(meta["sim_blend_draw"]), float(meta["sim_blend_away"]))
        poisson_tuple = (float(meta["sim_poisson_home"]), float(meta["sim_poisson_draw"]), float(meta["sim_poisson_away"]))
        outcome_class = int(meta["sim_outcome"])
        return (
            home_goals,
            away_goals,
            float(meta["sim_score_probability"]),
            outcome_class,
            float(meta["sim_outcome_probability"]),
            blend_tuple,
            poisson_tuple,
            {key: float(value) for key, value in meta.items()},
        )

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arena_ai import bolao  # noqa: E402
from arena_ai.worldcup_model import MODEL_PATH, WorldCupModel, sota  # noqa: E402
from bolao_knockout_calibration import actual_outcome, file_sha256, historical_form, metric_summary  # noqa: E402
from bolao_round16_calibration import context_after_results, results_for_round  # noqa: E402


REPORT_DIR = ROOT / "modeling" / "worldcup_2026_ml" / "reports"
DEFAULT_JSON = REPORT_DIR / "bolao_final_calibration.json"
DEFAULT_SEMIFINAL_CSV = REPORT_DIR / "bolao_semifinal_calibration.csv"
DEFAULT_MEDAL_CSV = REPORT_DIR / "bolao_medal_matches_calibration.csv"
TRAINING_MATRIX = ROOT / "modeling" / "worldcup_2026_ml" / "data" / "processed" / "sota_training_matches.csv"
OPERATIONAL_LOSS_MARGIN = bolao.CURRENT_FORM_REQUIRED_LOG_SCORE_GAIN_PER_MATCH
MONTE_CARLO_RUNS = 2000
MONTE_CARLO_SEED = 20260628
LOWER_IS_BETTER = (
    "log_loss_1x2",
    "brier_1x2",
    "exact_score_log_loss",
    "advance_log_loss",
    "advance_brier",
)


def audit_window_rows(
    model: WorldCupModel,
    board: bolao.GroupStageBoard,
    group_form: bolao.TournamentForm,
    candidate_form: bolao.TournamentForm,
    observed_before_window: dict[int, bolao.ObservedKnockoutResult],
    match_numbers: tuple[int, ...],
    *,
    window_name: str,
) -> pd.DataFrame:
    forms = {
        "historical": historical_form(group_form),
        "group_form": group_form,
        "frozen_form": candidate_form,
    }
    contexts = context_after_results(model, board, observed_before_window)
    fixtures = model.fixtures[model.fixtures["match_number"].isin(match_numbers)].sort_values(
        ["kickoff_at", "match_number"]
    )
    if set(int(value) for value in fixtures["match_number"]) != set(match_numbers):
        raise ValueError(f"janela {window_name} não corresponde ao fixture")

    classifier_weight = float(board.policy["classifier_weight"])
    rows: list[dict[str, object]] = []
    for game in fixtures.itertuples(index=False):
        observed = board.knockout_results[int(game.match_number)]
        context = sota.fixture_context(game, contexts, observed.home, observed.away)
        outcome = actual_outcome(observed.home_goals_90, observed.away_goals_90)
        row: dict[str, object] = {
            "window": window_name,
            "match_number": int(game.match_number),
            "round": observed.round_name,
            "kickoff_at": pd.Timestamp(game.kickoff_at).isoformat(),
            "home": observed.home,
            "away": observed.away,
            "home_goals_90": int(observed.home_goals_90),
            "away_goals_90": int(observed.away_goals_90),
            "extra_time_home_goals": int(observed.extra_time_home_goals),
            "extra_time_away_goals": int(observed.extra_time_away_goals),
            "winner": observed.winner,
            "resolution": observed.resolution,
            "shootout_home": observed.shootout_home,
            "shootout_away": observed.shootout_away,
            "actual_1x2": ("home", "draw", "away")[outcome],
            "actual_home_advanced": int(observed.winner == observed.home),
            "total_goals_90": int(observed.home_goals_90 + observed.away_goals_90),
        }
        for label, form in forms.items():
            distribution = bolao.form_aware_match(
                model,
                form,
                observed.home,
                observed.away,
                knockout=True,
                context=context,
            )
            score_probability = bolao.hybrid_score_probability(
                distribution.prediction,
                distribution.matrix,
                observed.home_goals_90,
                observed.away_goals_90,
                classifier_weight=classifier_weight,
            )
            row.update(
                {
                    f"{label}_p_home_90": float(distribution.blend[0]),
                    f"{label}_p_draw_90": float(distribution.blend[1]),
                    f"{label}_p_away_90": float(distribution.blend[2]),
                    f"{label}_score_probability": float(score_probability),
                    f"{label}_p_home_advances": float(distribution.prediction["p_home_advances"]),
                }
            )
            if label == "frozen_form":
                modal_outcome = int(np.argmax(distribution.blend))
                modal_score = bolao.candidate_for_outcome(
                    distribution.matrix,
                    distribution.blend,
                    modal_outcome,
                    "90min",
                    advance_multiplier=1.0,
                )
                row.update(
                    {
                        "frozen_form_home_xg": float(distribution.prediction["home_xg"]),
                        "frozen_form_away_xg": float(distribution.prediction["away_xg"]),
                        "frozen_form_weight": float(distribution.form_weight),
                        "frozen_form_modal_score_90": f"{modal_score.home_goals}-{modal_score.away_goals}",
                        "frozen_form_most_likely_to_advance": (
                            observed.home
                            if float(distribution.prediction["p_home_advances"]) >= 0.5
                            else observed.away
                        ),
                    }
                )
        rows.append(row)
        sota.update_team_context(contexts, observed.home, observed.away, game)
    return pd.DataFrame(rows).sort_values(["kickoff_at", "match_number"]).reset_index(drop=True)


def metric_block(rows: pd.DataFrame) -> dict[str, object]:
    historical_metrics = metric_summary(rows, "historical")
    group_form_metrics = metric_summary(rows, "group_form")
    frozen_form_metrics = metric_summary(rows, "frozen_form")
    loss_gaps = {
        metric: float(frozen_form_metrics[metric]) - float(historical_metrics[metric])
        for metric in LOWER_IS_BETTER
    }
    metrics_favoring_frozen_form = [metric for metric, gap in loss_gaps.items() if gap <= 0.0]
    metrics_within_operational_margin = [
        metric for metric, gap in loss_gaps.items() if gap <= OPERATIONAL_LOSS_MARGIN
    ]
    return {
        "historical_baseline": historical_metrics,
        "group_only_form": group_form_metrics,
        "frozen_form": frozen_form_metrics,
        "loss_gap_frozen_form_minus_historical": loss_gaps,
        "metrics_favoring_frozen_form": metrics_favoring_frozen_form,
        "metrics_within_operational_margin": metrics_within_operational_margin,
        "all_loss_gaps_within_operational_margin": (
            len(metrics_within_operational_margin) == len(LOWER_IS_BETTER)
        ),
    }


def champion_rows(ranking: list[bolao.ChampionOption]) -> list[dict[str, object]]:
    return [
        {
            "rank": int(option.rank),
            "team": option.team,
            "titles": int(option.wins),
            "probability": float(option.probability),
        }
        for option in ranking
    ]


def prediction_rows(rows: pd.DataFrame) -> list[dict[str, object]]:
    columns = (
        "window",
        "match_number",
        "round",
        "home",
        "away",
        "frozen_form_home_xg",
        "frozen_form_away_xg",
        "frozen_form_p_home_90",
        "frozen_form_p_draw_90",
        "frozen_form_p_away_90",
        "frozen_form_p_home_advances",
        "frozen_form_most_likely_to_advance",
        "frozen_form_modal_score_90",
        "home_goals_90",
        "away_goals_90",
        "winner",
        "resolution",
    )
    return rows.loc[:, columns].to_dict(orient="records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita a calibracao final do Bolao sem look-ahead.")
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON, help="JSON de saída.")
    parser.add_argument(
        "--semifinal-csv",
        type=Path,
        default=DEFAULT_SEMIFINAL_CSV,
        help="CSV da janela prospectiva das semifinais.",
    )
    parser.add_argument(
        "--medal-csv",
        type=Path,
        default=DEFAULT_MEDAL_CSV,
        help="CSV das janelas prospectivas de terceiro lugar e final.",
    )
    return parser.parse_args()


def run(*, out: Path, semifinal_csv: Path, medal_csv: Path) -> dict[str, object]:
    model = WorldCupModel()
    current_board = bolao.build_group_stage_board(model)
    group_form = bolao.build_tournament_form(model, current_board.form.observed_results)

    round32_results = results_for_round(current_board, "Round of 32")
    round16_results = results_for_round(current_board, "Round of 16")
    quarterfinal_results = results_for_round(current_board, "Quarterfinals")
    semifinal_results = results_for_round(current_board, "Semifinals")
    third_place_results = results_for_round(current_board, "Third Place Playoff")
    final_results = results_for_round(current_board, "Final")

    observed_before_semifinals = {**round32_results, **round16_results, **quarterfinal_results}
    observed_through_semifinals = {**observed_before_semifinals, **semifinal_results}
    observed_through_third_place = {**observed_through_semifinals, **third_place_results}
    observed_full_tournament = {**observed_through_third_place, **final_results}

    pre_semifinal_form = bolao.update_tournament_form_with_knockout_results(
        model,
        group_form,
        current_board.team_context,
        observed_before_semifinals,
    )
    pre_third_place_form = bolao.update_tournament_form_with_knockout_results(
        model,
        group_form,
        current_board.team_context,
        observed_through_semifinals,
    )
    pre_final_form = bolao.update_tournament_form_with_knockout_results(
        model,
        group_form,
        current_board.team_context,
        observed_through_third_place,
    )
    final_form = bolao.update_tournament_form_with_knockout_results(
        model,
        group_form,
        current_board.team_context,
        observed_full_tournament,
    )

    board_pre_semifinals = replace(
        current_board,
        form=pre_semifinal_form,
        knockout_results=observed_before_semifinals,
    )
    board_pre_final = replace(
        current_board,
        form=pre_final_form,
        knockout_results=observed_through_third_place,
    )
    board_final = replace(
        current_board,
        form=final_form,
        knockout_results=observed_full_tournament,
    )

    semifinal_rows = audit_window_rows(
        model,
        current_board,
        group_form,
        pre_semifinal_form,
        observed_before_semifinals,
        (101, 102),
        window_name="pre_semifinal",
    )
    third_place_rows = audit_window_rows(
        model,
        current_board,
        group_form,
        pre_third_place_form,
        observed_through_semifinals,
        (103,),
        window_name="pre_third_place",
    )
    final_rows = audit_window_rows(
        model,
        current_board,
        group_form,
        pre_final_form,
        observed_through_third_place,
        (104,),
        window_name="pre_final",
    )
    medal_rows = pd.concat((third_place_rows, final_rows), ignore_index=True)
    all_rows = pd.concat((semifinal_rows, medal_rows), ignore_index=True)

    pre_semifinal_ranking = bolao.build_champion_ranking(
        model,
        board_pre_semifinals,
        runs=MONTE_CARLO_RUNS,
        seed=MONTE_CARLO_SEED,
        top_n=4,
        workers=1,
    )
    pre_final_ranking = bolao.build_champion_ranking(
        model,
        board_pre_final,
        runs=MONTE_CARLO_RUNS,
        seed=MONTE_CARLO_SEED,
        top_n=2,
        workers=1,
    )

    semifinal_metrics = metric_block(semifinal_rows)
    third_place_metrics = metric_block(third_place_rows)
    final_metrics = metric_block(final_rows)
    combined_metrics = metric_block(all_rows)
    final_result = final_results.get(104)
    final_prediction = final_rows.iloc[0]
    extreme_score_rows = all_rows[all_rows["total_goals_90"] >= 8]
    training_columns = set(pd.read_csv(TRAINING_MATRIX, nrows=0).columns)
    stage_history_available = "stage" in training_columns or "stage_id" in training_columns
    pre_semifinal_spain = next(
        (option for option in pre_semifinal_ranking if option.team == "Spain"),
        None,
    )
    pre_final_spain = next(
        (option for option in pre_final_ranking if option.team == "Spain"),
        None,
    )

    expected_bracket = {
        101: ("France", "Spain"),
        102: ("England", "Argentina"),
        103: ("France", "England"),
        104: ("Spain", "Argentina"),
    }
    actual_bracket = {
        match_number: (result.home, result.away)
        for match_number, result in sorted(observed_full_tournament.items())
        if match_number >= 101
    }
    extra_time_goals_observed = int(
        all_rows["extra_time_home_goals"].sum() + all_rows["extra_time_away_goals"].sum()
    )
    hard_gates = {
        "complete_final_stage_snapshot": set(actual_bracket) == set(range(101, 105)),
        "official_final_stage_bracket": actual_bracket == expected_bracket,
        "candidate_frozen_before_semifinals": pre_semifinal_form.knockout_form_matches == 28,
        "candidate_frozen_before_third_place": pre_third_place_form.knockout_form_matches == 30,
        "candidate_frozen_before_final": pre_final_form.knockout_form_matches == 31,
        "semifinalists_identified_before_kickoff": (
            float(semifinal_metrics["frozen_form"]["advance_accuracy"]) == 1.0
        ),
        "champion_identified_before_final": (
            str(final_prediction["frozen_form_most_likely_to_advance"]) == "Spain"
        ),
        "spain_led_pre_semifinal_monte_carlo": (
            bool(pre_semifinal_ranking) and pre_semifinal_ranking[0].team == "Spain"
        ),
        "spain_led_pre_final_monte_carlo": (
            bool(pre_final_ranking) and pre_final_ranking[0].team == "Spain"
        ),
        "extra_time_goal_excluded_from_regulation_form": (
            extra_time_goals_observed == 1
            and final_form.knockout_form_policy == bolao.KNOCKOUT_FORM_POLICY
            and final_form.knockout_form_matches == 32
        ),
        "final_score_separated_correctly": bool(
            final_result is not None
            and final_result.home == "Spain"
            and final_result.away == "Argentina"
            and final_result.home_goals_90 == 0
            and final_result.away_goals_90 == 0
            and final_result.extra_time_home_goals == 1
            and final_result.extra_time_away_goals == 0
            and final_result.winner == "Spain"
            and final_result.resolution == "extra_time"
        ),
        "tournament_complete_with_spain_champion": (
            len(board_final.knockout_results) == 32
            and final_result is not None
            and final_result.winner == "Spain"
        ),
        "combined_loss_gaps_within_operational_margin": bool(
            combined_metrics["all_loss_gaps_within_operational_margin"]
        ),
    }

    report: dict[str, object] = {
        "report_type": "arena_bolao_final_calibration",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "scope": {
            "observed_windows": ["Semifinals", "Third Place Playoff", "Final"],
            "observed_matches": int(len(all_rows)),
            "evaluation_target": "90-minute 1X2, exact regulation score, and eventual advancement",
            "look_ahead": (
                "none; snapshots were frozen after the quarterfinals, after the semifinals, and after the "
                "third-place match respectively"
            ),
        },
        "observed_resolution": {
            "regulation": int((all_rows["resolution"] == "90min").sum()),
            "extra_time": int((all_rows["resolution"] == "extra_time").sum()),
            "penalties": int((all_rows["resolution"] == "penalties").sum()),
        },
        "window_metrics": {
            "semifinals": semifinal_metrics,
            "third_place": third_place_metrics,
            "final": final_metrics,
            "combined_prequential": combined_metrics,
        },
        "pre_match_predictions": {
            "semifinals": prediction_rows(semifinal_rows),
            "medal_matches": prediction_rows(medal_rows),
        },
        "monte_carlo_snapshots": {
            "runs": MONTE_CARLO_RUNS,
            "seed": MONTE_CARLO_SEED,
            "pre_semifinal": champion_rows(pre_semifinal_ranking),
            "pre_final": champion_rows(pre_final_ranking),
            "actual_champion": "Spain",
            "spain_probability_pre_semifinal": (
                float(pre_semifinal_spain.probability) if pre_semifinal_spain is not None else 0.0
            ),
            "spain_probability_pre_final": (
                float(pre_final_spain.probability) if pre_final_spain is not None else 0.0
            ),
            "post_tournament_interpretation": (
                "Once match 104 is locked, Spain at 100% is an observed result, not a model probability."
            ),
        },
        "current_form_update": {
            "policy": final_form.knockout_form_policy,
            "prior_goal_equivalents_frozen": float(final_form.prior_goal_equivalents),
            "median_weight_before_semifinals": float(pre_semifinal_form.median_current_weight),
            "median_weight_before_third_place": float(pre_third_place_form.median_current_weight),
            "median_weight_before_final": float(pre_final_form.median_current_weight),
            "median_weight_after_tournament": float(final_form.median_current_weight),
            "matches_appended_in_final_stage": int(len(all_rows)),
            "total_knockout_matches_appended": int(final_form.knockout_form_matches),
            "extra_time_goals_observed": extra_time_goals_observed,
            "extra_time_goals_appended": 0,
            "shootout_kicks_appended": 0,
        },
        "deviation_analysis": {
            "extreme_regulation_score_threshold": "at least 8 total goals",
            "extreme_matches": [
                {
                    "match_number": int(row.match_number),
                    "match": f"{row.home} {int(row.home_goals_90)}-{int(row.away_goals_90)} {row.away}",
                    "frozen_form_score_probability": float(row.frozen_form_score_probability),
                }
                for row in extreme_score_rows.itertuples(index=False)
            ],
            "third_place_stage_history_available_in_training_matrix": stage_history_available,
            "third_place_data_gap": (
                "The processed training matrix has no stage label, so the project cannot estimate whether "
                "third-place matches have a distinct scoring regime."
            ),
            "final_regulation_draw": bool(
                final_result is not None and final_result.home_goals_90 == final_result.away_goals_90
            ),
            "final_extra_time_goal_used_only_for_bracket_resolution": bool(
                final_result is not None
                and final_result.resolution == "extra_time"
                and extra_time_goals_observed > 0
                and final_form.knockout_form_matches == len(observed_full_tournament)
            ),
        },
        "decision": {
            "retain_frozen_current_form_policy_for_archival_snapshot": True,
            "append_final_stage_regulation_evidence": True,
            "retune_xgboost_or_global_hybrid_weights": False,
            "add_third_place_goal_multiplier": False,
            "recalibrate_draw_probability": False,
            "recalibrate_extra_time_probability": False,
            "recalibrate_penalty_probability": False,
            "reason": (
                "The frozen snapshots identified both finalists and Spain as champion, while the 6-4 third-place "
                "score is an extreme tail event that the current dataset cannot isolate by stage. Four dependent "
                "matches cannot justify a retrospective global retune. Regulation evidence is archived under the "
                "frozen policy, and the 2026 tournament must remain an out-of-sample block in any future retraining."
            ),
        },
        "limitations": [
            "Four final-stage matches are descriptive and cannot identify stable global recalibration.",
            "The 6-4 third-place result is one extreme observation, not evidence for a fitted multiplier.",
            "The processed historical matrix has no stage label for estimating a third-place scoring regime.",
            "The audit has final scores but no event-level xG, lineups, injuries, red-card state, or closing odds.",
            "The final extra-time goal resolves the champion but is excluded from the 90-minute posterior.",
            "Monte Carlo intervals measure sampling error only; after the final, simulation uncertainty is moot.",
        ],
        "artifacts": {
            "semifinal_csv": str(semifinal_csv),
            "medal_csv": str(medal_csv),
        },
        "source_fingerprints": {
            "model_package": file_sha256(MODEL_PATH),
            "sota_pipeline": file_sha256(ROOT / "modeling/worldcup_2026_ml/src/sota_pipeline.py"),
            "bolao": file_sha256(ROOT / "src/arena_ai/bolao.py"),
            "final_calibration_audit": file_sha256(Path(__file__).resolve()),
            "quarterfinal_calibration_audit": file_sha256(
                ROOT / "scripts/bolao_quarterfinal_calibration.py"
            ),
            "observed_group_results": file_sha256(bolao.OBSERVED_GROUP_RESULTS_PATH),
            "observed_knockout_results": file_sha256(bolao.OBSERVED_KNOCKOUT_RESULTS_PATH),
            "training_matrix": file_sha256(TRAINING_MATRIX),
        },
        "hard_gates": hard_gates,
        "approved": bool(all(hard_gates.values())),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    semifinal_csv.parent.mkdir(parents=True, exist_ok=True)
    medal_csv.parent.mkdir(parents=True, exist_ok=True)
    semifinal_rows.to_csv(semifinal_csv, index=False)
    medal_rows.to_csv(medal_csv, index=False)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = run(out=args.out, semifinal_csv=args.semifinal_csv, medal_csv=args.medal_csv)
    verdict = "OK" if report["approved"] else "FALHOU"
    combined = report["window_metrics"]["combined_prequential"]
    frozen = combined["frozen_form"]
    print(
        f"[bolao-final-calibration] {verdict} matches={report['scope']['observed_matches']} "
        f"advance_accuracy={frozen['advance_accuracy']:.1%} "
        f"spain_pre_sf={report['monte_carlo_snapshots']['spain_probability_pre_semifinal']:.1%} "
        f"spain_pre_final={report['monte_carlo_snapshots']['spain_probability_pre_final']:.1%}"
    )
    if not report["approved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

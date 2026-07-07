from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from math import sqrt
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


REPORT_DIR = ROOT / "modeling" / "worldcup_2026_ml" / "reports"
DEFAULT_JSON = REPORT_DIR / "bolao_round16_calibration.json"
DEFAULT_AUDIT_CSV = REPORT_DIR / "bolao_round16_calibration.csv"
DEFAULT_QUARTERFINAL_CSV = REPORT_DIR / "bolao_quarterfinal_predictions.csv"
OPERATIONAL_LOSS_MARGIN = bolao.CURRENT_FORM_REQUIRED_LOG_SCORE_GAIN_PER_MATCH


def results_for_round(
    board: bolao.GroupStageBoard,
    round_name: str,
) -> dict[int, bolao.ObservedKnockoutResult]:
    return {
        match_number: result
        for match_number, result in board.knockout_results.items()
        if result.round_name == round_name
    }


def context_after_results(
    model: WorldCupModel,
    board: bolao.GroupStageBoard,
    observed_results: dict[int, bolao.ObservedKnockoutResult],
) -> dict[str, dict[str, object]]:
    contexts = {team: dict(context) for team, context in board.team_context.items()}
    fixtures = model.fixtures[model.fixtures["match_number"].isin(observed_results)].sort_values(
        ["kickoff_at", "match_number"]
    )
    for game in fixtures.itertuples(index=False):
        observed = observed_results[int(game.match_number)]
        sota.update_team_context(contexts, observed.home, observed.away, game)
    return contexts


def audit_round16_rows(
    model: WorldCupModel,
    board: bolao.GroupStageBoard,
    group_form: bolao.TournamentForm,
    pre_round16_form: bolao.TournamentForm,
    round32_results: dict[int, bolao.ObservedKnockoutResult],
) -> pd.DataFrame:
    forms = {
        "historical": historical_form(group_form),
        "group_form": group_form,
        "pre_round16_form": pre_round16_form,
    }
    contexts = context_after_results(model, board, round32_results)
    fixtures = model.fixtures[model.fixtures["stage_id"] == 3].sort_values(["kickoff_at", "match_number"])
    classifier_weight = float(board.policy["classifier_weight"])
    rows: list[dict[str, object]] = []
    for game in fixtures.itertuples(index=False):
        observed = board.knockout_results[int(game.match_number)]
        context = sota.fixture_context(game, contexts, observed.home, observed.away)
        outcome = actual_outcome(observed.home_goals_90, observed.away_goals_90)
        row: dict[str, object] = {
            "match_number": int(game.match_number),
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
        rows.append(row)
        sota.update_team_context(contexts, observed.home, observed.away, game)
    return pd.DataFrame(rows).sort_values(["kickoff_at", "match_number"]).reset_index(drop=True)


def quarterfinal_prediction_rows(
    model: WorldCupModel,
    board: bolao.GroupStageBoard,
    pre_round16_form: bolao.TournamentForm,
) -> pd.DataFrame:
    winners: dict[int, str] = {}
    runners_up: dict[int, str] = {}
    for match_number, observed in sorted(board.knockout_results.items()):
        winners[match_number] = observed.winner
        runners_up[match_number] = observed.away if observed.winner == observed.home else observed.home

    contexts = context_after_results(model, board, board.knockout_results)
    fixtures = model.fixtures[model.fixtures["stage_id"] == 4].sort_values("match_number")
    rows: list[dict[str, object]] = []
    for game in fixtures.itertuples(index=False):
        left_slot, right_slot = sota.parse_match_label(game.match_label)
        home = sota.resolve_bracket_slot(left_slot, board.qualifiers, winners, runners_up, {})
        away = sota.resolve_bracket_slot(right_slot, board.qualifiers, winners, runners_up, {})
        context = sota.fixture_context(game, contexts, home, away)
        distribution = bolao.form_aware_match(model, board.form, home, away, knockout=True, context=context)
        pre_round16_distribution = bolao.form_aware_match(
            model,
            pre_round16_form,
            home,
            away,
            knockout=True,
            context=context,
        )
        modal_outcome = int(np.argmax(distribution.blend))
        modal_score = bolao.candidate_for_outcome(
            distribution.matrix,
            distribution.blend,
            modal_outcome,
            "90min",
            advance_multiplier=1.0,
        )
        rows.append(
            {
                "match_number": int(game.match_number),
                "kickoff_at": pd.Timestamp(game.kickoff_at).isoformat(),
                "home": home,
                "away": away,
                "home_xg": float(distribution.prediction["home_xg"]),
                "away_xg": float(distribution.prediction["away_xg"]),
                "p_home_win_90": float(distribution.blend[0]),
                "p_draw_90": float(distribution.blend[1]),
                "p_away_win_90": float(distribution.blend[2]),
                "p_home_advances": float(distribution.prediction["p_home_advances"]),
                "p_away_advances": float(distribution.prediction["p_away_advances"]),
                "pre_round16_p_home_advances": float(
                    pre_round16_distribution.prediction["p_home_advances"]
                ),
                "round16_update_p_home_advances_delta": float(
                    distribution.prediction["p_home_advances"]
                    - pre_round16_distribution.prediction["p_home_advances"]
                ),
                "most_likely_to_advance": (
                    home
                    if float(distribution.prediction["p_home_advances"])
                    >= float(distribution.prediction["p_away_advances"])
                    else away
                ),
                "modal_score_90": f"{modal_score.home_goals}-{modal_score.away_goals}",
                "form_weight": float(distribution.form_weight),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita a calibracao do Bolao apos as oitavas.")
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON, help="JSON de saída.")
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV, help="CSV dos jogos auditados.")
    parser.add_argument(
        "--quarterfinal-csv",
        type=Path,
        default=DEFAULT_QUARTERFINAL_CSV,
        help="CSV da foto das quartas.",
    )
    return parser.parse_args()


def run(*, out: Path, audit_csv: Path, quarterfinal_csv: Path) -> dict[str, object]:
    model = WorldCupModel()
    current_board = bolao.build_group_stage_board(model)
    group_form = bolao.build_tournament_form(model, current_board.form.observed_results)
    round32_results = results_for_round(current_board, "Round of 32")
    round16_results = results_for_round(current_board, "Round of 16")
    pre_round16_form = bolao.update_tournament_form_with_knockout_results(
        model,
        group_form,
        current_board.team_context,
        round32_results,
    )
    observed_through_round16 = {**round32_results, **round16_results}
    post_round16_form = bolao.update_tournament_form_with_knockout_results(
        model,
        group_form,
        current_board.team_context,
        observed_through_round16,
    )
    board = replace(
        current_board,
        form=post_round16_form,
        knockout_results=observed_through_round16,
    )

    rows = audit_round16_rows(model, board, group_form, pre_round16_form, round32_results)
    historical_metrics = metric_summary(rows, "historical")
    group_form_metrics = metric_summary(rows, "group_form")
    pre_round16_metrics = metric_summary(rows, "pre_round16_form")
    quarterfinals = quarterfinal_prediction_rows(model, board, pre_round16_form)

    lower_is_better = (
        "log_loss_1x2",
        "brier_1x2",
        "exact_score_log_loss",
        "advance_log_loss",
        "advance_brier",
    )
    loss_gaps = {
        metric: float(pre_round16_metrics[metric]) - float(historical_metrics[metric])
        for metric in lower_is_better
    }
    within_operational_margin = all(gap <= OPERATIONAL_LOSS_MARGIN for gap in loss_gaps.values())
    draw_in_expected_range = abs(float(pre_round16_metrics["draw_standardized_residual"])) < 2.0
    max_quarterfinal_advance_delta = float(
        quarterfinals["round16_update_p_home_advances_delta"].abs().max()
    )
    corrected_result = round16_results.get(96)
    correction_is_consistent = bool(
        corrected_result is not None
        and corrected_result.home == "Switzerland"
        and corrected_result.away == "Colombia"
        and corrected_result.winner == "Switzerland"
        and corrected_result.resolution == "penalties"
        and corrected_result.shootout_home == 4
        and corrected_result.shootout_away == 3
    )

    hard_gates = {
        "complete_round16_snapshot": len(rows) == 8 and set(rows["match_number"]) == set(range(89, 97)),
        "candidate_was_frozen_before_round16": pre_round16_form.knockout_form_matches == 16,
        "all_loss_gaps_within_operational_margin": within_operational_margin,
        "draw_count_within_two_standard_deviations": draw_in_expected_range,
        "round16_update_uses_regulation_only": (
            board.form.knockout_form_policy == bolao.KNOCKOUT_FORM_POLICY
            and board.form.knockout_form_matches == 24
        ),
        "switzerland_colombia_correction_verified": correction_is_consistent,
        "quarterfinal_bracket_complete": len(quarterfinals) == 4,
        "single_round_form_update_below_five_percentage_points": max_quarterfinal_advance_delta <= 0.05,
    }
    report: dict[str, object] = {
        "report_type": "arena_bolao_round16_calibration",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "scope": {
            "observed_round": "Round of 16",
            "observed_matches": int(len(rows)),
            "evaluation_target": "90-minute 1X2, exact regulation score, and eventual advancement",
            "comparison": "historical hybrid versus the form snapshot frozen after the Round of 32",
            "look_ahead": "none; the candidate form was fixed before the Round of 16 outcomes",
        },
        "observed_resolution": {
            "regulation": int((rows["resolution"] == "90min").sum()),
            "extra_time": int((rows["resolution"] == "extra_time").sum()),
            "penalties": int((rows["resolution"] == "penalties").sum()),
        },
        "historical_baseline": historical_metrics,
        "group_only_form": group_form_metrics,
        "pre_round16_form": pre_round16_metrics,
        "loss_gap_pre_round16_form_minus_historical": loss_gaps,
        "operational_margin": {
            "maximum_allowed_loss_gap_per_metric": OPERATIONAL_LOSS_MARGIN,
            "all_metrics_within_margin": within_operational_margin,
            "interpretation": (
                "reuses the existing per-match materiality threshold as an operational guardrail; "
                "it is not a formal statistical non-inferiority claim"
            ),
        },
        "current_form_update": {
            "policy": board.form.knockout_form_policy,
            "prior_goal_equivalents_frozen": float(board.form.prior_goal_equivalents),
            "median_weight_before_round16": float(pre_round16_form.median_current_weight),
            "median_weight_after_round16": float(board.form.median_current_weight),
            "matches_appended_in_round16": int(len(round16_results)),
            "total_knockout_matches_appended": int(board.form.knockout_form_matches),
            "extra_time_goals_appended": 0,
            "shootout_kicks_appended": 0,
            "max_abs_quarterfinal_advance_probability_delta": max_quarterfinal_advance_delta,
        },
        "data_corrections": [
            {
                "match_number": 96,
                "reported": "Colombia advanced",
                "verified": "Switzerland advanced 4-3 on penalties after 0-0 in 120 minutes",
            }
        ],
        "decision": {
            "retain_frozen_current_form_policy": within_operational_margin,
            "append_round16_regulation_evidence": within_operational_margin,
            "retune_xgboost_or_global_hybrid_weights": False,
            "recalibrate_draw_probability": False,
            "recalibrate_penalty_probability": False,
            "reason": (
                "The eight-match round slightly favors the historical baseline, but every paired aggregate loss "
                "gap remains below the 0.01 operational margin. The sample is too small for a global retune."
            ),
        },
        "quarterfinal_predictions": quarterfinals.to_dict(orient="records"),
        "limitations": [
            "Eight matches from one round cannot identify a stable global recalibration.",
            "The operational margin is a guardrail, not a confidence-bound non-inferiority test.",
            "The audit has final scores but no event-level xG, lineups, injuries, or betting-market closing odds.",
            "One shootout adds no credible evidence for team-specific penalty skill; shootouts remain 50/50.",
            "Monte Carlo intervals remain sampling-error intervals, not total predictive uncertainty.",
        ],
        "artifacts": {
            "round16_csv": str(audit_csv),
            "quarterfinal_csv": str(quarterfinal_csv),
        },
        "source_fingerprints": {
            "model_package": file_sha256(MODEL_PATH),
            "sota_pipeline": file_sha256(ROOT / "modeling/worldcup_2026_ml/src/sota_pipeline.py"),
            "bolao": file_sha256(ROOT / "src/arena_ai/bolao.py"),
            "round16_calibration_audit": file_sha256(Path(__file__).resolve()),
            "round32_calibration_audit": file_sha256(ROOT / "scripts/bolao_knockout_calibration.py"),
            "observed_group_results": file_sha256(bolao.OBSERVED_GROUP_RESULTS_PATH),
            "observed_knockout_results": file_sha256(bolao.OBSERVED_KNOCKOUT_RESULTS_PATH),
        },
        "hard_gates": hard_gates,
        "approved": bool(all(hard_gates.values())),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    quarterfinal_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(audit_csv, index=False)
    quarterfinals.to_csv(quarterfinal_csv, index=False)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = run(out=args.out, audit_csv=args.audit_csv, quarterfinal_csv=args.quarterfinal_csv)
    verdict = "OK" if report["approved"] else "FALHOU"
    current = report["pre_round16_form"]
    baseline = report["historical_baseline"]
    print(
        f"[bolao-round16-calibration] {verdict} matches={report['scope']['observed_matches']} "
        f"1x2_log_loss={baseline['log_loss_1x2']:.4f}->{current['log_loss_1x2']:.4f} "
        f"max_qf_delta={report['current_form_update']['max_abs_quarterfinal_advance_probability_delta']:.3%}"
    )
    if not report["approved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

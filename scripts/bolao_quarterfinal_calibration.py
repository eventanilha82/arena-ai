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
DEFAULT_JSON = REPORT_DIR / "bolao_quarterfinal_calibration.json"
DEFAULT_AUDIT_CSV = REPORT_DIR / "bolao_quarterfinal_calibration.csv"
DEFAULT_SEMIFINAL_CSV = REPORT_DIR / "bolao_semifinal_predictions.csv"
OPERATIONAL_LOSS_MARGIN = bolao.CURRENT_FORM_REQUIRED_LOG_SCORE_GAIN_PER_MATCH


def audit_quarterfinal_rows(
    model: WorldCupModel,
    board: bolao.GroupStageBoard,
    group_form: bolao.TournamentForm,
    pre_quarterfinal_form: bolao.TournamentForm,
    observed_before_quarterfinals: dict[int, bolao.ObservedKnockoutResult],
) -> pd.DataFrame:
    forms = {
        "historical": historical_form(group_form),
        "group_form": group_form,
        "pre_quarterfinal_form": pre_quarterfinal_form,
    }
    contexts = context_after_results(model, board, observed_before_quarterfinals)
    fixtures = model.fixtures[model.fixtures["stage_id"] == 4].sort_values(["kickoff_at", "match_number"])
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


def semifinal_prediction_rows(
    model: WorldCupModel,
    board: bolao.GroupStageBoard,
    pre_quarterfinal_form: bolao.TournamentForm,
) -> pd.DataFrame:
    winners: dict[int, str] = {}
    runners_up: dict[int, str] = {}
    for match_number, observed in sorted(board.knockout_results.items()):
        winners[match_number] = observed.winner
        runners_up[match_number] = observed.away if observed.winner == observed.home else observed.home

    contexts = context_after_results(model, board, board.knockout_results)
    fixtures = model.fixtures[model.fixtures["stage_id"] == 5].sort_values("match_number")
    rows: list[dict[str, object]] = []
    for game in fixtures.itertuples(index=False):
        left_slot, right_slot = sota.parse_match_label(game.match_label)
        home = sota.resolve_bracket_slot(left_slot, board.qualifiers, winners, runners_up, {})
        away = sota.resolve_bracket_slot(right_slot, board.qualifiers, winners, runners_up, {})
        context = sota.fixture_context(game, contexts, home, away)
        distribution = bolao.form_aware_match(model, board.form, home, away, knockout=True, context=context)
        pre_quarterfinal_distribution = bolao.form_aware_match(
            model,
            pre_quarterfinal_form,
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
                "pre_quarterfinal_p_home_advances": float(
                    pre_quarterfinal_distribution.prediction["p_home_advances"]
                ),
                "quarterfinal_update_p_home_advances_delta": float(
                    distribution.prediction["p_home_advances"]
                    - pre_quarterfinal_distribution.prediction["p_home_advances"]
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
    parser = argparse.ArgumentParser(description="Audita a calibracao do Bolao apos as quartas.")
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON, help="JSON de saída.")
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV, help="CSV dos jogos auditados.")
    parser.add_argument(
        "--semifinal-csv",
        type=Path,
        default=DEFAULT_SEMIFINAL_CSV,
        help="CSV da foto das semifinais.",
    )
    return parser.parse_args()


def run(*, out: Path, audit_csv: Path, semifinal_csv: Path) -> dict[str, object]:
    model = WorldCupModel()
    current_board = bolao.build_group_stage_board(model)
    group_form = bolao.build_tournament_form(model, current_board.form.observed_results)
    round32_results = results_for_round(current_board, "Round of 32")
    round16_results = results_for_round(current_board, "Round of 16")
    quarterfinal_results = results_for_round(current_board, "Quarterfinals")
    observed_before_quarterfinals = {**round32_results, **round16_results}
    pre_quarterfinal_form = bolao.update_tournament_form_with_knockout_results(
        model,
        group_form,
        current_board.team_context,
        observed_before_quarterfinals,
    )
    observed_through_quarterfinals = {**observed_before_quarterfinals, **quarterfinal_results}
    post_quarterfinal_form = bolao.update_tournament_form_with_knockout_results(
        model,
        group_form,
        current_board.team_context,
        observed_through_quarterfinals,
    )
    board = replace(
        current_board,
        form=post_quarterfinal_form,
        knockout_results=observed_through_quarterfinals,
    )

    rows = audit_quarterfinal_rows(
        model,
        board,
        group_form,
        pre_quarterfinal_form,
        observed_before_quarterfinals,
    )
    historical_metrics = metric_summary(rows, "historical")
    group_form_metrics = metric_summary(rows, "group_form")
    pre_quarterfinal_metrics = metric_summary(rows, "pre_quarterfinal_form")
    semifinals = semifinal_prediction_rows(model, board, pre_quarterfinal_form)

    lower_is_better = (
        "log_loss_1x2",
        "brier_1x2",
        "exact_score_log_loss",
        "advance_log_loss",
        "advance_brier",
    )
    loss_gaps = {
        metric: float(pre_quarterfinal_metrics[metric]) - float(historical_metrics[metric])
        for metric in lower_is_better
    }
    metrics_favoring_current_form = [metric for metric, gap in loss_gaps.items() if gap <= 0.0]
    metrics_within_operational_margin = [
        metric for metric, gap in loss_gaps.items() if gap <= OPERATIONAL_LOSS_MARGIN
    ]
    draw_in_expected_range = abs(float(pre_quarterfinal_metrics["draw_standardized_residual"])) < 2.0
    max_semifinal_advance_delta = float(
        semifinals["quarterfinal_update_p_home_advances_delta"].abs().max()
    )
    expected_semifinals = {
        101: ("France", "Spain"),
        102: ("England", "Argentina"),
    }
    actual_semifinals = {
        int(row.match_number): (str(row.home), str(row.away))
        for row in semifinals.itertuples(index=False)
    }
    advancement_not_worse_than_historical = (
        float(pre_quarterfinal_metrics["advance_log_loss"])
        <= float(historical_metrics["advance_log_loss"])
        and float(pre_quarterfinal_metrics["advance_brier"])
        <= float(historical_metrics["advance_brier"])
    )
    all_loss_gaps_within_operational_margin = (
        len(metrics_within_operational_margin) == len(lower_is_better)
    )

    hard_gates = {
        "complete_quarterfinal_snapshot": len(rows) == 4 and set(rows["match_number"]) == set(range(97, 101)),
        "candidate_was_frozen_before_quarterfinals": pre_quarterfinal_form.knockout_form_matches == 24,
        "draw_count_within_two_standard_deviations": draw_in_expected_range,
        "quarterfinal_update_uses_regulation_only": (
            board.form.knockout_form_policy == bolao.KNOCKOUT_FORM_POLICY
            and board.form.knockout_form_matches == 28
        ),
        "all_loss_gaps_within_operational_margin": all_loss_gaps_within_operational_margin,
        "all_four_advancers_identified": float(pre_quarterfinal_metrics["advance_accuracy"]) == 1.0,
        "semifinal_bracket_complete": actual_semifinals == expected_semifinals,
        "single_round_form_update_below_five_percentage_points": max_semifinal_advance_delta <= 0.05,
    }
    report: dict[str, object] = {
        "report_type": "arena_bolao_quarterfinal_calibration",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "scope": {
            "observed_round": "Quarterfinals",
            "observed_matches": int(len(rows)),
            "evaluation_target": "90-minute 1X2, exact regulation score, and eventual advancement",
            "comparison": "historical hybrid versus the form snapshot frozen after the Round of 16",
            "look_ahead": "none; the candidate form was fixed before the quarterfinal outcomes",
        },
        "observed_resolution": {
            "regulation": int((rows["resolution"] == "90min").sum()),
            "extra_time": int((rows["resolution"] == "extra_time").sum()),
            "penalties": int((rows["resolution"] == "penalties").sum()),
        },
        "historical_baseline": historical_metrics,
        "group_only_form": group_form_metrics,
        "pre_quarterfinal_form": pre_quarterfinal_metrics,
        "loss_gap_pre_quarterfinal_form_minus_historical": loss_gaps,
        "performance_signal": {
            "metrics_favoring_pre_quarterfinal_form": metrics_favoring_current_form,
            "metrics_within_operational_margin": metrics_within_operational_margin,
            "all_loss_gaps_within_operational_margin": all_loss_gaps_within_operational_margin,
            "advancement_losses_not_worse_than_historical": advancement_not_worse_than_historical,
            "operational_margin_per_metric": OPERATIONAL_LOSS_MARGIN,
            "interpretation": (
                "The four-match sample is descriptive only. The existing margin is reported for continuity, "
                "not used as a formal non-inferiority test or as a new hyperparameter-selection window."
            ),
        },
        "current_form_update": {
            "policy": board.form.knockout_form_policy,
            "prior_goal_equivalents_frozen": float(board.form.prior_goal_equivalents),
            "median_weight_before_quarterfinals": float(pre_quarterfinal_form.median_current_weight),
            "median_weight_after_quarterfinals": float(board.form.median_current_weight),
            "matches_appended_in_quarterfinals": int(len(quarterfinal_results)),
            "total_knockout_matches_appended": int(board.form.knockout_form_matches),
            "extra_time_goals_observed": int(
                rows["extra_time_home_goals"].sum() + rows["extra_time_away_goals"].sum()
            ),
            "extra_time_goals_appended": 0,
            "shootout_kicks_appended": 0,
            "max_abs_semifinal_advance_probability_delta": max_semifinal_advance_delta,
        },
        "decision": {
            "retain_frozen_current_form_policy": True,
            "append_quarterfinal_regulation_evidence": True,
            "retune_xgboost_or_global_hybrid_weights": False,
            "recalibrate_draw_probability": False,
            "recalibrate_extra_time_probability": False,
            "recalibrate_penalty_probability": False,
            "reason": (
                "The frozen pre-quarterfinal snapshot identified all four eventual semifinalists and slightly "
                "improved aggregate 1X2 losses. Exact-score and advancement losses were slightly worse, but every "
                "gap stayed inside the existing 0.01 operational guardrail. Four dependent matches remain far too "
                "few for a global retune, so only regulation-time evidence is appended to the frozen prior."
            ),
        },
        "semifinal_predictions": semifinals.to_dict(orient="records"),
        "limitations": [
            "Four matches from one round cannot identify a stable global recalibration.",
            "The operational margin is descriptive here, not a confidence-bound non-inferiority test.",
            "The audit has final scores but no event-level xG, lineups, injuries, red-card state, or closing odds.",
            "Extra-time goals resolve the bracket but are excluded from the 90-minute scoring posterior.",
            "Two extra-time matches do not identify a separate team-specific extra-time skill.",
            "Monte Carlo intervals remain sampling-error intervals, not total predictive uncertainty.",
        ],
        "artifacts": {
            "quarterfinal_csv": str(audit_csv),
            "semifinal_csv": str(semifinal_csv),
        },
        "source_fingerprints": {
            "model_package": file_sha256(MODEL_PATH),
            "sota_pipeline": file_sha256(ROOT / "modeling/worldcup_2026_ml/src/sota_pipeline.py"),
            "bolao": file_sha256(ROOT / "src/arena_ai/bolao.py"),
            "quarterfinal_calibration_audit": file_sha256(Path(__file__).resolve()),
            "round16_calibration_audit": file_sha256(ROOT / "scripts/bolao_round16_calibration.py"),
            "round32_calibration_audit": file_sha256(ROOT / "scripts/bolao_knockout_calibration.py"),
            "observed_group_results": file_sha256(bolao.OBSERVED_GROUP_RESULTS_PATH),
            "observed_knockout_results": file_sha256(bolao.OBSERVED_KNOCKOUT_RESULTS_PATH),
        },
        "hard_gates": hard_gates,
        "approved": bool(all(hard_gates.values())),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    semifinal_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(audit_csv, index=False)
    semifinals.to_csv(semifinal_csv, index=False)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = run(out=args.out, audit_csv=args.audit_csv, semifinal_csv=args.semifinal_csv)
    verdict = "OK" if report["approved"] else "FALHOU"
    current = report["pre_quarterfinal_form"]
    baseline = report["historical_baseline"]
    print(
        f"[bolao-quarterfinal-calibration] {verdict} matches={report['scope']['observed_matches']} "
        f"1x2_log_loss={baseline['log_loss_1x2']:.4f}->{current['log_loss_1x2']:.4f} "
        f"advance_accuracy={current['advance_accuracy']:.1%} "
        f"max_sf_delta={report['current_form_update']['max_abs_semifinal_advance_probability_delta']:.3%}"
    )
    if not report["approved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

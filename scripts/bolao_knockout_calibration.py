from __future__ import annotations

import argparse
import hashlib
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


REPORT_DIR = ROOT / "modeling" / "worldcup_2026_ml" / "reports"
DEFAULT_JSON = REPORT_DIR / "bolao_round32_calibration.json"
DEFAULT_AUDIT_CSV = REPORT_DIR / "bolao_round32_calibration.csv"
DEFAULT_ROUND16_CSV = REPORT_DIR / "bolao_round16_predictions.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_form(form: bolao.TournamentForm) -> bolao.TournamentForm:
    return bolao.historical_tournament_form(
        form.observed_results,
        status="audit_historical_baseline",
        validation_matches=form.validation_matches,
        validation_log_likelihood=form.validation_log_likelihood,
        historical_validation_log_likelihood=form.historical_validation_log_likelihood,
    )


def actual_outcome(home_goals: int, away_goals: int) -> int:
    return 0 if home_goals > away_goals else 2 if away_goals > home_goals else 1


def audit_round32_rows(
    model: WorldCupModel,
    board: bolao.GroupStageBoard,
    group_form: bolao.TournamentForm,
) -> pd.DataFrame:
    baseline = historical_form(group_form)
    contexts = {team: dict(context) for team, context in board.team_context.items()}
    fixtures = model.fixtures[model.fixtures["stage_id"] == 2].sort_values(["kickoff_at", "match_number"])
    rows: list[dict[str, object]] = []
    classifier_weight = float(board.policy["classifier_weight"])
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
            "actual_1x2": ("home", "draw", "away")[outcome],
            "actual_home_advanced": int(observed.winner == observed.home),
        }
        for label, form in (("historical", baseline), ("group_form", group_form)):
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


def metric_summary(rows: pd.DataFrame, label: str) -> dict[str, float | int]:
    probability_columns = [f"{label}_p_home_90", f"{label}_p_draw_90", f"{label}_p_away_90"]
    probabilities = rows[probability_columns].to_numpy(dtype=float)
    outcomes = np.array(
        [actual_outcome(int(row.home_goals_90), int(row.away_goals_90)) for row in rows.itertuples(index=False)],
        dtype=int,
    )
    one_hot = np.eye(3, dtype=float)[outcomes]
    selected = probabilities[np.arange(len(rows)), outcomes]
    actual_home_advanced = rows["actual_home_advanced"].to_numpy(dtype=float)
    home_advance = rows[f"{label}_p_home_advances"].to_numpy(dtype=float)
    expected_draws = float(probabilities[:, 1].sum())
    observed_draws = int(np.sum(outcomes == 1))
    draw_variance = float(np.sum(probabilities[:, 1] * (1.0 - probabilities[:, 1])))
    return {
        "matches": int(len(rows)),
        "log_loss_1x2": float(-np.log(np.maximum(1e-12, selected)).mean()),
        "brier_1x2": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "accuracy_1x2": float(np.mean(np.argmax(probabilities, axis=1) == outcomes)),
        "exact_score_log_loss": float(
            -np.log(np.maximum(1e-12, rows[f"{label}_score_probability"].to_numpy(dtype=float))).mean()
        ),
        "advance_log_loss": float(
            -np.mean(
                actual_home_advanced * np.log(np.maximum(1e-12, home_advance))
                + (1.0 - actual_home_advanced) * np.log(np.maximum(1e-12, 1.0 - home_advance))
            )
        ),
        "advance_brier": float(np.square(home_advance - actual_home_advanced).mean()),
        "advance_accuracy": float(np.mean((home_advance >= 0.5) == (actual_home_advanced == 1.0))),
        "observed_draws": observed_draws,
        "expected_draws": expected_draws,
        "draw_standardized_residual": float(
            (observed_draws - expected_draws) / sqrt(draw_variance) if draw_variance > 0 else 0.0
        ),
    }


def round16_prediction_rows(
    model: WorldCupModel,
    board: bolao.GroupStageBoard,
    group_form: bolao.TournamentForm,
) -> pd.DataFrame:
    knockout_games = model.fixtures[model.fixtures["stage_id"] > sota.GROUP_STAGE_ID].sort_values("match_number")
    round32_games = knockout_games[knockout_games["stage_id"] == 2]
    round32_slots = [
        slot
        for game in round32_games.itertuples(index=False)
        for slot in sota.parse_match_label(game.match_label)
        if slot.startswith("3")
    ]
    third_slot_assignment = sota.assign_third_slots(round32_slots, board.third_order)
    winners: dict[int, str] = {}
    runners_up: dict[int, str] = {}
    for match_number, observed in sorted(board.knockout_results.items()):
        loser = observed.away if observed.winner == observed.home else observed.home
        winners[match_number] = observed.winner
        runners_up[match_number] = loser

    contexts = {team: dict(context) for team, context in board.team_context.items()}
    observed_fixtures = knockout_games[knockout_games["match_number"].isin(board.knockout_results)].sort_values(
        ["kickoff_at", "match_number"]
    )
    for game in observed_fixtures.itertuples(index=False):
        observed = board.knockout_results[int(game.match_number)]
        sota.update_team_context(contexts, observed.home, observed.away, game)

    rows: list[dict[str, object]] = []
    for game in knockout_games[knockout_games["stage_id"] == 3].sort_values("match_number").itertuples(index=False):
        left_slot, right_slot = sota.parse_match_label(game.match_label)
        home = sota.resolve_bracket_slot(left_slot, board.qualifiers, winners, runners_up, third_slot_assignment)
        away = sota.resolve_bracket_slot(right_slot, board.qualifiers, winners, runners_up, third_slot_assignment)
        context = sota.fixture_context(game, contexts, home, away)
        distribution = bolao.form_aware_match(model, board.form, home, away, knockout=True, context=context)
        group_only_distribution = bolao.form_aware_match(
            model,
            group_form,
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
                "group_only_p_home_advances": float(group_only_distribution.prediction["p_home_advances"]),
                "knockout_update_p_home_advances_delta": float(
                    distribution.prediction["p_home_advances"]
                    - group_only_distribution.prediction["p_home_advances"]
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
    parser = argparse.ArgumentParser(description="Audita a calibracao do Bolao apos os 16 avos.")
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON, help="JSON de saída.")
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV, help="CSV dos jogos auditados.")
    parser.add_argument("--round16-csv", type=Path, default=DEFAULT_ROUND16_CSV, help="CSV da foto das oitavas.")
    return parser.parse_args()


def run(*, out: Path, audit_csv: Path, round16_csv: Path) -> dict[str, object]:
    model = WorldCupModel()
    current_board = bolao.build_group_stage_board(model)
    group_form = bolao.build_tournament_form(model, current_board.form.observed_results)
    round32_results = {
        match_number: result
        for match_number, result in current_board.knockout_results.items()
        if result.round_name == "Round of 32"
    }
    round32_form = bolao.update_tournament_form_with_knockout_results(
        model,
        group_form,
        current_board.team_context,
        round32_results,
    )
    board = replace(current_board, form=round32_form, knockout_results=round32_results)
    rows = audit_round32_rows(model, board, group_form)
    historical_metrics = metric_summary(rows, "historical")
    group_form_metrics = metric_summary(rows, "group_form")
    round16 = round16_prediction_rows(model, board, group_form)

    lower_is_better = (
        "log_loss_1x2",
        "brier_1x2",
        "exact_score_log_loss",
        "advance_log_loss",
        "advance_brier",
    )
    gains = {
        metric: float(historical_metrics[metric]) - float(group_form_metrics[metric])
        for metric in lower_is_better
    }
    directionally_confirmed = all(gain >= -1e-12 for gain in gains.values())
    draw_in_expected_range = abs(float(group_form_metrics["draw_standardized_residual"])) < 2.0
    max_round16_advance_delta = float(round16["knockout_update_p_home_advances_delta"].abs().max())
    hard_gates = {
        "complete_round32_snapshot": len(rows) == 16 and set(rows["match_number"]) == set(range(73, 89)),
        "group_form_was_temporally_validated": group_form.calibration_status == "enabled_validation",
        "out_of_sample_metrics_directionally_confirm_form": directionally_confirmed,
        "draw_count_within_two_standard_deviations": draw_in_expected_range,
        "knockout_update_uses_regulation_only": (
            board.form.knockout_form_policy == bolao.KNOCKOUT_FORM_POLICY
            and board.form.knockout_form_matches == 16
        ),
        "round16_bracket_complete": len(round16) == 8,
        "single_round_form_update_below_five_percentage_points": max_round16_advance_delta <= 0.05,
    }
    report: dict[str, object] = {
        "report_type": "arena_bolao_round32_calibration",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "scope": {
            "observed_round": "Round of 32",
            "observed_matches": int(len(rows)),
            "evaluation_target": "90-minute 1X2, exact regulation score, and eventual advancement",
            "comparison": "historical hybrid versus group-stage-validated current form",
            "look_ahead": "none; both candidates were fixed before the Round of 32 outcomes",
        },
        "observed_resolution": {
            "regulation": int((rows["resolution"] == "90min").sum()),
            "extra_time": int((rows["resolution"] == "extra_time").sum()),
            "penalties": int((rows["resolution"] == "penalties").sum()),
        },
        "historical_baseline": historical_metrics,
        "group_validated_form": group_form_metrics,
        "loss_reduction_historical_minus_group_form": gains,
        "current_form_update": {
            "policy": board.form.knockout_form_policy,
            "prior_goal_equivalents_frozen": float(board.form.prior_goal_equivalents),
            "prior_is_original_grid_upper_boundary": bool(
                board.form.prior_goal_equivalents == max(bolao.CURRENT_FORM_PRIOR_GOAL_GRID)
            ),
            "prior_boundary_action": "frozen_without_round32_hindsight_retuning",
            "median_weight_after_groups": float(group_form.median_current_weight),
            "median_weight_after_round32": float(board.form.median_current_weight),
            "matches_appended": int(board.form.knockout_form_matches),
            "extra_time_goals_appended": 0,
            "shootout_kicks_appended": 0,
            "max_abs_round16_advance_probability_delta": max_round16_advance_delta,
        },
        "decision": {
            "retain_group_validated_form": directionally_confirmed,
            "append_round32_regulation_evidence": directionally_confirmed,
            "retune_xgboost_or_global_hybrid_weights": False,
            "recalibrate_draw_probability": False,
            "recalibrate_penalty_probability": False,
            "reason": (
                "The independent Round of 32 check improves every audited loss metric, but 16 matches from one "
                "tournament are too few and too dependent for a new global fit. The existing prior remains frozen."
            ),
        },
        "round16_predictions": round16.to_dict(orient="records"),
        "limitations": [
            "Sixteen matches are enough for a directional check, not for global hyperparameter selection.",
            "All matches come from one tournament and are not an independent historical population.",
            "The prior is at the original grid boundary; broader temporal windows did not select a stable replacement.",
            "The audit has final scores but no event-level xG, lineups, injuries, or betting-market closing odds.",
            "Penalty shootouts remain 50/50 because three shootouts cannot identify a stable team skill.",
            "Monte Carlo intervals remain sampling-error intervals, not total predictive uncertainty.",
        ],
        "artifacts": {
            "round32_csv": str(audit_csv),
            "round16_csv": str(round16_csv),
        },
        "source_fingerprints": {
            "model_package": file_sha256(MODEL_PATH),
            "sota_pipeline": file_sha256(ROOT / "modeling/worldcup_2026_ml/src/sota_pipeline.py"),
            "bolao": file_sha256(ROOT / "src/arena_ai/bolao.py"),
            "calibration_audit": file_sha256(Path(__file__).resolve()),
            "observed_group_results": file_sha256(bolao.OBSERVED_GROUP_RESULTS_PATH),
            "observed_knockout_results": file_sha256(bolao.OBSERVED_KNOCKOUT_RESULTS_PATH),
        },
        "hard_gates": hard_gates,
        "approved": bool(all(hard_gates.values())),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    round16_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(audit_csv, index=False)
    round16.to_csv(round16_csv, index=False)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = run(out=args.out, audit_csv=args.audit_csv, round16_csv=args.round16_csv)
    verdict = "OK" if report["approved"] else "FALHOU"
    current = report["group_validated_form"]
    baseline = report["historical_baseline"]
    print(
        f"[bolao-knockout-calibration] {verdict} matches={report['scope']['observed_matches']} "
        f"1x2_log_loss={baseline['log_loss_1x2']:.4f}->{current['log_loss_1x2']:.4f} "
        f"draws={current['observed_draws']}/{current['expected_draws']:.2f}"
    )
    if not report["approved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

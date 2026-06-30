from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arena_ai import bolao  # noqa: E402
from arena_ai.worldcup_model import MODEL_PATH, WorldCupModel, sota  # noqa: E402


REPORT_DIR = ROOT / "modeling" / "worldcup_2026_ml" / "reports"
DEFAULT_JSON = REPORT_DIR / "bolao_top10_bias_audit.json"
DEFAULT_CSV = REPORT_DIR / "bolao_top10_bias_audit.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def ranking_map(ranking: list[bolao.ChampionOption]) -> dict[str, bolao.ChampionOption]:
    return {option.team: option for option in ranking}


def historical_form_board(board: bolao.GroupStageBoard) -> bolao.GroupStageBoard:
    return replace(
        board,
        form=bolao.historical_tournament_form(
            board.form.observed_results,
            status="audit_historical_baseline",
            validation_matches=board.form.validation_matches,
            validation_log_likelihood=board.form.validation_log_likelihood,
            historical_validation_log_likelihood=board.form.historical_validation_log_likelihood,
        ),
    )


def order_invariance_rows(
    model: WorldCupModel,
    form: bolao.TournamentForm,
    candidates: list[str],
    opponents: list[str],
) -> list[dict[str, object]]:
    rho = sota.dixon_coles_rho_from_package(model.package)
    rows: list[dict[str, object]] = []
    for team in candidates:
        for opponent in opponents:
            if team == opponent:
                continue
            forward = bolao.form_aware_match(model, form, team, opponent, knockout=True).prediction
            reverse = bolao.form_aware_match(model, form, opponent, team, knockout=True).prediction
            forward_resolution = bolao.knockout_resolution_policy(forward, rho=rho)
            reverse_resolution = bolao.knockout_resolution_policy(reverse, rho=rho)
            deltas = {
                "p90_win_order_delta": abs(float(forward["p_home_win_90"]) - float(reverse["p_away_win_90"])),
                "p90_draw_order_delta": abs(float(forward["p_draw_90"]) - float(reverse["p_draw_90"])),
                "p90_loss_order_delta": abs(float(forward["p_away_win_90"]) - float(reverse["p_home_win_90"])),
                "advance_order_delta": abs(float(forward["p_home_advances"]) - float(reverse["p_away_advances"])),
                "advance_if_draw_complement_delta": abs(
                    float(forward["p_home_advances_if_draw"]) + float(reverse["p_home_advances_if_draw"]) - 1.0
                ),
                "home_xg_order_delta": abs(float(forward["home_xg"]) - float(reverse["away_xg"])),
                "away_xg_order_delta": abs(float(forward["away_xg"]) - float(reverse["home_xg"])),
                "forward_probability_sum_error": abs(
                    float(forward["p_home_win_90"]) + float(forward["p_draw_90"]) + float(forward["p_away_win_90"]) - 1.0
                ),
                "reverse_probability_sum_error": abs(
                    float(reverse["p_home_win_90"]) + float(reverse["p_draw_90"]) + float(reverse["p_away_win_90"]) - 1.0
                ),
                "forward_penalty_neutrality_error": abs(float(forward_resolution.home_penalty_probability) - 0.5),
                "reverse_penalty_neutrality_error": abs(float(reverse_resolution.home_penalty_probability) - 0.5),
            }
            rows.append(
                {
                    "team": team,
                    "opponent": opponent,
                    "p_team_win_90": float(forward["p_home_win_90"]),
                    "p_draw_90": float(forward["p_draw_90"]),
                    "p_team_advances": float(forward["p_home_advances"]),
                    "p_team_advances_if_draw": float(forward["p_home_advances_if_draw"]),
                    "team_xg": float(forward["home_xg"]),
                    "opponent_xg": float(forward["away_xg"]),
                    "form_weight": float(forward.get("form_weight", 0.0)),
                    **deltas,
                }
            )
    return rows


def team_rows(
    active_ranking: list[bolao.ChampionOption],
    historical_ranking: list[bolao.ChampionOption],
    pair_rows: pd.DataFrame,
    runs: int,
) -> list[dict[str, object]]:
    historical = ranking_map(historical_ranking)
    rows: list[dict[str, object]] = []
    delta_columns = [
        "p90_win_order_delta",
        "p90_draw_order_delta",
        "p90_loss_order_delta",
        "advance_order_delta",
        "advance_if_draw_complement_delta",
        "home_xg_order_delta",
        "away_xg_order_delta",
        "forward_probability_sum_error",
        "reverse_probability_sum_error",
        "forward_penalty_neutrality_error",
        "reverse_penalty_neutrality_error",
    ]
    for option in active_ranking:
        subset = pair_rows[pair_rows["team"] == option.team]
        historical_option = historical.get(option.team)
        historical_probability = float(historical_option.probability) if historical_option is not None else 0.0
        lower, upper = bolao.mc_sampling_interval_95(float(option.probability), runs)
        row = {
            "rank": int(option.rank),
            "team": option.team,
            "titles": int(option.wins),
            "champion_probability": float(option.probability),
            "champion_probability_mc95_low": float(lower),
            "champion_probability_mc95_high": float(upper),
            "historical_form_probability": historical_probability,
            "current_form_delta": float(option.probability) - historical_probability,
            "max_single_match_advance": float(subset["p_team_advances"].max()),
            "min_single_match_advance": float(subset["p_team_advances"].min()),
        }
        row.update({f"max_{column}": float(subset[column].max()) for column in delta_columns})
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita vieses invisiveis nos dez principais candidatos do Bolao.")
    parser.add_argument("--runs", type=int, default=2000, help="Copas Monte Carlo por cenário de forma.")
    parser.add_argument("--seed", type=int, default=20260629, help="Seed reproduzível para a comparação de cenários.")
    parser.add_argument("--top", type=int, default=10, help="Quantidade de candidatos do ranking a auditar.")
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON, help="JSON de saída.")
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV, help="CSV de pares de confronto de saída.")
    return parser.parse_args()


def run(*, runs: int, seed: int, top: int, out: Path, csv_out: Path) -> dict[str, object]:
    if runs <= 0:
        raise ValueError("runs deve ser positivo")
    if top <= 0:
        raise ValueError("top deve ser positivo")

    model = WorldCupModel()
    board = bolao.build_group_stage_board(model)
    active_ranking = bolao.build_champion_ranking(
        model,
        board,
        runs=runs,
        seed=seed,
        top_n=min(top, len(board.qualified_teams)),
        workers=1,
    )
    baseline_ranking = bolao.build_champion_ranking(
        model,
        historical_form_board(board),
        runs=runs,
        seed=seed,
        top_n=len(board.qualified_teams),
        workers=1,
    )

    candidates = [option.team for option in active_ranking]
    opponents = sorted(board.qualified_teams)
    pairs = pd.DataFrame(order_invariance_rows(model, board.form, candidates, opponents))
    pairs = pairs.sort_values(["team", "opponent"]).reset_index(drop=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(csv_out, index=False)

    rows = team_rows(active_ranking, baseline_ranking, pairs, runs)
    observed_losers = sorted(
        result.away if result.winner == result.home else result.home for result in board.knockout_results.values()
    )
    all_delta_columns = [column for column in pairs.columns if column.endswith("_delta") or column.endswith("_error")]
    max_order_delta = max(float(pairs[column].max()) for column in all_delta_columns if "penalty" not in column)
    max_penalty_error = max(float(pairs[column].max()) for column in all_delta_columns if "penalty" in column)
    hard_gates = {
        "neutral_order_invariant": max_order_delta <= 1e-10,
        "probabilities_normalized": max(
            float(pairs["forward_probability_sum_error"].max()), float(pairs["reverse_probability_sum_error"].max())
        ) <= 1e-10,
        "penalties_neutral": max_penalty_error <= 1e-12,
        "observed_knockout_losers_excluded": not bool(set(candidates).intersection(observed_losers)),
        "current_form_validation_guarded": board.form.calibration_status
        in {"enabled_validation", "fallback_history_validation", "insufficient_observed_matches", "insufficient_training_matches"},
        "neutral_order_symmetrization_enabled": all(
            bool(sota.predict_match(model.package, team, opponents[0], neutral=True)["neutral_order_symmetrized"])
            for team in candidates
            if team != opponents[0]
        ),
    }
    report = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "scope": {
            "ranking": "fixed observed group stage, observed knockout results locked, form-aware hybrid Monte Carlo",
            "runs_per_form_scenario": int(runs),
            "seed": int(seed),
            "top": int(len(candidates)),
            "pair_scope": "each top candidate against every other qualified team with no fixture context, to isolate nominal bracket-order bias",
            "monte_carlo_interval": "Wilson 95% sampling error only; not total model uncertainty",
        },
        "current_form": {
            "is_enabled": bool(board.form.is_enabled),
            "calibration_status": board.form.calibration_status,
            "prior_goal_equivalents": float(board.form.prior_goal_equivalents),
            "median_weight": float(board.form.median_current_weight),
            "validation_matches": int(board.form.validation_matches),
            "validation_log_likelihood": float(board.form.validation_log_likelihood),
            "historical_validation_log_likelihood": float(board.form.historical_validation_log_likelihood),
        },
        "observed_knockout_results": [
            {
                "match_number": int(result.match_number),
                "home": result.home,
                "away": result.away,
                "winner": result.winner,
                "resolution": result.resolution,
            }
            for result in board.knockout_results.values()
        ],
        "observed_eliminated_teams": observed_losers,
        "top_10": rows,
        "order_invariance": {
            "pair_rows": int(len(pairs)),
            "max_delta_or_error": float(max_order_delta),
            "max_penalty_neutrality_error": float(max_penalty_error),
            "csv_path": str(csv_out),
        },
        "hard_gates": hard_gates,
        "approved": bool(all(hard_gates.values())),
        "source_fingerprints": {
            "model_package": file_sha256(MODEL_PATH),
            "sota_pipeline": file_sha256(ROOT / "modeling/worldcup_2026_ml/src/sota_pipeline.py"),
            "bolao": file_sha256(ROOT / "src/arena_ai/bolao.py"),
            "observed_group_results": file_sha256(bolao.OBSERVED_GROUP_RESULTS_PATH),
            "observed_knockout_results": file_sha256(bolao.OBSERVED_KNOCKOUT_RESULTS_PATH),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = run(runs=args.runs, seed=args.seed, top=args.top, out=args.out, csv_out=args.csv_out)
    verdict = "OK" if report["approved"] else "FALHOU"
    print(
        f"[bolao-top10-audit] {verdict} top={report['scope']['top']} "
        f"pairs={report['order_invariance']['pair_rows']} "
        f"max_delta={report['order_invariance']['max_delta_or_error']:.3e}"
    )
    if not report["approved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

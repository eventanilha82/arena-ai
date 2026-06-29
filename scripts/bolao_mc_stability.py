from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from math import sqrt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arena_ai import bolao  # noqa: E402
from arena_ai.worldcup_model import SOTA_ROOT, WorldCupModel, sota  # noqa: E402


DEFAULT_OUTPUT = ROOT / "modeling" / "worldcup_2026_ml" / "reports" / "bolao_monte_carlo_stability.json"
DEFAULT_RUNS = (1000, 2000)
DEFAULT_SEED = 20260628
DEFAULT_TOP_N = 10
DEFAULT_MAX_PROBABILITY_DELTA = 0.04
DEFAULT_MIN_TOP_OVERLAP = 0.60
DEFAULT_INDEPENDENT_SEEDS = (20260628, 20260629, 20260630)
DEFAULT_MAX_INDEPENDENT_PROBABILITY_DELTA = 0.06
DEFAULT_MIN_INDEPENDENT_TOP_OVERLAP = 0.60
DEFAULT_MAX_INDEPENDENT_SAMPLING_Z = 4.0


def parse_runs(value: str) -> tuple[int, ...]:
    try:
        runs = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--runs deve conter inteiros separados por vírgula") from error
    if len(runs) < 2 or any(item <= 0 for item in runs):
        raise argparse.ArgumentTypeError("--runs precisa ter ao menos dois tamanhos positivos")
    if tuple(sorted(set(runs))) != runs:
        raise argparse.ArgumentTypeError("--runs precisa estar em ordem estritamente crescente, sem repetição")
    return runs


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--independent-seeds deve conter inteiros separados por vírgula") from error
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--independent-seeds precisa ter ao menos duas seeds distintas")
    return seeds


def mc_sampling_interval_95(probability: float, runs: int) -> tuple[float, float]:
    """Wilson interval for Monte Carlo sampling error only."""
    total = max(1, int(runs))
    value = max(0.0, min(1.0, float(probability)))
    z = 1.96
    denominator = 1.0 + (z * z / total)
    center = (value + (z * z / (2.0 * total))) / denominator
    radius = (z / denominator) * sqrt((value * (1.0 - value) / total) + (z * z / (4.0 * total * total)))
    return max(0.0, center - radius), min(1.0, center + radius)


def payload_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_fingerprints() -> dict[str, dict[str, object]]:
    paths = {
        "bolao": ROOT / "src" / "arena_ai" / "bolao.py",
        "worldcup_model": ROOT / "src" / "arena_ai" / "worldcup_model.py",
        "sota_pipeline": SOTA_ROOT / "src" / "sota_pipeline.py",
        "model_package": SOTA_ROOT / "models" / "model_sota.pkl",
        "runtime_cache": SOTA_ROOT / "models" / "runtime_prediction_cache.pkl",
        "observed_results": bolao.OBSERVED_GROUP_RESULTS_PATH,
        "observed_snapshot": bolao.OBSERVED_SNAPSHOT_METADATA_PATH,
        "stability_audit": Path(__file__),
    }
    return {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(path),
        }
        for name, path in paths.items()
    }


def knockout_setup(model: WorldCupModel, board: bolao.GroupStageBoard) -> tuple[list[object], dict[str, str]]:
    knockout_games = list(
        model.fixtures[model.fixtures["stage_id"] > sota.GROUP_STAGE_ID].sort_values("match_number").itertuples(index=False)
    )
    round32_slots: list[str] = []
    for game in knockout_games:
        if int(game.stage_id) == 2:
            round32_slots.extend(slot for slot in sota.parse_match_label(game.match_label) if slot.startswith("3"))
    return knockout_games, sota.assign_third_slots(round32_slots, board.third_order)


def fixed_group_stage_payload(board: bolao.GroupStageBoard) -> dict[str, object]:
    matches = [
        {
            "match_number": match.match_number,
            "group": match.group,
            "home": match.home,
            "away": match.away,
            "home_goals": match.home_goals,
            "away_goals": match.away_goals,
            "is_observed": match.is_observed,
        }
        for group_matches in board.matches_by_group.values()
        for match in group_matches
    ]
    matches.sort(key=lambda item: int(item["match_number"]))
    form = board.form
    payload = {
        "is_fixed": True,
        "matches": matches,
        "qualified_teams": sorted(board.qualified_teams),
        "third_order": list(board.third_order),
        "form": {
            "is_enabled": form.is_enabled,
            "calibration_status": form.calibration_status,
            "observed_result_count": len(form.observed_results),
            "prior_goal_equivalents": form.prior_goal_equivalents,
            "median_current_weight": form.median_current_weight,
            "validation_matches": form.validation_matches,
            "validation_log_likelihood": form.validation_log_likelihood,
            "historical_validation_log_likelihood": form.historical_validation_log_likelihood,
        },
        "manual_snapshot": {
            "snapshot_kind": board.snapshot.snapshot_kind,
            "as_of": board.snapshot.as_of_utc_text,
            "source_label": board.snapshot.source_label,
            "official_source": board.snapshot.official_source,
            "results_sha256": board.snapshot.results_sha256,
        },
    }
    payload["fingerprint_sha256"] = payload_sha256(payload)
    return payload


def ranking_rows(counts: Counter[str], candidates: list[str], runs: int, top_n: int) -> list[dict[str, object]]:
    observed_champions = [team for team in candidates if counts[team] > 0]
    ordered = sorted(observed_champions, key=lambda team: (-counts[team], team))[: min(top_n, len(observed_champions))]
    rows = []
    for rank, team in enumerate(ordered, start=1):
        wins = int(counts[team])
        probability = wins / runs
        rows.append(
            {
                "rank": rank,
                "team": team,
                "titles": wins,
                "probability": probability,
                "mc_sampling_interval_95": {
                    "lower": mc_sampling_interval_95(probability, runs)[0],
                    "upper": mc_sampling_interval_95(probability, runs)[1],
                },
            }
        )
    return rows


def nested_snapshots(
    model: WorldCupModel,
    board: bolao.GroupStageBoard,
    *,
    runs: tuple[int, ...],
    seed: int,
    top_n: int,
) -> list[dict[str, object]]:
    candidates = sorted(board.qualified_teams)
    knockout_games, third_slot_assignment = knockout_setup(model, board)
    counts: Counter[str] = Counter()
    rng = random.Random(seed)
    targets = set(runs)
    snapshots: list[dict[str, object]] = []

    for completed in range(1, runs[-1] + 1):
        champion = bolao.simulate_form_aware_knockout(
            model,
            board,
            rng,
            knockout_games=knockout_games,
            third_slot_assignment=third_slot_assignment,
        )
        counts[champion] += 1
        if completed not in targets:
            continue
        snapshots.append(
            {
                "runs": completed,
                "top_champions": ranking_rows(counts, candidates, completed, top_n),
                "probabilities": {team: counts[team] / completed for team in candidates},
                "total_titles": int(sum(counts.values())),
            }
        )
    return snapshots


def compare_snapshots(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    top_n: int,
) -> dict[str, object]:
    baseline_probabilities = dict(baseline["probabilities"])
    candidate_probabilities = dict(candidate["probabilities"])
    teams = sorted(set(baseline_probabilities).union(candidate_probabilities))
    deltas = {
        team: abs(float(candidate_probabilities.get(team, 0.0)) - float(baseline_probabilities.get(team, 0.0)))
        for team in teams
    }
    baseline_top = [str(row["team"]) for row in list(baseline["top_champions"])[:top_n]]
    candidate_top = [str(row["team"]) for row in list(candidate["top_champions"])[:top_n]]
    overlap = len(set(baseline_top).intersection(candidate_top)) / max(1, min(len(baseline_top), len(candidate_top)))
    return {
        "from_runs": int(baseline["runs"]),
        "to_runs": int(candidate["runs"]),
        "max_abs_probability_delta": max(deltas.values(), default=0.0),
        "mean_abs_probability_delta": sum(deltas.values()) / max(1, len(deltas)),
        "top_n": top_n,
        "top_n_overlap": overlap,
        "baseline_top": baseline_top,
        "candidate_top": candidate_top,
    }


def gate_result(
    comparison: dict[str, object],
    *,
    max_probability_delta: float,
    min_top_overlap: float,
) -> dict[str, object]:
    max_delta = float(comparison["max_abs_probability_delta"])
    top_overlap = float(comparison["top_n_overlap"])
    checks = {
        "max_abs_probability_delta": {
            "actual": max_delta,
            "limit": max_probability_delta,
            "passed": max_delta <= max_probability_delta,
        },
        "top_n_overlap": {
            "actual": top_overlap,
            "limit": min_top_overlap,
            "passed": top_overlap >= min_top_overlap,
        },
    }
    return {"passed": all(bool(value["passed"]) for value in checks.values()), "checks": checks}


def independent_sampling_max_z(baseline: dict[str, object], candidate: dict[str, object]) -> tuple[float, str | None]:
    baseline_probabilities = dict(baseline["probabilities"])
    candidate_probabilities = dict(candidate["probabilities"])
    baseline_runs = int(baseline["runs"])
    candidate_runs = int(candidate["runs"])
    max_z = 0.0
    max_team: str | None = None
    for team in sorted(set(baseline_probabilities).union(candidate_probabilities)):
        first = float(baseline_probabilities.get(team, 0.0))
        second = float(candidate_probabilities.get(team, 0.0))
        pooled = ((first * baseline_runs) + (second * candidate_runs)) / (baseline_runs + candidate_runs)
        standard_error = sqrt(pooled * (1.0 - pooled) * ((1.0 / baseline_runs) + (1.0 / candidate_runs)))
        z_score = abs(first - second) / standard_error if standard_error > 0 else 0.0
        if z_score > max_z:
            max_z = z_score
            max_team = team
    return max_z, max_team


def independent_seed_audit(
    model: WorldCupModel,
    board: bolao.GroupStageBoard,
    *,
    seeds: tuple[int, ...],
    runs: int,
    top_n: int,
    max_probability_delta: float,
    min_top_overlap: float,
    max_sampling_z: float,
) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for seed in seeds:
        snapshot = nested_snapshots(model, board, runs=(runs,), seed=seed, top_n=top_n)[0]
        samples.append({"seed": seed, **snapshot})

    comparisons: list[dict[str, object]] = []
    for baseline, candidate in combinations(samples, 2):
        comparison = compare_snapshots(baseline, candidate, top_n=top_n)
        max_z, max_team = independent_sampling_max_z(baseline, candidate)
        comparison["baseline_seed"] = int(baseline["seed"])
        comparison["candidate_seed"] = int(candidate["seed"])
        comparison["max_two_sample_z"] = max_z
        comparison["max_two_sample_z_team"] = max_team
        comparisons.append(comparison)

    max_delta = max((float(item["max_abs_probability_delta"]) for item in comparisons), default=0.0)
    min_overlap = min((float(item["top_n_overlap"]) for item in comparisons), default=1.0)
    max_z = max((float(item["max_two_sample_z"]) for item in comparisons), default=0.0)
    checks = {
        "max_abs_probability_delta": {
            "actual": max_delta,
            "limit": max_probability_delta,
            "passed": max_delta <= max_probability_delta,
        },
        "min_top_n_overlap": {
            "actual": min_overlap,
            "limit": min_top_overlap,
            "passed": min_overlap >= min_top_overlap,
        },
        "max_two_sample_z": {
            "actual": max_z,
            "limit": max_sampling_z,
            "passed": max_z <= max_sampling_z,
        },
    }
    return {
        "runs_per_seed": runs,
        "seeds": list(seeds),
        "samples": samples,
        "comparisons": comparisons,
        "gate": {"passed": all(bool(value["passed"]) for value in checks.values()), "checks": checks},
    }


def build_report(
    model: WorldCupModel,
    board: bolao.GroupStageBoard,
    *,
    runs: tuple[int, ...],
    seed: int,
    top_n: int,
    max_probability_delta: float,
    min_top_overlap: float,
    independent_seeds: tuple[int, ...],
    independent_runs: int,
    max_independent_probability_delta: float,
    min_independent_top_overlap: float,
    max_independent_sampling_z: float,
) -> dict[str, object]:
    snapshots = nested_snapshots(model, board, runs=runs, seed=seed, top_n=top_n)
    comparisons = [
        compare_snapshots(snapshots[index - 1], snapshots[index], top_n=top_n)
        for index in range(1, len(snapshots))
    ]
    nested_gate = gate_result(
        comparisons[-1],
        max_probability_delta=max_probability_delta,
        min_top_overlap=min_top_overlap,
    )
    independent_audit = independent_seed_audit(
        model,
        board,
        seeds=independent_seeds,
        runs=independent_runs,
        top_n=top_n,
        max_probability_delta=max_independent_probability_delta,
        min_top_overlap=min_independent_top_overlap,
        max_sampling_z=max_independent_sampling_z,
    )
    independent_gate = dict(independent_audit["gate"])
    return {
        "report_type": "arena_bolao_monte_carlo_stability",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "runs": list(runs),
        "runtime_fingerprints": runtime_fingerprints(),
        "simulation_scope": {
            "group_stage": "fixed_board",
            "current_tournament_form": "included",
            "knockout": "form_aware_hybrid_sampled",
            "sampling": "nested_prefixes_one_seed_plus_independent_seeds",
        },
        "fixed_group_stage": fixed_group_stage_payload(board),
        "uncertainty": {
            "interval": "wilson_score_95_percent",
            "scope": "monte_carlo_sampling_error_only",
            "does_not_measure": [
                "model_misspecification",
                "data_quality_or_manual_snapshot_error",
                "parameter_uncertainty",
                "future_match_uncertainty_beyond_the_simulation_model",
            ],
        },
        "snapshots": snapshots,
        "comparisons": comparisons,
        "independent_seed_audit": independent_audit,
        "gate": {
            "passed": bool(nested_gate["passed"]) and bool(independent_gate["passed"]),
            "nested_prefixes": nested_gate,
            "independent_seeds": independent_gate,
        },
    }


def write_report(path: Path, report: dict[str, object]) -> Path:
    target = path if path.is_absolute() else ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita a estabilidade do Monte Carlo do bolão em amostras aninhadas.")
    parser.add_argument("--runs", type=parse_runs, default=DEFAULT_RUNS, help="Tamanhos aninhados, por exemplo 1000,2000.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-probability-delta", type=float, default=DEFAULT_MAX_PROBABILITY_DELTA)
    parser.add_argument("--min-top-overlap", type=float, default=DEFAULT_MIN_TOP_OVERLAP)
    parser.add_argument("--independent-seeds", type=parse_seeds, default=DEFAULT_INDEPENDENT_SEEDS)
    parser.add_argument(
        "--independent-runs",
        type=int,
        help="Copas por seed independente; por padrão usa o maior valor de --runs.",
    )
    parser.add_argument(
        "--max-independent-probability-delta",
        type=float,
        default=DEFAULT_MAX_INDEPENDENT_PROBABILITY_DELTA,
    )
    parser.add_argument(
        "--min-independent-top-overlap",
        type=float,
        default=DEFAULT_MIN_INDEPENDENT_TOP_OVERLAP,
    )
    parser.add_argument("--max-independent-sampling-z", type=float, default=DEFAULT_MAX_INDEPENDENT_SAMPLING_Z)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top <= 0:
        raise SystemExit("--top deve ser positivo")
    if not 0.0 <= args.max_probability_delta <= 1.0:
        raise SystemExit("--max-probability-delta deve estar entre 0 e 1")
    if not 0.0 <= args.min_top_overlap <= 1.0:
        raise SystemExit("--min-top-overlap deve estar entre 0 e 1")
    if args.independent_runs is not None and args.independent_runs <= 0:
        raise SystemExit("--independent-runs deve ser positivo")
    if not 0.0 <= args.max_independent_probability_delta <= 1.0:
        raise SystemExit("--max-independent-probability-delta deve estar entre 0 e 1")
    if not 0.0 <= args.min_independent_top_overlap <= 1.0:
        raise SystemExit("--min-independent-top-overlap deve estar entre 0 e 1")
    if args.max_independent_sampling_z <= 0.0:
        raise SystemExit("--max-independent-sampling-z deve ser positivo")

    model = WorldCupModel()
    board = bolao.build_group_stage_board(model)
    report = build_report(
        model,
        board,
        runs=tuple(args.runs),
        seed=int(args.seed),
        top_n=int(args.top),
        max_probability_delta=float(args.max_probability_delta),
        min_top_overlap=float(args.min_top_overlap),
        independent_seeds=tuple(args.independent_seeds),
        independent_runs=int(args.independent_runs or args.runs[-1]),
        max_independent_probability_delta=float(args.max_independent_probability_delta),
        min_independent_top_overlap=float(args.min_independent_top_overlap),
        max_independent_sampling_z=float(args.max_independent_sampling_z),
    )
    target = write_report(args.out, report)
    gate = dict(report["gate"])
    comparison = list(report["comparisons"])[-1]
    independent = dict(report["independent_seed_audit"])
    independent_gate = dict(independent["gate"])
    independent_checks = dict(independent_gate["checks"])
    try:
        display_target = target.relative_to(ROOT)
    except ValueError:
        display_target = target
    print(
        "[bolao-mc-stability] "
        f"{comparison['from_runs']}->{comparison['to_runs']} | "
        f"delta max={float(comparison['max_abs_probability_delta']):.2%} | "
        f"overlap top {int(comparison['top_n'])}={float(comparison['top_n_overlap']):.0%} | "
        f"seeds delta={float(independent_checks['max_abs_probability_delta']['actual']):.2%} | "
        f"seeds overlap={float(independent_checks['min_top_n_overlap']['actual']):.0%} | "
        f"seeds z={float(independent_checks['max_two_sample_z']['actual']):.2f} | "
        f"gate={'OK' if gate['passed'] else 'FALHOU'} | {display_target}"
    )
    if not bool(gate["passed"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "modeling" / "worldcup_2026_ml"
SOTA_SRC = MODEL_ROOT / "src"
MODEL_PATH = MODEL_ROOT / "models" / "model_sota.pkl"
RUNTIME_PREDICTION_CACHE = MODEL_ROOT / "models" / "runtime_prediction_cache.pkl"
MODEL_REPORT = MODEL_ROOT / "reports" / "sota_model_report.json"
TRAINING_PATH = MODEL_ROOT / "data" / "processed" / "sota_training_matches.csv"
SOTA_PIPELINE = SOTA_SRC / "sota_pipeline.py"
REPORT_JSON = MODEL_ROOT / "reports" / "sota_monte_carlo_stability.json"
REPORT_CSV = MODEL_ROOT / "reports" / "sota_monte_carlo_stability.csv"
REPORT_STAGE_CSV = MODEL_ROOT / "reports" / "sota_monte_carlo_stage_bracket_stability.csv"
sys.path.insert(0, str(SOTA_SRC))

import sota_pipeline as sota  # noqa: E402


PACKAGE: dict[str, Any] | None = None
BRACKET_STAGES = ("Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final")


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def load_runtime_prediction_cache(package: dict[str, Any]) -> bool:
    if not RUNTIME_PREDICTION_CACHE.exists():
        return False
    try:
        with RUNTIME_PREDICTION_CACHE.open("rb") as file:
            payload = pickle.load(file)
    except Exception:
        return False
    try:
        model_sha = str(file_fingerprint(MODEL_PATH)["sha256"])
        pipeline_sha = str(file_fingerprint(SOTA_PIPELINE)["sha256"])
    except OSError:
        return False
    if str(payload.get("model_sha256", "")) != model_sha:
        return False
    if str(payload.get("sota_pipeline_sha256", "")) != pipeline_sha:
        return False
    prediction_cache = payload.get("prediction_cache")
    if isinstance(prediction_cache, dict):
        package["prediction_cache"] = dict(prediction_cache)
    prediction_base_cache = payload.get("prediction_base_cache")
    if isinstance(prediction_base_cache, dict):
        package["prediction_base_cache"] = dict(prediction_base_cache)
    return bool(isinstance(prediction_cache, dict) or isinstance(prediction_base_cache, dict))


def source_fingerprints() -> dict[str, Any]:
    return {
        "model_package": file_fingerprint(MODEL_PATH),
        "model_report": file_fingerprint(MODEL_REPORT),
        "training_matches": file_fingerprint(TRAINING_PATH),
        "sota_pipeline": file_fingerprint(SOTA_PIPELINE),
        "mc_stability_script": file_fingerprint(Path(__file__).resolve()),
    }


def parse_runs(value: str) -> list[int]:
    runs = []
    for part in value.split(","):
        part = part.strip()
        if part:
            runs.append(max(1, int(part)))
    if not runs:
        raise argparse.ArgumentTypeError("informe pelo menos um volume de Monte Carlo")
    return sorted(set(runs))


def init_worker(model_path: str) -> None:
    global PACKAGE
    with Path(model_path).open("rb") as file:
        PACKAGE = pickle.load(file)
    # Runtime-only caches contain locks; each process owns its own clean cache.
    PACKAGE.pop("_prediction_cache_lock", None)
    PACKAGE.pop("prediction_cache", None)
    PACKAGE.pop("prediction_base_cache", None)
    load_runtime_prediction_cache(PACKAGE)


def run_champion_chunk(start_seed: int, run_count: int) -> dict[str, int]:
    if PACKAGE is None:
        raise RuntimeError("worker package not initialized")
    counts: Counter[str] = Counter()
    for offset in range(run_count):
        counts[sota.simulate_tournament_champion(PACKAGE, start_seed + offset)] += 1
    return dict(counts)


def simulate_stage_bracket_snapshot(package: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    fixtures = package["fixtures"]
    qualifiers, third_order, team_context = sota.simulate_group_stage_fast(package, fixtures, seed)
    winners: dict[int, str] = {}
    runners_up: dict[int, str] = {}
    stage_counts: dict[str, Counter[str]] = {stage: Counter() for stage in BRACKET_STAGES}
    pair_counts: dict[str, Counter[str]] = {stage: Counter() for stage in BRACKET_STAGES}
    knockout_games = fixtures[fixtures["stage_id"] > sota.GROUP_STAGE_ID].sort_values("match_number")
    round32_slots: list[str] = []
    for game in knockout_games[knockout_games["stage_id"] == 2].itertuples(index=False):
        round32_slots.extend(slot for slot in sota.parse_match_label(game.match_label) if slot.startswith("3"))
    third_slot_assignment = sota.assign_third_slots(round32_slots, third_order)
    for game in knockout_games.itertuples(index=False):
        stage = str(game.stage)
        left_slot, right_slot = sota.parse_match_label(game.match_label)
        home = sota.resolve_bracket_slot(left_slot, qualifiers, winners, runners_up, third_slot_assignment)
        away = sota.resolve_bracket_slot(right_slot, qualifiers, winners, runners_up, third_slot_assignment)
        stage_counts.setdefault(stage, Counter()).update([home, away])
        pair_counts.setdefault(stage, Counter()).update([" | ".join(sorted([home, away]))])
        context = sota.fixture_context(game, team_context, home, away)
        winner, _hg, _ag, _resolution, _sim_meta = sota.knockout_winner(package, home, away, rng, context=context)
        sota.update_team_context(team_context, home, away, game)
        loser = away if winner == home else home
        winners[int(game.match_number)] = winner
        runners_up[int(game.match_number)] = loser
    champion = winners[104]
    runner_up = runners_up[104]
    return {
        "champion": champion,
        "runner_up": runner_up,
        "stage_counts": {stage: dict(counter) for stage, counter in stage_counts.items()},
        "pair_counts": {stage: dict(counter) for stage, counter in pair_counts.items()},
    }


def run_stage_bracket_chunk(start_seed: int, run_count: int) -> dict[str, Any]:
    if PACKAGE is None:
        raise RuntimeError("worker package not initialized")
    stage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    champion_counts: Counter[str] = Counter()
    finalist_counts: Counter[str] = Counter()
    for offset in range(run_count):
        snapshot = simulate_stage_bracket_snapshot(PACKAGE, start_seed + offset)
        champion = str(snapshot["champion"])
        runner_up = str(snapshot["runner_up"])
        champion_counts.update([champion])
        finalist_counts.update([champion, runner_up])
        for stage, counts in snapshot["stage_counts"].items():
            stage_counts[str(stage)].update(dict(counts))
        for stage, counts in snapshot["pair_counts"].items():
            pair_counts[str(stage)].update(dict(counts))
    return {
        "champions": dict(champion_counts),
        "finalists": dict(finalist_counts),
        "stages": {stage: dict(counter) for stage, counter in stage_counts.items()},
        "pairs": {stage: dict(counter) for stage, counter in pair_counts.items()},
    }


def build_chunks(runs: int, seed: int, workers: int, chunk_size: int) -> list[tuple[int, int]]:
    target = max(1, int(chunk_size or max(25, runs // max(1, workers * 16))))
    chunks: list[tuple[int, int]] = []
    next_run = 0
    while next_run < runs:
        size = min(target, runs - next_run)
        chunks.append((seed + next_run, size))
        next_run += size
    return chunks


def odds_from_counts(counts: Counter[str], runs: int) -> list[dict[str, Any]]:
    rows = []
    for rank, (team, wins) in enumerate(counts.most_common(), start=1):
        rows.append(
            {
                "rank": rank,
                "team": team,
                "wins": int(wins),
                "probability": round(float(wins) / float(runs), 6),
            }
        )
    return rows


def appearance_rows(counts: Counter[str], runs: int, value_name: str = "appearances", limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for rank, (name, count) in enumerate(counts.most_common(limit), start=1):
        rows.append(
            {
                "rank": rank,
                "name": name,
                value_name: int(count),
                "probability": round(float(count) / float(max(1, runs)), 6),
            }
        )
    return rows


def compare_probability_rows(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    name_key: str = "name",
    top_limit: int | None = None,
) -> dict[str, Any]:
    previous_probs = {str(row[name_key]): float(row["probability"]) for row in previous}
    current_probs = {str(row[name_key]): float(row["probability"]) for row in current}
    previous_top_rows = previous[:top_limit] if top_limit else previous
    current_top_rows = current[:top_limit] if top_limit else current
    previous_top = {str(row[name_key]) for row in previous_top_rows}
    current_top = {str(row[name_key]) for row in current_top_rows}
    union = sorted(previous_top | current_top)
    deltas = [abs(current_probs.get(name, 0.0) - previous_probs.get(name, 0.0)) for name in union]
    return {
        "entered": sorted(current_top - previous_top),
        "exited": sorted(previous_top - current_top),
        "churn_count": int(len(current_top - previous_top) + len(previous_top - current_top)),
        "max_abs_delta": round(max(deltas) if deltas else 0.0, 6),
        "mean_abs_delta": round(sum(deltas) / len(deltas) if deltas else 0.0, 6),
    }


def compare_runs(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if previous is None:
        return {"baseline": True}
    previous_probs = {row["team"]: float(row["probability"]) for row in previous["top_16"]}
    current_probs = {row["team"]: float(row["probability"]) for row in current["top_16"]}
    previous_top = set(previous_probs)
    current_top = set(current_probs)
    union = sorted(previous_top | current_top)
    deltas = [abs(current_probs.get(team, 0.0) - previous_probs.get(team, 0.0)) for team in union]
    entered = sorted(current_top - previous_top)
    exited = sorted(previous_top - current_top)
    return {
        "baseline": False,
        "previous_runs": int(previous["runs"]),
        "leader_changed": str(previous["leader"]) != str(current["leader"]),
        "comparison": "union_top16_abs_delta",
        "union_team_count": int(len(union)),
        "entered_top16": entered,
        "exited_top16": exited,
        "top16_churn_count": int(len(entered) + len(exited)),
        "max_top16_abs_delta": round(max(deltas) if deltas else 0.0, 6),
        "mean_top16_abs_delta": round(sum(deltas) / len(deltas) if deltas else 0.0, 6),
    }


def run_size(
    executor: ThreadPoolExecutor | None,
    runs: int,
    seed: int,
    workers: int,
    chunk_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = time.perf_counter()
    chunks = build_chunks(runs, seed, workers, chunk_size)
    counts: Counter[str] = Counter()
    completed = 0
    next_progress = 0.1
    if executor is None:
        result_iter = (run_champion_chunk(chunk_seed, chunk_count) for chunk_seed, chunk_count in chunks)
    else:
        futures = [executor.submit(run_champion_chunk, chunk_seed, chunk_count) for chunk_seed, chunk_count in chunks]
        result_iter = (future.result() for future in as_completed(futures))
    for result in result_iter:
        counts.update(result)
        completed = sum(counts.values())
        progress = completed / float(runs)
        if progress >= next_progress or completed == runs:
            print(f"[mc-stability] runs={runs} progresso={completed}/{runs} ({progress:.0%})", flush=True)
            next_progress += 0.1
    elapsed = time.perf_counter() - start
    rows = odds_from_counts(counts, runs)
    leader = rows[0]["team"] if rows else "-"
    summary = {
        "runs": int(runs),
        "seed": int(seed),
        "workers": int(workers),
        "chunks": int(len(chunks)),
        "chunk_size": int(chunks[0][1] if chunks else 0),
        "elapsed_seconds": round(elapsed, 3),
        "rate_per_second": round(float(runs) / elapsed, 3) if elapsed > 0 else 0.0,
        "leader": leader,
        "leader_probability": float(rows[0]["probability"]) if rows else 0.0,
        "top_16": rows[:16],
    }
    csv_rows = [
        {
            "runs": int(runs),
            "seed": int(seed),
            "workers": int(workers),
            "rank": row["rank"],
            "team": row["team"],
            "wins": row["wins"],
            "probability": row["probability"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "rate_per_second": summary["rate_per_second"],
        }
        for row in rows
    ]
    return summary, csv_rows


def merge_stage_bracket_results(results: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str], dict[str, Counter[str]], dict[str, Counter[str]]]:
    champion_counts: Counter[str] = Counter()
    finalist_counts: Counter[str] = Counter()
    stage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        champion_counts.update(dict(result["champions"]))
        finalist_counts.update(dict(result["finalists"]))
        for stage, counts in result["stages"].items():
            stage_counts[str(stage)].update(dict(counts))
        for stage, counts in result["pairs"].items():
            pair_counts[str(stage)].update(dict(counts))
    return champion_counts, finalist_counts, stage_counts, pair_counts


def compare_stage_bracket(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if previous is None:
        return {"baseline": True}
    stage_comparisons = {}
    pair_comparisons = {}
    max_stage_delta = 0.0
    max_stage_churn = 0
    max_pair_delta = 0.0
    max_pair_churn = 0
    for stage in BRACKET_STAGES:
        stage_cmp = compare_probability_rows(previous["stages"].get(stage, []), current["stages"].get(stage, []), top_limit=16)
        pair_cmp = compare_probability_rows(previous["pairs"].get(stage, []), current["pairs"].get(stage, []), top_limit=8)
        stage_comparisons[stage] = stage_cmp
        pair_comparisons[stage] = pair_cmp
        max_stage_delta = max(max_stage_delta, float(stage_cmp["max_abs_delta"]))
        max_stage_churn = max(max_stage_churn, int(stage_cmp["churn_count"]))
        max_pair_delta = max(max_pair_delta, float(pair_cmp["max_abs_delta"]))
        max_pair_churn = max(max_pair_churn, int(pair_cmp["churn_count"]))
    finalist_cmp = compare_probability_rows(previous["finalists"], current["finalists"], top_limit=16)
    return {
        "baseline": False,
        "previous_runs": int(previous["runs"]),
        "stage_top16": stage_comparisons,
        "pair_top8": pair_comparisons,
        "finalist_top16": finalist_cmp,
        "max_stage_top16_abs_delta": round(max_stage_delta, 6),
        "max_stage_top16_churn": int(max_stage_churn),
        "max_pair_top8_abs_delta": round(max_pair_delta, 6),
        "max_pair_top8_churn": int(max_pair_churn),
        "max_finalist_top16_abs_delta": finalist_cmp["max_abs_delta"],
        "max_finalist_top16_churn": finalist_cmp["churn_count"],
    }


def run_stage_bracket_size(
    executor: ThreadPoolExecutor | None,
    runs: int,
    seed: int,
    workers: int,
    chunk_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = time.perf_counter()
    chunks = build_chunks(runs, seed, workers, chunk_size)
    results: list[dict[str, Any]] = []
    completed = 0
    next_progress = 0.1
    if executor is None:
        result_iter = (run_stage_bracket_chunk(chunk_seed, chunk_count) for chunk_seed, chunk_count in chunks)
    else:
        futures = [executor.submit(run_stage_bracket_chunk, chunk_seed, chunk_count) for chunk_seed, chunk_count in chunks]
        result_iter = (future.result() for future in as_completed(futures))
    for result in result_iter:
        results.append(result)
        completed += sum(int(value) for value in result["champions"].values())
        progress = completed / float(runs)
        if progress >= next_progress or completed == runs:
            print(f"[mc-stability] bracket runs={runs} progresso={completed}/{runs} ({progress:.0%})", flush=True)
            next_progress += 0.1
    champion_counts, finalist_counts, stage_counts, pair_counts = merge_stage_bracket_results(results)
    elapsed = time.perf_counter() - start
    stage_rows = {
        stage: appearance_rows(stage_counts.get(stage, Counter()), runs, value_name="appearances", limit=64)
        for stage in BRACKET_STAGES
    }
    pair_rows = {
        stage: appearance_rows(pair_counts.get(stage, Counter()), runs, value_name="matches", limit=64)
        for stage in BRACKET_STAGES
    }
    finalist_rows = appearance_rows(finalist_counts, runs, value_name="finals", limit=16)
    summary = {
        "runs": int(runs),
        "seed": int(seed),
        "workers": int(workers),
        "chunks": int(len(chunks)),
        "chunk_size": int(chunks[0][1] if chunks else 0),
        "elapsed_seconds": round(elapsed, 3),
        "rate_per_second": round(float(runs) / elapsed, 3) if elapsed > 0 else 0.0,
        "champions": odds_from_counts(champion_counts, runs)[:16],
        "finalists": finalist_rows,
        "stages": stage_rows,
        "pairs": pair_rows,
    }
    csv_rows: list[dict[str, Any]] = []
    for stage, rows in stage_rows.items():
        for row in rows:
            csv_rows.append({"kind": "stage", "runs": runs, "stage": stage, **row})
    for stage, rows in pair_rows.items():
        for row in rows:
            csv_rows.append({"kind": "pair", "runs": runs, "stage": stage, **row})
    for row in finalist_rows:
        csv_rows.append({"kind": "finalist", "runs": runs, "stage": "Final", **row})
    return summary, csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo offline para estabilidade do ranking de campeoes.")
    parser.add_argument("--runs", type=parse_runs, default=parse_runs("5000,10000"), help="Volumes separados por virgula.")
    parser.add_argument("--stage-runs", type=parse_runs, default=parse_runs("1000,2000"), help="Volumes full-bracket separados por virgula para estabilidade de fases/chaves.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed base da auditoria de convergencia; volumes maiores estendem a mesma amostra.")
    parser.add_argument("--workers", type=int, default=8, help="Threads paralelas. Evita subprocessos presos e segue o executor usado pelo runtime do jogo.")
    parser.add_argument("--chunk-size", type=int, default=0, help="Tamanho fixo de chunk; 0 calcula automaticamente.")
    parser.add_argument("--stage-chunk-size", type=int, default=0, help="Tamanho fixo de chunk para estabilidade full-bracket; 0 calcula automaticamente.")
    parser.add_argument("--max-top16-delta", type=float, default=0.015, help="Falha se o delta maximo do top 16 entre volumes passar deste limite.")
    parser.add_argument("--max-top16-churn", type=int, default=2, help="Falha se times entrarem/sairem do top 16 alem deste limite.")
    parser.add_argument("--max-stage-top16-delta", type=float, default=0.035, help="Falha se o delta maximo por fase entre volumes passar deste limite.")
    parser.add_argument("--max-stage-top16-churn", type=int, default=4, help="Falha se o churn por fase top 16 passar deste limite.")
    parser.add_argument("--max-pair-top8-delta", type=float, default=0.02, help="Falha se o delta maximo de confrontos top 8 por fase passar deste limite.")
    parser.add_argument("--max-pair-top8-churn", type=int, default=16, help="Falha se o churn de confrontos top 8 por fase passar deste limite.")
    parser.add_argument("--allow-leader-change", action="store_true", help="Nao falha se o lider mudar entre os volumes.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    runs_list = list(args.runs)
    stage_runs_list = list(args.stage_runs)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []
    all_csv_rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    worker_count = max(1, int(args.workers))
    init_worker(str(MODEL_PATH))
    if worker_count == 1:
        executor: ThreadPoolExecutor | None = None
        for index, runs in enumerate(runs_list):
            sample_seed = int(args.seed)
            print(f"[mc-stability] iniciando {runs} Copas com {args.workers} thread | seed={sample_seed}", flush=True)
            summary, csv_rows = run_size(executor, runs, sample_seed, args.workers, args.chunk_size)
            summary["stability_vs_previous"] = compare_runs(previous, summary)
            all_summaries.append(summary)
            all_csv_rows.extend(csv_rows)
            previous = summary
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="arena-ai-mc-stability") as executor:
            for index, runs in enumerate(runs_list):
                sample_seed = int(args.seed)
                print(f"[mc-stability] iniciando {runs} Copas com {args.workers} threads | seed={sample_seed}", flush=True)
                summary, csv_rows = run_size(executor, runs, sample_seed, args.workers, args.chunk_size)
                summary["stability_vs_previous"] = compare_runs(previous, summary)
                all_summaries.append(summary)
                all_csv_rows.extend(csv_rows)
                previous = summary

    stage_summaries: list[dict[str, Any]] = []
    stage_csv_rows: list[dict[str, Any]] = []
    previous_stage: dict[str, Any] | None = None
    if worker_count == 1:
        executor = None
        for index, runs in enumerate(stage_runs_list):
            sample_seed = int(args.seed) + 50_000_000
            print(f"[mc-stability] iniciando estabilidade de fases/chaves {runs} Copas com {args.workers} thread | seed={sample_seed}", flush=True)
            summary, csv_rows = run_stage_bracket_size(executor, runs, sample_seed, args.workers, args.stage_chunk_size)
            summary["stability_vs_previous"] = compare_stage_bracket(previous_stage, summary)
            stage_summaries.append(summary)
            stage_csv_rows.extend(csv_rows)
            previous_stage = summary
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="arena-ai-mc-stage") as executor:
            for index, runs in enumerate(stage_runs_list):
                sample_seed = int(args.seed) + 50_000_000
                print(f"[mc-stability] iniciando estabilidade de fases/chaves {runs} Copas com {args.workers} threads | seed={sample_seed}", flush=True)
                summary, csv_rows = run_stage_bracket_size(executor, runs, sample_seed, args.workers, args.stage_chunk_size)
                summary["stability_vs_previous"] = compare_stage_bracket(previous_stage, summary)
                stage_summaries.append(summary)
                stage_csv_rows.extend(csv_rows)
                previous_stage = summary

    pd.DataFrame(all_csv_rows).to_csv(REPORT_CSV, index=False)
    pd.DataFrame(stage_csv_rows).to_csv(REPORT_STAGE_CSV, index=False)
    output = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": "Offline convergence audit. Larger volumes reuse the same base seed and extend the smaller sample; champion ranking uses optimized champion-only path, phase/bracket stability uses full knockout simulation.",
        "source_fingerprints": source_fingerprints(),
        "stability_gate": {
            "max_top16_abs_delta": float(args.max_top16_delta),
            "max_top16_churn": int(args.max_top16_churn),
            "leader_change_allowed": bool(args.allow_leader_change),
            "max_stage_top16_abs_delta": float(args.max_stage_top16_delta),
            "max_stage_top16_churn": int(args.max_stage_top16_churn),
            "max_pair_top8_abs_delta": float(args.max_pair_top8_delta),
            "max_pair_top8_churn": int(args.max_pair_top8_churn),
        },
        "runs": all_summaries,
        "stage_bracket_runs": stage_summaries,
        "summary": {
            "max_runs": max(runs_list),
            "min_runs": min(runs_list),
            "max_stage_bracket_runs": max(stage_runs_list),
            "min_stage_bracket_runs": min(stage_runs_list),
            "leader_at_max_runs": all_summaries[-1]["leader"] if all_summaries else "-",
            "leader_probability_at_max_runs": all_summaries[-1]["leader_probability"] if all_summaries else 0.0,
            "csv_path": str(REPORT_CSV),
            "stage_bracket_csv_path": str(REPORT_STAGE_CSV),
        },
    }
    final_comparison = all_summaries[-1].get("stability_vs_previous", {}) if len(all_summaries) >= 2 else {}
    final_stage_comparison = stage_summaries[-1].get("stability_vs_previous", {}) if len(stage_summaries) >= 2 else {}
    passed = True
    if final_comparison and not bool(final_comparison.get("baseline", False)):
        if not args.allow_leader_change and bool(final_comparison.get("leader_changed", False)):
            passed = False
            raise AssertionError(f"Monte Carlo stability failed: leader changed between volumes: {final_comparison}")
        if float(final_comparison.get("max_top16_abs_delta", 0.0)) > float(args.max_top16_delta):
            passed = False
            raise AssertionError(f"Monte Carlo stability failed: top16 delta too high: {final_comparison}")
        if int(final_comparison.get("top16_churn_count", 0)) > int(args.max_top16_churn):
            passed = False
            raise AssertionError(f"Monte Carlo stability failed: top16 churn too high: {final_comparison}")
    stage_bracket_passed = True
    if final_stage_comparison and not bool(final_stage_comparison.get("baseline", False)):
        if float(final_stage_comparison.get("max_stage_top16_abs_delta", 1.0)) > float(args.max_stage_top16_delta):
            stage_bracket_passed = False
            raise AssertionError(f"Monte Carlo stage stability failed: stage delta too high: {final_stage_comparison}")
        if int(final_stage_comparison.get("max_stage_top16_churn", 99)) > int(args.max_stage_top16_churn):
            stage_bracket_passed = False
            raise AssertionError(f"Monte Carlo stage stability failed: stage churn too high: {final_stage_comparison}")
        if float(final_stage_comparison.get("max_pair_top8_abs_delta", 1.0)) > float(args.max_pair_top8_delta):
            stage_bracket_passed = False
            raise AssertionError(f"Monte Carlo bracket stability failed: pair delta too high: {final_stage_comparison}")
        if int(final_stage_comparison.get("max_pair_top8_churn", 99)) > int(args.max_pair_top8_churn):
            stage_bracket_passed = False
            raise AssertionError(f"Monte Carlo bracket stability failed: pair churn too high: {final_stage_comparison}")
    output["passed"] = passed
    output["stage_bracket_passed"] = stage_bracket_passed
    output["final_comparison"] = final_comparison
    output["stage_bracket_final_comparison"] = final_stage_comparison
    REPORT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[mc-stability] wrote {REPORT_JSON}")
    print(f"[mc-stability] wrote {REPORT_CSV}")
    print(f"[mc-stability] wrote {REPORT_STAGE_CSV}")


if __name__ == "__main__":
    main()

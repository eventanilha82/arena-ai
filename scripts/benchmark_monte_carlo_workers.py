from __future__ import annotations

import argparse
import os
import sys
import time

from arena_ai.worldcup_model import WorldCupModel, effective_monte_carlo_workers


def parse_workers(value: str) -> list[int]:
    workers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        workers.append(max(1, int(part)))
    if not workers:
        raise argparse.ArgumentTypeError("informe pelo menos um worker")
    return workers


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de workers Monte Carlo do Arena AI.")
    parser.add_argument("--runs", type=int, default=1000, help="Quantidade de Copas por teste.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed base da amostra.")
    parser.add_argument("--warmup-runs", type=int, default=96, help="Copas de aquecimento para carregar modelo/cache.")
    parser.add_argument("--warmup-workers", type=int, default=8, help="Workers usados no aquecimento.")
    parser.add_argument("--workers", type=parse_workers, default=parse_workers("1,2,4,8"), help="Lista separada por virgula.")
    parser.add_argument("--mode", choices=("fresh", "bootstrap"), default="fresh", help="fresh mede Monte Carlo real; bootstrap mede banco de cenarios do jogo.")
    parser.add_argument("--allow-over-cap", action="store_true", help="Permite stress test acima do limite de workers do runtime.")
    args = parser.parse_args()

    model = WorldCupModel()
    if args.allow_over_cap:
        sota = sys.modules["sota_pipeline"]
        sota.MAX_MONTE_CARLO_WORKERS = max(args.workers)
    use_scenario_bank = args.mode == "bootstrap"

    print(f"cpu_count={os.cpu_count()} mode={args.mode} runtime_worker_cap={effective_monte_carlo_workers(999)}")
    if args.warmup_runs > 0:
        print(f"warmup runs={args.warmup_runs} workers={args.warmup_workers}")
        model.champion_odds_with_representative(
            runs=args.warmup_runs,
            seed=args.seed + 999,
            workers=args.warmup_workers,
            progress_with_odds=False,
            use_scenario_bank=use_scenario_bank,
        )

    results: list[tuple[int, float, str, float]] = []
    for workers in args.workers:
        start = time.perf_counter()
        odds, _representative = model.champion_odds_with_representative(
            runs=args.runs,
            seed=args.seed,
            workers=workers,
            progress_with_odds=False,
            use_scenario_bank=use_scenario_bank,
        )
        elapsed = time.perf_counter() - start
        effective_workers = effective_monte_carlo_workers(workers)
        leader = odds[0][0] if odds else "-"
        leader_odds = float(odds[0][2]) if odds else 0.0
        results.append((workers, elapsed, leader, leader_odds))
        print(
            f"workers={workers:2d} effective={effective_workers:2d} runs={args.runs} "
            f"elapsed={elapsed:7.3f}s rate={args.runs / elapsed:6.1f}/s "
            f"leader={leader} {leader_odds:.1%}"
        )

    best = min(results, key=lambda item: item[1])
    print(f"best workers={best[0]} elapsed={best[1]:.3f}s")


if __name__ == "__main__":
    main()

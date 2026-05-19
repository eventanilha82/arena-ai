from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena_ai.worldcup_model import MODEL_PATH, RUNTIME_PREDICTION_CACHE_PATH, SOTA_PIPELINE_PATH, WorldCupModel, file_sha256  # noqa: E402


def load_existing_payload() -> dict[str, Any] | None:
    if not RUNTIME_PREDICTION_CACHE_PATH.exists():
        return None
    try:
        with RUNTIME_PREDICTION_CACHE_PATH.open("rb") as file:
            payload = pickle.load(file)
    except Exception as exc:
        print(f"[runtime-cache] existing cache ignored: unreadable payload ({exc})")
        return None
    if not isinstance(payload, dict):
        print(f"[runtime-cache] existing cache ignored: payload is {type(payload)!r}")
        return None
    return payload


def cache_status(payload: dict[str, Any] | None, runs: int, seed: int, workers: int, model_sha256: str, pipeline_sha256: str) -> tuple[bool, str]:
    if payload is None:
        return False, "missing"
    if payload.get("model_sha256") != model_sha256:
        return False, "model hash changed"
    if payload.get("sota_pipeline_sha256") != pipeline_sha256:
        return False, "pipeline hash changed"
    if int(payload.get("runs", 0) or 0) != runs:
        return False, f"runs changed ({payload.get('runs')!r} != {runs})"
    if int(payload.get("seed", 0) or 0) != seed:
        return False, f"seed changed ({payload.get('seed')!r} != {seed})"
    if int(payload.get("workers", 0) or 0) != workers:
        return False, f"workers changed ({payload.get('workers')!r} != {workers})"

    prediction_cache = payload.get("prediction_cache")
    prediction_base_cache = payload.get("prediction_base_cache")
    scenario_bank = payload.get("scenario_bank")
    if not isinstance(prediction_cache, dict) or not prediction_cache:
        return False, "prediction_cache missing"
    if not isinstance(prediction_base_cache, dict) or not prediction_base_cache:
        return False, "prediction_base_cache missing"
    if not isinstance(scenario_bank, list) or len(scenario_bank) < runs:
        size = len(scenario_bank) if isinstance(scenario_bank, list) else "missing"
        return False, f"scenario_bank incomplete ({size} < {runs})"

    return (
        True,
        f"runs={runs} seed={seed} workers={workers} "
        f"prediction_cache={len(prediction_cache)} "
        f"base_cache={len(prediction_base_cache)} "
        f"scenario_bank={len(scenario_bank)}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera cache persistente de predições para o Monte Carlo runtime.")
    parser.add_argument("--runs", type=int, default=1000, help="Copas usadas para aquecer os confrontos prováveis.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed base do aquecimento.")
    parser.add_argument("--workers", type=int, default=8, help="Workers usados no aquecimento.")
    parser.add_argument("--refresh-predictions", action="store_true", help="Ignora caches existentes e recalcula as predições base.")
    parser.add_argument("--check", action="store_true", help="Valida o cache existente sem regenerar.")
    args = parser.parse_args()

    if args.check and args.refresh_predictions:
        parser.error("--check não pode ser usado com --refresh-predictions")

    runs = max(1, args.runs)
    workers = max(1, args.workers)
    model_sha256 = file_sha256(MODEL_PATH)
    pipeline_sha256 = file_sha256(SOTA_PIPELINE_PATH)
    current, reason = cache_status(load_existing_payload(), runs, args.seed, workers, model_sha256, pipeline_sha256)
    if args.check:
        if not current:
            raise SystemExit(f"[runtime-cache] stale or missing: {reason}")
        print(f"[runtime-cache] current: {reason}")
        return
    if current and not args.refresh_predictions:
        print(f"[runtime-cache] current: {reason}; skipped")
        return
    if not args.refresh_predictions:
        print(f"[runtime-cache] rebuild needed: {reason}")

    model = WorldCupModel()
    if args.refresh_predictions:
        model.package["prediction_cache"] = {}
        model.package["prediction_base_cache"] = {}

    started = time.perf_counter()

    def progress(done: int, total: int, _odds: list[tuple[str, int, float]]) -> bool:
        if done == total or done % max(1, total // 10) == 0:
            elapsed = time.perf_counter() - started
            print(f"[runtime-cache] {done}/{total} Copas | {elapsed:.1f}s")
        return True

    sota = sys.modules["sota_pipeline"]
    representative_candidates: dict[str, list[object]] = {}
    sota.monte_carlo(
        model.package,
        runs=runs,
        seed=args.seed,
        workers=workers,
        progress_callback=progress,
        representative_candidates=representative_candidates,
        fast_champion_only=True,
    )
    scenario_bank = [candidate for candidates in representative_candidates.values() for candidate in candidates]

    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_sha256": model_sha256,
        "sota_pipeline_sha256": pipeline_sha256,
        "runs": runs,
        "seed": args.seed,
        "workers": workers,
        "prediction_cache": model.package.get("prediction_cache", {}),
        "prediction_base_cache": model.package.get("prediction_base_cache", {}),
        "scenario_bank": scenario_bank,
    }
    RUNTIME_PREDICTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_PREDICTION_CACHE_PATH.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - started
    print(
        "[runtime-cache] saved "
        f"{RUNTIME_PREDICTION_CACHE_PATH} | "
        f"prediction_cache={len(payload['prediction_cache'])} | "
        f"base_cache={len(payload['prediction_base_cache'])} | "
        f"scenario_bank={len(scenario_bank)} | "
        f"{elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()

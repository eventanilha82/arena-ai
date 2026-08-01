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

from arena_ai.worldcup_model import (  # noqa: E402
    MODEL_PATH,
    RUNTIME_PREDICTION_CACHE_PATH,
    SOTA_PIPELINE_PATH,
    WorldCupModel,
    file_sha256,
    incomplete_neutral_cache_keys,
    mirrored_neutral_base_cache_key,
    mirrored_neutral_prediction_cache_key,
)


NEUTRAL_CACHE_TOLERANCE = 1e-10


def incomplete_cache_pairs(
    prediction_cache: dict[object, object],
    prediction_base_cache: dict[object, object],
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    return (
        incomplete_neutral_cache_keys(
            prediction_cache,
            mirrored_neutral_prediction_cache_key,
        ),
        incomplete_neutral_cache_keys(
            prediction_base_cache,
            mirrored_neutral_base_cache_key,
        ),
    )


def neutral_cache_max_deltas(
    prediction_cache: dict[object, object],
    prediction_base_cache: dict[object, object],
) -> tuple[float, float]:
    prediction_delta = 0.0
    for key, value in prediction_cache.items():
        mirror_key = mirrored_neutral_prediction_cache_key(key)
        if mirror_key is None or mirror_key not in prediction_cache:
            continue
        mirror = prediction_cache[mirror_key]
        if not isinstance(value, dict) or not isinstance(mirror, dict):
            return float("inf"), float("inf")
        prediction_delta = max(
            prediction_delta,
            abs(float(value["p_home_win_90"]) - float(mirror["p_away_win_90"])),
            abs(float(value["p_draw_90"]) - float(mirror["p_draw_90"])),
            abs(float(value["p_away_win_90"]) - float(mirror["p_home_win_90"])),
            abs(float(value["home_xg"]) - float(mirror["away_xg"])),
            abs(float(value["away_xg"]) - float(mirror["home_xg"])),
        )

    base_delta = 0.0
    for key, value in prediction_base_cache.items():
        mirror_key = mirrored_neutral_base_cache_key(key)
        if mirror_key is None or mirror_key not in prediction_base_cache:
            continue
        mirror = prediction_base_cache[mirror_key]
        if not isinstance(value, dict) or not isinstance(mirror, dict):
            return float("inf"), float("inf")
        probabilities = tuple(float(item) for item in value["probs_90"])
        mirror_probabilities = tuple(float(item) for item in mirror["probs_90"])
        if len(probabilities) != 3 or len(mirror_probabilities) != 3:
            return float("inf"), float("inf")
        base_delta = max(
            base_delta,
            abs(probabilities[0] - mirror_probabilities[2]),
            abs(probabilities[1] - mirror_probabilities[1]),
            abs(probabilities[2] - mirror_probabilities[0]),
        )
    return prediction_delta, base_delta


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

    incomplete_predictions, incomplete_bases = incomplete_cache_pairs(
        prediction_cache,
        prediction_base_cache,
    )
    if incomplete_predictions or incomplete_bases:
        return False, (
            "neutral cache pairs incomplete "
            f"(prediction={len(incomplete_predictions)}, base={len(incomplete_bases)})"
        )
    prediction_delta, base_delta = neutral_cache_max_deltas(
        prediction_cache,
        prediction_base_cache,
    )
    if prediction_delta > NEUTRAL_CACHE_TOLERANCE or base_delta > NEUTRAL_CACHE_TOLERANCE:
        return False, (
            "neutral cache values are asymmetric "
            f"(prediction={prediction_delta:.3e}, base={base_delta:.3e})"
        )

    return (
        True,
        f"runs={runs} seed={seed} workers={workers} "
        f"prediction_cache={len(prediction_cache)} "
        f"base_cache={len(prediction_base_cache)} "
        f"scenario_bank={len(scenario_bank)}",
    )


def complete_neutral_cache_pairs(model: WorldCupModel) -> None:
    sota = sys.modules["sota_pipeline"]
    prediction_cache = model.package.setdefault("prediction_cache", {})
    prediction_base_cache = model.package.setdefault("prediction_base_cache", {})
    if not isinstance(prediction_cache, dict) or not isinstance(prediction_base_cache, dict):
        raise TypeError("runtime prediction caches must be dictionaries")

    incomplete_predictions, incomplete_bases = incomplete_cache_pairs(
        prediction_cache,
        prediction_base_cache,
    )
    for key in incomplete_bases:
        home, away, neutral, knockout = key
        forward_key = (home, away, neutral, knockout, ())
        reverse_key = (away, home, neutral, knockout, ())
        prediction_cache.pop(forward_key, None)
        prediction_cache.pop(reverse_key, None)
        sota.predict_match(
            model.package,
            str(away),
            str(home),
            neutral=True,
            knockout=bool(knockout),
        )
        sota.predict_match(
            model.package,
            str(home),
            str(away),
            neutral=True,
            knockout=bool(knockout),
        )

    incomplete_predictions, incomplete_bases = incomplete_cache_pairs(
        prediction_cache,
        prediction_base_cache,
    )
    for key in incomplete_predictions:
        home, away, neutral, knockout, context_key = key
        mirror_key = mirrored_neutral_prediction_cache_key(key)
        if mirror_key is None:
            raise RuntimeError(f"invalid neutral prediction cache key: {key!r}")
        prediction_cache.pop(key, None)
        prediction_cache.pop(mirror_key, None)
        sota.predict_match(
            model.package,
            str(home),
            str(away),
            neutral=True,
            knockout=bool(knockout),
            context=dict(context_key),
        )
        sota.predict_match(
            model.package,
            str(away),
            str(home),
            neutral=True,
            knockout=bool(knockout),
            context=dict(mirror_key[4]),
        )

    incomplete_predictions, incomplete_bases = incomplete_cache_pairs(
        prediction_cache,
        prediction_base_cache,
    )
    if incomplete_predictions or incomplete_bases:
        raise RuntimeError(
            "failed to complete neutral runtime cache pairs: "
            f"prediction={len(incomplete_predictions)} base={len(incomplete_bases)}"
        )
    prediction_delta, base_delta = neutral_cache_max_deltas(
        prediction_cache,
        prediction_base_cache,
    )
    if prediction_delta > NEUTRAL_CACHE_TOLERANCE or base_delta > NEUTRAL_CACHE_TOLERANCE:
        raise RuntimeError(
            "failed to symmetrize neutral runtime cache values: "
            f"prediction={prediction_delta:.3e} base={base_delta:.3e}"
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

    model = WorldCupModel(preserve_incomplete_runtime_cache=True)
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
    complete_neutral_cache_pairs(model)
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

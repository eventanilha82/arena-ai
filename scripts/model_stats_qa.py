from __future__ import annotations

import json
import hashlib
import math
import pickle
import sys
import time
from itertools import permutations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


ROOT = Path(__file__).resolve().parents[1]
SOTA_SRC = ROOT / "modeling" / "worldcup_2026_ml" / "src"
sys.path.insert(0, str(SOTA_SRC))

import sota_pipeline as sota  # noqa: E402


MODEL_ROOT = ROOT / "modeling" / "worldcup_2026_ml"
MODEL_PATH = MODEL_ROOT / "models" / "model_sota.pkl"
MODEL_REPORT = MODEL_ROOT / "reports" / "sota_model_report.json"
TRAINING_PATH = MODEL_ROOT / "data" / "processed" / "sota_training_matches.csv"
SOTA_PIPELINE = SOTA_SRC / "sota_pipeline.py"
CHAMPION_ODDS = MODEL_ROOT / "reports" / "sota_champion_odds.csv"
REPORT_JSON = MODEL_ROOT / "reports" / "sota_statistical_report.json"
REPORT_MD = ROOT / "docs" / "STATISTICAL_AUDIT.md"
CALIBRATION_CSV = MODEL_ROOT / "reports" / "sota_calibration_bins.csv"
CLASS_CALIBRATION_CSV = MODEL_ROOT / "reports" / "sota_class_calibration_summary.csv"
BLOCK_BOOTSTRAP_CSV = MODEL_ROOT / "reports" / "sota_block_bootstrap_intervals.csv"
ABLATION_CSV = MODEL_ROOT / "reports" / "sota_ablation_study.csv"
UNCERTAINTY_CSV = MODEL_ROOT / "reports" / "sota_uncertainty_intervals.csv"
STAGE_UNCERTAINTY_CSV = MODEL_ROOT / "reports" / "sota_stage_uncertainty_intervals.csv"
DC_CSV = MODEL_ROOT / "reports" / "sota_dixon_coles_rho_sensitivity.csv"
FRONTIER_CSV = MODEL_ROOT / "reports" / "sota_internal_frontier_experiments.csv"
RUNTIME_ADJUSTMENT_CSV = MODEL_ROOT / "reports" / "sota_runtime_adjustment_audit.csv"
MC_STABILITY_JSON = MODEL_ROOT / "reports" / "sota_monte_carlo_stability.json"
MC_STABILITY_CSV = MODEL_ROOT / "reports" / "sota_monte_carlo_stability.csv"
MC_STAGE_BRACKET_CSV = MODEL_ROOT / "reports" / "sota_monte_carlo_stage_bracket_stability.csv"
MC_STABILITY_SCRIPT = ROOT / "scripts" / "monte_carlo_stability.py"
RAW_DATA_ROOT = MODEL_ROOT / "data" / "raw"
RAW_MANIFEST_JSON = MODEL_ROOT / "reports" / "sota_raw_data_manifest.json"
RAW_MANIFEST_CSV = MODEL_ROOT / "reports" / "sota_raw_data_manifest.csv"


LABELS = {0: "casa", 1: "empate", 2: "fora"}


RAW_CSV_CONTRACTS: dict[str, dict[str, Any]] = {
    "data/raw/teams.csv": {
        "min_rows": 48,
        "required_columns": ["id", "team_name", "fifa_code", "group_letter", "is_placeholder"],
    },
    "data/raw/matches.csv": {
        "min_rows": 104,
        "required_columns": ["id", "match_number", "home_team_id", "away_team_id", "city_id", "stage_id", "kickoff_at"],
        "date_columns": ["kickoff_at"],
    },
    "data/raw/host_cities.csv": {
        "min_rows": 16,
        "required_columns": ["id", "city_name", "country", "venue_name", "region_cluster", "airport_code"],
    },
    "data/raw/tournament_stages.csv": {
        "min_rows": 7,
        "required_columns": ["id", "stage_name", "stage_order"],
    },
    "data/raw/fc26_players.csv": {
        "min_rows": 18_000,
        "required_columns": ["player_id", "short_name", "nationality_name", "overall", "potential", "value_eur", "player_positions"],
        "date_columns": ["fifa_update_date", "dob"],
    },
    "data/raw/fifa_rankings_1992_2024.csv": {
        "min_rows": 67_000,
        "required_columns": ["rank", "country_full", "country_abrv", "total_points", "confederation", "rank_date"],
        "date_columns": ["rank_date"],
    },
    "data/raw/candidates/pataterie_all_matches.csv": {
        "min_rows": 50_000,
        "required_columns": ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "country", "neutral"],
        "date_columns": ["date"],
    },
    "data/raw/candidates/pataterie_countries_names.csv": {
        "min_rows": 250,
        "required_columns": ["original_name", "current_name", "color_code"],
    },
    "data/raw/candidates/saifalnimri_eloratings.csv": {
        "min_rows": 6_000,
        "required_columns": ["date", "team", "rating", "change"],
        "date_columns": ["date"],
    },
    "data/raw/candidates/transfermarkt_player_injuries.csv": {
        "min_rows": 140_000,
        "required_columns": ["player_id", "season_name", "injury_reason", "from_date", "end_date", "days_missed", "games_missed"],
        "date_columns": ["from_date", "end_date"],
    },
    "data/raw/candidates/transfermarkt_player_market_value.csv": {
        "min_rows": 900_000,
        "required_columns": ["player_id", "date_unix", "value"],
    },
    "data/raw/candidates/transfermarkt_player_national_performances.csv": {
        "min_rows": 90_000,
        "required_columns": ["player_id", "team_id", "matches", "goals", "career_state"],
    },
    "data/raw/candidates/transfermarkt_player_profiles.csv": {
        "min_rows": 90_000,
        "required_columns": ["player_id", "player_name", "date_of_birth", "citizenship", "position", "current_club_name"],
        "date_columns": ["date_of_birth"],
    },
    "data/raw/candidates/transfermarkt_team_details.csv": {
        "min_rows": 2_000,
        "required_columns": ["club_id", "club_name", "country_name", "season_id", "source_url"],
    },
}


def log_step(start: float, message: str) -> None:
    print(f"[stats-qa] {message} ({time.perf_counter() - start:.1f}s)")


def round_float(value: Any, digits: int = 6) -> float:
    return round(float(value), digits)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, np.ndarray)) else False:
        return None
    return value


def load_artifacts() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if not MODEL_PATH.exists():
        raise AssertionError(f"missing model package: {MODEL_PATH}")
    if not MODEL_REPORT.exists():
        raise AssertionError(f"missing model report: {MODEL_REPORT}")
    if not TRAINING_PATH.exists():
        raise AssertionError(f"missing training frame: {TRAINING_PATH}")
    with MODEL_PATH.open("rb") as file:
        package = pickle.load(file)
    report = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
    training = pd.read_csv(TRAINING_PATH)
    training["date"] = pd.to_datetime(training["date"])
    return package, report, training


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


def parse_date_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if float(parsed.notna().mean()) < 0.95:
        try:
            parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):
            pass
    return parsed


def semantic_date_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    parsed = parse_date_column(frame[column])
    valid_ratio = float(parsed.notna().mean())
    return {
        "valid_ratio": round_float(valid_ratio, 6),
        "min": str(parsed.min().date()) if parsed.notna().any() else None,
        "max": str(parsed.max().date()) if parsed.notna().any() else None,
    }


def semantic_raw_checks(relative: str, path: Path, columns: list[str], row_count: int) -> dict[str, Any]:
    contract = RAW_CSV_CONTRACTS.get(relative)
    if contract is None:
        return {"checked": False, "passed": True, "checks": []}
    checks: list[dict[str, Any]] = []
    passed = True

    def record(name: str, ok: bool, **details: Any) -> None:
        nonlocal passed
        passed = passed and bool(ok)
        checks.append({"name": name, "passed": bool(ok), **details})

    min_rows = int(contract.get("min_rows", 0))
    record("min_rows", row_count >= min_rows, actual=row_count, expected_min=min_rows)
    required_columns = list(contract.get("required_columns", []))
    missing_columns = [column for column in required_columns if column not in columns]
    record("required_columns", not missing_columns, missing=missing_columns)
    if missing_columns:
        return {"checked": True, "passed": False, "checks": checks}

    frame = pd.read_csv(path, low_memory=False)
    duplicate_count = int(frame.duplicated().sum())
    duplicate_ratio = duplicate_count / max(1, len(frame))
    record("duplicate_rows_ratio", duplicate_ratio <= 0.001, duplicate_count=duplicate_count, duplicate_ratio=round_float(duplicate_ratio, 6), max_ratio=0.001)
    for column in contract.get("date_columns", []):
        if column not in frame.columns:
            record(f"{column}_present", False)
            continue
        summary = semantic_date_summary(frame, str(column))
        record(f"{column}_parse_rate", float(summary["valid_ratio"]) >= 0.95, **summary)

    if relative == "data/raw/teams.csv":
        group_counts = frame["group_letter"].astype(str).value_counts().sort_index().to_dict()
        expected_groups = {letter: 4 for letter in list("ABCDEFGHIJKL")}
        placeholder_count = int(frame["is_placeholder"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
        record("world_cup_48_teams", len(frame) == 48, actual=len(frame))
        record("group_balance_a_l", group_counts == expected_groups, actual=group_counts, expected=expected_groups)
        record("no_placeholders", placeholder_count == 0, placeholders=placeholder_count)
        record("unique_fifa_codes", int(frame["fifa_code"].nunique()) == 48, actual=int(frame["fifa_code"].nunique()))
        record("unique_team_names", int(frame["team_name"].nunique()) == 48, actual=int(frame["team_name"].nunique()))
    elif relative == "data/raw/matches.csv":
        stage_counts = {int(key): int(value) for key, value in frame["stage_id"].value_counts().sort_index().to_dict().items()}
        record("world_cup_104_matches", len(frame) == 104, actual=len(frame))
        record("group_stage_72_matches", stage_counts.get(1, 0) == 72, actual=stage_counts.get(1, 0))
        record("stage_distribution", stage_counts == {1: 72, 2: 16, 3: 8, 4: 4, 5: 2, 6: 1, 7: 1}, actual=stage_counts)
    elif relative == "data/raw/fifa_rankings_1992_2024.csv":
        ranks = pd.to_numeric(frame["rank"], errors="coerce")
        dates = semantic_date_summary(frame, "rank_date")
        record("rank_range", bool((ranks.dropna() >= 1).all() and (ranks.dropna() <= 250).all()), min_rank=float(ranks.min()), max_rank=float(ranks.max()))
        record("ranking_current_enough", int(str(dates["max"])[:4]) >= 2024, max_date=dates["max"])
    elif relative == "data/raw/candidates/pataterie_all_matches.csv":
        home_score = pd.to_numeric(frame["home_score"], errors="coerce")
        away_score = pd.to_numeric(frame["away_score"], errors="coerce")
        record("scores_non_negative", bool((home_score.dropna() >= 0).all() and (away_score.dropna() >= 0).all()))
    elif relative == "data/raw/fc26_players.csv":
        overall = pd.to_numeric(frame["overall"], errors="coerce")
        record("overall_range", bool((overall.dropna() >= 1).all() and (overall.dropna() <= 99).all()), min_overall=float(overall.min()), max_overall=float(overall.max()))
    elif relative == "data/raw/candidates/saifalnimri_eloratings.csv":
        rating = pd.to_numeric(frame["rating"], errors="coerce")
        record("elo_rating_range", bool((rating.dropna() >= 0).all() and (rating.dropna() <= 3000).all()), min_rating=float(rating.min()), max_rating=float(rating.max()))

    return {"checked": True, "passed": bool(passed), "checks": checks}


def raw_cross_file_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    passed = True

    def record(name: str, ok: bool, **details: Any) -> None:
        nonlocal passed
        passed = passed and bool(ok)
        checks.append({"name": name, "passed": bool(ok), **details})

    try:
        teams = pd.read_csv(RAW_DATA_ROOT / "teams.csv")
        matches = pd.read_csv(RAW_DATA_ROOT / "matches.csv")
        cities = pd.read_csv(RAW_DATA_ROOT / "host_cities.csv")
        stages = pd.read_csv(RAW_DATA_ROOT / "tournament_stages.csv")
    except Exception as exc:
        return {"passed": False, "checks": [{"name": "load_cross_file_inputs", "passed": False, "error": str(exc)}]}

    team_ids = set(pd.to_numeric(teams["id"], errors="coerce").dropna().astype(int))
    city_ids = set(pd.to_numeric(cities["id"], errors="coerce").dropna().astype(int))
    stage_ids = set(pd.to_numeric(stages["id"], errors="coerce").dropna().astype(int))
    home_ids = set(pd.to_numeric(matches["home_team_id"], errors="coerce").dropna().astype(int))
    away_ids = set(pd.to_numeric(matches["away_team_id"], errors="coerce").dropna().astype(int))
    match_city_ids = set(pd.to_numeric(matches["city_id"], errors="coerce").dropna().astype(int))
    match_stage_ids = set(pd.to_numeric(matches["stage_id"], errors="coerce").dropna().astype(int))
    match_numbers = set(pd.to_numeric(matches["match_number"], errors="coerce").dropna().astype(int))

    record("match_team_ids_exist", home_ids.issubset(team_ids) and away_ids.issubset(team_ids), missing_home=sorted(home_ids - team_ids), missing_away=sorted(away_ids - team_ids))
    record("match_city_ids_exist", match_city_ids.issubset(city_ids), missing=sorted(match_city_ids - city_ids))
    record("match_stage_ids_exist", match_stage_ids.issubset(stage_ids), missing=sorted(match_stage_ids - stage_ids))
    record("match_numbers_unique_1_104", len(match_numbers) == 104 and match_numbers == set(range(1, 105)), actual_count=len(match_numbers))
    group_stage = matches[matches["stage_id"] == 1]
    record(
        "group_stage_has_team_ids",
        bool(group_stage["home_team_id"].notna().all() and group_stage["away_team_id"].notna().all()),
        missing_home=int(group_stage["home_team_id"].isna().sum()),
        missing_away=int(group_stage["away_team_id"].isna().sum()),
    )
    knockout = matches[matches["stage_id"] != 1]
    record(
        "knockout_slots_are_placeholders_or_known_ids",
        bool(knockout["home_team_id"].isna().sum() == len(knockout) and knockout["away_team_id"].isna().sum() == len(knockout)),
        missing_home=int(knockout["home_team_id"].isna().sum()),
        missing_away=int(knockout["away_team_id"].isna().sum()),
        knockout_matches=int(len(knockout)),
    )
    return {"passed": bool(passed), "checks": checks}


def source_fingerprints() -> dict[str, Any]:
    return {
        "model_package": file_fingerprint(MODEL_PATH),
        "model_report": file_fingerprint(MODEL_REPORT),
        "training_matches": file_fingerprint(TRAINING_PATH),
        "sota_pipeline": file_fingerprint(SOTA_PIPELINE),
        "stats_qa_script": file_fingerprint(Path(__file__).resolve()),
    }


def raw_data_manifest() -> dict[str, Any]:
    if not RAW_DATA_ROOT.exists():
        raise AssertionError(f"missing raw data root: {RAW_DATA_ROOT}")
    rows: list[dict[str, Any]] = []
    for path in sorted(RAW_DATA_ROOT.rglob("*")):
        if not path.is_file():
            continue
        fingerprint = file_fingerprint(path)
        relative = path.relative_to(MODEL_ROOT).as_posix()
        row: dict[str, Any] = {
            "path": relative,
            "sha256": fingerprint["sha256"],
            "size_bytes": fingerprint["size_bytes"],
            "mtime_ns": fingerprint["mtime_ns"],
            "suffix": path.suffix.lower(),
        }
        if path.suffix.lower() == ".csv":
            try:
                header = pd.read_csv(path, nrows=0)
                row["columns"] = list(header.columns)
                with path.open("rb") as file:
                    line_count = sum(1 for _line in file)
                row["row_count"] = max(0, line_count - 1)
                semantic = semantic_raw_checks(relative, path, list(header.columns), int(row["row_count"]))
                row["semantic_checked"] = bool(semantic["checked"])
                row["semantic_passed"] = bool(semantic["passed"])
                row["semantic_summary"] = json.dumps(jsonable(semantic), ensure_ascii=False, sort_keys=True)
            except Exception as exc:  # pragma: no cover - reported in manifest instead of hiding the file.
                row["csv_error"] = str(exc)
                row["semantic_checked"] = bool(relative in RAW_CSV_CONTRACTS)
                row["semantic_passed"] = False
                row["semantic_summary"] = json.dumps({"checked": relative in RAW_CSV_CONTRACTS, "passed": False, "error": str(exc)}, ensure_ascii=False)
        rows.append(row)
    required_paths = set(RAW_CSV_CONTRACTS)
    present_paths = {str(row["path"]) for row in rows}
    missing_required = sorted(required_paths - present_paths)
    cross_file_checks = raw_cross_file_checks()
    semantic_failures = [
        {
            "path": row["path"],
            "summary": json.loads(str(row.get("semantic_summary", "{}") or "{}")),
        }
        for row in rows
        if row.get("semantic_checked") and not bool(row.get("semantic_passed"))
    ]
    if missing_required:
        semantic_failures.append({"path": "<missing>", "summary": {"missing_required": missing_required}})
    if not bool(cross_file_checks["passed"]):
        semantic_failures.append({"path": "<cross-file>", "summary": cross_file_checks})
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(str(row["sha256"]).encode("utf-8"))
        digest.update(str(row["size_bytes"]).encode("utf-8"))
    manifest = {
        "root": str(RAW_DATA_ROOT),
        "file_count": int(len(rows)),
        "total_size_bytes": int(sum(int(row["size_bytes"]) for row in rows)),
        "csv_file_count": int(sum(1 for row in rows if row["suffix"] == ".csv")),
        "manifest_sha256": digest.hexdigest(),
        "semantic": {
            "required_file_count": int(len(required_paths)),
            "required_files_present": not missing_required,
            "checked_file_count": int(sum(1 for row in rows if row.get("semantic_checked"))),
            "passed_file_count": int(sum(1 for row in rows if row.get("semantic_checked") and row.get("semantic_passed"))),
            "cross_file_checks": cross_file_checks,
            "passed": not semantic_failures,
            "failures": semantic_failures,
        },
        "files": rows,
    }
    RAW_MANIFEST_JSON.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(RAW_MANIFEST_CSV, index=False)
    return manifest


def fingerprint_is_fresh(reported: dict[str, Any], current: dict[str, Any]) -> bool:
    return str(reported.get("sha256", "")) == str(current.get("sha256", ""))


def metric_block(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 0.001, None)
    y = y.astype(int)
    pred = probs.argmax(axis=1)
    draw_mask = y == 1
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1).mean() / math.log(3)
    return {
        "rows": int(len(y)),
        "accuracy": round_float(accuracy_score(y, pred), 6),
        "log_loss": round_float(log_loss(y, probs, labels=[0, 1, 2]), 6),
        "rps": round_float(sota.rps_1x2(y, probs), 6),
        "brier": round_float(sota.brier_multiclass(y, probs), 6),
        "ece": round_float(sota.ece_multiclass(y, probs), 6),
        "draw_recall": round_float(np.mean(pred[draw_mask] == 1), 6) if draw_mask.any() else 0.0,
        "draw_expected_rate": round_float(probs[:, 1].mean(), 6),
        "draw_actual_rate": round_float(np.mean(y == 1), 6),
        "draw_gap": round_float(abs(probs[:, 1].mean() - np.mean(y == 1)), 6),
        "entropy": round_float(entropy, 6),
    }


def objective_from_metrics(metrics: dict[str, float]) -> float:
    return round_float(
        float(metrics["log_loss"])
        + 0.30 * float(metrics["rps"])
        + 0.08 * float(metrics["ece"])
        + 0.05 * float(metrics["brier"])
        + 0.42 * float(metrics["draw_gap"])
        + 0.05 * max(0.0, 0.68 - float(metrics["entropy"])),
        6,
    )


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-7, 1.0)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-7, None)


def confidence_calibration_bins(y: np.ndarray, probs: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index + 1 < bins else confidence <= high)
        if not mask.any():
            continue
        rows.append(
            {
                "section": "confianca",
                "label": "argmax",
                "bin_low": round_float(low, 3),
                "bin_high": round_float(high, 3),
                "count": int(mask.sum()),
                "avg_pred": round_float(confidence[mask].mean(), 6),
                "empirical_rate": round_float(correct[mask].mean(), 6),
                "abs_gap": round_float(abs(confidence[mask].mean() - correct[mask].mean()), 6),
            }
        )
    for class_index, label in LABELS.items():
        class_probs = probs[:, class_index]
        actual = (y == class_index).astype(float)
        for index in range(bins):
            low = index / bins
            high = (index + 1) / bins
            mask = (class_probs >= low) & (class_probs < high if index + 1 < bins else class_probs <= high)
            if not mask.any():
                continue
            rows.append(
                {
                    "section": "classe",
                    "label": label,
                    "bin_low": round_float(low, 3),
                    "bin_high": round_float(high, 3),
                    "count": int(mask.sum()),
                    "avg_pred": round_float(class_probs[mask].mean(), 6),
                    "empirical_rate": round_float(actual[mask].mean(), 6),
                    "abs_gap": round_float(abs(class_probs[mask].mean() - actual[mask].mean()), 6),
                }
            )
    frame = pd.DataFrame(rows)
    weighted_gap = float((frame["abs_gap"] * frame["count"]).sum() / max(1, frame["count"].sum())) if len(frame) else 0.0
    summary = {
        "bin_count": int(len(frame)),
        "max_abs_gap": round_float(frame["abs_gap"].max() if len(frame) else 0.0, 6),
        "weighted_abs_gap": round_float(weighted_gap, 6),
    }
    return frame, summary


def class_calibration_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    class_rows = frame[frame["section"] == "classe"].copy()
    rows: list[dict[str, Any]] = []
    for label in LABELS.values():
        section = class_rows[class_rows["label"] == label]
        if section.empty:
            continue
        total = max(1, int(section["count"].sum()))
        weighted_abs_gap = float((section["abs_gap"] * section["count"]).sum() / total)
        rows.append(
            {
                "label": label,
                "bin_count": int(len(section)),
                "sample_count": int(total),
                "weighted_abs_gap": round_float(weighted_abs_gap, 6),
                "max_abs_gap": round_float(section["abs_gap"].max(), 6),
                "mean_predicted_rate": round_float(float((section["avg_pred"] * section["count"]).sum() / total), 6),
                "mean_empirical_rate": round_float(float((section["empirical_rate"] * section["count"]).sum() / total), 6),
            }
        )
    summary_frame = pd.DataFrame(rows).sort_values("weighted_abs_gap", ascending=False)
    summary = {
        "path": str(CLASS_CALIBRATION_CSV),
        "worst_weighted_abs_gap": round_float(summary_frame["weighted_abs_gap"].max() if len(summary_frame) else 0.0, 6),
        "worst_max_abs_gap": round_float(summary_frame["max_abs_gap"].max() if len(summary_frame) else 0.0, 6),
        "rows": int(len(summary_frame)),
    }
    return summary_frame, summary


def bootstrap_metric_intervals(y: np.ndarray, probs: np.ndarray, samples: int = 400) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(2026)
    values: dict[str, list[float]] = {"log_loss": [], "rps": [], "brier": [], "ece": []}
    row_count = len(y)
    for _ in range(samples):
        indexes = rng.integers(0, row_count, row_count)
        block = metric_block(y[indexes], probs[indexes])
        for name in values:
            values[name].append(float(block[name]))
    intervals: dict[str, dict[str, float]] = {}
    for name, metric_values in values.items():
        arr = np.asarray(metric_values, dtype=float)
        intervals[name] = {
            "mean": round_float(arr.mean(), 6),
            "lower_95": round_float(np.quantile(arr, 0.025), 6),
            "upper_95": round_float(np.quantile(arr, 0.975), 6),
        }
    return intervals


def block_bootstrap_metric_intervals(
    diagnostic: pd.DataFrame,
    y: np.ndarray,
    probs: np.ndarray,
    samples: int = 400,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(2027)
    values: list[dict[str, Any]] = []

    def add_interval(block_type: str, block_keys: np.ndarray) -> None:
        unique_keys = [key for key in pd.unique(block_keys) if str(key)]
        blocks = [np.flatnonzero(block_keys == key) for key in unique_keys]
        blocks = [block for block in blocks if len(block)]
        if len(blocks) < 2:
            return
        metric_values: dict[str, list[float]] = {"log_loss": [], "rps": [], "brier": [], "ece": [], "draw_gap": []}
        for _ in range(samples):
            picked = rng.integers(0, len(blocks), len(blocks))
            indexes = np.concatenate([blocks[index] for index in picked])
            block_metrics = metric_block(y[indexes], probs[indexes])
            for name in metric_values:
                metric_values[name].append(float(block_metrics[name]))
        for metric, metric_samples in metric_values.items():
            arr = np.asarray(metric_samples, dtype=float)
            values.append(
                {
                    "block_type": block_type,
                    "metric": metric,
                    "block_count": int(len(blocks)),
                    "samples": int(samples),
                    "mean": round_float(arr.mean(), 6),
                    "lower_95": round_float(np.quantile(arr, 0.025), 6),
                    "upper_95": round_float(np.quantile(arr, 0.975), 6),
                    "width_95": round_float(np.quantile(arr, 0.975) - np.quantile(arr, 0.025), 6),
                }
            )

    add_interval("ano", pd.to_datetime(diagnostic["date"]).dt.year.astype(str).to_numpy())
    tournament = diagnostic["tournament"].fillna("unknown").astype(str).to_numpy() if "tournament" in diagnostic.columns else np.array([])
    if len(tournament):
        add_interval("torneio", tournament)
    frame = pd.DataFrame(values)
    summary = {
        "path": str(BLOCK_BOOTSTRAP_CSV),
        "samples": int(samples),
        "block_types": sorted(frame["block_type"].unique().tolist()) if len(frame) else [],
        "max_width_95": round_float(frame["width_95"].max() if len(frame) else 0.0, 6),
        "rows": int(len(frame)),
    }
    return frame, summary


def component_arrays(package: dict[str, Any], frame: pd.DataFrame, rho: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    x = frame[sota.BASE_FEATURES]
    models = package["models"]
    home_lam = np.clip(models["home_goals_poisson"].predict(x), 0.05, 5.5)
    away_lam = np.clip(models["away_goals_poisson"].predict(x), 0.05, 5.5)
    count_home_lam = np.clip(models["home_goals_xgb_count"].predict(x), 0.05, 5.5)
    count_away_lam = np.clip(models["away_goals_xgb_count"].predict(x), 0.05, 5.5)
    poisson_probs = sota.lambdas_to_prob_array(home_lam, away_lam, rho=rho)
    components = {
        "xgb": models["xgb_1x2"].predict_proba(x),
        "competitive": models["competitive_xgb_1x2"].predict_proba(x),
        "logistic": models["logistic_1x2"].predict_proba(x),
        "elo": sota.elo_policy_probs_from_frame(frame),
        "poisson": poisson_probs,
        "count_poisson": sota.lambdas_to_prob_array(count_home_lam, count_away_lam, rho=rho),
    }
    return components, poisson_probs


def blend_components(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(float(weight) for name, weight in weights.items() if name in components and weight > 0)
    if total <= 0:
        raise AssertionError("empty component weight set")
    blend = None
    for name, weight in weights.items():
        if name not in components or weight <= 0:
            continue
        part = components[name] * (float(weight) / total)
        blend = part if blend is None else blend + part
    if blend is None:
        raise AssertionError("component blend failed")
    return blend / np.clip(blend.sum(axis=1, keepdims=True), 0.001, None)


def apply_policy(y: np.ndarray, pre_draw: np.ndarray, poisson: np.ndarray, policy: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    candidate = sota.score_simulation_policy_candidate(
        y,
        pre_draw,
        poisson,
        float(policy["classifier_weight"]),
        float(policy["draw_floor"]),
        float(policy["draw_ceiling"]),
    )
    classifier = sota.policy_classifier_probs(pre_draw, float(policy["draw_floor"]), float(policy["draw_ceiling"]))
    probs = float(policy["classifier_weight"]) * classifier + (1.0 - float(policy["classifier_weight"])) * poisson
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 0.001, None)
    return probs, candidate


def ablation_study(
    y: np.ndarray,
    components: dict[str, np.ndarray],
    poisson_probs: np.ndarray,
    policy: dict[str, Any],
    runtime_weights: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(name: str, pre_draw: np.ndarray, classifier_weight: float | None = None) -> None:
        local_policy = dict(policy)
        if classifier_weight is not None:
            local_policy["classifier_weight"] = float(classifier_weight)
            local_policy["poisson_weight"] = float(1.0 - classifier_weight)
        _probs, metrics = apply_policy(y, pre_draw, poisson_probs, local_policy)
        rows.append({"ablation": name, **metrics})

    full_pre = blend_components(components, runtime_weights)
    add_row("runtime_policy", full_pre)
    for candidate in sota.manual_blend_candidate_grid():
        weights = dict(candidate["manual_blend_weights"])
        add_row(str(candidate["name"]), blend_components(components, weights))
    add_row("classifier_only_no_final_poisson", full_pre, classifier_weight=1.0)
    add_row("score_poisson_only", full_pre, classifier_weight=0.0)
    for name in ("xgb", "competitive", "logistic", "elo", "poisson", "count_poisson"):
        add_row(f"{name}_only_stack", components[name])

    frame = pd.DataFrame(rows).sort_values(["objective", "log_loss"], ascending=[True, True])
    return frame


def draw_policy_scan(y: np.ndarray, pre_draw: np.ndarray, poisson: np.ndarray, policy: dict[str, Any]) -> dict[str, Any]:
    _best, candidates = sota.select_simulation_policy_from_arrays(y, pre_draw, poisson)
    current_key = (
        round_float(policy["classifier_weight"], 4),
        round_float(policy["draw_floor"], 4),
        round_float(policy["draw_ceiling"], 4),
    )
    rank = next(
        (
            index + 1
            for index, item in enumerate(candidates)
            if (
                round_float(item["classifier_weight"], 4),
                round_float(item["draw_floor"], 4),
                round_float(item["draw_ceiling"], 4),
            )
            == current_key
        ),
        None,
    )
    return {
        "candidate_count": int(len(candidates)),
        "current_policy_rank": int(rank or -1),
        "current_policy": current_key,
        "top_10": candidates[:10],
        "best_by_classifier_weight": [
            min(
                [candidate for candidate in candidates if candidate["classifier_weight"] == weight],
                key=lambda item: item["objective"],
            )
            for weight in sorted({candidate["classifier_weight"] for candidate in candidates})
        ],
    }


def dixon_coles_sensitivity(
    y: np.ndarray,
    pre_draw: np.ndarray,
    frame: pd.DataFrame,
    package: dict[str, Any],
    policy: dict[str, Any],
) -> pd.DataFrame:
    x = frame[sota.BASE_FEATURES]
    models = package["models"]
    home_lam = np.clip(models["home_goals_poisson"].predict(x), 0.05, 5.5)
    away_lam = np.clip(models["away_goals_poisson"].predict(x), 0.05, 5.5)
    rows: list[dict[str, Any]] = []
    for rho in np.linspace(-0.18, 0.08, 14):
        poisson = sota.lambdas_to_prob_array(home_lam, away_lam, rho=float(rho))
        _probs, metrics = apply_policy(y, pre_draw, poisson, policy)
        pure = metric_block(y, poisson)
        rows.append(
            {
                "rho": round_float(rho, 4),
                "hybrid_objective": metrics["objective"],
                "hybrid_log_loss": metrics["log_loss"],
                "hybrid_rps": metrics["rps"],
                "hybrid_draw_gap": metrics["draw_gap"],
                "poisson_only_log_loss": pure["log_loss"],
                "poisson_only_rps": pure["rps"],
                "poisson_only_draw_gap": pure["draw_gap"],
            }
        )
    return pd.DataFrame(rows).sort_values(["hybrid_objective", "hybrid_log_loss"], ascending=[True, True])


def champion_uncertainty(report: dict[str, Any]) -> pd.DataFrame:
    if not CHAMPION_ODDS.exists():
        raise AssertionError(f"missing champion odds: {CHAMPION_ODDS}")
    odds = pd.read_csv(CHAMPION_ODDS)
    runs = int(max(1, odds["wins"].sum() if "wins" in odds else report.get("monte_carlo_runs", 1000)))
    rows: list[dict[str, Any]] = []
    for row in odds.head(16).itertuples(index=False):
        p = float(row.champion_probability)
        margin = 1.96 * math.sqrt(max(0.0, p * (1.0 - p)) / runs)
        rows.append(
            {
                "team": row.team,
                "wins": int(row.wins),
                "runs": runs,
                "probability": round_float(p, 6),
                "lower_95": round_float(max(0.0, p - margin), 6),
                "upper_95": round_float(min(1.0, p + margin), 6),
                "margin_95": round_float(margin, 6),
            }
        )
    return pd.DataFrame(rows)


def stage_uncertainty(report: dict[str, Any]) -> pd.DataFrame:
    stage_path = MODEL_ROOT / "reports" / "sota_stage_odds.csv"
    if not stage_path.exists():
        raise AssertionError(f"missing stage odds: {stage_path}")
    stage = pd.read_csv(stage_path)
    runs = int(max(1, report.get("monte_carlo_runs", 1000)))
    rows: list[dict[str, Any]] = []
    for row in stage.itertuples(index=False):
        p = float(row.probability)
        margin = 1.96 * math.sqrt(max(0.0, p * (1.0 - p)) / runs)
        rows.append(
            {
                "team": row.team,
                "stage": row.stage,
                "runs": runs,
                "probability": round_float(p, 6),
                "lower_95": round_float(max(0.0, p - margin), 6),
                "upper_95": round_float(min(1.0, p + margin), 6),
                "margin_95": round_float(margin, 6),
            }
        )
    return pd.DataFrame(rows)


def temperature_calibrate(probs: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.power(np.clip(probs, 1e-7, 1.0), 1.0 / float(temperature))
    return normalize_probs(scaled)


def isotonic_calibrate(
    y_cal: np.ndarray,
    probs_cal: np.ndarray,
    probs_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cal_parts = []
    eval_parts = []
    for class_index in range(3):
        iso = IsotonicRegression(out_of_bounds="clip")
        target = (y_cal == class_index).astype(float)
        iso.fit(probs_cal[:, class_index], target)
        cal_parts.append(iso.predict(probs_cal[:, class_index]))
        eval_parts.append(iso.predict(probs_eval[:, class_index]))
    return normalize_probs(np.vstack(cal_parts).T), normalize_probs(np.vstack(eval_parts).T)


def logistic_probability_calibrate(
    y_cal: np.ndarray,
    probs_cal: np.ndarray,
    probs_eval: np.ndarray,
    *,
    centered: bool,
    c_value: float,
    balanced: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x_cal = np.log(np.clip(probs_cal, 1e-7, 1.0))
    x_eval = np.log(np.clip(probs_eval, 1e-7, 1.0))
    if centered:
        x_cal = x_cal - x_cal.mean(axis=1, keepdims=True)
        x_eval = x_eval - x_eval.mean(axis=1, keepdims=True)
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced" if balanced else None,
        max_iter=2000,
    )
    model.fit(x_cal, y_cal)
    return normalize_probs(model.predict_proba(x_cal)), normalize_probs(model.predict_proba(x_eval))


def advanced_calibration_experiments(
    diagnostic: pd.DataFrame,
    y: np.ndarray,
    final_probs: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = diagnostic["date"].to_numpy()
    cal_mask = dates < np.datetime64("2025-01-01")
    eval_mask = dates >= np.datetime64("2025-01-01")
    if int(cal_mask.sum()) < 400 or int(eval_mask.sum()) < 400:
        split = max(1, len(diagnostic) // 2)
        cal_mask = np.zeros(len(diagnostic), dtype=bool)
        cal_mask[:split] = True
        eval_mask = ~cal_mask
    y_cal = y[cal_mask]
    y_eval = y[eval_mask]
    probs_cal = final_probs[cal_mask]
    probs_eval = final_probs[eval_mask]
    identity_eval = metric_block(y_eval, probs_eval)
    identity_objective = objective_from_metrics(identity_eval)
    rows: list[dict[str, Any]] = []

    def add_row(kind: str, candidate: str, cal_probs: np.ndarray, eval_probs: np.ndarray, note: str = "") -> None:
        cal_metrics = metric_block(y_cal, cal_probs)
        eval_metrics = metric_block(y_eval, eval_probs)
        rows.append(
            {
                "experiment": "advanced_calibration",
                "family": kind,
                "candidate": candidate,
                "calibration_rows": int(len(y_cal)),
                "evaluation_rows": int(len(y_eval)),
                "cal_objective": objective_from_metrics(cal_metrics),
                "eval_objective": objective_from_metrics(eval_metrics),
                "eval_objective_delta_vs_identity": round_float(objective_from_metrics(eval_metrics) - identity_objective, 6),
                "eval_log_loss": eval_metrics["log_loss"],
                "eval_log_loss_delta_vs_identity": round_float(float(eval_metrics["log_loss"]) - float(identity_eval["log_loss"]), 6),
                "eval_ece": eval_metrics["ece"],
                "eval_draw_gap": eval_metrics["draw_gap"],
                "eval_accuracy": eval_metrics["accuracy"],
                "promoted": False,
                "decision": "candidate",
                "note": note,
            }
        )

    add_row("identity", "runtime_sem_calibracao_extra", probs_cal, probs_eval, "referencia do runtime atual")
    for temperature in np.linspace(0.70, 1.50, 17):
        add_row(
            "temperature",
            f"T={temperature:.2f}",
            temperature_calibrate(probs_cal, float(temperature)),
            temperature_calibrate(probs_eval, float(temperature)),
        )
    iso_cal, iso_eval = isotonic_calibrate(y_cal, probs_cal, probs_eval)
    add_row("isotonic", "classwise_isotonic", iso_cal, iso_eval, "calibracao por classe, renormalizada")
    for centered, family in [(False, "dirichlet_logistic"), (True, "vector_scaling")]:
        for balanced in [False, True]:
            for c_value in [0.01, 0.03, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.00]:
                cal_probs, eval_probs = logistic_probability_calibrate(
                    y_cal,
                    probs_cal,
                    probs_eval,
                    centered=centered,
                    c_value=c_value,
                    balanced=balanced,
                )
                suffix = "balanced" if balanced else "plain"
                add_row(family, f"C={c_value:.2f}_{suffix}", cal_probs, eval_probs)

    frame = pd.DataFrame(rows).sort_values(["eval_objective", "eval_log_loss"], ascending=[True, True])
    best = frame.iloc[0].to_dict()
    identity = frame[frame["family"] == "identity"].iloc[0].to_dict()
    promote = (
        str(best["family"]) != "identity"
        and float(best["eval_objective"]) <= float(identity["eval_objective"]) - 0.002
        and float(best["eval_log_loss"]) <= float(identity["eval_log_loss"]) - 0.0005
        and float(best["eval_draw_gap"]) <= float(identity["eval_draw_gap"]) + 0.005
    )
    decision = "promoted" if promote else "not_promoted"
    frame.loc[frame.index == frame.index[0], "promoted"] = bool(promote)
    frame.loc[frame.index == frame.index[0], "decision"] = decision
    if not promote:
        frame.loc[frame.index == frame.index[0], "note"] = (
            "nao entrou: nenhum calibrador venceu o runtime em objetivo, log_loss e empate no split temporal 2024->2025+"
        )
    return frame, {
        "method": "2024 calibrates; 2025+ evaluates; no future leakage; candidates enter only if objective, log_loss and draw gap improve materially",
        "promoted": bool(promote),
        "identity": jsonable(identity),
        "best": jsonable(frame.iloc[0].to_dict()),
    }


def fit_team_strength(
    history: pd.DataFrame,
    shrinkage: float,
    half_life_years: float | None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    rows: list[tuple[str, float, float, float]] = []
    max_date = history["date"].max()
    for row in history.itertuples(index=False):
        weight = 1.0
        if half_life_years is not None:
            age_years = max(0.0, (max_date - row.date).days / 365.25)
            weight = 0.5 ** (age_years / float(half_life_years))
        rows.append((str(row.home_team), float(row.home_score), float(row.away_score), weight))
        rows.append((str(row.away_team), float(row.away_score), float(row.home_score), weight))
    team_games = pd.DataFrame(rows, columns=["team", "gf", "ga", "weight"])
    base_goal = float((team_games["gf"] * team_games["weight"]).sum() / max(1e-7, team_games["weight"].sum()))
    weighted = team_games.assign(gf_w=team_games["gf"] * team_games["weight"], ga_w=team_games["ga"] * team_games["weight"])
    grouped = weighted.groupby("team").agg(weight=("weight", "sum"), gf=("gf_w", "sum"), ga=("ga_w", "sum"))
    attack = ((grouped["gf"] + float(shrinkage) * base_goal) / (grouped["weight"] + float(shrinkage))) / base_goal
    defense = ((grouped["ga"] + float(shrinkage) * base_goal) / (grouped["weight"] + float(shrinkage))) / base_goal
    return base_goal, attack.to_dict(), defense.to_dict()


def team_strength_poisson_probs(
    history: pd.DataFrame,
    frame: pd.DataFrame,
    rho: float,
    shrinkage: float,
    half_life_years: float | None,
) -> np.ndarray:
    base_goal, attack, defense = fit_team_strength(history, shrinkage, half_life_years)
    home_lam: list[float] = []
    away_lam: list[float] = []
    for row in frame.itertuples(index=False):
        neutral = bool(row.neutral)
        home_factor = 1.08 if not neutral else 1.0
        away_factor = 0.94 if not neutral else 1.0
        home_lam.append(
            float(np.clip(base_goal * attack.get(str(row.home_team), 1.0) * defense.get(str(row.away_team), 1.0) * home_factor, 0.05, 5.5))
        )
        away_lam.append(
            float(np.clip(base_goal * attack.get(str(row.away_team), 1.0) * defense.get(str(row.home_team), 1.0) * away_factor, 0.05, 5.5))
        )
    return sota.lambdas_to_prob_array(np.asarray(home_lam), np.asarray(away_lam), rho=float(rho))


def dixon_coles_team_strength_experiments(
    history: pd.DataFrame,
    diagnostic: pd.DataFrame,
    y: np.ndarray,
    pre_draw: np.ndarray,
    poisson_probs: np.ndarray,
    policy: dict[str, Any],
    rho: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runtime_probs, runtime_policy_metrics = apply_policy(y, pre_draw, poisson_probs, policy)
    runtime_metrics = metric_block(y, runtime_probs)
    runtime_objective = float(runtime_policy_metrics["objective"])

    def add_row(
        family: str,
        candidate: str,
        candidate_poisson: np.ndarray,
        note: str,
    ) -> None:
        final_probs, metrics = apply_policy(y, pre_draw, candidate_poisson, policy)
        block = metric_block(y, final_probs)
        rows.append(
            {
                "experiment": "dixon_coles_team_strength",
                "family": family,
                "candidate": candidate,
                "calibration_rows": int(len(history)),
                "evaluation_rows": int(len(y)),
                "eval_objective": metrics["objective"],
                "eval_objective_delta_vs_runtime": round_float(float(metrics["objective"]) - runtime_objective, 6),
                "eval_log_loss": metrics["log_loss"],
                "eval_log_loss_delta_vs_runtime": round_float(float(metrics["log_loss"]) - float(runtime_metrics["log_loss"]), 6),
                "eval_ece": metrics["ece"],
                "eval_draw_gap": metrics["draw_gap"],
                "eval_accuracy": block["accuracy"],
                "promoted": False,
                "decision": "candidate",
                "note": note,
            }
        )

    add_row("runtime", "poisson_regressor_dixon_coles", poisson_probs, "referencia Poisson/DC atual do runtime")
    strength_candidates: list[tuple[str, np.ndarray]] = []
    for shrinkage in [4, 8, 12, 20, 32, 50]:
        for half_life in [None, 3.0, 5.0, 8.0]:
            candidate_poisson = team_strength_poisson_probs(history, diagnostic, rho, shrinkage, half_life)
            half_life_label = "none" if half_life is None else f"{half_life:g}"
            label = f"shrink={shrinkage}_half_life={half_life_label}"
            strength_candidates.append((label, candidate_poisson))
            add_row("team_attack_defense_dc", label, candidate_poisson, "ataque/defesa por selecao com shrinkage temporal")

    best_strength_label, best_strength_poisson = min(
        strength_candidates,
        key=lambda item: float(apply_policy(y, pre_draw, item[1], policy)[1]["objective"]),
    )
    for alpha in [0.05, 0.10, 0.20, 0.30, 0.50, 0.75]:
        mixed = normalize_probs(alpha * best_strength_poisson + (1.0 - alpha) * poisson_probs)
        add_row("poisson_regressor_plus_team_strength", f"{best_strength_label}_alpha={alpha:.2f}", mixed, "blend do Poisson atual com força ataque/defesa")

    frame = pd.DataFrame(rows).sort_values(["eval_objective", "eval_log_loss"], ascending=[True, True])
    best = frame.iloc[0].to_dict()
    runtime = frame[frame["family"] == "runtime"].iloc[0].to_dict()
    promote = (
        str(best["family"]) != "runtime"
        and float(best["eval_objective"]) <= float(runtime["eval_objective"]) - 0.002
        and float(best["eval_log_loss"]) <= float(runtime["eval_log_loss"]) - 0.0005
    )
    frame.loc[frame.index == frame.index[0], "promoted"] = bool(promote)
    frame.loc[frame.index == frame.index[0], "decision"] = "promoted" if promote else "not_promoted"
    if not promote:
        frame.loc[frame.index == frame.index[0], "note"] = (
            "nao entrou: ganho diagnostico insuficiente e/ou log_loss pior que o PoissonRegressor+Dixon-Coles atual"
        )
    return frame, {
        "method": "team attack/defense strengths estimated from pre-2024 history only, with shrinkage and temporal decay; candidates enter only with material objective and log_loss gains",
        "promoted": bool(promote),
        "runtime": jsonable(runtime),
        "best": jsonable(frame.iloc[0].to_dict()),
    }


def runtime_adjustment_audit(package: dict[str, Any], policy: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    fixtures = package["fixtures"]
    group_fixtures = fixtures[fixtures["stage_id"] == sota.GROUP_STAGE_ID]
    teams = sorted({str(team) for team in pd.concat([group_fixtures["home_team"], group_fixtures["away_team"]]).dropna() if str(team)})
    weights = sota.normalize_manual_blend_weights(dict(package.get("manual_blend_weights", sota.DEFAULT_MANUAL_BLEND_WEIGHTS)))
    states = sota.ensure_states(package)
    rows: list[dict[str, Any]] = []
    for home, away in permutations(teams, 2):
        h = sota.canonical_team(home)
        a = sota.canonical_team(away)
        x = sota.prepare_match_features(package, h, a, neutral=True)
        elo_home = sota.expected_score(states[h].elo, states[a].elo)
        draw = max(0.15, min(0.30, 0.27 - abs(elo_home - 0.5) * 0.20))
        p_elo = np.array([(1 - draw) * elo_home, draw, (1 - draw) * (1 - elo_home)])
        stack = sota.base_probability_stack(package, x, h, a, True, p_elo)
        base_probs = sum(float(weights.get(name, 0.0)) * probs for name, probs in stack.items() if name in weights)
        base_probs = normalize_probs(np.asarray([base_probs]))[0]
        pred = sota.predict_match(package, h, a, neutral=True)
        adjusted_probs = np.array(
            [
                float(pred["p_pre_draw_home_win_90"]),
                float(pred["p_pre_draw_draw_90"]),
                float(pred["p_pre_draw_away_win_90"]),
            ]
        )
        final_probs = np.array([float(pred["p_home_win_90"]), float(pred["p_draw_90"]), float(pred["p_away_win_90"])])
        shift = adjusted_probs - base_probs
        final_shift = final_probs - base_probs
        rows.append(
            {
                "home_team": h,
                "away_team": a,
                "base_home": round_float(base_probs[0], 6),
                "base_draw": round_float(base_probs[1], 6),
                "base_away": round_float(base_probs[2], 6),
                "adjusted_home": round_float(adjusted_probs[0], 6),
                "adjusted_draw": round_float(adjusted_probs[1], 6),
                "adjusted_away": round_float(adjusted_probs[2], 6),
                "final_home": round_float(final_probs[0], 6),
                "final_draw": round_float(final_probs[1], 6),
                "final_away": round_float(final_probs[2], 6),
                "max_abs_shift_pre_draw": round_float(np.abs(shift).max(), 6),
                "sum_abs_shift_pre_draw": round_float(np.abs(shift).sum(), 6),
                "max_abs_shift_final": round_float(np.abs(final_shift).max(), 6),
                "base_argmax": LABELS[int(base_probs.argmax())],
                "adjusted_argmax": LABELS[int(adjusted_probs.argmax())],
                "final_argmax": LABELS[int(final_probs.argmax())],
                "argmax_changed_pre_draw": bool(int(base_probs.argmax()) != int(adjusted_probs.argmax())),
                "argmax_changed_final": bool(int(base_probs.argmax()) != int(final_probs.argmax())),
                "squad_top26_diff": round_float(pred.get("squad_top26_diff", 0.0), 6),
                "tm_market_value_log_diff": round_float(pred.get("tm_market_value_log_diff", 0.0), 6),
                "tm_caps_diff": round_float(pred.get("tm_caps_diff", 0.0), 6),
                "tm_recent_injury_days_diff": round_float(pred.get("tm_recent_injury_days_diff", 0.0), 6),
                "context_shift": round_float(pred.get("context_shift", 0.0), 6),
            }
        )
    frame = pd.DataFrame(rows).sort_values("max_abs_shift_pre_draw", ascending=False)
    summary = {
        "path": str(RUNTIME_ADJUSTMENT_CSV),
        "method": "all ordered 2026 qualified-team pairs; compare base classifier blend before squad/Transfermarkt/context proxies against pre-draw and final runtime probabilities",
        "teams": int(len(teams)),
        "pairs": int(len(frame)),
        "max_abs_shift_pre_draw": round_float(frame["max_abs_shift_pre_draw"].max() if len(frame) else 0.0, 6),
        "p95_abs_shift_pre_draw": round_float(frame["max_abs_shift_pre_draw"].quantile(0.95) if len(frame) else 0.0, 6),
        "mean_abs_shift_pre_draw": round_float(frame["max_abs_shift_pre_draw"].mean() if len(frame) else 0.0, 6),
        "argmax_flip_rate_pre_draw": round_float(frame["argmax_changed_pre_draw"].mean() if len(frame) else 0.0, 6),
        "max_abs_shift_final": round_float(frame["max_abs_shift_final"].max() if len(frame) else 0.0, 6),
        "p95_abs_shift_final": round_float(frame["max_abs_shift_final"].quantile(0.95) if len(frame) else 0.0, 6),
        "decision": "audit_only_runtime_kept",
        "reason": (
            "Ajustes de elenco/Transfermarkt/contexto usam proxies 2026 sem backtest historico limpo; entram como camada operacional auditada, "
            "nao como novo tuning escondido. Limites de sanidade reprovam se o maximo passar de 35pp ou p95 passar de 18pp."
        ),
        "top_10": frame.head(10).to_dict(orient="records"),
        "policy_reference": {
            "classifier_weight": round_float(policy["classifier_weight"], 4),
            "poisson_weight": round_float(policy["poisson_weight"], 4),
            "draw_floor": round_float(policy["draw_floor"], 4),
            "draw_ceiling": round_float(policy["draw_ceiling"], 4),
        },
    }
    return frame, summary


def load_monte_carlo_stability() -> dict[str, Any]:
    if not MC_STABILITY_JSON.exists():
        return {
            "available": False,
            "fresh": False,
            "passed": False,
            "path": str(MC_STABILITY_JSON),
            "reason": "Rode `make mc-stability` para gerar estabilidade offline 5k/10k e fase/chave 1k/2k sem travar o jogo.",
        }
    data = json.loads(MC_STABILITY_JSON.read_text(encoding="utf-8"))
    runs = sorted(int(item.get("runs", 0)) for item in data.get("runs", []))
    current_fingerprints = {
        "model_package": file_fingerprint(MODEL_PATH),
        "model_report": file_fingerprint(MODEL_REPORT),
        "training_matches": file_fingerprint(TRAINING_PATH),
        "sota_pipeline": file_fingerprint(SOTA_PIPELINE),
        "mc_stability_script": file_fingerprint(MC_STABILITY_SCRIPT),
    }
    reported_fingerprints = data.get("source_fingerprints", {})
    fresh = bool(reported_fingerprints) and all(
        fingerprint_is_fresh(dict(reported_fingerprints.get(name, {})), current)
        for name, current in current_fingerprints.items()
    )
    final_comparison = data.get("final_comparison", {})
    stage_bracket_runs = sorted(int(item.get("runs", 0)) for item in data.get("stage_bracket_runs", []))
    stage_bracket_comparison = data.get("stage_bracket_final_comparison", {})
    stability_gate = data.get("stability_gate", {})
    max_delta = float(stability_gate.get("max_top16_abs_delta", 0.01))
    max_churn = int(stability_gate.get("max_top16_churn", 0))
    leader_change_allowed = bool(stability_gate.get("leader_change_allowed", False))
    final_passed = bool(data.get("passed", False))
    if final_comparison and not bool(final_comparison.get("baseline", False)):
        final_passed = (
            final_passed
            and (leader_change_allowed or not bool(final_comparison.get("leader_changed", False)))
            and float(final_comparison.get("max_top16_abs_delta", 1.0)) <= max_delta
            and int(final_comparison.get("top16_churn_count", 99)) <= max_churn
        )
    stage_bracket_passed = bool(data.get("stage_bracket_passed", False))
    if stage_bracket_comparison and not bool(stage_bracket_comparison.get("baseline", False)):
        stage_bracket_passed = (
            stage_bracket_passed
            and float(stage_bracket_comparison.get("max_stage_top16_abs_delta", 1.0)) <= float(stability_gate.get("max_stage_top16_abs_delta", 0.035))
            and int(stage_bracket_comparison.get("max_stage_top16_churn", 99)) <= int(stability_gate.get("max_stage_top16_churn", 4))
            and float(stage_bracket_comparison.get("max_pair_top8_abs_delta", 1.0)) <= float(stability_gate.get("max_pair_top8_abs_delta", 0.02))
            and int(stage_bracket_comparison.get("max_pair_top8_churn", 99)) <= int(stability_gate.get("max_pair_top8_churn", 8))
        )
    passed = bool(final_passed and fresh and stage_bracket_passed and 5_000 in runs and 10_000 in runs and 1_000 in stage_bracket_runs and 2_000 in stage_bracket_runs)
    return {
        "available": True,
        "fresh": bool(fresh),
        "passed": bool(passed),
        "stage_bracket_passed": bool(stage_bracket_passed),
        "path": str(MC_STABILITY_JSON),
        "csv_path": str(MC_STABILITY_CSV),
        "stage_bracket_csv_path": str(MC_STAGE_BRACKET_CSV),
        "runs": runs,
        "stage_bracket_runs": stage_bracket_runs,
        "stability_gate": stability_gate,
        "final_comparison": final_comparison,
        "stage_bracket_final_comparison": stage_bracket_comparison,
        "summary": data.get("summary", {}),
        "source_fingerprints": reported_fingerprints,
    }


def public_baseline_benchmark(model_report: dict[str, Any], runtime_metrics: dict[str, float]) -> dict[str, Any]:
    metrics = model_report.get("metrics", {})
    elo = metrics.get("baseline_elo_1x2", {})
    fifa = metrics.get("baseline_fifa_rank_1x2", {})
    xgb = metrics.get("xgb_temperature_calibrated_1x2", metrics.get("xgb_1x2", {}))
    competitive = metrics.get("competitive_xgb_1x2", {})

    runtime_accuracy = float(runtime_metrics.get("accuracy", 0.0))
    runtime_log_loss = float(runtime_metrics.get("log_loss", 99.0))
    elo_accuracy = float(elo.get("accuracy", 0.0))
    fifa_accuracy = float(fifa.get("accuracy", 0.0))
    xgb_log_loss = float(xgb.get("log_loss", 99.0))
    competitive_log_loss = float(competitive.get("log_loss", 99.0))

    return {
        "status": "available_public_style_baselines_only",
        "market_odds_benchmark": {
            "available": False,
            "reason": "O pacote atual nao contem odds historicas limpas de casas de aposta; nao inventamos benchmark externo sem dados auditaveis.",
        },
        "available_baselines": {
            "elo_accuracy": round_float(elo_accuracy, 6),
            "fifa_rank_accuracy": round_float(fifa_accuracy, 6),
            "xgb_calibrated_log_loss": round_float(xgb_log_loss, 6),
            "competitive_xgb_log_loss": round_float(competitive_log_loss, 6),
        },
        "runtime_policy": {
            "accuracy": round_float(runtime_accuracy, 6),
            "log_loss": round_float(runtime_log_loss, 6),
            "accuracy_gain_vs_elo_pp": round_float((runtime_accuracy - elo_accuracy) * 100.0, 3),
            "accuracy_gain_vs_fifa_pp": round_float((runtime_accuracy - fifa_accuracy) * 100.0, 3),
            "log_loss_gap_vs_xgb_calibrated": round_float(runtime_log_loss - xgb_log_loss, 6),
            "log_loss_gap_vs_competitive_xgb": round_float(runtime_log_loss - competitive_log_loss, 6),
        },
        "interpretation": (
            "O runtime precisa superar baselines publicos simples de forca/ranking e ficar perto da fronteira XGBoost, "
            "mas preservando Poisson/Dixon-Coles para placar, empate e variancia de futebol."
        ),
    }


def academic_stamp(
    report: dict[str, Any],
    benchmark: dict[str, Any],
    full_objective_gap: float,
    rho_objective_gap: float,
) -> dict[str, Any]:
    policy_metrics = report["calibration"]["runtime_2024_plus_metrics"]
    nested = report["nested_temporal"]
    component_ablation = nested.get("component_ablation", {})
    benchmark_runtime = benchmark["runtime_policy"]
    hard_gates = {
        "nested_temporal_no_leakage": bool(str(nested.get("version", "")).endswith("no_leakage_no_draw_xgb")),
        "complete_component_ablation_63_subsets": int(component_ablation.get("candidate_count", 0)) >= 63,
        "nested_component_and_policy_grid": int(component_ablation.get("policy_candidate_count_per_component", 0)) >= 690,
        "draw_xgb_removed": report["policy"].get("draw_xgb") == "removed_zero_weight_model",
        "runtime_draw_gap_lte_2pp": float(policy_metrics.get("draw_gap", 1.0)) <= 0.02,
        "runtime_log_loss_lte_0_82": float(policy_metrics.get("log_loss", 99.0)) <= 0.82,
        "runtime_near_ablation_frontier": float(full_objective_gap) <= 0.012,
        "dixon_coles_near_rho_frontier": float(rho_objective_gap) <= 0.01,
        "beats_elo_accuracy_by_5pp": float(benchmark_runtime.get("accuracy_gain_vs_elo_pp", 0.0)) >= 5.0,
        "beats_fifa_accuracy_by_7pp": float(benchmark_runtime.get("accuracy_gain_vs_fifa_pp", 0.0)) >= 7.0,
        "monte_carlo_uncertainty_reported": bool(report.get("uncertainty_intervals", {}).get("champion_top_16")),
        "stage_uncertainty_reported": bool(report.get("uncertainty_intervals", {}).get("stage_top_32")),
        "advanced_calibration_exhausted": "advanced_calibration" in report,
        "team_strength_dixon_coles_exhausted": "dixon_coles_team_strength" in report,
        "class_calibration_reported": bool(report.get("calibration", {}).get("class_summary", {}).get("rows", 0)),
        "block_bootstrap_reported": bool(report.get("uncertainty_intervals", {}).get("block_bootstrap", {}).get("rows", 0)),
        "runtime_adjustment_audit_reported": bool(report.get("runtime_adjustment_audit", {}).get("pairs", 0)),
        "runtime_adjustment_max_shift_lte_35pp": float(report.get("runtime_adjustment_audit", {}).get("max_abs_shift_pre_draw", 1.0)) <= 0.35,
        "runtime_adjustment_p95_shift_lte_18pp": float(report.get("runtime_adjustment_audit", {}).get("p95_abs_shift_pre_draw", 1.0)) <= 0.18,
        "raw_data_manifest_reported": int(report.get("raw_data_manifest", {}).get("file_count", 0)) >= 1,
        "raw_data_manifest_hash_reported": bool(report.get("raw_data_manifest", {}).get("manifest_sha256")),
        "raw_data_semantic_sanity_passed": bool(report.get("raw_data_manifest", {}).get("semantic", {}).get("passed")),
        "source_fingerprints_reported": all(
            name in report.get("source_fingerprints", {})
            for name in ["model_package", "model_report", "training_matches", "sota_pipeline", "stats_qa_script"]
        ),
        "external_elo_parse_complete": float(report.get("external_elo_audit", {}).get("parsed_ratio", 0.0)) >= 0.95,
        "external_elo_current": int(str(report.get("external_elo_audit", {}).get("max_date", "1900"))[:4]) >= 2024,
        "external_elo_qualified_coverage_complete": float(report.get("external_elo_audit", {}).get("qualified_team_coverage_ratio", 0.0)) >= 0.95,
        "mc_stability_available": bool(report.get("monte_carlo_stability", {}).get("available")),
        "mc_stability_fresh": bool(report.get("monte_carlo_stability", {}).get("fresh")),
        "mc_stability_passed": bool(report.get("monte_carlo_stability", {}).get("passed")),
        "mc_stage_bracket_stability_passed": bool(report.get("monte_carlo_stability", {}).get("stage_bracket_passed")),
    }
    approved = all(hard_gates.values())
    return {
        "stamp": "SOTA/KISS academico aplicado aos dados disponiveis" if approved else "SOTA/KISS pendente",
        "approved": approved,
        "scope": (
            "Carimbo academico pragmatico: valida componente, politica, empate, Poisson/Dixon-Coles, incerteza e baselines "
            "publicos disponiveis no pacote. Nao declara superioridade contra mercado sem odds historicas externas."
        ),
        "hard_gates": hard_gates,
        "not_promoted_by_design": [
            {
                "item": "odds de mercado/bookmakers",
                "reason": "ausentes no pacote atual; sem fonte auditavel nao ha benchmark externo honesto",
            },
            {
                "item": "Dixon-Coles hierarquico completo com shrinkage bayesiano",
                "reason": "aproximacao ataque/defesa por selecao foi testada com historico pre-2024; nao ganhou materialmente do PoissonRegressor+Dixon-Coles atual",
            },
            {
                "item": "isotonic/vector/Dirichlet calibration em runtime",
                "reason": "isotonic, temperature, vector scaling e Dirichlet foram testados em 2024->2025+; nenhum venceu objetivo, log_loss e empate ao mesmo tempo",
            },
            {
                "item": "ajustes 2026 de elenco/Transfermarkt/contexto como modelo calibrado historicamente",
                "reason": "foram auditados por limite de deslocamento probabilistico, mas nao promovidos a tuning academico porque faltam snapshots historicos equivalentes",
            },
        ],
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, sep, *body])


def write_markdown(report: dict[str, Any], ablation: pd.DataFrame, uncertainty: pd.DataFrame, dc: pd.DataFrame) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    policy = report["policy"]
    calibration = report["calibration"]
    nested = report["nested_temporal"]
    lines = [
        "# Auditoria estatística SOTA/KISS",
        "",
        "Este relatório executa os checks estatísticos ativos e a ablação completa dos subconjuntos do blend.",
        "",
        "## Model Card Resumido",
        "",
        "Esta seção preserva no relatório canônico o resumo que antes ficava em `model_card.md`.",
        "",
        "```json",
        json.dumps(report["model_card"], indent=2),
        "```",
        "",
        "## Veredito",
        "",
        f"- SOTA/KISS pragmático: `{report['verdict']['sota_kiss']}`",
        f"- Carimbo acadêmico: `{report['academic_stamp']['stamp']}`",
        f"- Motivo: {report['verdict']['reason']}",
        f"- Recomendações abertas: `{len(report.get('recommendations', []))}`",
        "",
        "Critérios duros do carimbo:",
        "",
        "```json",
        json.dumps(report["academic_stamp"]["hard_gates"], indent=2),
        "```",
        "",
        "Freshness dos artefatos:",
        "",
        "```json",
        json.dumps(report["source_fingerprints"], indent=2),
        "```",
        "",
        "Manifesto completo dos dados brutos:",
        "",
        "```json",
        json.dumps(
            {
                "path": report["raw_data_manifest"]["path"],
                "csv_path": report["raw_data_manifest"]["csv_path"],
                "file_count": report["raw_data_manifest"]["file_count"],
                "csv_file_count": report["raw_data_manifest"]["csv_file_count"],
                "total_size_bytes": report["raw_data_manifest"]["total_size_bytes"],
                "manifest_sha256": report["raw_data_manifest"]["manifest_sha256"],
                "semantic": report["raw_data_manifest"]["semantic"],
            },
            indent=2,
        ),
        "```",
        "",
        "## Política ativa",
        "",
        "```json",
        json.dumps(policy, indent=2),
        "```",
        "",
        "## 1. Calibração mais forte",
        "",
        "A seleção da política continua vindo da validação nested temporal sem vazamento. A curva de calibração abaixo é um diagnóstico do pacote atual em 2024+.",
        "",
        "```json",
        json.dumps(nested["selected_policy"], indent=2),
        "```",
        "",
        "```json",
        json.dumps(calibration["runtime_2024_plus_metrics"], indent=2),
        "```",
        "",
        "Resumo por classe:",
        "",
        markdown_table(
            report["calibration"]["class_summary_rows"],
            ["label", "sample_count", "weighted_abs_gap", "max_abs_gap", "mean_predicted_rate", "mean_empirical_rate"],
        ),
        "",
        "### Fronteira sem dataset externo",
        "",
        "Calibradores extras e Dixon-Coles por ataque/defesa foram testados com dados já existentes. Só entrariam no runtime se vencessem materialmente sem vazar futuro.",
        "",
        "```json",
        json.dumps(
            {
                "advanced_calibration": report["advanced_calibration"],
                "dixon_coles_team_strength": report["dixon_coles_team_strength"],
                "frontier_experiments": report["internal_frontier_experiments"],
            },
            indent=2,
        ),
        "```",
        "",
        "## 2. Intervalos de incerteza",
        "",
        "Campeão:",
        "",
        markdown_table(
            uncertainty.head(8).to_dict(orient="records"),
            ["team", "wins", "probability", "lower_95", "upper_95", "margin_95"],
        ),
        "",
        "Fases:",
        "",
        markdown_table(
            report["uncertainty_intervals"]["stage_top_32"][:8],
            ["team", "stage", "probability", "lower_95", "upper_95", "margin_95"],
        ),
        "",
        "Bootstrap por bloco temporal/torneio:",
        "",
        markdown_table(
            report["uncertainty_intervals"]["block_bootstrap_rows"],
            ["block_type", "metric", "block_count", "mean", "lower_95", "upper_95", "width_95"],
        ),
        "",
        "Estabilidade Monte Carlo offline:",
        "",
        "```json",
        json.dumps(report["monte_carlo_stability"], indent=2),
        "```",
        "",
        "## 3. Ablation study",
        "",
        "A tabela inclui todos os 63 subconjuntos dos seis sinais (`xgb`, `competitive`, `logistic`, `elo`, `poisson`, `count_poisson`) e compara contra a política ativa do runtime.",
        "",
        markdown_table(
            ablation.head(10).to_dict(orient="records"),
            ["ablation", "objective", "log_loss", "rps", "draw_gap", "entropy"],
        ),
        "",
        "## 4. Draw-specific calibration",
        "",
        "```json",
        json.dumps(report["draw_specific_calibration"], indent=2),
        "```",
        "",
        "## 5. Dixon-Coles",
        "",
        markdown_table(
            dc.head(8).to_dict(orient="records"),
            ["rho", "hybrid_objective", "hybrid_log_loss", "hybrid_draw_gap", "poisson_only_log_loss"],
        ),
        "",
        "## Benchmark externo disponível",
        "",
        "```json",
        json.dumps(report["external_benchmark"], indent=2),
        "```",
        "",
        "## Auditoria dos ajustes 2026",
        "",
        "Os ajustes de elenco, Transfermarkt e contexto entram no runtime porque a Copa 2026 precisa refletir força atual de elenco. Como não existem snapshots históricos equivalentes no pacote, eles são auditados por limite de deslocamento probabilístico, e não usados para retunar a validação temporal.",
        "",
        "```json",
        json.dumps(report["runtime_adjustment_audit"], indent=2),
        "```",
        "",
        "## Escopo do carimbo acadêmico",
        "",
        report["academic_stamp"]["scope"],
        "",
        "Itens não promovidos por desenho:",
        "",
        markdown_table(
            report["academic_stamp"]["not_promoted_by_design"],
            ["item", "reason"],
        ),
        "",
        "## 6. Relatório estatístico",
        "",
        "Arquivos gerados:",
        "",
        f"- `{REPORT_JSON}`",
        f"- `{CALIBRATION_CSV}`",
        f"- `{CLASS_CALIBRATION_CSV}`",
        f"- `{BLOCK_BOOTSTRAP_CSV}`",
        f"- `{ABLATION_CSV}`",
        f"- `{UNCERTAINTY_CSV}`",
        f"- `{STAGE_UNCERTAINTY_CSV}`",
        f"- `{DC_CSV}`",
        f"- `{FRONTIER_CSV}`",
        f"- `{RUNTIME_ADJUSTMENT_CSV}`",
        f"- `{MC_STABILITY_JSON}`",
        f"- `{RAW_MANIFEST_JSON}`",
        f"- `{RAW_MANIFEST_CSV}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    start = time.perf_counter()
    log_step(start, "loading model, report and training data")
    package, model_report, training = load_artifacts()
    log_step(start, "hashing complete raw data manifest")
    raw_manifest = raw_data_manifest()
    if "draw_xgb" in package.get("models", {}):
        raise AssertionError("draw_xgb is still present; SOTA/KISS policy requires it removed")
    if package.get("version") != model_report.get("version"):
        raise AssertionError("model package/report version mismatch")

    raw_policy = package.get("simulation_policy", {})
    policy = sota.simulation_policy_from_package(package)
    nested = raw_policy.get("nested_temporal_validation", {})
    selected = nested.get("selected_policy", {})
    if policy.get("selected_by") != "strict_nested_temporal_component_ablation_no_leakage_no_draw_xgb":
        raise AssertionError(f"unexpected policy selector: {policy.get('selected_by')}")
    runtime_weights = sota.normalize_manual_blend_weights(
        dict(package.get("manual_blend_weights", raw_policy.get("manual_blend_weights", sota.DEFAULT_MANUAL_BLEND_WEIGHTS)))
    )
    policy_weights = sota.normalize_manual_blend_weights(dict(raw_policy.get("manual_blend_weights", runtime_weights)))
    if runtime_weights != policy_weights:
        raise AssertionError("package manual_blend_weights and policy manual_blend_weights diverge")
    if int(selected.get("selected_folds", 0)) < 4:
        raise AssertionError(f"nested selected policy is too weak: {selected}")
    if float(selected.get("outer_draw_gap", 1.0)) > 0.02:
        raise AssertionError(f"nested draw calibration gap too high: {selected}")

    diagnostic = training[training["date"] >= "2024-01-01"].copy()
    if len(diagnostic) < 1000:
        raise AssertionError(f"diagnostic window too small: {len(diagnostic)}")
    log_step(start, f"building 2024+ policy arrays over {len(diagnostic)} rows")
    y, pre_draw, poisson = sota.backtest_policy_arrays(
        package["models"],
        diagnostic,
        rho=sota.dixon_coles_rho_from_package(package),
        use_stacking=bool(package.get("use_stacking_ensemble", False)),
        manual_weights=runtime_weights,
    )
    final_probs, policy_metrics = apply_policy(y, pre_draw, poisson, policy)
    runtime_metrics = metric_block(y, final_probs)
    calibration_bins, calibration_summary = confidence_calibration_bins(y, final_probs)
    calibration_bins.to_csv(CALIBRATION_CSV, index=False)
    class_calibration, class_calibration_report = class_calibration_summary(calibration_bins)
    class_calibration.to_csv(CLASS_CALIBRATION_CSV, index=False)

    log_step(start, "running bootstrap uncertainty and champion intervals")
    bootstrap = bootstrap_metric_intervals(y, final_probs)
    block_bootstrap, block_bootstrap_summary = block_bootstrap_metric_intervals(diagnostic, y, final_probs)
    block_bootstrap.to_csv(BLOCK_BOOTSTRAP_CSV, index=False)
    uncertainty = champion_uncertainty(model_report)
    uncertainty.to_csv(UNCERTAINTY_CSV, index=False)
    stage_intervals = stage_uncertainty(model_report)
    stage_intervals.to_csv(STAGE_UNCERTAINTY_CSV, index=False)

    log_step(start, "running component ablation")
    components, component_poisson = component_arrays(package, diagnostic, sota.dixon_coles_rho_from_package(package))
    ablation = ablation_study(y, components, component_poisson, policy, runtime_weights)
    ablation.to_csv(ABLATION_CSV, index=False)

    log_step(start, "scanning draw floor/ceiling policy")
    draw_scan = draw_policy_scan(y, pre_draw, poisson, policy)

    log_step(start, "scanning Dixon-Coles rho sensitivity")
    dc = dixon_coles_sensitivity(y, pre_draw, diagnostic, package, policy)
    dc.to_csv(DC_CSV, index=False)

    log_step(start, "testing advanced calibration candidates without external data")
    advanced_calibration, advanced_calibration_summary = advanced_calibration_experiments(diagnostic, y, final_probs)

    log_step(start, "testing team-strength Dixon-Coles candidates without external data")
    history = training[training["date"] < "2024-01-01"].copy()
    team_strength_dc, team_strength_dc_summary = dixon_coles_team_strength_experiments(
        history,
        diagnostic,
        y,
        pre_draw,
        poisson,
        policy,
        sota.dixon_coles_rho_from_package(package),
    )
    frontier = pd.concat([advanced_calibration, team_strength_dc], ignore_index=True, sort=False)
    frontier.to_csv(FRONTIER_CSV, index=False)
    if advanced_calibration_summary["promoted"]:
        raise AssertionError(f"advanced calibration produced a promotable candidate; integrate it before accepting: {advanced_calibration_summary}")
    if team_strength_dc_summary["promoted"]:
        raise AssertionError(f"team-strength Dixon-Coles produced a promotable candidate; integrate it before accepting: {team_strength_dc_summary}")

    log_step(start, "auditing 2026 squad/Transfermarkt/context adjustment bounds")
    adjustment_audit, adjustment_summary = runtime_adjustment_audit(package, policy)
    adjustment_audit.to_csv(RUNTIME_ADJUSTMENT_CSV, index=False)
    if float(adjustment_summary["max_abs_shift_pre_draw"]) > 0.35:
        raise AssertionError(f"runtime adjustment max shift exceeded sanity bound: {adjustment_summary}")
    if float(adjustment_summary["p95_abs_shift_pre_draw"]) > 0.18:
        raise AssertionError(f"runtime adjustment p95 shift exceeded sanity bound: {adjustment_summary}")

    mc_stability = load_monte_carlo_stability()

    best_ablation = ablation.iloc[0].to_dict()
    full_ablation = ablation[ablation["ablation"] == "runtime_policy"].iloc[0].to_dict()
    best_dc = dc.iloc[0].to_dict()
    package_rho = round_float(sota.dixon_coles_rho_from_package(package), 4)
    package_dc = dc[dc["rho"] == package_rho]
    if package_dc.empty:
        raise AssertionError(f"package rho {package_rho} not found in sensitivity grid")
    package_dc_row = package_dc.iloc[0].to_dict()

    full_objective_gap = float(full_ablation["objective"]) - float(best_ablation["objective"])
    rho_objective_gap = float(package_dc_row["hybrid_objective"]) - float(best_dc["hybrid_objective"])
    if float(policy_metrics["log_loss"]) > 0.82:
        raise AssertionError(f"runtime policy log_loss regressed: {policy_metrics}")
    if float(policy_metrics["draw_gap"]) > 0.02:
        raise AssertionError(f"runtime policy draw gap too high: {policy_metrics}")
    if int(draw_scan["current_policy_rank"]) <= 0 or int(draw_scan["current_policy_rank"]) > 8:
        raise AssertionError(f"current policy no longer near the draw-calibration frontier: {draw_scan['current_policy_rank']}")
    if full_objective_gap > 0.012:
        raise AssertionError(f"full policy trails ablation frontier too much: gap={full_objective_gap:.6f}")
    if rho_objective_gap > 0.01:
        raise AssertionError(f"package Dixon-Coles rho trails sensitivity frontier too much: gap={rho_objective_gap:.6f}")
    benchmark = public_baseline_benchmark(model_report, runtime_metrics)
    if float(benchmark["runtime_policy"]["accuracy_gain_vs_elo_pp"]) < 5.0:
        raise AssertionError(f"runtime policy does not beat ELO baseline enough: {benchmark}")
    if float(benchmark["runtime_policy"]["accuracy_gain_vs_fifa_pp"]) < 7.0:
        raise AssertionError(f"runtime policy does not beat FIFA rank baseline enough: {benchmark}")

    report = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "version": package.get("version"),
        "model_card": {
            "version": model_report.get("version"),
            "simulation_policy": model_report.get("simulation_policy", {}),
            "metrics": model_report.get("metrics", {}),
            "accepted_experiments": model_report.get("accepted_experiments", {}),
            "metric_gains_vs_previous_report": model_report.get("metric_gains_vs_previous_report", {}),
            "metric_gains_vs_sota_v1": model_report.get("metric_gains_vs_sota_v1", {}),
            "world_cup_backtest_aggregate": model_report.get("world_cup_backtest_aggregate", {}),
            "monte_carlo_runs": model_report.get("monte_carlo_runs"),
            "sample_champion_seed_2026": model_report.get("sample_champion_seed_2026"),
            "top_10_champion_odds": model_report.get("top_10_champion_odds", []),
        },
        "source_fingerprints": source_fingerprints(),
        "raw_data_manifest": {
            "path": str(RAW_MANIFEST_JSON),
            "csv_path": str(RAW_MANIFEST_CSV),
            "root": raw_manifest["root"],
            "file_count": raw_manifest["file_count"],
            "csv_file_count": raw_manifest["csv_file_count"],
            "total_size_bytes": raw_manifest["total_size_bytes"],
            "manifest_sha256": raw_manifest["manifest_sha256"],
            "semantic": raw_manifest["semantic"],
            "files": raw_manifest["files"],
        },
        "scope": {
            "selection": "nested temporal validation from existing model report; no retuning on the same outer holdout",
            "diagnostic": "2024+ package audit for calibration curves, ablation and rho sensitivity",
            "training_rows": int(model_report.get("training_rows", len(training))),
            "diagnostic_rows": int(len(diagnostic)),
        },
        "policy": {
            "classifier_weight": round_float(policy["classifier_weight"], 4),
            "poisson_weight": round_float(policy["poisson_weight"], 4),
            "draw_floor": round_float(policy["draw_floor"], 4),
            "draw_ceiling": round_float(policy["draw_ceiling"], 4),
            "manual_blend_weights": runtime_weights,
            "draw_xgb": "removed_zero_weight_model",
            "selected_by": policy.get("selected_by"),
        },
        "external_elo_audit": model_report.get("external_elo_audit", {}),
        "nested_temporal": {
            "version": nested.get("version"),
            "aggregate": nested.get("aggregate", {}),
            "selected_policy": selected,
            "component_ablation": nested.get("component_ablation", {}),
        },
        "calibration": {
            "runtime_2024_plus_metrics": policy_metrics,
            "runtime_2024_plus_classification_metrics": runtime_metrics,
            "bins_summary": calibration_summary,
            "class_summary": class_calibration_report,
            "class_summary_rows": class_calibration.to_dict(orient="records"),
            "bootstrap_95pct": bootstrap,
        },
        "uncertainty_intervals": {
            "method": "binomial normal approximation for champion/stage odds; row bootstrap and block bootstrap for diagnostic 1X2 metrics",
            "champion_top_16": uncertainty.to_dict(orient="records"),
            "stage_top_32": stage_intervals.head(32).to_dict(orient="records"),
            "metric_bootstrap": bootstrap,
            "block_bootstrap": block_bootstrap_summary,
            "block_bootstrap_rows": block_bootstrap.to_dict(orient="records"),
        },
        "ablation_study": {
            "method": "complete subset ablation across all 63 non-empty component subsets, plus runtime policy and single-signal stress tests",
            "best": jsonable(best_ablation),
            "full_policy": jsonable(full_ablation),
            "full_policy_objective_gap_vs_best": round_float(full_objective_gap, 6),
            "rows": ablation.to_dict(orient="records"),
        },
        "draw_specific_calibration": draw_scan,
        "dixon_coles": {
            "package_rho": package_rho,
            "best_rho": jsonable(best_dc),
            "package_rho_metrics": jsonable(package_dc_row),
            "package_rho_objective_gap_vs_best": round_float(rho_objective_gap, 6),
            "rows": dc.to_dict(orient="records"),
        },
        "advanced_calibration": advanced_calibration_summary,
        "dixon_coles_team_strength": team_strength_dc_summary,
        "internal_frontier_experiments": {
            "path": str(FRONTIER_CSV),
            "rows": int(len(frontier)),
            "promoted_candidates": int(frontier["promoted"].fillna(False).sum()),
            "rule": "promote only if a no-external-data candidate wins materially without hurting log_loss/draw calibration; otherwise document and keep runtime KISS",
        },
        "runtime_adjustment_audit": adjustment_summary,
        "monte_carlo_stability": mc_stability,
        "external_benchmark": benchmark,
        "recommendations": [],
        "verdict": {
            "sota_kiss": True,
            "reason": "Sem modelo de empate inativo, seleção nested temporal conjunta para componentes e política, gap de empate controlado, Poisson preservando variância de futebol e sensibilidade de rho/ablação perto da fronteira.",
        },
    }
    report["academic_stamp"] = academic_stamp(report, benchmark, full_objective_gap, rho_objective_gap)
    if not report["academic_stamp"]["approved"]:
        raise AssertionError(f"academic SOTA/KISS stamp failed: {report['academic_stamp']}")
    REPORT_JSON.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, ablation, uncertainty, dc)
    log_step(start, f"wrote {REPORT_JSON.name} and CSV diagnostics")
    return report


def main() -> None:
    report = run()
    policy = report["policy"]
    metrics = report["calibration"]["runtime_2024_plus_metrics"]
    best = report["dixon_coles"]["best_rho"]
    print(
        "[stats-qa] OK "
        f"policy={policy['classifier_weight']:.2f}/{policy['poisson_weight']:.2f} "
        f"draw={policy['draw_floor']:.2f}-{policy['draw_ceiling']:.2f} "
        f"log_loss={metrics['log_loss']:.4f} draw_gap={metrics['draw_gap']:.4f} "
        f"best_rho={best['rho']:.2f}"
    )


if __name__ == "__main__":
    main()

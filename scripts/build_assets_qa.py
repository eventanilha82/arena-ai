from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / "build" / "release_assets"
ASSET_MANIFEST = ROOT / "assets" / "asset_manifest.json"
FORBIDDEN_PARTS = {
    "candidates",
    "docs",
    "raw",
    "rejected_assets",
    "source",
    "sources",
}
FORBIDDEN_SUFFIXES = (
    "_sources",
    "_source",
)
FORBIDDEN_FILENAMES = {
    "downloaded_audio_manifest.csv",
}
SOTA_PIPELINE_REL = "modeling/worldcup_2026_ml/src/sota_pipeline.py"
MODEL_PACKAGE_REL = "modeling/worldcup_2026_ml/models/model_sota.pkl"
RUNTIME_PREDICTION_CACHE_REL = "modeling/worldcup_2026_ml/models/runtime_prediction_cache.pkl"
MODEL_REPORT_REL = "modeling/worldcup_2026_ml/reports/sota_model_report.json"
MODEL_RUNTIME_FILES = (
    SOTA_PIPELINE_REL,
    MODEL_PACKAGE_REL,
    RUNTIME_PREDICTION_CACHE_REL,
    MODEL_REPORT_REL,
)
MIN_RUNTIME_CACHE_RUNS = 1000


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_forbidden_release_path(path: str) -> bool:
    parts = Path(path).parts
    if any(part in FORBIDDEN_PARTS for part in parts):
        return True
    if any(part.endswith(FORBIDDEN_SUFFIXES) for part in parts):
        return True
    return Path(path).name in FORBIDDEN_FILENAMES


def is_app_payload_path(path: str) -> bool:
    parts = Path(path).parts
    return bool(parts) and parts[0] in {"assets", "docs", "modeling"}


def ensure_stage_path(stage: Path) -> Path:
    candidate = stage if stage.is_absolute() else ROOT / stage
    resolved = candidate.resolve(strict=False)
    build_root = (ROOT / "build").resolve(strict=False)
    try:
        inside_build = resolved.is_relative_to(build_root)
    except AttributeError:
        inside_build = build_root == resolved or build_root in resolved.parents
    if not inside_build or resolved == build_root:
        raise ValueError(f"--stage precisa resolver para um subdiretório de build/: {candidate}")
    return resolved


def ensure_sota_pipeline_importable() -> None:
    pipeline_dir = ROOT / "modeling" / "worldcup_2026_ml" / "src"
    pipeline_dir_text = str(pipeline_dir)
    if pipeline_dir_text not in sys.path:
        sys.path.insert(0, pipeline_dir_text)


def load_pickle_payload(data: bytes, label: str) -> object:
    ensure_sota_pipeline_importable()
    try:
        return pickle.loads(data)
    except Exception as exc:
        raise AssertionError(f"cache runtime inválido ou ilegível em {label}: {exc}") from exc


def validate_runtime_prediction_cache_bytes(cache_data: bytes, model_sha256: str, pipeline_sha256: str, label: str) -> str:
    if len(cache_data) < 1024:
        raise AssertionError(f"cache runtime pequeno demais em {label}: {len(cache_data)} bytes")
    payload = load_pickle_payload(cache_data, label)
    if not isinstance(payload, dict):
        raise AssertionError(f"cache runtime precisa ser dict em {label}, veio {type(payload)!r}")

    expected = {
        "model_sha256": model_sha256,
        "sota_pipeline_sha256": pipeline_sha256,
    }
    for key, current_hash in expected.items():
        cached_hash = str(payload.get(key, ""))
        if cached_hash != current_hash:
            raise AssertionError(
                f"cache runtime stale em {label}: {key}={cached_hash[:12]} != {current_hash[:12]}; rode make runtime-cache"
            )

    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        raise AssertionError(f"cache runtime sem generated_at em {label}")
    runs = int(payload.get("runs", 0) or 0)
    if runs <= 0:
        raise AssertionError(f"cache runtime sem runs válido em {label}: {payload.get('runs')!r}")
    if runs < MIN_RUNTIME_CACHE_RUNS:
        raise AssertionError(
            f"cache runtime pequeno demais em {label}: runs={runs} < {MIN_RUNTIME_CACHE_RUNS}; rode make runtime-cache"
        )

    prediction_cache = payload.get("prediction_cache")
    prediction_base_cache = payload.get("prediction_base_cache")
    scenario_bank = payload.get("scenario_bank")
    if not isinstance(prediction_cache, dict) or not prediction_cache:
        raise AssertionError(f"cache runtime sem prediction_cache útil em {label}")
    if not isinstance(prediction_base_cache, dict) or not prediction_base_cache:
        raise AssertionError(f"cache runtime sem prediction_base_cache útil em {label}")
    if not isinstance(scenario_bank, list) or not scenario_bank:
        raise AssertionError(f"cache runtime sem scenario_bank útil em {label}")
    if len(scenario_bank) < runs:
        raise AssertionError(
            f"cache runtime com scenario_bank incompleto em {label}: scenario_bank={len(scenario_bank)} < runs={runs}; "
            "rode make runtime-cache"
        )

    return (
        f"runtime_prediction_cache.pkl OK: runs={runs}, "
        f"prediction_cache={len(prediction_cache)}, "
        f"base_cache={len(prediction_base_cache)}, "
        f"scenario_bank={len(scenario_bank)}"
    )


def validate_runtime_prediction_cache_files(cache_path: Path, model_path: Path, pipeline_path: Path, label: str) -> str:
    for path in (cache_path, model_path, pipeline_path):
        if not path.exists():
            raise FileNotFoundError(f"arquivo obrigatório para validar cache ausente: {path}")
    return validate_runtime_prediction_cache_bytes(
        cache_path.read_bytes(),
        sha256_file(model_path),
        sha256_file(pipeline_path),
        label,
    )


def copy_file(source: Path, stage: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"asset de release ausente: {rel(source)}")
    target = stage / rel(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def manifest_paths() -> set[str]:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    paths: set[str] = set()

    for values in manifest.get("used_runtime_assets", {}).values():
        for item in values:
            path = str(item)
            if not path.startswith("assets/") or is_forbidden_release_path(path):
                raise AssertionError(f"used_runtime_assets contains non-runtime release payload: {path}")
            paths.add(path)

    for pattern in manifest.get("generated_runtime_globs", []):
        pattern = str(pattern)
        if not pattern.startswith("assets/"):
            continue
        if is_forbidden_release_path(pattern):
            raise AssertionError(f"runtime glob points at a non-release source path: {pattern}")
        for match in ROOT.glob(pattern):
            if match.is_file():
                path = rel(match)
                if not is_forbidden_release_path(path):
                    paths.add(path)
    return paths


def audio_manifest_paths() -> set[str]:
    sys.path.insert(0, str(ROOT / "src"))
    from arena_ai.audio_manifest import AUDIO_RUNTIME_FILES

    return {f"assets/sounds/runtime_assets/{filename}" for filename in AUDIO_RUNTIME_FILES}


def required_release_paths() -> set[str]:
    paths = manifest_paths()
    paths.update(audio_manifest_paths())
    paths.update(MODEL_RUNTIME_FILES)
    return paths


def git_tracked_paths(paths: set[str]) -> set[str]:
    if not paths:
        return set()
    if not (ROOT / ".git").exists():
        # Remote Windows builds receive a source tarball without .git. At that
        # point the local repo already enforced staging; the remote gate should
        # validate the payload it received instead of failing on git metadata.
        return {path for path in paths if (ROOT / path).exists()}
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", *sorted(paths)],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def validate_required_audio_assets_tracked(paths: set[str]) -> None:
    audio_paths = {path for path in paths if path.startswith("assets/sounds/runtime_assets/")}
    missing = sorted(audio_paths - git_tracked_paths(audio_paths))
    if missing:
        raise AssertionError(
            "runtime audio assets required for release are not tracked by git:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nStage/version these runtime files before running release/build-assets QA."
        )


def validate_stage(stage: Path) -> list[str]:
    validate_required_audio_assets_tracked(required_release_paths())
    if not stage.exists():
        raise FileNotFoundError(f"staging de assets de release não existe: {stage}")
    files = sorted(path for path in stage.rglob("*") if path.is_file())
    if not files:
        raise AssertionError(f"staging de assets de release está vazio: {stage}")

    violations = [path.relative_to(stage).as_posix() for path in files if is_forbidden_release_path(path.relative_to(stage).as_posix())]
    if violations:
        raise AssertionError("assets brutos proibidos entraram no bundle de release:\n" + "\n".join(f"  - {path}" for path in violations))
    cache_status = validate_runtime_prediction_cache_files(
        stage / RUNTIME_PREDICTION_CACHE_REL,
        stage / MODEL_PACKAGE_REL,
        stage / SOTA_PIPELINE_REL,
        str(stage),
    )
    print(f"[build-assets-qa] {cache_status}")
    return [path.relative_to(stage).as_posix() for path in files]


def stage_release_assets(stage: Path) -> list[str]:
    stage = ensure_stage_path(stage)
    paths = required_release_paths()
    validate_required_audio_assets_tracked(paths)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    for path in sorted(paths):
        if is_forbidden_release_path(path):
            raise AssertionError(f"caminho proibido selecionado para release: {path}")
        copy_file(ROOT / path, stage)
    return validate_stage(stage)


def zip_release_name(name: str) -> str:
    normalized = name.replace("\\", "/").lstrip("./")
    if normalized.startswith("_internal/"):
        normalized = normalized[len("_internal/") :]
    return normalized.rstrip("/")


def validate_zip_artifact(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"zip Windows não encontrado: {path}")
    with zipfile.ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if not infos:
            raise AssertionError(f"zip Windows vazio: {path}")
        rel_to_name = {zip_release_name(info.filename): info.filename for info in infos}
        violations = sorted(
            release_path for release_path in rel_to_name if is_app_payload_path(release_path) and is_forbidden_release_path(release_path)
        )
        if violations:
            raise AssertionError(
                "assets brutos proibidos entraram no zip Windows:\n" + "\n".join(f"  - {item}" for item in violations)
            )
        missing = [item for item in MODEL_RUNTIME_FILES if item not in rel_to_name]
        if missing:
            raise AssertionError(f"zip Windows sem arquivos obrigatórios para validar cache: {missing}")
        cache_status = validate_runtime_prediction_cache_bytes(
            archive.read(rel_to_name[RUNTIME_PREDICTION_CACHE_REL]),
            sha256_bytes(archive.read(rel_to_name[MODEL_PACKAGE_REL])),
            sha256_bytes(archive.read(rel_to_name[SOTA_PIPELINE_REL])),
            str(path),
        )
    print(f"[build-assets-qa] {cache_status}")
    return sorted(rel_to_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage and validate release assets for PyInstaller.")
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--check", action="store_true", help="Only validate an existing staged bundle.")
    parser.add_argument("--check-zip", type=Path, help="Validate a Windows release zip artifact.")
    args = parser.parse_args()

    if args.check_zip is not None:
        files = validate_zip_artifact(args.check_zip if args.check_zip.is_absolute() else ROOT / args.check_zip)
        print(f"[build-assets-qa] zip Windows OK: {len(files)} arquivos em {args.check_zip}")
        return 0

    stage = ensure_stage_path(args.stage)
    files = validate_stage(stage) if args.check else stage_release_assets(stage)
    print(f"[build-assets-qa] bundle de release OK: {len(files)} arquivos em {stage.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import plistlib
import shutil
import stat
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterable
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath


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
APP_PAYLOAD_PARTS = {
    "assets",
    "docs",
    "modeling",
}
MAC_APP_PAYLOAD_ROOTS = (
    "Contents/Resources",
    "Contents/Frameworks/_internal",
)
WINDOWS_LAUNCHER_NAME = "ArenaAI.exe"
SOTA_PIPELINE_REL = "modeling/worldcup_2026_ml/src/sota_pipeline.py"
MODEL_PACKAGE_REL = "modeling/worldcup_2026_ml/models/model_sota.pkl"
RUNTIME_PREDICTION_CACHE_REL = "modeling/worldcup_2026_ml/models/runtime_prediction_cache.pkl"
MODEL_REPORT_REL = "modeling/worldcup_2026_ml/reports/sota_model_report.json"
OBSERVED_GROUP_RESULTS_REL = "modeling/worldcup_2026_ml/data/observed/worldcup_2026_group_stage_results.csv"
OBSERVED_SNAPSHOT_METADATA_REL = "modeling/worldcup_2026_ml/data/observed/worldcup_2026_group_stage_snapshot.json"
MODEL_RUNTIME_FILES = (
    SOTA_PIPELINE_REL,
    MODEL_PACKAGE_REL,
    RUNTIME_PREDICTION_CACHE_REL,
    MODEL_REPORT_REL,
    OBSERVED_GROUP_RESULTS_REL,
    OBSERVED_SNAPSHOT_METADATA_REL,
)
MIN_RUNTIME_CACHE_RUNS = 1000
NEUTRAL_CACHE_TOLERANCE = 1e-10


def mirrored_neutral_cache_key(key: object, *, prediction: bool) -> tuple[object, ...] | None:
    expected_length = 5 if prediction else 4
    if not isinstance(key, tuple) or len(key) != expected_length or not bool(key[2]):
        return None
    if not prediction:
        return (key[1], key[0], key[2], key[3])
    if not isinstance(key[4], tuple):
        return None
    mirrored_context: list[tuple[object, object]] = []
    for item in key[4]:
        if not isinstance(item, tuple) or len(item) != 2:
            return None
        name, value = item
        if isinstance(name, str) and name.startswith("home_"):
            name = f"away_{name[5:]}"
        elif isinstance(name, str) and name.startswith("away_"):
            name = f"home_{name[5:]}"
        mirrored_context.append((name, value))
    context_key = tuple(sorted(mirrored_context, key=lambda item: str(item[0])))
    return (key[1], key[0], key[2], key[3], context_key)


def incomplete_neutral_cache_entries(cache: dict[object, object], *, prediction: bool) -> int:
    expected_length = 5 if prediction else 4
    count = 0
    for key in cache:
        if not isinstance(key, tuple) or len(key) != expected_length or not bool(key[2]):
            continue
        mirror = mirrored_neutral_cache_key(key, prediction=prediction)
        if mirror is None or mirror not in cache:
            count += 1
    return count


def neutral_cache_max_delta(cache: dict[object, object], *, prediction: bool) -> float:
    max_delta = 0.0
    for key, value in cache.items():
        mirror_key = mirrored_neutral_cache_key(key, prediction=prediction)
        if mirror_key is None or mirror_key not in cache:
            continue
        mirror = cache[mirror_key]
        if not isinstance(value, dict) or not isinstance(mirror, dict):
            return float("inf")
        if prediction:
            deltas = (
                abs(float(value["p_home_win_90"]) - float(mirror["p_away_win_90"])),
                abs(float(value["p_draw_90"]) - float(mirror["p_draw_90"])),
                abs(float(value["p_away_win_90"]) - float(mirror["p_home_win_90"])),
                abs(float(value["home_xg"]) - float(mirror["away_xg"])),
                abs(float(value["away_xg"]) - float(mirror["home_xg"])),
            )
        else:
            probabilities = tuple(float(item) for item in value["probs_90"])
            mirror_probabilities = tuple(float(item) for item in mirror["probs_90"])
            if len(probabilities) != 3 or len(mirror_probabilities) != 3:
                return float("inf")
            deltas = (
                abs(probabilities[0] - mirror_probabilities[2]),
                abs(probabilities[1] - mirror_probabilities[1]),
                abs(probabilities[2] - mirror_probabilities[0]),
            )
        max_delta = max(max_delta, *deltas)
    return max_delta


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


def portable_path_parts(path: str) -> tuple[str, ...]:
    return PurePosixPath(path.replace("\\", "/")).parts


def logical_release_key(path: str) -> str:
    return "/".join(part.casefold() for part in portable_path_parts(path))


def is_forbidden_release_path(path: str) -> bool:
    parts = tuple(part.casefold() for part in portable_path_parts(path))
    if any(part in FORBIDDEN_PARTS for part in parts):
        return True
    if any(part.endswith(FORBIDDEN_SUFFIXES) for part in parts):
        return True
    return bool(parts) and parts[-1] in FORBIDDEN_FILENAMES


def is_app_payload_path(path: str) -> bool:
    parts = portable_path_parts(path)
    return bool(parts) and parts[0].casefold() in APP_PAYLOAD_PARTS


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
    incomplete_predictions = incomplete_neutral_cache_entries(prediction_cache, prediction=True)
    incomplete_bases = incomplete_neutral_cache_entries(prediction_base_cache, prediction=False)
    if incomplete_predictions or incomplete_bases:
        raise AssertionError(
            f"cache runtime neutro incompleto em {label}: prediction={incomplete_predictions}, "
            f"base={incomplete_bases}; rode make runtime-cache"
        )
    prediction_delta = neutral_cache_max_delta(prediction_cache, prediction=True)
    base_delta = neutral_cache_max_delta(prediction_base_cache, prediction=False)
    if prediction_delta > NEUTRAL_CACHE_TOLERANCE or base_delta > NEUTRAL_CACHE_TOLERANCE:
        raise AssertionError(
            f"cache runtime neutro assimétrico em {label}: prediction={prediction_delta:.3e}, "
            f"base={base_delta:.3e}; rode make runtime-cache"
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

    classified_runtime_patterns = [str(pattern) for pattern in manifest.get("generated_runtime_globs", [])]
    runtime_patterns = [str(pattern) for pattern in manifest.get("release_runtime_globs", classified_runtime_patterns)]
    if set(classified_runtime_patterns) != set(runtime_patterns):
        raise AssertionError(
            "generated_runtime_globs must be the canonical release runtime inventory"
        )
    expected_counts_raw = manifest.get("release_runtime_glob_expected_counts")
    if not isinstance(expected_counts_raw, dict):
        raise AssertionError("asset manifest must declare release_runtime_glob_expected_counts")
    expected_counts = {str(pattern): int(count) for pattern, count in expected_counts_raw.items()}
    if set(runtime_patterns) != set(expected_counts):
        missing_counts = sorted(set(runtime_patterns) - set(expected_counts))
        stale_counts = sorted(set(expected_counts) - set(runtime_patterns))
        details = []
        if missing_counts:
            details.append("globs sem cardinalidade:\n" + "\n".join(f"  - {pattern}" for pattern in missing_counts))
        if stale_counts:
            details.append("cardinalidades sem glob ativo:\n" + "\n".join(f"  - {pattern}" for pattern in stale_counts))
        raise AssertionError("inventário de globs runtime inconsistente:\n" + "\n".join(details))

    for pattern in runtime_patterns:
        if not pattern.startswith("assets/"):
            continue
        if is_forbidden_release_path(pattern):
            raise AssertionError(f"runtime glob points at a non-release source path: {pattern}")
        matches = sorted(match for match in ROOT.glob(pattern) if match.is_file())
        expected_count = expected_counts[pattern]
        if len(matches) != expected_count:
            raise AssertionError(
                f"runtime glob cardinality mismatch: {pattern}={len(matches)} files, expected {expected_count}"
            )
        for match in matches:
            path = rel(match)
            if not any(fnmatch(path, classified_pattern) for classified_pattern in classified_runtime_patterns):
                raise AssertionError(f"release runtime asset is not classified by generated_runtime_globs: {path}")
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


def logical_path_index(paths: Iterable[str], label: str) -> dict[str, str]:
    entries: dict[str, list[str]] = {}
    for path in paths:
        entries.setdefault(logical_release_key(path), []).append(path)
    collisions = {
        key: values
        for key, values in entries.items()
        if len(values) != 1
    }
    if collisions:
        details = "\n".join(
            f"  - {key}: {values}"
            for key, values in sorted(collisions.items())
        )
        raise AssertionError(
            f"{label} contém caminhos duplicados após normalização case-insensitive:\n"
            + details
        )
    return {key: values[0] for key, values in entries.items()}


def validate_release_inventory(
    available_paths: Iterable[str],
    label: str,
) -> dict[str, str]:
    available_index = logical_path_index(available_paths, label)
    required_index = logical_path_index(required_release_paths(), "inventário canônico")
    missing_keys = sorted(set(required_index) - set(available_index))
    missing = [required_index[key] for key in missing_keys]
    if missing:
        raise AssertionError(
            f"payload de release incompleto em {label}:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    violations = sorted(
        path
        for path in available_index.values()
        if is_forbidden_release_path(path)
    )
    if violations:
        raise AssertionError(
            f"payload de release contém fontes/curadoria proibidas em {label}:\n"
            + "\n".join(f"  - {path}" for path in violations)
        )
    extra_keys = sorted(
        key
        for key, path in available_index.items()
        if is_app_payload_path(path) and key not in required_index
    )
    extras = [available_index[key] for key in extra_keys]
    if extras:
        raise AssertionError(
            f"payload de release contém arquivos fora do inventário canônico em {label}:\n"
            + "\n".join(f"  - {path}" for path in extras)
        )
    return {
        required_path: available_index[key]
        for key, required_path in required_index.items()
    }


def validate_release_payload_bytes(
    required_to_embedded: dict[str, str],
    read_embedded: Callable[[str], bytes],
    label: str,
) -> None:
    drifts = []
    for source_relative, embedded_relative in sorted(required_to_embedded.items()):
        source_path = ROOT / source_relative
        if (
            not source_path.is_file()
            or sha256_bytes(read_embedded(embedded_relative)) != sha256_file(source_path)
        ):
            drifts.append(f"{embedded_relative} != {source_relative}")
    if drifts:
        raise AssertionError(
            f"payload de release diverge dos bytes do source em {label}:\n"
            + "\n".join(f"  - {path}" for path in drifts)
        )


def git_tracked_paths(paths: set[str]) -> set[str]:
    if not paths:
        return set()
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--", *sorted(paths)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AssertionError("git is required by --require-git-tracked but was not found") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise AssertionError(
            "--require-git-tracked needs a Git worktree; use the iterative asset QA on source bundles without .git: "
            f"{detail}"
        )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def validate_required_runtime_assets_tracked(paths: set[str]) -> int:
    runtime_asset_paths = {path for path in paths if path.startswith("assets/")}
    missing = sorted(runtime_asset_paths - git_tracked_paths(runtime_asset_paths))
    if missing:
        raise AssertionError(
            "active runtime assets required for release are not tracked by git:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nTrack these final runtime files before running the local/CI release gate. "
            "Use build-assets-qa without --require-git-tracked while iterating."
        )
    return len(runtime_asset_paths)


def validate_stage(stage: Path) -> list[str]:
    if not stage.exists():
        raise FileNotFoundError(f"staging de assets de release não existe: {stage}")
    files = sorted(path for path in stage.rglob("*") if path.is_file())
    if not files:
        raise AssertionError(f"staging de assets de release está vazio: {stage}")

    violations = [path.relative_to(stage).as_posix() for path in files if is_forbidden_release_path(path.relative_to(stage).as_posix())]
    if violations:
        raise AssertionError("assets brutos proibidos entraram no bundle de release:\n" + "\n".join(f"  - {path}" for path in violations))
    release_files = [path.relative_to(stage).as_posix() for path in files]
    required_to_staged = validate_release_inventory(release_files, str(stage))
    validate_release_payload_bytes(
        required_to_staged,
        lambda embedded: (stage / embedded).read_bytes(),
        str(stage),
    )
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
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    for path in sorted(paths):
        if is_forbidden_release_path(path):
            raise AssertionError(f"caminho proibido selecionado para release: {path}")
        copy_file(ROOT / path, stage)
    return validate_stage(stage)


def zip_release_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    raw_parts = normalized.rstrip("/").split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or any(part in {"", ".", ".."} for part in raw_parts)
        or (raw_parts and raw_parts[0].endswith(":"))
    ):
        raise AssertionError(f"unsafe path in Windows zip: {name!r}")
    parts = list(PurePosixPath(normalized).parts)
    if parts and parts[0].casefold() == "_internal":
        parts = parts[1:]
    if not parts:
        raise AssertionError(f"empty normalized path in Windows zip: {name!r}")
    return "/".join(parts)


def zip_info_is_regular_file(info: zipfile.ZipInfo) -> bool:
    if info.is_dir():
        return False
    unix_mode = info.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    if file_type:
        return stat.S_ISREG(unix_mode)
    # ZIPs produced on Windows commonly omit Unix file-type bits. A non-directory
    # entry without those bits is the portable representation of a regular file.
    return not bool(info.external_attr & 0x10)


def validate_windows_launcher(
    archive: zipfile.ZipFile,
    infos: Iterable[zipfile.ZipInfo],
    label: str,
) -> zipfile.ZipInfo:
    launchers = [info for info in infos if info.filename == WINDOWS_LAUNCHER_NAME]
    if len(launchers) != 1:
        candidates = sorted(
            info.filename
            for info in infos
            if PurePosixPath(info.filename.replace("\\", "/")).name.casefold()
            == WINDOWS_LAUNCHER_NAME.casefold()
            or WINDOWS_LAUNCHER_NAME.casefold() in info.filename.casefold()
        )
        detail = f"; candidatos encontrados={candidates}" if candidates else ""
        raise AssertionError(
            f"{label} precisa conter exatamente um launcher raiz chamado "
            f"{WINDOWS_LAUNCHER_NAME!r}; encontrados={len(launchers)}{detail}"
        )
    launcher = launchers[0]
    if not zip_info_is_regular_file(launcher):
        raise AssertionError(
            f"launcher Windows precisa ser arquivo regular: {launcher.filename!r}"
        )
    if launcher.file_size <= 0:
        raise AssertionError(
            f"launcher Windows não pode estar vazio: {launcher.filename!r}"
        )
    if not archive.read(launcher):
        raise AssertionError(
            f"launcher Windows não pode estar vazio: {launcher.filename!r}"
        )
    return launcher


def validate_zip_artifact(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"zip Windows não encontrado: {path}")
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise AssertionError(
                f"zip Windows contém entrada corrompida: {corrupt_member!r}"
            )
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if not infos:
            raise AssertionError(f"zip Windows vazio: {path}")
        validate_windows_launcher(archive, infos, str(path))
        normalized_entries: dict[str, list[tuple[str, str]]] = {}
        for info in infos:
            release_name = zip_release_name(info.filename)
            normalized_entries.setdefault(release_name.casefold(), []).append(
                (release_name, info.filename)
            )
        collisions = {
            folded_name: entries
            for folded_name, entries in normalized_entries.items()
            if len(entries) != 1
        }
        if collisions:
            details = "\n".join(
                f"  - {folded_name}: "
                f"{[raw_name for _, raw_name in entries]}"
                for folded_name, entries in sorted(collisions.items())
            )
            raise AssertionError(
                "zip Windows contém caminhos duplicados após normalização "
                "case-insensitive:\n"
                + details
            )
        rel_to_name = {
            entries[0][0]: entries[0][1]
            for entries in normalized_entries.values()
        }
        required_to_zipped = validate_release_inventory(rel_to_name, str(path))
        validate_release_payload_bytes(
            required_to_zipped,
            lambda embedded: archive.read(rel_to_name[embedded]),
            str(path),
        )
        cache_status = validate_runtime_prediction_cache_bytes(
            archive.read(rel_to_name[required_to_zipped[RUNTIME_PREDICTION_CACHE_REL]]),
            sha256_bytes(
                archive.read(rel_to_name[required_to_zipped[MODEL_PACKAGE_REL]])
            ),
            sha256_bytes(
                archive.read(rel_to_name[required_to_zipped[SOTA_PIPELINE_REL]])
            ),
            str(path),
        )
    print(f"[build-assets-qa] {cache_status}")
    return sorted(rel_to_name)


def validate_bundle_executable_name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AssertionError(f"CFBundleExecutable precisa ser string em {label}")
    if (
        not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AssertionError(f"CFBundleExecutable inseguro em {label}: {value!r}")
    return value


def validate_mac_launcher(path: Path) -> tuple[dict[str, object], Path]:
    contents = path / "Contents"
    info_plist = contents / "Info.plist"
    executable_dir = contents / "MacOS"
    if not info_plist.is_file():
        raise AssertionError(f"app macOS sem Contents/Info.plist: {path}")
    try:
        with info_plist.open("rb") as handle:
            plist = plistlib.load(handle)
    except Exception as exc:
        raise AssertionError(f"Contents/Info.plist inválido em {path}: {exc}") from exc
    if not isinstance(plist, dict):
        raise AssertionError(f"Contents/Info.plist precisa conter um dicionário em {path}")
    executable_name = validate_bundle_executable_name(
        plist.get("CFBundleExecutable"),
        str(info_plist),
    )
    if not executable_dir.is_dir():
        raise AssertionError(f"app macOS sem Contents/MacOS: {path}")
    exact_launchers = [
        candidate
        for candidate in executable_dir.iterdir()
        if candidate.name == executable_name
    ]
    if len(exact_launchers) != 1:
        candidates = sorted(candidate.name for candidate in executable_dir.iterdir())
        raise AssertionError(
            f"app macOS precisa do launcher exato Contents/MacOS/{executable_name}; "
            f"encontrados={candidates}"
        )
    launcher = exact_launchers[0]
    launcher_stat = launcher.lstat()
    if not stat.S_ISREG(launcher_stat.st_mode):
        raise AssertionError(f"launcher macOS precisa ser arquivo regular: {launcher}")
    if launcher_stat.st_size <= 0:
        raise AssertionError(f"launcher macOS não pode estar vazio: {launcher}")
    if not launcher_stat.st_mode & 0o111:
        raise AssertionError(f"launcher macOS precisa ser executável: {launcher}")
    return plist, launcher


def mac_app_payload_roots(path: Path) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    resources = path / "Contents" / "Resources"
    if resources.is_dir():
        roots.append(("Contents/Resources", resources))

    frameworks = path / "Contents" / "Frameworks"
    if frameworks.is_dir():
        internal_candidates = [
            candidate
            for candidate in frameworks.iterdir()
            if candidate.is_dir() and candidate.name.casefold() == "_internal"
        ]
        if len(internal_candidates) > 1:
            names = sorted(candidate.name for candidate in internal_candidates)
            raise AssertionError(
                "app macOS contém roots _internal ambíguos em Contents/Frameworks: "
                f"{names}"
            )
        if internal_candidates:
            internal = internal_candidates[0]
            roots.append((f"Contents/Frameworks/{internal.name}", internal))
    return roots


def validate_mac_app(path: Path) -> tuple[list[str], tuple[str, ...]]:
    if not path.is_dir():
        raise FileNotFoundError(f"app macOS não encontrado: {path}")
    validate_mac_launcher(path)

    payload_entries: list[tuple[str, Path]] = []
    used_roots: list[str] = []
    for relative_root, payload_root in mac_app_payload_roots(path):
        root_entries = []
        for embedded_path in sorted(payload_root.rglob("*")):
            if not embedded_path.is_file():
                continue
            embedded_relative = embedded_path.relative_to(payload_root).as_posix()
            root_entries.append((embedded_relative, embedded_path))
        if root_entries:
            used_roots.append(relative_root)
            payload_entries.extend(root_entries)
    if not payload_entries:
        expected = ", ".join(MAC_APP_PAYLOAD_ROOTS)
        raise AssertionError(
            f"app macOS sem payload PyInstaller nos layouts suportados ({expected}): {path}"
        )

    required_to_embedded = validate_release_inventory(
        [release_name for release_name, _ in payload_entries],
        str(path),
    )
    embedded_by_key = {
        logical_release_key(release_name): embedded_path
        for release_name, embedded_path in payload_entries
    }
    validate_release_payload_bytes(
        required_to_embedded,
        lambda embedded: embedded_by_key[logical_release_key(embedded)].read_bytes(),
        str(path),
    )
    cache_status = validate_runtime_prediction_cache_files(
        embedded_by_key[logical_release_key(required_to_embedded[RUNTIME_PREDICTION_CACHE_REL])],
        embedded_by_key[logical_release_key(required_to_embedded[MODEL_PACKAGE_REL])],
        embedded_by_key[logical_release_key(required_to_embedded[SOTA_PIPELINE_REL])],
        str(path),
    )
    print(f"[build-assets-qa] {cache_status}")
    return sorted(release_name for release_name, _ in payload_entries), tuple(used_roots)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage and validate release assets for PyInstaller.")
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--check", action="store_true", help="Only validate an existing staged bundle.")
    parser.add_argument("--check-zip", type=Path, help="Validate a Windows release zip artifact.")
    parser.add_argument("--check-mac-app", type=Path, help="Validate a macOS PyInstaller .app bundle.")
    parser.add_argument(
        "--require-git-tracked",
        action="store_true",
        help="Require every active runtime asset to be tracked by Git (local/CI release gate only).",
    )
    args = parser.parse_args()

    if args.check_zip is not None:
        if args.require_git_tracked:
            parser.error("--require-git-tracked cannot be combined with --check-zip")
        if args.check_mac_app is not None or args.check:
            parser.error("--check-zip cannot be combined with another validation mode")
        files = validate_zip_artifact(args.check_zip if args.check_zip.is_absolute() else ROOT / args.check_zip)
        print(f"[build-assets-qa] zip Windows OK: {len(files)} arquivos em {args.check_zip}")
        return 0

    if args.check_mac_app is not None:
        if args.require_git_tracked:
            parser.error("--require-git-tracked cannot be combined with --check-mac-app")
        if args.check:
            parser.error("--check-mac-app cannot be combined with --check")
        mac_app = args.check_mac_app if args.check_mac_app.is_absolute() else ROOT / args.check_mac_app
        files, roots = validate_mac_app(mac_app)
        print(
            f"[build-assets-qa] app macOS OK: {len(files)} arquivos de payload "
            f"em {args.check_mac_app}; roots={','.join(roots)}"
        )
        return 0

    if args.require_git_tracked:
        tracked_count = validate_required_runtime_assets_tracked(required_release_paths())
        print(f"[build-assets-qa] Git tracking OK: {tracked_count} active runtime assets")

    stage = ensure_stage_path(args.stage)
    files = validate_stage(stage) if args.check else stage_release_assets(stage)
    print(f"[build-assets-qa] bundle de release OK: {len(files)} arquivos em {stage.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

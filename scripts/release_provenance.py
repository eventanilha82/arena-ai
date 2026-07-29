from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
SOURCE_KIND = "arena_release_source"
BUILD_KIND = "arena_build_provenance"
SOURCE_FINGERPRINT_DOMAIN = b"arena-ai-release-source-v1\0"


class ProvenanceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run_git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ProvenanceError(f"git command failed: git {' '.join(args)}") from exc
    return result.stdout.strip()


def source_fingerprint(git_tree: str) -> str:
    return sha256_bytes(SOURCE_FINGERPRINT_DOMAIN + git_tree.encode("ascii"))


def source_identity_from_git(*, require_clean: bool = True) -> dict[str, object]:
    status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    if require_clean and status:
        preview = "\n".join(status.splitlines()[:12])
        raise ProvenanceError(
            "release source must be a clean committed snapshot; pending paths:\n"
            f"{preview}"
        )
    git_head = run_git("rev-parse", "HEAD")
    git_tree = run_git("rev-parse", "HEAD^{tree}")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_KIND,
        "git_head": git_head,
        "git_tree": git_tree,
        "source_fingerprint_sha256": source_fingerprint(git_tree),
        "git_worktree_clean": not bool(status),
    }


def validate_source_identity(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ProvenanceError("source provenance must be a JSON object")
    required = {
        "schema_version",
        "kind",
        "git_head",
        "git_tree",
        "source_fingerprint_sha256",
        "git_worktree_clean",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ProvenanceError(f"source provenance missing fields: {', '.join(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ProvenanceError(
            f"unsupported source provenance schema: {payload['schema_version']}"
        )
    if payload["kind"] != SOURCE_KIND:
        raise ProvenanceError(f"invalid source provenance kind: {payload['kind']}")
    git_head = str(payload["git_head"])
    git_tree = str(payload["git_tree"])
    if len(git_head) != 40 or any(char not in "0123456789abcdef" for char in git_head):
        raise ProvenanceError("source git_head must be a full lowercase 40-char SHA")
    if len(git_tree) != 40 or any(char not in "0123456789abcdef" for char in git_tree):
        raise ProvenanceError("source git_tree must be a full lowercase 40-char SHA")
    expected_fingerprint = source_fingerprint(git_tree)
    if payload["source_fingerprint_sha256"] != expected_fingerprint:
        raise ProvenanceError("source fingerprint does not match git_tree")
    if payload["git_worktree_clean"] is not True:
        raise ProvenanceError("release source provenance is not marked clean")
    return {key: payload[key] for key in sorted(required)}


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read provenance JSON: {path}") from exc


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_source_identity(path: Path) -> dict[str, object]:
    return validate_source_identity(read_json(path))


def canonical_directory_fingerprint(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            payload = target.encode("utf-8")
            digest.update(b"L\0" + relative.encode("utf-8") + b"\0")
            digest.update(str(len(payload)).encode("ascii") + b"\0" + payload + b"\0")
            file_count += 1
            total_size += len(payload)
        elif path.is_file():
            size = path.stat().st_size
            digest.update(b"F\0" + relative.encode("utf-8") + b"\0")
            digest.update(str(size).encode("ascii") + b"\0")
            digest.update(sha256_file(path).encode("ascii") + b"\0")
            file_count += 1
            total_size += size
    if not file_count:
        raise ProvenanceError(f"artifact directory is empty: {root}")
    return digest.hexdigest(), file_count, total_size


def mac_executable(app: Path) -> tuple[str, str, int]:
    plist_path = app / "Contents" / "Info.plist"
    try:
        with plist_path.open("rb") as handle:
            plist = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException) as exc:
        raise ProvenanceError(f"invalid macOS Info.plist: {plist_path}") from exc
    executable_name = plist.get("CFBundleExecutable")
    if not isinstance(executable_name, str) or not executable_name:
        raise ProvenanceError("macOS Info.plist has no CFBundleExecutable")
    executable = app / "Contents" / "MacOS" / executable_name
    if not executable.is_file():
        raise ProvenanceError(f"macOS executable not found: {executable}")
    return (
        executable.relative_to(app).as_posix(),
        sha256_file(executable),
        executable.stat().st_size,
    )


def directory_executable(artifact: Path, app_name: str) -> tuple[str, str, int]:
    candidates = [
        path
        for path in artifact.rglob("*.exe")
        if path.name.casefold() == f"{app_name}.exe".casefold()
    ]
    if len(candidates) != 1:
        raise ProvenanceError(
            f"expected one {app_name}.exe in {artifact}, found {len(candidates)}"
        )
    executable = candidates[0]
    return (
        executable.relative_to(artifact).as_posix(),
        sha256_file(executable),
        executable.stat().st_size,
    )


def zip_executable(artifact: Path, app_name: str) -> tuple[str, str, int]:
    try:
        with ZipFile(artifact) as archive:
            candidates = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and PurePosixPath(info.filename).name.casefold()
                == f"{app_name}.exe".casefold()
            ]
            if len(candidates) != 1:
                raise ProvenanceError(
                    f"expected one {app_name}.exe in {artifact}, found {len(candidates)}"
                )
            info = candidates[0]
            payload = archive.read(info)
    except (OSError, ValueError) as exc:
        raise ProvenanceError(f"invalid Windows ZIP: {artifact}") from exc
    return info.filename, sha256_bytes(payload), len(payload)


def inspect_artifact(
    artifact: Path,
    *,
    platform: str,
    app_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    artifact = artifact.resolve()
    if platform == "macos":
        if not artifact.is_dir():
            raise ProvenanceError(f"macOS artifact must be an .app directory: {artifact}")
        artifact_sha, file_count, total_size = canonical_directory_fingerprint(artifact)
        executable_path, executable_sha, executable_size = mac_executable(artifact)
        artifact_payload = {
            "kind": "macos_app_tree",
            "name": artifact.name,
            "hash_kind": "canonical_tree_sha256",
            "sha256": artifact_sha,
            "file_count": file_count,
            "size_bytes": total_size,
        }
    elif platform == "windows":
        if artifact.is_dir():
            artifact_sha, file_count, total_size = canonical_directory_fingerprint(artifact)
            executable_path, executable_sha, executable_size = directory_executable(
                artifact, app_name
            )
            artifact_payload = {
                "kind": "windows_onedir_tree",
                "name": artifact.name,
                "hash_kind": "canonical_tree_sha256",
                "sha256": artifact_sha,
                "file_count": file_count,
                "size_bytes": total_size,
            }
        elif artifact.is_file():
            executable_path, executable_sha, executable_size = zip_executable(
                artifact, app_name
            )
            with ZipFile(artifact) as archive:
                file_count = len(archive.infolist())
            artifact_payload = {
                "kind": "windows_zip",
                "name": artifact.name,
                "hash_kind": "file_sha256",
                "sha256": sha256_file(artifact),
                "file_count": file_count,
                "size_bytes": artifact.stat().st_size,
            }
        else:
            raise ProvenanceError(f"Windows artifact not found: {artifact}")
    else:
        raise ProvenanceError(f"unsupported platform: {platform}")
    executable_payload = {
        "path": executable_path,
        "sha256": executable_sha,
        "size_bytes": executable_size,
    }
    return artifact_payload, executable_payload


def build_provenance_payload(
    artifact: Path,
    *,
    platform: str,
    app_name: str,
    source: dict[str, object],
) -> dict[str, object]:
    artifact_payload, executable_payload = inspect_artifact(
        artifact, platform=platform, app_name=app_name
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": BUILD_KIND,
        "app_name": app_name,
        "platform": platform,
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": validate_source_identity(source),
        "artifact": artifact_payload,
        "executable": executable_payload,
    }


def validate_build_provenance(
    artifact: Path,
    provenance_path: Path,
    *,
    platform: str,
    app_name: str,
    expected_source: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = read_json(provenance_path)
    if not isinstance(payload, dict):
        raise ProvenanceError("build provenance must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProvenanceError(
            f"unsupported build provenance schema: {payload.get('schema_version')}"
        )
    if payload.get("kind") != BUILD_KIND:
        raise ProvenanceError(f"invalid build provenance kind: {payload.get('kind')}")
    if payload.get("platform") != platform:
        raise ProvenanceError(
            f"build platform mismatch: {payload.get('platform')} != {platform}"
        )
    if payload.get("app_name") != app_name:
        raise ProvenanceError(
            f"build app mismatch: {payload.get('app_name')} != {app_name}"
        )
    recorded_source = validate_source_identity(payload.get("source"))
    if expected_source is not None:
        expected_source = validate_source_identity(expected_source)
        if recorded_source != expected_source:
            raise ProvenanceError(
                "build source fingerprint does not match the current release snapshot"
            )
    artifact_payload, executable_payload = inspect_artifact(
        artifact, platform=platform, app_name=app_name
    )
    if payload.get("artifact") != artifact_payload:
        raise ProvenanceError("artifact bytes do not match build provenance")
    if payload.get("executable") != executable_payload:
        raise ProvenanceError("primary executable does not match build provenance")
    return payload


def self_test() -> None:
    source = {
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_KIND,
        "git_head": "1" * 40,
        "git_tree": "2" * 40,
        "source_fingerprint_sha256": source_fingerprint("2" * 40),
        "git_worktree_clean": True,
    }
    with tempfile.TemporaryDirectory(prefix="arena-provenance-") as temp_value:
        temp = Path(temp_value)
        app = temp / "ArenaAI.app"
        executable = app / "Contents" / "MacOS" / "ArenaAI"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"mac-binary")
        with (app / "Contents" / "Info.plist").open("wb") as handle:
            plistlib.dump({"CFBundleExecutable": "ArenaAI"}, handle)
        mac_result = temp / "mac-build-result.json"
        write_json(
            mac_result,
            build_provenance_payload(
                app, platform="macos", app_name="ArenaAI", source=source
            ),
        )
        validate_build_provenance(
            app,
            mac_result,
            platform="macos",
            app_name="ArenaAI",
            expected_source=source,
        )
        executable.write_bytes(b"tampered")
        try:
            validate_build_provenance(
                app,
                mac_result,
                platform="macos",
                app_name="ArenaAI",
                expected_source=source,
            )
        except ProvenanceError:
            pass
        else:
            raise ProvenanceError("self-test accepted a tampered macOS executable")

        windows_zip = temp / "ArenaAI-windows-latest.zip"
        with ZipFile(windows_zip, "w", ZIP_DEFLATED) as archive:
            archive.writestr("ArenaAI.exe", b"windows-binary")
            archive.writestr("_internal/assets/example.bin", b"asset")
        windows_result = temp / "windows-build-result.json"
        write_json(
            windows_result,
            build_provenance_payload(
                windows_zip, platform="windows", app_name="ArenaAI", source=source
            ),
        )
        validate_build_provenance(
            windows_zip,
            windows_result,
            platform="windows",
            app_name="ArenaAI",
            expected_source=source,
        )
        mismatched_source = {
            **source,
            "git_head": "3" * 40,
        }
        try:
            validate_build_provenance(
                windows_zip,
                windows_result,
                platform="windows",
                app_name="ArenaAI",
                expected_source=mismatched_source,
            )
        except ProvenanceError:
            pass
        else:
            raise ProvenanceError("self-test accepted a build from another commit")
        with windows_zip.open("ab") as handle:
            handle.write(b"tampered")
        try:
            validate_build_provenance(
                windows_zip,
                windows_result,
                platform="windows",
                app_name="ArenaAI",
                expected_source=source,
            )
        except ProvenanceError:
            pass
        else:
            raise ProvenanceError("self-test accepted a tampered Windows ZIP")


def resolve_path(value: Path) -> Path:
    return (ROOT / value).resolve() if not value.is_absolute() else value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create and validate release source/build provenance."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    source_parser = subparsers.add_parser("write-source")
    source_parser.add_argument("--output", required=True, type=Path)

    check_source_parser = subparsers.add_parser("check-source")
    check_source_parser.add_argument("--provenance", type=Path)
    check_source_parser.add_argument("--against-current", action="store_true")

    source_ref_parser = subparsers.add_parser("source-ref")
    source_ref_parser.add_argument("--provenance", required=True, type=Path)

    build_parser = subparsers.add_parser("write-build")
    build_parser.add_argument("--platform", required=True, choices=("macos", "windows"))
    build_parser.add_argument("--artifact", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument("--source-provenance", type=Path)
    build_parser.add_argument("--app-name", default="ArenaAI")

    check_build_parser = subparsers.add_parser("check-build")
    check_build_parser.add_argument(
        "--platform", required=True, choices=("macos", "windows")
    )
    check_build_parser.add_argument("--artifact", required=True, type=Path)
    check_build_parser.add_argument("--provenance", required=True, type=Path)
    check_build_parser.add_argument("--source-provenance", type=Path)
    check_build_parser.add_argument("--app-name", default="ArenaAI")

    subparsers.add_parser("self-test")
    args = parser.parse_args()

    try:
        if args.command == "write-source":
            output = resolve_path(args.output)
            source = source_identity_from_git(require_clean=True)
            write_json(output, source)
            print(
                f"source provenance: {output} "
                f"head={source['git_head']} fingerprint={source['source_fingerprint_sha256']}"
            )
        elif args.command == "check-source":
            if args.provenance:
                source = load_source_identity(resolve_path(args.provenance))
                if args.against_current:
                    current = source_identity_from_git(require_clean=True)
                    if source != current:
                        raise ProvenanceError(
                            "recorded release source no longer matches the current "
                            "clean checkout"
                        )
            else:
                source = source_identity_from_git(require_clean=True)
            print(
                "release source OK: "
                f"head={source['git_head']} tree={source['git_tree']} "
                f"fingerprint={source['source_fingerprint_sha256']}"
            )
        elif args.command == "source-ref":
            source = load_source_identity(resolve_path(args.provenance))
            print(source["git_head"])
        elif args.command == "write-build":
            artifact = resolve_path(args.artifact)
            output = resolve_path(args.output)
            source = (
                load_source_identity(resolve_path(args.source_provenance))
                if args.source_provenance
                else source_identity_from_git(require_clean=True)
            )
            payload = build_provenance_payload(
                artifact,
                platform=args.platform,
                app_name=args.app_name,
                source=source,
            )
            write_json(output, payload)
            print(
                f"build provenance: {output} "
                f"artifact={payload['artifact']['sha256']} "
                f"executable={payload['executable']['sha256']}"
            )
        elif args.command == "check-build":
            artifact = resolve_path(args.artifact)
            provenance = resolve_path(args.provenance)
            expected_source = (
                load_source_identity(resolve_path(args.source_provenance))
                if args.source_provenance
                else source_identity_from_git(require_clean=True)
            )
            payload = validate_build_provenance(
                artifact,
                provenance,
                platform=args.platform,
                app_name=args.app_name,
                expected_source=expected_source,
            )
            print(
                f"build provenance OK: platform={args.platform} "
                f"artifact={payload['artifact']['sha256']} "
                f"source={payload['source']['source_fingerprint_sha256']}"
            )
        else:
            self_test()
            print("release provenance self-test passed")
    except ProvenanceError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

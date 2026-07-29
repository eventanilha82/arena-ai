from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from release_provenance import (
    ProvenanceError,
    source_identity_from_git,
    validate_source_identity,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "ArenaAI"


def run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def assert_same_source(expected: dict[str, object], label: str) -> None:
    current = source_identity_from_git(require_clean=True)
    if validate_source_identity(expected) != current:
        raise ProvenanceError(
            f"release source changed {label}; discard this build and rerun"
        )


def pyinstaller_executable() -> Path:
    scripts_dir = Path(sys.prefix) / (
        "Scripts"
        if sys.platform == "win32"
        else "bin"
    )
    candidates = (
        scripts_dir / "pyinstaller",
        scripts_dir / "pyinstaller.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ProvenanceError(
        f"PyInstaller executable not found beside {sys.executable}; run uv sync --dev"
    )


def extract_snapshot(
    destination: Path,
    *,
    git_head: str,
) -> None:
    archive = destination.parent / "arena-ai-source.tar"
    run(
        [
            "git",
            "archive",
            "--format=tar",
            f"--output={archive}",
            git_head,
        ],
        cwd=ROOT,
    )
    destination.mkdir(parents=True)
    with tarfile.open(archive) as handle:
        handle.extractall(destination, filter="data")


def copy_build_outputs(snapshot: Path, *, platform: str) -> tuple[Path, Path]:
    if platform == "macos":
        relative_artifact = Path("dist") / f"{APP_NAME}.app"
        relative_provenance = Path("dist") / f"{APP_NAME}.app.build-result.json"
    else:
        relative_artifact = Path("dist") / APP_NAME
        relative_provenance = Path("dist") / f"{APP_NAME}.build-result.json"
    source_artifact = snapshot / relative_artifact
    source_provenance = snapshot / relative_provenance
    if not source_artifact.exists() or not source_provenance.is_file():
        raise ProvenanceError(
            "snapshot build did not produce the expected artifact and sidecar"
        )

    output_dir = ROOT / "dist"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    target_artifact = ROOT / relative_artifact
    target_provenance = ROOT / relative_provenance
    shutil.copytree(source_artifact, target_artifact, symlinks=True)
    shutil.copy2(source_provenance, target_provenance)
    return target_artifact, target_provenance


def build(platform: str) -> None:
    host_platform = (
        "windows"
        if sys.platform == "win32"
        else "macos"
        if sys.platform == "darwin"
        else None
    )
    if platform != host_platform:
        raise ProvenanceError(
            f"snapshot build platform {platform} is incompatible with host {sys.platform}"
        )
    source = source_identity_from_git(require_clean=True)
    print(
        "[snapshot-build] captured "
        f"head={source['git_head']} tree={source['git_tree']}"
    )

    run(["make", "release-qa-local"], cwd=ROOT)
    assert_same_source(source, "during local/CI QA")

    with tempfile.TemporaryDirectory(prefix="arena-ai-release-") as temp_value:
        temp = Path(temp_value)
        snapshot = temp / "source"
        provenance = temp / "release-source-provenance.json"
        write_json(provenance, source)
        extract_snapshot(snapshot, git_head=str(source["git_head"]))
        run(
            [
                "make",
                "build-current",
                f"PYTHON={Path(sys.executable)}",
                f"PYINSTALLER={pyinstaller_executable()}",
                f"RELEASE_SOURCE_PROVENANCE={provenance}",
            ],
            cwd=snapshot,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (
                            str(snapshot / "src"),
                            os.environ.get("PYTHONPATH", ""),
                        ),
                    )
                ),
            },
        )
        assert_same_source(source, "while the immutable snapshot was building")
        artifact, sidecar = copy_build_outputs(snapshot, platform=platform)

    print(f"[snapshot-build] artifact={artifact}")
    print(f"[snapshot-build] provenance={sidecar}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Arena AI from an immutable committed Git snapshot."
    )
    parser.add_argument("--platform", choices=("macos", "windows"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            python = Path(sys.executable)
            if not python.is_file():
                raise ProvenanceError(f"active Python executable not found: {python}")
            pyinstaller = pyinstaller_executable()
            print(
                "snapshot build toolchain OK: "
                f"python={python} pyinstaller={pyinstaller}"
            )
        elif args.platform:
            build(args.platform)
        else:
            parser.error("--platform is required unless --self-test is used")
    except (ProvenanceError, subprocess.CalledProcessError, OSError, tarfile.TarError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

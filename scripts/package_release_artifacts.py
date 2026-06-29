from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
RELEASE_SOURCE_INTEGRITY_PATHS = (
    "src/arena_ai/main.py",
    "src/arena_ai/worldcup_model.py",
    "modeling/worldcup_2026_ml/src/sota_pipeline.py",
    "modeling/worldcup_2026_ml/models/model_sota.pkl",
    "modeling/worldcup_2026_ml/models/runtime_prediction_cache.pkl",
    "modeling/worldcup_2026_ml/reports/sota_model_report.json",
    "Makefile",
    "scripts/build_assets_qa.py",
    "scripts/package_release_artifacts.py",
    "pyproject.toml",
    "uv.lock",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def zip_directory(source: Path, target: Path) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    base = source.parent
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_dir():
                continue
            relative_path = path.relative_to(base)
            info = ZipInfo.from_file(path, relative_path.as_posix())
            info.external_attr = (path.stat().st_mode & 0o777) << 16
            archive.write(path, info.filename)


def zip_mac_app(mac_app: Path, target: Path) -> str:
    if sys.platform == "darwin" and shutil.which("ditto"):
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(mac_app), str(target)],
            check=True,
        )
        return "ditto"
    zip_directory(mac_app, target)
    return "zipfile"


def file_entry(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def source_integrity() -> dict[str, object]:
    status = run_text(["git", "status", "--porcelain=v1"])
    files = []
    for relative_path in RELEASE_SOURCE_INTEGRITY_PATHS:
        path = ROOT / relative_path
        if not path.is_file():
            raise SystemExit(f"missing release source file: {path}")
        files.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "git_head": run_text(["git", "rev-parse", "HEAD"]),
        "git_branch": run_text(["git", "branch", "--show-current"]),
        "git_worktree_clean": not bool(status),
        "git_status_porcelain": status.splitlines() if status else [],
        "files": files,
    }


def write_release_metadata(
    out: Path,
    *,
    app_name: str,
    artifacts: list[dict[str, object]],
    mac_zip_method: str,
) -> None:
    manifest = {
        "app_name": app_name,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_head": run_text(["git", "rev-parse", "--short", "HEAD"]),
        "git_branch": run_text(["git", "branch", "--show-current"]),
        "release_scope": "macos_windows_app",
        "source_integrity": source_integrity(),
        "mac_zip_method": mac_zip_method,
        "artifacts": artifacts,
    }
    manifest_path = out / "release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_lines = [f"{entry['sha256']}  {Path(str(entry['path'])).name}" for entry in artifacts]
    checksum_lines.append(f"{sha256(manifest_path)}  {manifest_path.name}")
    (out / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    for entry in artifacts:
        print(f"{entry['role']}: {entry['path']} {entry['size_bytes']} bytes {entry['sha256']}")
    print(f"manifest: {manifest_path.relative_to(ROOT)}")
    print(f"checksums: {(out / 'SHA256SUMS').relative_to(ROOT)}")


def resolve_path(value: Path) -> Path:
    return (ROOT / value).resolve() if not value.is_absolute() else value


def main() -> int:
    parser = argparse.ArgumentParser(description="Package Arena AI Mac/Windows release artifacts.")
    parser.add_argument("--app-name", default="ArenaAI")
    parser.add_argument("--mac-app", required=True, type=Path)
    parser.add_argument("--windows-zip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    mac_app = resolve_path(args.mac_app)
    windows_zip = resolve_path(args.windows_zip)
    out = resolve_path(args.out)
    if not mac_app.is_dir():
        raise SystemExit(f"missing mac app bundle: {mac_app}")
    if not windows_zip.is_file():
        raise SystemExit(f"missing windows zip: {windows_zip}")

    out.mkdir(parents=True, exist_ok=True)
    mac_zip = out / f"{args.app_name}-mac-latest.zip"
    windows_release_zip = out / f"{args.app_name}-windows-latest.zip"
    for artifact in (mac_zip, windows_release_zip):
        if artifact.exists():
            artifact.unlink()

    mac_zip_method = zip_mac_app(mac_app, mac_zip)
    shutil.copy2(windows_zip, windows_release_zip)
    artifacts = [
        file_entry(mac_zip, "mac_app_zip"),
        file_entry(windows_release_zip, "windows_app_zip"),
    ]
    write_release_metadata(out, app_name=args.app_name, artifacts=artifacts, mac_zip_method=mac_zip_method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

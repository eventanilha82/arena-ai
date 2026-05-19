#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
env_file="${ARENA_WIN_ENV_FILE:-$script_dir/.env}"

if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$env_file"
  set +a
fi

: "${ARENA_WIN_HOST:?Set ARENA_WIN_HOST in win/.env}"
ARENA_WIN_USER="${ARENA_WIN_USER:-opc}"
ARENA_WIN_REMOTE_ROOT="${ARENA_WIN_REMOTE_ROOT:-C:/arena-ai}"
ARENA_WIN_QA="${ARENA_WIN_QA:-smoke}"
case "$ARENA_WIN_QA" in
  none|smoke|validate|aaa) ;;
  *)
    printf "ARENA_WIN_QA inválido: %s (use none, smoke, validate ou aaa)\n" "$ARENA_WIN_QA" >&2
    exit 1
    ;;
esac

ssh_dest="${ARENA_WIN_USER}@${ARENA_WIN_HOST}"
ssh_opts=(
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$script_dir/known_hosts"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)

if [[ -f "$script_dir/id_ed25519_oci_win_build" ]]; then
  ssh_opts+=(-i "$script_dir/id_ed25519_oci_win_build")
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
tmp_dir="${TMPDIR:-/tmp}/arena-ai-win-build-$timestamp"
bundle="$tmp_dir/arena-ai-source.tar.gz"
mkdir -p "$tmp_dir"

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

echo "[arena-win-local] Empacotando área de trabalho atual"
COPYFILE_DISABLE=1 tar \
  --exclude "./.git" \
  --exclude "./.venv" \
  --exclude "./.venv*" \
  --exclude "./build" \
  --exclude "./dist" \
  --exclude "./*.spec" \
  --exclude "./artifacts" \
  --exclude "./win/.env" \
  --exclude "./win/id_*" \
  --exclude "./win/known_hosts" \
  --exclude "./win/authorized_keys.pub" \
  --exclude "./win/artifacts" \
  --exclude "./win/logs" \
  -czf "$bundle" \
  -C "$repo_root" \
  .

echo "[arena-win-local] Preparando diretórios remotos em $ssh_dest"
ssh "${ssh_opts[@]}" "$ssh_dest" \
  "powershell -NoProfile -ExecutionPolicy Bypass -Command \"New-Item -ItemType Directory -Force -Path '$ARENA_WIN_REMOTE_ROOT','$ARENA_WIN_REMOTE_ROOT/incoming','$ARENA_WIN_REMOTE_ROOT/out' | Out-Null\""

echo "[arena-win-local] Enviando executor de build"
scp "${ssh_opts[@]}" "$script_dir/build-remote.ps1" "${ssh_dest}:${ARENA_WIN_REMOTE_ROOT}/build-remote.ps1"

echo "[arena-win-local] Enviando bundle de código"
scp "${ssh_opts[@]}" "$bundle" "${ssh_dest}:${ARENA_WIN_REMOTE_ROOT}/incoming/arena-ai-source.tar.gz"

echo "[arena-win-local] Rodando build Windows"
ssh "${ssh_opts[@]}" "$ssh_dest" \
  "powershell -NoProfile -ExecutionPolicy Bypass -File $ARENA_WIN_REMOTE_ROOT/build-remote.ps1 -WorkRoot $ARENA_WIN_REMOTE_ROOT -Qa $ARENA_WIN_QA"

mkdir -p "$script_dir/artifacts"

echo "[arena-win-local] Baixando artefato"
scp "${ssh_opts[@]}" "${ssh_dest}:${ARENA_WIN_REMOTE_ROOT}/out/ArenaAI-windows-latest.zip" "$script_dir/artifacts/ArenaAI-windows-latest.zip"
scp "${ssh_opts[@]}" "${ssh_dest}:${ARENA_WIN_REMOTE_ROOT}/out/build-result.json" "$script_dir/artifacts/build-result.json" || true

echo "[arena-win-local] Pronto: $script_dir/artifacts/ArenaAI-windows-latest.zip"

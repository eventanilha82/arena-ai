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
source_provenance="$tmp_dir/release-source-provenance.json"
mkdir -p "$tmp_dir"

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

echo "[arena-win-local] Fixando snapshot Git antes do preflight"
"$repo_root/.venv/bin/python" "$repo_root/scripts/release_provenance.py" \
  write-source \
  --output "$source_provenance"

echo "[arena-win-local] Rodando preflight local/CI (Git, evidência visual e inventário)"
make -C "$repo_root" release-qa-local
"$repo_root/.venv/bin/python" "$repo_root/scripts/release_provenance.py" \
  check-source \
  --provenance "$source_provenance" \
  --against-current

source_commit="$(
  "$repo_root/.venv/bin/python" "$repo_root/scripts/release_provenance.py" \
    source-ref \
    --provenance "$source_provenance"
)"

echo "[arena-win-local] Empacotando exatamente o commit aprovado"
git -C "$repo_root" archive \
  --format=tar.gz \
  --output="$bundle" \
  "$source_commit"

echo "[arena-win-local] Preparando diretórios remotos em $ssh_dest"
ssh "${ssh_opts[@]}" "$ssh_dest" \
  "powershell -NoProfile -ExecutionPolicy Bypass -Command \"New-Item -ItemType Directory -Force -Path '$ARENA_WIN_REMOTE_ROOT','$ARENA_WIN_REMOTE_ROOT/incoming','$ARENA_WIN_REMOTE_ROOT/out' | Out-Null\""

echo "[arena-win-local] Enviando executor de build"
scp "${ssh_opts[@]}" "$script_dir/build-remote.ps1" "${ssh_dest}:${ARENA_WIN_REMOTE_ROOT}/build-remote.ps1"

echo "[arena-win-local] Enviando bundle de código"
scp "${ssh_opts[@]}" "$bundle" "${ssh_dest}:${ARENA_WIN_REMOTE_ROOT}/incoming/arena-ai-source.tar.gz"
scp "${ssh_opts[@]}" "$source_provenance" "${ssh_dest}:${ARENA_WIN_REMOTE_ROOT}/incoming/release-source-provenance.json"

echo "[arena-win-local] Rodando build Windows"
ssh "${ssh_opts[@]}" "$ssh_dest" \
  "powershell -NoProfile -ExecutionPolicy Bypass -File $ARENA_WIN_REMOTE_ROOT/build-remote.ps1 -WorkRoot $ARENA_WIN_REMOTE_ROOT -Qa $ARENA_WIN_QA"

mkdir -p "$script_dir/artifacts"

echo "[arena-win-local] Baixando artefato"
scp "${ssh_opts[@]}" "${ssh_dest}:${ARENA_WIN_REMOTE_ROOT}/out/ArenaAI-windows-latest.zip" "$script_dir/artifacts/ArenaAI-windows-latest.zip"
scp "${ssh_opts[@]}" "${ssh_dest}:${ARENA_WIN_REMOTE_ROOT}/out/build-result.json" "$script_dir/artifacts/build-result.json"

echo "[arena-win-local] Pronto: artefato e proveniência em $script_dir/artifacts/"

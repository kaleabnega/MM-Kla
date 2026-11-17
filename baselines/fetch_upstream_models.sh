#!/usr/bin/env bash
set -euo pipefail

BASELINES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESTINATION="${UPSTREAM_DIR:-${BASELINES_DIR}/upstream_sources}"

fetch_pinned() {
    local name="$1" url="$2" revision="$3"
    local target="${DESTINATION}/${name}"
    if [[ ! -e "${target}" ]]; then
        git clone --no-checkout "${url}" "${target}"
        git -C "${target}" checkout --detach "${revision}"
    fi
    if [[ ! -d "${target}/.git" || "$(git -C "${target}" rev-parse HEAD)" != "${revision}" ]]; then
        echo "Unexpected ${name} checkout at ${target}" >&2
        exit 2
    fi
    printf '%s %s\n' "${name}" "${target}"
}

mkdir -p "${DESTINATION}"
fetch_pinned DeepKla https://github.com/linDing-group/DeepKla.git d9f0f5ef987cd65f1562f1ba312e5608ddd44f9e
fetch_pinned Auto-Kla https://github.com/tubic/Auto-Kla.git 429f10ea9449dccc4630b670c3a6e734ce33ba56

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
URL=${1:-https://gpu-control.example.com}
URL=${URL%/}
[[ "$URL" == https://* ]] || { echo "public URL must use https://" >&2; exit 2; }
WHEEL=$(find "$ROOT/dist" -maxdepth 1 -type f -name '*.whl' -printf '%f\n' | sort | head -n1)
[[ -n "$WHEEL" ]] || { echo "no wheel found under $ROOT/dist" >&2; exit 2; }

# Fail before touching configuration unless the complete public trust boundary
# is already correct.
"$ROOT/scripts/verify_public_ingress.sh" "$URL"

sudo env NEW_URL="$URL" WHEEL="$WHEEL" python3 - <<'PY'
import os
from pathlib import Path

path = Path("/etc/sparehour/controller.env")
url = os.environ["NEW_URL"].rstrip("/")
wheel = os.environ["WHEEL"]
found = set()
lines = []
for line in path.read_text(encoding="utf-8").splitlines():
    if line.startswith("SPAREHOUR_PUBLIC_URL="):
        line = f"SPAREHOUR_PUBLIC_URL={url}"
        found.add("url")
    elif line.startswith("SPAREHOUR_WHEEL_URL="):
        line = f"SPAREHOUR_WHEEL_URL={url}/bootstrap/{wheel}"
        found.add("wheel")
    lines.append(line)
if found != {"url", "wheel"}:
    raise SystemExit("controller.env is missing SPAREHOUR_PUBLIC_URL or SPAREHOUR_WHEEL_URL")
tmp = path.with_name(path.name + ".tmp")
tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
os.chown(tmp, 0, 0)
os.replace(tmp, path)
PY

sudo systemctl restart sparehour-private.service
systemctl --quiet is-active sparehour-private.service
printf 'activated %s for new provider workers\n' "$URL"

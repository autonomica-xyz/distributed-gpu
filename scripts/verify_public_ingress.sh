#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
URL=${1:-https://gpu-control.example.com}
URL=${URL%/}
WHEEL=$(find "$ROOT/dist" -maxdepth 1 -type f -name '*.whl' -printf '%f\n' | sort | head -n1)
if [[ -z "$WHEEL" ]]; then
  echo "no wheel found under $ROOT/dist" >&2
  exit 2
fi
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

verify_route() {
  local label=$1
  shift
  local -a extra=("$@")
  local status

  echo "=== $label ==="
  printf 'health: '
  curl "${extra[@]}" --fail --silent --show-error --max-time 15 "$URL/healthz"
  echo

  curl "${extra[@]}" --fail --silent --show-error --max-time 120 \
    "$URL/bootstrap/$WHEEL" -o "$TMP"
  local local_sha remote_sha
  local_sha=$(sha256sum "$ROOT/dist/$WHEEL" | awk '{print $1}')
  remote_sha=$(sha256sum "$TMP" | awk '{print $1}')
  [[ "$local_sha" == "$remote_sha" ]] || {
    echo "wheel SHA-256 mismatch: local=$local_sha remote=$remote_sha" >&2
    exit 1
  }
  echo "wheel: $WHEEL sha256=$remote_sha"

  status=$(curl "${extra[@]}" --silent --show-error --output /dev/null \
    --write-out '%{http_code}' --max-time 15 "$URL/v1/workloads")
  [[ "$status" == 401 ]] || {
    echo "expected unauthenticated /v1/workloads=401, got $status" >&2
    exit 1
  }
  echo "unauthenticated API: 401"

  for path in / /dashboard /v1/dashboard/overview /v1/control/deployments; do
    status=$(curl "${extra[@]}" --silent --show-error --output /dev/null \
      --write-out '%{http_code}' --max-time 15 "$URL$path")
    [[ "$status" == 404 ]] || {
      echo "expected public $path=404, got $status" >&2
      exit 1
    }
  done
  echo "private UI/lifecycle routes: 404"

  # A stable HTTPS proxy should suppress FastAPI's interactive schema UI too.
  if [[ "$URL" == https://* ]]; then
    for path in /docs /redoc /openapi.json; do
      status=$(curl "${extra[@]}" --silent --show-error --output /dev/null \
        --write-out '%{http_code}' --max-time 15 "$URL$path")
      [[ "$status" == 404 ]] || {
        echo "expected public $path=404, got $status" >&2
        exit 1
      }
    done
    echo "public API documentation routes: 404"
  fi

  status=$(curl "${extra[@]}" --silent --show-error --output /dev/null \
    --write-out '%{http_code}' --max-time 15 "$URL/bootstrap/secret.txt")
  [[ "$status" == 404 ]] || {
    echo "expected non-wheel bootstrap=404, got $status" >&2
    exit 1
  }
  echo "non-wheel bootstrap: 404"
}

# Verify what this host resolves (often an internal split-DNS proxy address).
verify_route "resolved route"

# Split DNS can make an internal proxy look healthy while Internet workers hit
# a different edge. Resolve independently through public DoH and force curl to
# that address with the original TLS SNI/Host. Activation fails if public DNS
# cannot be independently checked.
if [[ "$URL" == https://* && "${SPAREHOUR_VERIFY_PUBLIC_DNS:-1}" != 0 ]]; then
  authority=${URL#https://}
  authority=${authority%%/*}
  host=${authority%%:*}
  python="$ROOT/.venv/bin/python"
  [[ -x "$python" ]] || python=python3
  dns_json=$(curl --fail --silent --show-error --max-time 15 \
    --get -H 'accept: application/dns-json' \
    --data-urlencode "name=$host" --data-urlencode 'type=A' \
    https://cloudflare-dns.com/dns-query)
  public_ip=$(printf '%s' "$dns_json" | "$python" -c '
import json, sys
answer = json.load(sys.stdin).get("Answer", [])
print(next((row["data"] for row in answer if row.get("type") == 1), ""))
')
  [[ -n "$public_ip" ]] || {
    echo "public DNS returned no A record for $host" >&2
    exit 1
  }
  verify_route "public DNS route ($host -> $public_ip)" \
    --resolve "$host:443:$public_ip"
fi

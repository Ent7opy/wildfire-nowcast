#!/usr/bin/env python3
"""Check data freshness health endpoint and exit with appropriate code.

Used by the ingest_scheduler health check to verify that source data
(FIRMS, weather, terrain, etc.) is fresh according to the configured
stale thresholds.

Exit codes:
  0 – overall_state == "healthy"
  1 – overall_state != "healthy" or endpoint unreachable
"""

import sys
import json
import urllib.request
import urllib.error

ENDPOINT = "http://api:8000/internal/health/data-freshness"
TIMEOUT = 10  # seconds


def main() -> int:
    try:
        req = urllib.request.Request(ENDPOINT, method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            if resp.status != 200:
                print(f"Unexpected status: {resp.status}", file=sys.stderr)
                return 1
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code} {e.reason}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    overall = data.get("overall_state")
    if overall == "healthy":
        print(f"Data freshness healthy: {overall}")
        return 0
    else:
        print(f"Data freshness not healthy: {overall}", file=sys.stderr)
        stale = data.get("stale_sources", [])
        if stale:
            print(f"Stale sources: {stale}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
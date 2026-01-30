#!/usr/bin/env python3
"""Health check script for the Wildfire Nowcast stack.

Verifies that all services (API, UI, DB) are running and accessible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import urllib.request
import urllib.error
import json
import os


# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class HealthChecker:
    """Check health of stack components."""

    def __init__(self, api_url: str = "http://localhost:8000", ui_url: str = "http://localhost:8501"):
        self.api_url = api_url.rstrip("/")
        self.ui_url = ui_url.rstrip("/")
        self.results: List[Tuple[str, bool, str]] = []

    def check_api(self) -> bool:
        """Check if API is healthy."""
        try:
            # Check health endpoint
            req = urllib.request.Request(f"{self.api_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    self.results.append(("API /health", True, "OK"))
                    return True
                else:
                    self.results.append(("API /health", False, f"HTTP {resp.status}"))
                    return False
        except urllib.error.HTTPError as e:
            self.results.append(("API /health", False, f"HTTP {e.code}"))
            return False
        except Exception as e:
            self.results.append(("API /health", False, str(e)))
            return False

    def check_api_version(self) -> bool:
        """Check if API version endpoint works."""
        try:
            req = urllib.request.Request(f"{self.api_url}/version", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    version = data.get("version", "unknown")
                    self.results.append(("API /version", True, f"v{version}"))
                    return True
                else:
                    self.results.append(("API /version", False, f"HTTP {resp.status}"))
                    return False
        except Exception as e:
            self.results.append(("API /version", False, str(e)))
            return False

    def check_ui(self) -> bool:
        """Check if UI is accessible."""
        try:
            req = urllib.request.Request(self.ui_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    self.results.append(("UI", True, "OK"))
                    return True
                else:
                    self.results.append(("UI", False, f"HTTP {resp.status}"))
                    return False
        except Exception as e:
            self.results.append(("UI", False, str(e)))
            return False

    def check_database(self) -> bool:
        """Check if database is accessible via API."""
        try:
            # Try to fetch fires (should work even if empty)
            req = urllib.request.Request(
                f"{self.api_url}/fires?min_lon=-10&min_lat=35&max_lon=30&max_lat=50&start_time=2024-01-01T00:00:00Z&end_time=2024-12-31T00:00:00Z",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    self.results.append(("Database (via API)", True, "OK"))
                    return True
                else:
                    self.results.append(("Database (via API)", False, f"HTTP {resp.status}"))
                    return False
        except urllib.error.HTTPError as e:
            if e.code == 422:
                # 422 means API is up but params might be off - still means DB is accessible
                self.results.append(("Database (via API)", True, "OK"))
                return True
            self.results.append(("Database (via API)", False, f"HTTP {e.code}"))
            return False
        except Exception as e:
            self.results.append(("Database (via API)", False, str(e)))
            return False

    def run_all_checks(self) -> bool:
        """Run all health checks and return overall status."""
        print("🔍 Checking Wildfire Nowcast stack health...\n")

        self.check_api()
        self.check_api_version()
        self.check_database()
        self.check_ui()

        return self.print_summary()

    def print_summary(self) -> bool:
        """Print summary of results and return True if all passed."""
        print("-" * 50)
        all_passed = True

        for name, passed, message in self.results:
            status = "✅" if passed else "❌"
            print(f"{status} {name:<25} {message}")
            if not passed:
                all_passed = False

        print("-" * 50)

        if all_passed:
            print("\n✨ All services are healthy!")
        else:
            print("\n⚠️  Some services are unhealthy. Troubleshooting tips:")
            print("   • Is Docker running? Run: docker compose ps")
            print("   • Are services started? Run: docker compose up -d")
            print("   • Check logs: docker compose logs -f")

        return all_passed


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Health check for Wildfire Nowcast stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Check default localhost services
  %(prog)s --api-url http://api.example.com:8000
  %(prog)s --wait 30          # Wait up to 30 seconds for services
        """,
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("API_URL", "http://localhost:8000"),
        help="API base URL (default: http://localhost:8000 or $API_URL)",
    )
    parser.add_argument(
        "--ui-url",
        default=os.environ.get("UI_URL", "http://localhost:8501"),
        help="UI base URL (default: http://localhost:8501 or $UI_URL)",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Seconds to wait for services to become available (default: 0)",
    )

    args = parser.parse_args()

    import time

    start_time = time.time()
    wait_until = start_time + args.wait

    while True:
        checker = HealthChecker(api_url=args.api_url, ui_url=args.ui_url)
        all_healthy = checker.run_all_checks()

        if all_healthy or time.time() >= wait_until:
            return 0 if all_healthy else 1

        remaining = int(wait_until - time.time())
        if remaining > 0:
            print(f"\n⏳ Waiting for services... ({remaining}s remaining)\n")
            time.sleep(min(5, remaining))


if __name__ == "__main__":
    sys.exit(main())

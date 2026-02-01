"""Print the path to the most recently created denoiser snapshot directory."""

import glob
import os
import sys


def main() -> None:
    pattern = os.path.join("data", "denoiser", "snapshots", "run_*")
    runs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not runs:
        print("ERROR: No snapshots found in data/denoiser/snapshots/", file=sys.stderr)
        sys.exit(1)
    print(runs[0])


if __name__ == "__main__":
    main()

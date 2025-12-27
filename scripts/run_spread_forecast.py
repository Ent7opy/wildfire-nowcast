#!/usr/bin/env python3
"""CLI for running wildfire spread forecasts manually."""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path if needed
# (though usually handled by the environment/uv)
sys.path.append(str(Path(__file__).parent.parent))

from ml.spread.service import run_spread_forecast, SpreadForecastRequest
from ml.spread.contract import DEFAULT_HORIZONS_HOURS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
LOGGER = logging.getLogger("run_spread_forecast_cli")


def main():
    parser = argparse.ArgumentParser(description="Run a wildfire spread forecast for an AOI.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        help="Bounding box: min_lon min_lat max_lon max_lat",
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="Region name (e.g., 'balkans')",
    )
    parser.add_argument(
        "--time",
        type=str,
        help="Forecast reference time (ISO format). Defaults to now.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS_HOURS),
        help=f"Forecast horizons in hours. Defaults to {DEFAULT_HORIZONS_HOURS}",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the output NetCDF file.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, plot the results (requires matplotlib).",
    )

    args = parser.parse_args()

    # 1. Parse time
    if args.time:
        try:
            ref_time = datetime.fromisoformat(args.time)
            if ref_time.tzinfo is None:
                ref_time = ref_time.replace(tzinfo=timezone.utc)
        except ValueError as e:
            LOGGER.error(f"Invalid time format: {e}")
            sys.exit(1)
    else:
        ref_time = datetime.now(timezone.utc)

    # 2. Build request
    request = SpreadForecastRequest(
        region_name=args.region,
        bbox=tuple(args.bbox),
        forecast_reference_time=ref_time,
        horizons_hours=args.horizons,
    )

    # 3. Run forecast
    try:
        LOGGER.info(f"Running forecast for {args.region} at {ref_time}...")
        forecast = run_spread_forecast(request)
    except Exception as e:
        LOGGER.exception(f"Forecast failed: {e}")
        sys.exit(1)

    # 4. Handle output
    LOGGER.info("Forecast completed successfully.")
    
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        forecast.probabilities.to_netcdf(out_path)
        LOGGER.info(f"Saved probabilities to {out_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            n_horizons = len(args.horizons)
            fig, axes = plt.subplots(1, n_horizons, figsize=(5 * n_horizons, 5), squeeze=False)
            
            for i, h in enumerate(args.horizons):
                ax = axes[0, i]
                p = forecast.probabilities.sel(lead_time_hours=h)
                im = p.plot(ax=ax, add_colorbar=False, cmap="YlOrRd", vmin=0, vmax=1)
                ax.set_title(f"Horizon: {h}h")
                plt.colorbar(im, ax=ax, label="Probability")
            
            plt.tight_layout()
            
            if args.output:
                plot_path = Path(args.output).with_suffix(".png")
                plt.savefig(plot_path)
                LOGGER.info(f"Saved plot to {plot_path}")
            else:
                plt.show()
                
        except ImportError:
            LOGGER.warning("matplotlib not found; skipping plot.")
        except Exception as e:
            LOGGER.error(f"Plotting failed: {e}")

    # Summary statistics
    probs = forecast.probabilities.values
    LOGGER.info(
        f"Output stats: min={probs.min():.4f}, max={probs.max():.4f}, "
        f"mean={probs.mean():.4f}, non-zero ratio={(probs > 0).mean():.4f}"
    )


if __name__ == "__main__":
    main()


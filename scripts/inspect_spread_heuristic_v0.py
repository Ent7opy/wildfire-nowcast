"""Visualization script for heuristic spread model v0."""

import os
import sys
from datetime import datetime, timezone
from dataclasses import dataclass

# Add repo root to path to allow imports from ml package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from ml.spread.heuristic_v0 import HeuristicSpreadModelV0, HeuristicSpreadV0Config
from ml.spread.contract import SpreadModelInput

@dataclass(frozen=True)
class MockGrid:
    crs: str = "EPSG:4326"
    cell_size_deg: float = 0.01
    origin_lat: float = 35.0
    origin_lon: float = 5.0
    n_lat: int = 100
    n_lon: int = 100

@dataclass(frozen=True)
class MockWindow:
    lat: np.ndarray
    lon: np.ndarray
    i0: int = 0
    i1: int = 50
    j0: int = 0
    j1: int = 50

@dataclass(frozen=True)
class MockFireHeatmap:
    heatmap: np.ndarray

@dataclass(frozen=True)
class MockTerrain:
    valid_data_mask: np.ndarray | None = None
    aoi_mask: np.ndarray | None = None
    slope: np.ndarray | None = None
    aspect: np.ndarray | None = None

def run_inspection():
    print("Running spread heuristic v0 inspection...")
    
    # Setup AOI
    ny, nx = 101, 101
    lat = np.arange(ny) * 0.01 + 35.0
    lon = np.arange(nx) * 0.01 + 5.0
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)
    
    # Ignition in center
    heatmap = np.zeros((ny, nx))
    heatmap[ny//2, nx//2] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)
    
    horizons = [6, 24, 48]
    ref_time = datetime(2025, 12, 26, tzinfo=timezone.utc)
    
    # Test cases: (name, u10, v10)
    cases = [
        ("no_wind", 0.0, 0.0),
        ("east_wind_10ms", 10.0, 0.0),
        ("northeast_wind_7ms", 5.0, 5.0),
    ]
    
    output_dir = "reports/spread_heuristic_v0"
    os.makedirs(output_dir, exist_ok=True)
    
    model = HeuristicSpreadModelV0(HeuristicSpreadV0Config(
        base_spread_km_h=0.1,
        wind_influence_km_h_per_ms=0.2,
        distance_decay_km=3.0
    ))
    
    for name, u, v in cases:
        print(f"Processing case: {name} (u={u}, v={v})")
        
        weather = xr.Dataset(
            data_vars={
                "u10": (("lat", "lon"), np.ones((ny, nx)) * u),
                "v10": (("lat", "lon"), np.ones((ny, nx)) * v),
            },
            coords={"lat": lat, "lon": lon}
        )
        
        inputs = SpreadModelInput(
            grid=MockGrid(),
            window=window,
            active_fires=fires,
            weather_cube=weather,
            terrain=MockTerrain(),
            forecast_reference_time=ref_time,
            horizons_hours=horizons
        )
        
        forecast = model.predict(inputs)
        
        # Plot
        fig, axes = plt.subplots(1, len(horizons), figsize=(18, 6), sharex=True, sharey=True)
        fig.suptitle(f"Heuristic Spread v0: {name} (u={u}m/s, v={v}m/s)", fontsize=16)
        
        for i, h in enumerate(horizons):
            ax = axes[i]
            probs_da = forecast.probabilities.isel(time=i)
            probs = probs_da.values
            
            # 1. Plot probability heatmap
            im = ax.imshow(
                probs, 
                origin="lower", 
                extent=[lon[0], lon[-1], lat[0], lat[-1]], 
                cmap="YlOrRd", 
                vmin=0, 
                vmax=1,
                alpha=0.8
            )
            
            # 2. Add contour lines for specific thresholds
            thresholds = [0.3, 0.5, 0.7]
            if probs.max() > thresholds[0]:
                ax.contour(
                    lon, lat, probs, 
                    levels=thresholds, 
                    colors=['black'], 
                    linewidths=0.5,
                    alpha=0.5
                )
            
            # 3. Plot ignition points (where input heatmap > 0)
            fire_y, fire_x = np.where(fires.heatmap > 0)
            if len(fire_x) > 0:
                ax.scatter(
                    lon[fire_x], lat[fire_y], 
                    marker='*', color='blue', s=100, label='Ignition' if i == 0 else ""
                )
            
            # 4. Draw wind vector field (quiver)
            # Plot a small grid of arrows to show wind direction
            skip = 20
            q_lon, q_lat = np.meshgrid(lon[::skip], lat[::skip])
            if u**2 + v**2 > 0:
                ax.quiver(
                    q_lon, q_lat, 
                    np.ones_like(q_lon) * u, np.ones_like(q_lat) * v,
                    color='blue', alpha=0.3, scale=500, width=0.005
                )

            ax.set_title(f"T+{h}h", fontsize=14)
            ax.set_xlabel("Lon")
            if i == 0:
                ax.set_ylabel("Lat")
                ax.legend(loc='upper left')

        # Add a single colorbar for all subplots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Probability")
        
        save_path = f"{output_dir}/spread_{name}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved plot to {save_path}")

    print("Inspection complete.")

if __name__ == "__main__":
    run_inspection()

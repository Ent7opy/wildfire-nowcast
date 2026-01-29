"""Legend component for map layers."""

from state import app_state
from config.theme import RiskThresholds


def get_legend_html() -> str:
    """Generate HTML for floating legend based on active layers."""
    legend_items = []

    if app_state.layers.show_fires:
        legend_items.append("🔥 <strong>Active fires</strong> — size and color indicate intensity (FRP)")

    if app_state.layers.show_forecast:
        legend_items.append("🟠 <strong>Forecast overlay</strong> — spread outlook")
        legend_items.append("<strong>Contours by horizon:</strong> T+24h (blue), T+48h (orange), T+72h (red)")
        legend_items.append("Shaded layer: higher = more likely spread")

    if app_state.layers.show_risk:
        legend_items.append("🔥 <strong>Risk index</strong> — fire risk heatmap:")
        legend_items.append(f"  • 🟢 Low (0.0-{RiskThresholds.MEDIUM}): minimal fire risk")
        legend_items.append(f"  • 🟡 Medium ({RiskThresholds.MEDIUM}-{RiskThresholds.HIGH}): moderate fire risk")
        legend_items.append(f"  • 🔴 High ({RiskThresholds.HIGH}-1.0): elevated fire risk")
        legend_items.append("  Based on land cover + weather conditions")

    if not legend_items:
        return ""  # Don't show legend if no layers active

    items_html = "".join([f"<div style='margin: 4px 0;'>{item}</div>" for item in legend_items])

    return f"""
    <div id="floating-legend">
        <div style="font-weight: bold; margin-bottom: 8px; font-size: 14px;">Legend</div>
        {items_html}
    </div>
    """


def render_legend() -> None:
    """Render the map legend based on active layers (deprecated - now rendered as floating overlay in app.py)."""
    pass

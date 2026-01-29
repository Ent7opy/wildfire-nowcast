"""Legend component for map layers."""

from state import app_state
from config.theme import FireColors, RiskColors, RiskThresholds


def _color_swatch(rgb: list, label: str) -> str:
    """Generate HTML for a colored dot + label pair."""
    r, g, b = rgb[:3]
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
        f'<span style="display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;background:rgb({r},{g},{b});'
        f'box-shadow:0 0 4px rgba({r},{g},{b},0.5);flex-shrink:0;"></span>'
        f'<span style="color:#e0e0e0;font-size:12px;">{label}</span></div>'
    )


def get_legend_html() -> str:
    """Generate HTML for floating legend based on active layers."""
    legend_items = []

    if app_state.layers.show_fires:
        legend_items.append(
            '<div style="font-weight:600;color:#e0e0e0;font-size:13px;margin-bottom:4px;">'
            'Active fires</div>'
        )
        legend_items.append(_color_swatch(FireColors.VERY_HIGH, "Very high likelihood"))
        legend_items.append(_color_swatch(FireColors.HIGH, "High likelihood"))
        legend_items.append(_color_swatch(FireColors.MEDIUM, "Medium likelihood"))
        legend_items.append(_color_swatch(FireColors.LOW, "Low likelihood"))
        legend_items.append(_color_swatch(FireColors.VERY_LOW, "Very low likelihood"))
        legend_items.append(_color_swatch(FireColors.UNSCORED, "Unscored"))
        legend_items.append(
            '<div style="color:rgba(255,255,255,0.5);font-size:11px;margin-top:2px;">'
            'Size indicates intensity (FRP)</div>'
        )

    if app_state.layers.show_forecast:
        legend_items.append(
            '<div style="font-weight:600;color:#e0e0e0;font-size:13px;margin:8px 0 4px;">'
            'Forecast overlay</div>'
        )
        legend_items.append(_color_swatch([255, 165, 0], "Spread outlook"))
        legend_items.append(
            '<div style="color:rgba(255,255,255,0.5);font-size:11px;">'
            'T+24h, T+48h, T+72h contours</div>'
        )

    if app_state.layers.show_risk:
        legend_items.append(
            '<div style="font-weight:600;color:#e0e0e0;font-size:13px;margin:8px 0 4px;">'
            'Risk index</div>'
        )
        legend_items.append(_color_swatch(RiskColors.LOW, f"Low (0.0\u2013{RiskThresholds.MEDIUM})"))
        legend_items.append(_color_swatch(RiskColors.MEDIUM, f"Medium ({RiskThresholds.MEDIUM}\u2013{RiskThresholds.HIGH})"))
        legend_items.append(_color_swatch(RiskColors.HIGH, f"High ({RiskThresholds.HIGH}\u20131.0)"))

    if not legend_items:
        return ""

    items_html = "".join(legend_items)

    return f"""
    <div id="floating-legend">
        <div style="font-weight:700;margin-bottom:8px;font-size:14px;color:#e0e0e0;">Legend</div>
        {items_html}
    </div>
    """


def render_legend() -> None:
    """Deprecated - now rendered as floating overlay in app.py."""
    pass

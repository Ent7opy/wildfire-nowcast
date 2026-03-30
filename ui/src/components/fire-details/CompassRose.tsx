interface CompassRoseProps {
  /** Meteorological direction the wind is coming *from* (0–360°, 0 = North). */
  directionDeg: number;
  /** Outer diameter in pixels. Defaults to 40. */
  size?: number;
}

/**
 * Compact SVG compass rose.  Renders a circular ring with N/E/S/W labels and
 * a directional arrow pointing toward where the wind is coming from.
 */
export function CompassRose({ directionDeg, size = 40 }: CompassRoseProps): JSX.Element {
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 2;          // ring radius with 2px inset
  const arrowLen = r * 0.62;       // arrow shaft length
  const labelOffset = r + 5.5;     // label distance from centre

  // Arrow tip points toward the direction the wind comes *from*, so we rotate
  // the upward-pointing arrow by directionDeg.
  const rotationDeg = ((directionDeg % 360) + 360) % 360;

  const labelFontSize = Math.max(5, size * 0.175);

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      aria-label={`Wind from ${rotationDeg.toFixed(0)}°`}
      style={{ flexShrink: 0 }}
    >
      <circle
        cx={cx}
        cy={cy}
        r={r}
        fill="none"
        stroke="rgba(255,255,255,0.15)"
        strokeWidth={1}
      />

      {(["N", "E", "S", "W"] as const).map((label, i) => {
        const angleDeg = i * 90 - 90; // N=top=-90, E=right=0, S=bottom=90, W=left=180
        const rad = (angleDeg * Math.PI) / 180;
        const lx = cx + labelOffset * Math.cos(rad);
        const ly = cy + labelOffset * Math.sin(rad);
        return (
          <text
            key={label}
            x={lx}
            y={ly}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={labelFontSize}
            fill="rgba(255,255,255,0.35)"
            fontWeight={700}
            fontFamily="inherit"
          >
            {label}
          </text>
        );
      })}

      <g transform={`rotate(${rotationDeg}, ${cx}, ${cy})`}>
        <line
          x1={cx}
          y1={cy + arrowLen * 0.25}
          x2={cx}
          y2={cy - arrowLen}
          stroke="#60a5fa"
          strokeWidth={1.5}
          strokeLinecap="round"
        />
        <polygon
          points={`${cx},${cy - arrowLen - 3} ${cx - 3},${cy - arrowLen + 3} ${cx + 3},${cy - arrowLen + 3}`}
          fill="#60a5fa"
        />
        <line
          x1={cx - 3}
          y1={cy + arrowLen * 0.25}
          x2={cx + 3}
          y2={cy + arrowLen * 0.25}
          stroke="#60a5fa"
          strokeWidth={1.5}
          strokeLinecap="round"
        />
      </g>
    </svg>
  );
}

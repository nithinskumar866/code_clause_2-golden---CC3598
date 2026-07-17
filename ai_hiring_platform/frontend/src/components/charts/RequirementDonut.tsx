import type { FC } from 'react';

export interface DonutSegment {
  label: string;
  value: number;
  color: string;
}

interface RequirementDonutProps {
  segments: DonutSegment[];
  size?: number;
  strokeWidth?: number;
  /** Large number rendered in the centre (defaults to the total). */
  centerValue?: number;
  centerLabel?: string;
}

/**
 * Donut chart of requirement outcomes (Matched / Partial / Missing).
 * Pure inline SVG; segments are drawn with stroke-dasharray arcs.
 */
export const RequirementDonut: FC<RequirementDonutProps> = ({
  segments,
  size = 160,
  strokeWidth = 18,
  centerValue,
  centerLabel = 'Total',
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const total = segments.reduce((sum, s) => sum + s.value, 0);
  const center = size / 2;

  let cumulative = 0;

  return (
    <div className="flex items-center gap-5">
      <svg
        width={size}
        height={size}
        role="img"
        aria-label="Requirement outcome breakdown"
        className="shrink-0"
      >
        <g transform={`rotate(-90 ${center} ${center})`}>
          {/* Track */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke="currentColor"
            className="text-white/5"
            strokeWidth={strokeWidth}
          />
          {total > 0 &&
            segments
              .filter((s) => s.value > 0)
              .map((seg) => {
                const fraction = seg.value / total;
                const dash = fraction * circumference;
                const offset = -cumulative * circumference;
                cumulative += fraction;
                return (
                  <circle
                    key={seg.label}
                    cx={center}
                    cy={center}
                    r={radius}
                    fill="none"
                    stroke={seg.color}
                    strokeWidth={strokeWidth}
                    strokeDasharray={`${dash} ${circumference - dash}`}
                    strokeDashoffset={offset}
                  />
                );
              })}
        </g>
        <text
          x={center}
          y={center - 4}
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-white text-2xl font-extrabold"
        >
          {centerValue ?? total}
        </text>
        <text
          x={center}
          y={center + 14}
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-gray-500 text-[9px] font-semibold uppercase tracking-widest"
        >
          {centerLabel}
        </text>
      </svg>

      <ul className="space-y-2 text-xs">
        {segments.map((seg) => (
          <li key={seg.label} className="flex items-center gap-2 text-gray-300">
            <span className="h-2.5 w-2.5 rounded-sm shrink-0" style={{ backgroundColor: seg.color }} />
            <span className="font-medium">{seg.label}</span>
            <span className="ml-auto font-semibold text-white">{seg.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

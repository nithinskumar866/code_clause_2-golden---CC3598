import type { FC } from 'react';

interface RadarAxis {
  label: string;
  /** Value 0–100. */
  value: number;
}

interface SubScoreRadarProps {
  data: RadarAxis[];
  size?: number;
}

const RING_LEVELS = [25, 50, 75, 100];

/**
 * Radar / spider chart of the explainable sub-scores. Pure inline SVG.
 * Works for any number of axes; the platform feeds it the four weighted
 * metrics (coverage, experience, project, confidence).
 */
export const SubScoreRadar: FC<SubScoreRadarProps> = ({ data, size = 260 }) => {
  const center = size / 2;
  const maxRadius = size / 2 - 46; // leave room for axis labels
  const n = data.length;

  // Vertex angle for axis i, starting at the top (−90°) and going clockwise.
  const angleFor = (i: number) => (-90 + (360 / n) * i) * (Math.PI / 180);

  const pointAt = (i: number, radius: number) => ({
    x: center + radius * Math.cos(angleFor(i)),
    y: center + radius * Math.sin(angleFor(i)),
  });

  const toPolygon = (radiusForIndex: (i: number) => number) =>
    data
      .map((_, i) => {
        const p = pointAt(i, radiusForIndex(i));
        return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
      })
      .join(' ');

  const dataPolygon = toPolygon((i) => (Math.max(0, Math.min(100, data[i].value)) / 100) * maxRadius);

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label="Radar chart of explainable sub-scores"
      className="overflow-visible"
    >
      {/* Background rings */}
      {RING_LEVELS.map((level) => (
        <polygon
          key={level}
          points={toPolygon(() => (level / 100) * maxRadius)}
          fill="none"
          stroke="currentColor"
          className="text-white/5"
          strokeWidth={1}
        />
      ))}

      {/* Axes + labels */}
      {data.map((axis, i) => {
        const outer = pointAt(i, maxRadius);
        const label = pointAt(i, maxRadius + 24);
        const anchor = label.x > center + 1 ? 'start' : label.x < center - 1 ? 'end' : 'middle';
        return (
          <g key={axis.label}>
            <line
              x1={center}
              y1={center}
              x2={outer.x}
              y2={outer.y}
              stroke="currentColor"
              className="text-white/10"
              strokeWidth={1}
            />
            <text
              x={label.x}
              y={label.y}
              textAnchor={anchor}
              dominantBaseline="middle"
              className="fill-gray-400 text-[10px] font-semibold uppercase tracking-wide"
            >
              {axis.label}
            </text>
            <text
              x={label.x}
              y={label.y + 12}
              textAnchor={anchor}
              dominantBaseline="middle"
              className="fill-indigo-300 text-[10px] font-bold"
            >
              {Math.round(axis.value)}%
            </text>
          </g>
        );
      })}

      {/* Data polygon */}
      <polygon
        points={dataPolygon}
        fill="#6366f1"
        fillOpacity={0.25}
        stroke="#818cf8"
        strokeWidth={2}
        strokeLinejoin="round"
      />
      {data.map((axis, i) => {
        const p = pointAt(i, (Math.max(0, Math.min(100, axis.value)) / 100) * maxRadius);
        return <circle key={axis.label} cx={p.x} cy={p.y} r={3} fill="#818cf8" />;
      })}
    </svg>
  );
};

import { useEffect, useState, type FC } from 'react';
import { scoreHex } from '../analysis/scoreColors';

interface ScoreRingProps {
  /** Score 0–100. */
  score: number;
  size?: number;
  strokeWidth?: number;
}

/**
 * Animated circular progress ring for the overall fit score.
 * Pure inline SVG — no charting dependency. Colour follows the shared
 * score thresholds. The arc animates from 0 on mount.
 */
export const ScoreRing: FC<ScoreRingProps> = ({ score, size = 128, strokeWidth = 10 }) => {
  const clamped = Math.max(0, Math.min(100, Math.round(score)));
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const color = scoreHex(clamped);

  const [progress, setProgress] = useState(0);
  useEffect(() => {
    const frame = requestAnimationFrame(() => setProgress(clamped));
    return () => cancelAnimationFrame(frame);
  }, [clamped]);

  const offset = circumference - (progress / 100) * circumference;

  return (
    <div
      className="relative inline-flex items-center justify-center"
      style={{ width: size, height: size }}
      role="img"
      aria-label={`Overall fit score ${clamped} percent`}
    >
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-white/5"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 900ms ease-out' }}
        />
      </svg>
      <span className="absolute text-3xl font-extrabold" style={{ color }}>
        {clamped}%
      </span>
    </div>
  );
};

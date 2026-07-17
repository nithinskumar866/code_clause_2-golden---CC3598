import type { FC } from 'react';

interface SkeletonProps {
  className?: string;
}

/** Neutral pulsing placeholder block. Compose with sizing utilities. */
export const Skeleton: FC<SkeletonProps> = ({ className = '' }) => (
  <div className={`animate-pulse rounded bg-white/5 ${className}`} />
);

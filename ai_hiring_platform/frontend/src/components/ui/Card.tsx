import type { FC, ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
}

/** Standard surface: rounded, subtle border, card background. */
export const Card: FC<CardProps> = ({ children, className = '' }) => (
  <div className={`rounded-xl border border-white/5 bg-card ${className}`}>{children}</div>
);

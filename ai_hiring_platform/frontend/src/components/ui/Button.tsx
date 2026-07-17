import type { ButtonHTMLAttributes, FC, ReactNode } from 'react';
import { Spinner } from './Spinner';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  leftIcon?: ReactNode;
}

const BASE =
  'inline-flex items-center justify-center gap-2 rounded-lg font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:opacity-50 disabled:cursor-not-allowed';

const SIZES: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-4 py-2.5 text-sm',
};

const VARIANTS: Record<ButtonVariant, string> = {
  primary: 'bg-indigo-600 text-white hover:bg-indigo-500 shadow-lg shadow-indigo-600/10',
  secondary: 'border border-white/10 text-gray-200 hover:bg-white/5 hover:text-white',
  ghost: 'text-gray-400 hover:bg-white/5 hover:text-white',
  danger: 'bg-rose-600 text-white hover:bg-rose-500',
};

/** Consistent, accessible button with variants, sizes and a loading state. */
export const Button: FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  loading = false,
  leftIcon,
  className = '',
  children,
  disabled,
  type = 'button',
  ...rest
}) => (
  <button
    type={type}
    className={`${BASE} ${SIZES[size]} ${VARIANTS[variant]} ${className}`}
    disabled={disabled || loading}
    aria-busy={loading || undefined}
    {...rest}
  >
    {loading ? <Spinner className="h-4 w-4" /> : leftIcon}
    {children}
  </button>
);

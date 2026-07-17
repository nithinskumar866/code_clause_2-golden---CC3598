import type { FC, SelectHTMLAttributes } from 'react';

export interface SelectOption {
  value: string;
  label: string;
}

interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'children'> {
  label?: string;
  options: SelectOption[];
  /** Render the label for screen readers only (keeps the control compact). */
  srLabel?: boolean;
}

/** Labeled select field. The wrapping label provides an accessible name. */
export const Select: FC<SelectProps> = ({ label, srLabel = false, options, className = '', ...rest }) => (
  <label className="block">
    {label && (
      <span
        className={
          srLabel ? 'sr-only' : 'mb-1.5 block text-xs font-semibold uppercase tracking-wider text-gray-400'
        }
      >
        {label}
      </span>
    )}
    <select
      className={`rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50 ${className}`}
      {...rest}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  </label>
);

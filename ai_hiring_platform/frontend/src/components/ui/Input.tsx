import type { FC, InputHTMLAttributes } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  id: string;
}

/** Labeled text/number/date input with an associated <label> for a11y. */
export const Input: FC<InputProps> = ({ label, id, className = '', ...rest }) => (
  <div>
    <label htmlFor={id} className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-gray-400">
      {label}
    </label>
    <input
      id={id}
      className={`w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white placeholder:text-gray-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 ${className}`}
      {...rest}
    />
  </div>
);

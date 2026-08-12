import { useRef, useState, type DragEvent, type FC } from 'react';
import { UploadCloud } from 'lucide-react';

interface FileDropProps {
  onFile: (file: File) => void;
  /** Shown in place of the idle prompt while the file is being processed. */
  busy?: boolean;
  busyLabel?: string;
  label: string;
  hint?: string;
  className?: string;
  /** Compact variant for the chat popup, where vertical space is scarce. */
  compact?: boolean;
}

const ACCEPT = '.pdf,.docx,.txt,.md';

/**
 * Drag-and-drop file input.
 *
 * Written against the native drag events rather than pulling `react-dropzone`
 * into the recruiter app: this is the whole of what the two portal sections
 * need, and the host app has no other use for the dependency.
 *
 * `dragCounter` rather than a boolean: dragenter/dragleave also fire when the
 * pointer crosses a *child* element, so a plain boolean flickers off as soon as
 * the cursor moves over the icon inside the zone.
 */
export const FileDrop: FC<FileDropProps> = ({
  onFile, busy = false, busyLabel = 'Working…', label, hint, className = '', compact = false,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);
  const [dragActive, setDragActive] = useState(false);

  const stop = (event: DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const onDrop = (event: DragEvent) => {
    stop(event);
    dragCounter.current = 0;
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (file && !busy) onFile(file);
  };

  return (
    <div
      onDragEnter={(e) => { stop(e); dragCounter.current += 1; setDragActive(true); }}
      onDragOver={stop}
      onDragLeave={(e) => {
        stop(e);
        dragCounter.current -= 1;
        if (dragCounter.current <= 0) setDragActive(false);
      }}
      onDrop={onDrop}
      onClick={() => !busy && inputRef.current?.click()}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          if (!busy) inputRef.current?.click();
        }
      }}
      role="button"
      tabIndex={0}
      aria-label={label}
      aria-busy={busy || undefined}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed text-center transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
        compact ? 'gap-1 px-4 py-5' : 'gap-2 px-6 py-10'
      } ${
        dragActive
          ? 'border-indigo-500 bg-indigo-500/10'
          : 'border-white/15 bg-black/20 hover:border-white/25 hover:bg-white/5'
      } ${busy ? 'cursor-wait opacity-70' : ''} ${className}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          // Cleared so choosing the same file twice in a row still fires change.
          e.target.value = '';
          if (file) onFile(file);
        }}
      />
      <UploadCloud className={`${compact ? 'h-5 w-5' : 'h-7 w-7'} text-indigo-400`} />
      <span className={`font-semibold text-white ${compact ? 'text-xs' : 'text-sm'}`}>
        {busy ? busyLabel : label}
      </span>
      <span className={`text-gray-500 ${compact ? 'text-[10px]' : 'text-xs'}`}>
        {hint ?? 'PDF, DOCX, TXT or MD — or click to browse'}
      </span>
    </div>
  );
};

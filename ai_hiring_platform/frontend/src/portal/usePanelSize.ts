import { useCallback, useEffect, useRef, useState } from 'react';

export interface PanelSize {
  width: number;
  height: number;
}

const STORAGE_KEY = 'portal.chatSize';

/** Small enough to still be a popup; large enough to read a cover letter in. */
const MIN: PanelSize = { width: 340, height: 380 };
const DEFAULT: PanelSize = { width: 420, height: 660 };

/**
 * Keeps the panel inside the window, however the window has changed since.
 *
 * A size saved on a wide monitor must not leave the panel hanging off a laptop
 * screen the next morning, so the stored value is clamped on every read rather
 * than trusted.
 */
function clamp(size: PanelSize): PanelSize {
  const maxWidth = Math.max(MIN.width, window.innerWidth - 40);
  const maxHeight = Math.max(MIN.height, window.innerHeight - 48);
  return {
    width: Math.min(Math.max(size.width, MIN.width), maxWidth),
    height: Math.min(Math.max(size.height, MIN.height), maxHeight),
  };
}

function read(): PanelSize {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return clamp(DEFAULT);
    const parsed = JSON.parse(stored) as Partial<PanelSize>;
    if (typeof parsed.width !== 'number' || typeof parsed.height !== 'number') return clamp(DEFAULT);
    return clamp({ width: parsed.width, height: parsed.height });
  } catch {
    // A corrupt or unavailable store is not a reason to fail to open a chat.
    return clamp(DEFAULT);
  }
}

/**
 * A panel the user can drag to whatever size suits them.
 *
 * The handle is on the TOP-LEFT because the panel is anchored bottom-right:
 * dragging that corner grows it into the empty screen rather than pushing it
 * off the edge, so width and height both increase as the pointer moves away
 * from the anchor.
 *
 * Pointer events rather than mouse events, so a trackpad, a pen and a touch
 * screen all work, and `setPointerCapture` keeps the drag alive when the pointer
 * outruns the 12px handle — which it always does.
 */
export function usePanelSize() {
  const [size, setSize] = useState<PanelSize>(read);
  const [resizing, setResizing] = useState(false);
  const origin = useRef<{ x: number; y: number; width: number; height: number } | null>(null);

  // Persisted per change rather than on unmount: the popup is never unmounted,
  // so an unmount-time save would never run.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(size));
    } catch {
      // Private browsing, quota, or a blocked store. The size still applies for
      // this session; losing it later is not worth surfacing.
    }
  }, [size]);

  useEffect(() => {
    const onWindowResize = () => setSize(current => clamp(current));
    window.addEventListener('resize', onWindowResize);
    return () => window.removeEventListener('resize', onWindowResize);
  }, []);

  const startResize = useCallback((event: React.PointerEvent<HTMLElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    origin.current = { x: event.clientX, y: event.clientY, width: size.width, height: size.height };
    setResizing(true);
  }, [size.width, size.height]);

  useEffect(() => {
    if (!resizing) return;

    const onMove = (event: PointerEvent) => {
      const from = origin.current;
      if (!from) return;
      // Anchored bottom-right, so leftward and upward movement is growth.
      setSize(clamp({
        width: from.width + (from.x - event.clientX),
        height: from.height + (from.y - event.clientY),
      }));
    };

    const stop = () => { origin.current = null; setResizing(false); };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', stop);
    window.addEventListener('pointercancel', stop);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', stop);
      window.removeEventListener('pointercancel', stop);
    };
  }, [resizing]);

  return { size, startResize, resizing };
}

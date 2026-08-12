import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

/**
 * Photo-gallery style multi-select over an ordered list.
 *
 * WHY NOT JUST CHECKBOXES
 * Picking 60 resumes out of 300 with one checkbox per row is 60 precise clicks, and a
 * recruiter doing it twice a week will not. Every gallery solves this the same way, so
 * this reproduces those exact gestures — they are already in the user's fingers:
 *
 *   click            select only this one
 *   ctrl/cmd + click add or remove this one
 *   shift + click    select the whole range from the anchor
 *   press and drag   sweep across items to select them continuously
 *   ctrl/cmd + A     select everything currently listed
 *   Escape           clear
 *
 * DRAG IS RANGE-BASED, NOT A RUBBER-BAND RECTANGLE. Pointer-down remembers an anchor,
 * and moving over any item selects the anchor..current range. That behaves correctly no
 * matter how the list is laid out — grid, table, one column on a phone — whereas a
 * geometric marquee has to be re-taught the layout and breaks when rows reflow.
 *
 * Selection is keyed by id and intersected with the visible list on every change, so
 * filtering or deleting cannot leave a phantom id selected.
 */
export interface GallerySelection {
  selected: Set<number>;
  isSelected: (id: number) => boolean;
  count: number;
  /** Props to spread on each selectable item. `index` is its position in `ids`. */
  itemProps: (id: number, index: number) => {
    onPointerDown: (e: React.PointerEvent) => void;
    onPointerEnter: () => void;
    onClick: (e: React.MouseEvent) => void;
    'data-selected': boolean;
  };
  selectAll: () => void;
  clear: () => void;
  invert: () => void;
  setSelected: (ids: number[]) => void;
  toggle: (id: number) => void;
  dragging: boolean;
}

export function useGallerySelection(ids: number[]): GallerySelection {
  const [selected, setSelectedState] = useState<Set<number>>(new Set());
  const [dragging, setDragging] = useState(false);
  // Anchor for shift-click and for a drag sweep.
  const anchorRef = useRef<number | null>(null);
  // The set as it was when the drag began, so a sweep can be recomputed from scratch
  // on every move instead of accumulating — that is what lets a drag shrink again when
  // you move back toward the anchor.
  const dragBaseRef = useRef<Set<number>>(new Set());
  const dragAdditiveRef = useRef(false);

  const idList = useMemo(() => ids, [ids]);

  // A selected id that is no longer listed (filtered out, deleted) must not survive —
  // acting on it later would hit something the recruiter can no longer see.
  useEffect(() => {
    setSelectedState((prev) => {
      const visible = new Set(idList);
      let changed = false;
      const next = new Set<number>();
      prev.forEach((id) => {
        if (visible.has(id)) next.add(id);
        else changed = true;
      });
      return changed ? next : prev;
    });
  }, [idList]);

  const applyRange = useCallback(
    (from: number, to: number, base: Set<number>, additive: boolean) => {
      const [lo, hi] = from <= to ? [from, to] : [to, from];
      const next = additive ? new Set(base) : new Set<number>();
      for (let i = lo; i <= hi; i += 1) {
        const id = idList[i];
        if (id !== undefined) next.add(id);
      }
      setSelectedState(next);
    },
    [idList],
  );

  // The drag ends wherever the pointer is released, including outside the list.
  useEffect(() => {
    if (!dragging) return undefined;
    const stop = () => setDragging(false);
    window.addEventListener('pointerup', stop);
    window.addEventListener('pointercancel', stop);
    return () => {
      window.removeEventListener('pointerup', stop);
      window.removeEventListener('pointercancel', stop);
    };
  }, [dragging]);

  const itemProps = useCallback(
    (id: number, index: number) => ({
      'data-selected': selected.has(id),
      onPointerDown: (e: React.PointerEvent) => {
        // Left button only, and never start a sweep from a real control inside the row.
        if (e.button !== 0) return;
        if ((e.target as HTMLElement).closest('button, a, input, select, textarea')) return;

        const additive = e.ctrlKey || e.metaKey;
        if (e.shiftKey && anchorRef.current !== null) {
          applyRange(anchorRef.current, index, selected, true);
          return;
        }

        anchorRef.current = index;
        dragAdditiveRef.current = additive;
        dragBaseRef.current = additive ? new Set(selected) : new Set();
        setDragging(true);

        // A plain press selects immediately, so a click that never moves still works.
        const next = additive ? new Set(selected) : new Set<number>();
        if (additive && next.has(id)) next.delete(id);
        else next.add(id);
        setSelectedState(next);
      },
      onPointerEnter: () => {
        if (!dragging || anchorRef.current === null) return;
        applyRange(anchorRef.current, index, dragBaseRef.current, dragAdditiveRef.current);
      },
      // Selection is decided on pointerdown; click only needs to not re-toggle it.
      onClick: (e: React.MouseEvent) => {
        e.preventDefault();
      },
    }),
    [applyRange, dragging, selected],
  );

  const selectAll = useCallback(() => setSelectedState(new Set(idList)), [idList]);
  const clear = useCallback(() => setSelectedState(new Set()), []);
  const invert = useCallback(
    () => setSelectedState(new Set(idList.filter((id) => !selected.has(id)))),
    [idList, selected],
  );
  const setSelected = useCallback((next: number[]) => setSelectedState(new Set(next)), []);
  const toggle = useCallback(
    (id: number) =>
      setSelectedState((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      }),
    [],
  );

  // Ctrl/Cmd+A and Escape, ignored while the user is typing in the filter box.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName)) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') {
        e.preventDefault();
        selectAll();
      } else if (e.key === 'Escape') {
        clear();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectAll, clear]);

  return {
    selected,
    isSelected: (id: number) => selected.has(id),
    count: selected.size,
    itemProps,
    selectAll,
    clear,
    invert,
    setSelected,
    toggle,
    dragging,
  };
}

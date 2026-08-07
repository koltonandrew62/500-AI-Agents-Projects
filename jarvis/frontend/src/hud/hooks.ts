/**
 * Shared internal hooks for the HUD layer.
 *
 * Kept deliberately tiny and dependency-free: every animated HUD component
 * drives itself from `useRafLoop`, which guarantees a single cancelled frame
 * on unmount, and consults `usePrefersReducedMotion` to degrade gracefully.
 */
import { useEffect, useRef, useState } from 'react';

/** Callback invoked once per animation frame. */
export type RafCallback = (deltaSeconds: number, elapsedSeconds: number) => void;

/**
 * Runs `callback` on every animation frame until unmount.
 *
 * The callback is stored in a ref so a new inline closure on each render does
 * not restart the loop. Pass `enabled: false` to suspend the loop entirely
 * (used for reduced-motion and for offscreen panels).
 */
export function useRafLoop(callback: RafCallback, enabled = true): void {
  const savedRef = useRef<RafCallback>(callback);
  savedRef.current = callback;

  useEffect(() => {
    if (!enabled) return;
    let frame = 0;
    let last = performance.now();
    const start = last;

    const tick = (now: number): void => {
      // Clamp: a backgrounded tab can produce multi-second deltas that would
      // fling every animation forward on return.
      const delta = Math.min((now - last) / 1000, 0.1);
      last = now;
      savedRef.current(delta, (now - start) / 1000);
      frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [enabled]);
}

/** True when the user has asked the OS to reduce motion. */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState<boolean>(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = (event: MediaQueryListEvent): void => setReduced(event.matches);
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  return reduced;
}

/**
 * Observes an element's pixel size. Returns `[ref, size]`; size is `null`
 * until the element has been laid out.
 */
export function useElementSize<T extends HTMLElement>(): [
  React.RefObject<T>,
  { width: number; height: number } | null,
] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const apply = (width: number, height: number): void => {
      setSize((prev) =>
        prev && prev.width === width && prev.height === height ? prev : { width, height },
      );
    };

    apply(node.clientWidth, node.clientHeight);

    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const box = entry.contentRect;
      apply(Math.round(box.width), Math.round(box.height));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return [ref, size];
}

/** Exponential smoothing toward a target — frame-rate independent. */
export function approach(current: number, target: number, rate: number, delta: number): number {
  const factor = 1 - Math.exp(-rate * delta);
  return current + (target - current) * factor;
}

/** Clamp helper used by every gauge in the HUD. */
export function clamp(value: number, min = 0, max = 1): number {
  return value < min ? min : value > max ? max : value;
}

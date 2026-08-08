/**
 * Diagnostics — the compact bottom-left key/value readout.
 *
 * Static text in a HUD looks dead, so one row at a time is briefly "glitched"
 * (a scrambled value + a hard flicker). The scheduler is a RAF accumulator
 * rather than a timer so it pauses with the tab and cancels cleanly on unmount,
 * and it is disabled outright under reduced motion.
 */
import { useMemo, useRef, useState } from 'react';
import type { ReactElement } from 'react';
import { Frame } from './Frame';
import { useRafLoop, usePrefersReducedMotion } from './hooks';
import styles from './Diagnostics.module.css';

export type DiagTone = 'cyan' | 'amber' | 'red' | 'dim';

export interface DiagnosticEntry {
  /** Left column text, rendered uppercase. Also the React key. */
  label: string;
  /** Right column value. Numbers are stringified by the caller. */
  value: string;
  /** Value color. Default `'cyan'`. */
  tone?: DiagTone;
}

export interface DiagnosticsProps {
  entries: readonly DiagnosticEntry[];
  /** Panel title. Default `"DIAGNOSTICS"`. */
  title?: string;
  /** Right-aligned frame status. */
  status?: string;
  /** Mean seconds between glitch events. Default 2.4. Set 0 to disable. */
  glitchIntervalSec?: number;
  /** Render two key/value columns instead of one. Default false. */
  dense?: boolean;
  className?: string;
}

const TONE_CLASS: Record<DiagTone, string> = {
  cyan: styles.cyan,
  amber: styles.amber,
  red: styles.red,
  dim: styles.dim,
};

const SCRAMBLE = '#%&$@!*0123456789ABCDEF';

/** Replaces a value with same-length noise, preserving separators. */
function scramble(value: string): string {
  let out = '';
  for (let i = 0; i < value.length; i += 1) {
    const char = value[i];
    out +=
      char === ' ' || char === ':' || char === '.' || char === '/' || char === '-'
        ? char
        : SCRAMBLE[Math.floor(Math.random() * SCRAMBLE.length)];
  }
  return out;
}

export function Diagnostics({
  entries,
  title = 'DIAGNOSTICS',
  status,
  glitchIntervalSec = 2.4,
  dense = false,
  className,
}: DiagnosticsProps): ReactElement {
  const reduced = usePrefersReducedMotion();
  const [glitch, setGlitch] = useState<{ index: number; text: string } | null>(null);

  const countRef = useRef(entries.length);
  countRef.current = entries.length;
  const entriesRef = useRef<readonly DiagnosticEntry[]>(entries);
  entriesRef.current = entries;

  // Accumulator state for the glitch scheduler.
  const timerRef = useRef({ nextIn: glitchIntervalSec, holdFor: 0 });
  const intervalRef = useRef(glitchIntervalSec);
  intervalRef.current = glitchIntervalSec;

  const glitchEnabled = !reduced && glitchIntervalSec > 0 && entries.length > 0;

  useRafLoop((delta) => {
    const timer = timerRef.current;

    if (timer.holdFor > 0) {
      timer.holdFor -= delta;
      if (timer.holdFor <= 0) {
        setGlitch(null);
        // Jittered gap so the flicker never falls into a visible rhythm.
        timer.nextIn = intervalRef.current * (0.55 + Math.random() * 1.1);
      }
      return;
    }

    timer.nextIn -= delta;
    if (timer.nextIn > 0) return;

    const count = countRef.current;
    if (count === 0) {
      timer.nextIn = intervalRef.current;
      return;
    }
    const index = Math.floor(Math.random() * count);
    const entry = entriesRef.current[index];
    setGlitch({ index, text: scramble(entry.value) });
    timer.holdFor = 0.06 + Math.random() * 0.1;
  }, glitchEnabled);

  const rows = useMemo(
    () =>
      entries.map((entry, index) => ({
        entry,
        index,
        tone: TONE_CLASS[entry.tone ?? 'cyan'],
      })),
    [entries],
  );

  return (
    <Frame
      title={title}
      status={status}
      scan
      className={className}
      bodyClassName={styles.body}
    >
      <dl className={`${styles.grid} ${dense ? styles.dense : ''}`.trim()}>
        {rows.map(({ entry, index, tone }) => {
          const glitched = glitch !== null && glitch.index === index;
          return (
            <div
              key={entry.label}
              className={`${styles.row} ${tone} ${glitched ? styles.glitch : ''}`.trim()}
            >
              <dt className={styles.key}>{entry.label}</dt>
              <dd className={styles.value}>{glitched ? glitch.text : entry.value}</dd>
            </div>
          );
        })}
      </dl>
    </Frame>
  );
}

export default Diagnostics;

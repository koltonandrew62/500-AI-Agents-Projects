/**
 * Telemetry — the scrolling event console.
 *
 * Newest-first (the list is reversed on render, so "auto-scroll" needs no
 * scroll math: new rows appear at the top and push the rest down). Rows are
 * capped at `maxRows` to bound both DOM size and memory when the backend is
 * chatty; the cap is applied at render so the caller may keep a longer history.
 */
import { useMemo } from 'react';
import type { ReactElement } from 'react';
import { Frame } from './Frame';
import styles from './Telemetry.module.css';

export type TelemetryLevel = 'debug' | 'info' | 'warn' | 'error' | 'ok';

export interface TelemetryEntry {
  /** Stable identity — used as the React key. */
  id: string;
  /** Severity; drives row color. Unknown strings degrade to `info`. */
  level: TelemetryLevel | string;
  /** Row body. Rendered uppercase, monospace, single line with ellipsis. */
  text: string;
  /** Epoch milliseconds. Defaults to render time if omitted. */
  ts?: number;
  /** Optional short source tag, e.g. `"VISION"`. */
  source?: string;
}

export interface TelemetryProps {
  /** Event log in arrival order (oldest first). Rendered newest-first. */
  entries: readonly TelemetryEntry[];
  /** Hard cap on rendered rows. Default 60. */
  maxRows?: number;
  /** Panel title. Default `"TELEMETRY"`. */
  title?: string;
  /** Right-aligned frame status. Defaults to the rendered row count. */
  status?: string;
  /** Show the HH:MM:SS gutter. Default true. */
  showTime?: boolean;
  className?: string;
}

const LEVEL_CLASS: Record<TelemetryLevel, string> = {
  debug: styles.debug,
  info: styles.info,
  warn: styles.warn,
  error: styles.error,
  ok: styles.ok,
};

const LEVEL_MARK: Record<TelemetryLevel, string> = {
  debug: '::',
  info: '>>',
  warn: '/!',
  error: 'XX',
  ok: '++',
};

function normalizeLevel(level: string): TelemetryLevel {
  switch (level) {
    case 'debug':
    case 'info':
    case 'warn':
    case 'error':
    case 'ok':
      return level;
    default:
      return 'info';
  }
}

function formatTime(ts: number): string {
  const date = new Date(ts);
  const pad = (n: number): string => String(n).padStart(2, '0');
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

export function Telemetry({
  entries,
  maxRows = 60,
  title = 'TELEMETRY',
  status,
  showTime = true,
  className,
}: TelemetryProps): ReactElement {
  const rows = useMemo(() => {
    const cap = Math.max(1, maxRows);
    const start = Math.max(0, entries.length - cap);
    // slice → reverse gives newest-first without mutating the caller's array.
    return entries.slice(start).reverse();
  }, [entries, maxRows]);

  return (
    <Frame
      title={title}
      status={status ?? `${rows.length}/${maxRows}`}
      scan
      className={className}
      bodyClassName={styles.body}
    >
      <ol className={styles.list} aria-live="polite" aria-relevant="additions">
        {rows.length === 0 ? (
          <li className={`${styles.row} ${styles.debug}`}>
            <span className={styles.mark}>::</span>
            <span className={styles.text}>awaiting telemetry stream</span>
          </li>
        ) : (
          rows.map((entry) => {
            const level = normalizeLevel(entry.level);
            return (
              <li key={entry.id} className={`${styles.row} ${LEVEL_CLASS[level]}`}>
                {showTime ? (
                  <span className={styles.time}>{formatTime(entry.ts ?? Date.now())}</span>
                ) : null}
                <span className={styles.mark} aria-hidden="true">
                  {LEVEL_MARK[level]}
                </span>
                {entry.source !== undefined ? (
                  <span className={styles.source}>{entry.source}</span>
                ) : null}
                <span className={styles.text} title={entry.text}>
                  {entry.text}
                </span>
              </li>
            );
          })
        )}
      </ol>
      <span className={styles.fadeTop} aria-hidden="true" />
      <span className={styles.fadeBottom} aria-hidden="true" />
    </Frame>
  );
}

export default Telemetry;

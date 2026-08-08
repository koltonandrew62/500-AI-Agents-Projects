/**
 * StatusChips — the bracketed subsystem pills under the brand block.
 *
 * Each chip is `[ • LABEL ]` where the leading dot pulses while the subsystem
 * is live and sits dark when it is not. Tone is derived from `active` unless
 * the caller overrides it, so a degraded-but-up subsystem can read amber
 * without the caller having to lie about `active`.
 */
import type { ReactElement } from 'react';
import styles from './StatusChips.module.css';

export type ChipTone = 'cyan' | 'amber' | 'red' | 'dim';

export interface StatusChip {
  /** Stable identity. Defaults to `label`, which is fine when labels are unique. */
  id?: string;
  /** Display text, rendered uppercase. */
  label: string;
  /** Drives the pulsing dot and the lit/unlit treatment. */
  active: boolean;
  /** Overrides the tone derived from `active`. */
  tone?: ChipTone;
  /** Optional trailing detail, e.g. a latency or a count. */
  detail?: string;
  /** Native title / accessible description. */
  hint?: string;
  /** Per-chip click handler. Makes just this chip an interactive button. */
  onClick?: () => void;
}

export interface StatusChipsProps {
  chips: readonly StatusChip[];
  /** Wrap onto multiple lines instead of a single row. Default true. */
  wrap?: boolean;
  /** Fires when any chip is clicked; makes every chip a focusable button. */
  onSelect?: (id: string, chip: StatusChip) => void;
  className?: string;
}

const TONE_CLASS: Record<ChipTone, string> = {
  cyan: styles.cyan,
  amber: styles.amber,
  red: styles.red,
  dim: styles.dim,
};

export function StatusChips({
  chips,
  wrap = true,
  onSelect,
  className,
}: StatusChipsProps): ReactElement {
  const listClasses = [styles.list, wrap ? styles.wrap : '', className ?? '']
    .filter(Boolean)
    .join(' ');

  return (
    <ul className={listClasses}>
      {chips.map((chip) => {
        const key = chip.id ?? chip.label;
        const tone: ChipTone = chip.tone ?? (chip.active ? 'cyan' : 'dim');
        const handler = chip.onClick ?? (onSelect ? () => onSelect(key, chip) : undefined);
        const classes = [styles.chip, TONE_CLASS[tone], chip.active ? styles.on : styles.off].join(
          ' ',
        );

        const inner = (
          <>
            <span className={styles.bracket} aria-hidden="true">
              [
            </span>
            <span className={styles.dot} aria-hidden="true" />
            <span className={styles.text}>{chip.label}</span>
            {chip.detail !== undefined ? (
              <span className={styles.detail}>{chip.detail}</span>
            ) : null}
            <span className={styles.bracket} aria-hidden="true">
              ]
            </span>
          </>
        );

        return (
          <li key={key} className={styles.item}>
            {handler ? (
              <button
                type="button"
                className={classes}
                title={chip.hint}
                aria-pressed={chip.active}
                onClick={handler}
              >
                {inner}
              </button>
            ) : (
              <span
                className={classes}
                title={chip.hint}
                role="status"
                aria-label={`${chip.label} ${chip.active ? 'online' : 'offline'}`}
              >
                {inner}
              </span>
            )}
          </li>
        );
      })}
    </ul>
  );
}

export default StatusChips;

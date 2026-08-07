/**
 * Frame — the shared corner-bracket panel wrapper.
 *
 * Every HUD panel composes this. It renders four bracket corners, a masked
 * hairline, an optional uppercase title row with a trailing rule, and an
 * optional slow vertical sweep. It never sets its own size: the parent grid
 * decides, and the body region is a flex child that can scroll.
 */
import type { ReactElement, ReactNode } from 'react';
import styles from './Frame.module.css';

export type FrameTone = 'cyan' | 'amber' | 'red' | 'dim';

export interface FrameProps {
  /** Uppercase panel label, e.g. "SYSTEM VITALS". Omit for a bare frame. */
  title?: string;
  /** Small right-aligned status text in the title row, e.g. "NOMINAL". */
  status?: string;
  /** Accent color for corners + title. Defaults to `'cyan'`. */
  tone?: FrameTone;
  /** Draw CRT scanlines over the body. Default `false`. */
  scan?: boolean;
  /** Animate a slow luminous band down the panel. Default `false`. */
  sweep?: boolean;
  /** Remove body padding — for edge-to-edge canvases. Default `false`. */
  flush?: boolean;
  /** Extra class on the outer frame element. */
  className?: string;
  /** Extra class on the inner body element. */
  bodyClassName?: string;
  children?: ReactNode;
}

const TONE_CLASS: Record<FrameTone, string> = {
  cyan: '',
  amber: styles.toneAmber,
  red: styles.toneRed,
  dim: styles.toneDim,
};

export function Frame({
  title,
  status,
  tone = 'cyan',
  scan = false,
  sweep = false,
  flush = false,
  className,
  bodyClassName,
  children,
}: FrameProps): ReactElement {
  const classes = [
    styles.frame,
    TONE_CLASS[tone],
    scan ? styles.scan : '',
    flush ? styles.flush : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <section className={classes}>
      <span className={`${styles.corner} ${styles.tl}`} aria-hidden="true" />
      <span className={`${styles.corner} ${styles.tr}`} aria-hidden="true" />
      <span className={`${styles.corner} ${styles.bl}`} aria-hidden="true" />
      <span className={`${styles.corner} ${styles.br}`} aria-hidden="true" />
      {sweep ? <span className={styles.sweep} aria-hidden="true" /> : null}

      {title !== undefined ? (
        <header className={styles.head}>
          <h2 className={styles.title}>{title}</h2>
          <span className={styles.rule} aria-hidden="true" />
          {status !== undefined ? <span className={styles.status}>{status}</span> : null}
        </header>
      ) : null}

      <div className={`${styles.body} ${bodyClassName ?? ''}`.trim()}>{children}</div>
    </section>
  );
}

export default Frame;

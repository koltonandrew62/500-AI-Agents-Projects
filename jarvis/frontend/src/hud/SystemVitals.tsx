/**
 * SystemVitals — labelled horizontal bar gauges for CPU / MEM / DISK / NET / PWR.
 *
 * Values arrive from the `telemetry` websocket frame at 1Hz, which is far too
 * coarse to look alive, so each bar keeps its own smoothed value and eases
 * toward the target on every animation frame. Threshold coloring is derived
 * from the smoothed value, so a bar warms to amber as it fills rather than
 * snapping a second later.
 */
import { useMemo, useRef, useState } from 'react';
import type { ReactElement } from 'react';
import { Frame } from './Frame';
import type { FrameTone } from './Frame';
import { approach, clamp, useRafLoop, usePrefersReducedMotion } from './hooks';
import styles from './SystemVitals.module.css';

export type VitalTone = 'cyan' | 'amber' | 'red';

export interface VitalGauge {
  /** Stable key + display label, e.g. `"CPU"`. Rendered uppercase. */
  label: string;
  /** 0–100 percentage used for the bar fill. */
  percent: number;
  /** Readout text. Defaults to `"NN%"` derived from `percent`. */
  readout?: string;
  /** Fill fraction above which the bar turns amber. Default 0.72. */
  warnAt?: number;
  /** Fill fraction above which the bar turns red. Default 0.89. */
  criticalAt?: number;
  /** Invert thresholds — low is bad (used for PWR / battery). Default false. */
  invert?: boolean;
}

export interface SystemVitalsProps {
  /** CPU load, 0–100. */
  cpu?: number;
  /** Memory used, 0–100. */
  mem?: number;
  /** Disk used, 0–100. */
  disk?: number;
  /** Aggregate network throughput in MB/s (up + down). */
  netUp?: number;
  netDown?: number;
  /** Scale that maps net MB/s onto the 0–100 bar. Default 12. */
  netScaleMbs?: number;
  /** Battery percentage, or `null` when on mains / unavailable. */
  battery?: number | null;
  /** Replaces the derived gauge set entirely when provided. */
  gauges?: readonly VitalGauge[];
  /** Panel title. Default `"SYSTEM VITALS"`. */
  title?: string;
  /** Right-aligned status text in the frame header. */
  status?: string;
  className?: string;
}

const WARN_DEFAULT = 0.72;
const CRIT_DEFAULT = 0.89;

const TONE_CLASS: Record<VitalTone, string> = {
  cyan: '',
  amber: styles.amber,
  red: styles.red,
};

function toneFor(gauge: VitalGauge, fill: number): VitalTone {
  const warn = gauge.warnAt ?? WARN_DEFAULT;
  const crit = gauge.criticalAt ?? CRIT_DEFAULT;
  if (gauge.invert === true) {
    if (fill <= 1 - crit) return 'red';
    if (fill <= 1 - warn) return 'amber';
    return 'cyan';
  }
  if (fill >= crit) return 'red';
  if (fill >= warn) return 'amber';
  return 'cyan';
}

/** Bars are drawn as N discrete segments — a solid gradient looks like a web app. */
const SEGMENTS = 28;

export function SystemVitals({
  cpu = 0,
  mem = 0,
  disk = 0,
  netUp = 0,
  netDown = 0,
  netScaleMbs = 12,
  battery = null,
  gauges,
  title = 'SYSTEM VITALS',
  status,
  className,
}: SystemVitalsProps): ReactElement {
  const reduced = usePrefersReducedMotion();

  const resolved = useMemo<readonly VitalGauge[]>(() => {
    if (gauges) return gauges;
    const netTotal = Math.max(netUp, 0) + Math.max(netDown, 0);
    const list: VitalGauge[] = [
      { label: 'CPU', percent: cpu, readout: `${cpu.toFixed(0)}%` },
      { label: 'MEM', percent: mem, readout: `${mem.toFixed(0)}%` },
      { label: 'DISK', percent: disk, readout: `${disk.toFixed(0)}%`, warnAt: 0.8, criticalAt: 0.93 },
      {
        label: 'NET',
        percent: (netTotal / Math.max(netScaleMbs, 0.001)) * 100,
        readout: `${netTotal.toFixed(1)} MB/S`,
        warnAt: 0.85,
        criticalAt: 0.97,
      },
    ];
    list.push(
      battery === null
        ? { label: 'PWR', percent: 100, readout: 'MAINS' }
        : { label: 'PWR', percent: battery, readout: `${battery.toFixed(0)}%`, invert: true },
    );
    return list;
  }, [gauges, cpu, mem, disk, netUp, netDown, netScaleMbs, battery]);

  // Smoothed fill fractions, one per gauge, keyed positionally.
  const smoothRef = useRef<number[]>([]);
  const targetsRef = useRef<number[]>([]);
  targetsRef.current = resolved.map((g) => clamp(g.percent / 100));
  const [fills, setFills] = useState<number[]>(() => targetsRef.current.slice());

  useRafLoop((delta) => {
    const targets = targetsRef.current;
    const smooth = smoothRef.current;
    if (smooth.length !== targets.length) smooth.length = targets.length;

    let changed = false;
    for (let i = 0; i < targets.length; i += 1) {
      const current = smooth[i] ?? 0;
      // Rise briskly, fall gently — a load spike should read as a spike.
      const rate = targets[i] > current ? 7 : 3.2;
      const next = approach(current, targets[i], rate, delta);
      if (Math.abs(next - current) > 0.0005) changed = true;
      smooth[i] = next;
    }
    if (changed) setFills(smooth.slice());
  }, !reduced);

  const active = reduced ? targetsRef.current : fills;

  // Any red gauge escalates the whole panel's bracket color.
  const panelTone: FrameTone = resolved.some((g, i) => toneFor(g, active[i] ?? 0) === 'red')
    ? 'red'
    : 'cyan';

  return (
    <Frame
      title={title}
      status={status}
      tone={panelTone}
      className={className}
      bodyClassName={styles.body}
    >
      <ul className={styles.list}>
        {resolved.map((gauge, index) => {
          const fill = active[index] ?? 0;
          const tone = toneFor(gauge, fill);
          const lit = Math.round(fill * SEGMENTS);
          return (
            <li key={gauge.label} className={`${styles.row} ${TONE_CLASS[tone]}`.trim()}>
              <span className={styles.label}>{gauge.label}</span>
              <div
                className={styles.track}
                role="meter"
                aria-label={gauge.label}
                aria-valuenow={Math.round(fill * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <span className={styles.fill} style={{ width: `${(fill * 100).toFixed(2)}%` }} />
                <span className={styles.segments} aria-hidden="true" />
                <span
                  className={styles.head}
                  style={{ left: `${(fill * 100).toFixed(2)}%` }}
                  aria-hidden="true"
                />
                <span className={styles.ticks} aria-hidden="true" data-lit={lit} />
              </div>
              <span className={styles.readout}>
                {gauge.readout ?? `${Math.round(gauge.percent)}%`}
              </span>
            </li>
          );
        })}
      </ul>
    </Frame>
  );
}

export default SystemVitals;

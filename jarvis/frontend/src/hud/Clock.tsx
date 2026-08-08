/**
 * Clock — the top-right time block plus a geo/coordinate readout.
 *
 * Centisecond precision means this must repaint ~100 times a second, which is
 * far too often for React state. Instead the loop writes `textContent` on two
 * refs; React renders exactly once per date change. Under reduced motion the
 * centiseconds field is frozen and the clock ticks once per second via
 * `setInterval`, which is both calmer and cheaper.
 */
import { useEffect, useRef, useState } from 'react';
import type { ReactElement } from 'react';
import { useRafLoop, usePrefersReducedMotion } from './hooks';
import styles from './Clock.module.css';

export interface GeoReadout {
  /** Decimal degrees; rendered as DDD°MM'SS" with hemisphere. */
  lat: number;
  lon: number;
  /** Metres above sea level. */
  altM?: number;
  /** Place label, e.g. `"MALIBU · CA"`. */
  place?: string;
}

export interface ClockProps {
  /** Small label above the time. Default `"LOCAL TIME"`. */
  label?: string;
  /** Coordinate block beneath the clock. Omit to hide it. */
  geo?: GeoReadout | null;
  /** Show the centisecond field. Default true. */
  showCentis?: boolean;
  /** 24-hour clock. Default true. */
  hour24?: boolean;
  /** Extra right-aligned status line, e.g. an uptime string. */
  note?: string;
  className?: string;
}

const pad = (value: number, width = 2): string => String(Math.floor(value)).padStart(width, '0');

function formatDate(date: Date): string {
  const days = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'];
  const months = [
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
  ];
  return `${days[date.getDay()]} ${pad(date.getDate())} ${months[date.getMonth()]} ${date.getFullYear()}`;
}

/** Decimal degrees → DDD°MM'SS"H, the way a HUD would actually print it. */
function formatCoord(value: number, axis: 'lat' | 'lon'): string {
  const hemisphere = axis === 'lat' ? (value >= 0 ? 'N' : 'S') : value >= 0 ? 'E' : 'W';
  const abs = Math.abs(value);
  const degrees = Math.floor(abs);
  const minutesFloat = (abs - degrees) * 60;
  const minutes = Math.floor(minutesFloat);
  const seconds = Math.round((minutesFloat - minutes) * 60);
  const width = axis === 'lat' ? 2 : 3;
  return `${pad(degrees, width)}°${pad(minutes)}'${pad(seconds)}"${hemisphere}`;
}

function timeParts(date: Date, hour24: boolean): { hms: string; suffix: string } {
  let hours = date.getHours();
  let suffix = '';
  if (!hour24) {
    suffix = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12 === 0 ? 12 : hours % 12;
  }
  return {
    hms: `${pad(hours)}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`,
    suffix,
  };
}

export function Clock({
  label = 'LOCAL TIME',
  geo = null,
  showCentis = true,
  hour24 = true,
  note,
  className,
}: ClockProps): ReactElement {
  const reduced = usePrefersReducedMotion();
  const hmsRef = useRef<HTMLSpanElement>(null);
  const centisRef = useRef<HTMLSpanElement>(null);

  // Only the date is React state — it changes once a day, not 100x a second.
  const [dateLabel, setDateLabel] = useState(() => formatDate(new Date()));
  const dateLabelRef = useRef(dateLabel);
  dateLabelRef.current = dateLabel;

  const write = (): void => {
    const now = new Date();
    const { hms, suffix } = timeParts(now, hour24);
    if (hmsRef.current) hmsRef.current.textContent = suffix ? `${hms} ${suffix}` : hms;
    if (centisRef.current && showCentis) {
      centisRef.current.textContent = pad(now.getMilliseconds() / 10);
    }
    const nextDate = formatDate(now);
    if (nextDate !== dateLabelRef.current) setDateLabel(nextDate);
  };

  useRafLoop(write, !reduced);

  useEffect(() => {
    write();
    if (!reduced) return;
    const timer = window.setInterval(write, 1000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced, hour24, showCentis]);

  const tz = typeof Intl !== 'undefined' ? Intl.DateTimeFormat().resolvedOptions().timeZone : '';

  return (
    <div className={`${styles.wrap} ${className ?? ''}`.trim()}>
      <div className={styles.head}>
        <span className={styles.label}>{label}</span>
        <span className={styles.tz}>{tz}</span>
      </div>

      <div className={styles.timeRow}>
        <span className={styles.hms} ref={hmsRef} />
        {showCentis ? (
          <>
            <span className={styles.dot}>.</span>
            <span className={styles.centis} ref={centisRef}>
              00
            </span>
          </>
        ) : null}
      </div>

      <div className={styles.date}>{dateLabel}</div>

      {geo ? (
        <dl className={styles.geo}>
          <div className={styles.geoRow}>
            <dt className={styles.geoKey}>LAT</dt>
            <dd className={styles.geoVal}>{formatCoord(geo.lat, 'lat')}</dd>
          </div>
          <div className={styles.geoRow}>
            <dt className={styles.geoKey}>LON</dt>
            <dd className={styles.geoVal}>{formatCoord(geo.lon, 'lon')}</dd>
          </div>
          {geo.altM !== undefined ? (
            <div className={styles.geoRow}>
              <dt className={styles.geoKey}>ALT</dt>
              <dd className={styles.geoVal}>{`${Math.round(geo.altM)} M`}</dd>
            </div>
          ) : null}
          {geo.place !== undefined ? <div className={styles.place}>{geo.place}</div> : null}
        </dl>
      ) : null}

      {note !== undefined ? <div className={styles.note}>{note}</div> : null}
    </div>
  );
}

export default Clock;

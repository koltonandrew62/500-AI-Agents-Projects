/**
 * Reticle — the orbital ring assembly that surrounds the particle core.
 *
 * Pure SVG on a 100x100 viewBox (matching `Reticle.module.css`, whose stroke
 * widths and 2.4px label size are authored in those units). Nothing here
 * re-renders per frame: `useRafLoop` mutates `transform` on a handful of <g>
 * refs directly, so a 60fps ring costs zero React work.
 *
 * Geometry, outward-in:
 *   r=48  8 orbiting triangular markers  (fast, forward)
 *   r=45  dashed arc ring + tick collar  (medium, reverse)
 *   r=39  dotted ring                    (medium, forward)
 *   r=31  inner dotted ring              (fast, reverse)
 *   plus four static corner brackets and a rotating quadrant label set.
 */
import { useMemo, useRef } from 'react';
import type { ReactElement } from 'react';
import type { CoreState } from './CoreSphere';
import { clamp, useRafLoop, usePrefersReducedMotion } from './hooks';
import styles from './Reticle.module.css';

export interface ReticleProps {
  /** Assistant state — sets base orbital speed and (on error) the amber tint. */
  state?: CoreState;
  /** 0–1 agitation; adds up to +120% speed on top of the state baseline. */
  intensity?: number;
  /** Quadrant labels drawn on the tick collar. Must be 4 entries. */
  labels?: readonly [string, string, string, string];
  className?: string;
}

/** Degrees per second for the reference (outer marker) orbit, per state. */
const STATE_SPEED: Record<CoreState, number> = {
  idle: 6,
  listening: 13,
  thinking: 30,
  speaking: 18,
  error: 9,
};

const RINGS = {
  markers: 48,
  arc: 45,
  collar: 43.2,
  dot: 39,
  hairOuter: 35.5,
  dotInner: 31,
  hairInner: 26,
} as const;

const MARKER_COUNT = 8;
const TICK_COUNT = 72;
const DEFAULT_LABELS = ['N', 'E', 'S', 'W'] as const;

interface Tick {
  key: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  major: boolean;
}

/** Polar → cartesian about the 50,50 centre. 0deg points up. */
function polar(radius: number, degrees: number): { x: number; y: number } {
  const rad = ((degrees - 90) * Math.PI) / 180;
  return { x: 50 + Math.cos(rad) * radius, y: 50 + Math.sin(rad) * radius };
}

function buildTicks(): Tick[] {
  const ticks: Tick[] = [];
  for (let i = 0; i < TICK_COUNT; i += 1) {
    const angle = (360 / TICK_COUNT) * i;
    const major = i % 6 === 0;
    const inner = polar(RINGS.collar - (major ? 3.1 : 1.5), angle);
    const outer = polar(RINGS.collar, angle);
    ticks.push({ key: `t${i}`, x1: inner.x, y1: inner.y, x2: outer.x, y2: outer.y, major });
  }
  return ticks;
}

/** An upward-pointing triangle of `size` centred on the origin. */
function trianglePoints(size: number): string {
  const h = size * 0.95;
  return `0,${-h} ${size * 0.62},${h * 0.62} ${-size * 0.62},${h * 0.62}`;
}

export function Reticle({
  state = 'idle',
  intensity = 0,
  labels = DEFAULT_LABELS,
  className,
}: ReticleProps): ReactElement {
  const reduced = usePrefersReducedMotion();

  const markersRef = useRef<SVGGElement>(null);
  const arcRef = useRef<SVGGElement>(null);
  const collarRef = useRef<SVGGElement>(null);
  const dotRef = useRef<SVGGElement>(null);
  const dotInnerRef = useRef<SVGGElement>(null);

  // Live props in refs: the loop reads them without ever restarting.
  const speedRef = useRef(STATE_SPEED[state]);
  speedRef.current = STATE_SPEED[state] * (1 + clamp(intensity) * 1.2);

  const ticks = useMemo(buildTicks, []);
  const markers = useMemo(
    () =>
      Array.from({ length: MARKER_COUNT }, (_, i) => {
        const angle = (360 / MARKER_COUNT) * i;
        const point = polar(RINGS.markers, angle);
        return { key: `m${i}`, angle, ...point, hollow: i % 2 === 1 };
      }),
    [],
  );
  const quadrants = useMemo(
    () =>
      labels.slice(0, 4).map((text, i) => {
        const point = polar(RINGS.collar + 3.4, i * 90);
        return { key: `q${i}`, text, ...point };
      }),
    [labels],
  );

  // Accumulated angles, mutated in place each frame.
  const anglesRef = useRef({ markers: 0, arc: 0, collar: 0, dot: 0, dotInner: 0 });

  useRafLoop((delta) => {
    const base = speedRef.current * delta;
    const a = anglesRef.current;
    a.markers = (a.markers + base) % 360;
    a.arc = (a.arc - base * 0.62) % 360;
    a.collar = (a.collar + base * 0.26) % 360;
    a.dot = (a.dot + base * 0.85) % 360;
    a.dotInner = (a.dotInner - base * 1.35) % 360;

    const spin = (node: SVGGElement | null, angle: number): void => {
      if (node) node.setAttribute('transform', `rotate(${angle.toFixed(2)} 50 50)`);
    };
    spin(markersRef.current, a.markers);
    spin(arcRef.current, a.arc);
    spin(collarRef.current, a.collar);
    spin(dotRef.current, a.dot);
    spin(dotInnerRef.current, a.dotInner);
  }, !reduced);

  const wrapClasses = [styles.wrap, state === 'error' ? styles.stateAmber : '', className ?? '']
    .filter(Boolean)
    .join(' ');

  return (
    <div className={wrapClasses} aria-hidden="true">
      <svg className={styles.svg} viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
        {/* Static hairlines — the quiet scaffold everything else moves against. */}
        <circle className={styles.hairRing} cx="50" cy="50" r={RINGS.hairOuter} />
        <circle className={styles.hairRing} cx="50" cy="50" r={RINGS.hairInner} />

        {/* Tick collar + quadrant labels: slowest element, reads as a compass. */}
        <g ref={collarRef}>
          {ticks.map((tick) => (
            <line
              key={tick.key}
              className={tick.major ? styles.tickMajor : styles.tick}
              x1={tick.x1}
              y1={tick.y1}
              x2={tick.x2}
              y2={tick.y2}
            />
          ))}
          {quadrants.map((q) => (
            <text
              key={q.key}
              className={styles.label}
              x={q.x}
              y={q.y}
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {q.text}
            </text>
          ))}
        </g>

        {/* Dashed arc ring — the segmented "gauge" band. */}
        <g ref={arcRef}>
          <circle className={styles.arcRing} cx="50" cy="50" r={RINGS.arc} />
        </g>

        {/* Two counter-rotating dotted rings. */}
        <g ref={dotRef}>
          <circle className={styles.dotRing} cx="50" cy="50" r={RINGS.dot} />
        </g>
        <g ref={dotInnerRef}>
          <circle className={styles.dotRingInner} cx="50" cy="50" r={RINGS.dotInner} />
        </g>

        {/* Orbiting triangular markers, alternating solid / hollow. */}
        <g ref={markersRef}>
          {markers.map((marker) => (
            <polygon
              key={marker.key}
              className={marker.hollow ? styles.markerHollow : styles.marker}
              points={trianglePoints(marker.hollow ? 2.0 : 1.7)}
              transform={`translate(${marker.x.toFixed(3)} ${marker.y.toFixed(3)}) rotate(${
                marker.angle + 180
              })`}
            />
          ))}
        </g>

        {/* Fixed corner brackets frame the whole assembly. */}
        <g className={styles.bracket}>
          <path d="M6 18 V6 H18" />
          <path d="M82 6 H94 V18" />
          <path d="M94 82 V94 H82" />
          <path d="M18 94 H6 V82" />
        </g>
      </svg>
    </div>
  );
}

export default Reticle;

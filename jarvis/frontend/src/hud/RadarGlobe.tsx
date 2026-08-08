/**
 * RadarGlobe — a wireframe globe under a sweeping radar arc.
 *
 * The globe is real geometry, not an image: latitude rings and meridian great
 * circles are sampled on the unit sphere, rotated about the polar axis, tilted,
 * then orthographically projected. Each circle yields two path strings — the
 * front-facing hemisphere (bright) and the back (dim) — which the RAF loop
 * writes straight onto the DOM via `setAttribute('d')`. React never re-renders
 * during rotation.
 *
 * Everything lives on a 100x100 viewBox so stroke widths read in "HUD units".
 */
import { useCallback, useLayoutEffect, useMemo, useRef } from 'react';
import type { ReactElement } from 'react';
import { Frame } from './Frame';
import { clamp, useRafLoop, usePrefersReducedMotion } from './hooks';
import styles from './RadarGlobe.module.css';

export type ContactTone = 'cyan' | 'amber' | 'red';

export interface RadarContact {
  /** Stable identity — used as the React key. */
  id: string;
  /** Latitude in degrees, -90..90. */
  lat: number;
  /** Longitude in degrees, -180..180. */
  lon: number;
  /** Dot color. Default `'cyan'`. */
  tone?: ContactTone;
  /** Optional 2–4 char tag drawn beside the dot. */
  tag?: string;
}

export interface RadarGlobeProps {
  /** Blinking markers pinned to the surface; they rotate with the globe. */
  contacts?: readonly RadarContact[];
  /** Globe rotation, degrees per second. Default 9. */
  spinDegPerSec?: number;
  /** Radar sweep rotation, degrees per second. Default 62. */
  sweepDegPerSec?: number;
  /** Axial tilt in degrees. Default 22. */
  tiltDeg?: number;
  /** Dim the whole assembly (e.g. no sensor link). Default true. */
  active?: boolean;
  /** Panel title. Default `"RADAR / GLOBE"`. */
  title?: string;
  /** Right-aligned frame status. Defaults to the contact count. */
  status?: string;
  className?: string;
}

const RADIUS = 33;
const CENTER = 50;
const SAMPLES = 64;
const LATITUDES = [-60, -40, -20, 0, 20, 40, 60] as const;
const MERIDIAN_COUNT = 8;

const TONE_CLASS: Record<ContactTone, string> = {
  cyan: styles.contactCyan,
  amber: styles.contactAmber,
  red: styles.contactRed,
};

interface Projected {
  x: number;
  y: number;
  front: boolean;
}

/** Unit-sphere point → tilted, spun, orthographically projected screen point. */
function project(latDeg: number, lonDeg: number, spinRad: number, tiltRad: number): Projected {
  const lat = (latDeg * Math.PI) / 180;
  const lon = (lonDeg * Math.PI) / 180 + spinRad;
  const cosLat = Math.cos(lat);

  const x = cosLat * Math.sin(lon);
  const y0 = Math.sin(lat);
  const z0 = cosLat * Math.cos(lon);

  // Tilt about the screen-horizontal axis so the poles lean toward the viewer.
  const y = y0 * Math.cos(tiltRad) - z0 * Math.sin(tiltRad);
  const z = y0 * Math.sin(tiltRad) + z0 * Math.cos(tiltRad);

  return { x: CENTER + x * RADIUS, y: CENTER - y * RADIUS, front: z >= 0 };
}

/**
 * Samples a closed curve and splits it into a bright front path and a dim back
 * path, starting a new subpath (`M`) whenever the curve crosses the limb.
 */
function splitPath(
  sample: (t: number) => Projected,
  count: number,
): { front: string; back: string } {
  let front = '';
  let back = '';
  let prevFront: boolean | null = null;

  for (let i = 0; i <= count; i += 1) {
    const point = sample(i / count);
    const cmd = point.front === prevFront ? 'L' : 'M';
    const segment = `${cmd}${point.x.toFixed(2)} ${point.y.toFixed(2)}`;
    if (point.front) front += segment;
    else back += segment;
    prevFront = point.front;
  }
  return { front, back };
}

interface CircleSpec {
  key: string;
  /** Maps t∈[0,1] to a (lat, lon) pair on the sphere. */
  at: (t: number) => { lat: number; lon: number };
  major: boolean;
}

function buildCircles(): CircleSpec[] {
  const circles: CircleSpec[] = LATITUDES.map((lat) => ({
    key: `lat${lat}`,
    at: (t: number) => ({ lat, lon: t * 360 }),
    major: lat === 0,
  }));

  for (let i = 0; i < MERIDIAN_COUNT; i += 1) {
    const lon = (180 / MERIDIAN_COUNT) * i;
    // Sweeping latitude through a full turn traces the meridian AND its
    // antipode — i.e. one complete great circle.
    circles.push({
      key: `lon${i}`,
      at: (t: number) => ({ lat: t * 360, lon }),
      major: i === 0,
    });
  }
  return circles;
}

export function RadarGlobe({
  contacts = [],
  spinDegPerSec = 9,
  sweepDegPerSec = 62,
  tiltDeg = 22,
  active = true,
  title = 'RADAR / GLOBE',
  status,
  className,
}: RadarGlobeProps): ReactElement {
  const reduced = usePrefersReducedMotion();
  const circles = useMemo(buildCircles, []);

  const frontRefs = useRef<(SVGPathElement | null)[]>([]);
  const backRefs = useRef<(SVGPathElement | null)[]>([]);
  const contactRefs = useRef<(SVGGElement | null)[]>([]);
  const sweepRef = useRef<SVGGElement>(null);

  const tiltRad = (tiltDeg * Math.PI) / 180;
  const contactsRef = useRef<readonly RadarContact[]>(contacts);
  contactsRef.current = contacts;

  /** Writes one full frame of geometry for a given spin angle. */
  const draw = useCallback(
    (spinRad: number): void => {
      circles.forEach((circle, index) => {
        const { front, back } = splitPath(
          (t) => {
            const { lat, lon } = circle.at(t);
            return project(lat, lon, spinRad, tiltRad);
          },
          SAMPLES,
        );
        frontRefs.current[index]?.setAttribute('d', front);
        backRefs.current[index]?.setAttribute('d', back);
      });

      contactsRef.current.forEach((contact, index) => {
        const node = contactRefs.current[index];
        if (!node) return;
        const point = project(contact.lat, contact.lon, spinRad, tiltRad);
        node.setAttribute('transform', `translate(${point.x.toFixed(2)} ${point.y.toFixed(2)})`);
        // Back-facing contacts fade rather than vanish — the globe reads as glass.
        node.setAttribute('opacity', point.front ? '1' : '0.22');
      });
    },
    [circles, tiltRad],
  );

  const spinRef = useRef(0.6);

  // One synchronous pass so contacts are positioned on the very first paint,
  // and so the reduced-motion pose is complete without ever starting a loop.
  useLayoutEffect(() => {
    draw(spinRef.current);
  }, [draw, contacts, reduced]);

  const sweepAngleRef = useRef(0);
  const spinRateRef = useRef(spinDegPerSec);
  spinRateRef.current = spinDegPerSec;
  const sweepRateRef = useRef(sweepDegPerSec);
  sweepRateRef.current = sweepDegPerSec;

  useRafLoop((delta) => {
    spinRef.current += (spinRateRef.current * delta * Math.PI) / 180;
    sweepAngleRef.current = (sweepAngleRef.current + sweepRateRef.current * delta) % 360;
    draw(spinRef.current);
    sweepRef.current?.setAttribute(
      'transform',
      `rotate(${sweepAngleRef.current.toFixed(2)} ${CENTER} ${CENTER})`,
    );
  }, !reduced);

  // Static pose for reduced motion / first paint before the loop runs.
  const initial = useMemo(() => {
    const map = new Map<string, { front: string; back: string }>();
    circles.forEach((circle) => {
      map.set(
        circle.key,
        splitPath((t) => {
          const { lat, lon } = circle.at(t);
          return project(lat, lon, 0.6, tiltRad);
        }, SAMPLES),
      );
    });
    return map;
  }, [circles, tiltRad]);

  const opacity = clamp(active ? 1 : 0.4, 0, 1);

  return (
    <Frame
      title={title}
      status={status ?? `${contacts.length} CONTACTS`}
      tone={active ? 'cyan' : 'dim'}
      flush
      className={className}
      bodyClassName={styles.body}
    >
      <svg
        className={styles.svg}
        viewBox="0 0 100 100"
        preserveAspectRatio="xMidYMid meet"
        style={{ opacity }}
        aria-hidden="true"
      >
        {/* Limb + range rings */}
        <circle className={styles.limb} cx={CENTER} cy={CENTER} r={RADIUS} />
        <circle className={styles.range} cx={CENTER} cy={CENTER} r={RADIUS * 1.22} />
        <circle className={styles.range} cx={CENTER} cy={CENTER} r={RADIUS * 1.4} />
        <line className={styles.cross} x1={CENTER - 46} y1={CENTER} x2={CENTER + 46} y2={CENTER} />
        <line className={styles.cross} x1={CENTER} y1={CENTER - 46} x2={CENTER} y2={CENTER + 46} />

        {/* Back hemisphere first, so front strokes overdraw it. */}
        {circles.map((circle, index) => (
          <path
            key={`b-${circle.key}`}
            ref={(node) => {
              backRefs.current[index] = node;
            }}
            className={styles.wireBack}
            d={initial.get(circle.key)?.back ?? ''}
          />
        ))}
        {circles.map((circle, index) => (
          <path
            key={`f-${circle.key}`}
            ref={(node) => {
              frontRefs.current[index] = node;
            }}
            className={circle.major ? styles.wireMajor : styles.wire}
            d={initial.get(circle.key)?.front ?? ''}
          />
        ))}

        {/* Radar sweep: a wedge with a hot leading edge. */}
        <g ref={sweepRef}>
          <path
            className={styles.sweepWedge}
            d={`M${CENTER} ${CENTER} L${CENTER} ${CENTER - RADIUS * 1.4} A${RADIUS * 1.4} ${
              RADIUS * 1.4
            } 0 0 1 ${(CENTER + RADIUS * 1.4 * Math.sin(Math.PI / 4)).toFixed(2)} ${(
              CENTER -
              RADIUS * 1.4 * Math.cos(Math.PI / 4)
            ).toFixed(2)} Z`}
          />
          <line
            className={styles.sweepEdge}
            x1={CENTER}
            y1={CENTER}
            x2={CENTER}
            y2={CENTER - RADIUS * 1.4}
          />
        </g>

        {/* Surface contacts */}
        {contacts.map((contact, index) => (
          <g
            key={contact.id}
            ref={(node) => {
              contactRefs.current[index] = node;
            }}
            className={`${styles.contact} ${TONE_CLASS[contact.tone ?? 'cyan']}`}
            style={{ animationDelay: `${(index % 7) * 0.29}s` }}
          >
            <circle className={styles.contactRing} r="2.6" />
            <circle className={styles.contactDot} r="1.05" />
            {contact.tag !== undefined ? (
              <text className={styles.contactTag} x="3.6" y="1">
                {contact.tag}
              </text>
            ) : null}
          </g>
        ))}
      </svg>
    </Frame>
  );
}

export default RadarGlobe;

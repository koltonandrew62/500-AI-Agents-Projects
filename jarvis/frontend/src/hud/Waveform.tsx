/**
 * Waveform — mirrored bar spectrum on a 2D canvas.
 *
 * `levels` is resampled onto a fixed bar count so the visual is stable no
 * matter what FFT size the caller uses. Each bar keeps its own smoothed value
 * with fast attack / slow release, which is what makes a voice meter feel
 * responsive rather than laggy; when input goes silent the bars decay into a
 * low idle ripple instead of collapsing to a dead flat line.
 *
 * Canvas is sized in device pixels via ResizeObserver — no window listeners,
 * and the RAF loop is fully cancelled on unmount.
 */
import { useEffect, useRef } from 'react';
import type { ReactElement } from 'react';
import { Frame } from './Frame';
import { approach, clamp, useElementSize, useRafLoop, usePrefersReducedMotion } from './hooks';
import styles from './Waveform.module.css';

export type WaveformTone = 'cyan' | 'amber' | 'red';

export interface WaveformProps {
  /**
   * Normalized magnitudes, 0–1, low frequency first. Any length; it is
   * resampled onto `bars`. Omit (or pass empty) to show the idle ripple.
   */
  levels?: Float32Array | null;
  /** Number of mirrored bars drawn. Default 48. */
  bars?: number;
  /** Master gain applied to `levels` before drawing. Default 1. */
  gain?: number;
  /** When false, the meter dims and decays to idle. Default true. */
  active?: boolean;
  /** Accent color. Default `'cyan'`. */
  tone?: WaveformTone;
  /** Render without the Frame chrome (for embedding in the command bar). */
  bare?: boolean;
  /** Panel title. Default `"AUDIO"`. */
  title?: string;
  /** Right-aligned frame status. */
  status?: string;
  className?: string;
}

const TONE_COLORS: Record<WaveformTone, { hot: string; cool: string }> = {
  cyan: { hot: '#7ee8ff', cool: '#1b8fe0' },
  amber: { hot: '#ffb545', cool: '#ff4d5e' },
  red: { hot: '#ff4d5e', cool: '#ffb545' },
};

/** Idle ripple amplitude — the meter is never completely dead. */
const IDLE_FLOOR = 0.035;

/** Nearest-neighbour resample of `levels` into `out`, with gain + clamp. */
function resample(levels: Float32Array | null | undefined, out: Float32Array, gain: number): void {
  const count = out.length;
  if (!levels || levels.length === 0) {
    out.fill(0);
    return;
  }
  const ratio = levels.length / count;
  for (let i = 0; i < count; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.max(start + 1, Math.floor((i + 1) * ratio));
    let peak = 0;
    for (let j = start; j < end && j < levels.length; j += 1) {
      const value = Math.abs(levels[j]);
      if (value > peak) peak = value;
    }
    out[i] = clamp(peak * gain);
  }
}

export function Waveform({
  levels = null,
  bars = 48,
  gain = 1,
  active = true,
  tone = 'cyan',
  bare = false,
  title = 'AUDIO',
  status,
  className,
}: WaveformProps): ReactElement {
  const reduced = usePrefersReducedMotion();
  const [hostRef, size] = useElementSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const count = Math.max(8, Math.floor(bars));
  const targetRef = useRef<Float32Array>(new Float32Array(count));
  const smoothRef = useRef<Float32Array>(new Float32Array(count));
  const phaseRef = useRef(0);

  // Reallocate only when the bar budget changes.
  useEffect(() => {
    targetRef.current = new Float32Array(count);
    smoothRef.current = new Float32Array(count);
  }, [count]);

  // Latest inputs in refs — the draw loop must not restart on prop churn.
  const levelsRef = useRef<Float32Array | null>(levels);
  levelsRef.current = levels;
  const gainRef = useRef(gain);
  gainRef.current = gain;
  const activeRef = useRef(active);
  activeRef.current = active;
  const toneRef = useRef(tone);
  toneRef.current = tone;

  // Backing store resize follows layout, in device pixels.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !size) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(size.width * dpr));
    const height = Math.max(1, Math.round(size.height * dpr));
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
  }, [size]);

  const paint = (delta: number): void => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const mid = height / 2;
    const smooth = smoothRef.current;
    const target = targetRef.current;

    resample(levelsRef.current, target, gainRef.current);
    phaseRef.current += delta * 2.1;

    const enabled = activeRef.current;
    const colors = TONE_COLORS[toneRef.current];

    ctx.clearRect(0, 0, width, height);

    // Centre axis.
    ctx.strokeStyle = 'rgba(28, 111, 146, 0.55)';
    ctx.lineWidth = Math.max(1, height * 0.004);
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();

    const slot = width / smooth.length;
    const barWidth = Math.max(1, slot * 0.56);

    ctx.shadowBlur = height * 0.07;
    ctx.shadowColor = colors.hot;

    for (let i = 0; i < smooth.length; i += 1) {
      // Idle ripple: a slow travelling sine so silence still breathes.
      const ripple =
        IDLE_FLOOR * (0.6 + 0.4 * Math.sin(phaseRef.current + i * 0.42)) * (enabled ? 1 : 0.5);
      const desired = Math.max(enabled ? target[i] : 0, ripple);
      const rate = desired > smooth[i] ? 22 : 5.5;
      smooth[i] = approach(smooth[i], desired, rate, delta);

      const amp = smooth[i];
      const half = Math.max(height * 0.008, amp * mid * 0.92);
      const x = i * slot + (slot - barWidth) / 2;

      const gradient = ctx.createLinearGradient(0, mid - half, 0, mid + half);
      gradient.addColorStop(0, colors.cool);
      gradient.addColorStop(0.5, colors.hot);
      gradient.addColorStop(1, colors.cool);
      ctx.fillStyle = gradient;
      ctx.globalAlpha = enabled ? 0.55 + amp * 0.45 : 0.3;
      ctx.fillRect(x, mid - half, barWidth, half * 2);

      // Hot caps on the mirrored peaks.
      ctx.globalAlpha = enabled ? 0.9 : 0.4;
      ctx.fillStyle = colors.hot;
      const cap = Math.max(1, height * 0.008);
      ctx.fillRect(x, mid - half, barWidth, cap);
      ctx.fillRect(x, mid + half - cap, barWidth, cap);
    }

    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;
  };

  useRafLoop((delta) => paint(delta), !reduced);

  // Reduced motion still shows a truthful, static snapshot of the input.
  useEffect(() => {
    if (!reduced) return;
    paint(0.5);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced, size, levels, active, tone]);

  const canvasNode = (
    <div className={styles.host} ref={hostRef}>
      <canvas ref={canvasRef} className={styles.canvas} aria-hidden="true" />
      <span className={styles.grid} aria-hidden="true" />
    </div>
  );

  if (bare) {
    return <div className={`${styles.bare} ${className ?? ''}`.trim()}>{canvasNode}</div>;
  }

  return (
    <Frame
      title={title}
      status={status ?? (active ? 'LIVE' : 'MUTED')}
      tone={tone === 'cyan' ? 'cyan' : tone}
      flush
      className={className}
      bodyClassName={styles.body}
    >
      {canvasNode}
    </Frame>
  );
}

export default Waveform;

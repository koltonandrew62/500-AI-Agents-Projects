/**
 * getUserMedia wrapper: start/stop, attach to a <video>, capture base64 JPEG
 * stills via an offscreen canvas, and a throttled frame sender.
 *
 * Frames are held in memory only — never written to disk (CONTRACTS §7).
 */

export type WebcamState = 'idle' | 'starting' | 'live' | 'denied' | 'error';

export interface WebcamOptions {
  width?: number;
  height?: number;
  facingMode?: 'user' | 'environment';
  /** Longest edge of the captured still, px. Frames are downscaled to this. */
  captureMaxEdge?: number;
}

export interface WebcamStatus {
  state: WebcamState;
  /** Human-readable, safe to render directly in the HUD. */
  message: string;
}

const DENIED_MESSAGE =
  'CAMERA ACCESS DENIED — grant permission in your browser, then re-enable VISION.';

/** Map a getUserMedia rejection to a user-facing HUD line. */
export function describeMediaError(err: unknown): WebcamStatus {
  const name =
    typeof err === 'object' && err !== null && 'name' in err
      ? String((err as { name: unknown }).name)
      : '';

  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return { state: 'denied', message: DENIED_MESSAGE };
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return { state: 'error', message: 'NO CAMERA DETECTED ON THIS DEVICE.' };
    case 'NotReadableError':
    case 'TrackStartError':
      return { state: 'error', message: 'CAMERA IS IN USE BY ANOTHER APPLICATION.' };
    case 'OverconstrainedError':
      return { state: 'error', message: 'CAMERA CANNOT MEET THE REQUESTED RESOLUTION.' };
    default:
      return { state: 'error', message: 'CAMERA UNAVAILABLE — OPTICAL FEED OFFLINE.' };
  }
}

export function isWebcamSupported(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    typeof navigator.mediaDevices !== 'undefined' &&
    typeof navigator.mediaDevices.getUserMedia === 'function'
  );
}

export class WebcamController {
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private sendTimer: ReturnType<typeof setInterval> | null = null;
  private status: WebcamStatus = { state: 'idle', message: 'OPTICAL FEED STANDBY' };
  private readonly listeners = new Set<(status: WebcamStatus) => void>();
  private readonly opts: Required<WebcamOptions>;
  private starting: Promise<boolean> | null = null;

  constructor(options: WebcamOptions = {}) {
    this.opts = {
      width: options.width ?? 1280,
      height: options.height ?? 720,
      facingMode: options.facingMode ?? 'user',
      captureMaxEdge: options.captureMaxEdge ?? 640,
    };
  }

  getStatus(): WebcamStatus {
    return this.status;
  }

  onStatus(listener: (status: WebcamStatus) => void): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  /** Bind the live stream to a <video>; safe to call before or after start(). */
  attach(video: HTMLVideoElement | null): void {
    if (this.video && this.video !== video) {
      this.video.srcObject = null;
    }
    this.video = video;
    if (video && this.stream) {
      video.srcObject = this.stream;
      video.muted = true;
      video.playsInline = true;
      void video.play().catch(() => { /* autoplay blocked; harmless */ });
    }
  }

  async start(): Promise<boolean> {
    if (this.stream) return true;
    if (this.starting) return this.starting;

    if (!isWebcamSupported()) {
      this.setStatus({ state: 'error', message: 'THIS BROWSER DOES NOT SUPPORT CAMERA CAPTURE.' });
      return false;
    }

    this.setStatus({ state: 'starting', message: 'INITIALIZING OPTICAL FEED…' });

    this.starting = (async (): Promise<boolean> => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: this.opts.width },
            height: { ideal: this.opts.height },
            facingMode: this.opts.facingMode,
          },
          audio: false,
        });
        this.stream = stream;
        if (this.video) this.attach(this.video);
        this.setStatus({ state: 'live', message: 'OPTICAL FEED ONLINE' });
        return true;
      } catch (err) {
        this.setStatus(describeMediaError(err));
        return false;
      } finally {
        this.starting = null;
      }
    })();

    return this.starting;
  }

  stop(): void {
    this.stopSending();
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    if (this.video) this.video.srcObject = null;
    this.canvas = null;
    if (this.status.state !== 'denied') {
      this.setStatus({ state: 'idle', message: 'OPTICAL FEED STANDBY' });
    }
  }

  isLive(): boolean {
    return this.stream !== null && this.status.state === 'live';
  }

  /** Natural resolution of the live track, or null when not running. */
  getDimensions(): { width: number; height: number } | null {
    const video = this.video;
    if (!video || !video.videoWidth || !video.videoHeight) return null;
    return { width: video.videoWidth, height: video.videoHeight };
  }

  /**
   * Grab one still as a bare base64 JPEG payload (no `data:` prefix — the
   * backend contract's `jpeg_b64` field). Returns null if no frame is ready.
   */
  captureFrame(quality = 0.7): string | null {
    const video = this.video;
    if (!video || !this.stream) return null;
    const sourceW = video.videoWidth;
    const sourceH = video.videoHeight;
    if (sourceW === 0 || sourceH === 0) return null;

    const scale = Math.min(1, this.opts.captureMaxEdge / Math.max(sourceW, sourceH));
    const width = Math.max(1, Math.round(sourceW * scale));
    const height = Math.max(1, Math.round(sourceH * scale));

    let canvas = this.canvas;
    if (!canvas) {
      canvas = document.createElement('canvas');
      this.canvas = canvas;
    }
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return null;

    try {
      ctx.drawImage(video, 0, 0, width, height);
      const dataUrl = canvas.toDataURL('image/jpeg', clamp(quality, 0.1, 0.95));
      const comma = dataUrl.indexOf(',');
      return comma >= 0 ? dataUrl.slice(comma + 1) : null;
    } catch {
      // Tainted canvas or the track died mid-draw.
      return null;
    }
  }

  /**
   * Push stills to `send` at `fps` (default 1). Idempotent — calling again
   * replaces the previous schedule.
   */
  startSending(send: (jpegB64: string) => void, fps = 1, quality = 0.7): void {
    this.stopSending();
    const interval = Math.max(100, Math.round(1000 / Math.max(0.1, fps)));
    this.sendTimer = setInterval(() => {
      if (!this.isLive()) return;
      const frame = this.captureFrame(quality);
      if (frame) send(frame);
    }, interval);
  }

  stopSending(): void {
    if (this.sendTimer !== null) {
      clearInterval(this.sendTimer);
      this.sendTimer = null;
    }
  }

  isSending(): boolean {
    return this.sendTimer !== null;
  }

  /** Release everything. The controller is reusable after this. */
  dispose(): void {
    this.stop();
    this.listeners.clear();
    this.video = null;
  }

  private setStatus(next: WebcamStatus): void {
    if (this.status.state === next.state && this.status.message === next.message) return;
    this.status = next;
    for (const listener of Array.from(this.listeners)) {
      try {
        listener(next);
      } catch (err) {
        console.error('[webcam] status listener threw', err);
      }
    }
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

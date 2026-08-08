/**
 * Vision orchestrator: base64 JPEG in, `VisionResult` out.
 *
 * Per PORT-CONTRACTS.md §5, the vision strategy changed from the Python
 * edition: the browser does cheap geometry (face/hand detection via the
 * native `FaceDetector` API, degrading to 0 silently), and this server does
 * SEMANTICS — the vision model chain describes the scene, reads any visible
 * text, and returns concrete next steps. There is no local (non-LLM) object
 * detector or OCR engine in this port; `objects`/`faces`/`hands` stay at
 * their zero value here and are populated client-side.
 *
 * `analyze()`:
 *   1. Decodes the incoming base64 JPEG to a raw buffer. The frame is NEVER
 *      written to disk (PORT-CONTRACTS.md §7) — everything stays in memory.
 *   2. Short-circuits on a near-identical frame (compared against the last
 *      *analyzed* frame) by returning the cached result — this is what
 *      keeps a webcam ticking at several FPS from burning free-tier vision
 *      LLM calls on a static scene.
 *   3. Otherwise calls the injected vision-completion function and parses
 *      its reply into a `VisionResult`.
 *
 * Never throws: any failure at any stage degrades to an emptier-than-ideal
 * result rather than rejecting.
 */

import type { Message, VisionResult } from '../types.js';

/** Downsample-free "signature": bytes sampled at fixed proportional offsets. */
const SIGNATURE_SAMPLES = 256;

/** Mean absolute byte difference (0-255) below which two frames are "near-identical". */
const DEFAULT_SIMILARITY_THRESHOLD = 2.0;

const SYSTEM_PROMPT =
  "You are J.A.R.V.I.S., a real-world visual assistant looking through the " +
  "user's webcam. You are not writing a photo caption — you are helping " +
  'someone in the moment. For every frame:\n' +
  '1. Identify the concrete object(s), scene, or situation in view. Be ' +
  'specific (brand, model, part names) whenever you can tell.\n' +
  '2. If there is any readable text, labels, error messages, or displays, ' +
  'read them out and use them.\n' +
  "3. If something looks broken, unfinished, or ambiguous, say what's " +
  'wrong with it.\n' +
  '4. ALWAYS end with concrete, actionable next steps — what the user ' +
  'should do next, in order. Prefer numbered steps for anything with more ' +
  'than one action.\n' +
  'Keep it tight: a few sentences of identification, then the steps. No ' +
  "generic filler like 'this image shows'. If the user asked a specific " +
  'question, answer it directly first.';

const DEFAULT_QUESTION = 'What am I looking at, and what should I do with it?';

const FALLBACK_TEXT =
  "I can't reach my vision model right now, so I can't describe what's in " +
  'frame. Try again in a moment.';

/** Injected so this module never has a hard dependency on the concrete LLM implementation. */
export type VisionCompleteFn = (
  messages: Message[],
  imageB64: string,
  signal?: AbortSignal,
) => Promise<string>;

interface FrameCache {
  signature: Float64Array;
  question: string;
  result: VisionResult;
}

function decodeB64Jpeg(jpegB64: string): Buffer | null {
  const payload = jpegB64.startsWith('data:') ? jpegB64.split(',', 2)[1] ?? '' : jpegB64;
  if (!payload) return null;
  try {
    const buf = Buffer.from(payload, 'base64');
    return buf.length > 0 ? buf : null;
  } catch {
    return null;
  }
}

/** Minimal JPEG SOF-marker scan for width/height — no decode, no dependency. */
function readJpegDimensions(buf: Buffer): { width: number; height: number } {
  if (buf.length < 4 || buf[0] !== 0xff || buf[1] !== 0xd8) {
    return { width: 0, height: 0 };
  }
  let offset = 2;
  while (offset + 9 <= buf.length) {
    if (buf[offset] !== 0xff) {
      offset += 1;
      continue;
    }
    const marker = buf[offset + 1] as number;
    // SOF0..SOF15 markers (excluding DHT 0xc4, JPG 0xc8, DAC 0xcc) carry dimensions.
    const isSof = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd9)) {
      offset += 2;
      continue;
    }
    const segmentLength = buf.readUInt16BE(offset + 2);
    if (isSof) {
      const height = buf.readUInt16BE(offset + 5);
      const width = buf.readUInt16BE(offset + 7);
      return { width, height };
    }
    offset += 2 + segmentLength;
  }
  return { width: 0, height: 0 };
}

/** Cheap, dependency-free frame signature: bytes sampled at fixed proportional offsets. */
function frameSignature(buf: Buffer): Float64Array {
  const sig = new Float64Array(SIGNATURE_SAMPLES);
  if (buf.length === 0) return sig;
  for (let i = 0; i < SIGNATURE_SAMPLES; i++) {
    const idx = Math.floor((i / SIGNATURE_SAMPLES) * buf.length);
    sig[i] = buf[idx] ?? 0;
  }
  return sig;
}

function signaturesSimilar(a: Float64Array, b: Float64Array, threshold: number): boolean {
  if (a.length !== b.length) return false;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
  }
  return sum / a.length < threshold;
}

function buildMessages(question: string): Message[] {
  const userText = question.trim() || DEFAULT_QUESTION;
  return [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: userText },
  ];
}

export interface VisionPipelineOptions {
  /** Called with the frame (and any question) to get the LLM scene description. */
  visionComplete: VisionCompleteFn;
  similarityThreshold?: number;
  enabled?: boolean;
}

export class VisionPipeline {
  private readonly visionComplete: VisionCompleteFn;
  private readonly similarityThreshold: number;
  private enabled: boolean;
  private cache: FrameCache | null = null;
  /** Serializes cache check-and-update; one instance is meant for one connection. */
  private chain: Promise<unknown> = Promise.resolve();

  constructor(opts: VisionPipelineOptions) {
    this.visionComplete = opts.visionComplete;
    this.similarityThreshold = opts.similarityThreshold ?? DEFAULT_SIMILARITY_THRESHOLD;
    this.enabled = opts.enabled ?? true;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Decode `jpegB64` and return a merged `VisionResult`. `question` grounds
   * the LLM description (falls back to a generic "what am I looking at").
   * Never throws.
   */
  async analyze(jpegB64: string, question = '', signal?: AbortSignal): Promise<VisionResult> {
    const empty: VisionResult = { objects: [], faces: 0, hands: 0, text: '', scene: '', width: 0, height: 0 };
    if (!this.enabled) return empty;

    const raw = decodeB64Jpeg(jpegB64);
    if (raw === null) return empty;

    const { width, height } = readJpegDimensions(raw);
    const signature = frameSignature(raw);

    const cached = await this.runExclusive(() => {
      const c = this.cache;
      if (c && c.question === question && signaturesSimilar(signature, c.signature, this.similarityThreshold)) {
        return c.result;
      }
      return null;
    });
    if (cached) return cached;

    let sceneText: string;
    try {
      const reply = await this.visionComplete(buildMessages(question), jpegB64, signal);
      sceneText = reply.trim() || FALLBACK_TEXT;
    } catch (err) {
      if (isAbortError(err)) throw err;
      console.warn('[vision] scene description failed', err);
      sceneText = FALLBACK_TEXT;
    }

    const result: VisionResult = { objects: [], faces: 0, hands: 0, text: '', scene: sceneText, width, height };

    await this.runExclusive(() => {
      this.cache = { signature, question, result };
      return null;
    });

    return result;
  }

  /** Serializes cache reads/writes without blocking concurrent `analyze` calls' network work. */
  private runExclusive<T>(fn: () => T): Promise<T> {
    const result = this.chain.then(fn);
    this.chain = result.catch(() => undefined);
    return result;
  }
}

function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === 'AbortError';
}

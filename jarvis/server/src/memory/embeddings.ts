/**
 * Text embedding backend for the memory subsystem.
 *
 * `Embedder` prefers `@xenova/transformers` running `all-MiniLM-L6-v2`
 * (384-dim) but never imports it at module load time — it's a heavyweight
 * optional dependency that can take real time to load and may not be
 * installed at all. The model is loaded lazily on first use. If the import
 * or model load fails for any reason, encoding transparently falls back to
 * a deterministic hash-based embedder of the same dimensionality: recall
 * still works, just without real semantic understanding, and the memory
 * subsystem never crashes for lack of the optional dependency.
 *
 * The fallback is the expected default path in most deployments, so it is
 * built to be genuinely decent rather than a toy: each distinct token hashes
 * (via SHA-256) into a seed for a deterministic pseudo-random unit-ish
 * vector, a text's embedding is the sum of its tokens' vectors, and the
 * result is L2-normalized. Same input text always produces the same output
 * vector, across processes and restarts, with no ML model required.
 */

import { createHash } from 'node:crypto';

export const EMBED_DIM = 384;

const WORD_RE = /[a-z0-9]+/g;

type FeatureExtractionPipeline = (
  text: string,
  options: { pooling: 'mean'; normalize: boolean },
) => Promise<{ data: Float32Array | number[] }>;

/**
 * Deterministic 32-bit PRNG (mulberry32) seeded from a numeric seed. Chosen
 * over `Math.random()` purely for reproducibility — the same seed always
 * produces the same stream, which is what makes the hash-embedder stable
 * across processes and restarts.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller transform: two uniform draws -> one standard-normal draw. */
function nextGaussian(rand: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Derive a 32-bit seed from a word via SHA-256, independent of platform. */
function seedFromWord(word: string): number {
  const digest = createHash('sha256').update(word, 'utf8').digest();
  // Use the first 4 bytes — plenty of entropy for a PRNG seed, and keeps the
  // seed within a safe 32-bit integer range.
  return digest.readUInt32BE(0);
}

function hashEmbed(text: string, dim: number = EMBED_DIM): Float32Array {
  const vec = new Float64Array(dim);
  const words = text.toLowerCase().match(WORD_RE) ?? [''];
  for (const word of words) {
    const rand = mulberry32(seedFromWord(word));
    for (let i = 0; i < dim; i += 1) {
      // vec is allocated with exactly `dim` elements and i < dim, so this
      // index is provably in-bounds; TypedArray reads never return
      // `undefined` at runtime regardless of what the type says here.
      vec[i] = (vec[i] as number) + nextGaussian(rand);
    }
  }
  let normSq = 0;
  for (let i = 0; i < dim; i += 1) normSq += (vec[i] as number) * (vec[i] as number);
  const norm = Math.sqrt(normSq);
  const out = new Float32Array(dim);
  if (norm > 0) {
    for (let i = 0; i < dim; i += 1) out[i] = (vec[i] as number) / norm;
  }
  return out;
}

type Backend = 'transformers' | 'hash';

/**
 * Lazily-loaded text embedder with a deterministic offline fallback. Safe to
 * construct at import time / app startup — no model is loaded and no
 * heavyweight import happens until `encode()` is first awaited.
 */
export class Embedder {
  static readonly DIM = EMBED_DIM;

  private readonly modelName: string;
  private pipeline: FeatureExtractionPipeline | null = null;
  private backendValue: Backend | null = null;
  private loading: Promise<void> | null = null;

  constructor(modelName = 'Xenova/all-MiniLM-L6-v2') {
    this.modelName = modelName;
  }

  /** Which backend is active: 'transformers', 'hash', or null if not yet loaded. */
  get backend(): Backend | null {
    return this.backendValue;
  }

  private async ensureLoaded(): Promise<void> {
    if (this.backendValue !== null) return;
    if (!this.loading) {
      this.loading = this.loadOnce();
    }
    await this.loading;
  }

  private async loadOnce(): Promise<void> {
    try {
      // Dynamic + string-built specifier so bundlers/TS don't try to resolve
      // this optional dependency at build time when it isn't installed.
      const specifier = '@xenova/transformers';
      const mod = (await import(/* @vite-ignore */ specifier)) as {
        pipeline: (task: string, model: string) => Promise<FeatureExtractionPipeline>;
      };
      this.pipeline = await mod.pipeline('feature-extraction', this.modelName);
      this.backendValue = 'transformers';
    } catch {
      // Missing dependency, no network to fetch weights, unsupported
      // platform, etc. — degrade gracefully rather than crash the memory
      // subsystem.
      this.pipeline = null;
      this.backendValue = 'hash';
    }
  }

  private async encodeOneSync(text: string): Promise<Float32Array> {
    if (this.backendValue === 'transformers' && this.pipeline) {
      try {
        const result = await this.pipeline(text, { pooling: 'mean', normalize: true });
        return Float32Array.from(result.data);
      } catch {
        // A runtime failure mid-session (e.g. OOM) still shouldn't crash
        // recall — fall back to the hash embedder for this call only.
        return hashEmbed(text);
      }
    }
    return hashEmbed(text);
  }

  /** Batch-encode texts into 384-dim float32 vectors, in input order. */
  async encode(texts: readonly string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    await this.ensureLoaded();
    const out: Float32Array[] = [];
    for (const text of texts) {
      out.push(await this.encodeOneSync(text));
    }
    return out;
  }

  /** Convenience wrapper for encoding a single string. */
  async encodeOne(text: string): Promise<Float32Array> {
    const [vec] = await this.encode([text]);
    // encode() pushes exactly one vector per input text, so a 1-element
    // input always yields a 1-element output; this fallback is unreachable
    // in practice but keeps the return type honest.
    return vec ?? hashEmbed(text);
  }

  /** Serialize an embedding vector to raw float32 bytes for BLOB storage. */
  static toBytes(vec: Float32Array): Buffer {
    return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
  }

  /** Deserialize a BLOB column back into a float32 vector. */
  static fromBytes(data: Buffer | Uint8Array): Float32Array {
    const buf = Buffer.isBuffer(data) ? data : Buffer.from(data);
    // Copy into a fresh, aligned ArrayBuffer — the source Buffer may be a
    // view into a larger, misaligned allocation (e.g. from better-sqlite3).
    const copy = Buffer.from(buf);
    return new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
  }
}

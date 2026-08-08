/**
 * Audio: mic level metering (for the Waveform), browser SpeechRecognition
 * (STT + wake-word), and SpeechSynthesis (TTS, deepest en-GB male voice).
 * All three degrade to safe no-ops where the underlying Web API is missing.
 */

// ---- Mic level meter --------------------------------------------------

export type MicMeterState = 'idle' | 'starting' | 'live' | 'denied' | 'unsupported' | 'error';
export interface MicMeterStatus { state: MicMeterState; message: string }

function isGetUserMediaSupported(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    typeof navigator.mediaDevices !== 'undefined' &&
    typeof navigator.mediaDevices.getUserMedia === 'function'
  );
}

type AudioCtor = typeof AudioContext;
function resolveAudioContextCtor(): AudioCtor | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as { AudioContext?: AudioCtor; webkitAudioContext?: AudioCtor };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

/**
 * Wraps getUserMedia({audio}) + AnalyserNode. `read()` fills the caller's
 * Float32Array with the current time-domain waveform (range roughly -1..1).
 */
export class MicLevelMeter {
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private status: MicMeterStatus = { state: 'idle', message: 'MIC STANDBY' };
  private readonly listeners = new Set<(status: MicMeterStatus) => void>();
  private readonly fftSize: number;
  private starting: Promise<boolean> | null = null;

  constructor(options: { fftSize?: number } = {}) {
    this.fftSize = options.fftSize ?? 1024;
  }

  getStatus(): MicMeterStatus { return this.status; }
  isLive(): boolean { return this.status.state === 'live'; }
  get binCount(): number { return this.analyser?.fftSize ?? this.fftSize; }

  onStatus(listener: (status: MicMeterStatus) => void): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  async start(): Promise<boolean> {
    if (this.status.state === 'live') return true;
    if (this.starting) return this.starting;

    if (!isGetUserMediaSupported()) {
      this.setStatus({ state: 'unsupported', message: 'MIC CAPTURE NOT SUPPORTED IN THIS BROWSER.' });
      return false;
    }
    const Ctor = resolveAudioContextCtor();
    if (!Ctor) {
      this.setStatus({ state: 'unsupported', message: 'WEB AUDIO NOT SUPPORTED IN THIS BROWSER.' });
      return false;
    }
    this.setStatus({ state: 'starting', message: 'INITIALIZING MIC…' });

    this.starting = (async (): Promise<boolean> => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        const ctx = new Ctor();
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = this.fftSize;
        analyser.smoothingTimeConstant = 0.75;
        source.connect(analyser);
        this.stream = stream;
        this.ctx = ctx;
        this.source = source;
        this.analyser = analyser;
        this.setStatus({ state: 'live', message: 'MIC ONLINE' });
        return true;
      } catch (err) {
        const name = typeof err === 'object' && err !== null && 'name' in err
          ? String((err as { name: unknown }).name) : '';
        if (name === 'NotAllowedError' || name === 'SecurityError') {
          this.setStatus({ state: 'denied', message: 'MIC ACCESS DENIED.' });
        } else {
          this.setStatus({ state: 'error', message: 'MIC UNAVAILABLE.' });
        }
        return false;
      } finally {
        this.starting = null;
      }
    })();
    return this.starting;
  }

  stop(): void {
    if (this.source) { try { this.source.disconnect(); } catch { /* noop */ } this.source = null; }
    this.analyser = null;
    if (this.stream) { for (const t of this.stream.getTracks()) t.stop(); this.stream = null; }
    if (this.ctx) { void this.ctx.close().catch(() => { /* noop */ }); this.ctx = null; }
    if (this.status.state !== 'denied' && this.status.state !== 'unsupported') {
      this.setStatus({ state: 'idle', message: 'MIC STANDBY' });
    }
  }

  /** Fills `target` with time-domain samples; returns a coarse 0-1 RMS level. */
  read(target: Float32Array<ArrayBuffer>): number {
    const analyser = this.analyser;
    if (!analyser) { target.fill(0); return 0; }
    analyser.getFloatTimeDomainData(target);
    let sumSquares = 0;
    for (let i = 0; i < target.length; i += 1) { const v = target[i] ?? 0; sumSquares += v * v; }
    return Math.min(1, Math.sqrt(sumSquares / target.length) * 4);
  }

  dispose(): void { this.stop(); this.listeners.clear(); }

  private setStatus(next: MicMeterStatus): void {
    if (this.status.state === next.state && this.status.message === next.message) return;
    this.status = next;
    for (const l of Array.from(this.listeners)) {
      try { l(next); } catch (err) { console.error('[audio] mic status listener threw', err); }
    }
  }
}

// ---- Speech recognition (STT) with wake-word detection -----------------

export type RecognizerState = 'idle' | 'listening' | 'unsupported' | 'error';
export interface RecognizerStatus { state: RecognizerState; message: string }

/** Minimal shape of the non-standard SpeechRecognition API. */
interface SpeechRecognitionLike extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  onend: (() => void) | null;
}
interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }>;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function resolveRecognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export interface SpeechRecognizerOptions {
  /** Word that must appear before `onWake` fires. Default 'jarvis'. */
  wakeWord?: string;
  lang?: string;
}
type ResultListener = (text: string, isFinal: boolean) => void;
type WakeListener = (utterance: string) => void;

/**
 * Continuous STT wrapper. Fires `onResult` for every interim/final chunk and
 * `onWake` the first time the configured wake word appears in a chunk.
 */
export class SpeechRecognizer {
  private recognition: SpeechRecognitionLike | null = null;
  private status: RecognizerStatus = { state: 'idle', message: 'VOICE INPUT STANDBY' };
  private readonly statusListeners = new Set<(status: RecognizerStatus) => void>();
  private readonly resultListeners = new Set<ResultListener>();
  private readonly wakeListeners = new Set<WakeListener>();
  private readonly wakeWord: string;
  private readonly lang: string;
  private restarting = false;
  private stoppedByUser = true;

  constructor(options: SpeechRecognizerOptions = {}) {
    this.wakeWord = (options.wakeWord ?? 'jarvis').toLowerCase();
    this.lang = options.lang ?? 'en-GB';
  }

  static isSupported(): boolean { return resolveRecognitionCtor() !== null; }
  getStatus(): RecognizerStatus { return this.status; }
  isListening(): boolean { return this.status.state === 'listening'; }

  onStatus(listener: (status: RecognizerStatus) => void): () => void {
    this.statusListeners.add(listener);
    return () => { this.statusListeners.delete(listener); };
  }
  onResult(listener: ResultListener): () => void {
    this.resultListeners.add(listener);
    return () => { this.resultListeners.delete(listener); };
  }
  /** Fires once per detected wake-word utterance with the full chunk text. */
  onWake(listener: WakeListener): () => void {
    this.wakeListeners.add(listener);
    return () => { this.wakeListeners.delete(listener); };
  }

  start(): void {
    if (this.status.state === 'listening') return;
    const Ctor = resolveRecognitionCtor();
    if (!Ctor) {
      this.setStatus({ state: 'unsupported', message: 'SPEECH RECOGNITION NOT SUPPORTED IN THIS BROWSER.' });
      return;
    }
    this.stoppedByUser = false;
    const recognition = new Ctor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = this.lang;

    recognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (!result || result.length === 0) continue;
        const alt = result[0];
        if (!alt) continue;
        const text = alt.transcript.trim();
        if (!text) continue;
        const isFinal = (result as { isFinal: boolean }).isFinal;
        this.emitResult(text, isFinal);
        if (text.toLowerCase().includes(this.wakeWord)) this.emitWake(text);
      }
    };
    recognition.onerror = (event) => {
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        this.setStatus({ state: 'error', message: 'VOICE INPUT ACCESS DENIED.' });
        this.stoppedByUser = true;
      }
    };
    recognition.onend = () => {
      if (this.stoppedByUser) {
        this.setStatus({ state: 'idle', message: 'VOICE INPUT STANDBY' });
        return;
      }
      // Browsers auto-stop continuous recognition after silence; restart
      // transparently unless the caller explicitly stopped us.
      if (!this.restarting) {
        this.restarting = true;
        setTimeout(() => {
          this.restarting = false;
          if (!this.stoppedByUser) { try { recognition.start(); } catch { /* already running */ } }
        }, 250);
      }
    };

    try {
      recognition.start();
      this.recognition = recognition;
      this.setStatus({ state: 'listening', message: 'VOICE INPUT LIVE' });
    } catch {
      this.setStatus({ state: 'error', message: 'VOICE INPUT FAILED TO START.' });
    }
  }

  stop(): void {
    this.stoppedByUser = true;
    if (this.recognition) { try { this.recognition.stop(); } catch { /* noop */ } }
    this.recognition = null;
    this.setStatus({ state: 'idle', message: 'VOICE INPUT STANDBY' });
  }

  dispose(): void {
    this.stop();
    this.statusListeners.clear();
    this.resultListeners.clear();
    this.wakeListeners.clear();
  }

  private emitResult(text: string, isFinal: boolean): void {
    for (const l of Array.from(this.resultListeners)) {
      try { l(text, isFinal); } catch (err) { console.error('[audio] result listener threw', err); }
    }
  }
  private emitWake(utterance: string): void {
    for (const l of Array.from(this.wakeListeners)) {
      try { l(utterance); } catch (err) { console.error('[audio] wake listener threw', err); }
    }
  }
  private setStatus(next: RecognizerStatus): void {
    if (this.status.state === next.state && this.status.message === next.message) return;
    this.status = next;
    for (const l of Array.from(this.statusListeners)) {
      try { l(next); } catch (err) { console.error('[audio] status listener threw', err); }
    }
  }
}

// ---- Speech synthesis (TTS) — deepest available en-GB male voice -------

/** Known deep-leaning en-GB male voice names across Chrome/Edge/Safari, best first. */
const PREFERRED_VOICE_NAMES = ['Daniel', 'Arthur', 'George', 'Ryan', 'Oliver', 'Google UK English Male'];

function isSpeechSynthesisSupported(): boolean {
  return typeof window !== 'undefined' && typeof window.speechSynthesis !== 'undefined';
}

function pickVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
  if (voices.length === 0) return null;
  for (const name of PREFERRED_VOICE_NAMES) {
    const hit = voices.find((v) => v.name === name);
    if (hit) return hit;
  }
  const enGbMale = voices.find(
    (v) => v.lang.toLowerCase() === 'en-gb' && /male/i.test(v.name) && !/female/i.test(v.name),
  );
  if (enGbMale) return enGbMale;
  const enGb = voices.find((v) => v.lang.toLowerCase() === 'en-gb');
  if (enGb) return enGb;
  const enAny = voices.find((v) => v.lang.toLowerCase().startsWith('en'));
  if (enAny) return enAny;
  return voices[0] ?? null;
}

export interface SpeakerOptions {
  /** Lower = deeper. Default 0.82. */
  pitch?: number;
  /** Speaking rate. Default 0.98. */
  rate?: number;
  volume?: number;
}

/**
 * SpeechSynthesis wrapper. Selects the deepest available en-GB male voice on
 * construction (and re-selects when the browser's voice list loads async).
 */
export class Speaker {
  private voice: SpeechSynthesisVoice | null = null;
  private speaking = false;
  private readonly listeners = new Set<(speaking: boolean) => void>();
  private readonly opts: Required<SpeakerOptions>;
  private readonly onVoicesChanged = (): void => this.refreshVoice();

  constructor(options: SpeakerOptions = {}) {
    this.opts = { pitch: options.pitch ?? 0.82, rate: options.rate ?? 0.98, volume: options.volume ?? 1 };
    if (isSpeechSynthesisSupported()) {
      this.refreshVoice();
      window.speechSynthesis.addEventListener('voiceschanged', this.onVoicesChanged);
    }
  }

  static isSupported(): boolean { return isSpeechSynthesisSupported(); }
  isSpeaking(): boolean { return this.speaking; }

  onSpeakingChange(listener: (speaking: boolean) => void): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  speak(text: string): void {
    const clean = text.trim();
    if (!clean || !isSpeechSynthesisSupported()) return;
    const utterance = new SpeechSynthesisUtterance(clean);
    if (this.voice) utterance.voice = this.voice;
    utterance.pitch = this.opts.pitch;
    utterance.rate = this.opts.rate;
    utterance.volume = this.opts.volume;
    utterance.onstart = () => this.setSpeaking(true);
    utterance.onend = () => this.setSpeaking(false);
    utterance.onerror = () => this.setSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }

  cancel(): void {
    if (!isSpeechSynthesisSupported()) return;
    window.speechSynthesis.cancel();
    this.setSpeaking(false);
  }

  dispose(): void {
    if (isSpeechSynthesisSupported()) {
      window.speechSynthesis.removeEventListener('voiceschanged', this.onVoicesChanged);
      this.cancel();
    }
    this.listeners.clear();
  }

  private refreshVoice(): void {
    const voices = window.speechSynthesis.getVoices();
    if (voices.length === 0) return;
    this.voice = pickVoice(voices);
  }
  private setSpeaking(next: boolean): void {
    if (this.speaking === next) return;
    this.speaking = next;
    for (const l of Array.from(this.listeners)) {
      try { l(next); } catch (err) { console.error('[audio] speaking listener threw', err); }
    }
  }
}

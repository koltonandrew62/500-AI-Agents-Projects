/**
 * Typed WebSocket client for the J.A.R.V.I.S. `/ws` protocol.
 * Protocol source of truth: docs/CONTRACTS.md section 3.
 *
 * Resilience contract: this client NEVER gives up. On any close/error it
 * schedules a reconnect with exponential backoff + jitter and reports
 * 'disconnected' so the HUD can render a DISCONNECTED state.
 */

// ---------------------------------------------------------------------------
// Client -> Server
// ---------------------------------------------------------------------------

export interface ChatMsg { type: 'chat'; text: string; id: string }
export interface FrameMsg { type: 'frame'; jpeg_b64: string; id: string }
export interface VoiceMsg { type: 'voice'; text: string; id: string }
export interface CancelMsg { type: 'cancel'; id: string }
export interface PingMsg { type: 'ping' }

export type ClientMessage = ChatMsg | FrameMsg | VoiceMsg | CancelMsg | PingMsg;

// ---------------------------------------------------------------------------
// Server -> Client
// ---------------------------------------------------------------------------

export interface TokenEvent { type: 'token'; id: string; text: string }
export interface DoneEvent { type: 'done'; id: string; text: string }

export type ThinkingStage = 'planning' | 'tool' | 'vision' | 'recall';
export interface ThinkingEvent { type: 'thinking'; id: string; stage: ThinkingStage | string }

export interface ToolCallEvent {
  type: 'tool_call';
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface ToolResultEvent {
  type: 'tool_result';
  id: string;
  name: string;
  ok: boolean;
  summary: string;
}

export interface TelemetryEvent {
  type: 'telemetry';
  cpu: number;
  mem: number;
  disk: number;
  net_up: number;
  net_down: number;
  battery: number | null;
  uptime_s: number;
  processes?: number;
  temp_c?: number | null;
}

export interface DetectedObject {
  label: string;
  confidence: number;
  /** x, y, w, h in source-frame pixels. */
  box: [number, number, number, number];
}

export interface VisionEvent {
  type: 'vision';
  objects: DetectedObject[];
  faces: number;
  hands?: number;
  text: string;
  scene: string;
  width?: number;
  height?: number;
}

export interface SpeakEvent { type: 'speak'; text: string; voice: string }

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';
export interface LogEvent { type: 'log'; level: LogLevel | string; text: string }

export interface ErrorEvent_ { type: 'error'; id: string | null; text: string }

export type ServerEvent =
  | TokenEvent
  | DoneEvent
  | ThinkingEvent
  | ToolCallEvent
  | ToolResultEvent
  | TelemetryEvent
  | VisionEvent
  | SpeakEvent
  | LogEvent
  | ErrorEvent_;

export type ServerEventType = ServerEvent['type'];
export type EventOf<T extends ServerEventType> = Extract<ServerEvent, { type: T }>;

// ---------------------------------------------------------------------------
// Connection status
// ---------------------------------------------------------------------------

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';

export type Unsubscribe = () => void;

export interface JarvisSocketOptions {
  /** Full ws:// URL. Defaults to same-origin `/ws` (Vite proxies it in dev). */
  url?: string;
  /** First retry delay, ms. */
  baseDelayMs?: number;
  /** Retry delay ceiling, ms. */
  maxDelayMs?: number;
  /** Heartbeat interval, ms. Set 0 to disable. */
  heartbeatMs?: number;
  /** Force reconnect if nothing arrives for this long, ms. Set 0 to disable. */
  stallTimeoutMs?: number;
}

const KNOWN_TYPES: ReadonlySet<string> = new Set<ServerEventType>([
  'token', 'done', 'thinking', 'tool_call', 'tool_result',
  'telemetry', 'vision', 'speak', 'log', 'error',
]);

/** Manual boundary guard — no schema library on the frontend (CONTRACTS §7). */
export function isServerEvent(value: unknown): value is ServerEvent {
  if (typeof value !== 'object' || value === null) return false;
  const t = (value as { type?: unknown }).type;
  return typeof t === 'string' && KNOWN_TYPES.has(t);
}

export function defaultWsUrl(): string {
  const fromEnv = import.meta.env.VITE_WS_URL;
  if (fromEnv) return fromEnv;
  if (typeof window === 'undefined') return 'ws://localhost:8000/ws';
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws`;
}

let idSeq = 0;
/** Stable unique turn id; crypto.randomUUID when available. */
export function newId(): string {
  const c = globalThis.crypto as Crypto | undefined;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  idSeq += 1;
  return `t-${Date.now().toString(36)}-${idSeq.toString(36)}`;
}

type Handler<T extends ServerEventType> = (event: EventOf<T>) => void;
type AnyHandler = (event: ServerEvent) => void;

export class JarvisSocket {
  private readonly url: string;
  private readonly baseDelay: number;
  private readonly maxDelay: number;
  private readonly heartbeatMs: number;
  private readonly stallTimeoutMs: number;

  private ws: WebSocket | null = null;
  private attempt = 0;
  private closedByUser = false;
  private status: ConnectionStatus = 'disconnected';

  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private stallTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly handlers = new Map<ServerEventType, Set<AnyHandler>>();
  private readonly statusHandlers = new Set<(s: ConnectionStatus) => void>();
  /** Messages queued while the socket is not OPEN (bounded). */
  private readonly outbox: ClientMessage[] = [];
  private static readonly MAX_OUTBOX = 32;

  constructor(options: JarvisSocketOptions = {}) {
    this.url = options.url ?? defaultWsUrl();
    this.baseDelay = options.baseDelayMs ?? 600;
    this.maxDelay = options.maxDelayMs ?? 15_000;
    this.heartbeatMs = options.heartbeatMs ?? 15_000;
    this.stallTimeoutMs = options.stallTimeoutMs ?? 45_000;
  }

  // -- lifecycle ------------------------------------------------------------

  connect(): void {
    this.closedByUser = false;
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    this.clearReconnect();
    this.setStatus('connecting');

    let socket: WebSocket;
    try {
      socket = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = socket;

    socket.onopen = () => {
      if (this.ws !== socket) return;
      this.attempt = 0;
      this.setStatus('connected');
      this.startHeartbeat();
      this.armStallTimer();
      this.flushOutbox();
    };

    socket.onmessage = (ev: MessageEvent<unknown>) => {
      if (this.ws !== socket) return;
      this.armStallTimer();
      if (typeof ev.data !== 'string') return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (!isServerEvent(parsed)) return;
      this.dispatch(parsed);
    };

    socket.onerror = () => {
      /* onclose always follows; reconnect is handled there. */
    };

    socket.onclose = () => {
      if (this.ws !== socket) return;
      this.ws = null;
      this.stopHeartbeat();
      this.clearStallTimer();
      this.setStatus('disconnected');
      if (!this.closedByUser) this.scheduleReconnect();
    };
  }

  /** Permanently close; call `connect()` again to resume. */
  close(): void {
    this.closedByUser = true;
    this.clearReconnect();
    this.stopHeartbeat();
    this.clearStallTimer();
    const socket = this.ws;
    this.ws = null;
    if (socket) {
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      try {
        socket.close(1000, 'client shutdown');
      } catch {
        /* already closing */
      }
    }
    this.setStatus('disconnected');
  }

  /** Drop the current socket and retry immediately (user-triggered retry). */
  reconnectNow(): void {
    this.attempt = 0;
    this.clearReconnect();
    const socket = this.ws;
    this.ws = null;
    if (socket) {
      socket.onclose = null;
      try { socket.close(4000, 'manual reconnect'); } catch { /* noop */ }
    }
    this.connect();
  }

  getStatus(): ConnectionStatus {
    return this.status;
  }

  // -- subscription ---------------------------------------------------------

  on<T extends ServerEventType>(type: T, handler: Handler<T>): Unsubscribe {
    let set = this.handlers.get(type);
    if (!set) {
      set = new Set();
      this.handlers.set(type, set);
    }
    const wrapped = handler as AnyHandler;
    set.add(wrapped);
    return () => {
      const current = this.handlers.get(type);
      if (!current) return;
      current.delete(wrapped);
      if (current.size === 0) this.handlers.delete(type);
    };
  }

  onStatus(handler: (status: ConnectionStatus) => void): Unsubscribe {
    this.statusHandlers.add(handler);
    return () => { this.statusHandlers.delete(handler); };
  }

  // -- send helpers ---------------------------------------------------------

  sendChat(text: string, id: string = newId()): string {
    this.send({ type: 'chat', text, id });
    return id;
  }

  sendFrame(jpegB64: string, id: string = newId()): string {
    // Frames are volatile: never queue them, drop if the socket is down.
    this.send({ type: 'frame', jpeg_b64: jpegB64, id }, false);
    return id;
  }

  sendVoice(text: string, id: string = newId()): string {
    this.send({ type: 'voice', text, id });
    return id;
  }

  cancel(id: string): void {
    this.send({ type: 'cancel', id }, false);
  }

  ping(): void {
    this.send({ type: 'ping' }, false);
  }

  /** Low-level send. `queue` buffers the message until the socket reopens. */
  send(message: ClientMessage, queue = true): boolean {
    const socket = this.ws;
    if (socket && socket.readyState === WebSocket.OPEN) {
      try {
        socket.send(JSON.stringify(message));
        return true;
      } catch {
        /* fall through to queue */
      }
    }
    if (queue) {
      if (this.outbox.length >= JarvisSocket.MAX_OUTBOX) this.outbox.shift();
      this.outbox.push(message);
      if (!this.closedByUser) this.connect();
    }
    return false;
  }

  // -- internals ------------------------------------------------------------

  private dispatch(event: ServerEvent): void {
    const set = this.handlers.get(event.type);
    if (!set) return;
    for (const handler of Array.from(set)) {
      try {
        handler(event);
      } catch (err) {
        console.error('[ws] handler threw for', event.type, err);
      }
    }
  }

  private setStatus(next: ConnectionStatus): void {
    if (this.status === next) return;
    this.status = next;
    for (const handler of Array.from(this.statusHandlers)) {
      try {
        handler(next);
      } catch (err) {
        console.error('[ws] status handler threw', err);
      }
    }
  }

  private flushOutbox(): void {
    while (this.outbox.length > 0) {
      const next = this.outbox[0];
      if (!next) { this.outbox.shift(); continue; }
      const socket = this.ws;
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      try {
        socket.send(JSON.stringify(next));
        this.outbox.shift();
      } catch {
        return;
      }
    }
  }

  private scheduleReconnect(): void {
    if (this.closedByUser || this.reconnectTimer !== null) return;
    const exponent = Math.min(this.attempt, 6);
    const raw = this.baseDelay * 2 ** exponent;
    const capped = Math.min(raw, this.maxDelay);
    const jitter = capped * 0.25 * Math.random();
    this.attempt += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, capped + jitter);
  }

  private clearReconnect(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    if (this.heartbeatMs <= 0) return;
    this.heartbeatTimer = setInterval(() => { this.ping(); }, this.heartbeatMs);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /** If the server goes silent past the threshold, treat the link as dead. */
  private armStallTimer(): void {
    this.clearStallTimer();
    if (this.stallTimeoutMs <= 0) return;
    this.stallTimer = setTimeout(() => {
      this.stallTimer = null;
      if (this.closedByUser) return;
      const socket = this.ws;
      this.ws = null;
      if (socket) {
        socket.onclose = null;
        try { socket.close(4001, 'stalled'); } catch { /* noop */ }
      }
      this.stopHeartbeat();
      this.setStatus('disconnected');
      this.scheduleReconnect();
    }, this.stallTimeoutMs);
  }

  private clearStallTimer(): void {
    if (this.stallTimer !== null) {
      clearTimeout(this.stallTimer);
      this.stallTimer = null;
    }
  }
}

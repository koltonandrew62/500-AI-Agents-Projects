/**
 * The `/ws` endpoint — PORT-CONTRACTS.md §3, implemented exactly against the
 * `ClientMessage`/`ServerEvent` unions in `types.ts`.
 *
 * One duplex JSON channel per client. `ConnectionManager` tracks every
 * connected client and supports fan-out broadcast (telemetry). Each
 * connection runs at most one turn at a time: a new `chat`/`voice` message
 * cancels whatever turn is still in flight (via `AbortController`) before
 * starting the next one.
 *
 * Isolation: every inbound message is dispatched inside a try/catch that
 * turns any failure into an `error` event for that one client. One client's
 * bug or disconnect must never affect another client or kill the broadcast
 * loop.
 *
 * `ws.ts` does not know the concrete shape of the agent/LLM/memory stack —
 * it depends only on the small `AgentLoopLike` surface below, composed and
 * injected by `index.ts`. This keeps the WebSocket transport decoupled from
 * how a turn is actually produced.
 */

import { randomUUID } from 'node:crypto';
import type { RawData, WebSocket } from 'ws';
import { WebSocketServer } from 'ws';
import type { Server as HttpServer } from 'node:http';
import type { ClientMessage, ServerEvent, Telemetry, TurnContext } from './types.js';
import type { VisionPipeline } from './vision/index.js';
import type { TelemetryMonitor } from './telemetry/index.js';

/** The minimal surface `ws.ts` needs from the agent stack (built by `ts-core`/`ts-memory`). */
export interface AgentLoopLike {
  handle(text: string, ctx: TurnContext, signal: AbortSignal): AsyncIterable<ServerEvent>;
}

interface ClientState {
  readonly id: string;
  readonly socket: WebSocket;
  ctx: TurnContext;
  turnAbort: AbortController | null;
  turnPromise: Promise<void> | null;
  /** Serializes turn start/cancel so two rapid `chat` messages can't race. */
  turnLock: Promise<unknown>;
}

function freshCtx(): TurnContext {
  return { turnId: '', history: [], recalled: [], telemetry: null, vision: null, lastFrameB64: null };
}

function isClientMessage(value: unknown): value is ClientMessage {
  if (typeof value !== 'object' || value === null) return false;
  const type = (value as Record<string, unknown>).type;
  return typeof type === 'string' && ['chat', 'frame', 'voice', 'cancel', 'ping'].includes(type);
}

export class ConnectionManager {
  private readonly clients = new Map<string, ClientState>();

  constructor(
    private readonly agentLoop: AgentLoopLike,
    private readonly visionPipeline: VisionPipeline | null,
  ) {}

  count(): number {
    return this.clients.size;
  }

  register(socket: WebSocket): string {
    const id = randomUUID();
    this.clients.set(id, {
      id,
      socket,
      ctx: freshCtx(),
      turnAbort: null,
      turnPromise: null,
      turnLock: Promise.resolve(),
    });
    return id;
  }

  unregister(id: string): void {
    const state = this.clients.get(id);
    if (!state) return;
    this.clients.delete(id);
    state.turnAbort?.abort();
  }

  /** Send `event` to every connected client. Dead sockets are dropped, never throw. */
  broadcast(event: ServerEvent): void {
    for (const [id, state] of [...this.clients]) {
      if (!this.send(state, event)) {
        this.unregister(id);
      }
    }
  }

  async dispatch(id: string, raw: RawData): Promise<void> {
    const state = this.clients.get(id);
    if (!state) return;

    let msg: unknown;
    try {
      msg = JSON.parse(raw.toString('utf8'));
    } catch {
      this.send(state, { type: 'error', text: 'invalid JSON' });
      return;
    }
    if (!isClientMessage(msg)) {
      const maybeId = typeof (msg as Record<string, unknown>)?.id === 'string' ? (msg as { id: string }).id : undefined;
      this.send(state, { type: 'error', id: maybeId, text: "message must be a JSON object with a valid 'type'" });
      return;
    }

    try {
      switch (msg.type) {
        case 'chat':
        case 'voice':
          await this.handleTurn(state, msg.text, msg.id);
          break;
        case 'frame':
          await this.handleFrame(state, msg.jpeg_b64, msg.id);
          break;
        case 'cancel':
          this.handleCancel(state, msg.id);
          break;
        case 'ping':
          break;
      }
    } catch (err) {
      const errId = 'id' in msg ? msg.id : undefined;
      console.error(`[ws] error handling '${msg.type}' from client ${id}`, err);
      this.send(state, { type: 'error', id: errId, text: err instanceof Error ? err.message : String(err) });
    }
  }

  // -- message handlers ----------------------------------------------------

  private async handleTurn(state: ClientState, text: string, turnId: string): Promise<void> {
    const run = async (): Promise<void> => {
      await this.cancelCurrentTurn(state);
      const controller = new AbortController();
      state.ctx.turnId = turnId;
      state.turnAbort = controller;
      state.turnPromise = this.runTurn(state, text, turnId, controller.signal);
      // Do not await here: the lock only serializes *starting* turns, not their duration.
    };
    state.turnLock = state.turnLock.then(run, run);
    await state.turnLock;
  }

  private async runTurn(state: ClientState, text: string, turnId: string, signal: AbortSignal): Promise<void> {
    try {
      for await (const event of this.agentLoop.handle(text, state.ctx, signal)) {
        if (signal.aborted) break;
        this.send(state, event);
      }
    } catch (err) {
      if (isAbortError(err) || signal.aborted) return;
      console.error(`[ws] agent loop raised for turn ${turnId}`, err);
      this.send(state, { type: 'error', id: turnId, text: err instanceof Error ? err.message : String(err) });
      this.send(state, { type: 'done', id: turnId, text: '' });
    }
  }

  private async cancelCurrentTurn(state: ClientState): Promise<void> {
    const controller = state.turnAbort;
    const prev = state.turnPromise;
    if (!controller || !prev) return;
    controller.abort();
    try {
      await prev;
    } catch {
      // already logged inside runTurn
    }
  }

  private handleCancel(state: ClientState, turnId: string): void {
    if (!turnId || turnId === state.ctx.turnId) {
      state.turnAbort?.abort();
    }
  }

  private async handleFrame(state: ClientState, jpegB64: string, frameId: string): Promise<void> {
    if (typeof jpegB64 !== 'string' || !jpegB64) {
      this.send(state, { type: 'error', id: frameId, text: 'frame message missing jpeg_b64' });
      return;
    }

    // Stash immediately so the next chat turn sees the frame even if
    // analysis below fails or vision is disabled.
    state.ctx.lastFrameB64 = jpegB64;

    if (!this.visionPipeline) return;
    try {
      const result = await this.visionPipeline.analyze(jpegB64);
      state.ctx.vision = result;
      this.send(state, { type: 'vision', ...result });
    } catch (err) {
      console.warn('[ws] vision analyze failed', err);
    }
  }

  // -- transport -------------------------------------------------------------

  private send(state: ClientState, event: ServerEvent): boolean {
    if (state.socket.readyState !== state.socket.OPEN) return false;
    try {
      state.socket.send(JSON.stringify(event));
      return true;
    } catch (err) {
      console.debug(`[ws] failed to deliver '${event.type}' event; client likely disconnected`, err);
      return false;
    }
  }
}

function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === 'AbortError';
}

export interface AttachWsOptions {
  httpServer: HttpServer;
  agentLoop: AgentLoopLike;
  visionPipeline: VisionPipeline | null;
  telemetryMonitor: TelemetryMonitor | null;
  telemetryIntervalMs: number;
  path?: string;
}

export interface WsHandle {
  manager: ConnectionManager;
  close(): void;
}

/** Wires the `/ws` upgrade handler onto `httpServer` and starts the telemetry broadcast loop. */
export function attachWebSocket(opts: AttachWsOptions): WsHandle {
  const wss = new WebSocketServer({ server: opts.httpServer, path: opts.path ?? '/ws' });
  const manager = new ConnectionManager(opts.agentLoop, opts.visionPipeline);

  wss.on('connection', (socket: WebSocket) => {
    const id = manager.register(socket);
    console.log(`[ws] client ${id} connected (${manager.count()} total)`);

    socket.on('message', (raw: RawData) => {
      manager.dispatch(id, raw).catch((err: unknown) => {
        console.error(`[ws] unhandled dispatch error for client ${id}`, err);
      });
    });

    socket.on('close', () => {
      manager.unregister(id);
      console.log(`[ws] client ${id} disconnected (${manager.count()} total)`);
    });

    socket.on('error', (err: Error) => {
      console.warn(`[ws] socket error for client ${id}`, err);
    });
  });

  const telemetryAbort = new AbortController();
  if (opts.telemetryMonitor) {
    opts.telemetryMonitor
      .run(
        (telemetry: Telemetry) => {
          manager.broadcast({ type: 'telemetry', ...telemetry });
        },
        opts.telemetryIntervalMs,
        telemetryAbort.signal,
      )
      .catch((err: unknown) => {
        console.error('[ws] telemetry broadcast loop crashed', err);
      });
  }

  return {
    manager,
    close(): void {
      telemetryAbort.abort();
      for (const client of wss.clients) {
        client.close();
      }
      wss.close();
    },
  };
}

/**
 * Dependency-free app store, subscribed to via `useSyncExternalStore`.
 *
 * Holds everything the shell renders: connection status, chat transcript,
 * streaming buffers, telemetry, vision results, agent state, tool activity,
 * log entries, and the vision/voice/memory toggles. `useJarvis.ts` is the
 * only module that mutates this store from the outside world (the socket).
 */
import type {
  ConnectionStatus,
  LogEvent,
  TelemetryEvent,
  ToolCallEvent,
  ToolResultEvent,
  VisionEvent,
} from '../lib/ws';

export type AgentState = 'idle' | 'listening' | 'thinking' | 'speaking';

export type ChatRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  /** True while a `done` for this id has not yet arrived. */
  streaming: boolean;
  ts: number;
}

export interface ToolActivityEntry {
  id: string;
  name: string;
  /** 'called' until the matching result arrives. */
  status: 'called' | 'ok' | 'failed';
  summary?: string;
  ts: number;
}

export interface LogEntry extends LogEvent {
  ts: number;
  key: string;
}

export interface JarvisToggles {
  vision: boolean;
  voice: boolean;
  memory: boolean;
}

export interface JarvisState {
  connection: ConnectionStatus;
  agentState: AgentState;
  messages: ChatMessage[];
  telemetry: TelemetryEvent | null;
  vision: VisionEvent | null;
  toolActivity: ToolActivityEntry[];
  logs: LogEntry[];
  toggles: JarvisToggles;
  /** Live (interim) voice transcript, cleared once committed as a chat message. */
  liveTranscript: string;
  /** Current mic input level, 0-1, for the waveform while listening. */
  micLevel: number;
  /** Downsampled waveform bars, 0-1 each, for the Waveform component. */
  waveform: number[];
}

const MAX_MESSAGES = 200;
const MAX_LOGS = 300;
const MAX_TOOL_ACTIVITY = 50;

function initialState(): JarvisState {
  return {
    connection: 'disconnected',
    agentState: 'idle',
    messages: [],
    telemetry: null,
    vision: null,
    toolActivity: [],
    logs: [],
    toggles: { vision: true, voice: true, memory: true },
    liveTranscript: '',
    micLevel: 0,
    waveform: new Array(24).fill(0) as number[],
  };
}

type Listener = () => void;

/** Minimal store compatible with `useSyncExternalStore`. */
class Store {
  private state: JarvisState = initialState();
  private readonly listeners = new Set<Listener>();

  getSnapshot = (): JarvisState => this.state;

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  private set(next: JarvisState): void {
    this.state = next;
    for (const listener of Array.from(this.listeners)) listener();
  }

  private patch(partial: Partial<JarvisState>): void {
    this.set({ ...this.state, ...partial });
  }

  // -- connection / agent state ---------------------------------------------

  setConnection(status: ConnectionStatus): void {
    this.patch({ connection: status });
  }

  setAgentState(next: AgentState): void {
    if (this.state.agentState === next) return;
    this.patch({ agentState: next });
  }

  // -- chat -------------------------------------------------------------

  /** Adds a fully-formed message (e.g. the local echo of what the user sent). */
  addMessage(message: ChatMessage): void {
    const messages = [...this.state.messages, message].slice(-MAX_MESSAGES);
    this.patch({ messages });
  }

  /** Appends a streamed token chunk to an assistant message, creating it if needed. */
  appendToken(id: string, chunk: string): void {
    const idx = this.state.messages.findIndex((m) => m.id === id);
    if (idx === -1) {
      this.addMessage({ id, role: 'assistant', text: chunk, streaming: true, ts: Date.now() });
      return;
    }
    const messages = this.state.messages.slice();
    const existing = messages[idx];
    if (!existing) return;
    messages[idx] = { ...existing, text: existing.text + chunk, streaming: true };
    this.patch({ messages });
  }

  /** Finalizes a streamed assistant message with the authoritative full text. */
  completeMessage(id: string, fullText: string): void {
    const idx = this.state.messages.findIndex((m) => m.id === id);
    if (idx === -1) {
      this.addMessage({ id, role: 'assistant', text: fullText, streaming: false, ts: Date.now() });
      return;
    }
    const messages = this.state.messages.slice();
    const existing = messages[idx];
    if (!existing) return;
    messages[idx] = { ...existing, text: fullText, streaming: false };
    this.patch({ messages });
  }

  // -- telemetry / vision -------------------------------------------------

  setTelemetry(event: TelemetryEvent): void {
    this.patch({ telemetry: event });
  }

  setVision(event: VisionEvent): void {
    this.patch({ vision: event });
  }

  // -- tools ----------------------------------------------------------------

  addToolCall(event: ToolCallEvent): void {
    const entry: ToolActivityEntry = { id: event.id, name: event.name, status: 'called', ts: Date.now() };
    const toolActivity = [...this.state.toolActivity, entry].slice(-MAX_TOOL_ACTIVITY);
    this.patch({ toolActivity });
  }

  addToolResult(event: ToolResultEvent): void {
    const idx = this.state.toolActivity.findIndex((t) => t.id === event.id && t.name === event.name);
    if (idx === -1) {
      const entry: ToolActivityEntry = {
        id: event.id,
        name: event.name,
        status: event.ok ? 'ok' : 'failed',
        summary: event.summary,
        ts: Date.now(),
      };
      this.patch({ toolActivity: [...this.state.toolActivity, entry].slice(-MAX_TOOL_ACTIVITY) });
      return;
    }
    const toolActivity = this.state.toolActivity.slice();
    const existing = toolActivity[idx];
    if (!existing) return;
    toolActivity[idx] = { ...existing, status: event.ok ? 'ok' : 'failed', summary: event.summary };
    this.patch({ toolActivity });
  }

  // -- logs -------------------------------------------------------------

  addLog(event: LogEvent): void {
    const entry: LogEntry = { ...event, ts: Date.now(), key: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}` };
    const logs = [...this.state.logs, entry].slice(-MAX_LOGS);
    this.patch({ logs });
  }

  // -- voice ------------------------------------------------------------

  setLiveTranscript(text: string): void {
    this.patch({ liveTranscript: text });
  }

  setMicLevel(level: number): void {
    this.patch({ micLevel: level });
  }

  // -- toggles ----------------------------------------------------------

  setToggle(key: keyof JarvisToggles, value: boolean): void {
    this.patch({ toggles: { ...this.state.toggles, [key]: value } });
  }
}

export const store = new Store();

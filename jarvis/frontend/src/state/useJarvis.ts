/**
 * `useJarvis` — the single hook that wires the WS client + audio devices to
 * the store and exposes actions to panels. Mount once (in `App.tsx`); every
 * panel reads from the store via this hook's returned snapshot.
 */
import { useCallback, useEffect, useRef, useSyncExternalStore } from 'react';
import { JarvisSocket, newId } from '../lib/ws';
import { MicLevelMeter, Speaker, SpeechRecognizer } from '../lib/audio';
import { store, type JarvisState } from './store';

// Module-level singletons: one socket / mic / speaker / recognizer for the
// life of the tab, regardless of how many times components re-render.
const socket = new JarvisSocket();
const micMeter = new MicLevelMeter();
const speaker = new Speaker();
const recognizer = new SpeechRecognizer({ wakeWord: 'jarvis' });

let wired = false;
let refCount = 0;

/** Wires socket/audio events into the store exactly once, ever. */
function wireOnce(): void {
  if (wired) return;
  wired = true;

  socket.onStatus((status) => store.setConnection(status));

  socket.on('token', (event) => {
    store.appendToken(event.id, event.text);
    store.setAgentState('speaking');
  });

  socket.on('done', (event) => {
    store.completeMessage(event.id, event.text);
    store.setAgentState('idle');
  });

  socket.on('thinking', () => {
    store.setAgentState('thinking');
  });

  socket.on('tool_call', (event) => {
    store.addToolCall(event);
    store.setAgentState('thinking');
  });

  socket.on('tool_result', (event) => {
    store.addToolResult(event);
  });

  socket.on('telemetry', (event) => {
    store.setTelemetry(event);
  });

  socket.on('vision', (event) => {
    store.setVision(event);
  });

  socket.on('speak', (event) => {
    store.setAgentState('speaking');
    speaker.speak(event.text);
  });

  socket.on('log', (event) => {
    store.addLog(event);
  });

  socket.on('error', (event) => {
    store.addLog({ type: 'log', level: 'error', text: event.text });
    store.setAgentState('idle');
  });

  speaker.onSpeakingChange((speaking) => {
    if (!speaking && store.getSnapshot().agentState === 'speaking') {
      store.setAgentState('idle');
    }
  });

  recognizer.onResult((text, isFinal) => {
    if (isFinal) {
      store.setLiveTranscript('');
    } else {
      store.setLiveTranscript(text);
      store.setAgentState('listening');
    }
  });

  recognizer.onWake(() => {
    store.setAgentState('listening');
  });

  micMeter.onStatus(() => { /* surfaced via isListening() below; no store field needed yet */ });
}

export interface JarvisActions {
  connect: () => void;
  reconnect: () => void;
  sendText: (text: string) => void;
  cancel: (id: string) => void;
  sendFrame: (jpegB64: string) => void;
  startVoice: () => void;
  stopVoice: () => void;
  toggleVision: () => void;
  toggleVoice: () => void;
  toggleMemory: () => void;
  micSupported: boolean;
  voiceSupported: boolean;
  ttsSupported: boolean;
}

export function useJarvis(): JarvisState & { actions: JarvisActions } {
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  const levelBuf = useRef(new Float32Array(256));
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    wireOnce();
    refCount += 1;
    socket.connect();

    return () => {
      refCount -= 1;
      if (refCount <= 0) {
        // Last consumer unmounted (e.g. HMR) — leave the socket connected;
        // it is intentionally a tab-lifetime singleton per CONTRACTS §3.
      }
    };
  }, []);

  // Drive the mic level + downsampled waveform -> store while listening.
  useEffect(() => {
    const BARS = 24;
    let lastTick = 0;
    const tick = (now: number): void => {
      // Throttle to ~20fps; the waveform doesn't need 60fps of store churn.
      if (now - lastTick >= 50) {
        lastTick = now;
        if (micMeter.isLive()) {
          const buf = levelBuf.current;
          const level = micMeter.read(buf);
          store.setMicLevel(level);
          const step = Math.floor(buf.length / BARS) || 1;
          const bars = new Float32Array(BARS);
          for (let i = 0; i < BARS; i += 1) {
            bars[i] = Math.min(1, Math.abs(buf[i * step] ?? 0) * 3);
          }
          store.setWaveform(bars);
        } else if (store.getSnapshot().micLevel !== 0) {
          store.setMicLevel(0);
          store.setWaveform(new Float32Array(BARS));
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  const connect = useCallback(() => socket.connect(), []);
  const reconnect = useCallback(() => socket.reconnectNow(), []);

  const sendText = useCallback((text: string) => {
    const clean = text.trim();
    if (!clean) return;
    const id = newId();
    store.addMessage({ id, role: 'user', text: clean, streaming: false, ts: Date.now() });
    socket.sendChat(clean, id);
    store.setAgentState('thinking');
  }, []);

  const cancel = useCallback((id: string) => socket.cancel(id), []);
  const sendFrame = useCallback((jpegB64: string) => socket.sendFrame(jpegB64), []);

  const startVoice = useCallback(() => {
    void micMeter.start();
    recognizer.start();
  }, []);

  const stopVoice = useCallback(() => {
    micMeter.stop();
    recognizer.stop();
    store.setLiveTranscript('');
    if (store.getSnapshot().agentState === 'listening') store.setAgentState('idle');
  }, []);

  const toggleVision = useCallback(() => {
    store.setToggle('vision', !store.getSnapshot().toggles.vision);
  }, []);
  const toggleVoice = useCallback(() => {
    const next = !store.getSnapshot().toggles.voice;
    store.setToggle('voice', next);
    if (!next) stopVoice();
  }, [stopVoice]);
  const toggleMemory = useCallback(() => {
    store.setToggle('memory', !store.getSnapshot().toggles.memory);
  }, []);

  return {
    ...state,
    actions: {
      connect,
      reconnect,
      sendText,
      cancel,
      sendFrame,
      startVoice,
      stopVoice,
      toggleVision,
      toggleVoice,
      toggleMemory,
      micSupported: typeof navigator !== 'undefined' && !!navigator.mediaDevices,
      voiceSupported: SpeechRecognizer.isSupported(),
      ttsSupported: Speaker.isSupported(),
    },
  };
}

/** Reused by VisionFeed so voice + vision share one wiring pass. */
export function getJarvisSocket(): JarvisSocket {
  wireOnce();
  return socket;
}

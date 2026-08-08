/**
 * CommandBar — bottom-center panel. Live transcript (ChatLog) on top, then a
 * waveform driven by mic input, then the input row (mic toggle + glowing
 * caret text input). This panel replaces the reference image's financial
 * panel — there is deliberately no money/financial UI anywhere in this app.
 */
import { useCallback, useRef, useState } from 'react';
import type { KeyboardEvent, ReactElement } from 'react';
import { Frame, Waveform } from '../hud';
import { useJarvis } from '../state/useJarvis';
import { ChatLog } from './ChatLog';
import styles from './CommandBar.module.css';

export function CommandBar(): ReactElement {
  const { messages, liveTranscript, waveform, agentState, connection, actions } = useJarvis();
  const [draft, setDraft] = useState('');
  const [micOn, setMicOn] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const submit = useCallback(() => {
    const text = draft.trim();
    if (!text) return;
    actions.sendText(text);
    setDraft('');
  }, [draft, actions]);

  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter') submit();
    },
    [submit],
  );

  const toggleMic = useCallback(() => {
    if (micOn) {
      actions.stopVoice();
      setMicOn(false);
    } else {
      actions.startVoice();
      setMicOn(true);
    }
  }, [micOn, actions]);

  const listening = agentState === 'listening' || micOn;
  const statusLabel =
    connection !== 'connected' ? connection.toUpperCase() : agentState.toUpperCase();

  return (
    <Frame title="COMMAND" status={statusLabel} sweep>
      <div className={styles.bar}>
        <div className={styles.transcript}>
          <ChatLog messages={messages} />
        </div>

        {liveTranscript ? <div className={styles.liveTranscript}>{liveTranscript}</div> : null}

        <div className={styles.waveRow}>
          <Waveform levels={waveform} active={listening} />
        </div>

        <div className={styles.inputRow}>
          <button
            type="button"
            className={`${styles.micButton} ${micOn ? styles.micButtonActive : ''}`.trim()}
            onClick={toggleMic}
            disabled={!actions.micSupported && !actions.voiceSupported}
            aria-pressed={micOn}
            aria-label={micOn ? 'Disable microphone' : 'Enable microphone'}
            title={
              actions.voiceSupported
                ? micOn
                  ? 'Disable microphone'
                  : 'Enable microphone'
                : 'Voice input unsupported in this browser'
            }
          >
            <svg className={styles.micIcon} viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <rect x="9" y="2" width="6" height="12" rx="3" stroke="currentColor" strokeWidth="1.6" />
              <path
                d="M5 11a7 7 0 0 0 14 0M12 18v3"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>
          </button>

          <div className={styles.inputWrap}>
            <input
              ref={inputRef}
              className={styles.input}
              type="text"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="TRANSMIT A COMMAND…"
              autoComplete="off"
              spellCheck={false}
            />
            {draft.length === 0 ? <span className={styles.caretGlow} aria-hidden="true" /> : null}
          </div>

          <button type="button" className={styles.sendButton} onClick={submit} disabled={!draft.trim()}>
            SEND
          </button>
        </div>
      </div>
    </Frame>
  );
}

export default CommandBar;

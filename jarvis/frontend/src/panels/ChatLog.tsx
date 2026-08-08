/**
 * ChatLog — conversation transcript styled as HUD comms traffic.
 * Pure presentational: renders whatever message list it's handed and
 * auto-scrolls to the newest entry. Embedded inside CommandBar.
 */
import { useEffect, useRef } from 'react';
import type { ReactElement } from 'react';
import type { ChatMessage } from '../state/store';
import styles from './ChatLog.module.css';

export interface ChatLogProps {
  messages: ChatMessage[];
  className?: string;
}

const TAG_CLASS: Record<ChatMessage['role'], string> = {
  user: styles.tagUser,
  assistant: styles.tagAssistant,
  system: styles.tagSystem,
};

const TAG_LABEL: Record<ChatMessage['role'], string> = {
  user: 'YOU',
  assistant: 'JARVIS',
  system: 'SYS',
};

function formatTime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString('en-GB', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function ChatLog({ messages, className }: ChatLogProps): ReactElement {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [messages]);

  return (
    <div ref={scrollRef} className={`${styles.log} ${className ?? ''}`.trim()}>
      {messages.length === 0 ? (
        <div className={styles.empty}>NO COMMS TRAFFIC YET — SAY OR TYPE SOMETHING.</div>
      ) : (
        messages.map((m) => (
          <div key={m.id} className={styles.row}>
            <span className={`${styles.tag} ${TAG_CLASS[m.role]}`}>{TAG_LABEL[m.role]}</span>
            <span className={styles.time}>{formatTime(m.ts)}</span>
            <span className={`${styles.text} ${m.streaming ? styles.streaming : ''}`.trim()}>
              {m.text}
            </span>
          </div>
        ))
      )}
    </div>
  );
}

export default ChatLog;

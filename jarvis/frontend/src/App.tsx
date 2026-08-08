/**
 * App — composes the full HUD layout per CONTRACTS.md section 6: header row
 * (brand + clock, status chips), a three-column main grid (vitals/telemetry,
 * core sphere + reticle, radar/waveform/vision), and a bottom row
 * (diagnostics + command bar). Collapses to a single column on narrow
 * screens without breaking.
 */
import { useMemo } from 'react';
import type { ReactElement } from 'react';
import {
  Clock,
  CoreSphere,
  Diagnostics,
  RadarGlobe,
  Reticle,
  StatusChips,
  SystemVitals,
  Telemetry,
  Waveform,
} from './hud';
import type { DiagnosticEntry } from './hud';
import type { StatusChip } from './hud';
import type { TelemetryEntry } from './hud';
import { CommandBar } from './panels/CommandBar';
import { VisionFeed } from './panels/VisionFeed';
import { useJarvis } from './state/useJarvis';
import styles from './App.module.css';

export function App(): ReactElement {
  const { connection, agentState, telemetry, vision, toggles, toolActivity, logs, waveform, actions } =
    useJarvis();

  const coreIntensity =
    agentState === 'listening' || agentState === 'speaking' ? 0.7 : agentState === 'thinking' ? 0.5 : 0.15;

  const chips: StatusChip[] = useMemo(
    () => [
      {
        id: 'link',
        label: connection === 'connected' ? 'LINK NOMINAL' : connection.toUpperCase(),
        active: connection === 'connected',
        tone: connection === 'connected' ? 'cyan' : 'red',
      },
      { id: 'vision', label: 'VISION', active: toggles.vision, tone: toggles.vision ? 'cyan' : 'dim' },
      { id: 'voice', label: 'VOICE', active: toggles.voice, tone: toggles.voice ? 'cyan' : 'dim' },
      { id: 'memory', label: 'MEMORY', active: toggles.memory, tone: toggles.memory ? 'cyan' : 'dim' },
      { id: 'state', label: agentState.toUpperCase(), active: agentState !== 'idle', tone: 'amber' },
    ],
    [connection, toggles, agentState],
  );

  const onChipSelect = (id: string): void => {
    if (id === 'vision') actions.toggleVision();
    else if (id === 'voice') actions.toggleVoice();
    else if (id === 'memory') actions.toggleMemory();
  };

  // Telemetry panel is the scrolling event console — fed by the `log` stream.
  const telemetryEntries: TelemetryEntry[] = useMemo(
    () =>
      logs.map((entry) => ({
        id: entry.key,
        level: entry.level === 'warn' ? 'warn' : entry.level === 'error' ? 'error' : entry.level === 'debug' ? 'debug' : 'info',
        text: entry.text,
        ts: entry.ts,
      })),
    [logs],
  );

  // Diagnostics is the compact key/value readout — recent tool activity + link stats.
  const diagnosticEntries: DiagnosticEntry[] = useMemo(() => {
    const entries: DiagnosticEntry[] = [
      { label: 'LINK', value: connection.toUpperCase(), tone: connection === 'connected' ? 'cyan' : 'red' },
      { label: 'AGENT', value: agentState.toUpperCase(), tone: 'cyan' },
      { label: 'UPTIME', value: telemetry ? `${Math.floor(telemetry.uptime_s)}S` : '—', tone: 'dim' },
      { label: 'VISION OBJ', value: String(vision?.objects.length ?? 0), tone: 'cyan' },
    ];
    const recentTools = toolActivity.slice(-4).reverse();
    for (const tool of recentTools) {
      entries.push({
        label: tool.name.slice(0, 12).toUpperCase(),
        value: tool.status.toUpperCase(),
        tone: tool.status === 'failed' ? 'red' : tool.status === 'ok' ? 'cyan' : 'amber',
      });
    }
    return entries;
  }, [connection, agentState, telemetry, vision, toolActivity]);

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.brandBlock}>
          <span className={styles.brand}>J.A.R.V.I.S.</span>
          <span className={styles.subtitle}>JUST A RATHER VERY INTELLIGENT SYSTEM</span>
          <div className={styles.chips}>
            <StatusChips chips={chips} onSelect={onChipSelect} />
          </div>
        </div>
        <div className={styles.headerRight}>
          <Clock />
        </div>
      </header>

      <main className={styles.main}>
        <div className={`${styles.col} ${styles.colLeft}`}>
          <SystemVitals
            cpu={telemetry?.cpu ?? 0}
            mem={telemetry?.mem ?? 0}
            disk={telemetry?.disk ?? 0}
            netUp={telemetry?.net_up ?? 0}
            netDown={telemetry?.net_down ?? 0}
            battery={telemetry?.battery ?? null}
          />
          <Telemetry entries={telemetryEntries} />
        </div>

        <div className={styles.col}>
          <div className={styles.core}>
            <div className={styles.coreSphereLayer}>
              <CoreSphere intensity={coreIntensity} state={agentState} />
            </div>
            <div className={styles.reticleLayer}>
              <Reticle state={agentState} intensity={coreIntensity} />
            </div>
          </div>
        </div>

        <div className={`${styles.col} ${styles.colRight}`}>
          <RadarGlobe active={connection === 'connected'} />
          <Waveform levels={waveform} active={agentState === 'listening' || agentState === 'speaking'} />
          <VisionFeed />
        </div>
      </main>

      <div className={styles.bottom}>
        <Diagnostics entries={diagnosticEntries} />
        <CommandBar />
      </div>
    </div>
  );
}

export default App;

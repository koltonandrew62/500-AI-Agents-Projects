/**
 * HUD barrel — the single import surface for the shell.
 *
 *   import { Frame, CoreSphere, Reticle, SystemVitals } from './hud';
 *
 * Components are value exports; every prop/model type is re-exported alongside
 * so the shell never has to reach into individual module paths.
 */

// ---- Chrome ---------------------------------------------------------------
export { Frame, default as FrameDefault } from './Frame';
export type { FrameProps, FrameTone } from './Frame';

// ---- Core + orbit ---------------------------------------------------------
export { CoreSphere } from './CoreSphere';
export type { CoreSphereProps, CoreState } from './CoreSphere';

export { Reticle } from './Reticle';
export type { ReticleProps } from './Reticle';

// ---- Left column ----------------------------------------------------------
export { SystemVitals } from './SystemVitals';
export type { SystemVitalsProps, VitalGauge, VitalTone } from './SystemVitals';

export { Telemetry } from './Telemetry';
export type { TelemetryProps, TelemetryEntry, TelemetryLevel } from './Telemetry';

export { Diagnostics } from './Diagnostics';
export type { DiagnosticsProps, DiagnosticEntry, DiagTone } from './Diagnostics';

// ---- Right column ---------------------------------------------------------
export { RadarGlobe } from './RadarGlobe';
export type { RadarGlobeProps, RadarContact, ContactTone } from './RadarGlobe';

export { Waveform } from './Waveform';
export type { WaveformProps, WaveformTone } from './Waveform';

// ---- Header ---------------------------------------------------------------
export { Clock } from './Clock';
export type { ClockProps, GeoReadout } from './Clock';

export { StatusChips } from './StatusChips';
export type { StatusChipsProps, StatusChip, ChipTone } from './StatusChips';

// ---- Shared hooks + math (used by shell-side panels too) ------------------
export {
  useRafLoop,
  usePrefersReducedMotion,
  useElementSize,
  approach,
  clamp,
} from './hooks';
export type { RafCallback } from './hooks';

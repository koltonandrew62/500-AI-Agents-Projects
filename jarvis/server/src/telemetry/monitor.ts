/**
 * System telemetry sampling via `systeminformation`.
 *
 * `TelemetryMonitor.sample()` returns one `Telemetry` snapshot (see
 * `types.ts`). `net_up`/`net_down` are instantaneous **KB/s rates**, derived
 * from the delta between this sample's cumulative byte counters and the
 * previous sample's — never a raw cumulative total, since a monotonically
 * growing total is not useful telemetry to stream to a HUD every second.
 *
 * Every individual sensor read is independently guarded: a machine with no
 * battery or no exposed CPU temperature degrades that one field to `null`
 * (or `0` for counters) and keeps every other field live. `sample()` itself
 * never rejects/throws.
 *
 * Ported from `backend/app/core/telemetry/monitor.py`.
 */

import si from 'systeminformation';
import type { Telemetry } from '../types.js';

const BYTES_PER_KB = 1024;

interface NetSnapshot {
  bytesSent: number;
  bytesRecv: number;
  timestampMs: number;
}

export class TelemetryMonitor {
  private prevNet: NetSnapshot | null = null;

  /** Take one point-in-time telemetry reading. Never throws. */
  async sample(): Promise<Telemetry> {
    const [cpu, mem, disk, net, battery, uptimeS, processes, temp] = await Promise.all([
      this.readCpuPercent(),
      this.readMemPercent(),
      this.readDiskPercent(),
      this.readNetRates(),
      this.readBatteryPercent(),
      this.readUptimeS(),
      this.readProcessCount(),
      this.readCpuTemp(),
    ]);

    return {
      cpu,
      mem,
      disk,
      net_up: net.up,
      net_down: net.down,
      battery,
      uptime_s: uptimeS,
      processes,
      temp_c: temp,
    };
  }

  /**
   * Sample on a fixed interval and invoke `onSample`. Runs until `signal` is
   * aborted. A failure in `onSample` (e.g. a closed socket) is swallowed and
   * the loop continues rather than dying silently.
   */
  async run(
    onSample: (telemetry: Telemetry) => void | Promise<void>,
    intervalMs: number,
    signal?: AbortSignal,
  ): Promise<void> {
    const interval = Math.max(50, intervalMs);
    while (!signal?.aborted) {
      try {
        const telemetry = await this.sample();
        await onSample(telemetry);
      } catch (err) {
        console.warn('[telemetry] broadcast callback failed, continuing loop', err);
      }
      await sleep(interval, signal);
    }
  }

  // -- individual sensor reads, each independently fail-safe ---------------

  private async readCpuPercent(): Promise<number> {
    try {
      const load = await si.currentLoad();
      return round2(load.currentLoad);
    } catch (err) {
      console.warn('[telemetry] failed to read CPU percent', err);
      return 0;
    }
  }

  private async readMemPercent(): Promise<number> {
    try {
      const mem = await si.mem();
      if (mem.total <= 0) return 0;
      return round2(((mem.total - mem.available) / mem.total) * 100);
    } catch (err) {
      console.warn('[telemetry] failed to read memory percent', err);
      return 0;
    }
  }

  private async readDiskPercent(): Promise<number> {
    try {
      const layout = await si.fsSize();
      if (!layout.length) return 0;
      // Weighted average across mounted filesystems, mirroring a single
      // root-usage percentage without assuming a "/" mount exists (Windows).
      const totalSize = layout.reduce((sum, d) => sum + (d.size || 0), 0);
      const totalUsed = layout.reduce((sum, d) => sum + (d.used || 0), 0);
      if (totalSize <= 0) return 0;
      return round2((totalUsed / totalSize) * 100);
    } catch (err) {
      console.warn('[telemetry] failed to read disk usage', err);
      return 0;
    }
  }

  private async readNetRates(): Promise<{ up: number; down: number }> {
    try {
      const stats = await si.networkStats();
      const bytesSent = stats.reduce((sum, s) => sum + (s.tx_bytes || 0), 0);
      const bytesRecv = stats.reduce((sum, s) => sum + (s.rx_bytes || 0), 0);
      const now = Date.now();
      const current: NetSnapshot = { bytesSent, bytesRecv, timestampMs: now };

      const previous = this.prevNet;
      this.prevNet = current;

      if (previous === null) return { up: 0, down: 0 };

      const elapsedS = (current.timestampMs - previous.timestampMs) / 1000;
      if (elapsedS <= 0) return { up: 0, down: 0 };

      const sentDelta = Math.max(0, current.bytesSent - previous.bytesSent);
      const recvDelta = Math.max(0, current.bytesRecv - previous.bytesRecv);

      return {
        up: round2(sentDelta / BYTES_PER_KB / elapsedS),
        down: round2(recvDelta / BYTES_PER_KB / elapsedS),
      };
    } catch (err) {
      console.warn('[telemetry] failed to read network counters', err);
      return { up: 0, down: 0 };
    }
  }

  private async readBatteryPercent(): Promise<number | null> {
    try {
      const battery = await si.battery();
      if (!battery.hasBattery) return null;
      return round2(battery.percent);
    } catch (err) {
      console.debug('[telemetry] battery sensor unavailable on this platform', err);
      return null;
    }
  }

  private async readUptimeS(): Promise<number> {
    try {
      return Math.max(0, Math.floor(si.time().uptime));
    } catch (err) {
      console.warn('[telemetry] failed to read system uptime', err);
      return 0;
    }
  }

  private async readProcessCount(): Promise<number> {
    try {
      const procs = await si.processes();
      return procs.all ?? 0;
    } catch (err) {
      console.warn('[telemetry] failed to read process count', err);
      return 0;
    }
  }

  private async readCpuTemp(): Promise<number | null> {
    try {
      const temp = await si.cpuTemperature();
      if (temp.main === null || temp.main === undefined || Number.isNaN(temp.main) || temp.main < 0) {
        return null;
      }
      return round2(temp.main);
    } catch (err) {
      console.debug('[telemetry] CPU temperature sensor unavailable on this platform', err);
      return null;
    }
  }
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

/**
 * System tools: get_time, get_system_info, take_note, set_volume, open_app.
 *
 * `set_volume` and `open_app` shell out to platform-specific utilities
 * behind a strict allowlist — never build a command line from unvalidated
 * free text.
 */

import { execFile } from 'node:child_process';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { config } from '../config.js';
import { registry } from './base.js';
import type { Tool, ToolResult } from '../types.js';

const VOLUME_TIMEOUT_MS = 5_000;

// Platform-aware, allowlisted app launchers. Only these keys may ever be
// opened — arbitrary strings from the caller are never passed to a shell.
type AppKey = 'browser' | 'terminal' | 'notes' | 'calculator' | 'finder';

const APP_COMMANDS: Record<string, Record<AppKey, string[]>> = {
  darwin: {
    browser: ['open', '-a', 'Safari'],
    terminal: ['open', '-a', 'Terminal'],
    notes: ['open', '-a', 'Notes'],
    calculator: ['open', '-a', 'Calculator'],
    finder: ['open', '-a', 'Finder'],
  },
  linux: {
    browser: ['xdg-open', 'https://'],
    terminal: ['x-terminal-emulator'],
    notes: ['gedit'],
    calculator: ['gnome-calculator'],
    finder: ['xdg-open', '.'],
  },
  win32: {
    browser: ['cmd', '/c', 'start', 'msedge'],
    terminal: ['cmd', '/c', 'start', 'cmd'],
    notes: ['notepad'],
    calculator: ['calc'],
    finder: ['explorer'],
  },
};

function fail(summary: string): ToolResult {
  return { ok: false, output: '', summary };
}

type ExecFileError = import('node:child_process').ExecFileException;

function runFile(cmd: string[], timeoutMs: number): Promise<{ ok: boolean; stderr: string; error?: ExecFileError }> {
  return new Promise((resolve) => {
    const program = cmd[0];
    if (program === undefined) {
      // Every PLATFORM_APPS/VOLUME_CMDS entry is a non-empty literal array,
      // so this is unreachable today — guarded because `cmd` is a general
      // `string[]` parameter and a future caller could pass one.
      resolve({ ok: false, stderr: 'runFile called with an empty command' });
      return;
    }
    const rest = cmd.slice(1);
    execFile(
      program,
      rest,
      { timeout: timeoutMs, encoding: 'utf8' as const },
      (error: ExecFileError | null, _stdout: string, stderr: string) => {
        resolve({ ok: !error, stderr: stderr ?? '', error: error ?? undefined });
      },
    );
  });
}

const getTimeTool: Tool = {
  name: 'get_time',
  description: 'Get the current local date and time.',
  parameters: { type: 'object', properties: {}, required: [], additionalProperties: false },
  async run(): Promise<ToolResult> {
    const now = new Date();
    const iso = now.toISOString();
    const summary = now.toLocaleString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
    return { ok: true, output: iso, summary };
  },
};

async function cpuPercent(sampleMs = 100): Promise<number> {
  const start = os.cpus();
  await new Promise((resolve) => setTimeout(resolve, sampleMs));
  const end = os.cpus();

  let idleDelta = 0;
  let totalDelta = 0;
  // `start`/`end` come from two separate os.cpus() calls; nothing types-level
  // guarantees they're the same length (hot-plug CPUs, containerized limits),
  // so bound the loop by the shorter of the two rather than asserting.
  const coreCount = Math.min(start.length, end.length);
  for (let i = 0; i < coreCount; i++) {
    const startCore = start[i];
    const endCore = end[i];
    if (startCore === undefined || endCore === undefined) continue;
    const s = startCore.times;
    const e = endCore.times;
    const sIdle = s.idle;
    const eIdle = e.idle;
    const sTotal = s.user + s.nice + s.sys + s.idle + s.irq;
    const eTotal = e.user + e.nice + e.sys + e.idle + e.irq;
    idleDelta += eIdle - sIdle;
    totalDelta += eTotal - sTotal;
  }
  if (totalDelta <= 0) return 0;
  return Math.max(0, Math.min(100, (1 - idleDelta / totalDelta) * 100));
}

async function diskUsage(target: string): Promise<{ percent: number; freeGb: number } | null> {
  try {
    const stat = await fs.statfs(target);
    const total = stat.blocks * stat.bsize;
    const free = stat.bfree * stat.bsize;
    const used = total - free;
    const percent = total > 0 ? (used / total) * 100 : 0;
    return { percent, freeGb: free / 1e9 };
  } catch {
    return null; // statfs unsupported on this platform/Node build — degrade gracefully
  }
}

const getSystemInfoTool: Tool = {
  name: 'get_system_info',
  description: 'Get CPU, memory, disk, and uptime information for the host machine.',
  parameters: { type: 'object', properties: {}, required: [], additionalProperties: false },
  async run(): Promise<ToolResult> {
    const cpu = await cpuPercent();
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const usedMem = totalMem - freeMem;
    const memPercent = totalMem > 0 ? (usedMem / totalMem) * 100 : 0;
    const disk = await diskUsage(os.platform() === 'win32' ? 'C:\\' : '/');
    const uptimeS = Math.round(os.uptime());

    const info: Record<string, unknown> = {
      cpu_percent: Math.round(cpu * 10) / 10,
      mem_percent: Math.round(memPercent * 10) / 10,
      mem_used_gb: Math.round((usedMem / 1e9) * 100) / 100,
      mem_total_gb: Math.round((totalMem / 1e9) * 100) / 100,
      disk_percent: disk ? Math.round(disk.percent * 10) / 10 : null,
      disk_free_gb: disk ? Math.round(disk.freeGb * 100) / 100 : null,
      uptime_s: uptimeS,
      platform: `${os.type()} ${os.release()} (${os.arch()})`,
    };
    const summary = `CPU ${info.cpu_percent}% · MEM ${info.mem_percent}% · DISK ${disk ? `${info.disk_percent}%` : 'n/a'}`;
    return { ok: true, output: JSON.stringify(info), summary, meta: info };
  },
};

const takeNoteTool: Tool = {
  name: 'take_note',
  description: 'Append a timestamped note to the workspace notes file.',
  parameters: {
    type: 'object',
    properties: { text: { type: 'string', minLength: 1 } },
    required: ['text'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const text = args.text as string;
    await fs.mkdir(config.workspaceRoot, { recursive: true });
    const notesPath = path.join(config.workspaceRoot, 'notes.md');
    const timestamp = new Date().toISOString();
    const line = `- [${timestamp}] ${text}\n`;
    await fs.appendFile(notesPath, line, 'utf-8');
    return { ok: true, output: line.trim(), summary: 'Note saved' };
  },
};

const setVolumeTool: Tool = {
  name: 'set_volume',
  description: 'Set the system output volume (0-100), platform-aware.',
  parameters: {
    type: 'object',
    properties: { level: { type: 'integer', minimum: 0, maximum: 100 } },
    required: ['level'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const level = args.level as number;
    const platform = os.platform();

    let cmd: string[];
    if (platform === 'darwin') {
      cmd = ['osascript', '-e', `set volume output volume ${level}`];
    } else if (platform === 'linux') {
      cmd = ['amixer', '-D', 'pulse', 'sset', 'Master', `${level}%`];
    } else if (platform === 'win32') {
      // No stock CLI volume control on Windows; report unsupported rather
      // than guessing at a third-party tool being installed.
      return fail('set_volume is not supported on Windows without a third-party tool');
    } else {
      return fail(`Unsupported platform: ${platform}`);
    }

    const { ok, stderr, error } = await runFile(cmd, VOLUME_TIMEOUT_MS);
    if (error?.code === 'ENOENT') {
      return fail(`Could not set volume: ${error.message}`);
    }
    if (!ok) {
      return fail(`Volume command failed: ${stderr}`);
    }
    return { ok: true, output: '', summary: `Volume set to ${level}%` };
  },
};

const openAppTool: Tool = {
  name: 'open_app',
  description: 'Open an allowlisted application: browser, terminal, notes, calculator, finder.',
  parameters: {
    type: 'object',
    properties: {
      app: { type: 'string', enum: ['browser', 'terminal', 'notes', 'calculator', 'finder'] },
    },
    required: ['app'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const app = args.app as AppKey;
    const platform = os.platform();
    const commands = APP_COMMANDS[platform];
    if (!commands) {
      return fail(`Unsupported platform: ${platform}`);
    }

    const cmd = commands[app];
    if (!cmd) {
      return fail(`App not allowlisted on ${platform}: ${JSON.stringify(app)}`);
    }

    const { error } = await runFile(cmd, VOLUME_TIMEOUT_MS);
    if (error?.code === 'ENOENT') {
      return fail(`Could not launch ${JSON.stringify(app)}: ${error.message}`);
    }
    return { ok: true, output: '', summary: `Opened ${app}` };
  },
};

registry.register(getTimeTool);
registry.register(getSystemInfoTool);
registry.register(takeNoteTool);
registry.register(setVolumeTool);
registry.register(openAppTool);

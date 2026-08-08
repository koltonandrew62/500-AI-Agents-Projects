/**
 * Sandboxed code execution: exec_js and exec_shell.
 *
 * `exec_js` runs arbitrary JavaScript in a throwaway `fork`ed Node
 * subprocess with a 5s timeout and a temp cwd, after a source-level
 * denylist rejects known-dangerous constructs. This is defense-in-depth,
 * not a real sandbox (no seccomp/chroot/network namespace) — the denylist
 * plus a short timeout and a scratch cwd are the guarantees actually
 * provided, mirroring the Python edition's AST-check approach.
 *
 * `exec_shell` runs a single allowlisted command with `execFile` (no shell
 * interpretation at all), so shell metacharacters are inert.
 */

import { fork, execFile } from 'node:child_process';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { registry } from './base.js';
import type { Tool, ToolResult } from '../types.js';

const JS_TIMEOUT_MS = 5_000;
const SHELL_TIMEOUT_MS = 5_000;

// Modules that are a direct route to the network, process control, or
// dynamic code loading — never allowed, regardless of import style.
const BANNED_MODULES = [
  'child_process', 'node:child_process',
  'net', 'node:net',
  'http', 'node:http',
  'https', 'node:https',
  'http2', 'node:http2',
  'dgram', 'node:dgram',
  'tls', 'node:tls',
  'cluster', 'node:cluster',
  'worker_threads', 'node:worker_threads',
  'vm', 'node:vm',
  'inspector', 'node:inspector',
  'repl', 'node:repl',
];

/** Matches `require('x')`, `require("x")`, `import ... from 'x'`, `import('x')`. */
function findModuleReferences(source: string): string[] {
  const found: string[] = [];
  const patterns = [
    /require\(\s*(['"])(.*?)\1\s*\)/g,
    /import\s+(?:[\s\S]*?\s+from\s+)?(['"])(.*?)\1/g,
    /import\(\s*(['"])(.*?)\1\s*\)/g,
  ];
  for (const re of patterns) {
    for (const match of source.matchAll(re)) {
      found.push(match[2]);
    }
  }
  return found;
}

/** `fs.writeFile*`/`appendFile*`/`unlink*`/`rm*` calls whose literal path arg looks like it escapes the sandbox cwd. */
function findUnsafeFsWrites(source: string): string[] {
  const violations: string[] = [];
  const writeCallRe = /\b(?:fs\.)?(writeFileSync|writeFile|appendFileSync|appendFile|unlinkSync|unlink|rmSync|rmdirSync|rmdir|rm)\s*\(\s*(['"])(.*?)\2/g;
  for (const match of source.matchAll(writeCallRe)) {
    const [, fn, , arg] = match;
    if (arg.startsWith('/') || arg.startsWith('~') || arg.includes('..')) {
      violations.push(`write-mode ${fn}() outside sandbox cwd: ${JSON.stringify(arg)}`);
    }
  }
  return violations;
}

/** Parse `source` and return a list of denylist violation messages. */
export function checkJsSource(source: string): string[] {
  const violations: string[] = [];

  for (const mod of findModuleReferences(source)) {
    const root = mod.split('/')[0];
    if (BANNED_MODULES.includes(mod) || BANNED_MODULES.includes(root)) {
      violations.push(`import of banned module: ${mod}`);
    }
  }

  // Escapes to dynamic module loading / eval-style code execution.
  if (/\brequire\s*\(\s*[^'"]/.test(source)) {
    violations.push('call to require() with a non-literal argument (dynamic require escape)');
  }
  if (/\bimport\s*\(\s*[^'")]/.test(source)) {
    violations.push('call to dynamic import() with a non-literal argument');
  }
  if (/\beval\s*\(/.test(source)) {
    violations.push('call to banned builtin: eval');
  }
  if (/\bnew\s+Function\s*\(/.test(source)) {
    violations.push('call to banned builtin: Function constructor');
  }
  if (/\bprocess\s*\.\s*exit\s*\(/.test(source)) {
    violations.push('call to banned attribute: process.exit');
  }
  if (/\bprocess\s*\.\s*kill\s*\(/.test(source)) {
    violations.push('call to banned attribute: process.kill');
  }
  if (/\bprocess\s*\.\s*binding\s*\(/.test(source)) {
    violations.push('call to banned attribute: process.binding');
  }
  if (/\bglobalThis\s*\[\s*['"]process['"]\s*\]/.test(source)) {
    violations.push('indirect access to process via globalThis indexing');
  }

  violations.push(...findUnsafeFsWrites(source));

  return violations;
}

function fail(summary: string): ToolResult {
  return { ok: false, output: '', summary };
}

const execJsTool: Tool = {
  name: 'exec_js',
  description:
    'Execute a short JavaScript snippet in an isolated Node subprocess with a 5s timeout and a temp ' +
    'working directory. Network/process/require-escape calls are statically rejected before execution.',
  parameters: {
    type: 'object',
    properties: {
      code: { type: 'string', minLength: 1, description: 'JavaScript source to execute.' },
    },
    required: ['code'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const code = args.code as string;

    const violations = checkJsSource(code);
    if (violations.length > 0) {
      return fail(`Rejected by sandbox denylist: ${violations.join('; ')}`);
    }

    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'jarvis-sandbox-'));
    const scriptPath = path.join(tmpDir, 'snippet.js');
    try {
      await fs.writeFile(scriptPath, code, 'utf-8');

      const result = await new Promise<ToolResult>((resolve) => {
        const stdoutChunks: Buffer[] = [];
        const stderrChunks: Buffer[] = [];

        const child = fork(scriptPath, [], {
          cwd: tmpDir,
          stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
          // No inherited env leakage beyond what the child needs to run node.
          env: { PATH: process.env.PATH ?? '' },
          execArgv: [], // no --inspect / loader escapes
        });

        let settled = false;
        const timer = setTimeout(() => {
          if (settled) return;
          settled = true;
          child.kill('SIGKILL');
          resolve(fail(`exec_js timed out after ${JS_TIMEOUT_MS / 1000}s`));
        }, JS_TIMEOUT_MS);

        child.stdout?.on('data', (chunk: Buffer) => stdoutChunks.push(chunk));
        child.stderr?.on('data', (chunk: Buffer) => stderrChunks.push(chunk));

        child.on('error', (err) => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          resolve(fail(`exec_js failed to start: ${err.message}`));
        });

        child.on('exit', (code2) => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          const outText = Buffer.concat(stdoutChunks).toString('utf-8');
          const errText = Buffer.concat(stderrChunks).toString('utf-8');
          const ok = code2 === 0;
          const combined = ok ? outText : `${outText}\n${errText}`.trim();
          resolve({
            ok,
            output: combined,
            summary: ok ? 'exec_js succeeded' : `exec_js exited ${code2}`,
            meta: { returncode: code2 },
          });
        });
      });

      return result;
    } finally {
      await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => undefined);
    }
  },
};

// Commands considered harmless enough to run with zero side effects beyond
// reading already-public system state.
const SHELL_ALLOWLIST = new Set(['ls', 'cat', 'date', 'uname', 'df', 'ps', 'echo', 'pwd', 'which']);

/** Minimal shell-style tokenizer supporting quoted strings — no metacharacter interpretation follows. */
function tokenize(command: string): string[] | null {
  const tokens: string[] = [];
  let i = 0;
  const n = command.length;
  while (i < n) {
    while (i < n && /\s/.test(command[i])) i++;
    if (i >= n) break;
    let token = '';
    let quote: '"' | "'" | null = null;
    while (i < n) {
      const ch = command[i];
      if (quote) {
        if (ch === quote) {
          quote = null;
          i++;
        } else {
          token += ch;
          i++;
        }
      } else if (ch === '"' || ch === "'") {
        quote = ch;
        i++;
      } else if (/\s/.test(ch)) {
        break;
      } else {
        token += ch;
        i++;
      }
    }
    if (quote) return null; // unterminated quote
    tokens.push(token);
  }
  return tokens;
}

const execShellTool: Tool = {
  name: 'exec_shell',
  description: 'Run a single allowlisted read-only shell command (ls, cat, date, uname, df, ps, echo, pwd, which).',
  parameters: {
    type: 'object',
    properties: {
      command: { type: 'string', minLength: 1, description: "Full command line, e.g. 'ls -la'." },
    },
    required: ['command'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const command = args.command as string;
    const tokens = tokenize(command);
    if (tokens === null) {
      return fail('Could not parse command: unterminated quote');
    }
    if (tokens.length === 0) {
      return fail('Empty command');
    }

    const [program, ...rest] = tokens;
    if (!SHELL_ALLOWLIST.has(program)) {
      return fail(`Command not allowlisted: ${JSON.stringify(program)}`);
    }

    return new Promise<ToolResult>((resolve) => {
      // execFile with an argument array — shell=false by construction, so
      // shell metacharacters in `rest` are inert, never interpreted.
      const child = execFile(
        program,
        rest,
        { timeout: SHELL_TIMEOUT_MS, windowsHide: true },
        (error, stdout, stderr) => {
          if (error && (error as NodeJS.ErrnoException).code === 'ENOENT') {
            resolve(fail(`Command not found: ${JSON.stringify(program)}`));
            return;
          }
          if (error && error.killed) {
            resolve(fail(`exec_shell timed out after ${SHELL_TIMEOUT_MS / 1000}s`));
            return;
          }
          const ok = !error;
          const returncode = ok ? 0 : typeof error?.code === 'number' ? error.code : 1;
          const combined = ok ? stdout : `${stdout}\n${stderr}`.trim();
          resolve({
            ok,
            output: combined,
            summary: `${program} exited ${returncode}`,
            meta: { returncode },
          });
        },
      );
      void child;
    });
  },
};

registry.register(execJsTool);
registry.register(execShellTool);

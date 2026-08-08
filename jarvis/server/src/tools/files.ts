/**
 * Filesystem tools: read_file, write_file, list_dir, search_files, file_info.
 *
 * Every path argument is resolved against `config.workspaceRoot` and must
 * stay inside it after resolution — the load-bearing security property of
 * this module. Binary files and anything over 1MB are refused outright.
 */

import { promises as fs } from 'node:fs';
import path from 'node:path';

import { config } from '../config.js';
import { registry } from './base.js';
import type { Tool, ToolResult } from '../types.js';

const MAX_FILE_BYTES = 1_000_000; // 1MB cap on both read and write

// Extensions considered text and therefore safe to read/write as strings.
// Anything else is refused rather than guessed at, to avoid returning
// mangled bytes decoded lossily.
const TEXT_SUFFIXES = new Set([
  '.txt', '.md', '.markdown', '.py', '.js', '.ts', '.tsx', '.jsx', '.json',
  '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.csv', '.tsv', '.html',
  '.htm', '.css', '.xml', '.sh', '.bash', '.sql', '.log', '.rst', '',
]);

class PathEscapeError extends Error {}

/**
 * Resolve `relative` against workspaceRoot, refusing any escape.
 *
 * Rejects `..` traversal, absolute-path escapes, and symlinks that resolve
 * outside the root — all three are checked against the *resolved* real
 * path, not the literal string, since string-only checks are trivially
 * bypassed (e.g. a symlink planted inside the workspace pointing outward).
 */
async function resolveInWorkspace(relative: string): Promise<string> {
  const root = await realpathOrSelf(path.resolve(config.workspaceRoot));
  const candidate = path.isAbsolute(relative) ? relative : path.join(root, relative);
  const merged = path.resolve(candidate);

  // realpath requires the path to exist; for not-yet-created write targets we
  // walk up to the nearest existing ancestor and resolve that instead, then
  // re-append the missing tail — this still defeats a symlink anywhere in the
  // existing prefix while allowing new-file creation.
  const resolved = await resolveRealOrNearestAncestor(merged);

  const relFromRoot = path.relative(root, resolved);
  if (relFromRoot === '..' || relFromRoot.startsWith(`..${path.sep}`) || path.isAbsolute(relFromRoot)) {
    throw new PathEscapeError(`path ${JSON.stringify(relative)} escapes workspace root`);
  }
  return resolved;
}

async function realpathOrSelf(p: string): Promise<string> {
  try {
    return await fs.realpath(p);
  } catch {
    return path.resolve(p);
  }
}

async function resolveRealOrNearestAncestor(p: string): Promise<string> {
  try {
    return await fs.realpath(p);
  } catch {
    const parent = path.dirname(p);
    if (parent === p) return p; // reached filesystem root without finding anything real
    const realParent = await resolveRealOrNearestAncestor(parent);
    return path.join(realParent, path.basename(p));
  }
}

function isTextFile(p: string): boolean {
  return TEXT_SUFFIXES.has(path.extname(p).toLowerCase());
}

function fail(summary: string): ToolResult {
  return { ok: false, output: '', summary };
}

const readFileTool: Tool = {
  name: 'read_file',
  description: 'Read a UTF-8 text file (max 1MB) from the workspace.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Path relative to the workspace root.' },
    },
    required: ['path'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const relPath = args.path as string;
    let resolved: string;
    try {
      resolved = await resolveInWorkspace(relPath);
    } catch (exc) {
      return fail(exc instanceof Error ? exc.message : String(exc));
    }

    let stat;
    try {
      stat = await fs.stat(resolved);
    } catch {
      return fail(`No such file: ${JSON.stringify(relPath)}`);
    }
    if (!stat.isFile()) {
      return fail(`Not a file: ${JSON.stringify(relPath)}`);
    }
    if (!isTextFile(resolved)) {
      return fail(`Refusing to read non-text file: ${JSON.stringify(relPath)}`);
    }
    if (stat.size > MAX_FILE_BYTES) {
      return fail(`File too large (${stat.size} bytes > 1MB cap)`);
    }

    const buf = await fs.readFile(resolved);
    let text: string;
    try {
      text = decodeStrictUtf8(buf);
    } catch {
      return fail(`File is not valid UTF-8 text: ${JSON.stringify(relPath)}`);
    }
    return { ok: true, output: text, summary: `Read ${stat.size} bytes from ${JSON.stringify(relPath)}` };
  },
};

function decodeStrictUtf8(buf: Buffer): string {
  const decoder = new TextDecoder('utf-8', { fatal: true });
  return decoder.decode(buf);
}

const writeFileTool: Tool = {
  name: 'write_file',
  description: 'Write UTF-8 text (max 1MB) to a file in the workspace, creating parent dirs as needed.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Path relative to the workspace root.' },
      content: { type: 'string', description: 'Text content to write.' },
      append: { type: 'boolean', description: 'Append instead of overwrite. Default false.' },
    },
    required: ['path', 'content'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const relPath = args.path as string;
    const content = args.content as string;
    const append = (args.append as boolean | undefined) ?? false;

    let resolved: string;
    try {
      resolved = await resolveInWorkspace(relPath);
    } catch (exc) {
      return fail(exc instanceof Error ? exc.message : String(exc));
    }

    const encoded = Buffer.byteLength(content, 'utf-8');
    if (encoded > MAX_FILE_BYTES) {
      return fail('Content exceeds 1MB cap');
    }
    if (!isTextFile(resolved)) {
      return fail(`Refusing to write non-text file: ${JSON.stringify(relPath)}`);
    }

    const root = path.resolve(config.workspaceRoot);
    await fs.mkdir(path.dirname(resolved), { recursive: true });
    // Existing symlinks must also resolve inside root even before writing
    // through them — re-check post-mkdir to catch a symlinked parent dir.
    const realParent = await realpathOrSelf(path.dirname(resolved));
    const relFromRoot = path.relative(root, realParent);
    if (relFromRoot === '..' || relFromRoot.startsWith(`..${path.sep}`) || path.isAbsolute(relFromRoot)) {
      return fail(`path ${JSON.stringify(relPath)} escapes workspace root`);
    }

    if (append) {
      await fs.appendFile(resolved, content, 'utf-8');
    } else {
      await fs.writeFile(resolved, content, 'utf-8');
    }

    const verb = append ? 'Appended to' : 'Wrote';
    return { ok: true, output: '', summary: `${verb} ${encoded} bytes at ${JSON.stringify(relPath)}` };
  },
};

const listDirTool: Tool = {
  name: 'list_dir',
  description: 'List files and subdirectories inside a workspace directory.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: "Directory path relative to workspace root. Default '.'." },
    },
    required: [],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const relPath = (args.path as string | undefined) ?? '.';
    let resolved: string;
    try {
      resolved = await resolveInWorkspace(relPath);
    } catch (exc) {
      return fail(exc instanceof Error ? exc.message : String(exc));
    }

    let stat;
    try {
      stat = await fs.stat(resolved);
    } catch {
      return fail(`No such directory: ${JSON.stringify(relPath)}`);
    }
    if (!stat.isDirectory()) {
      return fail(`Not a directory: ${JSON.stringify(relPath)}`);
    }

    const entries = await fs.readdir(resolved, { withFileTypes: true });
    entries.sort((a, b) => {
      if (a.isDirectory() !== b.isDirectory()) return a.isDirectory() ? -1 : 1;
      return a.name.toLowerCase().localeCompare(b.name.toLowerCase());
    });
    const lines = entries.map((e) => `${e.isDirectory() ? 'd' : 'f'}  ${e.name}`);
    return {
      ok: true,
      output: lines.join('\n'),
      summary: `${entries.length} entries in ${JSON.stringify(relPath)}`,
      meta: { count: entries.length },
    };
  },
};

const searchFilesTool: Tool = {
  name: 'search_files',
  description: 'Search for a text substring across text files under a workspace directory.',
  parameters: {
    type: 'object',
    properties: {
      query: { type: 'string', minLength: 1 },
      path: { type: 'string', description: "Directory to search under. Default '.'." },
      max_results: { type: 'integer', minimum: 1, maximum: 200 },
    },
    required: ['query'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const query = args.query as string;
    const relPath = (args.path as string | undefined) ?? '.';
    const maxResults = (args.max_results as number | undefined) ?? 50;

    let resolved: string;
    try {
      resolved = await resolveInWorkspace(relPath);
    } catch (exc) {
      return fail(exc instanceof Error ? exc.message : String(exc));
    }

    let stat;
    try {
      stat = await fs.stat(resolved);
    } catch {
      return fail(`Not a directory: ${JSON.stringify(relPath)}`);
    }
    if (!stat.isDirectory()) {
      return fail(`Not a directory: ${JSON.stringify(relPath)}`);
    }

    const root = path.resolve(config.workspaceRoot);
    const matches: string[] = [];

    async function walk(dir: string): Promise<void> {
      if (matches.length >= maxResults) return;
      const dirents = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
      const sorted = [...dirents].sort((a, b) => a.name.localeCompare(b.name));
      for (const dirent of sorted) {
        if (matches.length >= maxResults) return;
        const full = path.join(dir, dirent.name);
        if (dirent.isDirectory()) {
          await walk(full);
          continue;
        }
        if (!dirent.isFile() || !isTextFile(full)) continue;
        // Guard against a symlinked file inside the tree escaping root.
        const real = await realpathOrSelf(full);
        const relFromRoot = path.relative(root, real);
        if (relFromRoot === '..' || relFromRoot.startsWith(`..${path.sep}`) || path.isAbsolute(relFromRoot)) {
          continue;
        }
        const fstat = await fs.stat(full).catch(() => null);
        if (!fstat || fstat.size > MAX_FILE_BYTES) continue;
        let text: string;
        try {
          text = decodeStrictUtf8(await fs.readFile(full));
        } catch {
          continue;
        }
        const lines = text.split('\n');
        for (let i = 0; i < lines.length; i++) {
          if (matches.length >= maxResults) break;
          if (lines[i].includes(query)) {
            const rel = path.relative(root, full);
            matches.push(`${rel}:${i + 1}: ${lines[i].trim().slice(0, 200)}`);
          }
        }
      }
    }

    await walk(resolved);

    return {
      ok: true,
      output: matches.join('\n'),
      summary: `${matches.length} match(es) for ${JSON.stringify(query)}`,
      meta: { count: matches.length },
    };
  },
};

const fileInfoTool: Tool = {
  name: 'file_info',
  description: 'Get metadata (size, type, modified time) for a workspace path.',
  parameters: {
    type: 'object',
    properties: { path: { type: 'string' } },
    required: ['path'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const relPath = args.path as string;
    let resolved: string;
    try {
      resolved = await resolveInWorkspace(relPath);
    } catch (exc) {
      return fail(exc instanceof Error ? exc.message : String(exc));
    }

    let stat;
    try {
      stat = await fs.stat(resolved);
    } catch {
      return fail(`No such path: ${JSON.stringify(relPath)}`);
    }

    const kind = stat.isDirectory() ? 'dir' : 'file';
    const info: Record<string, unknown> = {
      type: kind,
      size_bytes: stat.size,
      modified: stat.mtimeMs / 1000,
    };
    return {
      ok: true,
      output: JSON.stringify(info),
      summary: `${kind} ${JSON.stringify(relPath)}: ${stat.size} bytes`,
      meta: info,
    };
  },
};

registry.register(readFileTool);
registry.register(writeFileTool);
registry.register(listDirTool);
registry.register(searchFilesTool);
registry.register(fileInfoTool);

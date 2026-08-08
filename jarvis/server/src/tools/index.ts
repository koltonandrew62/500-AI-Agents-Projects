/**
 * Tool package entrypoint.
 *
 * Importing this module imports every tool submodule so their
 * `registry.register(...)` calls run and populate the shared registry. The
 * agent loop only ever needs `registry` (to list specs) and `runTool` (to
 * execute by name) — never the individual tool implementations.
 */

import './files.js';
import './sandbox.js';
import './scheduler.js';
import './system.js';
import './web.js';

export { registry, runTool } from './base.js';
export { runScheduler } from './scheduler.js';

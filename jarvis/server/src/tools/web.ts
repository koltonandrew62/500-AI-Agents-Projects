/**
 * Web tools: web_search, fetch_url, get_weather, get_news.
 *
 * All network calls use global `fetch` with `AbortSignal.timeout` — nothing
 * here is allowed to hang the agent loop indefinitely. No API keys
 * required: DuckDuckGo HTML endpoint for search, open-meteo.com for
 * weather, Google News RSS for headlines.
 */

import { registry } from './base.js';
import type { Tool, ToolResult } from '../types.js';

const HTTP_TIMEOUT_MS = 10_000;
const FETCH_MAX_BYTES = 200_000;
const USER_AGENT = 'Mozilla/5.0 (compatible; JarvisAssistant/1.0)';

const TAG_RE = /<[^>]+>/g;
const WHITESPACE_RE = /\s+/g;

function stripHtml(html: string): string {
  const noScriptStyle = html.replace(/<(script|style)[^>]*>[\s\S]*?<\/\1>/gi, ' ');
  const text = noScriptStyle.replace(TAG_RE, ' ');
  return text.replace(WHITESPACE_RE, ' ').trim();
}

function decodeEntities(text: string): string {
  return text
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function fail(summary: string): ToolResult {
  return { ok: false, output: '', summary };
}

interface SearchResult {
  title: string;
  url: string;
  snippet: string;
}

function parseDdgHtml(html: string, maxResults: number): SearchResult[] {
  const results: SearchResult[] = [];
  const blockRe = /<div class="result results_links[^"]*"[\s\S]*?<\/div>\s*<\/div>/gi;
  const linkRe = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/i;
  const snippetRe = /<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>/i;

  for (const blockMatch of html.matchAll(blockRe)) {
    if (results.length >= maxResults) break;
    const block = blockMatch[0];
    const linkMatch = linkRe.exec(block);
    if (!linkMatch) continue;
    const url = linkMatch[1];
    const title = decodeEntities(stripHtml(linkMatch[2]));
    const snippetMatch = snippetRe.exec(block);
    const snippet = snippetMatch ? decodeEntities(stripHtml(snippetMatch[1])) : '';
    results.push({ title, url, snippet });
  }
  return results;
}

const webSearchTool: Tool = {
  name: 'web_search',
  description: 'Search the web via DuckDuckGo (no API key) and return top result titles/links/snippets.',
  parameters: {
    type: 'object',
    properties: {
      query: { type: 'string', minLength: 1 },
      max_results: { type: 'integer', minimum: 1, maximum: 20 },
    },
    required: ['query'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const query = args.query as string;
    const maxResults = (args.max_results as number | undefined) ?? 5;

    const url = new URL('https://html.duckduckgo.com/html/');
    url.searchParams.set('q', query);

    let text: string;
    try {
      const resp = await fetch(url, {
        headers: { 'User-Agent': USER_AGENT },
        signal: AbortSignal.timeout(HTTP_TIMEOUT_MS),
      });
      if (!resp.ok) return fail(`Search request failed: HTTP ${resp.status}`);
      text = await resp.text();
    } catch (exc) {
      return fail(`Search request failed: ${exc instanceof Error ? exc.message : String(exc)}`);
    }

    const results = parseDdgHtml(text, maxResults);
    if (results.length === 0) {
      return { ok: true, output: '', summary: 'No results found' };
    }

    const lines = results.map((r, i) => `${i + 1}. ${r.title} — ${r.url}\n   ${r.snippet}`);
    return {
      ok: true,
      output: lines.join('\n'),
      summary: `${results.length} result(s) for ${JSON.stringify(query)}`,
      meta: { results },
    };
  },
};

const fetchUrlTool: Tool = {
  name: 'fetch_url',
  description: 'Fetch a URL and return readable plain text extracted from the HTML, capped at 200KB.',
  parameters: {
    type: 'object',
    properties: { url: { type: 'string', minLength: 1 } },
    required: ['url'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const url = args.url as string;
    if (!(url.startsWith('http://') || url.startsWith('https://'))) {
      return fail('Only http:// and https:// URLs are allowed');
    }

    let buf: ArrayBuffer;
    try {
      const resp = await fetch(url, {
        headers: { 'User-Agent': USER_AGENT },
        redirect: 'follow',
        signal: AbortSignal.timeout(HTTP_TIMEOUT_MS),
      });
      if (!resp.ok) return fail(`Fetch failed: HTTP ${resp.status}`);
      if (!resp.body) {
        buf = await resp.arrayBuffer();
      } else {
        // Stream and stop once the cap is hit, rather than buffering
        // arbitrarily large responses first.
        const reader = resp.body.getReader();
        const chunks: Uint8Array[] = [];
        let total = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            chunks.push(value);
            total += value.length;
          }
          if (total >= FETCH_MAX_BYTES) {
            await reader.cancel().catch(() => undefined);
            break;
          }
        }
        const merged = new Uint8Array(Math.min(total, FETCH_MAX_BYTES));
        let offset = 0;
        for (const chunk of chunks) {
          const remaining = merged.length - offset;
          if (remaining <= 0) break;
          merged.set(chunk.subarray(0, remaining), offset);
          offset += Math.min(chunk.length, remaining);
        }
        buf = merged.buffer;
      }
    } catch (exc) {
      return fail(`Fetch failed: ${exc instanceof Error ? exc.message : String(exc)}`);
    }

    const bytes = new Uint8Array(buf).subarray(0, FETCH_MAX_BYTES);
    const html = new TextDecoder('utf-8', { fatal: false }).decode(bytes);
    const text = decodeEntities(stripHtml(html));
    return { ok: true, output: text, summary: `Fetched ${bytes.length} bytes from ${url}` };
  },
};

const getWeatherTool: Tool = {
  name: 'get_weather',
  description: 'Get current weather for a latitude/longitude via open-meteo.com (no API key).',
  parameters: {
    type: 'object',
    properties: {
      latitude: { type: 'number', minimum: -90, maximum: 90 },
      longitude: { type: 'number', minimum: -180, maximum: 180 },
    },
    required: ['latitude', 'longitude'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const latitude = args.latitude as number;
    const longitude = args.longitude as number;

    const url = new URL('https://api.open-meteo.com/v1/forecast');
    url.searchParams.set('latitude', String(latitude));
    url.searchParams.set('longitude', String(longitude));
    url.searchParams.set('current_weather', 'true');

    let data: Record<string, unknown>;
    try {
      const resp = await fetch(url, { signal: AbortSignal.timeout(HTTP_TIMEOUT_MS) });
      if (!resp.ok) return fail(`Weather request failed: HTTP ${resp.status}`);
      data = (await resp.json()) as Record<string, unknown>;
    } catch (exc) {
      return fail(`Weather request failed: ${exc instanceof Error ? exc.message : String(exc)}`);
    }

    const current = data.current_weather as Record<string, unknown> | undefined;
    if (!current) {
      return fail('No current_weather in response');
    }

    const summary = `${current.temperature}°C, wind ${current.windspeed} km/h`;
    return { ok: true, output: JSON.stringify(current), summary, meta: current };
  },
};

const NEWS_FEED_URL = 'https://news.google.com/rss';

function extractTagText(itemXml: string, tag: string): string {
  const re = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`, 'i');
  const match = re.exec(itemXml);
  if (!match) return '';
  return decodeEntities(match[1].replace(/^<!\[CDATA\[([\s\S]*?)\]\]>$/, '$1')).trim();
}

const getNewsTool: Tool = {
  name: 'get_news',
  description: 'Get current top news headlines from a free public RSS feed.',
  parameters: {
    type: 'object',
    properties: {
      topic: { type: 'string', description: 'Optional topic/query to search news for.' },
      max_results: { type: 'integer', minimum: 1, maximum: 20 },
    },
    required: [],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const topic = (args.topic as string | undefined) ?? '';
    const maxResults = (args.max_results as number | undefined) ?? 5;
    const url = topic ? `${NEWS_FEED_URL}/search?q=${encodeURIComponent(topic)}` : NEWS_FEED_URL;

    let text: string;
    try {
      const resp = await fetch(url, {
        headers: { 'User-Agent': USER_AGENT },
        signal: AbortSignal.timeout(HTTP_TIMEOUT_MS),
      });
      if (!resp.ok) return fail(`News request failed: HTTP ${resp.status}`);
      text = await resp.text();
    } catch (exc) {
      return fail(`News request failed: ${exc instanceof Error ? exc.message : String(exc)}`);
    }

    const items = [...text.matchAll(/<item>([\s\S]*?)<\/item>/gi)].slice(0, maxResults);
    if (items.length === 0 && !/<rss[\s>]/i.test(text) && !/<feed[\s>]/i.test(text)) {
      return fail('Could not parse RSS feed: no <rss> or <feed> root element found');
    }

    const headlines = items.map((m) => extractTagText(m[1], 'title')).filter((h) => h.length > 0);
    return {
      ok: true,
      output: headlines.map((h, i) => `${i + 1}. ${h}`).join('\n'),
      summary: `${headlines.length} headline(s)`,
      meta: { headlines },
    };
  },
};

registry.register(webSearchTool);
registry.register(fetchUrlTool);
registry.register(getWeatherTool);
registry.register(getNewsTool);

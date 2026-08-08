"""Web tools: web_search, fetch_url, get_weather, get_news.

All network calls go through a shared httpx.AsyncClient with explicit
timeouts — nothing here is allowed to hang the agent loop indefinitely. No
API keys are required: DuckDuckGo HTML endpoint for search, open-meteo.com
for weather, and public RSS for news.
"""

from __future__ import annotations

import re
from typing import Any
from xml.etree import ElementTree

import httpx

from app.core.tools.base import register
from app.models.schemas import ToolResult

_HTTP_TIMEOUT_S = 10.0
_FETCH_MAX_BYTES = 200_000
_USER_AGENT = "Mozilla/5.0 (compatible; JarvisAssistant/1.0)"

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    """Very small readable-text extractor: drop script/style, strip tags."""
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = _TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


@register
class WebSearchTool:
    name = "web_search"
    description = "Search the web via DuckDuckGo (no API key) and return top result titles/links/snippets."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def run(self, query: str, max_results: int = 5) -> ToolResult:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S, headers={"User-Agent": _USER_AGENT}) as client:
            try:
                resp = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                return ToolResult(ok=False, output="", summary=f"Search request failed: {exc}")

        results = _parse_ddg_html(resp.text, max_results)
        if not results:
            return ToolResult(ok=True, output="", summary="No results found")

        lines = [f"{i + 1}. {r['title']} — {r['url']}\n   {r['snippet']}" for i, r in enumerate(results)]
        return ToolResult(
            ok=True, output="\n".join(lines), summary=f"{len(results)} result(s) for {query!r}", meta={"results": results}
        )


def _parse_ddg_html(html: str, max_results: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for block in re.findall(r'(?is)<div class="result results_links[^"]*".*?</div>\s*</div>', html):
        link_match = re.search(r'(?is)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block)
        snippet_match = re.search(r'(?is)<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', block)
        if not link_match:
            continue
        url = link_match.group(1)
        title = _strip_html(link_match.group(2))
        snippet = _strip_html(snippet_match.group(1)) if snippet_match else ""
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


@register
class FetchUrlTool:
    name = "fetch_url"
    description = "Fetch a URL and return readable plain text extracted from the HTML, capped at 200KB."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"url": {"type": "string", "minLength": 1}},
        "required": ["url"],
        "additionalProperties": False,
    }

    async def run(self, url: str) -> ToolResult:
        if not (url.startswith("http://") or url.startswith("https://")):
            return ToolResult(ok=False, output="", summary="Only http:// and https:// URLs are allowed")

        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT_S, headers={"User-Agent": _USER_AGENT}, follow_redirects=True
        ) as client:
            try:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in resp.aiter_bytes():
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= _FETCH_MAX_BYTES:
                            break
                    body = b"".join(chunks)[:_FETCH_MAX_BYTES]
            except httpx.HTTPError as exc:
                return ToolResult(ok=False, output="", summary=f"Fetch failed: {exc}")

        text = _strip_html(body.decode("utf-8", errors="replace"))
        return ToolResult(ok=True, output=text, summary=f"Fetched {len(body)} bytes from {url}")


@register
class GetWeatherTool:
    name = "get_weather"
    description = "Get current weather for a latitude/longitude via open-meteo.com (no API key)."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "latitude": {"type": "number", "minimum": -90, "maximum": 90},
            "longitude": {"type": "number", "minimum": -180, "maximum": 180},
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False,
    }

    async def run(self, latitude: float, longitude: float) -> ToolResult:
        params = {"latitude": latitude, "longitude": longitude, "current_weather": "true"}
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            try:
                resp = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                return ToolResult(ok=False, output="", summary=f"Weather request failed: {exc}")

        current = data.get("current_weather")
        if not current:
            return ToolResult(ok=False, output="", summary="No current_weather in response")

        summary = (
            f"{current.get('temperature')}°C, wind {current.get('windspeed')} km/h"
        )
        return ToolResult(ok=True, output=str(current), summary=summary, meta=current)


_NEWS_FEED_URL = "https://news.google.com/rss"


@register
class GetNewsTool:
    name = "get_news"
    description = "Get current top news headlines from a free public RSS feed."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Optional topic/query to search news for."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": [],
        "additionalProperties": False,
    }

    async def run(self, topic: str = "", max_results: int = 5) -> ToolResult:
        url = f"{_NEWS_FEED_URL}/search?q={topic}" if topic else _NEWS_FEED_URL
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S, headers={"User-Agent": _USER_AGENT}) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                return ToolResult(ok=False, output="", summary=f"News request failed: {exc}")

        try:
            root = ElementTree.fromstring(resp.text)
        except ElementTree.ParseError as exc:
            return ToolResult(ok=False, output="", summary=f"Could not parse RSS feed: {exc}")

        items = root.findall(".//item")[:max_results]
        headlines = [(item.findtext("title") or "").strip() for item in items]
        headlines = [h for h in headlines if h]
        return ToolResult(
            ok=True,
            output="\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines)),
            summary=f"{len(headlines)} headline(s)",
            meta={"headlines": headlines},
        )

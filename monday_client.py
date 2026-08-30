"""monday.com GraphQL API v2 client.

Responsibilities (R1, R2, R3, R11):
  - Authenticated, **read-only** access (only ``boards`` / ``items_page`` queries;
    no mutation is ever constructed).
  - Cursor pagination so boards larger than one page are never silently truncated.
  - Flatten ``column_values`` into a wide DataFrame keyed by column *title*.
  - A short TTL cache so a multi-turn conversation does not re-hit the API on every
    turn, while still fetching live (satisfies "no hardcoded data").
  - Retry with exponential backoff on 429/5xx, then fall back to the last good
    (stale) cache with an explicit staleness marker rather than presenting nothing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

from config import get_settings

API_URL = "https://api.monday.com/v2"

# Read-only board query. items_page paginates via cursor; next_items_page continues.
_BOARD_QUERY = """
query ($boardId: [ID!], $limit: Int!) {
  boards(ids: $boardId) {
    id
    name
    columns { id title type }
    items_page(limit: $limit) {
      cursor
      items {
        id
        name
        group { title }
        column_values { id text value type column { title } }
      }
    }
  }
}
"""

_NEXT_PAGE_QUERY = """
query ($cursor: String!, $limit: Int!) {
  next_items_page(cursor: $cursor, limit: $limit) {
    cursor
    items {
      id
      name
      group { title }
      column_values { id text value type column { title } }
    }
  }
}
"""


class MondayError(RuntimeError):
    """Raised when the API cannot be reached after retries and no cache exists."""


@dataclass
class BoardData:
    board_id: str
    name: str
    columns: list[dict[str, str]]        # [{id, title, type}]
    df: pd.DataFrame                      # one row per item, columns keyed by title
    fetched_at: float
    stale: bool = False                  # True when served from an expired cache

    @property
    def age_seconds(self) -> float:
        return time.time() - self.fetched_at


@dataclass
class _CacheEntry:
    data: BoardData
    expires_at: float


class MondayClient:
    """Thin, cached, read-only GraphQL client for monday.com boards."""

    def __init__(
        self,
        token: str | None = None,
        api_version: str | None = None,
        ttl_seconds: int | None = None,
        page_limit: int = 500,
        session: requests.Session | None = None,
    ) -> None:
        settings = get_settings()
        self._token = token or settings.monday_api_token
        self._api_version = api_version or settings.monday_api_version
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
        self._page_limit = page_limit
        self._session = session or requests.Session()
        self._cache: dict[str, _CacheEntry] = {}

    # ---- HTTP -----------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self._token:
            raise MondayError("MONDAY_API_TOKEN is not configured.")
        return {
            "Authorization": self._token,
            "API-Version": self._api_version,
            "Content-Type": "application/json",
        }

    def _post(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """POST a GraphQL query with exponential backoff on 429/5xx."""
        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._session.post(
                    API_URL,
                    json={"query": query, "variables": variables},
                    headers=self._headers(),
                    timeout=30,
                )
                if resp.status_code == 401:
                    raise MondayError(
                        "monday.com rejected the API token (401). Check MONDAY_API_TOKEN."
                    )
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = MondayError(f"Transient API status {resp.status_code}")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("errors"):
                    # Complexity-budget errors also arrive here as GraphQL errors.
                    msg = "; ".join(e.get("message", "?") for e in payload["errors"])
                    if "complexity" in msg.lower() or "rate" in msg.lower():
                        last_exc = MondayError(f"Complexity/rate limit: {msg}")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise MondayError(f"GraphQL error: {msg}")
                return payload["data"]
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                time.sleep(backoff)
                backoff *= 2
        raise MondayError(f"monday.com unreachable after retries: {last_exc}")

    # ---- Flattening -----------------------------------------------------------

    @staticmethod
    def _flatten(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in items:
            row: dict[str, Any] = {
                "item_id": item.get("id"),
                "item_name": item.get("name"),
                "group": (item.get("group") or {}).get("title"),
            }
            for cv in item.get("column_values", []) or []:
                title = (cv.get("column") or {}).get("title")
                if not title:
                    continue
                # Prefer human-readable text; keep raw JSON value for date/number types
                # where text can be empty even when a value exists.
                row[title] = cv.get("text")
                row[f"__raw__{title}"] = cv.get("value")
            rows.append(row)
        return rows

    # ---- Public API -----------------------------------------------------------

    def get_board(self, board_id: str, *, force: bool = False) -> BoardData:
        """Fetch a full board (all pages) with TTL caching and stale fallback."""
        now = time.time()
        cached = self._cache.get(board_id)
        if cached and not force and cached.expires_at > now:
            return cached.data

        try:
            data = self._fetch_board(board_id)
            self._cache[board_id] = _CacheEntry(data, now + self._ttl)
            return data
        except MondayError:
            if cached is not None:
                stale = cached.data
                stale.stale = True
                return stale
            raise

    def _fetch_board(self, board_id: str) -> BoardData:
        data = self._post(_BOARD_QUERY, {"boardId": [board_id], "limit": self._page_limit})
        boards = data.get("boards") or []
        if not boards:
            raise MondayError(f"Board {board_id} not found or not accessible.")
        board = boards[0]
        page = board.get("items_page") or {}
        items = list(page.get("items") or [])
        cursor = page.get("cursor")

        # Follow the cursor until the board is exhausted.
        while cursor:
            nxt = self._post(_NEXT_PAGE_QUERY, {"cursor": cursor, "limit": self._page_limit})
            npage = nxt.get("next_items_page") or {}
            items.extend(npage.get("items") or [])
            cursor = npage.get("cursor")

        rows = self._flatten(items)
        df = pd.DataFrame(rows)
        columns = [
            {"id": c.get("id"), "title": c.get("title"), "type": c.get("type")}
            for c in (board.get("columns") or [])
        ]
        return BoardData(
            board_id=str(board.get("id")),
            name=board.get("name") or board_id,
            columns=columns,
            df=df,
            fetched_at=time.time(),
        )


@dataclass
class Boards:
    """Convenience container holding both boards for a request."""

    deals: BoardData
    work_orders: BoardData
    warnings: list[str] = field(default_factory=list)


def load_boards(client: MondayClient | None = None, *, force: bool = False) -> Boards:
    """Load both configured boards, tolerating a single-board failure."""
    settings = get_settings()
    client = client or MondayClient()
    warnings: list[str] = []

    if not settings.deals_board_id or not settings.work_orders_board_id:
        raise MondayError("Board IDs are not configured (DEALS_BOARD_ID / WORK_ORDERS_BOARD_ID).")

    deals = client.get_board(settings.deals_board_id, force=force)
    work_orders = client.get_board(settings.work_orders_board_id, force=force)
    for b in (deals, work_orders):
        if b.stale:
            warnings.append(f"Serving cached '{b.name}' data (~{int(b.age_seconds)}s old); live fetch failed.")
    return Boards(deals=deals, work_orders=work_orders, warnings=warnings)

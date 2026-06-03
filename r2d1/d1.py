"""
r2d1.d1 — Cloudflare D1 client.

Thin wrapper around the Cloudflare REST API for D1.
Used by Courier (upsert checkpoint rows) and Restarter (query latest row).
"""
from __future__ import annotations

import json
from typing import Any, Optional

import requests

from .secrets import require_secret, secret


class D1Error(RuntimeError):
    pass


class D1Client:
    """
    Minimal D1 client over the Cloudflare REST API.

    Only two operations are needed:
        upsert(table, row)      -- insert or replace a row
        query_one(sql, params)  -- return first row of a SELECT or None
    """

    BASE = "https://api.cloudflare.com/client/v4"

    def __init__(self, account_id: str, database_id: str, api_token: str):
        self.account_id  = account_id
        self.database_id = database_id
        self._session    = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_token}",
            "Content-Type":  "application/json",
        })

    @classmethod
    def from_env(cls) -> "D1Client":
        return cls(
            account_id  = require_secret("R2D1_ACCOUNT_ID"),
            database_id = require_secret("R2D1_D1_DATABASE_ID"),
            api_token   = require_secret("R2D1_API_TOKEN"),
        )

    @classmethod
    def from_env_optional(cls) -> Optional["D1Client"]:
        """Return None (with a warning) if D1 credentials are missing."""
        try:
            return cls.from_env()
        except Exception:
            return None

    # ── internal ──────────────────────────────────────────────────────────────

    def _url(self) -> str:
        return f"{self.BASE}/accounts/{self.account_id}/d1/database/{self.database_id}/query"

    def _execute(self, sql: str, params: list[Any] | None = None) -> list[dict]:
        payload = {"sql": sql}
        if params:
            payload["params"] = params
        resp = self._session.post(self._url(), json=payload, timeout=15)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise D1Error(f"D1 HTTP error: {e} — {resp.text[:300]}") from e
        data = resp.json()
        if not data.get("success"):
            errors = data.get("errors", [])
            raise D1Error(f"D1 query failed: {errors}")
        results = data.get("result", [])
        if not results:
            return []
        return results[0].get("results", [])

    # ── public ────────────────────────────────────────────────────────────────

    def upsert(self, table: str, row: dict[str, Any]) -> None:
        """
        Insert or replace a row.  Uses INSERT OR REPLACE so the PRIMARY KEY
        constraint handles duplicates (idempotent re-ship of the same sidecar).
        """
        cols   = list(row.keys())
        values = list(row.values())
        placeholders = ", ".join("?" * len(cols))
        col_list     = ", ".join(cols)
        sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
        self._execute(sql, values)

    def query_one(self, sql: str, params: list[Any] | None = None) -> Optional[dict]:
        """Execute a SELECT and return the first row, or None."""
        rows = self._execute(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: list[Any] | None = None) -> list[dict]:
        """Execute arbitrary SQL and return all rows."""
        return self._execute(sql, params)

    def ensure_table(self) -> None:
        """Create the checkpoints table if it does not exist."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                job_id    TEXT    NOT NULL,
                name      TEXT    NOT NULL,
                epoch     INTEGER NOT NULL,
                timestamp REAL    NOT NULL,
                r2_prefix TEXT    NOT NULL,
                metadata  TEXT    DEFAULT '{}',
                PRIMARY KEY (job_id, name)
            )
        """)

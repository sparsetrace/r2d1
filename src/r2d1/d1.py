"""
r2d1.d1 — Cloudflare D1 REST client.

Thin wrapper around the D1 HTTP API.
Used by Courier (upsert heartbeat rows) and Fetcher (query latest checkpoint).
"""
from __future__ import annotations

import json
from typing import Optional

import requests

from .secrets import r2d1_config, missing_d1, config_error


class D1Error(RuntimeError):
    pass


class D1Client:

    _BASE = "https://api.cloudflare.com/client/v4"

    def __init__(self, account_id: str, api_token: str, database_id: str):
        self._account_id  = account_id
        self._database_id = database_id
        self._session     = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_token}",
            "Content-Type":  "application/json",
        })

    @classmethod
    def from_env(cls) -> "D1Client":
        cfg     = r2d1_config()
        missing = missing_d1(cfg)
        if missing:
            raise config_error("D1", missing)
        return cls(
            account_id  = cfg["account_id"],
            api_token   = cfg["api_token"],
            database_id = cfg["d1_database_id"],
        )

    @classmethod
    def from_env_optional(cls) -> Optional["D1Client"]:
        """Return D1Client if credentials available, else None (R2-only mode)."""
        try:
            return cls.from_env()
        except RuntimeError:
            return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _url(self) -> str:
        return (
            f"{self._BASE}/accounts/{self._account_id}"
            f"/d1/database/{self._database_id}/query"
        )

    def _post(self, sql: str, params: list | None = None) -> dict:
        body = {"sql": sql}
        if params:
            body["params"] = params
        resp = self._session.post(self._url(), json=body, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise D1Error(f"D1 error: {data.get('errors', [])}")
        return data

    # ── Public ────────────────────────────────────────────────────────────────

    def query_one(self, sql: str, params: list | None = None) -> Optional[dict]:
        data    = self._post(sql, params)
        results = data.get("result", [{}])
        rows    = results[0].get("results", []) if results else []
        return rows[0] if rows else None

    def query_all(self, sql: str, params: list | None = None) -> list[dict]:
        data    = self._post(sql, params)
        results = data.get("result", [{}])
        return results[0].get("results", []) if results else []

    def execute(self, sql: str, params: list | None = None) -> None:
        self._post(sql, params)

    def upsert(self, table: str, row: dict) -> None:
        cols         = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        values       = [
            v if not isinstance(v, (dict, list)) else json.dumps(v)
            for v in row.values()
        ]
        self.execute(
            f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
            f"VALUES ({placeholders})",
            values,
        )

    def ensure_checkpoints_table(self) -> None:
        self.execute("""
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

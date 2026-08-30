"""Central configuration loader.

Secrets are read from Streamlit's ``st.secrets`` when running inside Streamlit,
falling back to environment variables / a local ``.env`` for CLI and test use.
Nothing here is ever committed — see ``.streamlit/secrets.toml.example``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _get(key: str, default: str | None = None) -> str | None:
    """Read a secret from st.secrets first, then the environment."""
    # st.secrets raises if there is no secrets file, so guard the import/access.
    try:
        import streamlit as st  # local import: keeps config usable without Streamlit

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    monday_api_token: str | None
    monday_api_version: str
    deals_board_id: str | None
    work_orders_board_id: str | None
    # LLM layer is OpenAI-compatible; defaults target NVIDIA NIM (Nemotron), but any
    # OpenAI-compatible endpoint works by overriding base_url / model / key.
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    cache_ttl_seconds: int
    # Fiscal year start month (1-12). April = Indian fiscal convention; documented
    # in the Decision Log. Overridable via secrets so the reviewer can change it.
    fiscal_year_start_month: int

    @property
    def board_ids(self) -> list[str]:
        return [b for b in (self.deals_board_id, self.work_orders_board_id) if b]

    def missing(self) -> list[str]:
        """Names of required secrets that are absent — surfaced in the UI."""
        required = {
            "MONDAY_API_TOKEN": self.monday_api_token,
            "DEALS_BOARD_ID": self.deals_board_id,
            "WORK_ORDERS_BOARD_ID": self.work_orders_board_id,
            "LLM_API_KEY": self.llm_api_key,
        }
        return [name for name, value in required.items() if not value]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        monday_api_token=_get("MONDAY_API_TOKEN"),
        monday_api_version=_get("MONDAY_API_VERSION", "2024-10") or "2024-10",
        deals_board_id=_get("DEALS_BOARD_ID"),
        work_orders_board_id=_get("WORK_ORDERS_BOARD_ID"),
        # Accept LLM_API_KEY, or NVIDIA_API_KEY as a convenience alias.
        llm_api_key=_get("LLM_API_KEY") or _get("NVIDIA_API_KEY"),
        llm_base_url=_get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        or "https://integrate.api.nvidia.com/v1",
        llm_model=_get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
        or "nvidia/nemotron-3-ultra-550b-a55b",
        cache_ttl_seconds=int(_get("CACHE_TTL_SECONDS", "300") or "300"),
        fiscal_year_start_month=int(_get("FISCAL_YEAR_START_MONTH", "4") or "4"),
    )

"""Thin wrapper around the OpenAI SDK pointed at our local vLLM server."""
from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


def get_client(base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1"),
        timeout=float(os.environ.get("OPENAI_TIMEOUT", "180")),
    )

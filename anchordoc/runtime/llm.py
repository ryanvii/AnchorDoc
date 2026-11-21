"""Deterministic OpenAI-compatible LLM runtime with SHA256 caching and offline replay."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import httpx
from openai import OpenAI


class LLMClient(Protocol):
    """Minimal protocol implemented by CachedLLM and deterministic test doubles."""

    model: str

    def chat_json(
        self,
        purpose: str,
        system: str,
        user: str,
        max_tokens: int = 700,
        images_b64: list[str] | None = None,
    ) -> tuple[dict[str, Any], str]: ...

    def snapshot_stats(self) -> dict[str, Any]: ...


def extract_json(text: str) -> dict[str, Any]:
    """Robustly extract a JSON object from raw LLM text."""
    s = (text or "").strip()
    if not s:
        return {}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.S)
    if m:
        s = m.group(1)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _resolve_env_settings() -> tuple[str, str, str]:
    base = (
        os.environ.get("ANCHORDOC_LLM_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).strip()
    model = (
        os.environ.get("ANCHORDOC_LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()
    key = (
        os.environ.get("ANCHORDOC_LLM_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    key_file = (os.environ.get("ANCHORDOC_LLM_KEY_FILE") or "").strip()
    if not key and key_file:
        p = Path(key_file).expanduser()
        if p.is_file():
            key = p.read_text(encoding="utf-8").strip()
    return base, model, key


class CachedLLM:
    """Thread-safe OpenAI-compatible client with SHA256 disk caching and token accounting."""

    def __init__(
        self,
        cache_dir: str | Path = "./cache",
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        offline: bool | None = None,
        timeout: int = 120,
        disable_thinking: bool = True,
        trust_env_proxy: bool = False,
    ) -> None:
        eb, em, ek = _resolve_env_settings()
        self.base_url = (base_url or eb).rstrip("/")
        self.model = model or em
        self.api_key = api_key if api_key is not None else ek
        self.offline = (
            offline
            if offline is not None
            else os.environ.get("ANCHORDOC_OFFLINE", "").strip() in {"1", "true", "yes"}
        )
        self.disable_thinking = disable_thinking
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = (
            None
            if self.offline
            else OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "not-set",
                timeout=timeout,
                http_client=httpx.Client(trust_env=trust_env_proxy, timeout=timeout),
            )
        )
        self._lock = threading.Lock()
        self.stats: dict[str, Any] = {
            "calls": 0,
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_sec": 0.0,
        }

    def reset_stats(self) -> None:
        with self._lock:
            for k in self.stats:
                self.stats[k] = 0.0 if k == "latency_sec" else 0

    def snapshot_stats(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.stats)

    def _cache_key(
        self,
        purpose: str,
        system: str,
        user: str,
        max_tokens: int,
        images_b64: list[str] | None,
    ) -> str:
        img_hashes = [hashlib.sha256(b.encode("ascii")).hexdigest()[:16] for b in (images_b64 or [])]
        payload = json.dumps(
            {
                "model": self.model,
                "purpose": purpose,
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": 0,
                "images": img_hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def chat_json(
        self,
        purpose: str,
        system: str,
        user: str,
        max_tokens: int = 700,
        images_b64: list[str] | None = None,
    ) -> tuple[dict[str, Any], str]:
        key = self._cache_key(purpose, system, user, max_tokens, images_b64)
        cpath = self.cache_dir / f"{purpose}_{key}.json"
        if cpath.exists():
            rec = json.loads(cpath.read_text(encoding="utf-8"))
            with self._lock:
                self.stats["calls"] += 1
                self.stats["cache_hits"] += 1
                self.stats["prompt_tokens"] += int(rec.get("prompt_tokens", 0))
                self.stats["completion_tokens"] += int(rec.get("completion_tokens", 0))
            return rec.get("parsed", {}), rec.get("raw", "")

        if self.offline or self._client is None:
            raise RuntimeError(f"Offline cache miss: {cpath.name}")

        if images_b64:
            content: Any = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                for img in images_b64
            ]
        else:
            content = user

        last_err: Exception | None = None
        for attempt in range(4):
            t0 = time.time()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                }
                if self.disable_thinking:
                    try:
                        resp = self._client.chat.completions.create(
                            **kwargs,
                            extra_body={"thinking": {"type": "disabled"}},
                        )
                    except Exception:
                        resp = self._client.chat.completions.create(**kwargs)
                else:
                    resp = self._client.chat.completions.create(**kwargs)
                dt = time.time() - t0
                raw = (resp.choices[0].message.content or "").strip()
                parsed = extract_json(raw)
                usage = getattr(resp, "usage", None)
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
                rec = {
                    "model": self.model,
                    "purpose": purpose,
                    "raw": raw,
                    "parsed": parsed,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "latency_sec": round(dt, 3),
                }
                tmp = cpath.with_suffix(".tmp")
                tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(cpath)
                with self._lock:
                    self.stats["calls"] += 1
                    self.stats["prompt_tokens"] += pt
                    self.stats["completion_tokens"] += ct
                    self.stats["latency_sec"] += dt
                return parsed, raw
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

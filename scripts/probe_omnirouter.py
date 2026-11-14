#!/usr/bin/env python3
"""Probe the OmniRoute gateway the lab's local LLM tier depends on.

Run this ON THE MACHINE that can reach the gateway (the lab host). It reads
the SAME configuration the lab uses (env vars -> configs/lab.yaml ->
configs/secrets.yaml), so what it validates is exactly what the lab will do:

  python scripts/probe_omnirouter.py            # health + model-id audit
  python scripts/probe_omnirouter.py --ask      # + one real completion
  python scripts/probe_omnirouter.py --ask --concurrency 3
                                                # + parallel burst (502 repro:
                                                #   live 2026-11-14 21:25 two
                                                #   same-second asks -> HTTP 502
                                                #   while one call worked)

Checks:
  1. GET {base}/v1/models answers (the same read-only probe `available()`
     uses to confirm the OpenAI-compatible tier);
  2. the configured local_llm_model (and local_llm_vl_model, when set) are
     LISTED by the gateway — an id the gateway doesn't route is the #1
     silent-failure cause (every ask would 404 -> the tier degrades to
     "absent" and the pool pays for everything again);
  3. --ask: one bounded /v1/chat/completions call against the configured
     model, honoring LOCAL_LLM_API_KEY / max_tokens.

Exit code 0 = gateway healthy and every configured id is listed;
1 = something is off (also printed, never raises).

Stdlib only — no aiohttp/pytest dependency, safe on any host.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.config import get_settings  # noqa: E402


def _http(url: str, payload: dict | None,
          api_key: str, timeout: float) -> tuple[int, dict | str]:
    """One HTTP call -> (status, parsed-json-or-body). Never raises."""
    headers = {"Content-Type": "application/json"}
    key = (api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        # the BODY matters (the bare reason 'Bad Gateway' hid the real
        # upstream error during the live 2026-11-14 21:25 incident)
        try:
            return exc.code, exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001 — probe must print, not raise
            return exc.code, str(exc.reason)
    except Exception as exc:  # noqa: BLE001 — probe must print, not raise
        return 0, f"{type(exc).__name__}: {exc}"


def _model_ids(doc: dict | str) -> list[str]:
    """Model ids from an OpenAI-shape /v1/models document (data[].id)."""
    if not isinstance(doc, dict):
        return []
    ids: list[str] = []
    data = doc.get("data")
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("id"):
                ids.append(str(entry["id"]))
    models = doc.get("models")           # Ollama-style fallback
    if isinstance(models, list):
        for entry in models:
            if isinstance(entry, dict) and entry.get("name"):
                ids.append(str(entry["name"]))
    return ids


def _content_of(doc: dict | str) -> str:
    """choices[0].message.content from an OpenAI-shape completion."""
    if isinstance(doc, dict):
        choices = doc.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                return str(msg.get("content") or "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ask", action="store_true",
                    help="also run one real bounded completion")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="with --ask: fire N PARALLEL completions to test "
                         "whether the upstream handles bursts (502 repro)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-request timeout seconds (default 120)")
    args = ap.parse_args()

    s = get_settings()
    base = str(s.local_llm_url or "").rstrip("/")
    if base.endswith("/v1"):
        url_base = base
    else:                                 # OmniRoute tolerance, same as lab
        url_base = f"{base}/v1"
    model = str(s.local_llm_model or "").strip()
    vl_model = str(getattr(s, "local_llm_vl_model", "") or "").strip()
    api_key = str(getattr(s, "local_llm_api_key", "") or "")

    print(f"OmniRoute probe — base={url_base}")
    print(f"  enabled={s.local_llm_enabled} backend={s.local_llm_backend}"
          f" model={model or '(empty)'}"
          f" vl_model={vl_model or '(empty)'}"
          f" api_key={'set' if api_key.strip() else 'no'}")
    if not s.local_llm_enabled:
        print("  !! local_llm_enabled is FALSE — the lab will never call "
              "the gateway (set it in configs/secrets.yaml)")

    status, doc = _http(f"{url_base}/models", None, api_key, args.timeout)
    if status != 200 or not isinstance(doc, dict):
        print(f"  FAIL GET /models -> HTTP {status}: {doc}")
        return 1
    ids = _model_ids(doc)
    print(f"  OK GET /models -> {len(ids)} model(s) routed by the gateway")
    for mid in ids:
        print(f"    - {mid}")

    ok = True
    if model and ids and model not in ids:
        print(f"  !! configured model '{model}' is NOT in the gateway list —"
              " every ask would fail; fix local_llm_model in "
              "configs/secrets.yaml")
        ok = False
    if vl_model and ids and vl_model not in ids:
        print(f"  !! configured vl_model '{vl_model}' is NOT in the gateway "
              "list — reserved tier, but fix it now so a future vision "
              "purpose doesn't 404")
        ok = False

    if args.ask:
        payload = {
            "model": model,
            "messages": [{"role": "user",
                          "content": "Answer with exactly: OMNIROUTE OK"}],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 64,
        }
        if args.concurrency > 1:
            n = max(1, args.concurrency)
            print(f"  .. firing {n} PARALLEL completions (502 burst repro)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(
                    lambda _: _http(f"{url_base}/chat/completions", payload,
                                    api_key, args.timeout), range(n)))
            burst_ok = True
            for i, (st, doc_) in enumerate(results):
                content = _content_of(doc_)
                if st == 200 and content.strip():
                    print(f"    OK [{i}] -> {content.strip()[:120]!r}")
                else:
                    print(f"    FAIL [{i}] -> HTTP {st}: {str(doc_)[:250]}")
                    burst_ok = False
            if burst_ok:
                print("  OK concurrent burst — the upstream handles "
                      "parallel completions; local_llm_max_concurrent may "
                      "be raised")
            else:
                print("  !! parallel burst FAILED -> keep "
                      "local_llm_max_concurrent=1 (the lab serializes "
                      "local asks by default)")
                ok = False
        else:
            t_status, t_doc = _http(f"{url_base}/chat/completions", payload,
                                    api_key, args.timeout)
            content = _content_of(t_doc)
            if t_status == 200 and content.strip():
                print(f"  OK completion via '{model}' -> "
                      f"{content.strip()[:150]!r}")
            else:
                hint = ("  <- 502/503 with a short body usually means the "
                        "gateway's UPSTREAM refused: re-check the model id "
                        "above and the provider key inside OmniRoute, and "
                        "keep local_llm_max_concurrent=1"
                        ) if t_status in (502, 503) else ""
                print(f"  FAIL completion -> HTTP {t_status}: "
                      f"{str(t_doc)[:300]}{hint}")
                ok = False

    print("RESULT:", "HEALTHY" if ok else "BROKEN (see !! above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

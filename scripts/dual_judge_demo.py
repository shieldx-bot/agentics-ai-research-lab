"""Live demo: TWO model families judge the SAME tournament match.

The anti-"one model judges everything" run (owner, 2026-11-13): one
pairwise debate prompt goes to BOTH juries —

  * primary  : Perplexity through a Hyperbrowser stealth session
               (accounts/account-XX.json cookie, the lab's usual edge)
  * second   : ChatGPT through its own cookie session (accounts/chatgpt-01.json)

and both verdicts are parsed with the tournament's own tolerant
``better idea`` parser, then compared — agree / CONTESTED, exactly the
arbitration run_match applies inside the lab.

With >= 2 Hyperbrowser tokens the two asks run truly in PARALLEL
(asyncio.gather on separate tokens). On the one-token lab the single
session slot forces the two cloud sessions to run back-to-back (the
code path is still the concurrent one the tournament uses; the token is
the limiter, not the logic).

Usage:
    .venv/bin/python scripts/dual_judge_demo.py \
        [--pplx-account accounts/account-03.json] \
        [--cookies accounts/chatgpt-01.json] [--goal "..."] [--timeout 540]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentics.config import Settings  # noqa: E402
from agentics.coscientist import pairing  # noqa: E402
from agentics.llm.chatgpt_client import ChatGPTClient  # noqa: E402
from agentics.llm.hyperbrowser_client import HyperbrowserClient  # noqa: E402
from agentics.llm.perplexity_client import CookieAccount  # noqa: E402

JURY_SYSTEM = (
    "You are the ranking jury of a self-improving AI research lab. You "
    "compare research hypotheses through a simulated panel debate and "
    "always end with the exact verdict line you are asked for. Be honest "
    "and decisive; never invent facts."
)

DEMO_HYP1 = ("A gradient-boosted tree model over 12 VNF resource-usage features "
             "predicts virtual network function placement success better than a "
             "linear baseline on public NFV traces; measurable as placement_auc.")
DEMO_HYP2 = ("An online-learning LSTM adapts VNF autoscaling thresholds faster "
             "than periodic batch retraining under diurnal 5G-core load; "
             "measurable as adaptation_time_s on a replayed trace.")


def _build_prompt(goal: str) -> str:
    return pairing.PAIRWISE_PROMPT.format(
        goal=goal[:600], id1="h1", text1=DEMO_HYP1,
        id2="h2", text2=DEMO_HYP2, feedback_block="",
    )


async def _run(cfg: argparse.Namespace) -> int:
    settings = Settings()
    if not getattr(settings, "hyperbrowser_api_key", ""):
        print("no Hyperbrowser token in settings/.env — cannot run live")
        return 2
    prompt = _build_prompt(cfg.goal)

    pplx = HyperbrowserClient(
        CookieAccount(name="demo-pplx", cookies={},
                      path=str(Path(cfg.pplx_account).resolve())),
        settings, answer_wait_s=cfg.timeout)
    gpt = ChatGPTClient(Path(cfg.cookies).resolve(), settings,
                        answer_wait_s=cfg.timeout)

    pool = list(getattr(settings, "hyperbrowser_api_keys", None) or [])
    parallel = len(pool) >= 2
    print(f"hyperbrowser tokens: {len(pool) or 1} -> "
          + ("PARALLEL asks (2 tokens)" if parallel
             else "sequential asks (1 token: one session slot)"))

    async def _ask(llm, tag):
        t0 = time.monotonic()
        try:
            resp = await llm.ask(prompt, mode="concise",
                                 system=JURY_SYSTEM, purpose="dual_judge_demo")
            return tag, (resp.answer or ""), time.monotonic() - t0, None
        except Exception as exc:  # noqa: BLE001 — demo prints, never raises
            return tag, "", time.monotonic() - t0, str(exc)

    if parallel:
        (t1, a1, d1, e1), (t2, a2, d2, e2) = await asyncio.gather(
            _ask(pplx, "perplexity"), _ask(gpt, "chatgpt"))
    else:
        # one session slot: run the SLOW judge first so the total wall
        # time is dominated by it instead of blown past by both
        (t2, a2, d2, e2) = await _ask(gpt, "chatgpt")
        (t1, a1, d1, e1) = await _ask(pplx, "perplexity")

    print(f"\n=== primary ({t1}) in {d1:.0f}s ===")
    print((a1[:1200] + " …") if len(a1) > 1200 else (a1 or f"FAILED: {e1}"))
    print(f"\n=== second judge ({t2}) in {d2:.0f}s ===")
    print((a2[:1200] + " …") if len(a2) > 1200 else (a2 or f"FAILED: {e2}"))

    v1 = pairing.parse_better_idea(a1)
    v2 = pairing.parse_better_idea(a2)
    print(f"\nparsed verdicts: primary={v1} second={v2}")
    if v1 is None:
        print("RESULT: primary verdict unparseable — match would be `invalid`")
        return 4
    if v2 is None:
        print(f"RESULT: judge2 unavailable/unparseable — primary h{v1} stands "
              "(noted in rationale)")
        return 3
    if v1 == v2:
        print(f"RESULT: DUAL-JUDGE AGREED — h{v1} wins, Elo moves")
        return 0
    print(f"RESULT: CONTESTED — primary h{v1} vs second h{v2}: "
          "recorded with NO Elo movement for the owner")
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pplx-account", default="accounts/account-03.json")
    ap.add_argument("--cookies", default="accounts/chatgpt-01.json")
    ap.add_argument("--goal", default=None)
    ap.add_argument("--timeout", type=int, default=420,
                    help="per-ask answer wait in seconds")
    args = ap.parse_args()
    if not args.goal:
        args.goal = Settings().research_topic
    started = time.monotonic()
    rc = asyncio.run(_run(args))
    print(f"\nrc={rc} wall={time.monotonic() - started:.0f}s")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

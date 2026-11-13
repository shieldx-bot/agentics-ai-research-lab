"""P19 — the peer panel: agents evaluate each OTHER's work by vote.

Motivation (owner): "cải tiến cơ chế tự đánh giá lẫn nhau, nhằm cải thiện
kết quả chung — nếu kết quả bỏ phiếu chỉ cần agent không chấp nhận, hoặc
không bảo vệ được quan điểm thì reject luôn." A single reviewer's opinion is
not the swarm's verdict: every MR is judged by a panel of peers, and ONE
unanswered objection is enough to kill it.

State machine (the ``mr_panels`` row per MR):

    voting(round 1) ── every panel vote accept ─────────▶ approved
          │ ONE reject vote (decisive, immediately)
          ▼
       defense ── author submits a defense covering ───▶ voting(round 2)
          │            EVERY objection
          │ deadline passes with no/short defense ───────▶ rejected (FINAL)
          ▼
    voting(round 2): the round-1 objectors re-vote
          │ any reject still standing ───────────────────▶ rejected (FINAL)
          └ every live objector withdrawn ───────────────▶ approved

The contract this module enforces (production rules):

1. NO SELF-APPROVAL (P5): the author of an MR can never vote on it — the
   engine refuses the vote outright, whatever the caller claims.
2. ANY reject is decisive: in round 1 it opens the defense round; in round 2
   (after the defense) it rejects the MR for good. Approval requires
   unanimity of the panel — never a majority, never a tiebreak.
3. "Không bảo vệ được quan điểm thì reject luôn": the defense must answer
   EVERY objection (keyed by the objector's agent id) with a non-empty
   rebuttal. An incomplete defense, an explicit concession, or a defense
   that misses the deadline each end the MR as REJECTED — final. The task
   returns to the open pool so a fresh attempt (by anyone) can try again;
   the rejection reason stays in the traces for the swarm to learn from.
4. The panel is the first ``peer_panel_size`` eligible agents that cast a
   vote (default 2). A reject arriving before approval is decisive even if
   it comes from outside the quorum; after approval the MR belongs to the
   merge queue (its own gates still apply).
5. Liveness follows the swarm's own heartbeat rule: an eligible voter that
   lost its heartbeat stops blocking the tally, the same way stale claims
   are swept. A dead objector's objection counts as withdrawn — the swarm
   never waits on the dead.

Nothing here talks to an LLM: the engine is the mechanical rulebook. The
graph nodes (node_review / node_defend) gather the human-style judgment and
call cast_vote / submit_defense with the results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from agentics.memory.shared_memory import SharedMemory

# panel states (mr_panels.state)
VOTING = "voting"
DEFENSE = "defense"
APPROVED = "approved"
REJECTED = "rejected"

# decision outcomes returned to the callers
OUT_APPROVED = "approved"        # panel reached unanimity -> MR approved
OUT_DEFENSE = "defense"          # a reject vote opened the defense round
OUT_REJECTED = "rejected"        # FINAL: the MR is dead, the task reopens
OUT_WAITING = "waiting"          # more votes needed before the tally decides
OUT_RE_VOTING = "re_voting"      # defense accepted -> objectors re-vote (r2)
OUT_IGNORED = "ignored"          # vote/defense on a closed or non-voting panel


@dataclass
class PanelDecision:
    """What the engine just decided (and why) — the caller journals this."""
    outcome: str
    mr_id: int
    reason: str = ""
    panel: dict = field(default_factory=dict)
    votes: list = field(default_factory=list)

    @property
    def decisive(self) -> bool:
        return self.outcome in (OUT_APPROVED, OUT_REJECTED)


class PeerPanel:
    """The mechanical rulebook for agents evaluating each other's MRs."""

    def __init__(self, memory: SharedMemory, panel_size: int = 2,
                 defense_ttl_minutes: int = 30, stale_after_s: float = 900.0,
                 now=lambda: time.time()) -> None:
        self.memory = memory
        self.panel_size = max(1, int(panel_size))
        self.defense_ttl_s = max(60, int(defense_ttl_minutes) * 60)
        self.stale_after_s = float(stale_after_s)
        self._now = now                 # injectable clock (tests)

    # ------------------------------------------------------------ helpers
    def _mr(self, mr_id: int) -> dict | None:
        return self.memory.get_review(mr_id)

    def alive_agents(self) -> set[str]:
        """Agents with a fresh heartbeat — the swarm's own liveness rule."""
        return {a["agent"] for a in self.memory.active_agents(self.stale_after_s)}

    def eligible_voters(self, author: str) -> set[str]:
        """Everyone alive except the author — P5: no self-approval, ever."""
        return self.alive_agents() - {author}

    def panel_digest(self, mr_id: int) -> str:
        """One owner-readable line: state, round, votes, deadline."""
        panel = self.memory.get_panel(mr_id)
        if panel is None:
            return ""
        votes = self.memory.panel_votes(mr_id, round=panel["round"])
        counts: dict[str, int] = {}
        for v in votes:
            counts[v["vote"]] = counts.get(v["vote"], 0) + 1
        breakdown = " ".join(f"{k}:{n}" for k, n in sorted(counts.items())) or "no votes"
        due = ""
        if panel["state"] == DEFENSE and panel["due_ts"]:
            left = max(0, int(panel["due_ts"] - self._now()))
            due = f", defense due in {left // 60}m"
        return (f"P19 panel !#{mr_id} r{panel['round']} {panel['state']} "
                f"[{breakdown}]{due}")

    # ------------------------------------------------------------ voting
    def cast_vote(
        self, mr_id: int, voter: str, vote: str, reason: str = "",
    ) -> PanelDecision:
        """One reviewer, one vote, one round. A reject is recorded WITH its
        objection (the author defends against reasons, not opinions)."""
        mr = self._mr(mr_id)
        if mr is None:
            return PanelDecision(OUT_IGNORED, mr_id, "MR record vanished")
        if voter == mr["author"]:
            # P5 hard guard: not a request, a refusal — self-approval must
            # never even reach the tally.
            return PanelDecision(OUT_IGNORED, mr_id, "P5: the author cannot vote on their own MR")
        if vote not in ("accept", "reject"):
            return PanelDecision(OUT_IGNORED, mr_id, f"unknown vote {vote!r}")
        panel = self.memory.ensure_panel(mr_id)
        if panel["state"] != VOTING:
            return PanelDecision(
                OUT_IGNORED, mr_id,
                f"panel is {panel['state']}, not {VOTING} — vote dropped")
        if not self.memory.record_panel_vote(mr_id, panel["round"], voter, vote, reason):
            return PanelDecision(
                OUT_IGNORED, mr_id, f"{voter} already voted in round {panel['round']}")
        if vote == "reject":
            self.memory.record_panel_objection(
                mr_id, panel["round"], voter, reason or "no reason given")
        return self.tally(mr_id, note=reason if vote == "accept" else "")

    # ------------------------------------------------------------ tally
    def tally(self, mr_id: int, note: str = "") -> PanelDecision:
        """Apply the panel rules to the CURRENT round's votes.

        ``note`` is the deciding voter's comment — it rides the MR record so
        the owner sees WHY the panel said yes (e.g. the mechanical tier-2
        prefix and the P16 evidence digest)."""
        mr = self._mr(mr_id)
        panel = self.memory.get_panel(mr_id)
        if mr is None or panel is None:
            return PanelDecision(OUT_IGNORED, mr_id, "MR or panel record missing")
        if panel["state"] != VOTING:
            return PanelDecision(OUT_IGNORED, mr_id, f"panel is {panel['state']}")
        rnd = panel["round"]
        votes = self.memory.panel_votes(mr_id, round=rnd)
        accepts = [v for v in votes if v["vote"] == "accept"]
        rejects = [v for v in votes if v["vote"] == "reject"]
        if rnd == 1:
            if rejects:
                # ONE not-accepting agent is enough: open the defense round
                # immediately — the MR is NOT approved "for now" while it waits.
                due = self._now() + self.defense_ttl_s
                self.memory.set_panel(mr_id, state=DEFENSE, due_ts=due)
                panel = self.memory.get_panel(mr_id)
                return PanelDecision(
                    OUT_DEFENSE, mr_id,
                    reason="; ".join(
                        f"{r['voter']}: {r['objection'][:120]}"
                        for r in self.memory.panel_objections(mr_id, round=1)),
                    panel=panel or {}, votes=votes)
            eligible = self.eligible_voters(mr["author"])
            quorum = min(self.panel_size, len(eligible))
            if accepts and len(accepts) >= quorum:
                return self._approve(mr_id, [v["voter"] for v in accepts], note=note)
            return PanelDecision(OUT_WAITING, mr_id,
                                 reason=f"{len(accepts)}/{quorum} accept votes",
                                 panel=panel, votes=votes)
        # round 2 — only the round-1 objectors hold a veto now; their peers
        # who accepted in round 1 keep that standing (the defense did not
        # concern them). A dead objector's objection is withdrawn.
        objectors = [o["voter"] for o in self.memory.panel_objections(mr_id, round=1)]
        alive = self.alive_agents()
        live_objectors = [o for o in objectors if o in alive]
        r2 = {v["voter"]: v["vote"] for v in votes}
        still_reject = [v for v in votes if v["vote"] == "reject"]
        if still_reject:
            return self._reject(
                mr_id, by=still_reject[0]["voter"],
                reason="defense rejected — "
                       + "; ".join(f"{v['voter']}: {v['reason'][:120]}" for v in still_reject))
        unswung = [o for o in live_objectors if r2.get(o) != "accept"]
        if unswung:
            return PanelDecision(OUT_WAITING, mr_id,
                                 reason=f"awaiting re-vote from {unswung}",
                                 panel=panel, votes=votes)
        return self._approve(mr_id, list(dict.fromkeys(
            [v["voter"] for v in self.memory.panel_votes(mr_id) if v["vote"] == "accept"])),
            note=note)

    def _approve(self, mr_id: int, voters: list[str], note: str = "") -> PanelDecision:
        by = f"panel[{','.join(voters)[:200]}]"
        comment = f"P19 unanimous panel: {voters}"
        if note.strip():
            comment = f"{comment} — {note[:400]}"
        self.memory.set_review_verdict(mr_id, "approved", by, comment)
        self.memory.set_panel(mr_id, state=APPROVED, clear_due=True)
        self.memory.set_trace_outcome(mr_id, "approved")
        panel = self.memory.get_panel(mr_id)
        self.memory.journal("peer-panel", "panel_approved",
                            {"mr": mr_id, "voters": voters[:10]})
        return PanelDecision(OUT_APPROVED, mr_id, reason=f"unanimous {by}",
                             panel=panel or {})

    def _reject(self, mr_id: int, by: str, reason: str) -> PanelDecision:
        """FINAL rejection — the MR is dead. The task returns to the open
        pool (a fresh attempt may try again); tools tied to the MR are
        dropped; the reason becomes a trace every agent can learn from."""
        self.memory.set_review_verdict(mr_id, "failed", by, reason[:3000])
        self.memory.set_panel(mr_id, state=REJECTED, clear_due=True)
        self.memory.set_trace_outcome(mr_id, "failed", reason[:1500])
        self.memory.drop_tools(mr_id)
        mr = self._mr(mr_id)
        if mr and mr.get("task_id"):
            self.memory.set_task_status(mr["task_id"], "open", claimed_by="")
        self.memory.journal(by, "panel_rejected",
                            {"mr": mr_id, "why": reason[:300],
                             "task": (mr or {}).get("task_id")})
        self.memory.post_message(
            by, "alert",
            f"P19 panel REJECTED MR #{mr_id} (final): {reason[:200]}",
            (mr or {}).get("task_id"),
        )
        panel = self.memory.get_panel(mr_id)
        return PanelDecision(OUT_REJECTED, mr_id, reason=reason, panel=panel or {})

    # ------------------------------------------------------------ defense
    def open_defense(self, mr_id: int) -> float:
        """Deadline for the author to defend (used by the defense round)."""
        panel = self.memory.get_panel(mr_id)
        return panel["due_ts"] if panel and panel["due_ts"] else 0.0

    def submit_defense(
        self, mr_id: int, author: str, rebuttals: dict[str, str],
        note: str = "", evidence_ids: list[int] | None = None,
    ) -> PanelDecision:
        """The author answers EVERY objection — one shot, mechanically checked.

        ``rebuttals`` maps objector id -> rebuttal text. Coverage is enforced
        here, not by hope: an incomplete defense IS a failed defense
        ("không bảo vệ được quan điểm thì reject luôn")."""
        mr = self._mr(mr_id)
        panel = self.memory.get_panel(mr_id)
        if mr is None or panel is None:
            return PanelDecision(OUT_IGNORED, mr_id, "MR or panel record missing")
        if author != mr["author"]:
            return PanelDecision(OUT_IGNORED, mr_id, "only the author defends")
        if panel["state"] != DEFENSE:
            return PanelDecision(OUT_IGNORED, mr_id, f"panel is {panel['state']}")
        objections = self.memory.panel_objections(mr_id, round=1)
        missing = [o["voter"] for o in objections
                   if not str(rebuttals.get(o["voter"], "")).strip()]
        defense = {
            "rebuttals": {k: str(v)[:2000] for k, v in rebuttals.items()},
            "note": note[:2000],
            "evidence_ids": (evidence_ids or [])[:20],
            "objectors": [o["voter"] for o in objections],
            "missing": missing,
            "ts": self._now(),
        }
        if missing:
            # The mechanical bar failed: the viewpoint was not defended.
            return self._reject(
                mr_id, by=author,
                reason=f"defense incomplete — no rebuttal for {missing}")
        self.memory.set_panel(mr_id, state=VOTING, round=2, clear_due=True,
                              defense=defense)
        self.memory.journal(author, "panel_defense",
                            {"mr": mr_id, "covers": list(rebuttals)[:10]})
        self.memory.post_message(
            author, "broadcast",
            f"P19 defense of MR #{mr_id} submitted — objectors {list(rebuttals)[:5]} "
            "re-vote now (one reject ends the MR).",
            mr.get("task_id"),
        )
        panel = self.memory.get_panel(mr_id)
        return PanelDecision(OUT_RE_VOTING, mr_id, panel=panel or {})

    def concede(self, mr_id: int, author: str, note: str = "") -> PanelDecision:
        """The author gives up the defense — honest, and immediately final."""
        mr = self._mr(mr_id)
        panel = self.memory.get_panel(mr_id)
        if mr is None or panel is None or author != mr["author"]:
            return PanelDecision(OUT_IGNORED, mr_id, "only the author concedes")
        if panel["state"] != DEFENSE:
            return PanelDecision(OUT_IGNORED, mr_id, f"panel is {panel['state']}")
        return self._reject(
            mr_id, by=author,
            reason=f"author conceded the defense: {note[:200] or '(no note)'}")

    # ------------------------------------------------------------ sweep
    def sweep(self) -> list[tuple[int, str]]:
        """Deadline + liveness janitor, run by the Supervisor's scan.

        Returns (mr_id, what-happened) pairs for the journal. Handles the
        three ways a panel can stall on its own:

        * a defense round whose deadline passed without a defense,
        * a voting panel whose MR record vanished,
        * a voting round-1 panel whose voters are all gone (approval was
          reachable but nobody ever voted) — left open on purpose: a fresh
          agent may still join; dead VOTES never block, dead SILENCE does
          not approve either.
        """
        events: list[tuple[int, str]] = []
        now = self._now()
        for panel in self.memory.panels_in_state(DEFENSE):
            mr_id = panel["mr_id"]
            if panel["due_ts"] and now >= panel["due_ts"]:
                mr = self._mr(mr_id)
                by = (mr or {}).get("author", "peer-panel")
                self._reject(mr_id, by=by,
                             reason="defense deadline passed without a defense")
                events.append((mr_id, "defense deadline -> rejected"))
        for panel in self.memory.panels_in_state(VOTING):
            mr_id = panel["mr_id"]
            if self._mr(mr_id) is None:
                self.memory.set_panel(mr_id, state=REJECTED, clear_due=True)
                events.append((mr_id, "MR record vanished -> panel dropped"))
        return events

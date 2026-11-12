"""ResearchState — the LangGraph state flowing through one research cycle."""
from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict


def merge_unique(current: list, incoming: list) -> list:
    """Append-only reducer that keeps the first occurrence of each item.

    Items may be unhashable (papers are dicts), so the seen-set is built
    from a stable JSON key instead of putting items into a set directly.
    """
    out: list = []
    seen: set[str] = set()
    for item in [*current, *incoming]:
        if isinstance(item, str):
            key = item
        else:
            try:
                key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                key = repr(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


class ResearchState(TypedDict, total=False):
    # identity
    agent_id: str
    role: str
    # workflow
    action: str                    # what this cycle will do
    issue_number: int
    issue_title: str
    issue_body: str
    branch: str
    # research artifacts
    query: str
    papers: Annotated[list[dict], merge_unique]
    citations: Annotated[list[str], merge_unique]
    bibtex: str
    # implementation artifacts
    plan: str
    changed_files: list[str]
    commit_message: str
    category: str                  # issue category (research/code/docs/...)
    verification: str              # tier-1 result, shown in the PR body
    reuses: list[str]              # registry tools the plan says it reuses
    # review artifacts
    pr_number: int
    pr_url: str
    review_verdict: str            # approve | request_changes
    review_comment: str
    # control
    memory_context: dict[str, Any]
    yield_reason: str
    summary: str
    error: str
    done: bool

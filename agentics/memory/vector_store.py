"""Vector store — semantic search over findings/papers (ChromaDB, persistent).

ChromaDB is optional at runtime: if it is not installed the lab still works,
just without semantic recall (the shared-memory tool/skill lookups fall back
to keyword-overlap ranking in SQLite).
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger("memory.vector")

try:
    import chromadb  # type: ignore
    from chromadb.config import Settings as ChromaSettings  # type: ignore
    _HAS_CHROMA = True
except Exception:  # noqa: BLE001
    _HAS_CHROMA = False


class VectorStore:
    def __init__(self, path: Path, collection: str = "findings") -> None:
        self._ok = _HAS_CHROMA
        self._col = None
        if not self._ok:
            log.warning("chromadb not installed — vector memory disabled")
            return
        try:
            client = chromadb.PersistentClient(
                path=str(path),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
            )
            self._col = client.get_or_create_collection(collection)
        except Exception as exc:  # noqa: BLE001
            log.error("vector store init failed: %s", exc)
            self._ok = False

    @property
    def available(self) -> bool:
        return self._ok

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        if not self._ok or not text.strip():
            return ""
        doc_id = uuid.uuid4().hex
        meta = {k: str(v)[:500] for k, v in (metadata or {}).items()}
        try:
            self._col.add(documents=[text[:8000]], ids=[doc_id], metadatas=[meta])
            return doc_id
        except Exception as exc:  # noqa: BLE001
            log.error("vector add failed: %s", exc)
            return ""

    def search(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        if not self._ok or not query.strip():
            return []
        try:
            res = self._col.query(query_texts=[query], n_results=n)
        except Exception as exc:  # noqa: BLE001
            log.error("vector query failed: %s", exc)
            return []
        out: list[dict[str, Any]] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            entry = {"text": doc, "meta": meta or {}}
            if i < len(dists):
                entry["distance"] = dists[i]
            out.append(entry)
        return out

    # v3 sync-trim: the findings table self-trims to its newest rows, and the
    # vector tier must follow — otherwise the index fills with docs whose
    # SQLite row is gone (ghosts). Recall skips ghosts at read time, but they
    # still eat the n_results budget and the index grows forever.
    def delete_many(self, doc_ids: list[str]) -> int:
        """Delete vector documents by id. Returns how many ids were accepted
        (0 when the store is offline or the list is empty)."""
        ids = [d for d in (doc_ids or []) if d]
        if not self._ok or not ids:
            return 0
        try:
            self._col.delete(ids=ids)
            return len(ids)
        except Exception as exc:  # noqa: BLE001
            log.error("vector delete failed (%d ids): %s", len(ids), exc)
            return 0

    def count(self) -> int:
        """Documents currently indexed — the `memory-stats` observability of
        the sync-trim (findings rows and vector docs should track together)."""
        if not self._ok:
            return 0
        try:
            return int(self._col.count())
        except Exception as exc:  # noqa: BLE001
            log.error("vector count failed: %s", exc)
            return 0

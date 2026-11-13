"""Co-Scientist architecture for the agentics lab.

A faithful behavioural port of the tournament-evolution core of
Co-Scientist (Gottweis et al., Nature 655, 2026 — "Accelerating scientific
discovery with Co-Scientist"), modelled on the Apache-2.0 open-source
re-implementation at https://github.com/Kaimen-Inc/Co-Scientist.

What is ported here, and what deliberately is NOT:

* PORTED — the self-improvement loop the paper's ablation credits most:
  an Elo tournament that ranks hypotheses by simulated scientific debate
  (pairwise verdicts -> zero-sum Elo, K scheduled by experience), pair
  selection that maximises information per match (new entrants, close-Elo
  but idea-distant pairs, occasional top-heavy random), evolution operators
  that breed the next generation from top-Elo hypotheses (combine /
  out-of-box, with parent lineage), and a meta-review that distils
  win/loss patterns into feedback for the next rounds.

* NOT PORTED — the supervisor/task queue (the lab already runs a
  claim/lease orchestrator over LangGraph cycles), the FAISS embedder
  (the lab has ChromaDB vector memory), and — crucially — the notion
  that an LLM review is the ground truth. In this lab a hypothesis is
  only SETTLED by running code (P16 exit-code + measured-metric verdicts,
  P24 independent replication); the tournament adds a QUALITY dimension
  on top of that truth dimension, never a replacement for it.

Every LLM call passes the lab's P20 budget gate and every operator is
failure-isolated: the tournament may never break a research cycle.
"""

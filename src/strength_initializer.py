"""
Strength Initializer — A priori strength initialization  (Step 5)
Reads bas_repair.jsonl and/or bas_no_repair.jsonl (from bas_assembler.py);
writes initialized_bas_repair.jsonl and initialized_bas_no_repair.jsonl.

  UI (s1) — Uniform Initialization
  SEI (s2) — Sructural Engagement Score
  CSI (s3) — Contextual Semantic Score
  HI (s4) — Hybrid Initialization
Usage:
    python src/strength_initializer.py \
    --input-repair      bas_repair.jsonl \
    --input-no-repair   bas_no_repair.jsonl \
    --output-repair     initialized_repair.jsonl \
    --output-no-repair  initialized_no_repair.jsonl \
    --strategy          all \

"""
import logging
import argparse
import math
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import networkx as nx

from sys_utils import (
    read_jsonl, JSONLWriter, log_progress, count_lines,
    SAMPLE_CONVERSATIONS, setup_logging, load_config, DEFAULT_CONFIG
)

log = logging.getLogger("strength_initializer")

# make mapping between new and old ids.
STRATEGY_MAP = {"ui": "s1", "sei": "s2", "csi": "s3", "hi": "s4"}
ALL_STRATEGIES = ["ui", "sei", "csi", "hi"]

# set this to true if we want to force central prop. to initialize at 1.0
ROOT_STRENGTH_OVERWRITE = False

# SEI hyper-parameters
SEI_ALPHA = 0.5   # OP authorship bonus factor
SEI_BETA  = 0.5   # proximity bonus factor

# CSI hyperparameters

# HI hyper-parameters
HI_DELTA  = 0.5   # SEI amplification weight (0=CSI only, 1=full amplification)
HI_MODE   = "amplifier"  # "additive" or "amplifier"

# ─── Helper functions
def _normalise(values: dict) -> dict:
    """Min-max normalise to [0, 1]. Returns all 1.0 if range is zero."""
    v = list(values.values())
    lo, hi = min(v), max(v)
    if math.isclose(lo, hi):
        return {k: 1.0 for k in values}
    return {k: (val - lo) / (hi - lo) for k, val in values.items()}


def _bas_to_digraph(bas: dict) -> nx.DiGraph:
    """
    develops a directed graph for each input BAS
    :param bas:
    :return:
    """
    G = nx.DiGraph()
    for node in bas.get("nodes", []):
        G.add_node(node["id"], **node)
    for edge in bas.get("edges", []):
        G.add_edge(edge["source"], edge["target"],
                   relation=edge["relation"],
                   weight=edge.get("weight", 1.0)
                   )
    return G


def _edge_weight_from_nodes(ns: dict, edges: list) -> dict:
    """Edge weights are uniform 1.0 for this study — only node strengths vary. This can be experimented
     with in future work.
    """
    return {(e["source"], e["target"]): 1.0 for e in edges}


def _apply_strengths(
        bas: dict, node_strengths: dict,
        edge_weights: dict, suffix: str
        ) -> dict:
    """Write initial_strength_{suffix} and initial_weight_{suffix} into bas in-place."""
    for node in bas["nodes"]:
        node[f"initial_strength_{suffix}"] = round(
            float(node_strengths.get(node["id"], 1.0)), 6
            )
    for edge in bas["edges"]:
        edge[f"initial_weight_{suffix}"] = round(
            float(edge_weights.get((edge["source"], edge["target"]), 1.0)), 6
            )
    return bas


# ─── Strategy implementations

def strategy_ui(bas: dict) -> dict:
    """UI — Uniform Initialization: all nodes and edges assigned strength 1.0."""
    log.info("[UI/s1] Uniform initialization")
    ids = [n["id"] for n in bas["nodes"]]
    pairs = [(e["source"], e["target"]) for e in bas["edges"]]
    return _apply_strengths(
        bas,
        {nid: 1.0 for nid in ids},
        {p: 1.0 for p in pairs},
        "s1",
        )


def strategy_sei(
        bas:   dict,
        alpha: float = SEI_ALPHA,
        beta:  float = SEI_BETA,
) -> dict:
    """
    SEI — Structural Engagement Initialization.

    Downstream-looking: evaluates each node through the argumentative activity
    it anchors below it in the graph.

    SEI_raw(a) = Σ_{d ∈ D(a)}  PageRank(d)
                                · [1 + α · OP(d)]
                                · [1 + β · (1 / depth(d, a))]

    where:
      D(a)          — all descendants of a in the directed graph
      PageRank(d)   — globally computed PageRank of d (relevance of incoming nodes)
      OP(d)         — 1 if d was authored by the OP, 0 otherwise
      depth(d, a)   — shortest directed path length from a to d
      α             — OP authorship bonus factor
      β             — proximity bonus factor

    Non-OP nodes and distant descendants are not penalised — bonuses are additive.
    Scores are min-max normalised to [0,1].
    """
    log.info("[SEI/s2] Structural Engagement Initialization  α=%.2f  β=%.2f",
             alpha, beta)

    nodes = bas.get("nodes", [])
    if not nodes:
        return bas

    G = _bas_to_digraph(bas)

    # Global PageRank over the full graph
    try:
        pr = nx.pagerank(G, alpha=0.85)
    except nx.PowerIterationFailedConvergence:
        log.warning("[SEI/s2] PageRank failed to converge — using degree centrality")
        pr = nx.degree_centrality(G)

    # OP speaker — stored on the root node or summary
    root_id  = bas.get("summary", {}).get("root_id", "")
    root_node = next((n for n in nodes if n["id"] == root_id), None)
    op_speaker = root_node.get("speaker_id", "") if root_node else ""
    log.info("[SEI/s2] OP speaker_id=%r", op_speaker)

    # Build speaker_id lookup
    speaker_map: dict[str, str] = {n["id"]: n.get("speaker_id", "") for n in nodes}

    raw: dict[str, float] = {}
    # Reverse the graph: in BAS, edges go source→target (challenger→challenged).
    # D(a) = all nodes that can reach a by following edges forward = all
    # predecessors of a = descendants on the reversed graph.
    G_rev = G.reverse(copy=True)

    for node in nodes:
        nid = node["id"]

        # All nodes that eventually point toward nid (following BAS edges)
        try:
            desc_lengths = dict(nx.single_source_shortest_path_length(G_rev, nid))
        except Exception:
            desc_lengths = {nid: 0}

        # Exclude self
        desc_lengths = {d: depth for d, depth in desc_lengths.items()
                        if d != nid and depth > 0}

        if not desc_lengths:
            raw[nid] = 0.0
            continue

        score = 0.0
        for d, depth in desc_lengths.items():
            pr_d   = pr.get(d, 0.0)
            op_d   = 1.0 if (op_speaker and speaker_map.get(d) == op_speaker) else 0.0
            score += pr_d * (1.0 + alpha * op_d) * (1.0 + beta * (1.0 / depth))

        raw[nid] = score

    ns = _normalise(raw)

    if root_id and root_id in ns:
        ns[root_id] = 1.0
        log.info("[SEI/s2] Root %s fixed to 1.0", root_id)

    ew = _edge_weight_from_nodes(ns, bas["edges"])
    return _apply_strengths(bas, ns, ew, "s2")



# ─── CSI helpers ──────────────────────────────────────────────────────────────

# Default hyper-parameters (can be overridden via CLI)
CSI_LAMBDA_CTX   = 0.5    # λ_ctx  — contextual influence diminishing factor
CSI_EPS          = 0.4    # ε_threshold — explicitness below which expansion triggers
CSI_ALPHA        = 0.5    # α — balance between LocalRelevance and TopicRelevance
CSI_GAMMA        = 0.5    # γ — balance between max and mean SimScore


def _linguistic_entropy(text: str) -> float:
    """
    Shannon entropy of the unigram token distribution of `text`.
    H(a) = -Σ p(w) log2 p(w)  over word tokens.
    Returns 0.0 for empty or single-token texts.
    """
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    from collections import Counter
    counts = Counter(tokens)
    total  = len(tokens)
    probs  = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _compute_explicitness(nodes: list[dict]) -> dict[str, float]:
    """
    Explicitness(a) = 1 - H(a) / H_max
    where H_max is the maximum entropy across all nodes in the graph.
    High entropy → low explicitness (vague, implicit language).
    """
    entropies = {n["id"]: _linguistic_entropy(n["text"]) for n in nodes}
    h_max = max(entropies.values()) if entropies else 1.0
    if math.isclose(h_max, 0.0):
        return {nid: 1.0 for nid in entropies}
    return {nid: 1.0 - h / h_max for nid, h in entropies.items()}


def _build_relation_maps(edges: list[dict]) -> tuple[
        dict[str, list[str]],   # support_parents[node]  → [source nodes]
        dict[str, list[str]],   # attack_parents[node]   → [source nodes]
        dict[str, list[str]],   # support_children[node] → [target nodes]
        dict[str, list[str]],   # attack_children[node]  → [target nodes]
]:
    """
    Build directional neighbour maps from the edge list.
    Edge direction: source → target (source argues about target).
    Parents of a = nodes that point TO a.
    Children of a = nodes that a points TO.
    """
    support_parents:  dict[str, list[str]] = {}
    attack_parents:   dict[str, list[str]] = {}
    support_children: dict[str, list[str]] = {}
    attack_children:  dict[str, list[str]] = {}

    for e in edges:
        src, tgt, rel = e["source"], e["target"], e.get("relation", "")
        if rel == "support":
            support_parents .setdefault(tgt, []).append(src)
            support_children.setdefault(src, []).append(tgt)
        elif rel == "attack":
            attack_parents  .setdefault(tgt, []).append(src)
            attack_children .setdefault(src, []).append(tgt)

    return support_parents, attack_parents, support_children, attack_children


def _find_expansion_node(
        node_id:          str,
        explicitness:     dict[str, float],
        support_parents:  dict[str, list[str]],
        support_children: dict[str, list[str]],
        attack_children:  dict[str, list[str]],
        eps:              float,
) -> tuple[str | None, int]:
    """
    Context expansion for implicit arguments (Explicitness < ε_threshold).

    Priority (preserving alignment with a's argumentative direction):
      1. Support-parent with highest explicitness  (depth 1, λ^1)
      2. Support-child  with highest explicitness  (depth 2, λ^2)
      3. Attacking grandchild (attack-child's attack-child)
         with highest explicitness                 (depth 3, λ^3)

    Returns (expansion_node_id, depth) or (None, 0) if no qualifying node found.
    """
    # 1. Support parents
    sp = [p for p in support_parents.get(node_id, [])
          if explicitness.get(p, 0.0) >= eps]
    if sp:
        best = max(sp, key=lambda p: explicitness.get(p, 0.0))
        return best, 1

    # 2. Support children
    sc = [c for c in support_children.get(node_id, [])
          if explicitness.get(c, 0.0) >= eps]
    if sc:
        best = max(sc, key=lambda c: explicitness.get(c, 0.0))
        return best, 2

    # 3. Attacking grandchildren: attack-child → attack-grandchild
    #    (attack of attack = indirect support — preserves alignment)
    grandchildren = []
    for ac in attack_children.get(node_id, []):
        for gc in attack_children.get(ac, []):
            if explicitness.get(gc, 0.0) >= eps:
                grandchildren.append(gc)
    if grandchildren:
        best = max(grandchildren, key=lambda g: explicitness.get(g, 0.0))
        return best, 3

    return None, 0


def _contextual_embeddings(
        nodes:       list[dict],
        edges:       list[dict],
        encoder,                   # SentenceTransformer instance
        explicitness: dict[str, float],
        lambda_ctx:  float,
        eps:         float,
) -> dict[str, np.ndarray]:
    """
    Compute contextual embeddings for all nodes:

      ContextEmbed(a) =
        Embed(a)                                  if Explicitness(a) ≥ ε
        Embed(a) + λ    · Embed(parent*(a))       if depth=1
        Embed(a) + λ²   · Embed(child*(a))        if depth=2
        Embed(a) + λ³   · Embed(grand*(a))        if depth=3
        Embed(a)                                  otherwise

    Embeddings are L2-normalised after combination.
    """
    texts  = [n["text"] for n in nodes]
    nids   = [n["id"]   for n in nodes]

    raw_embeds: np.ndarray = encoder.encode(texts, normalize_embeddings=True,
                                            show_progress_bar=False)
    embed_map: dict[str, np.ndarray] = dict(zip(nids, raw_embeds))

    sp, ap, sc, ac = _build_relation_maps(edges)

    ctx_embeds: dict[str, np.ndarray] = {}
    for nid in nids:
        e_a = embed_map[nid]
        if explicitness.get(nid, 1.0) >= eps:
            ctx_embeds[nid] = e_a
            continue

        exp_id, depth = _find_expansion_node(
            nid, explicitness, sp, sc, ac, eps
        )
        if exp_id is None or exp_id not in embed_map:
            ctx_embeds[nid] = e_a
            continue

        lam_k = lambda_ctx ** depth
        combined = e_a + lam_k * embed_map[exp_id]
        # L2-normalise the combined vector
        norm = np.linalg.norm(combined)
        ctx_embeds[nid] = combined / norm if norm > 0 else e_a

    return ctx_embeds


def strategy_csi(
        bas:        dict,
        model_name: str   = "paraphrase-multilingual-MiniLM-L12-v2",
        lambda_ctx: float = CSI_LAMBDA_CTX,
        eps:        float = CSI_EPS,
        alpha:      float = CSI_ALPHA,
        gamma:      float = CSI_GAMMA,
) -> dict:
    """
    CSI — Contextual Semantic Initialization.

    Steps:
      1. Compute Explicitness via linguistic (Shannon) entropy.
      2. Context expansion for implicit nodes (Explicitness < ε):
           support-parents → support-children → attack-grandchildren.
      3. Contextual embeddings = Embed(a) + λ^k · Embed(expansion_node).
      4. CSI(a) = α · LocalRelevance(a) + (1-α) · TopicRelevance(a)
           LocalRelevance(a) = (1-γ)·max_t SimScore(a,t) + γ·mean_t SimScore(a,t)
           TopicRelevance(a) = cosine_sim(ContextEmbed(a), Embed(title))
      5. Root (P_central) is set to 1.0 after normalisation.
      6. Min-max normalise to [0,1].
    """
    log.info("[CSI/s3] Contextual Semantic Initialization  "
             "λ=%.2f  ε=%.2f  α=%.2f  γ=%.2f", lambda_ctx, eps, alpha, gamma)

    nodes = bas.get("nodes", [])
    edges = bas.get("edges", [])
    if not nodes:
        return bas

    nids   = [n["id"]   for n in nodes]
    root_id = bas.get("summary", {}).get("root_id")
    title   = bas.get("summary", {}).get("root_text", "")

    # ── 1. Explicitness ────────────────────────────────────────────────────────
    explicitness = _compute_explicitness(nodes)
    log.info("[CSI/s3] Explicitness computed — mean=%.3f  min=%.3f  max=%.3f",
             float(np.mean(list(explicitness.values()))),
             min(explicitness.values()), max(explicitness.values()))

    # ── 2 & 3. Contextual embeddings ──────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model_name)

    ctx_embeds = _contextual_embeddings(
        nodes, edges, encoder, explicitness, lambda_ctx, eps
    )

    # ── 4. CSI scores ──────────────────────────────────────────────────────────
    # Build relation maps for T(a) — the nodes a directly points to
    sp, ap, sc, ac = _build_relation_maps(edges)

    # T(a) = nodes that a directly points to (children: both support and attack)
    all_children: dict[str, list[str]] = {}
    for e in edges:
        all_children.setdefault(e["source"], []).append(e["target"])

    # Topic embedding
    if title:
        topic_embed = encoder.encode([title], normalize_embeddings=True)[0]
    else:
        log.warning("[CSI/s3] No title found — TopicRelevance will be 0 for all nodes")
        topic_embed = None

    raw_scores: dict[str, float] = {}
    for nid in nids:
        ce_a = ctx_embeds[nid]

        # Local Relevance — similarity to direct targets T(a)
        targets = all_children.get(nid, [])
        if targets:
            sim_scores = np.array([
                float(np.dot(ce_a, ctx_embeds[t]))   # both L2-normalised
                for t in targets if t in ctx_embeds
            ])
            local_rel = ((1 - gamma) * sim_scores.max()
                         + gamma * sim_scores.mean())
        else:
            # No outgoing edges — node doesn't engage any target directly
            local_rel = 0.0

        # Topic Relevance
        topic_rel = float(np.dot(ce_a, topic_embed)) if topic_embed is not None else 0.0

        raw_scores[nid] = alpha * local_rel + (1 - alpha) * topic_rel

    # ── 5 & 6. Normalise and fix root to 1.0 ──────────────────────────────────
    ns = _normalise(raw_scores)

    if root_id and root_id in ns:
        ns[root_id] = 1.0
        log.info("[CSI/s3] Root %s fixed to 1.0 (central proposition)", root_id)

    ew = _edge_weight_from_nodes(ns, edges)
    return _apply_strengths(bas, ns, ew, "s3")



def strategy_hi(
        bas:        dict,
        model_name: str   = "paraphrase-multilingual-MiniLM-L12-v2",
        lambda_ctx: float = CSI_LAMBDA_CTX,
        eps:        float = CSI_EPS,
        csi_alpha:  float = CSI_ALPHA,
        gamma:      float = CSI_GAMMA,
        sei_alpha:  float = SEI_ALPHA,
        beta:       float = SEI_BETA,
        delta:      float = HI_DELTA,
        mode:       str   = HI_MODE,
) -> dict:
    """
    HI — Hybrid Initialization combining CSI and SEI.

    CSI (s3) — upstream, semantic: how relevant and well-formed is this argument?
    SEI (s2) — downstream, structural: how much argumentative activity does it anchor?

    Two modes:
      "additive"   : HI(a) = normalise( CSI(a) + SEI(a))
      "amplifier"  : HI(a) = normalise( CSI(a) · (1 + δ · SEI(a)) )
                     δ=0 → HI=CSI;  δ=1 → up to 2× CSI for fully engaged nodes.
    """
    log.info("[HI/s4] Hybrid init  mode=%s  δ=%.2f", mode, delta)

    sei_key = "initial_strength_s2"
    csi_key = "initial_strength_s3"

    sei_present = all(sei_key in n for n in bas["nodes"])
    csi_present = all(csi_key in n for n in bas["nodes"])

    if not sei_present:
        bas = strategy_sei(deepcopy(bas), sei_alpha, beta)
    if not csi_present:
        bas = strategy_csi(deepcopy(bas), model_name, lambda_ctx, eps, csi_alpha, gamma)

    sei_scores = {n["id"]: n[sei_key] for n in bas["nodes"]}
    csi_scores = {n["id"]: n[csi_key] for n in bas["nodes"]}

    ns: dict[str, float] = {}
    for nid in sei_scores:
        csi_v = csi_scores[nid]
        sei_v = sei_scores[nid]
        if mode == "amplifier":
            ns[nid] = csi_v * (1.0 + delta * sei_v)
        else:  # additive
            ns[nid] = csi_v + sei_v

    ns = _normalise(ns)
    ew = _edge_weight_from_nodes(ns, bas["edges"])
    return _apply_strengths(bas, ns, ew, "s4")


# ─── Main initialization orchestrator ─────────────────────────────────────────

def initialize_strengths(
        bas:        dict,
        strategies: list,
        model_name: str   = "paraphrase-multilingual-MiniLM-L12-v2",
        lambda_ctx: float = CSI_LAMBDA_CTX,
        eps:        float = CSI_EPS,
        csi_alpha:  float = CSI_ALPHA,
        gamma:      float = CSI_GAMMA,
        sei_alpha:  float = SEI_ALPHA,
        beta:       float = SEI_BETA,
        delta:      float = HI_DELTA,
        hi_mode:    str   = HI_MODE,
) -> dict:
    """
    Run requested strategies on a single BAS dict.
    Returns a lean output dict containing only fields needed by Step 6.
    """
    resolved = ALL_STRATEGIES if "all" in strategies else [
        s for s in strategies if s in ALL_STRATEGIES
        ]

    result = deepcopy(bas)
    for strat in resolved:
        if strat == "ui":
            result = strategy_ui(result)
        elif strat == "sei":
            result = strategy_sei(result, sei_alpha, beta)
        elif strat == "csi":
            result = strategy_csi(result, model_name, lambda_ctx, eps, csi_alpha, gamma)
        elif strat == "hi":
            result = strategy_hi(result, model_name, lambda_ctx, eps,
                                 csi_alpha, gamma, sei_alpha, beta, delta, hi_mode)

    # ── Override root node strength to 1.0 across all strategies ─────────────
    root_id = bas.get("summary", {}).get("root_id")
    if root_id and ROOT_STRENGTH_OVERWRITE:
        for node in result["nodes"]:
            if node["id"] == root_id:
                for strat in resolved:
                    suffix = STRATEGY_MAP[strat]
                    key = f"initial_strength_{suffix}"
                    if key in node and node[key] != 1.0:
                        log.info(
                            "Root %s initial_strength_%s overridden: %.4f → 1.0",
                            root_id, suffix, node[key],
                            )
                        node[key] = 1.0
                break

    # ── Ensure root is influentiable by gradual semantics ─────────────────────
    # If the root has zero incoming edges, the iterative update p^(t) = W+^T s^(t)
    # will always be zero for the root — the discourse can never affect P_central.
    # Fix: reverse all direct outgoing edges from the root so they point inward,
    # making the root's children sources of influence rather than targets.
    # Relation type is preserved — support stays support, attack stays attack —
    # since the semantic relationship is symmetric in this context.
    if root_id:
        edges = result["edges"]
        incoming_ids = {e["source"] for e in edges if e["target"] == root_id}
        outgoing = [e for e in edges if e["source"] == root_id]

        if not incoming_ids and outgoing:
            log.info(
                "Root %s has 0 incoming edges — reversing %d outgoing edge(s) "
                "to make it influentiable by gradual semantics",
                root_id, len(outgoing),
                )
            for edge in outgoing:
                log.info(
                    "  Reversing: %s → %s (%s)  →  %s → %s (%s)",
                    edge["source"], edge["target"], edge["relation"],
                    edge["target"], edge["source"], edge["relation"],
                    )
                edge["source"], edge["target"] = edge["target"], edge["source"]

    # ── Build lean output — only what Step 6 (gradual_semantics) needs ────────
    strength_keys = [f"initial_strength_{STRATEGY_MAP[s]}" for s in resolved]
    weight_keys = [f"initial_weight_{STRATEGY_MAP[s]}" for s in resolved]

    lean_nodes = [
        {
            "id": n["id"],
            **{k: n[k] for k in strength_keys if k in n},
            }
        for n in result["nodes"]
        ]
    lean_edges = [
        {
            "source": e["source"],
            "target": e["target"],
            "relation": e["relation"],
            "synthetic": e.get("synthetic", False),  # needed by gradual_semantics weight matrices
            **{k: e[k] for k in weight_keys if k in e},
            }
        for e in result["edges"]
        ]

    return {
        "thread_id": result.get("thread_id", "unknown"),
        "conv_id": result.get("conv_id", "unknown"),
        "is_delta": result.get("is_delta", False),
        "mode": result.get("mode", "unknown"),
        "summary": result.get("summary", {}),
        "nodes": lean_nodes,
        "edges": lean_edges,
        }


# ─── JSONL batch runner ───────────────────────────────────────────────────────

def run_on_jsonl(
        input_path: Path, output_path: Path, strategies: list,
        model_name: str, lambda_ctx: float, eps: float,
        csi_alpha: float, gamma: float,
        sei_alpha: float, beta: float,
        delta: float, hi_mode: str,
) -> None:
    total = count_lines(input_path)
    log.info("Strength init starting — %d conversations  input=%s", total, input_path)
    with JSONLWriter(output_path) as writer:
        for line_no, bas in read_jsonl(input_path):
            log_progress(line_no, total, bas.get("thread_id", ""), "INIT", log)
            writer.write(initialize_strengths(
                bas, strategies, model_name,
                lambda_ctx, eps, csi_alpha, gamma,
                sei_alpha, beta, delta, hi_mode,
            ))
    log.info("Finished. Output → %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strength initializer — BAS JSONL in / initialized BAS JSONL out"
        )
    parser.add_argument("--input-repair",    "-r", help="Path to bas_repair.jsonl")
    parser.add_argument("--input-no-repair", "-n", help="Path to bas_no_repair.jsonl")
    parser.add_argument("--output-repair",    default="initialized_bas_repair.jsonl")
    parser.add_argument("--output-no-repair", default="initialized_bas_no_repair.jsonl")
    parser.add_argument("--configs", "-c", default=None,
                        help="Unified use-case profile JSON (see pipeline_config.py). "
                             "Supplies defaults for the strategy hyper-parameters from "
                             "the profile's 'strength' section; any explicit flag below "
                             "overrides it. e.g. configs/social_media.json. If omitted, "
                             "the built-in default is used.")
    parser.add_argument("--strategy", "-s", nargs="+",
                        choices=["ui", "sei", "csi", "hi", "all"], default=["all"],
                        help="UI=uniform  SEI=structural-engagement  CSI=contextual-semantic  HI=hybrid"
                        )
    parser.add_argument("--model-name", default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="Sentence encoder model name (default: paraphrase-multilingual-MiniLM-L12-v2)")
    # CSI hyper-parameters
    parser.add_argument("--lambda-ctx", type=float, default=CSI_LAMBDA_CTX,
                        help=f"λ_ctx: contextual influence decay (default: {CSI_LAMBDA_CTX})")
    parser.add_argument("--eps", type=float, default=CSI_EPS,
                        help=f"ε: explicitness threshold for context expansion (default: {CSI_EPS})")
    parser.add_argument("--csi-alpha", type=float, default=CSI_ALPHA,
                        help=f"α: LocalRelevance vs TopicRelevance balance (default: {CSI_ALPHA})")
    parser.add_argument("--gamma", type=float, default=CSI_GAMMA,
                        help=f"γ: max vs mean SimScore balance (default: {CSI_GAMMA})")
    # SEI hyper-parameters
    parser.add_argument("--sei-alpha", type=float, default=SEI_ALPHA,
                        help=f"α: OP authorship bonus factor (default: {SEI_ALPHA})")
    parser.add_argument("--beta", type=float, default=SEI_BETA,
                        help=f"β: proximity bonus factor (default: {SEI_BETA})")
    # HI hyper-parameters
    parser.add_argument("--delta", type=float, default=HI_DELTA,
                        help=f"δ: SEI amplification weight in HI (default: {HI_DELTA})")
    parser.add_argument("--hi-mode", choices=["additive", "amplifier"], default=HI_MODE,
                        help=f"HI combination mode (default: {HI_MODE})")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log-file", "-l", default="strength_initializer.log")
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)
    # Load configs profile.
    try:
        cfg = load_config(args.configs, DEFAULT_CONFIG)
    except ValueError as exc:
        parser.error(str(exc))
    s = cfg.get("strength", {})
    csi = s.get("csi", {}); sei = s.get("sei", {})
    hi = s.get("hi", {})

    def pick(flag, val, fallback):
        return flag if flag is not None else (val if val is not None else fallback)

    model_name = pick(args.model_name, s.get("model_name"), "all-MiniLM-L6-v2")
    lambda_ctx = pick(args.lambda_ctx, csi.get("lambda_ctx"), CSI_LAMBDA_CTX)
    eps = pick(args.eps, csi.get("eps"), CSI_EPS)
    csi_alpha = pick(args.csi_alpha, csi.get("alpha"), CSI_ALPHA)
    gamma = pick(args.gamma, csi.get("gamma"), CSI_GAMMA)
    sei_alpha = pick(args.sei_alpha, sei.get("alpha"), SEI_ALPHA)
    beta = pick(args.beta, sei.get("beta"), SEI_BETA)
    delta = pick(args.delta, hi.get("delta"), HI_DELTA)
    hi_mode = pick(args.hi_mode, hi.get("mode"), HI_MODE)

    log.info(
        "configs=%s  strategies=%s  model=%s  "
        "CSI[λ=%.2f ε=%.2f α=%.2f γ=%.2f]  "
        "SEI[α=%.2f β=%.2f]  "
        "HI[δ=%.2f mode=%s]",
        (args.config or "builtin_default"), args.strategy, model_name,
        lambda_ctx, eps, csi_alpha, gamma,
        sei_alpha, beta,
        delta, hi_mode,
        )

    kwargs = dict(
        strategies=args.strategy, model_name=model_name,
        lambda_ctx=lambda_ctx, eps=eps,
        csi_alpha=csi_alpha, gamma=gamma,
        sei_alpha=sei_alpha, beta=beta,
        delta=delta, hi_mode=hi_mode,
        )

    if args.input_repair:
        run_on_jsonl(Path(args.input_repair), Path(args.output_repair), **kwargs)

    if args.input_no_repair:
        run_on_jsonl(Path(args.input_no_repair), Path(args.output_no_repair), **kwargs)

    if not args.input_repair and not args.input_no_repair:
        log.info("No --input — chaining from built-in sample")
        import extract_edu as ee, pac_selector as ps
        import llm_reasoner as lr, bas_assembler as ba
        with JSONLWriter(Path(args.output_repair)) as writer:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS), conv.get("thread_id", ""), "INIT", log)
                repair_bas, _ = ba.assemble_bas(
                    lr.run_reasoning(ps.select_all_pacs(ee.extract_edus(conv)))
                    )
                if repair_bas:
                    writer.write(initialize_strengths(repair_bas, **kwargs))


if __name__ == "__main__":
    main()

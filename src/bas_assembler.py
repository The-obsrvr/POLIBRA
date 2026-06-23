"""
BAS Assembler — Bipolar Argument Structure assembly (Step 4)
Builds on llm_reasoner_old.py output.

Constructs G(A, E, P_central) where:
  A          = set of argumentative EDUs (nodes)
  E          = set of support / attack relations (edges)
  P_central  = central proposition — identified in Step 1 and
               carried verbatim in the `central_proposition` field; this
               module only matches it against the node set

Two output modes are always produced in a single pass:
  repair    — disconnected components are stitched to P_central's component
              via synthetic sentiment-based edges, yielding a single unified
              graph that preserves all argument activity. The ENTIRE dataset
              is preserved
  no-repair — only the connected subgraph containing P_central is retained;
              isolated components are discarded. Discussions whose central
              component has ≤ 3 argumentative units are dropped

Usage:
    python bas_assembler.py --input reasoning_input.jsonl \\
                            --output-repair    bas_repair.jsonl \\
                            --output-no-repair bas_no_repair.jsonl
    python bas_assembler.py --export-dot bas.dot   # GraphViz DOT export
    python bas_assembler.py                        # built-in sample
"""

import logging
import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx

from sys_utils import (read_jsonl, log_progress, JSONLWriter, count_lines,
                       SAMPLE_CONVERSATIONS, setup_logging, load_config, DEFAULT_CONFIG)


# ─── Logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger("bas_assembler")

# ─── Constants ────────────────────────────────────────────────────────────────

EMBED_MODEL             = DEFAULT_CONFIG["bas"]["embed_model"]   # only used when repair runs
MIN_ARGUMENTATIVE_UNITS = DEFAULT_CONFIG["bas"]["min_argumentative_units"]
_ENCODER                = None


def apply_config(cfg: dict) -> None:
    """Install Step-4 (BAS) settings from a resolved configs dict."""
    global EMBED_MODEL, MIN_ARGUMENTATIVE_UNITS
    bas = cfg.get("bas", {})
    EMBED_MODEL = bas.get("embed_model", EMBED_MODEL)
    MIN_ARGUMENTATIVE_UNITS = bas.get("min_argumentative_units", MIN_ARGUMENTATIVE_UNITS)


def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading encoder for repair mode")
        _ENCODER = SentenceTransformer(EMBED_MODEL)
    return _ENCODER

# 1 · Extraction helpers
def extract_nodes_and_edges(
    reasoning_output: dict,
) -> tuple[dict[str, dict], list[dict]]:
    """
    Walk the reasoning output and collect:
      nodes  — dict keyed by node-id ("edu_<global_idx>") for every
               argumentative EDU, carrying all speaker and structural
               metadata required by downstream steps.
      edges  — list of raw relation dicts for every support/attack PAC
               pair where the target is argumentative.  Speaker IDs of
               both endpoints are carried on each edge so downstream
               tasks need no node lookup.

    Non-argumentative EDUs and neutral relations are silently dropped.
    """
    nodes: dict[str, dict] = {}
    raw_edges: list[dict]  = []

    for turn in reasoning_output.get("conversation", []):
        if turn.get("deleted"):
            continue
        turn_post_id        = turn.get("post_id", "")
        turn_parent_post_id = turn.get("parent_post_id", "")
        turn_speaker_id     = turn.get("speaker_id", "")

        for edu in turn.get("edus", []):
            if not isinstance(edu, dict):
                continue

            if not edu.get("target_is_argumentative"):
                log.debug("  Dropping non-arg EDU [%s]: %.55s",
                          edu.get("global_idx"), edu.get("text", ""))
                continue

            g_idx      = edu["global_idx"]
            node_id    = f"edu_{g_idx}"
            # EDU dicts from Step 1 carry their own speaker_id; fall back to turn
            speaker_id = edu.get("speaker_id") or turn_speaker_id

            nodes[node_id] = {
                "id":             node_id,
                "global_idx":     g_idx,
                "post_id":        turn_post_id,
                "parent_post_id": turn_parent_post_id,
                "speaker_id":     speaker_id,
                "text":           edu.get("text", ""),
                "is_root":        False,
                "in_degree":      0,
                "out_degree":     0,
            }

            for pac in edu.get("pacs", []):
                rel = pac.get("relation", "neutral")
                if rel not in ("support", "attack"):
                    continue
                src_g_idx = pac.get("source_global_idx")
                if src_g_idx is None:
                    continue

                raw_edges.append({
                    "source":           f"edu_{src_g_idx}",
                    "target":           node_id,
                    "relation":         rel,
                    # source speaker resolved later in promote_source_nodes
                    # once the full node set is available
                    "source_speaker_id": pac.get("source_speaker_id", ""),
                    "target_speaker_id": speaker_id,
                    "synthetic":        False,
                })

    log.info("Extracted %d argumentative nodes, %d raw edges (before pruning)",
             len(nodes), len(raw_edges))
    return nodes, raw_edges


def promote_source_nodes(
    nodes:     dict[str, dict],
    raw_edges: list[dict],
    reasoning_output: dict,
) -> tuple[dict[str, dict], list[dict]]:
    """
    Promote source EDUs that are not yet in the argumentative node set:
    an EDU that is the source of a support/attack relation is argumentative
    by virtue of being engaged with, regardless of the LLM's direct
    classification of that EDU.

    Duplicate and bidirectional edge resolution (INTRA PAC pairs can produce
    both directions):
      1. PRIMARY rule — preserve post flow: keep the edge whose source has a
         lower global_idx than its target (earlier → later in the discussion).
         If both directions have the same orientation relative to idx order,
         keep the one that is already in the canonical (src_idx < tgt_idx) order.
      2. SECONDARY rule — when the directionality is equal, prefer the stronger
         relation (attack > support).

    Speaker IDs on edges are resolved here once the full node set is known.
    """
    RELATION_RANK = {"attack": 1, "support": 0}

    # Build a full lookup of all EDUs (argumentative or not) for promotion
    all_edus: dict[str, dict] = {}
    for turn in reasoning_output.get("conversation", []):
        if turn.get("deleted"):
            continue
        turn_speaker_id     = turn.get("speaker_id", "")
        turn_post_id        = turn.get("post_id", "")
        turn_parent_post_id = turn.get("parent_post_id", "")
        for edu in turn.get("edus", []):
            if not isinstance(edu, dict):
                continue
            g_idx      = edu.get("global_idx")
            node_id    = f"edu_{g_idx}"
            speaker_id = edu.get("speaker_id") or turn_speaker_id
            all_edus[node_id] = {
                "id":             node_id,
                "global_idx":     g_idx,
                "post_id":        turn_post_id,
                "parent_post_id": turn_parent_post_id,
                "speaker_id":     speaker_id,
                "text":           edu.get("text", ""),
                "is_root":        False,
                "in_degree":      0,
                "out_degree":     0,
            }

    promoted = 0
    # best[(canonical_src, canonical_tgt)] → edge dict
    # canonical order: lower global_idx first in the key
    best: dict[tuple, dict] = {}

    for edge in raw_edges:
        src, tgt = edge["source"], edge["target"]

        # Promote source if not yet in the argumentative set
        if src not in nodes:
            if src in all_edus:
                nodes[src] = all_edus[src]
                promoted += 1
                log.debug("Promoted source EDU %s to argumentative node", src)
            else:
                continue   # source not found in conversation — skip

        src_idx = nodes[src]["global_idx"]
        tgt_idx = nodes[tgt]["global_idx"] if tgt in nodes else float("inf")

        # Canonical key: always (lower_idx_node, higher_idx_node)
        # so forward and reverse edges between the same pair share one slot
        if src_idx <= tgt_idx:
            canon_key    = (src, tgt)
            flow_correct = True    # source is earlier — preserves post flow
        else:
            canon_key    = (tgt, src)
            flow_correct = False   # source is later — against post flow

        if canon_key not in best:
            best[canon_key] = {**edge, "flow_correct": flow_correct}
        else:
            existing      = best[canon_key]
            existing_flow = existing["flow_correct"]
            # Primary: prefer the edge that goes with post flow
            if flow_correct and not existing_flow:
                best[canon_key] = {**edge, "flow_correct": flow_correct}
            # Secondary: same flow status — prefer stronger relation
            elif flow_correct == existing_flow:
                if (RELATION_RANK.get(edge["relation"], 0) >
                        RELATION_RANK.get(existing["relation"], 0)):
                    best[canon_key] = {**edge, "flow_correct": flow_correct}

    # Reconstruct final edge list with canonical (earlier → later) direction
    # and resolved speaker IDs, stripping the internal flow_correct flag
    edges: list[dict] = []
    for (canon_src, canon_tgt), edge in best.items():
        src_speaker = nodes.get(canon_src, {}).get("speaker_id", "")
        tgt_speaker = nodes.get(canon_tgt, {}).get("speaker_id", "")
        edges.append({
            "source":            canon_src,
            "target":            canon_tgt,
            "relation":          edge["relation"],
            "source_speaker_id": src_speaker,
            "target_speaker_id": tgt_speaker,
            "synthetic":         edge.get("synthetic", False),
        })

    log.info("Promoted %d source EDUs → %d total nodes  %d edges (after dedup)",
             promoted, len(nodes), len(edges))
    return nodes, edges


# 2 · Degree annotation
def annotate_degrees(
    nodes: dict[str, dict],
    edges: list[dict],
) -> None:
    """Fill in_degree / out_degree on each node in-place."""
    for node in nodes.values():
        node["in_degree"]  = 0
        node["out_degree"] = 0

    for edge in edges:
        if edge["source"] in nodes:
            nodes[edge["source"]]["out_degree"] += 1
        if edge["target"] in nodes:
            nodes[edge["target"]]["in_degree"]  += 1


# 3 · repair mode
def _embed_nodes(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings for a list of texts."""
    encoder = get_encoder()
    return encoder.encode(texts, convert_to_numpy=True,
                          normalize_embeddings=True
                          ).astype(np.float32)


def repair_connectivity(
    nodes: dict[str, dict],
    edges: list[dict],
) -> list[dict]:
    """
    Guarantee that the argument graph is weakly connected.

    For each disconnected component:
      1. Find the best connection to the main component — the pair
         (minor_nid, main_nid) with the highest cosine similarity.
      2. Determine relation from polarity between the two nodes:
         sim ≥ 0 → SUPPORT, sim < 0 → ATTACK.
      3. Connect with conversational directionality (earlier → later).
    """
    if not nodes:
        return edges

    G = nx.DiGraph()
    G.add_nodes_from(nodes.keys())
    for e in edges:
        G.add_edge(e["source"], e["target"], relation=e["relation"])
    comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    if len(comps) == 1:
        log.info("Graph is already connected — no repair needed")
        return edges

    log.info("Graph has %d weakly-connected components — repairing", len(comps))

    node_id_list = list(nodes.keys())
    texts        = [nodes[nid]["text"] for nid in node_id_list]
    embs         = _embed_nodes(texts)
    id_to_emb    = {nid: embs[i] for i, nid in enumerate(node_id_list)}
    synthetic_edges: list[dict] = []
    main_component:  set[str]   = set(comps[0])

    for minor_comp in comps[1:]:
        # ── Step 1: find the best connection across the component boundary ────
        best_score = -1.0
        best_src   = None
        best_tgt   = None
        for minor_nid in minor_comp:
            for main_nid in main_component:
                score = float(id_to_emb[minor_nid] @ id_to_emb[main_nid])
                if score > best_score:
                    best_score = score
                    best_src   = minor_nid
                    best_tgt   = main_nid

        if best_src and best_tgt:
            # Relation determined by polarity between the two connected nodes:
            # positive similarity → support, negative → attack
            relation = "support" if best_score >= 0 else "attack"

            # ── Step 3: enforce conversational directionality ─────────────────
            if nodes[best_src]["global_idx"] > nodes[best_tgt]["global_idx"]:
                best_src, best_tgt = best_tgt, best_src

            syn_edge = {
                "source":            best_src,
                "target":            best_tgt,
                "relation":          relation,
                "source_speaker_id": nodes[best_src]["speaker_id"],
                "target_speaker_id": nodes[best_tgt]["speaker_id"],
                "synthetic":         True,
            }
            synthetic_edges.append(syn_edge)
            main_component |= minor_comp
            log.info(
                "  Synthetic edge: %s → %s  relation=%s  sim=%.3f",
                best_src, best_tgt, relation, best_score,
            )

    if synthetic_edges:
        log.info("Added %d synthetic edge(s) to repair connectivity", len(synthetic_edges))

    return edges + synthetic_edges


# 5 · BAS summary statistics
def compute_summary(
    nodes:   dict[str, dict],
    edges:   list[dict],
    root_id: Optional[str],
) -> dict:
    support_edges   = [e for e in edges if e["relation"] == "support"]
    attack_edges    = [e for e in edges if e["relation"] == "attack"]
    synthetic_edges = [e for e in edges if e.get("synthetic")]

    # Speaker breakdown
    speaker_counts = Counter(n["speaker_id"] for n in nodes.values())

    return {
        "total_nodes":       len(nodes),
        "total_edges":       len(edges),
        "support_edges":     len(support_edges),
        "attack_edges":      len(attack_edges),
        "synthetic_edges":   len(synthetic_edges),
        "root_id":           root_id,
        "root_text":         nodes[root_id]["text"] if root_id else None,
        "speaker_breakdown": dict(speaker_counts),
    }


def extract_central_subgraph(
    nodes:   dict[str, dict],
    edges:   list[dict],
    root_id: str,
) -> tuple[dict[str, dict], list[dict]]:
    """
    No-repair mode: retain only the weakly-connected component that contains
    P_central (root_id). All other components are discarded.
    Returns filtered (nodes, edges).
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes.keys())
    for e in edges:
        G.add_edge(e["source"], e["target"])

    # Find the component containing root_id
    for comp in nx.weakly_connected_components(G):
        if root_id in comp:
            central_comp = comp
            break
    else:
        log.warning("root_id %s not found in any component — returning full graph", root_id)
        return nodes, edges

    discarded = len(nodes) - len(central_comp)
    if discarded:
        log.info(
            "No-repair: retaining central component (%d nodes), "
            "discarding %d node(s) in isolated components",
            len(central_comp), discarded,
        )

    filtered_nodes = {nid: nodes[nid] for nid in central_comp}
    filtered_edges = [e for e in edges
                      if e["source"] in central_comp and e["target"] in central_comp]
    return filtered_nodes, filtered_edges


def is_too_small(nodes: dict[str, dict]) -> bool:
    """Return True if the BAS has insufficient argument activity (≤ 3 units)."""
    return len(nodes) <= MIN_ARGUMENTATIVE_UNITS


# 6 · DOT / GraphViz export
def export_dot(bas: dict, path: Path) -> None:
    """
    Write a GraphViz DOT file for visualisation.
    Nodes are labelled with a short text snippet.
    Support edges = green solid; attack edges = red dashed.
    Root node is double-circled.
    """
    root_id = bas.get("summary", {}).get("root_id")
    lines   = ["digraph BAS {", '  rankdir=LR;', '  node [shape=box fontsize=10];']

    for node in bas["nodes"]:
        nid     = node["id"]
        speaker = node.get("speaker_id", "?")
        label   = f"[{speaker}] {node['text'][:35]}".replace('"', "'")
        shape   = 'doublecircle' if nid == root_id else 'box'
        lines.append(f'  "{nid}" [label="{label}…" shape={shape}];')

    for edge in bas["edges"]:
        src   = edge["source"]
        tgt   = edge["target"]
        rel   = edge["relation"]
        synth = edge.get("synthetic", False)
        color = "green4" if rel == "support" else "red3"
        style = "dashed" if synth else "solid"
        lines.append(
            f'  "{src}" -> "{tgt}" [label="{rel}" color={color} style={style}];'
        )

    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("DOT file written to %s", path)


# 7 · Main assembly pipeline
def assemble_bas(
    reasoning_output: dict,
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Full BAS assembly pipeline — produces both modes in a single pass.

    P_central is taken verbatim from Step 1's `central_proposition` field;
    no root identification is performed here.

    Args:
        reasoning_output: Output dict from llm_reasoner.py (contains thread_id,
                          title, conversation, central_proposition, and
                          per-EDU reasoning results).

    Returns (repair_bas, no_repair_bas).
      repair_bas    — always produced when at least one argumentative node
                      exists (the full dataset is preserved in repair mode).
      no_repair_bas — None when its central component is too small (≤ 3 units).
    """
    conv_id    = reasoning_output.get("conv_id", "unknown")
    thread_id  = reasoning_output.get("thread_id", "")
    is_delta   = reasoning_output.get("is_delta")
    # central_proposition is the verbatim string identified in Step 1
    central_proposition_text = reasoning_output.get("central_proposition")
    log.info("─── BAS Assembly thread_id=%s ──", thread_id)

    # ── Step 1: extract argumentative nodes and non-neutral edges ─────────────
    nodes, raw_edges = extract_nodes_and_edges(reasoning_output)

    # ── Guarantee the CP node is always in the node set ───────────────────────
    # The central proposition must participate in the graph by definition.
    # If it was not extracted as argumentative (e.g. all its PACs were neutral
    # even after the Step 3 review call), force-promote it from the conversation
    # so that P_central identification and connectivity never fail silently.
    if central_proposition_text and nodes is not None:
        cp_text   = central_proposition_text.strip()
        cp_in_set = any(n["text"].strip() == cp_text for n in nodes.values())
        if not cp_in_set:
            for turn in reasoning_output.get("conversation", []):
                if turn.get("deleted"):
                    continue
                turn_speaker = turn.get("speaker_id", "")
                for edu in turn.get("edus", []):
                    if not isinstance(edu, dict):
                        continue
                    if edu.get("text", "").strip() == cp_text:
                        g_idx   = edu["global_idx"]
                        node_id = f"edu_{g_idx}"
                        nodes[node_id] = {
                            "id":             node_id,
                            "global_idx":     g_idx,
                            "post_id":        turn.get("post_id", ""),
                            "parent_post_id": turn.get("parent_post_id", ""),
                            "speaker_id":     edu.get("speaker_id") or turn_speaker,
                            "text":           edu.get("text", ""),
                            "is_root":        False,
                            "in_degree":      0,
                            "out_degree":     0,
                        }
                        log.info(
                            "Force-promoted CP node %s into argumentative set "
                            "(was not extracted as argumentative by Step 3)",
                            node_id,
                        )
                        break
                if cp_in_set or any(n["text"].strip() == cp_text for n in nodes.values()):
                    break

    if not nodes:
        log.warning("thread_id=%s — no argumentative nodes found, skipping", thread_id)
        return None, None

    # ── Step 2: promote source EDUs and deduplicate/reorient edges ───────────
    nodes, edges = promote_source_nodes(nodes, raw_edges, reasoning_output)

    # ── Step 3: first-pass degree annotation ──────────────────────────────────
    annotate_degrees(nodes, edges)

    # ── Step 4: designate P_central from Step 1 ──────────────────────────────
    # Root identification is performed in Step 1 (central_proposition).
    # Here we only match the verbatim CP text against the node set — the CP
    # node is guaranteed present (force-promoted above if needed).
    root_id = None
    if central_proposition_text:
        cp_text = central_proposition_text.strip()
        for nid, node in nodes.items():
            if node["text"].strip() == cp_text:
                root_id = nid
                log.info("P_central matched from Step 1 central_proposition: %s  "
                         "text=%.80s", root_id, node["text"])
                break

    if root_id is None:
        # Step 1 CP missing or unmatched — minimal positional fallback only.
        root_id = min(nodes.keys(), key=lambda nid: nodes[nid]["global_idx"])
        log.warning(
            "thread_id=%s — central_proposition from Step 1 missing/unmatched; "
            "falling back to earliest argumentative EDU as P_central: %s",
            thread_id, root_id,
        )

    if root_id and root_id in nodes:
        nodes[root_id]["is_root"] = True

    # ══ REPAIR mode ═══════════════════════════════════════════════════════════
    # The entire dataset is preserved in repair mode — no minimum-size filter.
    repair_edges = repair_connectivity(nodes, edges)
    annotate_degrees(nodes, repair_edges)
    repair_summary = compute_summary(nodes, repair_edges, root_id)

    if is_too_small(nodes):
        log.info("thread_id=%s — only %d argumentative units (≤ %d); "
                 "kept in repair mode (full dataset preserved)",
                 thread_id, len(nodes), MIN_ARGUMENTATIVE_UNITS)

    repair_bas = {
        "conv_id":   conv_id,
        "thread_id": thread_id,
        "is_delta":  is_delta,
        "mode":      "repair",
        "nodes":     list(nodes.values()),
        "edges":     repair_edges,
        "summary":   repair_summary,
    }
    log.info(
        "repair BAS — %d nodes  %d edges  (support=%d  attack=%d  synthetic=%d)  root=%s",
        repair_summary["total_nodes"], repair_summary["total_edges"],
        repair_summary["support_edges"], repair_summary["attack_edges"],
        repair_summary["synthetic_edges"], root_id,
    )

    # ══ NO-REPAIR mode ════════════════════════════════════════════════════════
    # Use original (non-synthetic) edges and extract only P_central's component
    nr_nodes, nr_edges = extract_central_subgraph(
        {nid: dict(n) for nid, n in nodes.items()},  # shallow copy to avoid mutation
        [e for e in edges if not e.get("synthetic")],
        root_id,
    )
    annotate_degrees(nr_nodes, nr_edges)
    nr_summary = compute_summary(nr_nodes, nr_edges, root_id)

    if is_too_small(nr_nodes):
        log.info(
            "thread_id=%s — no-repair central component has only %d units (≤ %d), "
            "dropping no-repair output only (repair mode preserves the full dataset)",
            thread_id, len(nr_nodes), MIN_ARGUMENTATIVE_UNITS,
        )
        # only return repaired version
        return repair_bas, None
    no_repair_bas = {
        "conv_id":   conv_id,
        "thread_id": thread_id,
        "is_delta":  is_delta,
        "mode":      "no_repair",
        "nodes":     list(nr_nodes.values()),
        "edges":     nr_edges,
        "summary":   nr_summary,
    }
    log.info(
        "no-repair BAS — %d nodes  %d edges  (support=%d  attack=%d)  root=%s",
        nr_summary["total_nodes"], nr_summary["total_edges"],
        nr_summary["support_edges"], nr_summary["attack_edges"], root_id,
    )

    return repair_bas, no_repair_bas


def run_on_jsonl(
    input_path:        Path,
    output_repair:     Path,
    output_no_repair:  Path,
) -> None:
    total = count_lines(input_path)
    log.info("BAS assembly starting — %d conversations", total)

    skipped_empty   = 0   # conversations with no argumentative nodes at all
    written_repair  = 0
    written_no_rep  = 0

    with JSONLWriter(output_repair) as wr_repair, \
         JSONLWriter(output_no_repair) as wr_no_rep:
        for line_no, conv in read_jsonl(input_path):
            log_progress(line_no, total, conv.get("thread_id", ""), "BAS", log)
            repair_bas, no_repair_bas = assemble_bas(conv)

            if repair_bas is None:
                skipped_empty += 1
                continue

            wr_repair.write(repair_bas)
            written_repair += 1

            if no_repair_bas is not None:
                wr_no_rep.write(no_repair_bas)
                written_no_rep += 1

    log.info(
        "Finished.  repair→%s (%d written)  no_repair→%s (%d written)  "
        "skipped (no argumentative nodes)=%d",
        output_repair, written_repair,
        output_no_repair, written_no_rep,
        skipped_empty,
    )


# 9 · CLI
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble Bipolar Argument Structures "
                    "from LLM reasoner output"
    )
    parser.add_argument("--input",            "-i", help="Path to llm_reasoner JSONL output")
    parser.add_argument("--output-repair",    "-r", help="Path to save repair-mode BAS JSONL",
                        default="bas_repair.jsonl")
    parser.add_argument("--output-no-repair", "-n", help="Path to save no-repair-mode BAS JSONL",
                        default="bas_no_repair.jsonl")
    parser.add_argument("--configs", "-c", default=None,
                        help="Unified use-case profile JSON (see pipeline_config.py). "
                             "Reads Step-4 settings from the profile's 'bas' section: "
                             "'embed_model' (connectivity-repair encoder) and "
                             "'min_argumentative_units'. e.g. configs/social_media.json "
                             "or configs/italian_interrogations.json. If omitted, the "
                             "built-in default is used."
                        )
    parser.add_argument("--export-dot",       "-d", help="Path to export GraphViz DOT file")
    parser.add_argument("--verbose",          "-v", action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--log-file", "-l", default="bas_assembler.log",
                        help="Log file path (default: bas_assembler.log)"
                        )
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)
    try:
        cfg = load_config(args.configs, DEFAULT_CONFIG)
    except ValueError as exc:
        parser.error(str(exc))
    apply_config(cfg)
    log.info("configs=%s  embed_model=%s  min_argumentative_units=%d",
             (args.configs or "builtin_default"), EMBED_MODEL, MIN_ARGUMENTATIVE_UNITS
             )
    if args.input:
        run_on_jsonl(
            Path(args.input),
            Path(args.output_repair),
            Path(args.output_no_repair),
            )

    else:
        log.info("No --input — chaining from built-in sample")
        import extract_edu as ee
        import pac_selector as ps
        import llm_reasoner as lr
        with JSONLWriter(Path(args.output_repair)) as wr_repair, \
             JSONLWriter(Path(args.output_no_repair)) as wr_no_rep:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS), conv.get("thread_id", ""), "BAS", log)
                repair_bas, no_repair_bas = assemble_bas(
                    lr.run_reasoning(ps.select_all_pacs(ee.extract_edus(conv))),
                )
                if repair_bas:
                    wr_repair.write(repair_bas)
                if no_repair_bas:
                    wr_no_rep.write(no_repair_bas)


if __name__ == "__main__":
    main()

"""
PAC Selector — Probable Argument Candidate Selector

For every target EDU, this module selects its probable argumentative candidates based on contextual, semantic and implicit characteristics.
"""
import logging
import argparse
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from sys_utils import (
    read_jsonl, JSONLWriter, log_progress, count_lines, SAMPLE_CONVERSATIONS,
    setup_logging, load_config, DEFAULT_CONFIG
    )

# Logging
log = logging.getLogger("pac_selector")

# Default Initialization / can be overridden by --configs
DEFAULT_MODEL = "all-mpnet-base-v2"
DEFAULT_K = 25
DEFAULT_THRESHOLD = 0.45
DEFAULT_WINDOW = 10

_ENCODER = None


def get_encoder(model_name: str):
    global _ENCODER
    if _ENCODER is None:
        log.info("Loading sentence-transformer: %s", model_name)
        _ENCODER = SentenceTransformer(model_name)
    return _ENCODER


# EDU index
def build_edu_index(
        conversation: list[dict],
        central_proposition: str | None = None,
        ) -> list[dict]:
    """
    Flatten all EDUs from all (non-deleted) turns into a global ordered list.
    Each entry carries:
        global_idx          – position in the full document (0-based)
        post_id             – turn this EDU belongs to
        parent_post_id      – parent post of this turn (empty string for root)
        speaker_id          – speaker
        local_idx           – position within the turn (0-based)
        text                – the EDU text string
        is_central_prop     – True if this EDU is the central proposition
    """
    index: list[dict] = []
    for turn in conversation:
        if turn.get("deleted"):
            continue
        post_id = turn.get("post_id", "")
        # Data carries "parent_id" (samples_test.jsonl, interrogazioni);
        # "parent_post_id" kept as a legacy fallback.
        parent_post_id = turn.get("parent_id", turn.get("parent_post_id", ""))
        speaker_id = turn.get("speaker_id", "")
        for local_idx, edu in enumerate(turn.get("edus", [])):
            # EDUs are dicts {"text": ..., "speaker_id": ...} from Step 1
            edu_text = edu["text"] if isinstance(edu, dict) else edu
            is_cp = (
                    central_proposition is not None
                    and edu_text.strip() == central_proposition.strip()
            )
            index.append({
                "global_idx": int(len(index)),
                "post_id": post_id,
                "parent_post_id": parent_post_id,
                "speaker_id": speaker_id,
                "local_idx": int(local_idx),
                "text": edu_text,
                "is_central_prop": is_cp,
                }
                )
    return index


# Discussion graph
def build_discussion_graph(conversation: list[dict]) -> dict[str, list[str]]:
    """
    Build a mapping of post_id → list of direct child post_ids from the
    conversation turn list.

    Only non-deleted turns are included. The root post (no parent) maps to
    an empty list by default and appears as a key so it is always present.
    """
    children: dict[str, list[str]] = {}
    for turn in conversation:
        if turn.get("deleted"):
            continue
        post_id = turn.get("post_id", "")
        parent_post_id = turn.get("parent_id", turn.get("parent_post_id", ""))
        # Ensure every post appears as a key
        if post_id not in children:
            children[post_id] = []
        if parent_post_id:
            children.setdefault(parent_post_id, [])
            if post_id not in children[parent_post_id]:
                children[parent_post_id].append(post_id)
    return children


def embed_edus(edu_index: list[dict], model_name: str) -> np.ndarray:
    """
    Encode all EDU texts and return a (N, D) float32 matrix.
    """
    encoder = get_encoder(model_name)
    texts = [e["text"] for e in edu_index]
    log.info("Encoding %d EDUs…", len(texts))
    t0 = time.time()
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        )
    log.info("Encoding done in %.2fs  shape=%s", time.time() - t0, embeddings.shape)
    return embeddings.astype(np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Returns (N, N) cosine similarity matrix (dot product between embeddings)
    """
    return embeddings @ embeddings.T  # shape (N, N)


def select_pacs_for_edu(
        target_idx: int,
        sim_row: np.ndarray,  # cosine similarities of target against all EDUs
        edu_index: list[dict],
        children_map: dict[str, list[str]],  # post_id → [child post_ids]
        k: int,
        threshold: float,
        implicit_window: int,
        ) -> list[dict]:
    """
    Select PACs for a single target EDU under localised scoping rules.

    Scope rules
    -----------
    INTRA  — candidates from the target's own post.
               No directionality restriction: a source may appear before or
               after the target within the same post.
    INTER  — candidates from direct children posts of the target's post only.
               Directionality preserved: source global_idx > target global_idx.
    Excluded — EDUs from unrelated (parallel) branches and the central
               proposition (is_central_prop=True) are never candidates.

    Implicit EDU fallback
    ---------------------
    If the best semantic similarity across all INTRA+INTER candidates falls
    below `threshold`, the EDU is treated as implicit. In this case semantic
    selection is skipped entirely and up to `implicit_window` surrounding EDUs
    from the same post (excluding the target itself) are returned as
    "contextual" PACs. Direct children are not considered for implicit EDUs.

    Returns a list of PAC dicts, each with:
        source_global_idx  – global index of the source EDU
        source_post_id     – post the source belongs to
        source_speaker_id  – speaker of the source
        source_local_idx   – local index within that post
        source_text        – source EDU text
        cosine_sim         – similarity score
        selection_type     – "intra_semantic" | "inter_semantic" | "contextual"
        pac_scope          – "intra" | "inter" | "implicit"
    """
    target = edu_index[target_idx]
    target_post = target["post_id"]
    child_posts = set(children_map.get(target_post, []))

    # ── Build candidate pools ─────────────────────────────────────────────────
    intra_indices: list[int] = []  # same post, no directionality restriction
    inter_indices: list[int] = []  # direct children posts, source after target

    for i, entry in enumerate(edu_index):
        if i == target_idx:
            continue
        if entry.get("is_central_prop"):
            continue  # Pcentral is never a source candidate
        post = entry["post_id"]
        # consider all EDUs emerging from the same post : INTRA PAC
        if post == target_post:
            intra_indices.append(i)
        # consider only EDUs from the children posts emerging from the target EDU: INTER PAC
        elif post in child_posts and i > target_idx:
            inter_indices.append(i)
        # any other post is a parallel thread — excluded

    # Implicit EDU check
    # if there is any EDU candidate that has good similarity above threshold, then target is likely
    # explicit, otherwise implicit.
    all_candidate_indices = intra_indices + inter_indices
    best_sim = (
        float(np.max(sim_row[all_candidate_indices]))
        if all_candidate_indices else 0.0
    )

    if best_sim < threshold:
        # no semantically-similar candidate has a good similarity score with target EDU
        # EDU is implicit — return surrounding intra-post window only
        intra_set = set(intra_indices)
        window_pacs: list[dict] = []

        # Take proximal EDUs surrounding the target EDU
        # Walk outward from target: alternating before/after within the post
        before = [i for i in intra_indices if i < target_idx][-implicit_window:]
        after = [i for i in intra_indices if i > target_idx][:implicit_window]

        # Merge preserving document order, capped at implicit_window total
        candidates = sorted(set(before + after))[:implicit_window]

        for src_idx in candidates:
            e = edu_index[src_idx]
            window_pacs.append({
                "source_global_idx": int(src_idx),
                "source_post_id": e["post_id"],
                "source_speaker_id": e["speaker_id"],
                "source_local_idx": int(e["local_idx"]),
                "source_text": e["text"],
                "cosine_sim": float(round(float(sim_row[src_idx]), 4)),
                "selection_type": "contextual",
                "pac_scope": "implicit",
                }
                )
        log.debug(
            "  EDU[%03d] implicit (best_sim=%.3f) → %d contextual PACs",
            target_idx, best_sim, len(window_pacs),
            )
        return window_pacs

    # ── Normal semantic selection ─────────────────────────────────────────────
    def top_semantic(candidate_pool: list[int], label: str) -> list[dict]:
        """Sort pool by similarity descending, apply threshold, return top-k."""
        ranked = sorted(
            [i for i in candidate_pool if sim_row[i] >= threshold],
            key=lambda i: sim_row[i],
            reverse=True,
            )[:k]
        return [
            {
                "source_global_idx": int(src_idx),
                "source_post_id": edu_index[src_idx]["post_id"],
                "source_speaker_id": edu_index[src_idx]["speaker_id"],
                "source_local_idx": int(edu_index[src_idx]["local_idx"]),
                "source_text": edu_index[src_idx]["text"],
                "cosine_sim": float(round(float(sim_row[src_idx]), 4)),
                "selection_type": f"{label}_semantic",
                "pac_scope": label,
                }
            for src_idx in ranked
            ]

    intra_pacs = top_semantic(intra_indices, "intra")
    inter_pacs = top_semantic(inter_indices, "inter")
    pacs = intra_pacs + inter_pacs

    log.debug(
        "  EDU[%03d] %-48s → %d PACs (%d intra / %d inter)",
        target_idx, target["text"][:46],
        len(pacs), len(intra_pacs), len(inter_pacs),
        )
    return pacs


# Main PAC selection pass
def select_all_pacs(
        edu_output: dict,
        model_name: str = DEFAULT_MODEL,
        k: int = DEFAULT_K,
        threshold: float = DEFAULT_THRESHOLD,
        implicit_window: int = DEFAULT_WINDOW,
        ) -> dict:
    """
    Build EDU index → discussion graph → embed → similarity matrix →
    localised PAC selection per EDU.

    Returns an enriched version of edu_output with 'pacs' attached to each
    EDU and a top-level 'pac_summary'.
    """
    conversation = edu_output.get("conversation", [])
    central_proposition = edu_output.get("central_proposition")
    thread_id = edu_output.get("thread_id", "?")

    edu_index = build_edu_index(conversation, central_proposition)
    children_map = build_discussion_graph(conversation)
    N = len(edu_index)

    if N == 0:
        log.warning("thread_id=%s  No EDUs found — returning input unchanged", thread_id)
        return edu_output

    log.info("thread_id=%s  EDUs=%d  posts=%d", thread_id, N, len(children_map))

    # Embed all EDUs
    embeddings = embed_edus(edu_index, model_name)

    # Full (N×N) cosine similarity matrix
    sim_matrix = cosine_similarity_matrix(embeddings)

    log.info(
        "Selecting PACs  (k=%d  threshold=%.2f implicit_window=%d)…",
        k, threshold, implicit_window,
        )

    total_pacs, pac_counts, enriched_flat = 0, [], []

    for g_idx, edu_entry in enumerate(edu_index):
        pacs = select_pacs_for_edu(
            target_idx=g_idx,
            sim_row=sim_matrix[g_idx],
            edu_index=edu_index,
            children_map=children_map,
            k=k,
            threshold=threshold,
            implicit_window=implicit_window,
            )
        total_pacs += len(pacs)
        pac_counts.append(len(pacs))
        enriched_flat.append({**edu_entry, "pacs": pacs, "pac_count": len(pacs)})

    avg_pacs = total_pacs / N if N else 0
    log.info(
        "PAC selection done — %d total PACs  avg=%.1f/EDU  min=%d  max=%d",
        total_pacs, avg_pacs, min(pac_counts), max(pac_counts),
        )

    # ── Re-embed PAC lists back into the conversation structure ───────────────
    pac_map: dict[tuple, list] = {
        (e["post_id"], e["local_idx"]): e["pacs"]
        for e in enriched_flat
        }
    pac_count_map: dict[tuple, int] = {
        (e["post_id"], e["local_idx"]): e["pac_count"]
        for e in enriched_flat
        }
    g_idx_map: dict[tuple, int] = {
        (e["post_id"], e["local_idx"]): int(e["global_idx"])
        for e in enriched_flat
        }

    enriched_turns = []
    for turn in conversation:
        if turn.get("deleted"):
            enriched_turns.append(turn)
            continue
        post_id = turn.get("post_id", "")
        enriched_edus = []
        for li, edu in enumerate(turn.get("edus", [])):
            # EDUs are dicts {"text": ..., "speaker_id": ...} from Step 1
            edu_text = edu["text"] if isinstance(edu, dict) else edu
            key = (post_id, li)
            enriched_edus.append({
                **(edu if isinstance(edu, dict) else {"text": edu_text}),
                "global_idx": g_idx_map.get(key, 0),
                "pacs": pac_map.get(key, []),
                "pac_count": pac_count_map.get(key, 0),
                }
                )
        enriched_turns.append({**turn, "edus": enriched_edus})

    return {
        **edu_output,
        "conversation": enriched_turns,
        "pac_summary": {
            "model": model_name,
            "k": k,
            "threshold": threshold,
            "implicit_window": implicit_window,
            "total_edus": N,
            "total_pacs": total_pacs,
            "avg_pacs_per_edu": round(avg_pacs, 2),
            "min_pacs": int(min(pac_counts)),
            "max_pacs": int(max(pac_counts)),
            },
        }


# JSONL write
def run_on_jsonl(
        input_path: Path,
        output_path: Path,
        model_name: str = DEFAULT_MODEL,
        k: int = DEFAULT_K,
        threshold: float = DEFAULT_THRESHOLD,
        implicit_window: int = DEFAULT_WINDOW,
        ) -> None:
    # Preload the encoder once before the loop
    get_encoder(model_name)
    total = count_lines(input_path)
    log.info("PAC selection starting — %d conversations", total)

    with JSONLWriter(output_path) as writer:
        for line_no, conv in read_jsonl(input_path):
            log_progress(line_no, total, conv.get("thread_id", ""), "PAC", log)
            result = select_all_pacs(conv, model_name, k, threshold, implicit_window)
            writer.write(result)

    log.info("Finished. Output → %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select Probable Argumentative Candidates (PACs) for each EDU"
        )
    parser.add_argument("--input", "-i", help="Path to EDU extractor JSON output")
    parser.add_argument("--output", "-o", help="Path to save enriched JSON output")
    parser.add_argument("--configs", "-c", default=None,
                        help="Use-case profile JSON. Its 'pac' section supplies "
                             "the defaults for --model/--k/--threshold/--window "
                             "(e.g. configs/social_media.json or "
                             "configs/italian_interrogations.json). Any explicit "
                             "flag below overrides the profile. If omitted, the "
                             "built-in social-media default is used."
                        )
    parser.add_argument("--model", "-m", default=None,
                        help=f"Sentence-transformer model (default: None)"
                        )
    parser.add_argument("--k", type=int, default=None,
                        help=f"kNN neighbourhood size (default: None)"
                        )
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Cosine similarity threshold (default: None)"
                        )
    parser.add_argument("--window", type=int, default=None,
                        help=(
                            f"Implicit EDU surrounding window size (default: None). "
                            "Used when an EDU produces no semantic matches above threshold — "
                            "selects this many surrounding intra-post EDUs instead."
                        )
                        )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging"
                        )
    parser.add_argument("--log-file", "-l", default="edu_extractor.log",
                        help="Log file path (default: edu_extractor.log)"
                        )
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)
    # Load the configs.
    try:
        cfg = load_config(args.configs, DEFAULT_CONFIG)
    except ValueError as exc:
        parser.error(str(exc))

    # Resolve each knob: explicit CLI flag > profile value > built-in default.
    pac = cfg.get("pac", {})
    model = args.model if args.model is not None else pac.get("model", DEFAULT_MODEL)
    k = args.k if args.k is not None else pac.get("k", DEFAULT_K)
    threshold = args.threshold if args.threshold is not None else pac.get("threshold", DEFAULT_THRESHOLD)
    window = args.window if args.window is not None else pac.get("implicit_window", DEFAULT_WINDOW)

    log.info(
        "Config=%s  model=%s  k=%d  threshold=%.2f  implicit_window=%d",
        (args.configs or "builtin_default"), model, k, threshold, window,
        )

    if args.input:
        run_on_jsonl(Path(args.input), Path(args.output),
                     model, k, threshold, window
                     )
    else:
        # Default to sample conversation
        log.info("No --input — using built-in sample")
        # Import edu_extractor to generate EDU output on the fly
        import extract_edu as ee
        get_encoder(args.model)
        with JSONLWriter(Path(args.output)) as writer:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS), conv.get("thread_id", ""), "PAC", log)
                edu_conv = ee.extract_edus(conv)
                writer.write(select_all_pacs(edu_conv, args.model,
                                             args.k, args.threshold, args.window
                                             )
                             )


if __name__ == "__main__":
    main()

"""
LLM Reasoner — Iterative argumentative relation classification (Step 3)

For every (target EDU, PAC source EDU) pair this module determines:
  (i)  Is the target EDU argumentative?
  (ii) Does the source EDU SUPPORT, ATTACK, or is NEUTRAL toward the target?
"""
import json
import re
import time
import logging
import argparse
from pathlib import Path
from typing import Optional
from collections import Counter

import requests

from sys_utils import (
read_jsonl, log_progress, JSONLWriter, count_lines, SAMPLE_CONVERSATIONS,
setup_logging, load_config, DEFAULT_CONFIG
    )

# Logging
log = logging.getLogger("llm_reasoner")

# ─── default Initialization / overwritten by configs file

OLLAMA_URL      = "http://localhost:11434/api/chat"
MODEL           = "qwen3.6:27b"
TIMEOUT         = 500
MAX_RETRIES     = 2
VALID_RELATIONS      = {"support", "attack", "neutral"}
CONFIDENCE_THRESHOLD_SUPPORT = 0.45   # support below/at this → neutral
CONFIDENCE_THRESHOLD_ATTACK  = 0.35   # attack is relaxed a bit since it is harder to identify
ENABLE_THINKING = True
THINKING_BUDGET: Optional[str] = None

# ── Context window and batch size heuristic ───────────────────────────────────
CTX_WINDOW = 16_384

# BATCH_SIZE is kept at 10
_REASONER_SYSTEM_OVERHEAD  = 700
_REASONER_RESPONSE_FRAC    = 0.50   # 75% of ctx reserved for output (thinking + JSON)

# Per-batch-item cost estimates (conservative):
TOKENS_PER_TARGET    = 150
TOKENS_PER_SOURCE    = 25
AVG_SOURCES_PER_EDU  = 8
_CHARS_PER_TOKEN     = 4   # Italian tokenizes denser than English (was 4)

# Fixed batch size — number of target EDUs per LLM call.
BATCH_SIZE = 10


def _build_batches(all_items: list[dict], batch_size: int = BATCH_SIZE) -> list[list[dict]]:
    """Split all_items into fixed-size batches of batch_size"""
    return [all_items[i: i + batch_size]
            for i in range(0, len(all_items), batch_size)]


# ─── Prompts ──────────────────────────────────────────────────────────────────
REASONING: dict = DEFAULT_CONFIG.get("reasoning", {})


def build_user_prompt(
    target_text:   str,
    context_before: list[str],
    context_after:  list[str],
    source_edus:   list[str],
) -> str:
    """Single-target prompt — used only for tie-breaking on disputed pairs."""
    ctx_before_str = (
        "\n".join(f"  [{i}] {t}" for i, t in enumerate(context_before))
        if context_before else "  (none)"
    )
    ctx_after_str = (
        "\n".join(f"  [{i}] {t}" for i, t in enumerate(context_after))
        if context_after else "  (none)"
    )
    sources_str = "\n".join(
        f"  [{i}] {text}" for i, text in enumerate(source_edus)
    )
    return f"""=== TARGET EDU [0] ===
{target_text}

=== LOCAL CONTEXT (before target) ===
{ctx_before_str}

=== LOCAL CONTEXT (after target) ===
{ctx_after_str}

=== SOURCE EDUs (candidates to classify with respect to the target) ===
{sources_str}

Classify each SOURCE EDU's relation to the TARGET EDU.
"""


def build_batch_prompt(batch_items: list[dict]) -> str:
    """
    Build a single prompt covering a batch of target EDUs.

    Each batch_item contains:
        batch_position  : int         (0-based index within this batch)
        target_text     : str
        context_before  : list[str]
        context_after   : list[str]
        source_texts    : list[str]
    """
    sections = []
    for item in batch_items:
        pos       = item["batch_position"]
        ctx_b_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["context_before"]))
            if item["context_before"] else "  (none)"
        )
        ctx_a_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["context_after"]))
            if item["context_after"] else "  (none)"
        )
        sources_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["source_texts"]))
            if item["source_texts"] else "  (none)"
        )
        sections.append(
            f"=== TARGET EDU [{pos}] ===\n{item['target_text']}\n\n"
            f"=== LOCAL CONTEXT (before target [{pos}]) ===\n{ctx_b_str}\n\n"
            f"=== LOCAL CONTEXT (after target [{pos}]) ===\n{ctx_a_str}\n\n"
            f"=== SOURCE EDUs for target [{pos}] ===\n{sources_str}"
        )

    divider = "\n\n" + "─" * 40 + "\n\n"
    return divider.join(sections) + "\n\nClassify all target EDUs and their source candidates as specified."


def build_tiebreak_prompt(tiebreak_items: list[dict], reason: str) -> str:
    """
    Build a tie-breaking prompt that tells the model why it is being invoked:
      reason = "disagreement"  — two calls produced conflicting labels
      reason = "failed"        — both calls failed to produce a valid response
    """
    if reason == "disagreement":
        preamble = (
            "Two classifications of the following EDUs disagreed. "
            "Output your verdict as JSON immediately — no explanation, no preamble."
        )
    else:
        preamble = (
            "Previous classification attempts failed to produce a valid response. "
            "Output your verdict as JSON immediately — no explanation, no preamble."
        )

    sections = []
    for item in tiebreak_items:
        pos       = item["batch_position"]
        ctx_b_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["context_before"]))
            if item["context_before"] else "  (none)"
        )
        ctx_a_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["context_after"]))
            if item["context_after"] else "  (none)"
        )
        sources_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["source_texts"]))
            if item["source_texts"] else "  (none)"
        )
        sections.append(
            f"=== TARGET EDU [{pos}] ===\n{item['target_text']}\n\n"
            f"=== LOCAL CONTEXT (before target [{pos}]) ===\n{ctx_b_str}\n\n"
            f"=== LOCAL CONTEXT (after target [{pos}]) ===\n{ctx_a_str}\n\n"
            f"=== SOURCE EDUs for target [{pos}] ===\n{sources_str}"
        )

    divider = "\n\n" + "─" * 40 + "\n\n"
    body    = divider.join(sections)
    return f"{preamble}\n\n{body}\n\nClassify all target EDUs and their source candidates."


# Ollama client
def call_ollama(
    user_prompt:   str,
    temperature:   float,
    retries:       int = MAX_RETRIES,
    system_prompt: str = None,
) -> Optional[str]:
    """Single inference call; strips Qwen3 <think> block.
    Retries on timeout AND on empty response body.
    num_predict mirrors _REASONER_RESPONSE_FRAC so input/output budgets
    are consistent with the batch size heuristic."""

    num_predict = int(CTX_WINDOW * _REASONER_RESPONSE_FRAC)

    options: dict = {"temperature": temperature, "num_ctx": CTX_WINDOW,
                     "num_predict": num_predict}

    payload: dict = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt or REASONING["system"]},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": options,
    }

    think_val = (THINKING_BUDGET if THINKING_BUDGET else "medium") if "gpt" in MODEL.lower() \
                else (ENABLE_THINKING if "qwen3" in MODEL.lower() else None)

    for attempt in range(1, retries + 1):
        try:
            if think_val is not None:
                payload["think"] = think_val

            log.debug("Ollama call  temp=%.2f  attempt=%d/%d  think=%s",
                      temperature, attempt, retries, think_val)
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            msg  = resp.json()["message"]
            raw  = msg.get("content", "")
            if not raw:
                thinking_len = len(msg.get("thinking", "") or "")
                log.warning(
                    "Empty content on attempt %d/%d — msg keys: %s  thinking_len: %d",
                    attempt, retries, list(msg.keys()), thinking_len,
                )
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if not raw:
                log.warning("Empty body on attempt %d/%d — retrying", attempt, retries)
                time.sleep(2 ** attempt)
                continue
            return raw
        except requests.exceptions.ConnectionError:
            log.error("Cannot reach Ollama at %s — is it running?", OLLAMA_URL)
            return None
        except requests.exceptions.Timeout:
            log.warning("Timeout on attempt %d/%d", attempt, retries)
            if attempt == retries:
                return None
            time.sleep(2 ** attempt)
        except Exception as exc:
            log.error("Unexpected error: %s", exc)
            return None
    return None


def _apply_confidence_gate(rel: str, confidence: float) -> str:
    """
    Downgrade support/attack to neutral when the model's stated confidence
    is at or below the relation-specific threshold. The gate is asymmetric:
    ATTACK uses a lower threshold than SUPPORT so that harder-to-detect,
    indirect attacks are not silently suppressed. Neutral is always kept.
    """
    threshold = (CONFIDENCE_THRESHOLD_SUPPORT if rel == "support"
                 else CONFIDENCE_THRESHOLD_ATTACK)
    if rel in ("support", "attack") and confidence <= threshold:
        log.debug(
            "  Confidence gate: %s (conf=%.3f ≤ %.2f) → neutral",
            rel, confidence, threshold,
        )
        return "neutral"
    return rel


def parse_batch_response(raw: str, batch_items: list[dict]) -> Optional[list[Optional[dict]]]:
    """
    Parse a multi-target batch response into a list of per-target result dicts.
    Missing target indices are left as None — caller decides whether to retry.
    Returns None if parsing fails entirely (bad JSON, wrong structure).
    """
    if not raw or not raw.strip():
        log.warning("Empty batch response from model")
        return None

    log.debug("parse_batch_response raw (first 2000): %.2000s", raw)

    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Try direct parse first; if the model emitted prose before/after the JSON,
        # fall back to extracting the first JSON object or array from the response.
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'(\{[\s\S]*}|\[[\s\S]*])', cleaned)
            if not match:
                log.warning("Batch response parse failed: no JSON found  raw=%.80s", raw)
                return None
            log.warning(
                "Prose preamble in response — extracted JSON fragment  raw=%.80s", raw
                )
            data = json.loads(match.group(1))
        n_targets = len(batch_items)

        if isinstance(data, dict) and "batch" in data:
            batch_list = data["batch"]
        elif isinstance(data, list):
            batch_list = data
        else:
            log.warning("Unexpected batch response structure: %s", type(data))
            return None

        results: list[Optional[dict]] = [None] * n_targets

        for entry in batch_list:
            if not isinstance(entry, dict):
                log.warning("Skipping non-dict batch entry: %s", type(entry))
                continue
            t_idx = entry.get("target_idx")
            if not isinstance(t_idx, int) or not (0 <= t_idx < n_targets):
                continue
            n_srcs  = len(batch_items[t_idx]["source_texts"])
            rel_map: dict[int, str] = {}
            for rel_entry in entry.get("relations", []):
                if not isinstance(rel_entry, dict):
                    continue
                s_idx      = rel_entry.get("source_idx")
                rel        = str(rel_entry.get("relation", "neutral")).lower().strip()
                confidence = float(rel_entry.get("confidence", 1.0))
                if rel not in VALID_RELATIONS:
                    rel = "neutral"
                rel = _apply_confidence_gate(rel, confidence)
                if isinstance(s_idx, int) and 0 <= s_idx < n_srcs:
                    rel_map[s_idx] = rel
            results[t_idx] = {
                "relations": [rel_map.get(i, "neutral") for i in range(n_srcs)],
            }

        found_indices = [r is not None for r in results]
        n_found = sum(found_indices)
        if n_found < len(batch_items):
            log.warning(
                "parse_batch_response: found %d/%d targets. "
                "batch_positions sent: %s  found: %s",
                n_found, len(batch_items),
                [item["batch_position"] for item in batch_items],
                [i for i, r in enumerate(results) if r is not None],
            )
        return results

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning("Batch response parse failed: %s  raw=%.80s", exc, raw)
        return None


# ─── Self-consistency voting ──────────────────────────────────────────────────
def majority_vote(responses: list[dict], n_sources: int) -> dict:
    """Majority vote across relation labels for each source position."""
    relations = []
    for i in range(n_sources):
        votes = [r["relations"][i] for r in responses if i < len(r["relations"])]
        rel   = Counter(votes).most_common(1)[0][0] if votes else "neutral"
        relations.append(rel)
    return {"relations": relations}


# Context window builder
def get_intra_post_context(
    post_edu_texts: list[str],
    local_idx:      int,
    window:         int,
) -> tuple[list[str], list[str]]:
    """
    Return (edus_before, edus_after) within ±window positions of the target,
    restricted strictly to the target EDU's own post.

    post_edu_texts — ordered list of EDU texts from the target's post only.
    local_idx      — position of the target EDU within that post list.
    window         — number of surrounding EDUs to include on each side.
    """
    before = post_edu_texts[max(0, local_idx - window): local_idx]
    after  = post_edu_texts[local_idx + 1: local_idx + 1 + window]
    return before, after


# Per-EDU inference with cascaded self-consistency
def _disputed_batch_pairs(
    res1: list[dict],
    res2: list[dict],
    batch_items: list[dict],
) -> list[tuple[int, int]]:
    """
    Compare two batch responses and return list of (target_idx, source_idx)
    tuples where the relation label differs between the two calls.
    """
    pair_disputes: list[tuple[int, int]] = []
    for t_idx in range(len(batch_items)):
        # Skip if either result is missing for this target
        if res1[t_idx] is None or res2[t_idx] is None:
            continue
        n_srcs = len(batch_items[t_idx]["source_texts"])
        for s_idx in range(n_srcs):
            if res1[t_idx]["relations"][s_idx] != res2[t_idx]["relations"][s_idx]:
                pair_disputes.append((t_idx, s_idx))
    return pair_disputes


def _log_tiebreak_resolution(
        batch_item: dict,
        s_idx: int,
        label1: str,
        label2: str,
        resolved: str,
        method: str,
        ) -> None:
    """
    Record a disagreement between the two main calls and how the third call
    resolved it. Written to the log file with a fixed 'TIEBREAK_RESOLUTION'
    tag so the records can be grepped out for qualitative analysis, e.g.:

        grep TIEBREAK_RESOLUTION llm_reasoner.log

    method distinguishes genuine third-call resolutions ('tiebreak_call')
    from majority-vote fallbacks when the third call failed or dropped the
    target.
    """
    source_texts = batch_item.get("source_texts", [])
    source_text = source_texts[s_idx] if s_idx < len(source_texts) else ""
    log.info(
        "TIEBREAK_RESOLUTION  method=%s  target_gidx=%s  source_idx=%d  "
        "call1=%s  call2=%s  resolved=%s  |  target=%.120s  |  source=%.120s",
        method, batch_item.get("global_idx", "?"), s_idx,
        label1, label2, resolved,
        batch_item.get("target_text", "").replace("\n", " "),
        source_text.replace("\n", " "),
        )


def reason_for_edu_batch(
    batch_items:   list[dict],
    temp1:         float,
    temp2:         float,
    temp_tiebreak: float,
    system_prompt: str = None,
) -> list[dict]:
    """
    Run two inference calls on a batch of target EDUs with self-consistency.

    If some target indices are missing from the response (the model dropped them
    mid-generation), the missing items are split in half and each half is
    retried recursively. This continues up to depth 3 or until batch_size=1,
    at which point missing relations are defaulted to neutral.
    """

    def _recurse(batch_items: list[dict], depth: int) -> list[dict]:
        n_targets = len(batch_items)
        default   = [
            {"relations": ["neutral"] * len(item["source_texts"])}
            for item in batch_items
        ]

        if all(len(item["source_texts"]) == 0 for item in batch_items):
            return [{"relations": []} for _ in batch_items]

        prompt = build_batch_prompt(batch_items)

        raw1 = call_ollama(prompt, temperature=temp1, system_prompt=system_prompt)
        res1 = parse_batch_response(raw1, batch_items)

        raw2 = call_ollama(prompt, temperature=temp2, system_prompt=system_prompt)
        res2 = parse_batch_response(raw2, batch_items)

        valid = [r for r in [res1, res2] if r is not None]

        if len(valid) == 0:
            log.warning("Both calls failed — invoking third prompt as self-refinement")
            tb_prompt = build_tiebreak_prompt(
                [{**item, "batch_position": i} for i, item in enumerate(batch_items)],
                reason="failed",
            )
            raw3 = call_ollama(tb_prompt, temperature=temp_tiebreak,
                               system_prompt=REASONING["tiebreak"])
            res3 = parse_batch_response(raw3, batch_items)
            valid = [res3] if res3 is not None else []

        if len(valid) == 1:
            log.warning("Only one run has been successful. Preserving it.")
            merged = valid[0]
        elif len(valid) == 2:
            pair_disputes = _disputed_batch_pairs(valid[0], valid[1], batch_items)
            if not pair_disputes:
                # there are no disputes. 100% agreement
                merged = valid[0]
            else:
                log.info("  Pair-level disagreement — disputed_pairs=%d", len(pair_disputes))
                disputed_target_indices = sorted({t for t, _ in pair_disputes})
                tiebreak_items = [
                    {**batch_items[t], "batch_position": new_pos}
                    for new_pos, t in enumerate(disputed_target_indices)
                ]
                # rebuild prompt only over the disagreed target EDUs
                tb_prompt = build_tiebreak_prompt(tiebreak_items, reason="disagreement")
                raw3      = call_ollama(tb_prompt, temperature=temp_tiebreak,
                                        system_prompt=REASONING["tiebreak"])
                res3      = parse_batch_response(raw3, tiebreak_items)
                # add in disputed pairs. defaulted to neutral.
                merged = [{"relations": list(r["relations"])} if r is not None else
                          {"relations": ["neutral"] * len(batch_items[i]["source_texts"])}
                          for i, r in enumerate(valid[0])]
                # third prompt completed.
                if res3 is not None:
                    for new_pos, orig_t_idx in enumerate(disputed_target_indices):
                        if res3[new_pos] is None:
                            log.warning("Tiebreak missing target new_pos=%d orig_t_id=%d - falling back to"
                                        " majority vote for that target.", new_pos, orig_t_idx)
                            disputed_sources = [s for t, s in pair_disputes if t == orig_t_idx]
                            for s_idx in disputed_sources:
                                votes = [valid[0][orig_t_idx]["relations"][s_idx],
                                         valid[1][orig_t_idx]["relations"][s_idx]]
                                resolved = Counter(votes).most_common(1)[0][0]
                                merged[orig_t_idx]["relations"][s_idx] = resolved
                                _log_tiebreak_resolution(
                                    batch_items[orig_t_idx], s_idx,
                                    votes[0], votes[1], resolved,
                                    method="majority_vote(tiebreak_missing_target)",
                                    )
                            continue

                        disputed_sources = [s for t, s in pair_disputes if t == orig_t_idx]
                        for s_idx in disputed_sources:
                            if s_idx < len(res3[new_pos]["relations"]):
                                resolved = res3[new_pos]["relations"][s_idx]
                                merged[orig_t_idx]["relations"][s_idx] = resolved
                                _log_tiebreak_resolution(
                                    batch_items[orig_t_idx], s_idx,
                                    valid[0][orig_t_idx]["relations"][s_idx],
                                    valid[1][orig_t_idx]["relations"][s_idx],
                                    resolved,
                                    method="tiebreak_call",
                                    )
                else:
                    log.warning("Tie-break failed — majority vote for disputed pairs")
                    for t_idx, s_idx in pair_disputes:
                        votes = [valid[0][t_idx]["relations"][s_idx],
                                 valid[1][t_idx]["relations"][s_idx]]
                        resolved = Counter(votes).most_common(1)[0][0]
                        merged[t_idx]["relations"][s_idx] = resolved
                        _log_tiebreak_resolution(
                            batch_items[t_idx], s_idx,
                            votes[0], votes[1], resolved,
                            method="majority_vote(tiebreak_failed)",
                            )
        else:
            merged = None

        missing_indices = [i for i, r in enumerate(merged) if r is None] \
                          if merged is not None else list(range(n_targets))

        if not missing_indices:
            return merged  # type: ignore[return-value]

        log.warning(
            "  %d/%d target(s) missing — splitting and retrying (depth=%d)",
            len(missing_indices), n_targets, depth,
        )

        if depth >= 2 or n_targets == 1:
            log.error("  Max split depth reached — defaulting missing targets to neutral")
            if merged is None:
                return default
            for i in missing_indices:
                merged[i] = {"relations": ["neutral"] * len(batch_items[i]["source_texts"])}
            return merged  # type: ignore[return-value]

        mid        = max(1, len(missing_indices) // 2)
        half_a_idx = missing_indices[:mid]
        half_b_idx = missing_indices[mid:]
        result     = list(merged) if merged is not None else list(default)

        for half in [half_a_idx, half_b_idx]:
            if not half:
                continue
            sub_items = [{**batch_items[i], "batch_position": new_pos}
                         for new_pos, i in enumerate(half)]
            sub_results = _recurse(sub_items, depth + 1)
            for new_pos, orig_idx in enumerate(half):
                result[orig_idx] = sub_results[new_pos]

        return result

    return _recurse(batch_items, depth=0)


# ─── Central Proposition validation ──────────────────────────────────────────
def validate_central_proposition(
    pac_output:      dict,
    enriched_turns:  list[dict],
    post_edu_texts:  dict[str, list[str]],
    edu_location:    dict[int, tuple[str, int]],
    context_window:  int,
    temp1:           float,
    temp2:           float,
    temp_tiebreak:   float,
) -> list[dict]:
    """
    Post-reasoning validation for the Central Proposition (P_central).

    If the CP EDU has all-neutral PAC relations after the main batch loop,
    re-run it with PROMPTS["cp_review"] which instructs the model to
    scrutinise more carefully. The result patches the CP EDU in-place.
    """
    cp_text   = pac_output.get("central_proposition", "")
    thread_id = pac_output.get("thread_id", "?")

    if not cp_text:
        return enriched_turns

    cp_edu = cp_turn = cp_g_idx = None
    for turn in enriched_turns:
        if turn.get("deleted"):
            continue
        for edu in turn.get("edus", []):
            if not isinstance(edu, dict):
                continue
            if edu.get("text", "").strip() == cp_text.strip():
                cp_edu   = edu
                cp_turn  = turn
                cp_g_idx = edu.get("global_idx")
                break
        if cp_edu:
            break

    if cp_edu is None:
        log.warning("CP validation: central proposition EDU not found in enriched output")
        return enriched_turns

    pacs = cp_edu.get("pacs", [])
    if not pacs:
        return enriched_turns

    if any(p.get("relation", "neutral") in ("support", "attack") for p in pacs):
        return enriched_turns

    log.warning(
        "CP validation: thread_id=%s  CP EDU (global_idx=%s) has all-neutral PACs — "
        "re-running with PROMPTS['cp_review']", thread_id, cp_g_idx,
    )

    post_id, local_pos = edu_location.get(cp_g_idx, ("", 0))
    post_texts         = post_edu_texts.get(post_id, [cp_edu.get("text", "")])
    ctx_b, ctx_a       = get_intra_post_context(post_texts, local_pos, context_window)

    cp_batch_item = [{
        "batch_position": 0, "global_idx": cp_g_idx,
        "target_text":    cp_edu.get("text", ""),
        "context_before": ctx_b, "context_after": ctx_a,
        "source_texts":   [p["source_text"] for p in pacs],
        "pacs":           pacs,
    }]

    review_results = reason_for_edu_batch(
        batch_items   = cp_batch_item,
        temp1         = temp1, temp2 = temp2, temp_tiebreak = temp_tiebreak,
        system_prompt = REASONING["cp_review"],
    )

    if not review_results:
        log.warning("CP validation: review call returned no result — keeping original")
        return enriched_turns

    new_relations = review_results[0].get("relations", [])
    if not any(r in ("support", "attack") for r in new_relations):
        log.warning("CP validation: review also produced all-neutral — keeping original")
        return enriched_turns

    patched_pacs = [{**p, "relation": new_relations[i] if i < len(new_relations) else "neutral"}
                    for i, p in enumerate(pacs)]
    rel_counts   = Counter(p["relation"] for p in patched_pacs)
    patched_edu  = {
        **cp_edu, "pacs": patched_pacs,
        "target_is_argumentative": True,
        "relation_summary": dict(rel_counts),
        "cp_review_applied": True,
    }
    log.info("CP validation: patched  support=%d  attack=%d  neutral=%d",
             rel_counts.get("support", 0), rel_counts.get("attack", 0),
             rel_counts.get("neutral", 0))

    patched_turns = []
    for turn in enriched_turns:
        if turn is not cp_turn:
            patched_turns.append(turn)
            continue
        patched_turns.append({**turn, "edus": [
            patched_edu if (isinstance(e, dict) and e.get("global_idx") == cp_g_idx) else e
            for e in turn.get("edus", [])
        ]})
    return patched_turns


# ─── Main reasoning pass ──────────────────────────────────────────────────────

def run_reasoning(
    pac_output:     dict,
    ctx_window:     int   = CTX_WINDOW,
    context_window: int   = 2,
    temp1:          float = 0.1,
    temp2:          float = 0.2,
    temp_tiebreak:  float = 0.15,
    batch_size:     int   = BATCH_SIZE,
) -> dict:
    """
    Collect all EDUs across all turns, group them into fixed-size batches,
    and run one LLM call (with self-consistency) per batch.
    """
    conversation = pac_output.get("conversation", [])

    # ── Build per-post EDU text lists and a global_idx lookup ────────────────
    # post_edu_texts: post_id → ordered list of EDU texts within that post
    # edu_location:   global_idx → (post_id, local_idx_within_post)
    post_edu_texts: dict[str, list[str]] = {}
    edu_location:   dict[int, tuple[str, int]] = {}

    for turn in conversation:
        if turn.get("deleted"):
            continue
        post_id = turn.get("post_id", "")
        if post_id not in post_edu_texts:
            post_edu_texts[post_id] = []
        for edu in turn.get("edus", []):
            if not isinstance(edu, dict):
                continue
            g_idx     = edu.get("global_idx", 0)
            local_pos = len(post_edu_texts[post_id])
            post_edu_texts[post_id].append(
                edu.get("text", "") if isinstance(edu, dict) else str(edu)
            )
            edu_location[g_idx] = (post_id, local_pos)

    thread_id = pac_output.get("thread_id", "?")
    N         = sum(len(texts) for texts in post_edu_texts.values())
    log.info(
        "Starting LLM reasoning — thread_id:%s  %d EDUs  batch_size=%d  context_window=±%d",
        thread_id, N, batch_size, context_window,
    )

    # Counters for summary
    total_argumentative = 0
    total_support       = 0
    total_attack        = 0
    total_neutral       = 0
    total_pacs          = 0

    # ── Collect all EDU dicts in order across turns ───────────────────────────
    all_edu_dicts: list[dict] = []
    for turn in conversation:
        if turn.get("deleted"):
            continue
        for edu in turn.get("edus", []):
            if isinstance(edu, dict):
                all_edu_dicts.append(edu)

    # ── Process in EDU-level batches ──────────────────────────────────────────
    # results_map: global_idx → result dict {relations: [...]}
    results_map: dict[int, dict] = {}

    # Build all batch_items first, then group into prompt-size-fitting batches
    # using _build_batches. This produces the minimum number of batches — each
    # is filled as large as possible — and no item is ever dropped.
    all_items: list[dict] = []
    for pos, edu in enumerate(all_edu_dicts):
        g_idx       = edu.get("global_idx", 0)
        target_text = edu.get("text", "")
        pacs        = edu.get("pacs", [])

        post_id, local_pos = edu_location.get(g_idx, ("", 0))
        post_texts         = post_edu_texts.get(post_id, [target_text])
        ctx_b, ctx_a       = get_intra_post_context(post_texts, local_pos, context_window)

        all_items.append({
            "batch_position": pos,
            "global_idx":     g_idx,
            "target_text":    target_text,
            "context_before": ctx_b,
            "context_after":  ctx_a,
            "source_texts":   [p["source_text"] for p in pacs],
            "pacs":           pacs,
        })

    batches   = _build_batches(all_items, batch_size)
    n_batches = len(batches)
    log.info("  thread_id=%s  %d EDUs → %d batch(es)", thread_id, len(all_items), n_batches)

    for batch_no, batch_items in enumerate(batches, 1):
        # Re-index batch_position to 0-based within this batch so the prompt
        # labels targets 0..N-1 and the parser's range check always matches.
        batch_items = [{**item, "batch_position": local_pos}
                       for local_pos, item in enumerate(batch_items)]
        log.info(
            "  thread_id=%s  batch [%d/%d]  EDUs=%d",
            thread_id, batch_no, n_batches, len(batch_items),
        )

        t0            = time.time()
        batch_results = reason_for_edu_batch(
            batch_items   = batch_items,
            temp1         = temp1,
            temp2         = temp2,
            temp_tiebreak = temp_tiebreak,
        )
        elapsed = time.time() - t0
        log.info("  Batch [%d/%d] done in %.1fs", batch_no, n_batches, elapsed)

        for pos, result in enumerate(batch_results):
            g_idx = batch_items[pos]["global_idx"]
            results_map[g_idx] = result

    # ── Re-attach results back onto the conversation structure ─────────────────
    enriched_turns = []

    for turn in conversation:
        if turn.get("deleted"):
            enriched_turns.append(turn)
            continue

        enriched_edus = []

        for edu in turn.get("edus", []):
            if not isinstance(edu, dict):
                enriched_edus.append(edu)
                continue

            g_idx  = edu.get("global_idx", 0)
            pacs   = edu.get("pacs", [])
            result = results_map.get(g_idx)

            if result is None:
                enriched_edus.append({
                    **edu,
                    "target_is_argumentative": False,
                    "reasoning_skipped":       True,
                })
                continue

            relations = result["relations"]

            # Argumentativeness is inferred from relations — an EDU is
            # argumentative if any PAC relation is support or attack.
            is_arg = any(r in ("support", "attack") for r in relations)

            enriched_pacs = []
            for i, pac in enumerate(pacs):
                rel = relations[i] if i < len(relations) else "neutral"
                enriched_pacs.append({**pac, "relation": rel})

            rel_counts = Counter(p.get("relation", "neutral") for p in enriched_pacs)

            if is_arg:
                total_argumentative += 1
            total_support += rel_counts.get("support", 0)
            total_attack  += rel_counts.get("attack",  0)
            total_neutral += rel_counts.get("neutral", 0)
            total_pacs    += len(enriched_pacs)

            log.info(
                "  [EDU %03d] arg=%s  support=%d  attack=%d  neutral=%d",
                g_idx, is_arg,
                rel_counts.get("support", 0),
                rel_counts.get("attack",  0),
                rel_counts.get("neutral", 0),
            )

            enriched_edus.append({
                **edu,
                "pacs":                    enriched_pacs,
                "target_is_argumentative": is_arg,
                "relation_summary": {
                    "support": rel_counts.get("support", 0),
                    "attack":  rel_counts.get("attack",  0),
                    "neutral": rel_counts.get("neutral", 0),
                },
            })

        enriched_turns.append({**turn, "edus": enriched_edus})

    # ── P_central validation ──────────────────────────────────────────────────
    enriched_turns = validate_central_proposition(
        pac_output     = pac_output,
        enriched_turns = enriched_turns,
        post_edu_texts = post_edu_texts,
        edu_location   = edu_location,
        context_window = context_window,
        temp1          = temp1, temp2 = temp2, temp_tiebreak = temp_tiebreak,
    )

    reasoning_summary = {
        "total_edus":            N,
        "argumentative_edus":    total_argumentative,
        "non_argumentative":     N - total_argumentative,
        "total_pacs_classified": total_pacs,
        "support_relations":     total_support,
        "attack_relations":      total_attack,
        "neutral_relations":     total_neutral,
        "ctx_window":            ctx_window,
        "batch_size":            batch_size,
        "context_window":        context_window,
        "temp1":                 temp1,
        "temp2":                 temp2,
        "temp_tiebreak":         temp_tiebreak,
    }

    log.info(
        "Reasoning complete — thread_id=%s  %d/%d argumentative EDUs  "
        "support=%d  attack=%d  neutral=%d",
        thread_id, total_argumentative, N,
        total_support, total_attack, total_neutral,
    )

    return {
        **pac_output,
        "conversation":      enriched_turns,
        "reasoning_summary": reasoning_summary,
    }


def load_completed_thread_ids(output_path: Path) -> set[str]:
    """
    Scan an existing output JSONL file and return the set of thread_ids that
    have already been processed (i.e. have a reasoning_summary present).
    Returns an empty set if the file does not exist.
    """
    completed: set[str] = set()
    if not output_path.exists():
        return completed

    for _, record in read_jsonl(output_path):
        thread_id = record.get("thread_id")
        # Only count as done if reasoning actually ran (not just copied through)
        if thread_id and "reasoning_summary" in record:
            completed.add(thread_id)

    log.info("Resume: found %d already-completed conversations in %s",
             len(completed), output_path)
    return completed


def run_on_jsonl(
        input_path, output_path, ctx_window, context_window,
        temp1, temp2, temp_tiebreak, batch_size=BATCH_SIZE
        ):
    total = count_lines(input_path)
    out_path = Path(output_path)

    # ── Resume: scan output file for already-processed thread_ids
    completed = load_completed_thread_ids(out_path)
    n_skip = 0

    if completed:
        log.info(
            "Resuming — %d/%d conversations already done",
            len(completed), total,
            )
    else:
        log.info("LLM reasoning starting fresh — %d conversations", total)

    # ── Open in append mode when resuming, write mode when starting fresh ──────
    file_mode = "a" if completed else "w"
    log.info("Output file mode=%s → %s", file_mode, out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, file_mode, encoding="utf-8") as out_f:
        for line_no, conv in read_jsonl(input_path):
            thread_id = conv.get("thread_id", "")

            if thread_id in completed:
                log.debug("Skipping already-completed thread_id=%s", thread_id)
                n_skip += 1
                continue

            log_progress(line_no, total, thread_id, "REASON", log)
            result = run_reasoning(conv, ctx_window, context_window,
                                   temp1, temp2, temp_tiebreak, batch_size
                                   )
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()  # flush after each conversation — safe against crashes

    log.info(
        "Finished. Skipped=%d  Processed=%d  Output → %s",
        n_skip, total - n_skip, out_path,
        )


# CLI
def main() -> None:
    global MODEL, ENABLE_THINKING, OLLAMA_URL, THINKING_BUDGET, CTX_WINDOW, BATCH_SIZE, \
        CONFIDENCE_THRESHOLD_SUPPORT, CONFIDENCE_THRESHOLD_ATTACK, REASONING

    parser = argparse.ArgumentParser(
        description="Iterative LLM reasoning for argumentative relation classification"
        )
    parser.add_argument("--input", "-i", help="PAC selector JSON output")
    parser.add_argument("--output", "-o", help="Path to save enriched JSON output")
    parser.add_argument("--model", "-m", default=MODEL,
                        help=f"Ollama model name (default: {MODEL})"
                        )
    parser.add_argument("--ollama-url", default=OLLAMA_URL,
                        help=f"Ollama API endpoint (default: {OLLAMA_URL})"
                        )
    parser.add_argument("--ctx-window", type=int, default=CTX_WINDOW,
                        help=(
                            f"Model context window in tokens (default: {CTX_WINDOW}). "
                            "Batch size is derived automatically: "
                            f"budget = ctx × {1 - _REASONER_RESPONSE_FRAC:.0%} "
                            f"- {_REASONER_SYSTEM_OVERHEAD} overhead, "
                            f"÷ ~{TOKENS_PER_TARGET + AVG_SOURCES_PER_EDU * TOKENS_PER_SOURCE} tokens/EDU."
                        )
                        )
    parser.add_argument("--context-window", type=int, default=2,
                        help="±N EDUs of intra-post context added to prompt (default: 2)"
                        )
    parser.add_argument("--temp1", type=float, default=0.1,
                        help="Temperature for call 1 (default: 0.1)"
                        )
    parser.add_argument("--temp2", type=float, default=0.2,
                        help="Temperature for call 2 (default: 0.2)"
                        )
    parser.add_argument("--temp-tiebreak", type=float, default=0.15,
                        help="Temperature for tie-breaking call (default: 0.15)"
                        )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=(
                            f"EDUs per LLM call (default: {BATCH_SIZE}). "
                            "Try 10 (stable baseline), 15, 20 to find the "
                            "point where the thinking trace overwhelms num_predict."
                        )
                        )
    parser.add_argument("--conf-support", type=float, default=CONFIDENCE_THRESHOLD_SUPPORT,
                        help=f"Confidence gate for SUPPORT; at/below → neutral "
                             f"(default: {CONFIDENCE_THRESHOLD_SUPPORT})"
                        )
    parser.add_argument("--conf-attack", type=float, default=CONFIDENCE_THRESHOLD_ATTACK,
                        help=f"Confidence gate for ATTACK; at/below → neutral. "
                             f"Lower than support to boost attack recall "
                             f"(default: {CONFIDENCE_THRESHOLD_ATTACK})"
                        )
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking mode (faster but less reasoning depth)"
                        )
    parser.add_argument("--thinking-budget", default="medium",
                        choices=["low", "medium", "high"],
                        help="Thinking budget for gpt-oss (default: medium)."
                        )
    parser.add_argument("--configs", "-c", default=None,
                        help="Unified use-case profile JSON shared across all "
                             "pipeline steps (see pipeline_config.py). The "
                             "reasoner reads its three prompts from the profile's "
                             "'reasoning' section: 'system', 'tiebreak', "
                             "'cp_review'. e.g. configs/social_media.json or "
                             "configs/italian_interrogations.json. If omitted, "
                             "the built-in social-media default is used."
                        )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging"
                        )
    parser.add_argument("--log-file", "-l", default="llm_reasoner.log",
                        help="Log file path (default: llm_reasoner.log)"
                        )
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    # Load Config file
    try:
        cfg = load_config(args.configs, DEFAULT_CONFIG)
    except ValueError as exc:
        parser.error(str(exc))

    REASONING = cfg["reasoning"]

    MODEL = args.model
    OLLAMA_URL = args.ollama_url
    ENABLE_THINKING = not args.no_think
    THINKING_BUDGET = args.thinking_budget
    CTX_WINDOW = args.ctx_window
    BATCH_SIZE = args.batch_size
    CONFIDENCE_THRESHOLD_SUPPORT = args.conf_support
    CONFIDENCE_THRESHOLD_ATTACK = args.conf_attack

    log.info(
        "Model=%s  ollama_url=%s  thinking=%s  thinking_budget=%s  "
        "ctx_window=%d  batch_size=%d  context_window=%d  "
        "conf_gate[support=%.2f attack=%.2f]  configs=%s",
        MODEL, OLLAMA_URL, ENABLE_THINKING, THINKING_BUDGET,
        CTX_WINDOW, BATCH_SIZE, args.context_window,
        CONFIDENCE_THRESHOLD_SUPPORT, CONFIDENCE_THRESHOLD_ATTACK,
        (args.configs or "builtin_default"),
        )

    if args.input:
        run_on_jsonl(Path(args.input), Path(args.output),
                     CTX_WINDOW, args.context_window,
                     args.temp1, args.temp2, args.temp_tiebreak, BATCH_SIZE
                     )

    else:
        log.info("No --input — chaining from built-in sample")
        import extract_edu as ee
        import pac_selector as ps
        with JSONLWriter(Path(args.output)) as writer:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS), conv.get("thread_id", ""), "REASON", log)
                edu = ee.extract_edus(conv)
                pac = ps.select_all_pacs(edu)
                writer.write(run_reasoning(pac, CTX_WINDOW, args.context_window,
                                           args.temp1, args.temp2, args.temp_tiebreak,
                                           BATCH_SIZE
                                           )
                             )

if __name__ == "__main__":
    main()
import json
import re
import time
import logging
import argparse
from pathlib import Path
from typing import Optional
import requests

from sys_utils import (
    read_jsonl, JSONLWriter, log_progress, count_lines,
    SAMPLE_CONVERSATIONS, setup_logging, DEFAULT_CONFIG, load_config
    )

# Logging — configured after CLI args are parsed

log = logging.getLogger("edu_extractor")

# ─── Initialization (also set in configs file)
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:27b" # or gpt-oss:20B
TIMEOUT = 500
MAX_RETRIES = 2

# Context window of the model (tokens). Drives both num_ctx in Ollama and
# the heuristic batch budget — keep these in sync.
CTX_WINDOW = 8_192
CHARS_PER_TOKEN = 4

# initialize default configs
CFG: dict = dict(DEFAULT_CONFIG)

_DELETED_RE: re.Pattern | None = None
_CLEAN_RES: list = []
_CONNECTIVE_RE: re.Pattern | None = None


def apply_config(cfg: dict) -> None:
    """Install `cfg` as the active configs and (re)compile its regex helpers."""
    global CFG, _DELETED_RE, _CLEAN_RES, _CONNECTIVE_RE
    CFG = cfg
    edu = cfg["edu"]

    markers = edu.get("deleted_markers") or [r"\[removed\]", r"\[deleted\]"]
    _DELETED_RE = re.compile("|".join(markers), re.IGNORECASE)

    _CLEAN_RES = [re.compile(p) for p in edu.get("clean_patterns", [])]

    conns = edu.get("fallback_connectives") or []
    if conns:
        alt = "|".join(re.escape(c) for c in conns)
        _CONNECTIVE_RE = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)
    else:
        _CONNECTIVE_RE = None


# Compile defaults at import so helpers work before main() runs.
apply_config(CFG)


def compute_batch_token_budget(ctx_window: int = CTX_WINDOW) -> int:
    """
    Derive the maximum number of input tokens per batch from the model's
    context window.

    Budget = ctx_window - system_prompt_overhead - response_headroom
           = ctx_window * (1 - RESPONSE_HEADROOM_FRAC) - SYSTEM_PROMPT_OVERHEAD

    The response headroom is proportional to the context window so that
    larger windows still leave room for longer EDU lists. The system prompt
    overhead is fixed regardless of window size.

    Returns the budget in tokens (minimum 256 to avoid degenerate batches).
    """
    edu = CFG["edu"]
    if ctx_window is None:
        ctx_window = edu["ctx_window"]
    reserve = int(ctx_window * edu["response_headroom_frac"])
    budget = ctx_window - edu["system_prompt_overhead"] - reserve
    budget = max(budget, 256)
    log.debug("Batch budget: ctx=%d sys_overhead=%d reserve=%d budget=%d",
              ctx_window, edu["system_prompt_overhead"], reserve, budget
              )
    return budget


# # EDU content filter
# DELETED_RE = re.compile(r'\[removed\]|\[deleted\]', re.IGNORECASE)
#
# # ─── Raw-text pre-processing filters ─────────────────────────────────────────
#
# # Matches CMV moderator bot opening lines, e.g.:
# #   "Hello, users of CMV! This is a..."
# #   "Hello, users of r/changemyview!..."
# _MOD_HELLO_RE = re.compile(
#     r'^Hello,\s+users\s+of\s+(?:CMV|r/changemyview)[^\n]*[\n]?',
#     re.IGNORECASE | re.MULTILINE,
# )
#
# # Matches delta-confirmation lines produced by DeltaBot, e.g.:
# #   "Confirmed: 1 delta awarded to /u/aleph473 (..."
# #   "Confirmed: 2 deltas awarded to /u/foo (..."
# _DELTA_CONFIRMED_RE = re.compile(
#     r'^Confirmed:\s+\d+\s+deltas?\s+awarded\s+to\s+/u/\S+[^\n]*[\n]?',
#     re.IGNORECASE | re.MULTILINE,
# )
#
# # Matches any remaining DeltaBot / moderator-bot boilerplate blocks that begin
# # with a known bot signature line, running to the next blank line.
# # Covers patterns like "^/u/DeltaBot...", "^*I am a bot*...", etc.
# _BOT_BLOCK_RE = re.compile(
#     r'(?:^/u/DeltaBot\b[^\n]*|^\*I am a bot\b[^\n]*)'
#     r'(?:\n.+)*',
#     re.IGNORECASE | re.MULTILINE,
# )
#
# # ─── Parliamentary (interrogazione) boilerplate filters ──────────────────────
# # These target the non-argumentative ritual of Italian question-time turns.
#
# # Stage directions and stenographic annotations in parentheses, e.g.:
# #   "(Applausi dei deputati del gruppo Fratelli d'Italia)"
# #   "(Commenti)" / "(Vedi l'allegato A)" / "(Proteste del deputato ...)"
# _STAGE_DIRECTION_RE = re.compile(
#     r'\(\s*(?:Applausi|Commenti|Proteste|Vedi\s+l[\u2019\']allegato|Dalla\s+tribuna)'
#     r'[^)]*\)',
#     re.IGNORECASE,
# )
#
# # Courtesy openers at the start of a turn, possibly chained, e.g.:
# #   "Grazie, Presidente." / "Grazie Presidente e grazie Ministro."
# #   "Grazie, Ministro per la sua risposta," (only the leading formula)
# _COURTESY_OPENER_RE = re.compile(
#     r'^(?:\[\s*Speaker[^\]]*\]:\s*)?'                 # keep optional speaker tag intact below
#     r'((?:Grazie(?:\s+ancora)?[,]?\s+(?:signor\s+|signora\s+)?'
#     r'(?:Presidente|Ministro|Ministra|Sottosegretari[oa])'
#     r'(?:\s+e\s+grazie[,]?\s+(?:signor\s+|signora\s+)?'
#     r'(?:Presidente|Ministro|Ministra|Sottosegretari[oa]))?'
#     r'[.,]\s*)+)',
#     re.IGNORECASE,
# )
#
# # Procedural time-keeping asides anywhere in a turn, e.g.:
# #   "mi avvio a concludere" / "Concludo, Presidente." / "- mi avvio a concludere -"
# _PROCEDURAL_ASIDE_RE = re.compile(
#     r'(?:[-\u2013\u2014]\s*)?'
#     r'(?:mi\s+avvio\s+a\s+concludere|[Cc]oncludo,?\s+Presidente\.?|'
#     r'[Hh]o\s+concluso,?\s+Presidente\.?)'
#     r'(?:\s*[-\u2013\u2014])?',
# )
#
#
# def _clean_parliamentary_text(text: str) -> str:
#     """
#     Strip Italian question-time ritual that carries no argumentative content:
#       - stenographic stage directions: "(Applausi ...)", "(Commenti)", "(Vedi l'allegato A)"
#       - courtesy openers: "Grazie, Presidente.", "Grazie Presidente e grazie Ministro."
#       - procedural asides: "mi avvio a concludere", "Concludo, Presidente."
#     The speaker tag prefix "[Speaker N - Role]: " is preserved.
#     """
#     # Preserve the speaker tag prefix, clean the body
#     tag = ""
#     m = re.match(r'^(\[\s*Speaker[^\]]*\]:\s*)', text)
#     if m:
#         tag = m.group(1)
#         text = text[m.end():]
#
#     text = _STAGE_DIRECTION_RE.sub("", text)
#     text = _PROCEDURAL_ASIDE_RE.sub("", text)
#     # Courtesy openers can be chained ("Grazie Presidente, grazie Ministro, ...")
#     # — apply repeatedly until the head of the turn is substantive.
#     while True:
#         new = _COURTESY_OPENER_RE.sub("", text, count=1)
#         if new == text:
#             break
#         text = new.lstrip()
#
#     text = re.sub(r'\s{2,}', ' ', text)
#     # Leading vocative left behind by courtesy removal: "Signor Ministro, ..."
#     text = re.sub(
#         r'^(?:[Ss]ignor[ae]?\s+(?:Presidente|Ministr[oa]|Sottosegretari[oa])[,.]\s*)+',
#         '', text,
#     )
#     # Dangling punctuation after stage-direction removal: "Conferenza ." → "Conferenza."
#     text = re.sub(r'\s+([.,;:])', r'\1', text)
#     return tag + text.strip()


def clean_turn_text(text: str) -> str:
    """Strip configured boilerplate, collapse blank-line runs, trim."""
    for rx in _CLEAN_RES:
        text = rx.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



# SYSTEM_PROMPT = """You are a discourse analysis expert. Segment the given text into Elementary Discourse Units (EDUs).
#
# The input text may be in English or Italian. ALWAYS return EDUs verbatim in the
# original language of the input — never translate, paraphrase, or alter wording.
#
# RULES:
# - An EDU must be a coherent segment of text extracted VERBATIM from the input — do not paraphrase or alter wording
# - Simple sentences containing a single clear clause are returned as a single EDU unchanged
# - Composite sentences containing multiple clauses must be split into separate EDUs by identifying:
#     * Coordinating and subordinating connectives:
#         English: and, but, or, so, yet, because, although, however, therefore, since, while, whereas, unless, if, when, after, before, that, which, who
#         Italian: e, ed, ma, però, o, oppure, quindi, perché, poiché, siccome, sebbene, benché, anche se, tuttavia, pertanto, mentre, invece, se, quando, dopo che, prima che, che, il quale, la quale, i quali, affinché, dato che, in quanto
#     * Punctuation boundaries: commas, semicolons, colons, em-dashes that separate clauses
#     * Other discourse markers:
#         English: moreover, furthermore, in addition, as a result, for example, in contrast, on the other hand
#         Italian: inoltre, infatti, peraltro, di conseguenza, ad esempio, per esempio, al contrario, d'altra parte, in particolare, in conclusione
# - Drop any resulting EDU that contains TWO WORDS OR FEWER — these are not argumentatively useful
# - If a clause cannot be split further, return it as a single EDU
#
# OUTPUT FORMAT — return ONLY this JSON, no markdown, no explanation:
# {"edus": ["EDU 1", "EDU 2", "EDU 3"]}
#
# EXAMPLES:
# Input: I love coffee, but I avoid it because it affects my sleep. Climate change is real and it threatens our future.
# Output: {"edus": ["I love coffee", "but I avoid it", "because it affects my sleep", "Climate change is real", "and it threatens our future"]}
#
# Input: Il Governo ha adottato misure concrete, ma i risultati restano insufficienti perché gli sbarchi continuano ad aumentare.
# Output: {"edus": ["Il Governo ha adottato misure concrete", "ma i risultati restano insufficienti", "perché gli sbarchi continuano ad aumentare"]}
#
# Input: L'accordo prevede un consistente stanziamento economico; inoltre, rafforza la cooperazione tra i Paesi del Mediterraneo, che rappresenta una priorità strategica.
# Output: {"edus": ["L'accordo prevede un consistente stanziamento economico", "inoltre, rafforza la cooperazione tra i Paesi del Mediterraneo", "che rappresenta una priorità strategica"]}
#
# Input: Yes. But still.
# Output: {"edus": []}
# """
#
# # ─── Central Proposition prompt ───────────────────────────────────────────────
#
# CENTRAL_PROP_SYSTEM_PROMPT = """You are an argument analysis expert. Your task is to identify the Central Proposition from the opening contribution of a discussion. The discussion may be an online debate (e.g. Reddit) or a parliamentary interrogation (Italian "interrogazione a risposta immediata"); the EDUs may be in English or Italian.
#
# The Central Proposition is the single EDU that best represents the opening speaker's main claim or opinion. It must:
# 1. Come from the opening contribution (the opening post, or the deputy's question turn in a parliamentary interrogation)
# 2. Best reflect the stance or opinion expressed in the discussion title (for interrogations, the title is the official agenda topic of the interrogation)
# 3. Assert something the speaker believes or is defending
#
# For parliamentary interrogations specifically:
# - Prefer the EDU stating the deputy's substantive premise/claim about the issue (the position the Government's answer will confirm or contradict)
# - The formulaic request clause ("si chiede al Governo quali iniziative intenda intraprendere...") is acceptable only if no substantive premise EDU exists
# - Courtesy formulas ("Grazie, Presidente", "Signor Ministro") are never the central proposition
#
# Rhetorical questions are valid — they assert a position in question form.
# Exclude: genuine open questions, greetings, procedural statements ("I'll explain below", "mi avvio a concludere").
#
# You will be given the discussion title and a numbered list of EDUs from the opening contribution.
# Select the single best EDU by returning its index number.
#
# OUTPUT FORMAT — return ONLY this JSON, no markdown, no explanation:
# {"central_proposition_idx": <integer>}
# """


# ─── EDU filter ───────────────────────────────────────────────────────────────
def filter_edus(edus: list[dict]) -> tuple[list[dict], int]:
    """
    Remove EDU dicts that:
      - have empty / whitespace-only text
      - contain [deleted] or [removed] in their text
      - have two words or fewer (not argumentatively interesting)
    Each EDU dict has at minimum {"text": str, "speaker_id": str}.
    Returns (filtered_list, n_removed).
    """
    min_words = CFG["edu"]["min_edu_words"]
    result: list[dict] = []
    removed: int = 0
    for edu in edus:
        stripped = edu["text"].strip()
        if not stripped:
            removed += 1
            continue
        if _DELETED_RE.search(stripped):
            log.debug("  Dropping EDU (deleted marker): %.70s", stripped)
            removed += 1
            continue
        if len(stripped.split()) <= min_words:
            log.debug("  Dropping EDU (<= %d words): %.70s", min_words, stripped)
            removed += 1
            continue
        result.append({**edu, "text": stripped})
    return result, removed


# ─── Deduplication
def deduplicate_edus(edus: list[dict]) -> list[dict]:
    """
    Remove duplicate EDUs. Order-preserving; keeps first occurrence.
    Comparison is case-insensitive on the text field.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for edu in edus:
        key = edu["text"].strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(edu)
    return result


# ─── Ollama client ────────────────────────────────────────────────────────────
def call_ollama(user_text: str, retries: int = MAX_RETRIES) -> Optional[str]:
    """Send a prompt to Ollama; return raw response string or None on failure."""
    num_predict = int(CFG["edu"]["ctx_window"] * CFG["edu"]["response_headroom_frac"])
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": CFG["prompts"]["segmentation"]},
            {"role": "user", "content": user_text},
            ],
        "stream": False,
        "format": "json",
        "think": False,  # disable or enable thinking. Disabled for segmentation
        "options": {
            "temperature": 0.1,
            "num_predict": num_predict,
            "num_ctx": CFG["edu"]["ctx_window"],
            },
        }
    for attempt in range(1, retries + 1):
        try:
            log.debug("Ollama request attempt %d/%d  chars=%d",
                      attempt, retries, len(user_text)
                      )
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            # Strip Qwen3 thinking block in case it slips through format=json
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            log.debug("Raw output (first 300 chars): %.300s", raw)
            return raw
        except requests.exceptions.ConnectionError:
            log.error("Cannot reach Ollama at %s — is it running?", OLLAMA_URL)
            return None
        except requests.exceptions.Timeout:
            log.warning("Timeout (attempt %d/%d)", attempt, retries)
            if attempt == retries:
                return None
            time.sleep(2 ** attempt)
        except Exception as exc:
            log.error("Unexpected error: %s", exc)
            return None
    return None


# ─── Fallback splitter ────────────────────────────────────────────────────────
def fallback_split_edus(text: str) -> list[str]:
    """
    Rule-based EDU splitter used when LLM generation fails.

      1. Split on sentence boundaries (. ! ?) first.
      2. Within each sentence, split further on clause-separating punctuation
         (comma, semicolon, colon, em-dash) that precede a known connective or
         discourse marker — or standalone semicolons/colons.
      3. Drop any resulting segment of ≤2 words.
    """
    min_words = CFG["edu"]["min_edu_words"]
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    candidates: list[str] = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        for part in re.split(r"[;:\u2014]", sent):
            part = part.strip()
            if not part:
                continue
            if _CONNECTIVE_RE is not None:
                sub = re.split(r",\s*(?=" + _CONNECTIVE_RE.pattern + r")", part)
                candidates.extend(s.strip() for s in sub if s.strip())
            else:
                candidates.append(part)
    return [seg for seg in candidates if len(seg.split()) > min_words]


# ─── EDU parser ───────────────────────────────────────────────────────────────
def parse_edus(raw: str, fallback_text: str = "") -> list[str]:
    """
    Parse model output into a flat list of EDU strings.
    Falls back to rule-based splitting on failure.
    """
    if not raw or not raw.strip():
        log.warning("Empty response — using fallback splitter")
        return fallback_split_edus(fallback_text) if fallback_text.strip() else []

    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "edus" in data:
            return [e.strip() for e in data["edus"] if e.strip()]
        if isinstance(data, list):
            return [e.strip() for e in data if isinstance(e, str) and e.strip()]
    except json.JSONDecodeError:
        pass

    try:
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if m:
            data = json.loads(m.group())
            if isinstance(data, dict) and "edus" in data:
                return [e.strip() for e in data["edus"] if e.strip()]
    except json.JSONDecodeError:
        pass

    log.warning("JSON parse failed — using fallback splitter")
    return fallback_split_edus(fallback_text) if fallback_text.strip() else []


# ─── Step 1.1: Central Proposition extraction ────────────────────────────────
def call_ollama_central_prop(title: str, op_edus: list[str],
                             retries: int = MAX_RETRIES) -> Optional[str]:
    """
    LLM call for central proposition identification.
    Sends a numbered list of OP EDUs and asks the model to return
    the index of the best one — avoids any verbatim copying/validation issues.
    """
    edu_lines = "\n".join(f"[{i}] {edu}" for i, edu in enumerate(op_edus))
    user_content = (
        f"Discussion title: {title}\n\n"
        f"Opening post EDUs:\n{edu_lines}\n\n"
        f"Which EDU index is the central proposition?"
    )
    num_predict = int(CFG["edu"]["ctx_window"] * CFG["edu"]["response_headroom_frac"])
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": CFG["prompts"]["central_proposition"]},
            {"role": "user", "content": user_content},
            ],
        "stream": False,
        "format": "json",
        "think": True,
        "options": {
            "temperature": 0.3,
            "num_predict": num_predict,
            "num_ctx": CFG["edu"]["ctx_window"],
            },
        }
    for attempt in range(1, retries + 1):
        try:
            log.debug("Central-prop Ollama request attempt %d/%d  n_edus=%d",
                      attempt, retries, len(op_edus)
                      )
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if not raw:
                log.warning("Central-prop empty body on attempt %d/%d — retrying",
                            attempt, retries
                            )
                time.sleep(2 ** attempt)
                continue
            log.debug("Central-prop raw output: %.300s", raw)
            return raw
        except requests.exceptions.ConnectionError:
            log.error("Cannot reach Ollama at %s — is it running?", OLLAMA_URL)
            return None
        except requests.exceptions.Timeout:
            log.warning("Central-prop timeout (attempt %d/%d)", attempt, retries)
            if attempt == retries:
                return None
            time.sleep(2 ** attempt)
        except Exception as exc:
            log.error("Central-prop unexpected error: %s", exc)
            return None
    return None


def call_ollama_central_prop_old(
        title: str, op_text: str,
        retries: int = MAX_RETRIES
        ) -> Optional[str]:
    """
    LLM call specifically for central proposition extraction.
    Uses the same Ollama endpoint but a dedicated system prompt.
    """
    user_content = (
        f"Discussion title: {title}\n\n"
        f"Opening post:\n{op_text}"
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": CFG["prompts"]["central_proposition"]},
            {"role": "user", "content": user_content},
            ],
        "stream": False,
        "format": "json",
        "think": True,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
            "num_ctx": 8192,
            },
        }
    for attempt in range(1, retries + 1):
        try:
            log.debug("Central-prop Ollama request attempt %d/%d", attempt, retries)
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            log.debug("Central-prop raw output: %.300s", raw)
            return raw
        except requests.exceptions.ConnectionError:
            log.error("Cannot reach Ollama at %s — is it running?", OLLAMA_URL)
            return None
        except requests.exceptions.Timeout:
            log.warning("Central-prop timeout (attempt %d/%d)", attempt, retries)
            if attempt == retries:
                return None
            time.sleep(2 ** attempt)
        except Exception as exc:
            log.error("Central-prop unexpected error: %s", exc)
            return None
    return None


def extract_central_proposition(
        title: str,
        op_edus: list[str]
        ) -> Optional[str]:
    """
    Step 1.1 — Identify the Central Proposition (Pcentral).

    Sends the already-extracted OP EDUs as a numbered list to the LLM and
    asks it to return the index of the best one. Index-based selection
    eliminates verbatim-copy failures entirely — the result is always a
    valid EDU string taken directly from op_edus.

    Falls back to the OP EDU with the highest title word-overlap if the
    LLM fails or returns an out-of-range index.
    """
    if not op_edus:
        log.warning("Central-prop: no OP EDUs to select from")
        return None

    raw = call_ollama_central_prop(title, op_edus)

    # ── Parse index from LLM response ────────────────────────────────────────
    selected_idx: Optional[int] = None
    if raw:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            val = data.get("central_proposition_idx") if isinstance(data, dict) else None
            if isinstance(val, int):
                selected_idx = val
            elif isinstance(val, str) and val.isdigit():
                selected_idx = int(val)
        except (json.JSONDecodeError, AttributeError):
            m = re.search(r'"central_proposition_idx"\s*:\s*(\d+)', cleaned)
            if m:
                selected_idx = int(m.group(1))

    if selected_idx is not None and 0 <= selected_idx < len(op_edus):
        result = op_edus[selected_idx]
        log.info("  Central proposition [idx=%d]: %.80s", selected_idx, result)
        return result

    if selected_idx is not None:
        log.warning("  Central-prop LLM returned out-of-range idx=%d (n=%d) — falling back",
                    selected_idx, len(op_edus)
                    )

    # ── Fallback: OP EDU with highest title word-overlap ─────────────────────
    title_words = set(title.lower().split())
    best_edu = max(op_edus, key=lambda e: len(set(e.lower().split()) & title_words))
    best_score = len(set(best_edu.lower().split()) & title_words)
    log.info("  Central proposition via fallback (overlap=%d): %.80s", best_score, best_edu)
    return best_edu


# ─── EDU → turn attribution ───────────────────────────────────────────────────
def attribute_edus_to_turns(
        edus: list[str],
        batch: list[tuple[int, str]],
        ) -> dict[int, list[str]]:
    """
    Match each EDU to the turn whose text contains it (substring match).

    Strategy:
      For each EDU, check which turn text contains it as a substring
      (case-insensitive). Assign it to the first matching turn in batch order.
      If no turn contains the EDU (e.g. the model rephrased slightly),
      assign it to the turn with the highest character overlap ratio.

    Falls back to distributing unmatched EDUs proportionally across turns
    if overlap scoring also fails.

    Returns dict of turn_idx → [edus belonging to that turn].
    """
    result: dict[int, list[str]] = {t_idx: [] for t_idx, _ in batch}

    # Pre-normalise turn texts for matching
    turn_texts_lower = [(t_idx, txt.lower()) for t_idx, txt in batch]

    for edu in edus:
        edu_lower = edu.lower().strip()

        # 1. Exact substring match
        matched = False
        for t_idx, txt_lower in turn_texts_lower:
            if edu_lower in txt_lower:
                result[t_idx].append(edu)
                matched = True
                break

        if matched:
            continue

        # 2. Best overlap — find turn with most characters in common
        best_score = -1
        best_idx = batch[0][0]  # default to first turn
        for t_idx, txt_lower in turn_texts_lower:
            # Count shared words as a simple overlap score
            edu_words = set(edu_lower.split())
            turn_words = set(txt_lower.split())
            score = len(edu_words & turn_words)
            if score > best_score:
                best_score = score
                best_idx = t_idx

        log.debug("  EDU not found in any turn — assigned to turn %d by overlap: %.50s",
                  best_idx, edu
                  )
        result[best_idx].append(edu)

    return result


# ─── Paragraph (turn-batch) builder ───────────────────────────────────────────
def build_turn_batches(
        turns: list[dict],
        ctx_window: int = None,
        ) -> list[list[tuple[int, str]]]:
    """
    Group consecutive turns into batches whose combined text fits within the
    heuristic input budget derived from the model's context window.

    Budget = ctx_window * (1 - RESPONSE_HEADROOM_FRAC) - SYSTEM_PROMPT_OVERHEAD

    Each batch is a list of (turn_idx, turn_text) pairs. A single turn that
    exceeds the budget forms its own batch — it will still be processed but
    may produce a truncated response if it is genuinely too large for the model.
    """
    if ctx_window is None:
        ctx_window = CFG["edu"]["ctx_window"]
    max_tokens = compute_batch_token_budget(ctx_window)
    max_chars = max_tokens * CFG["edu"]["chars_per_token"]
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_chars = 0

    for t_idx, turn in enumerate(turns):
        text = clean_turn_text(turn.get("text", ""))
        if not text:
            continue
        turn_chars = len(text)
        if current and current_chars + turn_chars > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append((t_idx, text))
        current_chars += turn_chars

    if current:
        batches.append(current)

    log.debug(
        "Built %d turn-batch(es) from %d turns  (ctx=%d → budget=%d tokens / %d chars)",
        len(batches), len(turns), ctx_window, max_tokens, max_chars,
        )
    return batches


def build_prompt(batch: list[tuple[int, str]]) -> str:
    """Concatenate all turn texts into one plain paragraph for the LLM."""
    return " ".join(text for _, text in batch)


# ─── Per-conversation logic ───────────────────────────────────────────────────

def extract_edus(conv: dict) -> dict:
    """
    Process one conversation by merging turns into token-bounded paragraphs.

    Strategy:
      1. Group consecutive turns into batches of ≤ MAX_TOKENS combined tokens.
      2. Concatenate turn texts into a single plain paragraph per batch.
      3. One LLM call per batch — model returns {"edus": [...]} for the paragraph.
      4. Post-generation: attribute each EDU to the turn whose text contains it
         (substring match → word-overlap fallback). No turn-labelling in prompt.
      5. Deduplicate and filter per turn (including ≤2-word EDU removal).

    Step 1.1 — Central Proposition:
      After EDU extraction, the opening post (first non-deleted turn) is used
      to identify the Central Proposition via a dedicated LLM call, with a
      word-overlap fallback to OP EDUs. The result is stored at the top level
      as `central_proposition`.

    NOTE: The central proposition should always be manually inspected before
    downstream use.
    """
    turns = [t for t in conv.get("conversation", []) if not t.get("deleted")]
    total = len(turns)
    thread_id = conv.get("thread_id", "?")
    title = conv.get("title", "")

    log.info("thread_id=%s  turns=%d  title=%.55s",
             thread_id, total, title
             )

    if total == 0:
        return {**conv,
                "conversation": [],
                "central_proposition": None,
                "edu_summary": {"total_turns": 0, "total_edus": 0}
                }

    batches = build_turn_batches(turns, CTX_WINDOW)
    n_batches = len(batches)
    # log.info("  %d batch(es)  ctx=%d tokens", n_batches, CFG["edu"]["ctx_window"])
    # Accumulator: turn_idx → EDU list (plain strings; speaker tagged later)
    turn_edus: dict[int, list[str]] = {i: [] for i in range(total)}
    t0 = time.time()

    for b_idx, batch in enumerate(batches, 1):
        t_indices = [t for t, _ in batch]
        approx_tokens = sum(len(txt) for _, txt in batch) // CHARS_PER_TOKEN
        log.info("  batch [%d/%d]  turns=%d  ~%d tokens",
                 b_idx, n_batches, len(t_indices), approx_tokens
                 )

        paragraph = build_prompt(batch)
        raw = call_ollama(paragraph)
        edus = parse_edus(raw, fallback_text=paragraph)

        log.info("    → %d EDUs before attribution", len(edus))

        # Attribute EDUs back to their originating turns by text matching
        if len(batch) == 1:
            # Single turn — all EDUs belong to it
            t_idx = batch[0][0]
            turn_edus[t_idx].extend(edus)
        else:
            attributed = attribute_edus_to_turns(edus, batch)
            for t_idx, t_edus in attributed.items():
                turn_edus[t_idx].extend(t_edus)

    elapsed = time.time() - t0

    # ── Assemble enriched turn dicts ──────────────────────────────────────────
    enriched: list[dict] = []
    grand_total: int = 0

    for t_idx, turn in enumerate(turns):
        speaker_id = turn.get("speaker_id", "")

        # Wrap plain EDU strings into dicts carrying the speaker
        tagged: list[dict] = [
            {"text": e, "speaker_id": speaker_id}
            for e in turn_edus[t_idx]
        ]
        deduped = deduplicate_edus(tagged)
        filtered, n_drop = filter_edus(deduped)

        if n_drop:
            log.debug("  turn %d: filtered %d EDU(s)", t_idx, n_drop)

        # log.info("  turn [%d/%d] %s  post_id=%s  → %d EDUs",
        #          t_idx + 1, total,
        #          speaker_id or "?",
        #          turn.get("post_id", ""),
        #          len(filtered)
        #          )

        enriched.append({**turn, "edus": filtered, "edu_count": len(filtered)})
        grand_total += len(filtered)

    # ── Step 1.1: Central Proposition extraction ──────────────────────────────
    # The opening post is the first turn (index 0 in filtered turns list).
    op_turn = enriched[0] if enriched else None
    op_edus = [e["text"] for e in op_turn.get("edus", [])] if op_turn else []

    log.info("  Extracting central proposition for thread_id=%s  op_edus=%d",
             thread_id, len(op_edus)
             )
    central_proposition = extract_central_proposition(title, op_edus)

    # log.info(f"Title:{title} and central_prop:{central_proposition[:100]}")

    log.info("thread_id=%s  done — %d EDUs across %d turns in %.1fs",
             thread_id, grand_total, total, elapsed
             )

    return {
        **conv,
        "conversation": enriched,
        "central_proposition": central_proposition,
        "edu_summary": {"total_turns": total, "total_edus": grand_total},
        }


# ─── JSONL batch runner ───────────────────────────────────────────────────────
def load_completed_thread_ids(output_path: Path) -> set[str]:
    """
    Scan an existing output JSONL file and return the set of thread_ids that
    have already been successfully processed (have a non-empty edu_summary).
    Returns an empty set if the file does not exist.
    """
    completed: set[str] = set()
    if not output_path.exists():
        return completed

    for _, record in read_jsonl(output_path):
        thread_id = record.get("thread_id")
        summary = record.get("edu_summary", {})
        # Only count as done if EDU extraction actually ran
        if thread_id and "total_edus" in summary:
            completed.add(thread_id)

    log.info("Resume: found %d already-completed conversations in %s",
             len(completed), output_path
             )
    return completed


def run_on_jsonl(input_path: Path, output_path: Path) -> None:
    total = count_lines(input_path)
    out_path = Path(output_path)

    # ── Resume: scan output file for already-processed thread_ids ───────────────
    completed = load_completed_thread_ids(out_path)
    n_skip = 0

    if completed:
        log.info(
            "Resuming — %d/%d conversations already done",
            len(completed), total,
            )
    else:
        log.info("EDU extraction starting fresh — %d conversations in %s",
                 total, input_path
                 )

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

            log_progress(line_no, total, thread_id, "EDU", log)
            result = extract_edus(conv)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()  # flush after each conversation — safe against crashes

    log.info(
        "Finished. Skipped=%d  Processed=%d  Output → %s",
        n_skip, total - n_skip, out_path,
        )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:

    global MODEL, TIMEOUT, OLLAMA_URL

    parser = argparse.ArgumentParser(description="EDU extractor — JSONL in / JSONL out")
    parser.add_argument("--input", "-i",
                        help="Input JSONL (one conversation per line)"
                        )
    parser.add_argument("--output", "-o", default="edu_output.jsonl",
                        help="Output JSONL (default: edu_output.jsonl)"
                        )
    parser.add_argument("--configs", "-c", default=None,
                        help="Use-case profile JSON selecting the segmentation "
                             "and central-proposition prompts, the boilerplate "
                             "cleaning / fallback-splitter rules, and the budget "
                             "knobs (e.g. configs/social_media.json or "
                             "configs/italian_interrogations.json). If omitted, "
                             "the built-in social-media default is used."
                        )

    parser.add_argument("--model", "-m", default=MODEL,
                        help=f"Ollama model name (default: {MODEL})"
                        )
    parser.add_argument("--ollama-url", default=OLLAMA_URL,
                        help=f"Ollama API endpoint (default: {OLLAMA_URL})"
                        )
    parser.add_argument("--log-file", "-l", default="edu_extractor.log",
                        help="Log file path (default: edu_extractor.log)"
                        )
    parser.add_argument("--ctx-window", type=int, default=None,
                        help="Override the context window (tokens) from the "
                             "configs profile. Batch budget is derived as "
                             "ctx * (1 - response_headroom_frac) - "
                             "system_prompt_overhead, using the profile's "
                             "fractions. If omitted, the profile's ctx_window "
                             "is used."
                        )
    parser.add_argument("--timeout", type=int, default=TIMEOUT,
                        help=f"Per-request timeout seconds (default: {TIMEOUT})"
                        )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging"
                        )
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    try:
        cfg = load_config(args.configs, DEFAULT_CONFIG)
    except ValueError as exc:
        parser.error(str(exc))
    if args.ctx_window is not None:
        cfg["edu"]["ctx_window"] = args.ctx_window  # CLI override of the profile
    apply_config(cfg)

    MODEL = args.model
    TIMEOUT = args.timeout
    OLLAMA_URL = args.ollama_url

    budget = compute_batch_token_budget()
    log.info(
        "Model=%s  configs=%s  ctx_window=%d  chars/token=%d  batch_budget=%d tokens  "
        "min_edu_words=%d  timeout=%ds  log=%s",
        MODEL, (args.configs or "builtin_default"), CFG["edu"]["ctx_window"],
        CFG["edu"]["chars_per_token"], budget, CFG["edu"]["min_edu_words"],
        TIMEOUT, args.log_file,
        )

    if args.input:
        run_on_jsonl(Path(args.input), Path(args.output))
    else:
        log.info("No --input — using built-in sample")
        with JSONLWriter(Path(args.output)) as writer:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS),
                             conv.get("thread_id", ""), "EDU", log
                             )
                writer.write(extract_edus(conv))


if __name__ == "__main__":
    main()

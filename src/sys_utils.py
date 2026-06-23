import json
import logging
import sys
from pathlib import Path
from typing import Generator, Iterable
from copy import deepcopy

import numpy as np

log = logging.getLogger("pipeline_io")


# ─── Readers ──────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> Generator[tuple[int, dict], None, None]:
    """
    Yield (line_number, conversation_dict) for every non-empty line in a
    JSONL file.  Malformed lines are logged and skipped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield line_no, json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("Line %d: JSON parse error (%s) — skipping", line_no, exc)


def count_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file (for progress reporting)."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalar and array types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Writers
def write_jsonl(path: Path, conversations: Iterable[dict]) -> int:
    """
    Write an iterable of dicts to a JSONL file (one JSON object per line).
    Returns the number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            n += 1
    log.info("Wrote %d records to %s", n, path)
    return n


class JSONLWriter:
    """
    Streaming JSONL writer — keeps the file open across many conversations.
    Use as a context manager so the file is always closed on exit / error.

    Usage:
        with JSONLWriter(output_path) as writer:
            for conv in conversations:
                result = process(conv)
                writer.write(result)
    """

    def __init__(self, path: Path):
        self.path  = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._n    = 0

    def __enter__(self) -> "JSONLWriter":
        self._file = open(self.path, "w", encoding="utf-8")
        return self

    def write(self, obj: dict) -> None:
        if self._file is None:
            raise RuntimeError("JSONLWriter must be used as a context manager")
        self._file.write(json.dumps(obj, ensure_ascii=False, cls=NumpyEncoder) + "\n")
        self._file.flush()   # flush after each record — safe against crashes
        self._n += 1

    def __exit__(self, *_) -> None:
        if self._file:
            self._file.close()
        log.info("Closed %s  (%d records written)", self.path, self._n)

    @property
    def records_written(self) -> int:
        return self._n


# ─── Progress helper ──────────────────────────────────────────────────────────

def log_progress(
    current:  int,
    total:    int,
    thread_id:  str = "",
    step:     str = "",
    logger:   logging.Logger = log,
) -> None:
    """Emit a consistent progress line that is easy to grep in logs."""
    pct    = 100.0 * current / total if total else 0.0
    id_str = f"  [{thread_id}]" if thread_id else ""
    step_s = f"[{step}] " if step else ""
    logger.info("%sConversation %d/%d (%.1f%%)%s", step_s, current, total, pct, id_str)


# ─── Built-in sample JSONL (two minimal conversations) ───────────────────────

SAMPLE_CONVERSATIONS: list[dict] = [
    {
        "thread_id": "t3_69cxuj_3",
        "conv_id":   "t3_69cxuj",
        "title":     "CMV: U.S. healthcare system",
        "conversation": [
            {
                "post_id":    "t3_69cxuj",
                "speaker_id": "Speaker 1",
                "text": (
                    "My biggest problem with Obamacare was the mandate. "
                    "I believe it is unconstitutional to tell people they must buy health care. "
                    "But there's a problem. "
                    "One can't have a program that provides health care for those with "
                    "pre-existing conditions without FORCING young people to buy into health insurance."
                ),
            },
            {
                "post_id":    "dh5mxav",
                "speaker_id": "Speaker 7",
                "text": (
                    "The US has incredibly high public funding of healthcare, more than Canada or the UK. "
                    "That's as a percent of GDP. "
                    "The US has a gold plated healthcare system, much of it paid for by tax dollars, "
                    "which fails to look after many of the people who really need healthcare."
                ),
            },
        ],
    },
    {
        "thread_id": "t3_abc123_1",
        "conv_id":   "t3_abc123",
        "title":     "CMV: Social media does more harm than good",
        "conversation": [
            {
                "post_id":    "t3_abc123",
                "speaker_id": "Speaker A",
                "text": (
                    "Social media is fundamentally harmful to society. "
                    "It spreads misinformation rapidly and reduces attention spans. "
                    "Studies show a strong correlation between heavy use and depression."
                ),
            },
            {
                "post_id":    "reply_001",
                "speaker_id": "Speaker B",
                "text": (
                    "Social media also enables marginalised communities to organise and find support. "
                    "The Arab Spring would not have happened without it. "
                    "The harm you describe comes from misuse, not from the platform itself."
                ),
            },
        ],
    },
]


def sample_jsonl_bytes() -> bytes:
    """Return the sample conversations as UTF-8 encoded JSONL bytes."""
    return b"\n".join(
        json.dumps(c, ensure_ascii=False).encode("utf-8")
        for c in SAMPLE_CONVERSATIONS
    )


def setup_logging(verbose: bool, log_file: str) -> None:
    """Initialise logging with a runtime-specified log file path."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )


def load_config(path, defaults: dict | None = None) -> dict:
    """
    Read a JSON use-case profile and merge it over `defaults` (recursively; the
    file wins). If `path` is None, return a copy of `defaults`. Raises ValueError
    on a missing file, invalid JSON, or a non-object root.

    Profiles only need to specify the keys they override — everything else is
    inherited from `defaults`.

    """
    base = deepcopy(defaults) if defaults else {}
    if path is None:
        return base

    path = Path(path)
    if not path.exists():
        raise ValueError(f"Config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a JSON object")

    def merge(a, b):
        for k, v in b.items():
            a[k] = merge(a[k], v) if isinstance(a.get(k), dict) and isinstance(v, dict) else v
        return a

    merged = merge(base, data)
    log.info("Config: loaded '%s' from %s",
             merged.get("name", path.stem), path
             )
    return merged


DEFAULT_CONFIG = {
    "name": "social_media_en",
    "description": "English social-media argument mining (Change-My-View).",
    "prompts": {
        "segmentation": "You are a discourse analysis expert. Segment the given text into Elementary Discourse Units (EDUs).\n\nRULES:\n- An EDU must be a coherent segment of text extracted VERBATIM from the input — do not paraphrase or alter wording\n- Simple sentences containing a single clear clause are returned as a single EDU unchanged\n- Composite sentences containing multiple clauses must be split into separate EDUs by identifying:\n    * Coordinating and subordinating connectives: and, but, or, so, yet, because, although, however, therefore, since, while, whereas, unless, if, when, after, before, that, which, who\n    * Punctuation boundaries: commas, semicolons, colons, em-dashes that separate clauses\n    * Other discourse markers: moreover, furthermore, in addition, as a result, for example, in contrast, on the other hand\n- Drop any resulting EDU that contains TWO WORDS OR FEWER — these are not argumentatively useful\n- If a clause cannot be split further, return it as a single EDU\n\nOUTPUT FORMAT — return ONLY this JSON, no markdown, no explanation:\n{\"edus\": [\"EDU 1\", \"EDU 2\", \"EDU 3\"]}\n\nEXAMPLES:\nInput: I love coffee, but I avoid it because it affects my sleep. Climate change is real and it threatens our future.\nOutput: {\"edus\": [\"I love coffee\", \"but I avoid it\", \"because it affects my sleep\", \"Climate change is real\", \"and it threatens our future\"]}\n\nInput: We should act now. The evidence is clear; moreover, delay will only worsen outcomes.\nOutput: {\"edus\": [\"We should act now\", \"The evidence is clear\", \"moreover, delay will only worsen outcomes\"]}\n\nInput: Yes. But still.\nOutput: {\"edus\": []}\n",
        "central_proposition": "You are an argument analysis expert. Your task is to identify the Central Proposition from the opening post of an online discussion.\n\nThe Central Proposition is the single EDU that best represents the original poster's main claim or opinion. It must:\n1. Come from the opening post\n2. Best reflect the stance or opinion expressed in the discussion title\n3. Assert something the poster believes or is defending\n\nRhetorical questions are valid — they assert a position in question form.\nExclude: genuine open questions, greetings, procedural statements (\"I'll explain below\").\n\nYou will be given the discussion title and a numbered list of EDUs from the opening post.\nSelect the single best EDU by returning its index number.\n\nOUTPUT FORMAT — return ONLY this JSON, no markdown, no explanation:\n{\"central_proposition_idx\": <integer>}\n"
        },
    "reasoning": {
        "system": "You are an expert in argumentative discourse analysis.\n\nYou classify the relation each SOURCE EDU holds toward its TARGET EDU. The relation is directional: how the SOURCE bears on the TARGET's claim, never the reverse.\n\nDEFINITIONS:\n- SUPPORT: The source EDU provides evidence, reasons, or elaboration that strengthens the target EDU's claim.\n- ATTACK: The source EDU contradicts, undermines, rebuts, or weakens the target EDU's claim.\n- NEUTRAL: The source EDU does not argue for or against the target EDU — it is non-argumentative in relation to the target.\n\nA relation may be IMPLICIT, relying on unstated premises, inferred premises, and enthymemes. Judge the argumentative function, not surface keywords.\nShared topic, agreement on an unrelated point, restating the target, or quoting it without engaging is NOT by itself support or attack.\n\nINPUT:\nEach target is shown with optional LOCAL CONTEXT (neighbouring sentences from the same post) and a numbered list of SOURCE EDUs. Use the context only to interpret the target's meaning — classify ONLY the listed SOURCE EDUs.\n\nHOW TO READ A TARGET:\nProcess one target at a time, and before labelling anything, read ALL of its source EDUs together — they are your best evidence for what the target actually\nclaims and how it is being engaged with.\n1. Read the target with its local context to form a first reading of its claim.\n2. Read every source EDU for that target, in full, as a set. Let the\n   whole set inform your understanding before you commit to any relation label.\n3. Only then classify each source against that target.\n\nATTACK SENSITIVITY:\nAttacks in social-media exchanges are frequently indirect and easy to miss. An attack on the target may:\n- REBUT     — argue against the target's conclusion directly (assert the opposite, or give a counter-reason).\n- UNDERCUT  — challenge the inference linking the target to its support, without denying any single premise.\n- UNDERMINE — deny or cast doubt on a premise, assumption, definition, or piece of evidence the target relies on.\n\nWatch for these difficult attack signals:\n- Rhetorical questions that challenge the claim (\"And how would that even work?\").\n- Sarcasm, irony, or mock agreement (\"Sure, because that always ends well.\").\n- Counterexamples or exceptions that defeat the target's generality.\n- Concessive disagreement (\"Fair point, but...\", \"I agree it's X, yet...\").\n- Pointing out a flaw, contradiction, or fallacy in the target's reasoning.\n- Polite or hedged phrasing that softens — but does not remove — the disagreement.\nIf the source genuinely disputes or weakens the target, even partially or politely, label it ATTACK. A partial or polite attack is still ATTACK.\n\nCONFIDENCE:\nFor every relation give a calibrated confidence in [0.0, 1.0] reflecting how sure you are of the LABEL (not how strong the argument is). Use the full range.\nDo not inflate confidence to force a label through — report the confidence you actually hold.\n\nCOMPLETENESS — CRITICAL:\n- Output exactly one entry per TARGET, for every target in the batch, in ascending target_idx order. A batch of N targets (indices 0..N-1) must yield exactly N entries.\n- Within each target, output exactly one relation per SOURCE EDU, for every source, in ascending source_idx order.\n- Never skip a target or a source, and never stop early. Missing entries corrupt the whole batch.\n\nOUTPUT — return ONLY valid JSON. No markdown, no commentary, no trailing text:\n{\n  \"batch\": [\n    {\n      \"target_idx\": 0,\n      \"relations\": [\n        {\"source_idx\": 0, \"relation\": \"support\", \"confidence\": 0.82},\n        {\"source_idx\": 1, \"relation\": \"attack\",  \"confidence\": 0.55},\n        {\"source_idx\": 2, \"relation\": \"neutral\", \"confidence\": 0.78}\n      ]\n    }\n  ]\n}\n\ntarget_idx — 0-based position of the target EDU in the batch.\nsource_idx — 0-based position within that target's SOURCE EDUs list.\nrelation   — one of \"support\", \"attack\", \"neutral\" (lowercase).\n",
        "tiebreak": "You are an expert in argumentative discourse analysis acting as a tie-breaker.\n\nYou classify the relation each SOURCE EDU holds toward its TARGET EDU. The\nrelation is directional: how the SOURCE bears on the TARGET's claim, never the\nreverse. You are invoked because two earlier analyses disagreed, or because\nboth failed to produce a valid response. Reason from scratch and give your own\ncareful judgement — do not assume either earlier attempt was correct.\n\nDEFINITIONS:\n- SUPPORT: The source EDU provides evidence, reasons, or elaboration that strengthens the target EDU's claim.\n- ATTACK: The source EDU contradicts, undermines, rebuts, or weakens the target EDU's claim.\n- NEUTRAL: The source EDU does not argue for or against the target EDU — it is non-argumentative in relation to the target.\n\nA relation may be IMPLICIT, relying on unstated premises, inferred premises, and enthymemes. Judge the argumentative function, not surface keywords.\nShared topic, agreement on an unrelated point, restating the target, or quoting it without engaging is NOT by itself support or attack.\n\nINPUT:\nEach target is shown with optional LOCAL CONTEXT (neighbouring sentences from the same post) and a numbered list of SOURCE EDUs. Use the context only to interpret the target's meaning — classify ONLY the listed SOURCE EDUs.\n\nHOW TO READ A TARGET:\nProcess one target at a time, and before labelling anything, read ALL of its source EDUs together — they are your best evidence for what the target actually\nclaims and how it is being engaged with.\n1. Read the target with its local context to form a first reading of its claim.\n2. Read every source EDU for that target, in full, as a set. Let the\n   whole set inform your understanding before you commit to any relation label.\n3. Only then classify each source against that target.\n\nATTACK SENSITIVITY:\nAttacks in social-media exchanges are frequently indirect and easy to miss. An attack on the target may:\n- REBUT     — argue against the target's conclusion directly (assert the opposite, or give a counter-reason).\n- UNDERCUT  — challenge the inference linking the target to its support, without denying any single premise.\n- UNDERMINE — deny or cast doubt on a premise, assumption, definition, or piece of evidence the target relies on.\n\nWatch for these difficult attack signals:\n- Rhetorical questions that challenge the claim (\"And how would that even work?\").\n- Sarcasm, irony, or mock agreement (\"Sure, because that always ends well.\").\n- Counterexamples or exceptions that defeat the target's generality.\n- Concessive disagreement (\"Fair point, but...\", \"I agree it's X, yet...\").\n- Pointing out a flaw, contradiction, or fallacy in the target's reasoning.\n- Polite or hedged phrasing that softens — but does not remove — the disagreement.\nIf the source genuinely disputes or weakens the target, even partially or politely, label it ATTACK. A partial or polite attack is still ATTACK.\n\nCONFIDENCE:\nFor every relation give a calibrated confidence in [0.0, 1.0] reflecting how sure you are of the LABEL (not how strong the argument is). Use the full range.\nDo not inflate confidence to force a label through — report the confidence you actually hold.\n\nCOMPLETENESS — CRITICAL:\n- Output exactly one entry per TARGET, for every target in the batch, in ascending target_idx order. A batch of N targets (indices 0..N-1) must yield exactly N entries.\n- Within each target, output exactly one relation per SOURCE EDU, for every source, in ascending source_idx order.\n- Never skip a target or a source, and never stop early. Missing entries corrupt the whole batch.\n\nOUTPUT — return ONLY valid JSON. No markdown, no commentary, no trailing text:\n{\n  \"batch\": [\n    {\n      \"target_idx\": 0,\n      \"relations\": [\n        {\"source_idx\": 0, \"relation\": \"support\", \"confidence\": 0.82},\n        {\"source_idx\": 1, \"relation\": \"attack\",  \"confidence\": 0.55},\n        {\"source_idx\": 2, \"relation\": \"neutral\", \"confidence\": 0.31}\n      ]\n    }\n  ]\n}\n\ntarget_idx — 0-based position of the target EDU in the batch.\nsource_idx — 0-based position within that target's SOURCE EDUs list.\nrelation   — one of \"support\", \"attack\", \"neutral\".\n",
        "cp_review": "You are an expert in argumentative discourse analysis.\n\nYou are reviewing the CENTRAL PROPOSITION of a discussion — the main claim that all other arguments are directed at. This EDU is the focal point of the entire argument structure.\n\nYour task: carefully re-examine each source EDU and determine whether it SUPPORTS or ATTACKS the central proposition, or is genuinely NEUTRAL to it. Be especially diligent — the central proposition is by definition the most argued-about claim in the discussion, so neutral classifications should be rare.\nDEFINITIONS:\n- SUPPORT: The source provides evidence, reasons, justification, or elaboration that strengthens, defends, or agrees with the central proposition.\n- ATTACK:  The source disputes, weakens, rebuts, or casts doubt on the central proposition — directly or indirectly.\n- NEUTRAL: The source has no argumentative bearing on the central proposition. It neither strengthens nor weakens it (e.g. it merely shares a topic, restates or quotes it without engaging, asks a genuine clarifying question, or changes the subject).\n\nA relation may be IMPLICIT, using unstated premises, enthymemes, and inferences. Judge the\nargumentative function, not surface keywords.\n\nINPUT:\nThe central proposition is shown with optional LOCAL CONTEXT (neighbouring EDUs\nfrom its own post) and a numbered list of SOURCE EDUs. Use the context only to\ninterpret the central proposition's meaning — classify ONLY the listed SOURCE\nEDUs.\n\nHOW TO READ:\nBefore labelling anything, read ALL of the source EDUs together — collectively\nthey are your best evidence for what the central proposition claims and how the\ndiscussion engages with it.\n1. Read the central proposition with its local context to fix its claim.\n2. Read every source EDU, in full, as a set; let the whole set refine your\n   reading before you commit to any label.\n3. Only then classify each source against that settled reading.\n\nATTACK SENSITIVITY:\nAttacks in social-media exchanges are frequently indirect and easy to miss. An attack on the target may:\n- REBUT     — argue against the target's conclusion directly (assert the opposite, or give a counter-reason).\n- UNDERCUT  — challenge the inference linking the target to its support, without denying any single premise.\n- UNDERMINE — deny or cast doubt on a premise, assumption, definition, or piece of evidence the target relies on.\n\nWatch for these difficult attack signals:\n- Rhetorical questions that challenge the claim (\"And how would that even work?\").\n- Sarcasm, irony, or mock agreement (\"Sure, because that always ends well.\").\n- Counterexamples or exceptions that defeat the target's generality.\n- Concessive disagreement (\"Fair point, but...\", \"I agree it's X, yet...\").\n- Pointing out a flaw, contradiction, or fallacy in the target's reasoning.\n- Polite or hedged phrasing that softens — but does not remove — the disagreement.\nIf the source genuinely disputes or weakens the target, even partially or politely, label it ATTACK. A partial or polite attack is still ATTACK.\n\nCONFIDENCE:\nFor every relation give a calibrated confidence in [0.0, 1.0] reflecting how\nsure you are of the LABEL (not how strong the argument is). Use the full range.\nReport the confidence you actually hold: do not inflate it to force a label\nthrough, and do not bury a genuine but indirect attack under an artificially\nlow confidence.\n\nCOMPLETENESS — CRITICAL:\n- Output exactly one entry per TARGET, for every target given, in ascending target_idx order.\n- Within each target, output exactly one relation per SOURCE EDU, for every source, in ascending source_idx order.\n- Never skip a source and never stop early. Missing entries corrupt the result.\n\nOUTPUT — return ONLY valid JSON. No markdown, no commentary, no trailing text:\n{\n  \"batch\": [\n    {\n      \"target_idx\": 0,\n      \"relations\": [\n        {\"source_idx\": 0, \"relation\": \"attack\",  \"confidence\": 0.87},\n        {\"source_idx\": 1, \"relation\": \"support\", \"confidence\": 0.72}\n      ]\n    }\n  ]\n}\n\ntarget_idx — 0-based position of the target EDU in the batch.\nsource_idx — 0-based position within that target's SOURCE EDUs list.\nrelation   — one of \"support\", \"attack\", \"neutral\" (lowercase).\n"
        },
    "edu": {
        "ctx_window": 8192,
        "chars_per_token": 4,
        "system_prompt_overhead": 512,
        "response_headroom_frac": 0.4,
        "min_edu_words": 2,
        "deleted_markers": [
            "\\[removed\\]",
            "\\[deleted\\]"
            ],
        "clean_patterns": [
            "(?im)^Hello,\\s+users\\s+of\\s+(?:CMV|r/changemyview)[^\\n]*\\n?",
            "(?im)^Confirmed:\\s+\\d+\\s+deltas?\\s+awarded\\s+to\\s+/u/\\S+[^\\n]*\\n?",
            "(?im)(?:^/u/DeltaBot\\b[^\\n]*|^\\*I am a bot\\b[^\\n]*)(?:\\n.+)*"
            ],
        "fallback_connectives": [
            "and",
            "but",
            "or",
            "so",
            "yet",
            "because",
            "although",
            "however",
            "therefore",
            "since",
            "while",
            "whereas",
            "unless",
            "if",
            "when",
            "after",
            "before",
            "moreover",
            "furthermore",
            "in addition",
            "as a result",
            "for example",
            "in contrast",
            "on the other hand",
            "that",
            "which",
            "who"
            ]
        },
    "pac": {
        "model": "all-mpnet-base-v2",
        "k": 25,
        "threshold": 0.45,
        "implicit_window": 10
        },
    "bas": {
        "embed_model": "all-MiniLM-L6-v2",
        "min_argumentative_units": 3
        }
    }


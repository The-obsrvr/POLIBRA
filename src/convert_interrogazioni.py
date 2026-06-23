#!/usr/bin/env python3
"""
Convert the Italian parliamentary interrogations dataset (final_merged_interrogazioni.json)
into the JSONL conversation format consumed by the neuro-symbolic pipeline
(same schema as Data/samples_test.jsonl).

Segmentation logic
------------------
An "interrogazione a risposta immediata" transcript has a fixed rhetorical structure:
  1. QUESTION       — a deputy illustrates the interrogation (illustrazione)
  2. RESPONSE       — the Minister / Vice-Minister / Undersecretary answers
  3. COUNTER-REPLY  — a deputy exercises the right of reply (replica)
PRESIDENTE turns are procedural (giving the floor) and are dropped.

Output schema per line (mirrors samples_test.jsonl):
  thread_id, conv_id, title, is_delta, delta_ts, conversation[ post_id,
  parent_id, conv_id, speaker_id, timestamp, text ], turn_count, metadata{...}
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

# A turn header is an (almost) all-caps speaker name at line start, optionally
# followed by "(PARTY)" or ", Ministro ...", and terminated by ". "
TURN_HEADER_RE = re.compile(
    r"(?m)^("
    r"[A-ZÀÈÉÌÒÙÁÍÓÚ][A-ZÀÈÉÌÒÙÁÍÓÚ'\u2019\.\s\-]{3,60}?"          # NAME (caps)
    r"(?:\s*\([A-Z0-9ÀÈÉÌÒÙ'\u2019\-\.\s]+\)"                       # (PARTY)
    r"|,\s*(?:Ministr[oa]|Vice\s*[Mm]inistr[oa]|Sottosegretari[oa])[^.\n]*(?:\n[^.\n]*)?"  # , Ministro ...
    r")?"
    r")\.\s"
)

MINISTER_RE = re.compile(r"Ministr|Sottosegretar|MINISTR|SOTTOSEGRETAR")


def normalize_ws(text: str) -> str:
    """Collapse hard line-wraps from the PDF extraction into flowing text."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    # join wrapped lines, keep paragraph feel out of it (single space)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def split_turns(full_text: str):
    """Return (title, [(speaker_header, body), ...])."""
    matches = list(TURN_HEADER_RE.finditer(full_text))
    if not matches:
        return None, []
    # Text before the first header = the parenthesised agenda title
    title_raw = full_text[: matches[0].start()].strip()
    title = normalize_ws(title_raw).strip("()– ")
    title = re.sub(r"\s*[–-]\s*n\.\s*[\d\-]+\)?$", "", title)  # drop "– n. 3-00545)"

    turns = []
    for i, m in enumerate(matches):
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        header = normalize_ws(m.group(1))
        body = normalize_ws(full_text[m.end(): body_end])
        if body:
            turns.append((header, body))
    return title, turns


def segment_qrc(turns):
    """Identify question / response / counter-reply among the speaker turns."""
    question = response = counter = None
    for header, body in turns:
        if header.upper().startswith("PRESIDENTE"):
            continue                       # procedural — drop
        is_minister = bool(MINISTER_RE.search(header))
        if question is None and not is_minister:
            question = (header, body)
        elif response is None and is_minister:
            response = (header, body)
        elif response is not None and counter is None and not is_minister:
            counter = (header, body)
    return question, response, counter


def speaker_name(header: str) -> str:
    """'ANGELO ROSSI (FDI)' -> 'Angelo Rossi'; 'ANTONIO TAJANI, Ministro…' -> 'Antonio Tajani'."""
    name = re.split(r"\(|,", header)[0].strip()
    return name.title()


def party_or_role(header: str) -> str:
    m = re.search(r"\(([^)]+)\)", header)
    if m:
        return m.group(1).strip()
    m = re.search(r",\s*(.+)$", header)
    return m.group(1).strip() if m else ""


def convert_record(rec: dict, idx: int):
    full_text = rec.get("full_interro_text")
    if not full_text:
        return None  # no transcript available
    title, turns = split_turns(full_text)
    question, response, counter = segment_qrc(turns)
    if not (question and response and counter):
        return None  # incomplete transcript — skip

    conv_id = rec["id"]
    thread_id = f"{conv_id}_1"
    roles = [("Question", question), ("Response", response), ("Counter-Reply", counter)]

    # Speaker IDs reflect the SIDE, not the person:
    #   Speaker 1 — the interrogating side (questioner and replier, who may be
    #               different deputies of the same group). Required so that
    #               SEI's OP-authorship bonus (strength_initializer, s2)
    #               recognises the counter-reply as OP-side activity.
    #   Speaker 2 — the responding Government member.
    # Per-person identity is preserved in speaker_name / affiliation.
    conversation = []
    prev_post_id = ""
    for j, (role, (header, body)) in enumerate(roles, start=1):
        name = speaker_name(header)
        sid = "Speaker 2" if role == "Response" else "Speaker 1"
        post_id = f"{conv_id}_t{j}"
        conversation.append({
            "post_id":      post_id,
            "parent_id":    prev_post_id,
            "conv_id":      conv_id,
            "speaker_id":   sid,
            "timestamp":    rec.get("date_modified", ""),
            "text":         f"[{sid} - {role}]: {body}",
            "deleted":      False,
            "role":         role.lower(),
            "speaker_name": name,
            "affiliation":  party_or_role(header),
        })
        prev_post_id = post_id

    return {
        "thread_id":  thread_id,
        "conv_id":    conv_id,
        "title":      title or rec.get("pol_topic", ""),
        "is_delta":   None,
        "delta_ts":   None,
        "conversation": conversation,
        "turn_count": len(conversation),
        "sentence_count": sum(len(re.findall(r"[.!?]+", t["text"])) for t in conversation),
        "metadata": {
            "source":         "final_merged_interrogazioni.json",
            "act_id":         rec.get("id"),
            "type":           rec.get("type"),
            "legislature":    rec.get("legislature"),
            "branch":         rec.get("branch"),
            "presenter":      rec.get("presenter"),
            "date_presented": rec.get("date_presented"),
            "date_modified":  rec.get("date_modified"),
            "seduta":         rec.get("seduta"),
            "pol_topic":      rec.get("pol_topic"),
            "block":          rec.get("block"),
            "language":       "it",
        },
    }


def main():
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("final_merged_interrogazioni.json")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("interrogazioni_samples.jsonl")
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    data = json.loads(inp.read_text(encoding="utf-8"))
    written, skipped = 0, 0
    with out.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(data):
            if written >= limit:
                break
            row = convert_record(rec, i)
            if row is None:
                skipped += 1
                print(f"  skipped {rec.get('id')} — could not segment Q/R/C")
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            print(f"  ok {rec.get('id')} — title: {row['title'][:70]}…")
    print(f"Done: {written} written, {skipped} skipped → {out}")


if __name__ == "__main__":
    main()

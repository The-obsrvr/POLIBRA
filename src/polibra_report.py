"""
NARS Discussion Report Generator
Generates a clean PDF report for each conversation processed by the NARS pipeline.

Inputs:
  --predictions   : output of persuasiveness_detector.py  (predictions_*.jsonl)
  --bas           : output of bas_assembler.py             (bas_repair*.jsonl or bas_no_repair*.jsonl)
  --output-dir    : directory to write one PDF per conversation (default: reports/)
  --thread-id     : generate report for a single thread_id only

Usage:
  python polibra_report.py --predictions predictions_repair.jsonl \
                        --bas bas_repair.jsonl \
                        --output-dir reports/

  python polibra_report.py --predictions predictions_repair.jsonl \
                        --bas bas_repair.jsonl \
                        --thread-id t3_64yzdt_5
"""
import argparse
import json
import logging
import math
import textwrap
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

log = logging.getLogger("nars_report")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Colour palette (matching the UI in the screenshots) ───────────────────────
C_BG         = colors.HexColor("#F9F9F7")
C_CARD       = colors.white
C_BORDER     = colors.HexColor("#E5E5E3")
C_TEXT       = colors.HexColor("#1A1A1A")
C_MUTED      = colors.HexColor("#6B6B6B")
C_GREEN      = colors.HexColor("#2D7A3A")
C_RED        = colors.HexColor("#C0392B")
C_ORANGE     = colors.HexColor("#D4640A")
C_LABEL_BG   = colors.HexColor("#F0F0EE")
C_GREEN_PILL = colors.HexColor("#E8F5E9")
C_RED_PILL   = colors.HexColor("#FDECEA")

W, H = A4
MARGIN = 18 * mm
INNER_W = W - 2 * MARGIN

STRATEGY_LABELS = {"s1": "UI (uniform)", "s2": "SEI (structural)",
                   "s3": "CSI (contextual)", "s4": "HI (hybrid)"}

# ── Style helpers ─────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()

    def s(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    return {
        "meta":      s("meta",      fontSize=9,  textColor=C_MUTED,
                       fontName="Helvetica"),
        "label":     s("label",     fontSize=7.5, textColor=C_MUTED,
                       fontName="Helvetica-Bold", spaceAfter=2),
        "cp":        s("cp",        fontSize=12, textColor=C_TEXT,
                       fontName="Helvetica-Oblique", leading=17),
        "section":   s("section",   fontSize=7.5, textColor=C_MUTED,
                       fontName="Helvetica-Bold", spaceBefore=4),
        "body":      s("body",      fontSize=9.5, textColor=C_TEXT,
                       fontName="Helvetica", leading=14),
        "small":     s("small",     fontSize=8.5, textColor=C_MUTED,
                       fontName="Helvetica", leading=12),
        "kv_key":    s("kv_key",    fontSize=9,  textColor=C_TEXT,
                       fontName="Helvetica"),
        "kv_val_g":  s("kv_val_g",  fontSize=9,  textColor=C_GREEN,
                       fontName="Helvetica-Bold"),
        "kv_val_r":  s("kv_val_r",  fontSize=9,  textColor=C_RED,
                       fontName="Helvetica-Bold"),
        "kv_val":    s("kv_val",    fontSize=9,  textColor=C_TEXT,
                       fontName="Helvetica-Bold"),
        "arg_text":  s("arg_text",  fontSize=9,  textColor=C_TEXT,
                       fontName="Helvetica-Oblique", leading=13),
        "arg_small": s("arg_small", fontSize=8,  textColor=C_MUTED,
                       fontName="Helvetica", leading=11),
        "ens_pers":  s("ens_pers",  fontSize=9.5, textColor=C_GREEN,
                       fontName="Helvetica-Bold"),
        "ens_npers": s("ens_npers", fontSize=9.5, textColor=C_RED,
                       fontName="Helvetica-Bold"),
        "gt":        s("gt",        fontSize=9,  textColor=C_MUTED,
                       fontName="Helvetica"),
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if "__eval__" not in obj:
                        out.append(obj)
                except json.JSONDecodeError:
                    pass
    return out


def _index_by_thread(records: list[dict]) -> dict[str, dict]:
    return {r.get("thread_id", r.get("conv_id", str(i))): r
            for i, r in enumerate(records)}


# ── BAS helpers ───────────────────────────────────────────────────────────────

def _get_root_id(bas: dict) -> str:
    return bas.get("summary", {}).get("root_id", "")


def _root_node(bas: dict) -> dict:
    root_id = _get_root_id(bas)
    for n in bas.get("nodes", []):
        if n["id"] == root_id:
            return n
    nodes = bas.get("nodes", [])
    return nodes[0] if nodes else {}


def _graph_stats(bas: dict) -> dict:
    edges    = bas.get("edges", [])
    nodes    = bas.get("nodes", [])
    root_id  = _get_root_id(bas)
    supports = [e for e in edges if e.get("relation") == "support"]
    attacks  = [e for e in edges if e.get("relation") == "attack"]
    atk_root = [e for e in attacks if e.get("target") == root_id]

    speaker_ids = {n.get("speaker_id") for n in nodes if n.get("speaker_id")}

    return {
        "n_units":    len(nodes),
        "n_support":  len(supports),
        "n_attack":   len(attacks),
        "n_atk_root": len(atk_root),
        "n_speakers": len(speaker_ids),
        "ratio":      f"{len(supports)}:{max(len(attacks),1)}",
    }


def _key_arguments(bas: dict, n: int = 3) -> tuple[list[dict], list[dict]]:
    """Return top-N attacking and supporting nodes by in-degree to P_central."""
    edges   = bas.get("edges", [])
    nodes   = bas.get("nodes", [])
    root_id = _get_root_id(bas)
    nmap    = {nd["id"]: nd for nd in nodes}

    attackers  = [nmap[e["source"]] for e in edges
                  if e.get("relation") == "attack" and e.get("target") == root_id
                  and e["source"] in nmap]
    supporters = [nmap[e["source"]] for e in edges
                  if e.get("relation") == "support" and e.get("target") == root_id
                  and e["source"] in nmap]

    # Also collect indirect attackers (attack nodes that attack supporters)
    support_ids = {e["source"] for e in edges
                   if e.get("relation") == "support" and e.get("target") == root_id}
    indirect_atk = [nmap[e["source"]] for e in edges
                    if e.get("relation") == "attack" and e.get("target") in support_ids
                    and e["source"] in nmap and nmap[e["source"]] not in attackers]

    all_atk = attackers + indirect_atk[:max(0, n - len(attackers))]
    return all_atk[:n], supporters[:n]


def _failure_reason(bas: dict, pred: dict) -> str:
    """Generate a human-readable explanation of why predictions may be wrong."""
    stats = _graph_stats(bas)
    gt    = pred.get("ground_truth")
    preds = pred.get("thresholds", {})
    if not preds:
        return ""
    first_thresh = next(iter(preds.values()), {})
    predictions  = first_thresh.get("predictions", {})
    ensemble     = first_thresh.get("ensemble", {})
    predicted    = ensemble.get("predicted", False)

    if gt is None:
        return ""

    # Correct prediction — explain why it worked
    if predicted == gt:
        if gt:
            return (f"The single direct attack on P_central combined with "
                    f"{stats['n_attack']} total attack relation(s) produced sufficient "
                    f"strength shift across all strategies.")
        else:
            return (f"Despite {stats['n_attack']} attack(s), the "
                    f"{stats['n_support']}:{stats['n_attack']} support-to-attack ratio "
                    f"kept P_central's posterior above the threshold — "
                    f"correctly indicating no persuasion occurred.")

    # Wrong prediction — explain the failure mode
    if predicted and not gt:
        return (f"The combined weight of {stats['n_attack']} attack(s) — "
                f"including {stats['n_atk_root']} direct hit(s) on P_central — "
                f"pulls the posterior below the threshold. The system mistakes "
                f"structural argumentative pressure for a genuine view change "
                f"the OP never made.")
    if not predicted and gt:
        return (f"Despite a confirmed view change (delta), the "
                f"{stats['n_support']}:{stats['n_attack']} support-to-attack ratio "
                f"kept P_central's posterior too stable. The system missed the "
                f"persuasion event because the supporting majority suppressed "
                f"the attacking signal.")
    return ""


# ── PDF building blocks ───────────────────────────────────────────────────────

def _card(content_rows: list, col_widths: list = None) -> Table:
    """Wrap content in a white card with a light border."""
    t = Table(content_rows, colWidths=col_widths or [INNER_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), C_CARD),
        ("BOX",         (0, 0), (-1, -1), 0.5, C_BORDER),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",  (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def _strategy_card(strat_id: str, initial: float, final: float,
                   abs_delta: float, predicted: bool, styles: dict) -> Table:
    """One strategy tile: label, strength arrow, |Δ|, verdict."""
    label   = STRATEGY_LABELS.get(strat_id, strat_id.upper())
    arrow   = f"{initial:.2f} \u2192 {final:.2f}"
    delta_s = f"|Δ| = {abs_delta:.3f}"
    verdict = "persuaded \u2713" if predicted else "not persuaded \u00d7"
    v_color = C_GREEN if predicted else C_RED
    bg      = C_GREEN_PILL if predicted else C_RED_PILL

    label_p   = Paragraph(label,   styles["small"])
    arrow_p   = Paragraph(f"<b>{arrow}</b>", ParagraphStyle(
        "arrow", parent=styles["body"], fontSize=13, leading=16))
    delta_p   = Paragraph(delta_s, ParagraphStyle(
        "delta", parent=styles["small"],
        textColor=C_RED if abs_delta >= 0.3 else C_GREEN))
    verdict_p = Paragraph(verdict, ParagraphStyle(
        "verdict", parent=styles["small"], textColor=v_color, fontName="Helvetica-Bold"))

    inner = Table([[label_p], [arrow_p], [delta_p], [verdict_p]],
                  colWidths=[INNER_W / 4 - 8 * mm])
    inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 1), (0, 2), "LEFT"),
    ]))
    return inner


def _arg_block(node: dict, rel: str, styles: dict, note: str = "") -> Table:
    """A single argument card with relation type badge."""
    text    = node.get("text", "")[:200]
    spk     = node.get("speaker_id", "")
    rel_col = C_RED if rel == "attack" else C_GREEN
    rel_lbl = "⚔ attack" if rel == "attack" else "✦ support"

    badge = Paragraph(f'<font color="#{rel_col.hexval()[2:]}"><b>{rel_lbl}</b></font>',
                      styles["small"])
    body  = Paragraph(f'"{text}"', styles["arg_text"])
    meta  = Paragraph(f"Speaker: {spk}" + (f"  —  {note}" if note else ""),
                      styles["arg_small"])

    inner = Table([[badge], [body], [meta]],
                  colWidths=[INNER_W / 2 - 12 * mm])
    inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_LABEL_BG),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    return inner


# ── Main report builder ───────────────────────────────────────────────────────

def build_report(thread_id: str, pred: dict, bas: dict,
                 output_path: Path) -> None:
    """Build a single-conversation PDF report."""
    st = _styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"NARS Report — {thread_id}",
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    stats    = _graph_stats(bas)
    n_rel    = stats["n_support"] + stats["n_attack"]
    meta_txt = (f"{thread_id}    "
                f"{stats['n_speakers']} speaker(s)  ·  "
                f"{stats['n_units']} units  ·  "
                f"{n_rel} relations")
    story.append(Paragraph(meta_txt, st["meta"]))
    story.append(Spacer(1, 3 * mm))

    # ── Central Proposition card ───────────────────────────────────────────────
    root = _root_node(bas)
    cp_text = root.get("text", bas.get("summary", {}).get("root_text", ""))
    op_id   = root.get("speaker_id", "Speaker 1")
    story.append(Paragraph("CENTRAL PROPOSITION P<sub>CENTRAL</sub> "
                            f"— {op_id.upper()} (ORIGINAL POSTER)", st["label"]))
    story.append(Spacer(1, 1 * mm))

    cp_card = _card([[Paragraph(f'"{cp_text}"', st["cp"])]])
    story.append(cp_card)
    story.append(Spacer(1, 4 * mm))

    # ── Graph stats + Key Attacking Argument (two-column) ─────────────────────
    key_atk, key_sup = _key_arguments(bas, n=1)

    # Left: graph stats
    kv_rows = [
        [Paragraph("Argumentative units",  st["kv_key"]),
         Paragraph(str(stats["n_units"]),  st["kv_val"])],
        [Paragraph("Support relations",     st["kv_key"]),
         Paragraph(str(stats["n_support"]), st["kv_val_g"])],
        [Paragraph("Attack relations",      st["kv_key"]),
         Paragraph(str(stats["n_attack"]),  st["kv_val_r"])],
        [Paragraph("Attacks on P<sub>CENTRAL</sub>", st["kv_key"]),
         Paragraph(str(stats["n_atk_root"]), st["kv_val_r"])],
    ]
    ratio_row = [[Paragraph(f"Support-to-attack ratio: {stats['ratio']}",
                             st["small"]), Paragraph("", st["small"])]]

    graph_inner = Table(kv_rows + ratio_row,
                        colWidths=[INNER_W * 0.28, INNER_W * 0.12])
    graph_inner.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("SPAN",          (0, 4), (1, 4)),
        ("TEXTCOLOR",     (0, 4), (1, 4), C_MUTED),
    ]))
    graph_section = Table([[Paragraph("ARGUMENT GRAPH", st["section"])],
                           [graph_inner]],
                          colWidths=[INNER_W * 0.45])
    graph_section.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("BACKGROUND",    (0, 0), (-1, -1), C_CARD),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
    ]))

    # Right: key argument analysis
    preds_map   = pred.get("thresholds", {})
    first_t_key = next(iter(preds_map), None)
    first_preds = preds_map.get(first_t_key, {}).get("predictions", {}) if first_t_key else {}
    ensemble    = preds_map.get(first_t_key, {}).get("ensemble", {}) if first_t_key else {}
    gt          = pred.get("ground_truth")
    predicted   = ensemble.get("predicted", False)
    correct     = (predicted == gt) if gt is not None else None

    if key_atk:
        ka_node = key_atk[0]
        ka_text = ka_node.get("text", "")[:160]
        reason  = _failure_reason(bas, pred)
        why_label = "WHY THE SYSTEM SUCCEEDED" if correct else (
            "WHY THE SYSTEM FAILED" if correct is False else "ANALYSIS")
        atk_content = [
            [Paragraph("KEY ATTACKING ARGUMENT", st["section"])],
            [Paragraph(f'"{ka_text}"', st["arg_text"])],
            [Paragraph(why_label, st["section"])],
            [Paragraph(reason, st["small"])],
        ]
    else:
        atk_content = [
            [Paragraph("KEY ATTACKING ARGUMENT", st["section"])],
            [Paragraph("No direct attacks on P_central found.", st["small"])],
        ]

    atk_section = Table(atk_content,
                        colWidths=[INNER_W * 0.50])
    atk_section.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("BACKGROUND",    (0, 0), (-1, -1), C_CARD),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
    ]))

    two_col = Table([[graph_section, atk_section]],
                    colWidths=[INNER_W * 0.47, INNER_W * 0.53],
                    spaceBefore=0)
    two_col.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ("COLPADDING",   (0, 0), (-1, -1), 3),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 4 * mm))

    # ── Key arguments section (expanded) ──────────────────────────────────────
    all_atk, all_sup = _key_arguments(bas, n=3)
    if all_atk or all_sup:
        story.append(Paragraph("KEY ARGUMENTATIVE SIGNALS", st["section"]))
        story.append(Spacer(1, 1.5 * mm))
        story.append(HRFlowable(width=INNER_W, thickness=0.5, color=C_BORDER))
        story.append(Spacer(1, 2 * mm))

        arg_cells = []
        for i in range(max(len(all_atk), len(all_sup))):
            left  = _arg_block(all_atk[i], "attack",  st, "direct attack on P_central" if i == 0 else "") \
                    if i < len(all_atk) else Paragraph("", st["small"])
            right = _arg_block(all_sup[i], "support", st, "direct support of P_central" if i == 0 else "") \
                    if i < len(all_sup) else Paragraph("", st["small"])
            arg_cells.append([left, right])

        if arg_cells:
            arg_table = Table(arg_cells,
                              colWidths=[INNER_W * 0.49, INNER_W * 0.49],
                              spaceBefore=0)
            arg_table.setStyle(TableStyle([
                ("VALIGN",       (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",  (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING",   (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
                ("COLPADDING",   (0, 0), (-1, -1), 4),
            ]))
            story.append(arg_table)
        story.append(Spacer(1, 4 * mm))

    # ── Symbolic Reasoning section ────────────────────────────────────────────
    # Find the threshold matching the screenshots (default 0.3)
    available_thresholds = sorted(preds_map.keys(), key=float)
    display_thresh = "0.3" if "0.3" in available_thresholds else (
        available_thresholds[1] if len(available_thresholds) > 1 else
        available_thresholds[0] if available_thresholds else None)

    if display_thresh:
        thresh_data  = preds_map[display_thresh]
        thresh_preds = thresh_data.get("predictions", {})
        thresh_ens   = thresh_data.get("ensemble", {})

        story.append(Paragraph(
            f"SYMBOLIC REASONING — STRENGTH PROPAGATION ON P<sub>CENTRAL</sub> "
            f"(Δ THRESHOLD = {float(display_thresh):.1f})", st["section"]))
        story.append(Spacer(1, 1.5 * mm))
        story.append(HRFlowable(width=INNER_W, thickness=0.5, color=C_BORDER))
        story.append(Spacer(1, 2 * mm))

        # Strategy tiles — read initial/final from root node
        root_id  = _get_root_id(bas)
        strat_tiles = []
        for strat_id in ["s1", "s2", "s3", "s4"]:
            sp = thresh_preds.get(strat_id)
            if sp is None:
                continue
            tile = _strategy_card(
                strat_id,
                sp["initial"], sp["final"],
                sp["abs_delta"], sp["predicted"], st)
            strat_tiles.append(tile)

        if strat_tiles:
            # Pad to 4
            while len(strat_tiles) < 4:
                strat_tiles.append(Paragraph("", st["small"]))
            tiles_table = Table([strat_tiles],
                                colWidths=[INNER_W / 4] * 4)
            tiles_table.setStyle(TableStyle([
                ("VALIGN",       (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",  (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING",   (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
            ]))
            story.append(tiles_table)
        story.append(Spacer(1, 3 * mm))

        # Ensemble verdict row
        ens_pred   = thresh_ens.get("predicted", False)
        votes_yes  = thresh_ens.get("votes_persuasive", 0)
        votes_tot  = thresh_ens.get("votes_total", 4)
        ens_label  = ("persuaded" if ens_pred else "not persuaded")
        ens_votes  = f"{votes_yes}/{votes_tot} strategies agree"
        tick       = "\u2713" if (correct is True) else ("\u00d7" if correct is False else "")
        gt_label   = ("delta confirmed" if gt else "no delta") if gt is not None else "unknown"

        ens_style  = st["ens_pers"] if ens_pred else st["ens_npers"]
        ens_bg     = C_GREEN_PILL if ens_pred else C_RED_PILL

        ens_pill   = Table([[Paragraph(
            f"{ens_label} — {ens_votes} {tick}", ens_style)]],
            colWidths=[INNER_W * 0.45])
        ens_pill.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), ens_bg),
            ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ]))

        ens_row = Table(
            [[Paragraph("Majority vote (ensemble):", st["kv_key"]),
              ens_pill,
              Paragraph(f"Ground truth: {gt_label}", st["gt"])]],
            colWidths=[INNER_W * 0.22, INNER_W * 0.48, INNER_W * 0.30])
        ens_row.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING",   (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ]))
        story.append(ens_row)
        story.append(Spacer(1, 4 * mm))

        # ── All-thresholds summary table ──────────────────────────────────────
        story.append(Paragraph("THRESHOLD SWEEP — ALL STRATEGIES", st["section"]))
        story.append(Spacer(1, 1.5 * mm))
        story.append(HRFlowable(width=INNER_W, thickness=0.5, color=C_BORDER))
        story.append(Spacer(1, 2 * mm))

        headers = ["δ", "UI", "SEI", "CSI", "HI", "ENS"]
        hdr_style = ParagraphStyle("th", parent=st["small"],
                                   fontName="Helvetica-Bold", textColor=C_MUTED)
        tbl_rows  = [[Paragraph(h, hdr_style) for h in headers]]

        for tkey in available_thresholds:
            td     = preds_map[tkey]
            tp     = td.get("predictions", {})
            te     = td.get("ensemble", {})
            row = [Paragraph(f"{float(tkey):.2f}", st["small"])]
            for sid in ["s1", "s2", "s3", "s4"]:
                sp = tp.get(sid)
                if sp:
                    v = "\u2713" if sp["predicted"] else "\u00d7"
                    c = C_GREEN if sp["predicted"] else C_RED
                    row.append(Paragraph(f'<font color="#{c.hexval()[2:]}">'
                                         f'<b>{v}</b></font> {sp["abs_delta"]:.3f}',
                                         st["small"]))
                else:
                    row.append(Paragraph("—", st["small"]))
            ep = te.get("predicted", False)
            ev = "\u2713" if ep else "\u00d7"
            ec = C_GREEN if ep else C_RED
            row.append(Paragraph(f'<font color="#{ec.hexval()[2:]}"><b>{ev}</b></font>',
                                  st["small"]))
            tbl_rows.append(row)

        sweep_table = Table(tbl_rows,
                            colWidths=[INNER_W * 0.10] + [INNER_W * 0.18] * 4 + [INNER_W * 0.10])
        sweep_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  C_LABEL_BG),
            ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(sweep_table)

    doc.build(story)
    log.info("Report written → %s", output_path)


# ── Batch runner ──────────────────────────────────────────────────────────────

def run(pred_path: Path, bas_path: Path, output_dir: Path,
        filter_thread: str = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = _index_by_thread(_load_jsonl(pred_path))
    bass  = _index_by_thread(_load_jsonl(bas_path))

    common = set(preds) & set(bass)
    if filter_thread:
        common = {filter_thread} if filter_thread in common else set()
        if not common:
            log.error("thread_id %r not found in both files", filter_thread)
            return

    log.info("%d conversations to report", len(common))
    for tid in sorted(common):
        safe = tid.replace("/", "_").replace("\\", "_")
        out  = output_dir / f"{safe}.pdf"
        try:
            build_report(tid, preds[tid], bass[tid], out)
        except Exception as exc:
            log.error("Failed for %s: %s", tid, exc, exc_info=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PDF discussion reports from NARS pipeline output")
    parser.add_argument("--predictions", "-p", required=True,
                        help="predictions_*.jsonl from persuasiveness_detector.py")
    parser.add_argument("--bas",         "-b", required=True,
                        help="bas_repair*.jsonl or bas_no_repair*.jsonl from bas_assembler.py")
    parser.add_argument("--output-dir",  "-o", default="reports",
                        help="Directory to write PDFs into (default: reports/)")
    parser.add_argument("--thread-id",   "-t", default=None,
                        help="Generate report for a single thread_id only")
    parser.add_argument("--verbose",     "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(Path(args.predictions), Path(args.bas),
        Path(args.output_dir), args.thread_id)


if __name__ == "__main__":
    main()

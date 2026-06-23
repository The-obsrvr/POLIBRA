#!/bin/bash
set -e

# ── Quick run: 3 Italian interrogations through steps 1–6 ─────────────────────
# Prereqs: Ollama installed; sentence-transformers will download the two
# multilingual encoders from HuggingFace on first use:
#   paraphrase-multilingual-mpnet-base-v2   (Step 2 — PAC selection)
#   paraphrase-multilingual-MiniLM-L12-v2   (Steps 4/5 — repair + CSI/HI)

VERSION="interro_v1"
OUT="outputs/${VERSION}"
EDU_MODEL="qwen3.6:27b"
REASONING_MODEL="qwen3.6:27b"
INPUT="Data/interrogazioni_samples.jsonl"
OLLAMA="http://127.0.0.1:11436"

mkdir -p "${OUT}"
echo "Pipeline ${VERSION} — Italian interrogations → ${OUT}"
echo "Input : ${INPUT}"

export OLLAMA_KEEP_ALIVE=1h

start_ollama () {
    OLLAMA_HOST=${OLLAMA} ollama serve &
    OLLAMA_PID=$!
    until curl -sf ${OLLAMA}/api/tags > /dev/null 2>&1; do sleep 1; done
    if ! ollama list | grep -q "^$1"; then
        OLLAMA_HOST=${OLLAMA} ollama pull "$1"
    fi
    until curl -sf ${OLLAMA}/api/chat \
        -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
        | grep -q '"done":true'; do sleep 2; done
    echo "$1 ready."
}

stop_ollama () {
    kill "${OLLAMA_PID}" && wait "${OLLAMA_PID}" 2>/dev/null || true
}

# ── Step 1: EDU extraction (Italian-aware prompt) ─────────────────────────────
start_ollama "${EDU_MODEL}"
python src/extract_edu.py \
    --input      "${INPUT}" \
    --output     "${OUT}/edu.jsonl" \
    --ollama-url ${OLLAMA}/api/chat \
    --model      "${EDU_MODEL}" \
    --log-file   "${OUT}/edu.log"
stop_ollama
echo "Step 1 complete — GPU free."

# ── Step 2: PAC selection (multilingual mpnet, parent_id chain active) ────────
# k lowered 25 → 10: interrogations have ~40–90 EDUs over 3 turns, so k=25
# would select nearly every above-threshold candidate and bloat Step 3.
python src/pac_selector.py \
    --input     "${OUT}/edu.jsonl" \
    --output    "${OUT}/pac.jsonl" \
    --k         10 \
    --threshold 0.45 \
    --log-file  "${OUT}/pac.log"
echo "Step 2 complete."

# ── Step 2.1: PAC sanity check — similarity distribution + implicit rate ──────
# Use this to decide whether 0.45 needs re-tuning for the multilingual encoder.
python3 - "${OUT}/pac.jsonl" << 'PYEOF'
import json, sys
sims, scopes = [], {}
for line in open(sys.argv[1]):
    conv = json.loads(line)
    for turn in conv.get("conversation", []):
        for edu in turn.get("edus", []):
            for p in edu.get("pacs", []):
                sims.append(p["cosine_sim"])
                scopes[p["pac_scope"]] = scopes.get(p["pac_scope"], 0) + 1
if sims:
    sims.sort()
    n = len(sims)
    pct = lambda q: sims[int(q * (n - 1))]
    print(f"PAC similarity: n={n}  p10={pct(.1):.3f}  p50={pct(.5):.3f}  p90={pct(.9):.3f}")
    print(f"PAC scopes: {scopes}")
    impl = scopes.get("implicit", 0)
    if impl / max(n, 1) > 0.5:
        print("WARNING: >50% implicit PACs — threshold 0.45 is likely too high "
              "for the multilingual encoder; retry Step 2 with --threshold 0.35–0.40")
PYEOF

# ── Step 3: LLM relation reasoning (batch lowered for denser Italian tokens) ──
start_ollama "${REASONING_MODEL}"
python src/llm_reasoner.py \
    --input      "${OUT}/pac.jsonl" \
    --output     "${OUT}/reasoned.jsonl" \
    --ollama-url ${OLLAMA}/api/chat \
    --model      "${REASONING_MODEL}" \
    --no-think \
    --batch-size 6 \
    --log-file   "${OUT}/reasoned.log"
echo "Step 3 complete."

# ── Step 4: BAS construction (multilingual repair encoder, LLM root id) ───────
python src/bas_assembler.py \
    --input            "${OUT}/reasoned.jsonl" \
    --output-repair    "${OUT}/bas_repair.jsonl" \
    --output-no-repair "${OUT}/bas_no_repair.jsonl" \
    --log-file         "${OUT}/bas.log"
stop_ollama
echo "Step 4 complete — GPU free."

# ── Step 5: Strength initialization (UI, SEI, CSI, HI — multilingual MiniLM) ──
python src/strength_initializer.py \
    --input-repair     "${OUT}/bas_repair.jsonl" \
    --input-no-repair  "${OUT}/bas_no_repair.jsonl" \
    --output-repair    "${OUT}/initialized_repair.jsonl" \
    --output-no-repair "${OUT}/initialized_no_repair.jsonl" \
    --strategy all \
    --log-file "${OUT}/strength_init.log"
echo "Step 5 complete."

# ── Step 6: Gradual semantics ─────────────────────────────────────────────────
python src/gradual_semantics.py \
    --input-repair     "${OUT}/initialized_repair.jsonl" \
    --input-no-repair  "${OUT}/initialized_no_repair.jsonl" \
    --output-repair    "${OUT}/semantics_repair.jsonl" \
    --output-no-repair "${OUT}/semantics_no_repair.jsonl" \
    --strategy all \
    --lambda 0.5 --tau 0.3 --tau-agg 1.0 --beta 0.1 --max-iter 1000 \
    --log-file "${OUT}/gradual_semantics.log"
echo "Step 6 complete."

# ── Step 6.1: Root strength summary — what did the run say? ───────────────────
python3 - "${OUT}/semantics_repair.jsonl" << 'PYEOF'
import json, sys
print(f"\n{'thread_id':<14} {'strategy':<6} {'init':>7} {'final':>7} {'|Δ|':>7}")
for line in open(sys.argv[1]):
    sem = json.loads(line)
    root_id = sem.get("summary", {}).get("root_id")
    root = next((n for n in sem.get("nodes", []) if n["id"] == root_id), None)
    if not root:
        continue
    for s in ["s1", "s2", "s3", "s4"]:
        ik, fk = f"initial_strength_{s}", f"acceptability_{s}"
        if ik in root and fk in root:
            d = abs(root[fk] - root[ik])
            print(f"{sem.get('thread_id',''):<14} {s:<6} {root[ik]:>7.4f} {root[fk]:>7.4f} {d:>7.4f}")
    print(f"  root: {root.get('text','')[:90]}")
PYEOF

# ── Step 7 (optional — no ground truth yet) ───────────────────────────────────
# Interrogations have no is_delta. Once is_satisfied labels are populated, run:
# python src1/persuasiveness_detector.py \
#     --input-repair "${OUT}/semantics_repair.jsonl" \
#     --output-repair "${OUT}/predictions_repair.jsonl" \
#     --threshold 0.1 0.3 0.5 \
#     --ground-truth-key is_satisfied \
#     --log-file "${OUT}/persuasiveness.log"

echo "Pipeline ${VERSION} complete → ${OUT}"

#!/bin/bash
set -euo pipefail

# running on gpu 0

# ══════════════════════════════════════════════════════════════════════════════
# run.sh — Steps 1–4 (EDU → PAC → reasoning → BAS) on a selectable range
#
# Set the discussion range below (or pass it as args: ./run.sh START END).
# Range is 1-based and INCLUSIVE — line N of the JSONL = discussion N.
# Use non-overlapping ranges across parallel runs, e.g.:
#     ./run.sh 1   250        # first half   (e.g. on GPU 0)
#     ./run.sh 251 500        # second half  (e.g. on GPU 1)
#     ./run.sh 1   10         # quick smoke test
#
# GPU / Ollama placement is yours to manage. Point OLLAMA_URL at whichever
# server (and therefore GPU) you want this run to use; run this script in
# separate shells with different OLLAMA_URL / CUDA_VISIBLE_DEVICES per range.
#
# All outputs are suffixed with the range (e.g. bas_repair_251-500.jsonl) so
# concurrent runs over different ranges never clobber each other.
# ══════════════════════════════════════════════════════════════════════════════

# ── Range to process (edit here, or override via args) ────────────────────────
START="${1:-11}"
END="${2:-250}"

# ── Configuration (override any via environment) ──────────────────────────────
SRC="${SRC:-src}"                                          # dir with the .py steps
INPUT="${INPUT:-Data/samples.jsonl}"                       # 500-discussion JSONL
OUT="${OUT:-outputs/v21}"
CONFIG="${CONFIG:-configs/social_cmv.json}"              # unified use-case profile

EDU_MODEL="${EDU_MODEL:-qwen3.6:27b}"
REASONING_MODEL="${REASONING_MODEL:-qwen3.6:27b}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11435/api/chat}"
BATCH_SIZE="${BATCH_SIZE:-10}"

TAG="${START}-${END}"
mkdir -p "${OUT}"

echo "Range    : discussions ${TAG} (inclusive)"
echo "Input    : ${INPUT}"
echo "Config   : ${CONFIG}"
echo "Ollama   : ${OLLAMA_URL}"
echo "Output   : ${OUT}  (files suffixed _${TAG})"

# ── Slice the input to the selected range ─────────────────────────────────────
SHARD="${OUT}/samples_${TAG}.jsonl"
sed -n "${START},${END}p" "${INPUT}" > "${SHARD}"
echo "Selected $(wc -l < "${SHARD}") discussions → ${SHARD}"

# ── Step 1: EDU extraction (Ollama) ───────────────────────────────────────────
#OLLAMA_HOST=${OLLAMA_URL} ollama serve &
#OLLAMA_PID=$!
#until curl -sf http://127.0.0.1:11435/api/tags > /dev/null 2>&1; do sleep 1; done
#
#if ! ollama list | grep -q "^${EDU_MODEL}"; then
#    OLLAMA_HOST=http://127.0.0.1:11435 ollama pull "${EDU_MODEL}"
#fi
#
#until curl -sf http://127.0.0.1:11435/api/chat \
#    -d "{\"model\":\"${EDU_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
#    | grep -q '"done":true'; do sleep 2; done
#echo "${EDU_MODEL} ready."
#
#echo "--- Step 1: EDU extraction ---"
#python "${SRC}/extract_edu.py" \
#    --input      "${SHARD}" \
#    --output     "${OUT}/edu_${TAG}.jsonl" \
#    --configs     "${CONFIG}" \
#    --ollama-url "${OLLAMA_URL}" \
#    --model      "${EDU_MODEL}" \
#    --log-file   "${OUT}/edu_${TAG}.log"
#
#kill "${OLLAMA_PID}" && wait "${OLLAMA_PID}" 2>/dev/null || true
#echo "Step 1 complete — GPU free."
#
## ── Step 2: PAC selection (sentence-transformers) ─────────────────────────────
#echo "--- Step 2: PAC selection ---"
#python "${SRC}/pac_selector.py" \
#    --input    "${OUT}/edu_${TAG}.jsonl" \
#    --output   "${OUT}/pac_${TAG}.jsonl" \
#    --configs   "${CONFIG}" \
#    --log-file "${OUT}/pac_${TAG}.log"

# ── Step 3: LLM reasoning (Ollama) ────────────────────────────────────────────
echo "--- Step 3: LLM reasoning ---"

OLLAMA_HOST=${OLLAMA_URL} ollama serve &
OLLAMA_PID=$!
until curl -sf http://127.0.0.1:11435/api/tags > /dev/null 2>&1; do sleep 1; done

if ! ollama list | grep -q "^${REASONING_MODEL}"; then
    OLLAMA_HOST=http://127.0.0.1:11435 ollama pull "${REASONING_MODEL}"
fi

until curl -sf http://127.0.0.1:11435/api/chat \
    -d "{\"model\":\"${REASONING_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
    | grep -q '"done":true'; do sleep 2; done
echo "${REASONING_MODEL} ready."

python "${SRC}/llm_reasoner.py" \
    --input      "${OUT}/pac_${TAG}.jsonl" \
    --output     "${OUT}/reasoned_${TAG}.jsonl" \
    --configs     "${CONFIG}" \
    --ollama-url "${OLLAMA_URL}" \
    --model      "${REASONING_MODEL}" \
    --no-think \
    --batch-size "${BATCH_SIZE}" \
    --log-file   "${OUT}/reasoned_${TAG}1.log"

kill "${OLLAMA_PID}" && wait "${OLLAMA_PID}" 2>/dev/null || true
echo "Step 3 complete — GPU free."

# ── Step 4: BAS construction ──────────────────────────────────────────────────
echo "--- Step 4: BAS construction ---"
python "${SRC}/bas_assembler.py" \
    --input            "${OUT}/reasoned_${TAG}.jsonl" \
    --output-repair    "${OUT}/bas_repair_${TAG}.jsonl" \
    --output-no-repair "${OUT}/bas_no_repair_${TAG}.jsonl" \
    --configs           "${CONFIG}" \
    --log-file         "${OUT}/bas_${TAG}.log"

echo "Done ${TAG} → ${OUT}"
echo "  repair    : ${OUT}/bas_repair_${TAG}.jsonl"
echo "  no-repair : ${OUT}/bas_no_repair_${TAG}.jsonl"
# Merge ranges afterward with, e.g.:
#   cat ${OUT}/bas_repair_*.jsonl > ${OUT}/bas_repair.jsonl
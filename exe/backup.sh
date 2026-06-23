#!/bin/bash
set -e

# ── Configuration ─────────────────────────────────────────────────────────────
VERSION="v127"
BATCH="test"
OUT="outputs/${VERSION}"
EDU_MODEL="qwen3.6:27b"
REASONING_MODEL="qwen3.6:27b"

mkdir -p "${OUT}"
echo "Pipeline ${VERSION} — batch ${BATCH} → ${OUT}"

export OLLAMA_KEEP_ALIVE=1h

# ── Step 1: EDU extraction ────────────────────────────────────────────────────
#OLLAMA_HOST=http://127.0.0.1:11434 ollama serve &
#OLLAMA_PID=$!
#until curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do sleep 1; done
#
#if ! curl -sf http://127.0.0.1:11434/api/tags | grep -q "\"${EDU_MODEL}\""; then
#    OLLAMA_HOST=http://127.0.0.1:11434 ollama pull "${EDU_MODEL}"
#fi
#
#until curl -sf http://127.0.0.1:11434/api/chat \
#    -d "{\"model\":\"${EDU_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
#    | grep -q '"done":true'; do sleep 2; done
#echo "${EDU_MODEL} ready."
#
#python src1/extract_edu.py \
#    --input      "Data/samples_test.jsonl" \
#    --output     "${OUT}/edu_test.jsonl" \
#    --ollama-url http://127.0.0.1:11434/api/chat \
#    --model      "${EDU_MODEL}" \
#    --log-file   "${OUT}/edu_test.log"
#
#kill "${OLLAMA_PID}" && wait "${OLLAMA_PID}" 2>/dev/null || true
#echo "Step 1 complete — GPU free."
#
## ── Step 2: PAC selection ─────────────────────────────────────────────────────
#python src1/pac_selector.py \
#    --input    "${OUT}/edu_test.jsonl" \
#    --output   "${OUT}/pac_test.jsonl" \
#    --log-file "${OUT}/pac_test.log"
#echo "Step 2 complete — GPU free."


# ── Step 3: LLM reasoning — three parallel servers ───────────────────────────
OLLAMA_HOST=http://127.0.0.1:11434 ollama serve &
PID0=$!
OLLAMA_HOST=http://127.0.0.1:11435 ollama serve &
PID1=$!

until curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do sleep 1; done
until curl -sf http://127.0.0.1:11435/api/tags > /dev/null 2>&1; do sleep 1; done

if ! curl -sf http://127.0.0.1:11434/api/tags | grep -q "\"${REASONING_MODEL}\""; then
    OLLAMA_HOST=http://127.0.0.1:11434 ollama pull "${REASONING_MODEL}"
fi

until curl -sf http://127.0.0.1:11434/api/chat \
    -d "{\"model\":\"${REASONING_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
    | grep -q '"done":true'; do sleep 2; done &
WP0=$!
until curl -sf http://127.0.0.1:11435/api/chat \
    -d "{\"model\":\"${REASONING_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
    | grep -q '"done":true'; do sleep 2; done &
WP1=$!

wait $WP0 $WP1
echo "${REASONING_MODEL} ready on all two servers."

# Split PAC output into three shards
python3 - "outputs/v23/pac_test.jsonl" << 'PYEOF'
import sys, pathlib
lines  = [l for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
shards = [[], []]
for i, line in enumerate(lines):
    shards[i % 2].append(line)
for idx, shard in enumerate(shards):
    pathlib.Path(f"pac_shard_{idx}.jsonl").write_text("\n".join(shard) + "\n")
    print(f"pac_shard_{idx}: {len(shard)} conversations")
PYEOF

python src/llm_reasoner.py \
    --input           pac_shard_0.jsonl \
    --output          "${OUT}/reasoned_shard_0.jsonl" \
    --ollama-url      http://127.0.0.1:11434/api/chat \
    --model           "${REASONING_MODEL}" \
    --thinking-budget medium \
    --batch-size 10 \
    --log-file        "${OUT}/reasoned_shard_0.log" &
RPID0=$!
python src/llm_reasoner.py \
    --input           pac_shard_1.jsonl \
    --output          "${OUT}/reasoned_shard_1.jsonl" \
    --ollama-url      http://127.0.0.1:11435/api/chat \
    --model           "${REASONING_MODEL}" \
    --thinking-budget medium \
    --batch-size 10 \
    --log-file        "${OUT}/reasoned_shard_1.log" &
RPID1=$!

wait $RPID0 || { echo "ERROR: reasoning shard 0 failed"; exit 1; }
echo "Shard 0 complete."
wait $RPID1 || { echo "ERROR: reasoning shard 1 failed"; exit 1; }
echo "Shard 1 complete."

cat "${OUT}/reasoned_shard_0.jsonl" \
    "${OUT}/reasoned_shard_1.jsonl" \
    > "${OUT}/reasoned_test.jsonl"
echo "Reasoning merged → ${OUT}/reasoned_test.jsonl"

kill $PID0 $PID1 && wait $PID0 $PID1 2>/dev/null || true
rm -f pac_shard_0.jsonl pac_shard_1.jsonl
echo "Step 3 complete — GPU free."

# ── Step 4: BAS construction ──────────────────────────────────────────────────
python src/bas_assembler.py \
    --input            "${OUT}/reasoned_test.jsonl" \
    --output-repair    "${OUT}/bas_repair_test.jsonl" \
    --output-no-repair "${OUT}/bas_no_repair_test.jsonl" \
    --log-file         "${OUT}/bas_test.log"

echo "Pipeline ${VERSION} complete → ${OUT}"

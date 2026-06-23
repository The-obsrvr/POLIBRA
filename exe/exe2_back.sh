#!/bin/bash
set -e

VERSION="v2"
BATCH="nothink"
OUT="outputs/${VERSION}"
EDU_MODEL="qwen3.6:27b"
REASONING_MODEL="qwen3.6:27b"
INPUT="Data/samples.jsonl"
SKIP_FILE="Data/samples_test.jsonl"

mkdir -p "${OUT}"
echo "Pipeline ${VERSION} — batch ${BATCH} → ${OUT}"
echo "Input : ${INPUT}"
echo "Skipping conversations already in: ${SKIP_FILE}"

export OLLAMA_KEEP_ALIVE=1h

# ── Pre-filter: remove conversations already in samples_test.jsonl ────────────
echo "Filtering input — removing conversations already being processed..."
python3 - "${OUT}" "${INPUT}" "${SKIP_FILE}" << 'PYEOF'
import json, pathlib, sys

out_dir    = pathlib.Path(sys.argv[1])
input_path = pathlib.Path(sys.argv[2])
skip_path  = pathlib.Path(sys.argv[3])
out_path   = out_dir / "samples_filtered.jsonl"

# Collect thread_ids already being processed
skip_ids = set()
if skip_path.exists():
    for line in skip_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            tid = obj.get("thread_id") or obj.get("id")
            if tid:
                skip_ids.add(tid)
        except json.JSONDecodeError:
            continue
    print(f"Skipping {len(skip_ids)} conversations from {skip_path}")

kept = []
skipped = 0
for line in input_path.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        tid = obj.get("thread_id") or obj.get("id")
        if tid in skip_ids:
            skipped += 1
        else:
            kept.append(line)
    except json.JSONDecodeError:
        kept.append(line)

out_path.write_text("\n".join(kept) + "\n")
print(f"Kept {len(kept)} conversations, skipped {skipped} duplicates → {out_path}")
PYEOF

FILTERED="${OUT}/samples_filtered.jsonl"

# ── Step 1: EDU extraction ────────────────────────────────────────────────────
OLLAMA_HOST=http://127.0.0.1:11436 ollama serve &
OLLAMA_PID=$!
until curl -sf http://127.0.0.1:11436/api/tags > /dev/null 2>&1; do sleep 1; done

if ! ollama list | grep -q "^${EDU_MODEL}"; then
    OLLAMA_HOST=http://127.0.0.1:11436 ollama pull "${EDU_MODEL}"
fi

until curl -sf http://127.0.0.1:11436/api/chat \
    -d "{\"model\":\"${EDU_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
    | grep -q '"done":true'; do sleep 2; done
echo "${EDU_MODEL} ready."

python src/extract_edu.py \
    --input      "${FILTERED}" \
    --output     "${OUT}/edu.jsonl" \
    --ollama-url http://127.0.0.1:11436/api/chat \
    --model      "${EDU_MODEL}" \
    --log-file   "${OUT}/edu.log"

kill "${OLLAMA_PID}" && wait "${OLLAMA_PID}" 2>/dev/null || true
echo "Step 1 complete — GPU free."

# ── Step 2: PAC selection ─────────────────────────────────────────────────────
python src/pac_selector.py \
    --input    "${OUT}/edu.jsonl" \
    --output   "${OUT}/pac.jsonl" \
    --log-file "${OUT}/pac.log"
echo "Step 2 complete — GPU free."

# ── Step 3: LLM reasoning — single Qwen3.6:27b, thinking disabled ────────────
# One instance only
OLLAMA_HOST=http://127.0.0.1:11436 ollama serve &
OLLAMA_PID=$!
until curl -sf http://127.0.0.1:11436/api/tags > /dev/null 2>&1; do sleep 1; done

if ! ollama list | grep -q "^${REASONING_MODEL}"; then
    OLLAMA_HOST=http://127.0.0.1:11436 ollama pull "${REASONING_MODEL}"
fi

until curl -sf http://127.0.0.1:11436/api/chat \
    -d "{\"model\":\"${REASONING_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
    | grep -q '"done":true'; do sleep 2; done
echo "${REASONING_MODEL} ready."

python src/llm_reasoner.py \
    --input      "${OUT}/pac.jsonl" \
    --output     "${OUT}/reasoned.jsonl" \
    --ollama-url http://127.0.0.1:11436/api/chat \
    --model      "${REASONING_MODEL}" \
    --no-think \
    --batch-size 10 \
    --log-file   "${OUT}/reasoned.log"

kill "${OLLAMA_PID}" && wait "${OLLAMA_PID}" 2>/dev/null || true
echo "Step 3 complete — GPU free."

# ── Step 4: BAS construction ──────────────────────────────────────────────────
python src/bas_assembler.py \
    --input            "${OUT}/reasoned.jsonl" \
    --output-repair    "${OUT}/bas_repair.jsonl" \
    --output-no-repair "${OUT}/bas_no_repair.jsonl" \
    --log-file         "${OUT}/bas.log"

echo "Pipeline ${VERSION} complete → ${OUT}"
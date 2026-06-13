# POLIBRA: Balancing POLItical Opinions via neuro-symbolic Reasoning and Argumentation

This repository contains documentation on a project that forms part of a broader PhD research effort titled "Identifying the Stance of Argumentative Opinions in Political Discourse", conducted under the HYBRIDS Project within the Horizon Europe framework.

The primary contributor and point of contact for this repository is Siddharth Bhargava (sbhargava@fbk.eu).

---

## Summary

**POLIBRA** (Balancing POLItical opinions via neuro-symbolic Reasoning and Argumentation) is an open-source, modular neuro-symbolic framework that transforms raw discussions into **Bipolar Argument Structures** (BAS)—graph-based representations of supporting and attacking arguments.

The framework combines neural argument mining with symbolic reasoning. First, it extracts argumentative structures from long-form discussions through retrieval-augmented, iterative LLM reasoning. It then applies a symbolic evaluation component that propagates support and attack influences gradually to assess the relative strength of competing arguments.

Designed for complex and lengthy discussions, POLIBRA addresses noisy, implicit, and context-dependent discourse while providing **transparent and explainable reasoning** for persuasiveness, argumentative effectiveness, and political strategy analysis.

Key Features
- A _neural argument mining component_ that performs discourse unit segmentation, retrieval-augmented candidate pair selection, relation prediction through iterative LLM-based reasoning, and automatic Bipolar Argument Structure (BAS) construction. This component enables large-scale annotation and argument structure prediction for multi-threaded, multi-party discussions.
- A _symbolic reasoning component_ that quantifies initial argument strengths using semantic and structural characteristics, and evaluates relative argument influence through gradual strength propagation based on argumentation semantics tailored for long, sparsely-connected argument chains.
- _Downstream analytical capabilities_ supporting tasks such as persuasiveness detection (i.e., whether argumentation leads to view change) and explainable summarization through the identification of the most and least influential supporting and attacking arguments within a discussion.

---

## Data

Refer to this repository, [CMV2BAS Pipeline](https://github.com/The-obsrvr/CMV-BAS-Data-Pipeline), to access our data preparation and CMV2BAS dataset containing discussion threads derived from the public Webis ChangeMyView 2020 Corpus.

Alternatively, the system accepts .JSONL input files containing the following JSON format:

```json
{
        "thread_id": "t3_69cxuj_3",
        "conv_id":   "t3_69cxuj",
        "title":     "CMV: U.S. healthcare system",
        "conversation": [
            {
                "post_id":    "t3_69cxuj",
                "speaker_id": "Speaker 1",
                "text": (
                    "."
                ),
            },
            {
                "post_id":    "dh5mxav",
                "speaker_id": "Speaker 7",
                "text": (
                    "T"
                ),
            },
        ],
    }

```


---

## Methodology

- An end-to-end framework for transforming long-form discussions into argumentative structures that support logical reasoning, knowledge discovery, and downstream analytical tasks.
- A modular and extensible architecture with support for version control, logging, visualization, intermediate output inspection, and system customization.
- A scalable platform that accommodates diverse argumentation schemas, reasoning frameworks, and argument strength initialization strategies for domain-specific applications.

Our framework contains a neural component comprising of four steps, a symbolic component comprising of two steps and then downstream analytical tasks such as summary report generation and persuasiveness detection.

---

## System Implementation

The system requires access to a GPU, ideally having sufficient memory to support a medium LLM (<30B) and long context (up to 16834 max tokens). The system is designed to work with smaller LLMs and context spaces as well but this may result in a significantly longer processing time, especially for longer tasks. Note only the neural component primarily requires access to the GPU processing. 

### Building the System Environment

Our system accesses the open-source LLMs through the Ollama Client. Consequently, it will be required to be installed. To streamline installing all the dependencies and python packages, we share our DockerFile image. It can be built by running the following command:


```shell
$ docker build --rm -t {YOUR_CONTAINER_NAME} .
```

### System Execution

To run our system end-to-end, execute the following command:

```shell 
$ docker run  --gpus='"device={DEVICE NUMBER(S)}"' --runtime=nvidia --rm -ti --shm-size=32gb -v $PWD:/app {YOUR_CONTAINER_IMAGE} ./POLIBRA_exe.sh
```

The above command runs our ```POLIBRA_exe.sh``` that contains our end-to-end framework, described below completely with all the command-line arguments possible for each step. The steps are sequential, individually producing intermediate outputs and logging to enable increased inspection and system customization. The pipeline could be paused and resumed from any step, given its prior steps have been implemented and the file pathways are correct. 

```shell
#!/bin/bash
set -e

# ── Configuration ─────────────────────────────────────────────────────────────
VERSION="v1"
BATCH="cmv2bas"
OUT="outputs/${VERSION}"
EDU_MODEL="qwen3.6:27b"
REASONING_MODEL="qwen3.6:27b"

mkdir -p "${OUT}"
echo "Pipeline ${VERSION} — batch ${BATCH} → ${OUT}"

export OLLAMA_KEEP_ALIVE=1h

# ── Step 1: EDU extraction ────────────────────────────────────────────────────
OLLAMA_HOST=http://127.0.0.1:11434 ollama serve &
OLLAMA_PID=$!
until curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do sleep 1; done

if ! curl -sf http://127.0.0.1:11434/api/tags | grep -q "\"${EDU_MODEL}\""; then
    OLLAMA_HOST=http://127.0.0.1:11434 ollama pull "${EDU_MODEL}"
fi

until curl -sf http://127.0.0.1:11434/api/chat \
    -d "{\"model\":\"${EDU_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
    | grep -q '"done":true'; do sleep 2; done
echo "${EDU_MODEL} ready."

python src/extract_edu.py \
    --input      "Data/samples.jsonl" \
    --output     "${OUT}/edu.jsonl" \
    --ollama-url http://127.0.0.1:11434/api/chat \
    --model      "${EDU_MODEL}" \
    --log-file   "${OUT}/edu.log"

kill "${OLLAMA_PID}" && wait "${OLLAMA_PID}" 2>/dev/null || true
echo "Step 1 complete — GPU free."


## ── Step 2: PAC selection ─────────────────────────────────────────────────────
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


# ── Step 5: Strength initialization ──────────────────────────────────────────
# Runs all four strategies: UI (s1), SEI (s2), CSI (s3), HI (s4).
echo "--- Step 5: Strength initialization ---"
python src/strength_initializer.py \
    --input-repair    "${OUT}/bas_repair.jsonl" \
    --input-no-repair "${OUT}/bas_no_repair.jsonl" \
    --output-repair   "${OUT}/initialized_repair.jsonl" \
    --output-no-repair "${OUT}/initialized_no_repair.jsonl" \
    --strategy all \
    --log-file "${OUT}/strength_init.log"
echo "Step 5 complete."
#
## ── Step 6: Gradual semantics ─────────────────────────────────────────────────
## Iterative strength propagation with softmax-weighted aggregation.
echo "--- Step 6: Gradual semantics ---"
python src/gradual_semantics.py \
    --input-repair    "${OUT}/initialized_repair_.jsonl" \
    --input-no-repair "${OUT}/initialized_no_repair.jsonl" \
    --output-repair   "${OUT}/semantics_repair.jsonl" \
    --output-no-repair "${OUT}/semantics_no_repair.jsonl" \
    --strategy all \
    --lambda 0.5 \
    --tau 0.3 \
    --tau-agg 1.0 \
    --beta 0.1 \
    --max-iter 1000 \
    --log-file "${OUT}/gradual_semantics.log"
echo "Step 6 complete."
#
## ── Step 7: Persuasiveness detection ─────────────────────────────────────────
## Sweep δ thresholds 0.1 → 0.9; tune-threshold finds the best δ per strategy.
echo "--- Step 7: Persuasiveness detection ---"
python src/persuasiveness_detector.py \
    --input-repair    "${OUT}/semantics_repair.jsonl" \
    --input-no-repair "${OUT}/semantics_no_repair.jsonl" \
    --output-repair   "${OUT}/predictions_repair.jsonl" \
    --output-no-repair "${OUT}/predictions_no_repair.jsonl" \
    --threshold 0.1 0.3 0.5 0.7 0.9 \
    --ground-truth-key is_delta \
    --tune-threshold \
    --tune-metric f1 \
    --tune-strategy ensemble \
    --log-file "${OUT}/persuasiveness.log"
echo "Step 7 complete."
#
## ── Evaluation ────────────────────────────────────────────────────────────────
echo "--- Evaluation ---"
python src/evaluate.py \
    --input-repair    "${OUT}/predictions_repair.jsonl" \
    --input-no-repair "${OUT}/predictions_no_repair.jsonl" \
    --output          "${OUT}/evaluation.json"
echo "Evaluation complete → ${OUT}/evaluation.json"

# ── Report generation ─────────────────────────────────────────────────────────
# One PDF per conversation, using repair-mode BAS + predictions.
echo "--- Report generation ---"
python src/nars_report.py \
    --predictions "${OUT}/predictions_repair.jsonl" \
    --bas         "${OUT}/bas_repair.jsonl" \
    --output-dir  "${OUT}/reports_summary"
echo "Reports written → ${OUT}/reports_summary/"

echo "Pipeline ${VERSION} complete → ${OUT}"

```




---

## System Use-case and Limitations

---

## Acknowledgements 

This research work has received funding from the European Union's Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101073351. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.

---

## Citation

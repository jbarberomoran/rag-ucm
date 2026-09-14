# Balanced-dataset benchmark

The [completed 350-observation run](qwen-balanced-benchmark.md) includes verified
results, limitations and provenance hashes.

## Dataset correction

The original answer positions were A=0, B=5, C=64, D=1: always answering C
achieved 91.43% without reading a question. Now A=18, B=18, C=17, D=17.
The majority-letter control is 25.71%; uniform guessing has expected accuracy
25%. These are analytical controls, not additional model requests.

Only options and the corresponding answer letter were permuted. Question text,
answer content, correct answer meaning and references are preserved. The seed-2026
transformation is src/dataset_balance.py; its audit records reversible mappings
and original file/content hashes. Tests reconstruct the source and check its
canonical hash. Position-dependent options are rejected for manual review.

This fixes positional imbalance, not label truth, distractor quality, leakage,
duplicated concepts or generalization beyond one paper. Do not mix historical
results or the original smoke test with new results.

## Reproduce

Use Python 3.11, the platform lockfile from README, and Ollama with
qwen3:4b-instruct. Initial model/tokenizer downloads require network access.

```bash
python main.py --name qwen-balanced-full --runs 1 --sleep 0
# Add --rebuild-db only if the generated index is incompatible.
# Retry failures with unchanged source, inputs and environment:
python main.py --name qwen-balanced-full --runs 1 --sleep 0 --resume
```

This is 70 questions × five methods = 350 local requests. Retrieval supplies
at most five whole chunks within 2,048 tokens; oversized chunks are skipped.
Model context window: 32,768; output limit: 32; temperature: 0; seed: 2026.
Prompt plus output must fit the window. Qwen prompts use the pinned tokenizer
and Ollama's [raw generate API](https://docs.ollama.com/api/generate); server
input token counts must equal local counts, preventing silent truncation.

Inspect summary.json before plots: it includes all expected observations,
failures, both accuracy denominators and controls. Paired statistics report
exclusions and exploratory question-cluster intervals. One pass does not
characterize run-to-run variation. No confirmatory comparison was preregistered.
Timing is end-to-end on a shared machine, not an isolated hardware benchmark.
CSV, manifest and plots remain under results/persistent_results/qwen-balanced-full/
and are excluded from Git.

## Human relevance review

```bash
python -m scripts.review_annotations prepare --packet results/relevance-review.json
```

The packet contains every question and the complete chunk catalog. Read the paper
and all chunks independently of predictions. Fill relevant_chunk_ids, reviewer,
notes and reviewed=true. Include all relevant chunks, not only one method's hits.
A second independent reviewer and adjudication are recommended for a gold standard.

```bash
python -m scripts.review_annotations export --packet results/relevance-review.json \
  --output results/relevance-labels.json
python main.py --name reviewed-benchmark --annotations results/relevance-labels.json
```

Export requires every question to be reviewed and attributed, validates IDs and
binds the packet to dataset/catalog hashes. Reviewer fields are attestations,
not proof of who did the review. Keep the packet alongside labels for provenance.
Reviewed questions with no positive chunks are omitted from positive-relevance
metrics: null is not zero. No human gold labels are supplied; the prepared local
packet intentionally starts unreviewed.

## Dependency locks

requirements-dev-macos.lock targets Apple Silicon macOS 14+ / Python 3.11;
requirements-dev-linux.lock targets Linux x86_64 / Python 3.11 with CPU PyTorch
and is used by CI (no multi-gigabyte CUDA runtime is needed for the tests).
They pin transitive versions using uv 0.12.13 and constraints from the tested
local environment. Linux-specific dependencies were resolved separately.
Other platforms and optional cloud/notebook extras are not locked or claimed
to be validated.

To refresh deliberately, run uv pip compile requirements-dev.txt with
--python-version 3.11, the appropriate --python-platform and --output-file.
Set MACOSX_DEPLOYMENT_TARGET=14.0 for macOS. Review changes and rerun unit and
integration tests. For Linux use --torch-backend cpu and retain the official
https://download.pytorch.org/whl/cpu extra index in the generated lock.
Add tools before starting a run:
resume rejects any change to the installed package environment.

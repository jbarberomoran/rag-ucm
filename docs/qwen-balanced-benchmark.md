# Full balanced Qwen benchmark — 2026-09-13

Completed one pass of all 70 questions across all five methods: **350/350 valid
observations, zero failed or missing observations**. This is a full-dataset run,
not the historical three-question smoke. No cloud API or Google account was used.

| Method | Correct | Accuracy | Mean end-to-end seconds |
|---|---:|---:|---:|
| baseline | 52/70 | 74.29% | 0.334 |
| bm25 | 69/70 | 98.57% | 2.693 |
| dense | 60/70 | 85.71% | 2.733 |
| hybrid | 64/70 | 91.43% | 2.862 |
| cross_encoder | 69/70 | 98.57% | 2.688 |

The majority-letter control is 18/70 = 25.71%; uniform random guessing has
expected accuracy 25%. Because no observations failed, conditional and
end-to-end accuracy agree. Machine-readable aggregates are in
[qwen-balanced-summary.csv](qwen-balanced-summary.csv).

BM25 and cross-encoder had identical correctness outcomes on all 70 questions.
Their paired difference and empirical question-bootstrap interval are 0 percentage
points. This degenerate empirical interval does **not** establish equivalence on
new questions or other papers. All pairwise analyses are exploratory; there was
no preregistered confirmatory comparison or multiple-comparison correction.

## Configuration and verification

- Qwen3 4B Instruct via Ollama 0.33.3; Apple Silicon, 24 GiB RAM.
- Python 3.11.16; temperature 0; seed 2026; output limit 32; model context 32,768.
- Semantic index rebuilt from 30 PDF pages into 80 chunks.
- Equal retrieval cap: five whole chunks / 2,048 tokens, separators included.
- All 350 saved contexts were checked: maximum 2,047 context tokens, maximum
  five chunks, zero empty RAG contexts; maximum full prompt 2,257 tokens.
- Local and server prompt token counts matched for every valid observation.
- Source hash still matched the manifest at completion.
- Resume returned successfully without adding observations; both attempt and
  final CSV hashes were unchanged. Four plots were generated and accuracy was
  visually checked.
- 93 offline unit tests passed, with 86.84% source coverage; real offline Chroma
  integration, Ruff and diff checks passed.
- [GitHub Actions validation](https://github.com/jbarberomoran/rag-ucm/actions/runs/34780903228)
  passed using the Linux CPU dependency lock.

Generation is stateless, but latency includes local processing and server caches.
Method order was fixed, the machine was shared, and unit tests overlapped with
part of the run. These times are operational observations, not a controlled
hardware or performance comparison. One deterministic pass does not characterize
run-to-run variability.

## Provenance

- Model digest: `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`
- Dataset SHA-256: `f54f8358558570c1a89b66e4a9476c191e83496148dcb185de6a0b85fa279a73`
- Paper SHA-256: `5970d37adcf424431a9dc1c9d541af35aa0b830c9bad29da5c1d2c5771bdb1f5`
- Evaluated Python-source SHA-256: `9738bd189d7b0a05d9cbf6840a8a432e7f0d2fa5e8741215144f41c9282eba34`
- Attempt CSV SHA-256: `f0b36280bc62ff43c83905efa71825f67c71193bf211bf97d7c86328ba864e46`
- Final CSV SHA-256: `d0b910f90fb69c4faf1507bdb890b8dbd962982ee24847250adc0f0253b2ffb0`

The run began from a working tree based on main commit
`ff00824a76f0c66d006ea9e600d69026455ba2b3`; the source hash above identifies the evaluated changes.
The complete manifest also records exact package versions, model revisions and
settings. Full local artifacts remain in
`results/persistent_results/qwen-balanced-full/` and are deliberately not committed.

See [benchmark validity](benchmark-validity.md) for the reproduction command,
dataset permutation audit and review workflow. Human relevance labels remain
**pending**: `results/relevance-review.json` is prepared but unreviewed.
Retrieval recall, precision and reciprocal-rank gold metrics are therefore not
claimed. Balancing correct-option positions does not independently validate
question truth, distractor quality or generalization beyond this paper.

# Evaluation protocol

The question dataset and option order are unchanged. Conclusions apply only to
this dataset and paper; the known option imbalance remains a limitation.

## Separate the measurements
Baseline uses its own internal-knowledge prompt. Retrieved-context methods use
the same context-grounded prompt and temperature.
Correctness is answer-key agreement. Reference overlap is lexical matching
(longest normalized matching span / reference length, threshold 0.5).
It is not semantic support, faithfulness, reasoning quality or evidence of luck.
The legacy retrieval_score column remains an alias for compatibility.

For human relevance evaluation, inspect data/chroma_db/chunks.json after indexing.
Create a JSON file mapping 1-based question IDs to relevant chunk IDs:
```json
{"1": ["actual-chunk-id-from-chunks.json"], "2": ["another-actual-chunk-id"]}
```
These example IDs are placeholders, not annotations. Annotate independently of
retrieval output and include all relevant chunks. Pass --annotations PATH.
Recall@k, precision@k and reciprocal rank are computed from returned unique chunk
IDs. Here k is the number of returned chunks: hybrid can return a union larger
than the per-retriever k. Missing annotations yield null metrics. Reannotate after
changing the corpus or chunking; chunk IDs refer to a particular index.
No human relevance labels have been fabricated or added to the question dataset.

## Paired analysis
Each result identifies run_id, question_id, method and any request error.
The final CSV keeps the latest attempt per combination. Paired analysis uses
successful observations shared by each method pair, reporting exclusions on
both sides. Failures remain in the final CSV; they must be reported alongside
conditional accuracy to avoid hiding availability differences.

The paired difference is averaged across runs within each question, then across
questions. A seeded question-cluster bootstrap supplies exploratory 95% percentile
intervals (2,000 resamples); repeated API calls are not independent questions.
No multiple-comparison correction is applied. Predeclare a primary comparison
for confirmatory use. With fewer than two paired questions, no interval is reported.
Latency and cost should be considered alongside accuracy. Do not select a winner
solely from a small difference in aggregate accuracy.

Historical summary counts are baseline/BM25/dense 821, hybrid 820 and cross-encoder
819. Their missing attempts and run alignment cannot be reconstructed from the
aggregate CSV alone. Historical outcomes predate these protocol corrections and
are descriptive, not validation of the updated code. In particular, 98.53% versus
97.32% does not establish superiority, while mean latency was 2.64 s versus 0.64 s.
New paired claims require new experiments with the corrected prompts.

## Storage and resume
Use a distinct --name for each configuration. Each directory holds manifest.json,
resultados_parciales.csv (attempt log), resultados_finales.csv and
paired_statistics.json. --resume (also --keep-existing for compatibility) verifies
the saved configuration and retries missing/failed observations, retaining successes.
Request failures record exception classes without potentially sensitive provider messages.
A second resume can retry persistent failures; there is no unbounded automatic retry.
Only one process may write a given experiment directory at a time.

Manifest records source/input hashes, code revision, Python/packages and settings.
Secrets are never included. Provider model aliases are not immutable revisions.
Index manifests detect changes in PDF hash, embedding model and chunking parameters;
unknown or incompatible indexes require explicit --rebuild-db.
Baseline-only runs do not build or load Chroma.

## Verification
Offline unit tests cover prompts, paired alignment, errors/resume and provenance.
The separate integration suite uses actual temporary Chroma storage and deterministic
local embeddings, without Gemini or model downloads:
```bash
python -m pytest
python -m pytest integration_tests
```
These checks verify software behavior; they do not replace live retrieval/model
evaluation or human relevance annotation.


# Local generation validation

Historical smoke: this predates dataset rebalancing and token-budget fixes.
It is retained for provenance, not as validation of the current benchmark.

The default generation path was exercised with Qwen3 4B Instruct on an Apple
Silicon machine with 24 GiB RAM, without a Google API key or cloud generation.

- Ollama: 0.33.3
- Model: qwen3:4b-instruct
- Model digest: 0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0
- Temperature: 0; seed: 2026; context: 32768; output limit: 32; thinking: disabled.
- Source PDF: 30 pages, 80 semantic chunks.
- Questions: original indices 0, 1, 2; dataset and option ordering unchanged.
- Methods: baseline, BM25, dense, hybrid and hybrid plus cross-encoder.
- Result: 15 completed requests, zero request errors, 14 correct answers.
  Dense missed one answer; all other method/question combinations were correct.
- Final CSV, paired statistics and all four plots were generated successfully.

This is an end-to-end smoke test, not a representative benchmark or evidence that
one method is superior. Historical Gemini results are a separate experiment.
The initial 16K conservative context budget rejected two hybrid prompts; the
validated configuration uses 32K and preserves the entire prompt.

Reproduce after setup:
```bash
python -m scripts.check_environment --check-generation
python main.py --name qwen-local-validated --questions 0 1 2 \
  --methods baseline bm25 dense hybrid cross_encoder --sleep 0
```
If the directory already exists, choose another --name or use --resume with
the same configuration. Model downloads and generated results stay outside Git.

Automated checks: 79 offline unit tests (86.46% src coverage), one real Chroma
storage integration test, and Ruff. Provider transport is mocked in unit tests;
the above smoke test used the real local server and model.

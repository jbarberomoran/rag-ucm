# Two-Stage Retrieval RAG

University research project developed with Management Solutions and the Faculty
of Mathematics at Universidad Complutense de Madrid. It compares how retrieval
strategies affect a configurable language model answering multiple-choice questions about a
technical research paper.

## Compared methods

- `baseline`: the selected language model without retrieved context.
- `bm25`: keyword retrieval.
- `dense`: embedding similarity with Chroma.
- `hybrid`: weighted BM25 and dense retrieval.
- `cross_encoder`: hybrid candidate retrieval followed by reranking.

Answer correctness and retrieval evidence are measured separately. This avoids
confusing answer-key agreement with lexical reference overlap. Neither metric
alone establishes whether the model used the context. See [evaluation protocol](docs/evaluation.md).

## Repository layout

```text
.
├── data/                  # Source paper, assignment, and question dataset
├── docs/                  # Small, reviewable research summaries
├── scripts/               # Local diagnostics
├── src/
│   ├── config.py          # Paths and experiment defaults
│   ├── domain.py          # Pure scoring and serialization logic
│   ├── ingestion.py       # PDF loading, chunking, and vector-store creation
│   ├── retrieval.py       # BM25, dense, hybrid, and cross-encoder retrieval
│   ├── rag_pipeline.py    # Prompt construction and provider-independent generation
│   ├── generation.py      # Ollama default; optional Gemini adapter
│   ├── question_data.py   # Dataset validation and selection
│   ├── queries.py         # Experiment orchestration
│   └── evaluation.py      # Metrics and plots
├── tests/                 # Offline unit tests
├── main.py                # Command-line entry point
└── Two-Stage-Retrieval LLM RAG.ipynb
```

Generated vector data and experiment outputs are intentionally excluded from
Git. A compact summary of the previously published final experiment is kept in
`docs/final-results-summary.csv`.

## Requirements

- Python 3.11
- Ollama running locally, with `qwen3:4b-instruct` downloaded (default model)
- Internet access for initial Ollama and Hugging Face model downloads

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Install [Ollama](https://docs.ollama.com/quickstart), start its server in another
terminal and download the default [model](https://ollama.com/library/qwen3:4b-instruct):

```bash
ollama serve
# In another terminal:
ollama pull qwen3:4b-instruct
```

No Google account or API key is required. Generation runs on your machine.
Model weights occupy about 2.5 GB; working memory also depends on the context window.
`.env.example` documents `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL` and limits.
CLI flags override environment values. Keep provider and model names consistent.

Never commit `.env`. To use the notebook, install
`requirements-notebook.txt` instead of `requirements-dev.txt`.

Validate the local setup without making an API call:

```bash
python -m scripts.check_environment
```

Use `python -m scripts.check_environment --check-generation` to check the
installed model and make one request. `--check-api` remains a compatibility alias.

### Optional cloud generation

Install `requirements-gemini.txt`, configure `GOOGLE_API_KEY` in `.env`, then run
`python main.py --provider gemini --model YOUR_AVAILABLE_MODEL --name cloud-run`.
The optional adapter is loaded only when selected. Model availability and billing
are controlled by the provider. No automatic provider fallback mixes experiments.
Historical result summaries used Gemini and are not Qwen benchmarks.
See [local validation](docs/local-validation.md) for a real 15-request smoke test
using Qwen, including the exact model digest and reproduction command.

## Tests and quality checks

Unit tests do not use the network, API keys, Chroma, or downloaded ML models.

```bash
python -m pytest
python -m pytest --cov=src --cov-report=term-missing
python -m ruff check .
python -m pytest integration_tests
```

GitHub Actions runs linting and unit tests for pushes to `main` and for pull
requests.

## Run an experiment

Run one complete pass with all methods:

```bash
python main.py
```

Run a small smoke experiment over the first three questions:

```bash
python main.py --questions 0 1 2 --methods baseline bm25 --skip-plots
```

Repeat the experiment and keep it under a named result directory:

```bash
python main.py --name final-comparison --runs 10
python main.py --name final-comparison --runs 10 --resume
```

Use `python main.py --help` for every option. The default is one run; this
prevents an accidental invocation from making ten full batches of paid API
calls, which was the previous behavior.

Select another locally installed model with `--provider ollama --model MODEL`.
Manifests include provider, model digest/server version for Ollama and generation
limits; each result row also records provider and model. Resume rejects changes
to generation identity. Long prompts fail conservatively instead of silently
truncating retrieved evidence; adjust `--context-tokens` to fit your machine.

Existing experiment directories require `--resume` or a new `--name`.
`--keep-existing` is a compatibility alias for resume. Each experiment has its
own attempt log, manifest and paired statistics. `--chunking recursive` selects
recursive splitting; incompatible existing indexes require `--rebuild-db`.
Optional `--annotations PATH` enables metrics against human-labeled chunk IDs.
The notebook now analyzes saved runs or displays the historical summary when no
run exists, without requiring an API key. Historical results predate the fixes
and are not evidence that the corrected pipeline achieves the same accuracy.

## Data and reproducibility

- `data/questions.json` contains 70 validated multiple-choice questions.
- `data/chroma_db/` is regenerated locally and must not be committed.
- `results/` contains generated CSV and plot outputs and must not be committed.
- Retrieved documents are stored as JSON in result rows so the evidence remains
  machine-readable.
- The deterministic evidence judge uses normalized exact/fuzzy overlap. It does
  not make a second LLM call, avoiding circular and nondeterministic grading.

The included PDFs may have licensing terms separate from the MIT-licensed
source code. Verify redistribution rights before publishing copies elsewhere.

## Authors

- Jorge Barbero Morán — UCM, Faculty of Mathematics
- David Marcos Jimeno — UCM, Faculty of Mathematics

## License

The source code is available under the [MIT License](LICENSE).

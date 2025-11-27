# Two-Stage-Retrieval LLM RAG

This project has been developed following the guidelines of the "Modelización de problemas de la Empresa", proposed by Management Solutions and the Faculty of Mathematics at UCM. A basic understanding of LLMs and RAG is highly recommended.

It implements a question-answering (Q&A) system over a technical research paper from Meta SuperIntelligence Labs. The goal is to compare how different retrieval-augmented generation (RAG) setups help a large language model (LLM) answer multiple-choice questions.

The following retrieval pipelines are compared:

- **BM25** – Classic keyword search.

- **Dense Retrieval** – Embedding-based semantic retrieval.

- **LLM Baseline** – LLM without any retrieval.

- **(Bonus) Hybrid Retrieval** – Combination of BM25 and Dense Retrieval.

- **Hybrid Retrieval + Cross-Encoder** – Reranking of the chunks retrieved by hybrid retrieval.


---
## Features


The project includes:

- Splitting the paper into sensible chunks and storing them in a **vector database** (ChromaDB).

- Comparison between different recursive and semantic **chunk strategies**.

- Running multiple executions for statistically relevant results with **high reproducibility**.

- Evaluation of **accuracy** and source attribution.

- Generating a concise **dashboard** comparing the different pipelines.

 
---
## Project Structure

 The project is organized into multiple files under the src/ folder, with a clear separation of responsibilities. Here’s how it works:

1. **Input Data**

- **data/questions.json** - Contains 50 multiple-choice questions extracted from a technical research paper.
Each question includes: the correct answer, three distractors, and an optional reference to the source paper.

- This is the only input you need to provide for the experiment.


2. **Source Code**
- **main.py** - Entry point of the project.
Supports *Local* mode (`results/local_results/`) and *Persistent* mode (`results/persistent_results/<test_name>/`).  
  - Calls the pipeline, saves final results, and generates plots.

- **launcher.py** - Sets up the environment for experiments.
 - Initializes the vector database (ChromaDB).
 - Clears previous results if needed.
 - Creates all necessary directories (plots, final CSVs, etc.).

- **queries.py** - Main logic to execute questions.
 - For each question and method it:
  - Sends the query to the LLM and retrieves documents.
  - Computes accuracy and evidence verification.
  - Stores partial results in results/resultados_parciales.csv.
  - Returns a DataFrame with all results for further evaluation.

- **rag_pipeline.py** - Implements RAG logic for different retrieval methods
 - Contains functions to verify ground truth against retrieved documents.
 - Computes retrieval scores and status tags for each answer.

- **retrieval.py** - Implements the retrieval engine.
 - Provides a singleton engine to handle different retrieval methods efficiently.

- **evaluation.py** - Evaluates the results and generates dashboards.
 - `evaluate_results(df, final_file)` → prints accuracy and summary metrics.  
 - `generate_dashboard(dir_input, dir_output)` → generates plots:  
  - Accuracy per method  
  - RAG quality distribution  
  - Response latency  
  - Retrieval fidelity

3. **Output Data**

- **Partial results** - Always stored in `results/resultados_parciales.csv`.
 - Updated after each question is processed.

- **Final results** - Stored in `results/local_results/` or `results/persistent_results/<test_name>`.
 - File name: `resultados_finales.csv`.

- **Plots/Dashboard** - Stored in `plots/` inside the corresponding results folder.
 - Include:
  - Bar charts for accuracy and RAG quality
  - Boxplots for response latency
  - Violin plots for retrieval fidelity

**Tip:** Run `main.py try_n` on the terminal in orden to save the try number n results in that directory.
---
##  Workflow Summary

1. Load questions dataset (`questions.json`).

2. Initialize vector database (ChromaDB).

3. For each question:
 - Retrieve relevant documents (BM25 / Dense / Hybrid / Hybrid + Cross-Encoder).
 - Query LLM for an answer.
 - Verify against ground truth.
 - Save partial results.

4. After all questions are processed:
 - Concatenate all results.
 - Save final results CSV.
 - Generate plots/dashboard for comparison and analysis.



---
## Authors
- Jorge Barbero Morán – UCM, Faculty of Mathematics
- David Marcos Jimeno – UCM, Faculty of Mathematics

---
## License
This project is licensed under the **MIT License**
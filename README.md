This project has been fully developed under the guidelines of the "Modelización de problemas de la Empresa", proposed by Management Solutions and the Faculty of Mathematics from the UCM. 
 It implements a question-answering (Q&A) system over a technical research paper from Meta SuperIntelligence Labs. The goal is to compare how different retrieval-augmented generation (RAG) setups help a large language model (LLM) answer multiple-choice questions.

The following retrieval pipelines are compared:

- BM25 – Classic keyword search.

- Dense Retrieval – Embedding-based semantic retrieval.

- LLM Baseline – LLM without any retrieval.

- (Bonus) Hybrid Retrieval – Combination of BM25 and Dense Retrieval.

The project includes:

Splitting the paper into sensible chunks and storing them in a vector database (FAISS or ChromaDB).

Comparison between different chunk strategies.

Running multiple executions for statistically relevant results with high reproducibility.

Evaluating accuracy and source attribution.

Generating a concise dashboard (tables/plots) comparing the different pipelines.

 

 Proyect Structure & Workflow

 The project is organized into multiple files under the src/ folder, with a clear separation of responsibilities. Here’s how it works:


1. Input Data

- data/questions.json
Contains 50 multiple-choice questions extracted from a technical research paper.
Each question includes: the correct answer, three distractors, and an optional reference to the source paper.

- This is the only input you need to provide for the experiment.


2. Source Code
- main.py
Entry point of the project.
Handles execution modes:

 - Local mode: temporary results saved under results/local_results/.

 - Persistent mode: results saved under results/persistent_results/<test_name>/.

Calls the pipeline to run questions, save final results, and generate dashboards.
Reads the API key from .env if needed.

- src/launcher.py
Sets up the environment for experiments.
Initializes the vector database (FAISS or ChromaDB).
Clears previous results if clear_results=True.
Creates all necessary directories (plots, final CSVs, etc.).

- src/queries.py
Main logic to execute questions.
For each question and method:

 - Sends the query to the LLM and retrieves documents.

 - Computes accuracy and evidence verification.

 - Stores partial results in results/resultados_parciales.csv.

 - Returns a DataFrame with all results for further evaluation.

- src/rag_pipeline.py
Implements RAG logic for different retrieval methods:

 - BM25

 - Dense Retrieval

 - Hybrid (bonus)

Contains functions to verify ground truth against retrieved documents.
Computes retrieval scores and status tags for each answer.

- src/retrieval.py
Implements the retrieval engine.
Provides a singleton engine to handle different retrieval methods efficiently.

- src/evaluation.py
Evaluates the results and generates dashboards.

Key functions:

 - evaluate_results(df, final_file) → prints accuracy and summary metrics.

 - generate_dashboard(dir_input, dir_output) → generates plots:
    Accuracy per method
    RAG quality distribution
    Response latency
    Retrieval fidelity


3. Output Data

- Partial results:
Always stored in results/resultados_parciales.csv.
Updated after each question is processed.

- Final results:
Stored in results/local_results/ or results/persistent_results/<test_name>/.
File name: resultados_finales.csv.

- Plots / Dashboard:
Stored in plots/ folder inside the corresponding results folder.
Visualizations include:

 - Bar charts for accuracy and RAG quality

 - Boxplots for response latency

 - Violin plots for retrieval fidelity


4. Workflow Summary

- Load questions dataset (questions.json).

- Initialize vector database (FAISS/ChromaDB).

- For each question:
    Retrieve relevant documents (BM25 / Dense / Hybrid).
    Query LLM for an answer.
    Verify against ground truth.
    Save partial results.

- After all questions are processed:
    Concatenate all results.
    Save final results CSV.
    Generate plots/dashboard for comparison and analysis.

    ## Authors
- Jorge Barbero Morán – UCM, Faculty of Mathematics
- David Marcos Jimeno – UCM, Faculty of Mathematics

    ## License
This project is licensed under the MIT License
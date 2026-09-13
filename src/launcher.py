"""Prepare local storage for an experiment run."""

import shutil
from pathlib import Path

from src.config import RESULTS_DIR
from src.ingestion import db_setup


def setup_environment(
    rebuild_db: bool = False,
    clear_results: bool = True,
    results_dir: str | Path = RESULTS_DIR,
) -> None:
    """Create output directories, clear generated files, and prepare Chroma."""
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    partial_file = RESULTS_DIR / "resultados_parciales.csv"
    final_file = results_dir / "resultados_finales.csv"

    results_dir.mkdir(parents=True, exist_ok=True)
    if clear_results:
        final_file.unlink(missing_ok=True)
        partial_file.unlink(missing_ok=True)
        if plots_dir.exists():
            shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    db_setup(rebuild_db)


# Preserve the original public name for notebook compatibility.
setup_enviroment = setup_environment

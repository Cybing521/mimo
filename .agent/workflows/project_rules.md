---
description: Enforce project structure and maintenance standards
---

# Project Structure & Maintenance Rules

## 1. Directory Structure

*   **Root**: Contains the main simulation entry point (`universal_simulation.py`) and `README.md`.
*   **`docs/`**: Contains detailed documentation, parameter explanations, and figure summaries.
    *   Move `experiment_parameters.md`, `figures_summary.md`, and other non-readme documentation here.
*   **`utils/`**: All auxiliary tools (e.g., PDF extractors, data converters) MUST be placed here.
*   **`results/`**: All simulation outputs (images, .json data, .csv logs) MUST be saved here.
*   **`papers/`**: Keep all reference PDFs and extracted text files here.

## 2. Documentation Standards

*   **README.md**: Should be the single source of truth for *getting started* and *running simulations*. It should link to detailed docs in `docs/`.
*   **Update Protocol**: Update `README.md` and relevant files in `docs/` IMMEDIATELY after any code modification or feature addition.

## 3. Coding Standards for Simulations

*   **Modular Design**:
    *   **Core Logic**: Keep in `mimo_optimized.py` (or similar library files).
    *   **Execution**: Keep in `universal_simulation.py` (the runner).
    *   *Do not duplicate core logic in the runner script.*
*   **Reproducibility**: Set random seeds, use relative paths.


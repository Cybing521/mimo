---
description: Enforce project structure and maintenance standards
---

# Project Structure & Maintenance Rules

## 1. Directory Structure

*   **Root**: Contains the main simulation entry point (`universal_simulation.py`) and `README.md`.
*   **`core/`**: **NEW** Contains all core algorithm libraries (one per paper).
    *   **Naming**: `<topic>_core.py` (e.g., `mimo_core.py`, `swipt_core.py`)
    *   Each file should have a header comment indicating the paper it implements.
*   **`docs/`**: Contains detailed documentation, parameter explanations, and figure summaries.
    *   Move `experiment_parameters.md`, `figures_summary.md`, and other non-readme documentation here.
*   **`utils/`**: All auxiliary tools (e.g., PDF extractors, data converters) MUST be placed here.
*   **`results/`**: All simulation outputs (images, .json data, .csv logs) MUST be saved here.
    *   **NEW RULE**: Results from different papers MUST be stored in separate subdirectories.
    *   Structure: `results/<paper_short_name>/` (e.g., `results/ma2023/`, `results/swipt2017/`)
*   **`papers/`**: Keep all reference PDFs and extracted text files here.

## 2. Documentation Standards

*   **README.md**: Should be the single source of truth for *getting started* and *running simulations*. It should link to detailed docs in `docs/`.
*   **Update Protocol**: Update `README.md` and relevant files in `docs/` IMMEDIATELY after any code modification or feature addition.

## 3. Coding Standards for Simulations

*   **Modular Design**:
    *   **Core Logic**: Keep in `mimo_core.py` (or similar library files named `*_core.py`).
    *   **Execution**: Keep in `universal_simulation.py` (the runner).
    *   *Do not duplicate core logic in the runner script.*
*   **Reproducibility**: Set random seeds, use relative paths.
*   **Naming Convention for Core Libraries**: Use `*_core.py` pattern for algorithm implementation modules (e.g., `mimo_core.py`, `swipt_core.py`).

## 4. Multi-Paper Project Management

*   **Separate Results**: When implementing algorithms from different papers, results MUST be stored in subdirectories named after the paper.
    *   Example: `results/ma2023/fig6_capacity.png`, `results/swipt2017/rate_energy_region.png`
*   **Naming Convention**: Use lowercase, short identifiers for paper directories (e.g., `ma2023`, `swipt2017`).
*   **Script Headers**: Each simulation script MUST include a comment header indicating which paper it implements.

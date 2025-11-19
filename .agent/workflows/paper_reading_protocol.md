---
description: Enforce paper reading before implementation
---

# Paper Reading Protocol

Before proposing any new scripts or implementations based on a research paper, the agent MUST:

1.  **Locate and Read the Paper**:
    *   Check the `papers/` directory for relevant PDF files.
    *   If the paper is not found, ask the user to provide it.
    *   Use available tools (e.g., `read_resource` if supported, or text extraction) to understand the paper's content.

2.  **Analyze Key Figures**:
    *   Identify the specific figure to be reproduced (e.g., Fig. 5, Fig. 6).
    *   Extract the exact simulation parameters for that figure:
        *   X-axis and Y-axis definitions.
        *   Fixed parameters (Antenna count, SNR, Region size, etc.).
        *   Variable parameters.
        *   Comparison schemes (Baselines).

3.  **Confirm with User**:
    *   Summarize the findings to the user.
    *   Ask for confirmation *before* writing any code.

4.  **Implementation**:
    *   Only after confirmation, proceed to create or modify scripts.

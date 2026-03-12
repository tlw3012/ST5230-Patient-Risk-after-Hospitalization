## Notebook-to-module restructuring plan

This document records how existing notebooks map to Python modules
under `src/`, and the current status of each notebook after the
repository cleanup for GitHub presentation.

### Current notebook layout (after cleanup)

- **Main demo notebook**: `CP4 Project - BioCliDistilBERT.ipynb` (repository root).  
  This is the primary user-facing demo for the BioClinical DistilBERT pipeline.
- **Archived**: `P4 Project - BioCliDistilBERT.ipynb` → `notebooks/archive/`.  
  Historical reference; logic migrated into `src/`.
- **Reference RAG notebook**: `CP4-1 Project - BioCliDistilBERT RAG.ipynb` → `notebooks/archive/rag/`.  
  Secondary reference for the RAG variant; not the main demo.

### Legend for notebook status

- **keep**: notebook will remain as a primary, user-facing artifact
  (possibly with light cleanup later).
- **merge**: notebook's logic will be merged into a smaller number of
  consolidated demonstration notebooks; the original may eventually be
  moved to `notebooks/archive/`.
- **archive**: notebook is kept for historical reference only; may live
  under `notebooks/archive/` (or `notebooks/archive/rag/` for RAG).

### Mapping table

| Current notebook name / location                                         | Future Python module destinations (main ones)                                                                                           | Planned notebook status |
|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| `FJY - data_clean&Integration.ipynb`                                    | `src/data/raw_loading.py`, `src/data/cleaning.py`, `src/data/building.py`                                                               | keep (for now)          |
| `CP4 Project - BioCliDistilBERT.ipynb` (root — **main demo**)            | `src/data/building.py`, `src/features/text_embedding.py`, `src/features/engineering.py`, `src/features/selection.py`                    | keep (main demo)        |
| `notebooks/archive/P4 Project - BioCliDistilBERT.ipynb`                  | `src/data/building.py`, `src/features/text_embedding.py`, `src/features/engineering.py`                                                 | archive                  |
| `notebooks/archive/rag/CP4-1 Project - BioCliDistilBERT RAG.ipynb`       | `src/features/text_embedding.py`, `src/features/engineering.py`, `src/features/selection.py`, `src/data/io.py`                          | archive (reference RAG) |
| `P7 Project - drgcode Norm.ipynb`                   | `src/features/engineering.py`, `src/features/selection.py`                                                                              | archive later           |
| `P5 Project - Regression.ipynb`                     | `src/features/engineering.py`, `src/models/tree_based.py`, `src/models/training.py`, `src/models/evaluation.py`                         | archive later           |
| `P6 Project - Regression Norm.ipynb`                | `src/features/engineering.py`, `src/features/labels.py`, `src/models/tree_based.py`, `src/models/training.py`, `src/models/evaluation.py` | archive later        |
| `CP5 Project - Reg-dod.ipynb`                       | `src/features/labels.py`, `src/features/engineering.py`, `src/models/logistic.py`, `src/models/figs.py`, `src/models/training.py`, `src/models/evaluation.py` | merge / archive later |
| `CP6 Project - Reg-severity.ipynb`                  | `src/features/labels.py`, `src/features/engineering.py`, `src/models/logistic.py`, `src/models/tree_based.py`, `src/models/training.py`, `src/models/evaluation.py` | merge / archive later |
| `CP7 Project - Reg-morality.ipynb`                  | `src/features/labels.py`, `src/features/engineering.py`, `src/models/logistic.py`, `src/models/tree_based.py`, `src/models/training.py`, `src/models/evaluation.py` | merge / archive later |
| `CP8 Project - Reg-acuity.ipynb`                    | `src/features/labels.py`, `src/features/engineering.py`, `src/models/logistic.py`, `src/models/tree_based.py`, `src/models/training.py`, `src/models/evaluation.py` | keep (representative) |

### Notes

- The mapping above is intentionally many-to-many: a single notebook
  often contains logic that will eventually land in several modules, and
  each module will aggregate logic from multiple notebooks.
- The `notebooks/archive/` directory holds archived notebooks (e.g. P4).
  The main demo is `CP4 Project - BioCliDistilBERT.ipynb` in the repository
  root; the RAG reference notebook is in `notebooks/archive/rag/`.


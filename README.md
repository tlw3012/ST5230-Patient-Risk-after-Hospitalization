
# Patient-Risk-after-Hospitalization

Course project repository for **ST5230**: A multimodal EHR prediction pipeline integrating vital signs and clinical notes via ClinicalBERT embeddings and ensemble ML models.
pipelines for clinical note embedding (BioClinical DistilBERT), feature engineering, and prediction tasks (regression/classification) using structured and text-derived features.

---

## Problem statement

Clinical notes and structured EHR data (vitals, demographics, DRG codes, etc.) can be combined to support prediction of outcomes such as in-hospital death, length of stay, disease severity, and acuity. This project explores:

- **Representing clinical text** via a lightweight transformer (DistilClinicalBERT) with block-level encoding and document-level aggregation.
- **Combining text embeddings with structured features** (numerical, categorical) into a single feature space for modeling.
- **Training and evaluating** logistic and tree-based models for multiple prediction targets.

The goal is a reproducible, modular workflow that can be adapted to other clinical prediction tasks and datasets.

---

## Methods

- **Text encoding**: [DistilClinicalBERT](https://huggingface.co/nlpie/distil-clinicalbert) (or similar) — tokenization, fixed-size blocking, optional [CLS]/[SEP] framing, mean-pooling over tokens and blocks to obtain a single 768-d vector per document.
- **Feature engineering**: Parsing and aligning text embeddings with structured tables; building sparse/dense design matrices; optional PCA or pooling over multi-vector columns.
- **Modeling**: Logistic regression and tree-based models (e.g. Random Forest, XGBoost) for classification/regression; standard train/validation or cross-validation; evaluation metrics (e.g. AUC-ROC, accuracy, calibration) and optional visualizations.
- **Code organization**: Reusable logic lives in `src/` (data loading, cleaning, embedding, feature construction); notebooks orchestrate runs and display results without duplicating core implementations.

---

## Prediction tasks

The modeling notebooks target different outcomes derived from the same or related EHR/clinical data:

| Task type | Description | Example notebooks |
|-----------|-------------|-------------------|
| **Death (in-hospital)** | Binary: patient died during admission (e.g. `dod`). | CP5 Reg-dod |
| **Severity** | Ordinal/numeric severity (e.g. DRG-based). | CP6 Reg-severity |
| **Mortality risk** | Mortality-related labels (e.g. DRG mortality). | CP7 Reg-morality |
| **Acuity** | Triage or acuity level (e.g. ED acuity). | CP8 Reg-acuity |
| **Regression (generic)** | Continuous or count outcomes. | P5 Regression, P6 Regression Norm |
| **DRG / coding** | DRG code or normalized coding targets. | P7 drgcode Norm |

See **MODELING_NOTEBOOK_GUIDE.md** for a short guide to each CP5–CP8 and P5–P7 notebook.

---

## Recommended reading order

1. **README.md** (this file) — overview, methods, and structure.
2. **CP4 Project - BioCliDistilBERT.ipynb** — main demo: from raw text to embeddings and downstream use. Start here for the full pipeline.
3. **FJY - data_clean&Integration.ipynb** — data cleaning and integration; how structured + text data are prepared.
4. **MODELING_NOTEBOOK_GUIDE.md** — what each modeling notebook (CP5–CP8, P5–P7) does and when to open it.
5. **RESTRUCTURE_PLAN.md** — which notebooks map to which `src/` modules (for contributors).

---

## Project structure

```
.
├── README.md
├── MODELING_NOTEBOOK_GUIDE.md     # Guide to CP5–CP8 and P5–P7
├── RESTRUCTURE_PLAN.md            # Notebook ↔ src/ mapping
│
├── CP4 Project - BioCliDistilBERT.ipynb   # ★ Main demo: embedding + pipeline
├── FJY - data_clean&Integration.ipynb      # Data cleaning & integration
├── CP5–CP8, P5–P7, …                      # Prediction / modeling notebooks
│
├── notebooks/
│   └── archive/
│       ├── P4 Project - BioCliDistilBERT.ipynb   # Archived; logic in src/ and CP4
│       └── rag/
│           └── CP4-1 Project - BioCliDistilBERT RAG.ipynb   # RAG variant (reference)
│
└── src/
    ├── config/
    ├── data/          # raw_loading, cleaning, building
    └── features/     # text_embedding, engineering, …
```

---

## Notebook roles (quick reference)

| Notebook | Role |
|----------|------|
| **CP4 Project - BioCliDistilBERT.ipynb** | **Main demo.** End-to-end: load data, compute text embeddings with DistilClinicalBERT, build feature matrices, and run downstream steps. This is the primary entry point. |
| **FJY - data_clean&Integration.ipynb** | **Data prep.** Cleans and integrates raw tables; shows how structured and text inputs are combined. Used before or alongside embedding/modeling. |
| **notebooks/archive/P4 Project - BioCliDistilBERT.ipynb** | **Archived.** Earlier version of the embedding pipeline; logic has been moved into `src/` and reflected in CP4. Kept for reference only. |
| **notebooks/archive/rag/CP4-1 Project - BioCliDistilBERT RAG.ipynb** | **RAG reference.** Retrieval-augmented variant of the pipeline. Secondary reference, not the main demo. |
| **CP5–CP8, P5–P7** | **Modeling notebooks.** Each focuses on a specific prediction task (death, severity, mortality, acuity, regression, DRG). See MODELING_NOTEBOOK_GUIDE.md. |

---

## Core logic in `src/`

Reusable code lives under `src/` so notebooks stay readable and consistent:

- **`src/features/text_embedding.py`** — Tokenize-and-block, add special tokens, extract block embeddings, reduce to document vector (768-d).
- **`src/features/engineering.py`** — Parse vector columns, build sparse matrices from embedding columns, and related helpers.
- **`src/data/`** — Raw loading, cleaning, and dataset building used by data and embedding pipelines.

Notebooks import from these modules instead of re-implementing logic. See **RESTRUCTURE_PLAN.md** for the full notebook–module mapping.

---

## Restricted data note

**This repository does not include restricted or proprietary datasets.** Code and notebooks assume that appropriate data (e.g. MIMIC-III/IV or other de-identified clinical data) are obtained separately and placed in a path you configure. You must comply with your institution’s and the dataset’s data-use agreements; paths and sample usage in the repo may need to be adjusted for your environment.

# Patient-Risk-after-Hospitalization

Course project repository for ST5230: a multimodal EHR prediction pipeline that integrates structured clinical data and clinical notes using BioClinical DistilBERT embeddings and machine learning models.

## Overview

This project studies how structured EHR features (for example, vital signs, demographics, and DRG-related variables) can be combined with clinical note representations to support outcome prediction after hospitalization.

The repository covers:

- clinical note embedding with BioClinical DistilBERT
- feature engineering for structured and text-derived features
- prediction tasks including classification and regression
- modularized data, feature, and modeling utilities under `src/`

## Problem Statement

Clinical notes contain rich information that is often not captured by structured variables alone. This project explores a multimodal workflow for combining note embeddings with structured EHR data to predict outcomes such as in/out-hospital death, severity, mortality-related risk, acuity and so on.

The goal is to build a reusable and reasonably modular prediction pipeline that can be adapted to similar clinical modeling settings.

## Methods

### Text Representation
Clinical notes are encoded using BioClinical DistilBERT-style embeddings. Long notes are split into fixed-size blocks, encoded block by block, and then aggregated into a single document-level vector.

### Feature Engineering
Text embeddings are aligned with structured clinical tables and combined into downstream modeling matrices. The pipeline supports vector parsing, sparse/dense feature construction, and optional dimensionality reduction.

### Modeling
The project includes logistic and tree-based models for multiple prediction tasks, together with shared helpers for train/test splitting, evaluation, and experiment utilities.

## Prediction Tasks

```text
.
| Task | Description |
|---|---|
| Death | Binary prediction of in-hospital death |
| Severity | Severity-related target |
| Mortality risk | Mortality-related target | 
| Acuity | Acuity / triage-related target |

## Repository Structure

```text
.
├── README.md
├── MODELING_NOTEBOOK_GUIDE.md
├── RESTRUCTURE_PLAN.md
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── pipelines/
├── notebooks/
│   └── archive/
│       └── rag/
└── *.ipynb
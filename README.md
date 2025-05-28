# SOMD 2025: Finetuning ModernBERT for In- and Out-of-Distribution NER and Relation Extraction of Software Mentions in Scientific Texts

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Steps to Run](#steps-to-run)
- [Results for NER and RE in Each Phase](#results-for-ner-and-re-in-each-phase)
- [Findings](#findings)
- [Limitations](#limitations)

## Overview

In this project, we utilize the dataset and evaluation criteria defined by Software Mention Detection - (SOMD 2025)  competition to solve the problem of Named Entity Recognition and Relation classification in input sentences from the scientific texts. During the competition, by finetuning ModernBERT and building a joint model on top of it, we achieve best SOMD F1 score of $0.89$ in Phase I. Using the same model, we achieve the second best SOMD score of $0.55$ in Phase II. In the Open Submission phase, we experiment with Adapative finetuning, achieving a SOMD score of $0.6$, with best macro average for NER being $0.69$. Our work shows the efficiency of finetuning even with a small dataset and the promise of adaptive finetuning on Out-of-Distribution (OOD) dataset. 

## Project Structure

```bash
.
├── EntityModel_checkpoint/
├── FewShot_checkpoint/
├── JointModel_checkpoint/
├── ModernBERT_checkpoint/
├── data/
│   ├── phase_1/
│   │   ├── test_texts.txt
│   │   ├── train_entities.txt
│   │   ├── train_relations.txt
│   │   └── train_texts.txt
│   ├── phase_2/
│   │   ├── entities.txt
│   │   ├── relations.txt
│   │   ├── test_texts.txt
│   │   └── texts.txt
│   └── predictions/
│       ├── phase_1/
│       └── phase_2/
├── results/
│   ├── phase_1.zip
│   ├── phase_2_0.55.zip
│   └── phase_2_0.6.zip
├── src/
│   ├── __init__.py
│   ├── phase_1/
│   │   ├── __init__.py
│   │   ├── config_.py
│   │   ├── infer.py
│   │   └── train.py
│   └── phase_2/
│       ├── __init__.py
│       ├── adapter_weighted_inference.py
│       ├── config.py
│       ├── dataloader.py
│       ├── model.py
│       ├── relation_adapter_weighted.py
│       ├── relation_dataset.py
│       ├── relation_model.py
│       └── utils.py
├── requirements.txt
├── LICENSE
└── README.md
```

**Note:** The model files are available at the link : [SOMD-2025-models](https://drive.google.com/drive/folders/1OUHnB04Ljye_0_SSD_zu9TTDjJAXDvPc?usp=drive_link). After cloning the repo, download all the model files in their respective directory for further processing.

## Installation

### Clone the repository

```
git clone https://github.com/ekbanasolutions/somd-2025
cd somd-2025
```

### Create a virtual environment and activate it:

#### On Linux / Mac:

```
python -m venv venv
source venv/bin/activate 
```

#### On Windows

```
python -m venv venv
venv\Scripts\activate
```

### Install dependencies:

```
pip install -r requirements.txt
```

## Steps to run

### Phase - I

- You can modify the parameters for Phase I in the [config file](./src/phase_1/config_.py).

#### Train the model

```
cd SOMD_2025/src/phase_1/
python3 train.py
```

#### For inference

```
cd SOMD_2025/src/phase_1/
python3 infer.py
```

### Phase - II

- You can modify the parameters for Phase II in the [config file](./src/phase_2/config.py).

#### Train the model

```
cd SOMD_2025/src/phase_2/
python3 relation_adapter_weighted.py
```

#### For Inference

```
cd SOMD_2025/src/phase_2/
python3 adapter_weighted_inference.py
```

## Results for NER and RE in Each Phase

| Phase               | F1 SOMD | NER F1 | NER Precision | NER Recall | RE F1 | RE Precision | RE Recall |
|---------------------|---------|--------|----------------|------------|-------|---------------|-----------|
| Phase I             | 0.89    | 0.93   | 0.93           | 0.95       | 0.84  | 0.85          | **0.86**      |
| Phase I (Modified Joint Model)   | **0.92**    | **0.95**   | **0.95**           | **0.96**       | **0.89**  | **0.95**          | 0.85      |
| Phase II            | 0.55    | 0.64   | 0.67           | 0.65       | 0.46  | 0.69          | 0.39      |
| Open Submission     | **0.60**    | **0.69**   | **0.74**           | **0.69**       | **0.51**  | **0.71**          | **0.42**      |

## Findings

- During Phase I, the Joint Model using ModernBERT achieved the highest overall performance with an F1 score of 0.89.

- After Phase I, a refined approach — the Modified Joint Model — was developed, which improved the F1 SOMD score to 0.92.

- The model failed to generalize well to Out-of-Distribution (OOD) dataset in Phase II, resulting in a significant drop in performance with a SOMD F1 score of 0.55.

- Following multiple post-Phase II experiments, the best SOMD F1 score achieved was 0.60.

## Limitations

- Poor generalization to Out-of-Distribution (OOD) dataset.

- Relation Extraction depends heavily on accurate Entity Extraction.

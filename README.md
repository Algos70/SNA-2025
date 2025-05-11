# Temporal Link Prediction

This repository implements a Temporal Link Prediction pipeline using PyTorch Geometric Temporal (GConvGRU) to forecast edge formation in dynamic graphs. It covers data preprocessing, model training, inference, validation generation, and evaluation.

## Table of Contents

- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Data Organization](#data-organization)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Validation & Evaluation](#validation--evaluation)
- [Generating Validation Queries](#generating-validation-queries)
- [Evaluation Parameters](#evaluation-parameters)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Notes & Tips](#notes--tips)

## Project Structure

```
temporal_link_prediction/
├── README.md
├── requirements.txt
├── preprocess.py          # remap & sample edges
├── scripts/
│   └── make_val.py        # generate validation queries
├── data/                  # input CSVs
│   ├── edges_train_A.csv
│   ├── node_features.csv
│   ├── edge_type_features.csv
│   └── val_queries.csv    # generated validation queries
├── output/                # model predictions
│   ├── output_A.csv       # test-set predictions
│   └── output_val.csv     # validation-set predictions
└── src/
    ├── dataset.py         # TemporalLinkPredictionDataset
    ├── model.py           # GConvGRU link predictor
    ├── train.py           # training script
    ├── inference.py       # inference script
    └── evaluate.py        # evaluation script
```

## Dependencies

```bash
# Python 3.10+
pip install -r requirements.txt
```

**requirements.txt**:
```text
torch>=1.13
torch-geometric>=2.4.0
torch-geometric-temporal>=0.3.1
pandas
numpy
scikit-learn
tqdm
``` 

## Setup

1. Clone the repo and navigate:
  ```bash
   git@github.com:Algos70/SNA-2025.git
   git clone <repo_url>
   cd temporal_link_prediction
  ```
2. Create & activate virtual environment:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

## Data Organization

Place original CSV files under `data/`:
- `edges_train_A.csv`
- `node_features.csv`
- `edge_type_features.csv`
- `input_A.csv`

## Data Preprocessing

Generate remapped & sampled edges and mapped features:

```bash
python src/preprocess.py \
  --chunksize 1000000 \
  --sample_frac 0.0001
```

Outputs:
- `data/edges_train_A_mapped.csv`
- `data/node_features_mapped.csv`
- `data/edge_type_features_mapped.csv`

## Training

Train the model on sampled data:

```bash
python src/train.py \
  --edges data/edges_train_A_mapped.csv \
  --node_feats data/node_features_mapped.csv \
  --etype_feats data/edge_type_features_mapped.csv \
  --num_nodes 19442 \
  --hidden 64 \
  --epochs 20 \
  --lr 1e-3 \
  --save_path model/model.pth
```

## Inference

#### Test Set

Generate predictions for `input_A.csv` (official test queries):

```bash
python src/inference.py \
  --model_path model/model.pth \
  --edges data/edges_train_A_mapped.csv \
  --node_feats data/node_features_mapped.csv \
  --query data/input_A.csv \
  --output output/output_A.csv \
  --hidden 64
```


## Generating Validation Queries

Use the provided script to generate your validation queries from the remapped edges:

```bash
python src/make_val.py
```

This will create `data/val_queries.csv` containing rows:
```
src,dst,etype,t0,t1,label
```

#### Validation Set

Generate predictions for your validation queries in `data/val_queries.csv`:

```bash
python src/inference.py \
  --model_path model/model.pth \
  --edges data/edges_train_A_mapped.csv \
  --node_feats data/node_features_mapped.csv \
  --query data/val_queries.csv \
  --output output/output_val.csv \
  --hidden 64
```

## Evaluation Parameters

To evaluate your model on the validation set, run:

```bash
python src/evaluate.py \
  --preds output/output_val.csv \
  --labels data/val_queries.csv
```

- `--preds`: path to the CSV of model output scores (sixth column is used).
- `--labels`: path to CSV of validation queries with last column as ground-truth label (0/1).

The script will report:
- **ROC AUC**
- **Average Precision**
- **PR AUC**

## Hyperparameter Tuning

Adjust in `src/train.py` or command line:
- `--hidden` (hidden size)
- `--epochs`
- `--lr` (learning rate)
- Negative-sampling ratio in `dataset.py`
- Number of GConvGRU hops (`K` parameter)

## Notes & Tips

- **Mac M1 GPU**: Use `mps` device and enable AMP for speed-ups.
- **Full dataset**: For production, switch to a streaming TGN loop to handle all 27M edges efficiently.
- **Validation size**: Increase `val_queries.csv` size for robust metrics.
- **Error analysis**: Inspect high-confidence false positives/negatives per edge type or time gap.

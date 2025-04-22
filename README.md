# SNA-2025

## Temporal Link Prediction in Dynamic Graphs

### Project Overview
This project implements temporal link prediction models for dynamic graphs using PyTorch Geometric Temporal. The system predicts the probability of edge formation between nodes within a specified future time window, making it valuable for applications like social network event forecasting, e-commerce demand prediction, and recommender systems.

### Features
- Graph Neural Network-based temporal link prediction
- Support for large-scale temporal graph data
- Edge formation probability prediction
- Specialized handling for dynamic event graphs
### Dataset link
Download and extract datasetA from the following link to datasetA folder:
https://www.dgl.ai/WSDM2022-Challenge/

### Dataset Structure
The project uses a dynamic event graph dataset (Dataset A) with the following components:

```
datasetA/
├── edges_train_A.csv      # Temporal edges data
├── node_features.csv      # Node attribute information
└── edge_type_features.csv # Edge type characteristics
```

#### Data Description
- **edges_train_A.csv**: Contains temporal edge information
  - Source node
  - Destination node
  - Edge type
  - Timestamp

- **node_features.csv**: Node-level features
  - Categorical attributes
  - Missing values marked as -1

- **edge_type_features.csv**: Edge type characteristics
  - Anonymized categorical attributes

### Requirements
```
pip install -r requirements.txt
```

### Project Structure
```
├── datasetA/             # Dataset directory
├── src/                  # Source code
│   └── main.py          # Main implementation
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

### Technology Stack
- PyTorch Geometric Temporal
- Python
- Additional dependencies listed in requirements.txt

### Getting Started
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python src/main.py
   ```

### Implementation Details
The project implements a unified machine learning model that:
- Uses PyTorch Geometric Temporal for graph neural network implementation
- Handles temporal aspects of dynamic graphs
- Predicts edge formation probabilities
- Generalizes across different graph datasets

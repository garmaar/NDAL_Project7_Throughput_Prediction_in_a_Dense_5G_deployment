# 5G Throughput Prediction in Dense Deployments & Federated Learning

End-to-end machine learning and distributed computing pipeline to predict user-level downlink throughput in dense 5G New Radio (NR) networks[cite: 2]. The project evaluates centralized regression architectures against a decentralized **Federated Learning (FedAvg)** framework implemented in **PyTorch**[cite: 2], preserving raw measurement privacy across simulated network edge nodes[cite: 2].

---

## Architecture Overview

```text
5g-throughput-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── acc_arena_subset.csv         # 108k+ spatial-temporal radio measurement samples
│
├── scripts/
│   └── extract_acc_arena_subset.py  # Pre-filtering and dataset reduction script
│
├── notebooks/
│   └── throughput_prediction.ipynb  # End-to-end pipeline: EDA, ML & Federated Learning
│
└── docs/
    └── presentation.pdf             # Technical architectural slide deck
```

---

## Key Technical Highlights

* **Large-Scale 5G Radio Dataset:** Ingested and processed 108,240 spatial-temporal network traces collected at the **ACC Arena**[cite: 2].
* **Radio-Frequency (RF) Metrics:** Features include Downlink/Uplink Signal-to-Interference-plus-Noise Ratio (`sinr_dl`, `sinr_ul`), Block Error Rate (`bler`), Physical Resource Blocks (`prb`), 3D spatial positioning (`x`, `y`, `z`), and serving Radio Unit (`ru_id`)[cite: 2].
* **Feature Engineering & Operator Aggregates:** Engineered a time-windowed aggregation mechanism extracting statistical profiles (count, mean, median, min, max, std) from operator measurement devices to capture local cell load without label leakage[cite: 2].
* **Leakage-Free Temporal Splitting:** Implemented strict per-user chronological train/test partitioning (`TEST_FRACTION = 0.2`) followed by a global time-series sort, avoiding future-sample lookahead bias[cite: 2].
* **Decentralized Model Convergence:** Built a custom **Federated Averaging (FedAvg)** simulator using PyTorch tensors and DataLoaders to aggregate local user weights over multiple communication rounds[cite: 2].

---

## Machine Learning Pipeline

### 1. Preprocessing Pipeline (`scikit-learn`)
* **Numerical Features:** Imputed via median replacement and scaled using `StandardScaler`[cite: 2].
* **Categorical Features:** Traffic types (e.g., video, constant rate, HTTP) and Radio Unit IDs encoded via `OneHotEncoder(handle_unknown='ignore')`[cite: 2].
* **Pipeline Integration:** Unified under a modular `ColumnTransformer` to guarantee zero data leakage between training folds and test sets[cite: 2].

### 2. Evaluated Architectures
* **Centralized Baseline:** 
  * **Random Forest Regressor:** Non-linear decision trees tuned with hyperparameter search[cite: 2].
  * **Multi-Layer Perceptron (MLP):** Feed-forward neural network trained on normalized tabular inputs[cite: 2].
* **Federated Learning (FedAvg - PyTorch):**
  * Local node updates trained over client-specific epochs (`LOCAL_EPOCHS = 3`, `LOCAL_MIN_SAMPLES = 50`)[cite: 2].
  * Global coordinator averaging server-side parameters across 20 federated rounds to study decentralized convergence against centralized models[cite: 2].
* **Sensitivity Experiment ($X$ Parameter):** Evaluated predictive robustness against varying numbers of available measurement devices ($X \in \{3, 5, 10, 20\}$)[cite: 2].

---

## Performance Metrics

Models are evaluated using standard regression criteria[cite: 2]:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| \qquad \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} \qquad R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## Quickstart & Environment Setup

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone [https://github.com/garmaar/5g-throughput-prediction.git](https://github.com/garmaar/5g-throughput-prediction.git)
cd 5g-throughput-prediction
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Notebook Execution
```bash
jupyter notebook notebooks/throughput_prediction.ipynb
```

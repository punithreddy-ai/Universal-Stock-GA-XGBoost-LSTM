# 📈 Universal Stock Direction Prediction
### GA-Selected XGBoost + Universal LSTM for NSE Stock Prediction

<p align="center">
  <strong>AI-powered stock direction classification using technical indicators, Genetic Algorithm feature selection, XGBoost, and LSTM.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/XGBoost-ML-green?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/NSE-India-red?style=for-the-badge" alt="NSE">
  <img src="https://img.shields.io/badge/Status-Academic%20Project-purple?style=for-the-badge" alt="Status">
</p>

---

## 🧭 Table of Contents

- [📌 About the Project](#-about-the-project)
- [🎯 Objectives](#-objectives)
- [✨ Key Features](#-key-features)
- [🧠 System Architecture](#-system-architecture)
- [🔬 Methodology](#-methodology)
- [🎨 UI/UX Design](#-uiux-design)
- [📊 Machine Learning Pipeline](#-machine-learning-pipeline)
- [📁 Project Structure](#-project-structure)
- [⚙️ Technologies Used](#️-technologies-used)
- [🚀 Installation](#-installation)
- [▶️ How to Run](#️-how-to-run)
- [📈 Evaluation](#-evaluation)
- [📦 Generated Outputs](#-generated-outputs)
- [🔐 Data & Security](#-data--security)
- [🧪 Research Integrity](#-research-integrity)
- [🔮 Future Scope](#-future-scope)
- [👨‍💻 Author](#-author)
- [⚠️ Disclaimer](#️-disclaimer)

---

# 📌 About the Project

**Universal Stock Direction Prediction** is an academic machine-learning project designed to classify the **next-day direction of NSE stocks** as:

- 🟢 **UP**
- 🔴 **DOWN**

The project combines traditional technical analysis with machine learning and deep learning.

The core pipeline integrates:

```text
NSE Historical Data
        ↓
Data Cleaning
        ↓
Technical Indicators
        ↓
Feature Engineering
        ↓
Genetic Algorithm
        ↓
Feature Selection
        ↓
XGBoost Classification
        ↓
Universal LSTM
        ↓
Threshold Selection
        ↓
Evaluation & Leakage Audit
        ↓
Prediction Results
```

The project is designed around a **universal multi-stock approach**, allowing the same modeling framework to work across multiple supported NSE stocks.

---

# 🎯 Objectives

### 1. 📊 Data Processing
Prepare historical NSE stock data using chronological processing and stock-wise handling.

### 2. 🧮 Feature Engineering
Generate technical indicators and derived market features for model training.

### 3. 🧬 Intelligent Feature Selection
Use a **Genetic Algorithm (GA)** to identify a useful subset of technical features.

### 4. 🌲 XGBoost Classification
Use XGBoost to classify the next trading day's direction.

### 5. 🧠 LSTM Sequence Learning
Use a Universal LSTM model to learn temporal patterns from historical sequences.

### 6. 🔍 Robust Evaluation
Evaluate models using multiple classification metrics and perform a leakage audit.

### 7. 📦 Deployment Readiness
Save model artifacts and feature information for future integration with a Streamlit or web-based prediction interface.

---

# ✨ Key Features

| Feature | Description |
|---|---|
| 📈 NSE Stock Data | Historical Indian stock-market data |
| 🧹 Data Cleaning | Stock-wise chronological preprocessing |
| 📊 Technical Indicators | Momentum, trend, volatility and volume features |
| 🧬 Genetic Algorithm | Automated feature subset selection |
| 🌲 XGBoost | Tree-based direction classifier |
| 🧠 Universal LSTM | Sequence-based deep-learning model |
| 🎯 Threshold Optimization | Validation-based classification threshold |
| 🧪 Ablation Study | Comparison of different feature/model configurations |
| 🔐 Leakage Audit | Checks for inappropriate information flow |
| 📉 Confusion Matrix | Detailed classification analysis |
| 📊 ROC-AUC | Ranking-based model evaluation |
| 💾 Model Export | Saves models and inference artifacts |
| 🚀 Deployment Ready | Designed for future Streamlit integration |

---

# 🧠 System Architecture

```text
┌──────────────────────────────┐
│       NSE Historical Data    │
│          2015 – 2024         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Data Preprocessing     │
│  Cleaning • Sorting • Checks │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Technical Indicators     │
│ Trend • Momentum • Volume    │
│ Volatility • Price Features  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Genetic Algorithm (GA)    │
│       Feature Selection      │
└──────────────┬───────────────┘
               │
               ▼
      ┌───────────────────┐
      │ Selected Features │
      └─────────┬─────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌───────────────┐  ┌───────────────┐
│   XGBoost     │  │  Universal    │
│ Classifier    │  │     LSTM      │
└───────┬───────┘  └───────┬───────┘
        │                  │
        └────────┬─────────┘
                 ▼
      ┌─────────────────────┐
      │ Prediction &        │
      │ Performance Analysis│
      └─────────────────────┘
```

---

# 🔬 Methodology

## Step 1 — Data Acquisition

The project uses the Kaggle dataset:

```text
manavanghan/nse-india-stock-market-data-2015-2024
```

The notebook downloads the dataset through **KaggleHub**.

Expected primary dataset:

```text
nifty500_stocks.csv
```

---

## Step 2 — Data Preprocessing

The data is processed independently for each stock.

Main operations include:

- Date conversion
- Chronological sorting
- Duplicate handling
- Missing-value handling
- Stock-wise processing
- Target generation
- Train/validation/test separation

The project uses chronological splitting rather than random shuffling for the main time-series workflow.

---

## Step 3 — Technical Feature Engineering

The model uses market-derived features based on:

### 📈 Trend
- Moving averages
- Exponential moving averages
- Trend-related indicators

### ⚡ Momentum
- RSI
- MACD
- Momentum-related features

### 🌪️ Volatility
- ATR
- Rolling volatility
- Price-range features

### 📊 Volume
- Volume changes
- Volume-related indicators

### 💹 Price
- Open
- High
- Low
- Close
- Returns
- Price relationships

---

# 🧬 Genetic Algorithm + XGBoost

The Genetic Algorithm searches for useful feature subsets.

Conceptually:

```text
Initial Population
       ↓
Feature Subsets
       ↓
Model Fitness
       ↓
Selection
       ↓
Crossover
       ↓
Mutation
       ↓
New Generation
       ↓
Best Feature Subset
       ↓
XGBoost
```

The objective is to reduce unnecessary features while retaining predictive information.

---

# 🧠 Universal LSTM

The project also includes a Universal LSTM component.

The LSTM processes historical sequences using a configurable lookback window:

```python
WINDOW = 60
```

Conceptually:

```text
Day 1 ─┐
Day 2  │
Day 3  │
 ...   ├──► LSTM ───► Next-Day Direction
Day 60 │
       ┘
```

This allows the model to learn temporal dependencies that may not be represented by individual observations.

---

# 🎨 UI/UX Design

Although the current repository is centered around the research notebook, the project is structured for a future **interactive stock-prediction dashboard**.

## 🖥️ Proposed Dashboard

```text
┌────────────────────────────────────────────────────────────┐
│  📈 UNIVERSAL STOCK AI                         ● LIVE      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Select Stock       Prediction Horizon                    │
│  ┌─────────────┐    ┌──────────────┐                       │
│  │ TCS       ▼ │    │ Next Day     │                       │
│  └─────────────┘    └──────────────┘                       │
│                                                            │
│  ┌──────────────────────┐  ┌────────────────────────────┐  │
│  │   MARKET SIGNAL      │  │       CONFIDENCE            │  │
│  │                      │  │                             │  │
│  │       🟢 UP          │  │          78.4%              │  │
│  │                      │  │     Model Confidence       │  │
│  └──────────────────────┘  └────────────────────────────┘  │
│                                                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  PRICE / SIGNAL CHART                  │ │
│  │                                                        │ │
│  │       ╱╲      ╱╲                                      │ │
│  │  ╱╲  ╱  ╲____╱  ╲___                                 │ │
│  │                                                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                            │
│  Technical Indicators                                     │
│  RSI     MACD     ATR     Volume     Trend               │
│  61.2    Bullish  Normal  ↑          Positive             │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 🎨 UI/UX Principles

### 1. Clean Dashboard
Use a clean card-based layout so users can understand the prediction quickly.

### 2. Visual Prediction Signal

```text
🟢 UP
🔴 DOWN
🟡 NEUTRAL / UNCERTAIN
```

The interface should make the prediction immediately visible.

### 3. Confidence Visualization

Use a progress indicator or gauge:

```text
Confidence
████████████████░░░░ 78%
```

### 4. Interactive Stock Selection

Users should be able to select a supported NSE stock from a searchable dropdown.

### 5. Interactive Charts

The future dashboard can provide:

- Candlestick chart
- Moving averages
- RSI
- MACD
- Volume
- Model prediction markers

### 6. Explainability

A future version can show:

```text
Why this prediction?

✓ RSI contribution
✓ Momentum contribution
✓ Trend contribution
✓ Volume contribution
✓ Selected GA features
```

### 7. Responsive Design

The planned interface should work across:

- 💻 Desktop
- 📱 Mobile
- 🖥️ Large displays

### 8. User Experience Flow

```text
Select Stock
     ↓
Load Market Data
     ↓
Generate Features
     ↓
Run Model
     ↓
Display Signal
     ↓
Show Confidence
     ↓
Explain Prediction
```

---

# 📊 Machine Learning Pipeline

```text
Raw Stock Data
      │
      ▼
Cleaning & Validation
      │
      ▼
Technical Indicators
      │
      ▼
Feature Matrix
      │
      ├───────────────┐
      │               │
      ▼               ▼
   GA Feature      LSTM Sequence
   Selection          Creation
      │               │
      ▼               ▼
   XGBoost          LSTM
      │               │
      └───────┬───────┘
              ▼
       Validation Set
              │
              ▼
       Threshold Selection
              │
              ▼
          Test Set
              │
              ▼
      Performance Metrics
              │
              ▼
        Leakage Audit
```

---

# 📁 Project Structure

```text
Universal-Stock-GA-XGBoost-LSTM/
│
├── 📓 Universal_Stock_GA_XGBoost_LSTM_FINAL.ipynb
│
├── 📄 README.md
│
├── 📦 requirements.txt
│
├── 🔒 .gitignore
│
├── 📜 LICENSE
│
└── 📂 universal_stock_project/
    │
    ├── universal_ga_xgboost.json
    ├── universal_lstm.keras
    ├── universal_lstm_scaler.pkl
    ├── universal_features.json
    ├── universal_config.json
    ├── feature_engineering.py
    ├── supported_stocks.json
    ├── feature_importance.csv
    ├── per_stock_results.csv
    ├── universal_model_results.csv
    ├── ablation_study.csv
    ├── latest_predictions_all_stocks.csv
    ├── ga_history.csv
    ├── test_predictions.csv
    ├── split_dates.csv
    └── leakage_audit.json
```

> The `universal_stock_project/` directory is generated during notebook execution. Large datasets and model artifacts are intentionally excluded from the default GitHub commit.

---

# ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming language |
| 🐼 Pandas | Data processing |
| 🔢 NumPy | Numerical computation |
| 📊 Matplotlib | Visualization |
| 🤖 Scikit-learn | Preprocessing and evaluation |
| 🌲 XGBoost | Classification |
| 🧠 TensorFlow / Keras | LSTM |
| 🧬 Genetic Algorithm | Feature selection |
| 📦 Joblib | Artifact serialization |
| ☁️ KaggleHub | Dataset acquisition |
| 📓 Jupyter | Experiment environment |
| 🚀 Streamlit | Planned UI/deployment layer |

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Universal-Stock-GA-XGBoost-LSTM.git
cd Universal-Stock-GA-XGBoost-LSTM
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Register the Jupyter Kernel

```bash
python -m ipykernel install --user \
  --name universal-stock \
  --display-name "Universal Stock (Python 3.10)"
```

## 5. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
Universal_Stock_GA_XGBoost_LSTM_FINAL.ipynb
```

Select the:

```text
Universal Stock (Python 3.10)
```

kernel.

---

# ▶️ How to Run

## Quick Test

For the first execution, use:

```python
TARGET_STOCK = "TCS"
QUICK_MODE = True
```

This reduces the computational workload and is recommended for checking whether the environment and dataset are configured correctly.

## Full Experiment

After confirming the notebook works:

```python
QUICK_MODE = False
```

The full experiment can require significantly more CPU/RAM/time.

---

# ⚙️ Main Configuration

Important configuration parameters include:

```python
TARGET_STOCK = "TCS"
QUICK_MODE = False
SEED = 42
USE_ADJ_CLOSE = False
WINDOW = 60
THRESHOLD_MODE = "val_balanced_accuracy"
RUN_TUNING = True
RUN_LOSO = False
RUN_SHAP = False
```

Example stocks:

```python
TARGET_STOCK = "TCS"
TARGET_STOCK = "RELIANCE"
TARGET_STOCK = "INFY"
TARGET_STOCK = "HDFCBANK"
TARGET_STOCK = "ICICIBANK"
TARGET_STOCK = "SBIN"
```

---

# 📈 Evaluation

The project evaluates classification performance using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Balanced Accuracy
- Specificity
- Confusion Matrix
- Per-stock performance
- Ablation study
- Majority-class baseline
- Leakage audit

The validation set is used for model/threshold decisions, while the test set is reserved for final evaluation.

---

# 📦 Generated Outputs

After execution, the notebook can generate:

### 🤖 Model Artifacts

```text
universal_ga_xgboost.json
universal_lstm.keras
universal_lstm_scaler.pkl
```

### ⚙️ Configuration & Features

```text
universal_features.json
universal_config.json
supported_stocks.json
feature_engineering.py
```

### 📊 Results

```text
feature_importance.csv
per_stock_results.csv
universal_model_results.csv
ablation_study.csv
latest_predictions_all_stocks.csv
ga_history.csv
test_predictions.csv
split_dates.csv
```

### 🔍 Audit

```text
leakage_audit.json
```

---

# 🔐 Data & Security

Never upload sensitive credentials to GitHub.

The repository's `.gitignore` excludes:

```text
.env
.env.*
.kaggle/
kaggle.json
venv/
.venv/
*.keras
*.pkl
*.joblib
```

Before pushing:

```bash
git status
```

Check that no API keys, passwords, credentials, or private files are staged.

---

# 🧪 Research Integrity

This project is an academic implementation focused on NSE stock-direction classification.

It should **not** be described as a direct reproduction of another paper's experimental results.

The implementation differs according to:

- Market
- Dataset
- Stock universe
- Feature engineering
- Train/validation/test protocol
- Genetic Algorithm configuration
- Model configuration
- Target definition
- Universal LSTM integration

The project does not hard-code a target accuracy.

Unexpectedly high performance should be investigated for possible data leakage or experimental issues.

---

# 🔮 Future Scope

## 🌐 Web Dashboard

Develop a Streamlit-based dashboard containing:

```text
Stock Selection
      ↓
Live/Latest Market Data
      ↓
Feature Engineering
      ↓
GA-XGBoost + LSTM
      ↓
Prediction
      ↓
Interactive Dashboard
```

## 🎨 Advanced UI/UX

Future interface features:

- 🌙 Dark / light mode
- 📊 Interactive TradingView-style charts
- 🔎 Searchable stock selector
- 🟢 Real-time prediction cards
- 📈 Confidence gauge
- 🧠 Explainable AI panel
- 📱 Responsive mobile layout
- 📊 Historical prediction performance
- 🔔 Optional alerts
- 📋 Model comparison dashboard

## 🤖 Explainable AI

Future versions can integrate feature-level explanations such as SHAP-based interpretation to help users understand which features influenced the prediction.

---

# 👨‍💻 Author

### Punithreddy K R

**Computer Science and Engineering**  
**Sri Krishna Institute of Technology, Bengaluru**

Academic project focused on:

> Artificial Intelligence • Machine Learning • Deep Learning • Financial Data Analytics

---

# ⚠️ Disclaimer

This project is developed for **academic, educational, and research purposes only**.

Stock-market prediction is inherently uncertain. Model predictions are not guaranteed to be accurate and should **not** be interpreted as financial advice, investment recommendations, or guarantees of future returns.

Always perform independent financial research and consult a qualified financial professional before making investment decisions.

---

<p align="center">
  <strong>📈 Universal Stock AI • Research • Experiment • Analyze</strong>
</p>

<p align="center">
  Made with Python, XGBoost, TensorFlow & ❤️
</p>

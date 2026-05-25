# ML From Scratch

From-scratch implementations of classical machine learning algorithms using NumPy, validated against sklearn baselines. Built as part of MLE interview preparation — the goal is to understand the math and code behind each algorithm, not just use library APIs.

---

## Models

| Model | Dataset | R² | Status |
|---|---|---|---|
| Linear Regression (Gradient Descent) | Energy Efficiency | 0.8983 | ✅ Complete |
| Logistic Regression | Coming soon | — | 🔜 |

---

## Project Structure

```
ml-from-scratch/
├── models/
│   ├── guides.py              # Abstract base class — enforces fit/predict/score interface
│   └── supervised/
│       └── regression.py     # LinearRegressionGD — gradient descent from scratch
├── notebooks/
│   └── 01_linear_regression_energy_efficiency.ipynb
├── data/
│   └── README.md             # Download instructions
├── requirements.txt
└── README.md
```

---

## Architecture

All models inherit from a shared abstract base class (`Model`) that mirrors the sklearn estimator convention:

```python
class Model(ABC):
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def score(self, X, y): ...
```

This enforces a consistent interface so every model is drop-in compatible across notebooks.

---

## Linear Regression — Energy Efficiency Dataset

**Dataset:** [UCI Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency)  
**Paper:** Tsanas & Xifara (2012), *Energy and Buildings*  
**Task:** Predict building Heating Load (Y1, kWh/m²) from 8 architectural parameters

### Dataset Details

768 building simulations generated in Ecotect, varying 12 building shapes across glazing area, distribution, and orientation parameters. No missing values.

| Feature | Description |
|---|---|
| X1 | Relative Compactness |
| X2 | Surface Area |
| X3 | Wall Area |
| X4 | Roof Area |
| X5 | Overall Height |
| X6 | Orientation |
| X7 | Glazing Area |
| X8 | Glazing Area Distribution |
| **Y1** | **Heating Load (target)** |
| Y2 | Cooling Load (not used) |

### EDA Findings

- **Multicollinearity:** X1, X2, X4, X5 form a geometry cluster. X1↔X2 correlation = −0.99, X4↔X5 = −0.97. Including all would make X^T X near-singular.
- **Feature selection:** Dropped X1, X2, X4 (redundant with X5) and X6 (zero target correlation). Kept X3, X5, X7, X8.
- **Bimodal target:** Y1 distribution reflects two building height sub-populations (3.5m vs 7.0m), a known limitation for a single linear model.

### Results

| | CV R² (10-fold) | Test R² | RMSE | MAE |
|---|---|---|---|---|
| sklearn (normal equation) | 90.94% ± 0.02 | 89.83% | 3.35 kWh/m² | 2.39 kWh/m² |
| From-scratch GD | — | **89.83%** | — | — |

GD converges at iteration ~451 / 1000 via early stopping (tolerance 1e-6), matching the sklearn closed-form solution to 4 decimal places on all coefficients.

### Implementation — Gradient Descent

The bias term is folded into the weight vector by prepending a column of ones to X.

```
Forward pass:  ŷ = X @ w
Loss (MSE):    L = (1/n) ||y - ŷ||²
Gradient:      ∂L/∂w = -(2/n) X^T (y - ŷ)
Update:        w = w - α * ∂L/∂w
```

Features must be normalized before training — GD is sensitive to scale, the normal equation is not.

---

## Setup

```bash
git clone https://github.com/whood23/ml-from-scratch.git
cd ml-from-scratch
pip install -r requirements.txt
```

Download the Energy Efficiency dataset from [UCI](https://archive.ics.uci.edu/dataset/242/energy+efficiency) and place `ENB2012_data.xlsx` in the `data/` directory, then run the notebook.

---

## References

- Tsanas, A., & Xifara, A. (2012). Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools. *Energy and Buildings*, 49, 560–567.
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI*.

# 🧠 Machine Learning with scikit-learn: A Comprehensive Comparative Study

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)

A **systematic, reproducible, and visually rich comparative analysis** of **10 classical machine learning algorithms** using `scikit-learn`. Each model is rigorously evaluated on real-world datasets with full preprocessing, hyperparameter tuning, cross-validation, performance visualization, and in-depth model comparison.

---

## 🎯 Features

### 🔍 Unified Experimental Framework
- **Consistent workflow** across all 10 models
- Full **Exploratory Data Analysis (EDA)** with visualizations
- **Stratified train/test splits** and **StratifiedKFold** cross-validation
- **Hyperparameter tuning** via `GridSearchCV`
- **Comprehensive evaluation**: accuracy, precision, recall, F1, ROC-AUC, log loss, confusion matrices
- **Rich visual diagnostics** for every model
- **Model comparison** with summary tables and performance charts

### 📊 Advanced Visualization Suite
Every notebook includes:
- Confusion matrices with Seaborn heatmaps
- ROC and Precision-Recall curves
- Feature importance plots (trees, coefficients)
- Cross-validation score distributions
- Prediction probability histograms
- Decision boundaries (SVM, Naive Bayes, MLP)
- Training loss curves (MLP)
- Learning curves (Random Forest)
- Per-class performance breakdowns
- Hyperparameter sensitivity analysis

### 🧪 Model Coverage
| Model | Dataset | Task Type |
|------|--------|-----------|
| **AdaBoost** | Breast Cancer Wisconsin | Binary Classification |
| **Decision Tree** | Titanic Survival | Binary Classification |
| **KNN** | MNIST Digits | Multi-class Classification |
| **Linear Regression** | California Housing | Regression |
| **Logistic Regression** | Pima Indians Diabetes | Binary Classification |
| **MLP (Neural Net)** | MNIST Digits | Multi-class Classification |
| **Naive Bayes** | Iris Species | Multi-class Classification |
| **Random Forest** | Iris Species | Multi-class Classification |
| **SVM** | Iris Species | Multi-class Classification |

---

## 📦 Installation

```bash
git clone https://github.com/hghaemi/ml_with_sklearn.git
cd ml_with_sklearn
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `matplotlib >= 3.4.0`
- `seaborn >= 0.11.0`
- `scikit-learn >= 1.0.0`
- `jupyter >= 1.0.0`
- `scipy >= 1.7.0`

---

## 🚀 Project Structure

```
ml_with_sklearn/
├── AdaBoost/
│   └── adaboost.ipynb                 # Ensemble boosting, feature importance
├── DecisionTree/
│   └── decision_tree.ipynb            # Tree visualization, depth tuning
├── KNN/
│   └── knn.ipynb                      # Distance metrics, k-analysis, speed vs accuracy
├── LinearRegression/
│   └── linear_regression.ipynb        # Regression fit, MSE, R²
├── LogisticRegression/
│   └── logistic_regression.ipynb      # L1/L2/ElasticNet penalties
├── MLP/
│   └── mlp.ipynb                      # Hidden layers, activation functions, loss curves
├── NaiveBayes/
│   └── naive_bayes.ipynb              # Gaussian/Multinomial/Bernoulli variants
├── RandomForest/
│   └── random_forest.ipynb            # Ensemble diversity, learning curves
├── SVM/
│   └── svm.ipynb                      # Kernel comparison, margins, PCA projection
├── .gitignore
├── LICENSE                            # MIT License
├── requirements.txt                   # Dependencies
└── README.md                          
```

---

## 📊 Model Evaluation Workflow

### 1. Data Loading & EDA
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
```

- Class distribution analysis
- Missing value handling
- Correlation matrices
- Feature distributions by class

### 2. Preprocessing & Splitting
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
```

### 3. Model Training & Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 1.0]}
grid = GridSearchCV(AdaBoostClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

### 4. Evaluation & Visualization
- Confusion matrix
- ROC-AUC curves
- Cross-validation stability
- Per-class metrics
- Performance vs complexity trade-offs

### 5. Comparative Analysis
- Bar charts of accuracy, F1, AUC
- Summary tables
- Best model selection and interpretation

---

## 🏗️ Architecture & Design Philosophy

### 🧩 Modular & Reproducible
Each algorithm has its own folder, enabling:
- Easy navigation
- Independent experimentation
- Reusable templates for new models

### 🎨 Visualization Standards
- Consistent color schemes (`husl`, `RdYlBu`)
- Professional figure sizing and layout
- Clear titles and axis labels
- Annotated performance metrics
- Grids and legends for readability

---

## 📈 Example: SVM Kernel Comparison

```python
models = {
    'SVM Linear': SVC(kernel='linear'),
    'SVM RBF (C=1)': SVC(kernel='rbf', C=1),
    'SVM Polynomial': SVC(kernel='poly', degree=3),
    'SVM (Best)': SVC(**grid_search.best_params_)
}
```

✅ **Visual Output Includes**:
- Decision boundaries via PCA projection
- Support vector count analysis
- Kernel-wise performance ranking
- C and gamma sensitivity curves
- Training time vs accuracy trade-offs

---

## 🧪 Testing & Validation

- **Cross-validated** (5-fold StratifiedKFold)
- **Stratified** to preserve class balance
- **Reproducible** with fixed `random_state=42`
- **Benchmarked** against sklearn defaults

---

## 🧠 Key Insights Across Models

### 1. **Dataset Matters**
- Linear models excel on linearly separable data (Iris, Breast Cancer)
- Non-linear models (MLP, RBF SVM) shine on complex patterns (MNIST)
- Tree-based models handle mixed data types well (Titanic)

### 2. **Preprocessing is Critical**
- SVM and KNN require scaling
- Naive Bayes benefits from binarization/discretization
- Neural networks converge faster with normalized inputs

### 3. **Hyperparameter Tuning Pays Off**
- Optimal `k` in KNN improves accuracy by 5–10%
- `C` and `gamma` in SVM drastically affect margins
- Depth and pruning in trees prevent overfitting

### 4. **Interpretability vs Performance**
- Linear models offer coefficient insights
- Trees provide visual decision paths
- Ensembles and neural nets trade interpretability for accuracy

---


## 🤝 Contributing

- Add new models (XGBoost, LightGBM, etc.)
- Improve visualizations
- Add more datasets
- Translate summaries to other languages

are welcome via **Pull Request**.

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use this code for academic or educational purposes
- ✅ Modify and extend it
- ✅ Share it with proper attribution

---

## 📬 Contact

**M. Hossein Ghaemi**  
📧 h.ghaemi.2003@gmail.com  
🔗 [GitHub: @hghaemi](https://github.com/hghaemi)  
🌐 Project Link: [https://github.com/hghaemi/ml_with_sklearn](https://github.com/hghaemi/ml_with_sklearn)

---

**Thank you for exploring this project!** 🚀  

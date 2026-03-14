# 🧠 Kernel Support Vector Machine

A classification project demonstrating Kernel SVM on the Social Network Ads dataset. Implements non-linear decision boundaries using kernel tricks in both Python and R, with visualization of training and test set results.

## 📖 Methodology

When data is not linearly separable, Kernel SVM projects it into a higher-dimensional space where a linear hyperplane can separate the classes, then maps the decision boundary back to the original space.

### Kernel Types

| Kernel | Description |
|--------|-------------|
| **Gaussian RBF** | Radial Basis Function — maps data into infinite-dimensional space. Used in this project. |
| **Polynomial** | Maps data using polynomial combinations of features. |
| **Sigmoid** | Based on the hyperbolic tangent function, similar to neural network activation. |
| **Linear** | Standard dot product — equivalent to regular SVM (no projection). |

This project uses the **Gaussian RBF kernel** (`kernel='rbf'`), which is well-suited for this dataset's non-linear purchase decision boundary based on Age and Estimated Salary.

### Pipeline

1. Load and preprocess the Social Network Ads dataset
2. Split into 75% training / 25% test sets
3. Apply feature scaling (StandardScaler)
4. Train SVC with RBF kernel
5. Evaluate with confusion matrix
6. Visualize decision boundaries for both sets

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Primary implementation |
| 📊 R | Alternative implementation |
| 🔬 scikit-learn | SVM classifier, preprocessing, model evaluation |
| 🔢 NumPy | Numerical operations |
| 🐼 pandas | Data loading and manipulation |
| 📈 matplotlib | Decision boundary visualization |
| 📦 e1071 (R) | SVM implementation in R |
| 📦 caTools (R) | Train/test splitting in R |

## 📋 Dependencies

### Python

```
numpy
pandas
matplotlib
scikit-learn
```

### R

```
caTools
e1071
ElemStatLearn
```

## 🚀 How to Run

### Python

```bash
pip install numpy pandas matplotlib scikit-learn
python kernel_svm.py
```

### R

```r
source("kernel_svm.R")
```

Make sure `Social_Network_Ads.csv` is in the same directory as the script.

## ⚠️ Known Issues

- The R script depends on `ElemStatLearn`, which has been archived from CRAN. You may need to install it from a mirror or archive.
- Visualization uses a fine mesh grid (`step=0.01`) which can be slow on large feature ranges. Increase the step size if performance is an issue.
- External image links in the original README are broken / expired.

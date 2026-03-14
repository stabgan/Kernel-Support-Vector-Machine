# Kernel Support Vector Machine

Binary classifier using an RBF-kernel SVM to predict whether a user purchases a product based on age and estimated salary.

## What It Does

Trains a Support Vector Machine with a Gaussian RBF kernel on the **Social Network Ads** dataset, evaluates accuracy via a confusion matrix and classification report, and renders decision-boundary plots for both training and test sets.

### Why a Kernel?

When data is not linearly separable, the kernel trick projects features into a higher-dimensional space where a linear separator exists, then maps the boundary back to 2-D — without explicitly computing the high-dimensional coordinates (the "kernel trick").

## Dataset

`Social_Network_Ads.csv` — 400 rows, 5 columns:

| Column | Description |
|---|---|
| User ID | Unique identifier (unused) |
| Gender | Male / Female (unused) |
| Age | User age |
| EstimatedSalary | Annual salary estimate |
| Purchased | Target — 0 or 1 |

## 🛠 Tech Stack

| | Tool | Purpose |
|---|---|---|
| 🐍 | Python 3 | Primary implementation |
| 📊 | scikit-learn | SVM, scaling, metrics |
| 🔢 | NumPy | Array operations |
| 🐼 | pandas | CSV loading |
| 📈 | matplotlib | Decision-boundary plots |
| 📉 | R (e1071) | Alternative R implementation |

## Getting Started

```bash
pip install numpy pandas matplotlib scikit-learn
python kernel_svm.py
```

The script prints a confusion matrix, accuracy score, and classification report, then shows two decision-boundary plots (training and test sets).

### R Version

```r
# Requires: caTools, e1071, ElemStatLearn
Rscript kernel_svm.R
```

## ⚠️ Known Issues

- The R script depends on `ElemStatLearn`, which was removed from CRAN. Install from archive or use an alternative plotting approach.
- External images in the old README used hotlinked URLs that may be broken.

## License

[MIT](LICENSE)

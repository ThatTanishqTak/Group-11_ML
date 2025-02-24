# G6 Connected Games Development Project

## Coursework 1 Group 11

**[Name] [K-Number]**

## Loading Modules

```python
# Loading load_digits from scikit-learn
from sklearn.datasets import load_digits
# Loading train_test_split and GridSearchCV from scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
# Importing Random Forest and Decision Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# Importing all the necessary metrics for evaluation
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
# For plotting
import matplotlib.pyplot as plt
# Importing numpy
import numpy as np
```

*Might need more or fewer modules according to ur needs.*

## Loading Data

```python
# Assign load_digits here
digits = load_digits()
# Assigning load_digits's data and target to X and Y
x, y = digits.data, digits.target
# Splitting the data for training and testing: 80% for training and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
```

*Ensure every **`random_state`** is set to 7 for consistency in data.*

## Classification Methods

Work on the two classification methods from the list below:

- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest
- Gradient Boosting Machines (GBM)
- XGBoost
- LightGBM
- CatBoost
- k-Nearest Neighbors (KNN)
- Decision Trees
- Naive Bayes
- AdaBoost

### Example Code:

```python
rfD = RandomForestClassifier(random_state=7)
dtD = DecisionTreeClassifier(random_state=7)
```

## Hyperparameter Tuning

Example Code:

```python
rfGrid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 10, 20]}
dtGrid = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 10, 20]}
```

## Training

Add training code here. Example Code:

```python
# Train the default models
rfD.fit(x_train, y_train)
dtD.fit(x_train, y_train)

# Train the hyperparameter-tuned models
rfH.fit(x_train, y_train)
dtH.fit(x_train, y_train)
```

## Evaluation

This section will contain all the graphs and confusion matrix printing.

## References

Just add links for all sources used in the loading modules and some AI-related references, cause lets be real they know we gonna use AI.
Also if u used Gemeni mention that u used its auto correct stuff....

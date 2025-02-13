# **Classical Machine Learning**

## **Part I – Group Submission (.ipynb)**  

This project focuses on implementing multiple classification models on the `digits` dataset from `scikit-learn`. The notebook follows a structured template and includes:  

### **1. Load the Dataset**  
- Use `sklearn.datasets.load_digits()`.  
- No external CSVs or websites.  
- Split data into **80% training** and **20% testing**.  

### **2. Classification Models (3 to 8 Methods)**  
Choose at least three (ideally up to eight) classification methods:

#### **Baseline Models:**
> We can use different models, however these are the ones mostly discused in class.
- k-Nearest Neighbors (KNN)
  
#### **Advanced Models:**  
- Support Vector Machine (SVM)  

#### **With Dimensionality Reduction (Optional for More Variations)**  
- PCA + SVM
  
### **3. Model Training & Testing (Separate Sections in Notebook)**  
- Train each classifier using the training set.  
- Tune hyperparameters (Grid Search for SVM).  
- Evaluate using:  
  - **Balanced Accuracy**  
  - **ROC AUC (macro-averaged approach)**  
- Plot a **single ROC curve** comparing all models.  
- Display **confusion matrices** for each model.  

### **4. Hyperparameter Tuning**  
- Show results for default settings first.  
- Use **GridSearchCV** or **RandomizedSearchCV** for optimization.  
- Compare both results.  

### **5. Result Presentation**  
- Label all tables and graphs clearly.  
- Print accuracy, ROC AUC, confusion matrices, and final comparisons.  
- Provide references for any method outside the module.  
---
## **TLDR**  

### **Group Submission**  
- Code sections: Load dataset, train classifiers, evaluate models, plot comparisons.  
- Includes results, hyperparameter tuning, ROC curve, confusion matrices.  

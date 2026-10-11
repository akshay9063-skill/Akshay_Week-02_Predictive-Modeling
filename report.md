# Week 2 Project Report

## Title
Predictive Modeling Using Machine Learning

## Aim
To build and evaluate supervised machine-learning models that classify sample student outcomes as Pass or Fail using performance-related features.

## Dataset
The included `student_performance.csv` contains 240 synthetic sample records and four input features: study hours per day, attendance percentage, assignment completion percentage, and previous exam score. The target is `outcome` (Pass/Fail). This is synthetic educational data, not real student data.

## Methodology
1. Load and inspect the CSV dataset using pandas.
2. Separate the input features and target label.
3. Encode the target labels.
4. Split data into 80% training and 20% testing using stratification.
5. Train Decision Tree and Random Forest classifiers.
6. Evaluate accuracy, precision, recall, F1-score, and classification reports.
7. Generate a confusion matrix and ROC curve for Random Forest.
8. Save evaluation results and charts in `outputs/`.

## Results from the included run
- Decision Tree test accuracy: 72.9%
- Random Forest test accuracy: 83.3%
- Random Forest ROC-AUC: 0.875
- Best model by test F1-score in this run: Random Forest

These are results from one run on this synthetic dataset. They are included for reproducibility and demonstration, not as a claim about real students or real-world performance.

## Expected Outcome
The project demonstrates a complete supervised-learning workflow: data preparation, train/test split, model training, prediction, evaluation, and visualization. The confusion matrix summarizes correct and incorrect predictions; the ROC curve illustrates the trade-off between true-positive and false-positive rates.

## Conclusion
The Random Forest model performed better than the Decision Tree model on the included held-out synthetic test set. Since the dataset is synthetic and simplified, these results are educational only and should not be used for real academic decisions.

# Predictive Modeling Using Machine Learning
**Internship:** TiRANEX — Week 2  
**Prepared by:** A. Akshay

## Project overview
This project demonstrates a supervised machine-learning classification workflow. It predicts a sample student's outcome (`Pass` or `Fail`) from study hours, attendance, assignment completion, and previous exam score.

> **Dataset note:** `student_performance.csv` is synthetic sample data created for educational demonstration. It is not real student data and must not be used to make real academic decisions.

## Objectives
- Load and inspect a CSV dataset.
- Separate input features and target labels.
- Split data into training and testing sets.
- Train and compare Decision Tree and Random Forest classifiers.
- Evaluate models using accuracy, precision, recall, F1-score, a confusion matrix, and ROC-AUC when available.
- Save evaluation charts and a text report.

## Files
- `student_performance.csv` — included sample dataset (240 records).
- `train_model.py` — complete training and evaluation script.
- `requirements.txt` — Python dependencies.
- `report.md` — concise project report.
- `.gitignore` — excludes generated files and Python cache.
- `outputs/` — included evaluation charts and metrics; regenerated when the script runs.

## How to run
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python train_model.py
   ```
5. Open the generated files in the `outputs` folder.

## Methods
The script uses a stratified 80/20 train-test split and a fixed random seed so the experiment can be reproduced. Both models are fitted only on the training data and evaluated on held-out test data.

## Expected outcome
The script prints evaluation metrics and creates:
- `outputs/model_metrics.txt`
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`

The included output files are from one run on this included synthetic dataset. Re-run the script to regenerate them. Do not interpret these scores as real-world student performance.

## Limitations
The dataset is synthetic and intentionally simplified. Results are only for demonstrating a machine-learning workflow; they do not establish real-world predictive ability or fairness.

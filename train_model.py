"""Train and evaluate student outcome classification models.

Dataset is synthetic and for educational demonstration only.
Run: python train_model.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_performance.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    data = pd.read_csv(DATA_PATH)
    target = "outcome"
    features = [
        "study_hours_per_day",
        "attendance_percent",
        "assignment_completion_percent",
        "previous_exam_score",
    ]
    X = data[features]
    y_text = data[target]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)  # alphabetical order: Fail=0, Pass=1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        ),
    }

    lines = [
        "PREDICTIVE MODELING USING MACHINE LEARNING",
        "Synthetic student performance dataset — educational use only",
        f"Total records: {len(data)}",
        f"Training records: {len(X_train)}",
        f"Testing records: {len(X_test)}",
        f"Features: {', '.join(features)}",
        f"Target classes: {', '.join(encoder.classes_)}",
        "",
    ]

    best_model = None
    best_name = None
    best_f1 = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        lines.extend([
            f"MODEL: {name}",
            f"Accuracy:  {accuracy:.3f}",
            f"Precision (Pass class): {precision:.3f}",
            f"Recall (Pass class):    {recall:.3f}",
            f"F1-score (Pass class):   {f1:.3f}",
            "Classification report:",
            classification_report(
                y_test, predictions, labels=[0, 1],
                target_names=list(encoder.classes_), zero_division=0
            ),
            "",
        ])

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

        if name == "Random Forest":
            cm = confusion_matrix(y_test, predictions, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6, 5))
            image = ax.imshow(cm, interpolation="nearest")
            ax.set_title("Confusion Matrix — Random Forest")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_xticks([0, 1], labels=list(encoder.classes_))
            ax.set_yticks([0, 1], labels=list(encoder.classes_))
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
            plt.close(fig)

            probabilities = model.predict_proba(X_test)[:, list(encoder.classes_).index("Pass")]
            fpr, tpr, _ = roc_curve(y_test, probabilities, pos_label=1)
            auc = roc_auc_score(y_test, probabilities)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.3f})")
            ax.plot([0, 1], [0, 1], linestyle="--", label="Chance baseline")
            ax.set_title("ROC Curve — Random Forest")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=160)
            plt.close(fig)
            lines.append(f"Random Forest ROC-AUC: {auc:.3f}")
            lines.append("")

    lines.extend([
        f"Best model by test F1-score in this run: {best_name}",
        "",
        "Interpretation: Metrics describe performance on this held-out synthetic test set.",
        "They do not validate performance on real students or real academic decisions.",
    ])
    report_path = OUTPUT_DIR / "model_metrics.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved evaluation files to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

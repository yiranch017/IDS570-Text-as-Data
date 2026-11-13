import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "l1"


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing saved Week 09 file: {path}. "
            "Place the original train/test JSON files in the data folder."
        )
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path.name} must contain a nonempty list of [text, label] pairs.")
    for row in rows:
        if (not isinstance(row, list) or len(row) != 2
                or not isinstance(row[0], str) or row[1] not in (0, 1)):
            raise ValueError(f"Invalid [text, label] pair in {path.name}.")
    texts = [t for t, y in rows]
    labels = np.array([y for t, y in rows])
    if set(labels) != {0, 1}:
        raise ValueError(f"{path.name} must contain both NEG=0 and CORE=1.")
    return texts, labels


def main():
    train_path = DATA_DIR / "train_core_vs_neg.json"
    test_path = DATA_DIR / "test_core_vs_neg.json"
    X_train_texts, y_train = load_data(train_path)
    X_test_texts, y_test = load_data(test_path)

    vectorizer = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.9)
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    clf = LogisticRegression(l1_ratio=1, solver="liblinear", C=1.0, max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    coefficients = clf.coef_[0]
    words = vectorizer.get_feature_names_out()
    nonzero = int(np.count_nonzero(coefficients))
    report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["NEG", "CORE"],
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    auc = float(roc_auc_score(y_test, y_prob))

    print("Model: l1")
    print("scikit-learn:", sklearn.__version__)
    print("Train/test shapes:", X_train.shape, X_test.shape)
    print("Confusion matrix (rows=labels, columns=predictions; NEG, CORE):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["NEG", "CORE"],
        digits=4, zero_division=0
    ))
    print(f"ROC AUC: {auc:.6f}")
    print(f"Non-zero coefficients: {nonzero} / {len(coefficients)}")
    print(f"Zero coefficients: {len(coefficients) - nonzero}")
    print("Iterations:", clf.n_iter_.tolist())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top_words = {}
    for name, indices in (
        ("positive", np.flatnonzero(coefficients > 0)),
        ("negative", np.flatnonzero(coefficients < 0)),
    ):
        ordered = sorted(indices, key=lambda i: coefficients[i], reverse=name == "positive")[:15]
        top_words[name] = [
            {"word": str(words[i]), "coefficient": float(coefficients[i])} for i in ordered
        ]
        print(f"\nTop {len(ordered)} strictly {name} words:")
        for row in top_words[name]:
            print(f"{row['word']:25s} {row['coefficient']:+.6f}")

    error_groups = {
        "FP": np.flatnonzero((y_test == 0) & (y_pred == 1)).tolist(),
        "FN": np.flatnonzero((y_test == 1) & (y_pred == 0)).tolist(),
    }
    rng = np.random.default_rng(42)
    for values in error_groups.values():
        rng.shuffle(values)
    selected = []
    for j in range(max(len(v) for v in error_groups.values())):
        for kind, values in error_groups.items():
            if j < len(values) and len(selected) < 10:
                selected.append((kind, values[j]))
        if len(selected) >= 10:
            break
    with (OUTPUT_DIR / "misclassified_review.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_index", "error_type", "weak_label", "prediction", "p_core", "text",
                         "human_label", "model_or_label_right", "reason"])
        for kind, i in selected:
            writer.writerow([i, kind, int(y_test[i]), int(y_pred[i]), float(y_prob[i]),
                             X_test_texts[i], "", "", ""])
    print(f"\nSaved {len(selected)} misclassified chunks for close reading.")
    if len(selected) < 10:
        print("Fewer than 10 misclassifications exist; all available errors were saved.")
    with (OUTPUT_DIR / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_index", "weak_label", "prediction", "p_core"])
        for i in range(len(y_test)):
            writer.writerow([i, int(y_test[i]), int(y_pred[i]), float(y_prob[i])])
    results = {
        "model": "l1", "sklearn_version": sklearn.__version__,
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "train_shape": list(X_train.shape), "test_shape": list(X_test.shape),
        "confusion_matrix": cm.tolist(), "classification_report": report,
        "roc_auc": auc, "nonzero_coefficients": nonzero,
        "total_coefficients": len(coefficients), "top_words": top_words,
        "model_parameters": clf.get_params(),
        "vectorizer_settings": {"lowercase": True, "min_df": 5, "max_df": 0.9},
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (OUTPUT_DIR / "coefficients.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "coefficient"])
        writer.writerows(zip(words, coefficients))
    print("Outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

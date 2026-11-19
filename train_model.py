"""
train_model.py - Trains a Naive Bayes Disease Classifier on the symptoms dataset.
Pure Python standard library - zero external DLL dependencies, robust against Application Control policies.
Saves model to model/disease_model.pkl and symptoms list to model/symptoms.pkl
"""
import csv
import math
import os
import pickle
from collections import defaultdict

class DiseaseClassifier:
    """
    Multivariate Bernoulli Naive Bayes Classifier with Bayesian symptom likelihood estimation.
    Provides predict() and predict_proba() compatible with scikit-learn interfaces.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.classes_ = []
        self.symptoms_ = []
        self.class_priors_ = {}
        self.feature_prob_pos_ = {}  # P(symptom=1 | disease)
        self.feature_prob_neg_ = {}  # P(symptom=0 | disease)

    def fit(self, X_rows, y_labels, symptoms):
        self.symptoms_ = list(symptoms)
        total_samples = len(y_labels)
        
        disease_counts = defaultdict(int)
        symptom_pos_counts = defaultdict(lambda: defaultdict(int))
        
        for row, disease in zip(X_rows, y_labels):
            disease_counts[disease] += 1
            for sym in self.symptoms_:
                if row.get(sym, 0) == 1:
                    symptom_pos_counts[disease][sym] += 1

        self.classes_ = sorted(list(disease_counts.keys()))

        # Compute log priors and feature likelihoods with Laplace smoothing
        for disease in self.classes_:
            n_d = disease_counts[disease]
            self.class_priors_[disease] = math.log(n_d / total_samples)
            self.feature_prob_pos_[disease] = {}
            self.feature_prob_neg_[disease] = {}
            
            for sym in self.symptoms_:
                pos_count = symptom_pos_counts[disease][sym]
                p_pos = (pos_count + self.alpha) / (n_d + 2 * self.alpha)
                p_neg = 1.0 - p_pos
                self.feature_prob_pos_[disease][sym] = math.log(max(p_pos, 1e-9))
                self.feature_prob_neg_[disease][sym] = math.log(max(p_neg, 1e-9))

        return self

    def _score_sample(self, sample_dict):
        # Compute unnormalized log-posterior for each disease
        scores = {}
        for disease in self.classes_:
            log_p = self.class_priors_[disease]
            pos_probs = self.feature_prob_pos_[disease]
            neg_probs = self.feature_prob_neg_[disease]
            
            for sym in self.symptoms_:
                val = sample_dict.get(sym, 0)
                if val == 1:
                    # Give weight to present symptoms
                    log_p += pos_probs[sym]
                else:
                    # Small penalty/weight for absent symptoms
                    log_p += 0.05 * neg_probs[sym]
            scores[disease] = log_p

        # Softmax with log-sum-exp trick
        max_s = max(scores.values())
        exp_scores = {d: math.exp(s - max_s) for d, s in scores.items()}
        total_exp = sum(exp_scores.values())
        return [exp_scores[d] / total_exp for d in self.classes_]

    def predict_proba(self, X):
        """Accepts DataFrame, list of dicts, or single dict."""
        if hasattr(X, "to_dict"):
            rows = X.to_dict(orient="records")
        elif isinstance(X, list):
            rows = X
        elif isinstance(X, dict):
            rows = [X]
        else:
            rows = [dict(zip(self.symptoms_, X[0]))]

        return [self._score_sample(r) for r in rows]

    def predict(self, X):
        probas = self.predict_proba(X)
        predictions = []
        for p in probas:
            max_idx = p.index(max(p))
            predictions.append(self.classes_[max_idx])
        return predictions


def load_dataset(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        symptoms = [c for c in reader.fieldnames if c != "prognosis"]
        X = []
        y = []
        for row in reader:
            y.append(row["prognosis"])
            X.append({sym: int(row[sym]) for sym in symptoms})
    return X, y, symptoms


def main():
    print("Loading data...")
    X_train, y_train, symptoms = load_dataset("data/Training.csv")
    X_test, y_test, _ = load_dataset("data/Testing.csv")

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Symptoms count:   {len(symptoms)}")
    print(f"Unique diseases:  {len(set(y_train))}")

    print("\nTraining Disease Classifier (Bayesian Symptom Inference)...")
    model = DiseaseClassifier(alpha=1.0)
    model.fit(X_train, y_train, symptoms)

    # Evaluate
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    correct = sum(1 for yt, yp in zip(y_test, y_pred) if yt == yp)
    accuracy = correct / len(y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{len(y_test)})")

    # Save
    os.makedirs("model", exist_ok=True)
    with open("model/disease_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/symptoms.pkl", "wb") as f:
        pickle.dump(symptoms, f)

    print("\nModel saved successfully -> model/disease_model.pkl")
    print("Symptoms list saved       -> model/symptoms.pkl")
    print("Ready to run Streamlit app!")


if __name__ == "__main__":
    main()

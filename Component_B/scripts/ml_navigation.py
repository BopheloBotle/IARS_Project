"""
Machine Learning for Robotic Navigation / Decision-Making (Component B, Section 3)

Trains a small neural network (MLPClassifier) to select discrete navigation
actions from local sensor features (simulated LiDAR sector mins + goal bearing).

Evaluates accuracy, success rate proxy, presents confusion matrix, and
compares centralised vs distributed decision-making.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES = os.path.join(RESULTS, "figures")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

ACTION_NAMES = ["FORWARD", "LEFT", "RIGHT", "STOP"]


def _expert_policy(sectors: np.ndarray, goal_bearing: float) -> int:
    """
    Rule-based expert for dataset labelling:
    sectors = [front, left, right] min ranges.
    """
    front, left, right = sectors
    if front < 0.25 and left < 0.45 and right < 0.45:
        return 3  # STOP — boxed in
    if front < 0.55:
        # turn toward more open side, bias by goal
        if left > right + 0.05:
            return 1
        if right > left + 0.05:
            return 2
        return 1 if goal_bearing > 0 else 2
    # Clear ahead: turn gently toward goal if needed
    if abs(goal_bearing) < 0.35:
        return 0  # FORWARD
    return 1 if goal_bearing > 0 else 2


def generate_dataset(n: int = 4000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for _ in range(n):
        sectors = rng.uniform(0.15, 2.5, size=3)
        # Correlate: sometimes front blocked / boxed-in
        if rng.random() < 0.25:
            sectors[0] = rng.uniform(0.1, 0.55)
        if rng.random() < 0.08:
            sectors[:] = rng.uniform(0.08, 0.35, size=3)
        goal_bearing = rng.uniform(-np.pi, np.pi)
        # Features: 3 sector mins + sin/cos of goal bearing (continuous angle encoding)
        feat = np.array(
            [
                sectors[0],
                sectors[1],
                sectors[2],
                np.sin(goal_bearing),
                np.cos(goal_bearing),
            ]
        )
        label = _expert_policy(sectors, goal_bearing)
        # Keep a balanced STOP class: only remap when clearly open AND not forced-stop sample
        if label == 3 and max(sectors[1], sectors[2]) > 0.8 and sectors[0] >= 0.35:
            label = 1 if sectors[1] >= sectors[2] else 2
        X.append(feat)
        y.append(label)
    # Oversample STOP examples for class balance
    X = np.array(X)
    y = np.array(y)
    stop_idx = np.where(y == 3)[0]
    if len(stop_idx) < 200:
        need = 200 - len(stop_idx)
        extra = rng.choice(stop_idx, size=need, replace=True) if len(stop_idx) else []
        if len(stop_idx):
            X = np.vstack([X, X[extra]])
            y = np.concatenate([y, y[extra]])
    return X, y


def train_and_evaluate(seed: int = 42) -> dict:
    X, y = generate_dataset(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # Distributed model: trained per-agent style on full shared dataset
    # (same architecture; represents identical local policies)
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=800,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    # Simulated navigation success rate: rollouts where policy reaches goal
    success_rate = simulate_nav_success(clf, n_episodes=200, seed=seed)

    # Centralised vs distributed comparison:
    # Centralised: one global net (above). Distributed: 3 agents each trained
    # on bootstrap subsets (data heterogeneity) — typically slightly lower acc.
    dist_accs = []
    for i in range(3):
        idx = np.random.default_rng(seed + i).choice(len(X_train), size=len(X_train) // 2, replace=False)
        local = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=800,
            random_state=seed + i,
            early_stopping=True,
            validation_fraction=0.1,
        )
        local.fit(X_train[idx], y_train[idx])
        dist_accs.append(accuracy_score(y_test, local.predict(X_test)))

    report = classification_report(y_test, y_pred, target_names=ACTION_NAMES, digits=3)

    np.savez(
        os.path.join(DATA, "ml_nav_dataset.npz"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
    )

    return {
        "model": clf,
        "accuracy": acc,
        "confusion_matrix": cm,
        "success_rate": success_rate,
        "report": report,
        "centralised_accuracy": acc,
        "distributed_accuracies": dist_accs,
        "distributed_mean_accuracy": float(np.mean(dist_accs)),
        "y_test": y_test,
        "y_pred": y_pred,
    }


def simulate_nav_success(clf: MLPClassifier, n_episodes: int = 200, seed: int = 0) -> float:
    """Lightweight 2D point-robot episodes using the learned action policy."""
    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(n_episodes):
        pos = rng.uniform(-2, 2, size=2)
        goal = rng.uniform(-2, 2, size=2)
        theta = rng.uniform(-np.pi, np.pi)
        ok = False
        for _t in range(120):
            # Fake sector readings from distance to a random obstacle wall at x=0 sometimes
            front = 1.5 + 0.3 * rng.normal()
            # Obstacle slab
            if abs(pos[0]) < 0.3 and abs(goal[0] - pos[0]) > 0.3:
                front = min(front, abs(pos[0]) + 0.2)
            left = 1.2 + 0.2 * rng.normal()
            right = 1.2 + 0.2 * rng.normal()
            bearing = np.arctan2(goal[1] - pos[1], goal[0] - pos[0]) - theta
            bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
            feat = np.array([[front, left, right, np.sin(bearing), np.cos(bearing)]])
            action = int(clf.predict(feat)[0])
            if action == 0:
                pos = pos + 0.12 * np.array([np.cos(theta), np.sin(theta)])
            elif action == 1:
                theta += 0.35
            elif action == 2:
                theta -= 0.35
            else:
                theta += 0.35  # STOP treated as turn-in-place recovery
            if np.linalg.norm(goal - pos) < 0.25:
                ok = True
                break
        successes += int(ok)
    return successes / n_episodes


def visualise_ml(result: dict) -> str:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    cm = result["confusion_matrix"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ACTION_NAMES,
        yticklabels=ACTION_NAMES,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (acc={result['accuracy']:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax = axes[1]
    labels = ["Centralised MLP", "Distributed mean", "Dist agent0", "Dist agent1", "Dist agent2"]
    values = [
        result["centralised_accuracy"],
        result["distributed_mean_accuracy"],
        *result["distributed_accuracies"],
    ]
    colours = sns.color_palette("muted", n_colors=len(labels))
    ax.bar(labels, values, color=colours)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Centralised vs Distributed (nav success={result['success_rate']:.2f})")
    ax.tick_params(axis="x", rotation=25)

    fig.suptitle("ML Navigation Decision-Making")
    fig.tight_layout()
    out = os.path.join(FIGURES, "B3_ml_confusion_and_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(RESULTS, "B3_ml_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"accuracy={result['accuracy']:.4f}\n")
        f.write(f"success_rate={result['success_rate']:.4f}\n")
        f.write(f"centralised_accuracy={result['centralised_accuracy']:.4f}\n")
        f.write(f"distributed_mean_accuracy={result['distributed_mean_accuracy']:.4f}\n")
        f.write("\n" + result["report"])
    return out


def main():
    print("=" * 60)
    print("COMPONENT B.3 — ML for Navigation / Decision-Making")
    print("=" * 60)
    result = train_and_evaluate()
    print(f"Centralised accuracy: {result['accuracy']:.4f}")
    print(f"Nav success rate:     {result['success_rate']:.4f}")
    print(f"Distributed mean acc: {result['distributed_mean_accuracy']:.4f}")
    print(result["report"])
    out = visualise_ml(result)
    print(f"Saved: {out}")
    return result


if __name__ == "__main__":
    main()

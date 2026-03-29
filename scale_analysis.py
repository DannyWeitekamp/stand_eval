"""
Scale analysis: compare TreeClassifier (decision_tree), TreeClassifier (option_tree),
and STANDClassifier on the satellite-image UCI dataset across training set sizes.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from stand.tree_classifier import TreeClassifier
from stand.stand import STANDClassifier

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading satellite dataset...")
data = fetch_openml("satimage", version=1, as_frame=False, parser="liac-arff")
X = data.data.astype(np.float64)
le = LabelEncoder()
Y = le.fit_transform(data.target).astype(np.int64)

# All features are continuous; no nominal features
X_nom_full = np.zeros((len(X), 0), dtype=np.int64)

# Fixed test set: samples past index 6000
X_cont_test = X[6000:]
X_nom_test = X_nom_full[6000:]
Y_test = Y[6000:]

# Pool to sample training sets from
X_cont_pool = X[:6000]
X_nom_pool = X_nom_full[:6000]
Y_pool = Y[:6000]

TRAIN_SIZES = [50, 100, 200, 400, 800, 1600, 3200, 6000]

N_REPEATS = 10
RNG_SEED = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def count_nodes(model):
    """Return number of nodes in the tree (works for both TreeClassifier and
    STANDClassifier, which exposes its op_tree via op_tree_classifier)."""
    if isinstance(model, STANDClassifier):
        return len(model.op_tree_classifier.nodes)
    return len(model.nodes)


def run_condition(make_model, X_nom_tr, X_cont_tr, Y_tr):
    """Fit model, return (accuracy, n_nodes) on the fixed test set."""
    model = make_model()
    model.fit(X_nom_tr, X_cont_tr, Y_tr)
    preds = model.predict(X_nom_test, X_cont_test)
    acc = np.mean(preds == Y_test)
    n_nodes = count_nodes(model)
    return acc, n_nodes


CONDITIONS = [
    ("TreeClassifier (decision_tree)", lambda: TreeClassifier(preset_type="decision_tree")),
    # ("TreeClassifier (option_tree)",   lambda: TreeClassifier(preset_type="option_tree")),
    ("STANDClassifier",               lambda: STANDClassifier()),
]

# ── Experiment ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)

# results[condition_name][train_size] = {"acc": [...], "nodes": [...]}
results = {name: {n: {"acc": [], "nodes": []} for n in TRAIN_SIZES}
           for name, _ in CONDITIONS}

total = len(TRAIN_SIZES) * N_REPEATS * len(CONDITIONS)
done = 0

for train_size in TRAIN_SIZES:
    for rep in range(N_REPEATS):
        idx = rng.choice(len(X_cont_pool), size=train_size, replace=False)
        X_cont_tr = X_cont_pool[idx]
        X_nom_tr  = X_nom_pool[idx]
        Y_tr      = Y_pool[idx]

        for name, make_model in CONDITIONS:
            acc, n_nodes = run_condition(make_model, X_nom_tr, X_cont_tr, Y_tr)
            results[name][train_size]["acc"].append(acc)
            results[name][train_size]["nodes"].append(n_nodes)
            done += 1
            print(f"  [{done}/{total}] {name} | n={train_size} rep={rep+1} "
                  f"acc={acc:.3f} nodes={n_nodes}")

# ── Table ─────────────────────────────────────────────────────────────────────

def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


print("\n\n" + "=" * 100)
print("RESULTS TABLE")
print("=" * 100)

# --- Accuracy table ---
print("\n[ Test Accuracy (mean ± std over 10 random subsets) ]\n")

col_w = 32
header = f"{'Train size':>12}" + "".join(f"{name:>{col_w}}" for name, _ in CONDITIONS)
print(header)
print("-" * len(header))
for n in TRAIN_SIZES:
    row = f"{n:>12}"
    for name, _ in CONDITIONS:
        accs = results[name][n]["acc"]
        row += f"{fmt(np.mean(accs), np.std(accs)):>{col_w}}"
    print(row)

# --- Node count table ---
print("\n[ Number of Nodes (mean ± std over 10 random subsets) ]\n")

header2 = f"{'Train size':>12}" + "".join(f"{name:>{col_w}}" for name, _ in CONDITIONS)
print(header2)
print("-" * len(header2))
for n in TRAIN_SIZES:
    row = f"{n:>12}"
    for name, _ in CONDITIONS:
        nodes = results[name][n]["nodes"]
        row += f"{fmt(np.mean(nodes), np.std(nodes)):>{col_w}}"
    print(row)

print("\n" + "=" * 100)
